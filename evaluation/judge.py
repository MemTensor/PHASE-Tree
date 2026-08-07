"""LLM-as-Judge scoring and embedding cosine similarity.

For each prediction produced by ``predict_prompt.py`` or ``predict_hypernet.py``, this script:

  1. **LLM Judge** — asks GPT-4.1 to score two dimensions (1-5 each):
     * ``character_score``: profile consistency — how consistently the
       prediction reflects the described character traits.
     * ``semantic_score``: contextual coherence — how coherently the
       prediction continues the dialogue context.
  2. **Embedding similarity** — computes cosine similarity between the
     prediction embedding and the ground-truth embedding.

The character profile injected into the judge prompt is always taken from the
``--persona_data`` file (typically ``m6_phase_tree/all_dialogues.json``) so
that all experiments are evaluated against the same high-quality persona
ground truth.

Both scoring passes are parallelized with ``ThreadPoolExecutor`` and display
real-time progress via ``tqdm``.  Both support resume: existing
``question_id`` entries in the output files are skipped on re-run.

Usage example::

    python evaluation/judge.py \\
        --predictions_dir results/RAIDEN/prompt/main/m6_phase_tree/random_test \\
        --persona_data   LongEvoRoleBench/processed/RAIDEN/m6_phase_tree/all_dialogues.json \\
        --num_workers 10
"""

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------

_RUBRIC_CACHE: dict[str, str] = {}


def load_rubric(path: str) -> str:
    if path not in _RUBRIC_CACHE:
        with open(path, "r", encoding="utf-8") as f:
            _RUBRIC_CACHE[path] = f.read()
    return _RUBRIC_CACHE[path]


def _default_rubric_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona_rubric.md")


# ---------------------------------------------------------------------------
# Judge prompt — profile is injected as persona ground truth
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are an expert at evaluating dialogue quality. Assess how well the predicted line is consistent with the character profile and how coherently it continues the dialogue context. Strictly follow the scoring rubric below.

### Character Profile
{character_profile}

### Dialogue Context
{context}

### Ground-truth Line
{ground_truth}

### Predicted Line
{prediction}

### Scoring Rubric
{rubric}

### Instructions
1. Evaluate the predicted line on two independent dimensions defined in the rubric: **character_score** (profile consistency, 1-5) and **semantic_score** (contextual coherence, 1-5).
2. For character_score, measure ONLY consistency with the Character Profile provided above — do not use external knowledge about the character. If a trait is not mentioned in the profile, do not reward or penalize its presence.
3. For semantic_score, evaluate ONLY whether the response is a coherent continuation of the dialogue context. The ground-truth is a reference point, not the only correct answer.
4. Score each dimension INDEPENDENTLY. Poor profile consistency does not imply poor contextual coherence, and vice versa.
5. Write a brief justification referencing the rubric level that best matches each score.
6. Output ONLY the following JSON, nothing else:

{{"character_score": <1-5>, "semantic_score": <1-5>, "reasoning": "<brief justification>"}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> list[dict]:
    items = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_done_keys(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                keys.add(json.loads(line)["question_id"])
    return keys


def get_env(key: str, fallback: str | None = None) -> str | None:
    val = os.getenv(key, fallback)
    return val if val else fallback


def judge_scores_filename(model: str) -> str:
    """Return the judge-scores filename for a given model.

    ``gpt-4.1`` keeps the legacy ``judge_scores.jsonl`` name so existing GPT
    results and downstream report scripts remain unchanged.  Other models are
    written to model-specific files, e.g. ``judge_scores_glm-5.2.jsonl``.
    """
    if model == "gpt-4.1":
        return "judge_scores.jsonl"
    safe = re.sub(r"[^\w.-]+", "_", model)
    return f"judge_scores_{safe}.jsonl"


def should_disable_thinking(model: str, explicit: bool | None = None) -> bool:
    """Return whether to disable chain-of-thought thinking for judge models."""
    if explicit is not None:
        return explicit
    env = get_env("JUDGE_DISABLE_THINKING")
    if env is not None:
        return env.lower() in {"1", "true", "yes", "on"}
    m = model.lower()
    return m.startswith("glm") or "deepseek" in m


def model_supports_thinking_param(model: str) -> bool:
    """Only GLM / DeepSeek endpoints accept extra_body.thinking."""
    m = model.lower()
    return m.startswith("glm") or "deepseek" in m


# ---------------------------------------------------------------------------
# LLM-as-Judge  (parallel via ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def parse_judge_response(text: str) -> dict:
    text = text.strip()
    try:
        data = json.loads(text)
        return {
            "character_score": int(data["character_score"]),
            "semantic_score": int(data["semantic_score"]),
            "reasoning": str(data.get("reasoning", "")),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    json_match = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return {
                "character_score": int(data["character_score"]),
                "semantic_score": int(data["semantic_score"]),
                "reasoning": str(data.get("reasoning", "")),
            }
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    cs = re.search(r"character_score[\"']?\s*[:=]\s*(\d)", text)
    ss = re.search(r"semantic_score[\"']?\s*[:=]\s*(\d)", text)
    if cs and ss:
        return {
            "character_score": int(cs.group(1)),
            "semantic_score": int(ss.group(1)),
            "reasoning": "parsed via regex fallback",
        }
    raise ValueError(f"Could not parse judge response: {text[:200]}")


def judge_one(
    client: OpenAI,
    model: str,
    sample: dict,
    prediction: str,
    rubric: str,
    max_retries: int = 3,
    rate_limit_sleep: float = 0.1,
    disable_thinking: bool = False,
    record_model: str | None = None,
) -> dict:
    api_model = model
    label_model = record_model or model
    prompt = JUDGE_PROMPT.format(
        character_profile=sample["profile_text"],
        context=sample["input"],
        ground_truth=sample["output"],
        prediction=prediction,
        rubric=rubric,
    )
    create_kwargs: dict = {
        "model": api_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 400,
        "timeout": 60,
    }
    if disable_thinking and model_supports_thinking_param(api_model):
        create_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            if rate_limit_sleep > 0:
                time.sleep(rate_limit_sleep)
            resp = client.chat.completions.create(**create_kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            if not raw:
                raise ValueError("empty judge response content")
            scores = parse_judge_response(raw)
            return {
                "question_id": sample["question_id"],
                "role": sample["role"],
                "judge_model": label_model,
                **scores,
            }
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                backoff = 2 ** attempt + rate_limit_sleep
                time.sleep(backoff)
    raise last_err


def run_llm_judge(
    client: OpenAI,
    model: str,
    samples: dict[str, dict],
    predictions: dict[str, str],
    output_path: str,
    rubric: str,
    num_workers: int = 10,
    max_retries: int = 3,
    rate_limit_sleep: float = 0.1,
    disable_thinking: bool = False,
    api_model: str | None = None,
    record_model: str | None = None,
):
    call_model = api_model or model
    label_model = record_model or model
    done_keys = load_done_keys(output_path)
    tasks = [
        (qid, pred) for qid, pred in predictions.items()
        if qid not in done_keys and qid in samples
    ]
    print(f"  LLM Judge: total={len(predictions)}, done={len(done_keys)}, "
          f"remaining={len(tasks)}, workers={num_workers}, "
          f"retries={max_retries}, thinking={'OFF' if disable_thinking else 'ON'}"
          + (f", api_model={call_model}" if call_model != label_model else ""),
          flush=True)
    if not tasks:
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    f = open(output_path, "a", encoding="utf-8")
    errors = 0
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(judge_one, client, call_model, samples[qid], pred, rubric,
                        max_retries, rate_limit_sleep, disable_thinking, label_model): qid
            for qid, pred in tasks
        }
        pbar = tqdm(total=len(tasks), desc="llm-judge", unit="sample",
                    file=sys.stderr, dynamic_ncols=True)
        done_count = 0
        for future in as_completed(futures):
            qid = futures[future]
            try:
                result = future.result()
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                errors += 1
                tqdm.write(f"  ERROR judging {qid}: {e}", file=sys.stderr)
            done_count += 1
            pbar.update(1)
            elapsed = time.perf_counter() - t0
            pbar.set_postfix_str(f"{done_count/elapsed:.1f} it/s, err={errors}")
        pbar.close()

    f.close()
    elapsed = time.perf_counter() - t0
    print(f"  LLM Judge done: {len(tasks)-errors} scored, {errors} errors, "
          f"{elapsed:.1f}s ({len(tasks)/elapsed:.1f} it/s)", flush=True)


# ---------------------------------------------------------------------------
# Embedding cosine similarity  (parallel via ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


EMBED_BATCH_SIZE = 64


def _embed_batch(
    client: OpenAI,
    model: str,
    texts: list[str],
    max_retries: int = 3,
) -> list[list[float]]:
    """Embed a batch of texts in a single API call with retry."""
    clean = [t[:8000] or " " for t in texts]
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.embeddings.create(model=model, input=clean, timeout=60)
            sorted_data = sorted(resp.data, key=lambda x: x.index)
            return [d.embedding for d in sorted_data]
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                raise


def run_embedding_scoring(
    embed_client: OpenAI,
    embed_model: str,
    samples: dict[str, dict],
    predictions: dict[str, str],
    output_path: str,
    num_workers: int = 10,
):
    done_keys = load_done_keys(output_path)
    tasks = [
        (qid, pred) for qid, pred in predictions.items()
        if qid not in done_keys and qid in samples
    ]
    print(f"  Embedding: total={len(predictions)}, done={len(done_keys)}, "
          f"remaining={len(tasks)}, batch_size={EMBED_BATCH_SIZE}, "
          f"workers={num_workers}", flush=True)
    if not tasks:
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    batches = []
    for i in range(0, len(tasks), EMBED_BATCH_SIZE):
        batch = tasks[i: i + EMBED_BATCH_SIZE]
        batches.append(batch)

    f = open(output_path, "a", encoding="utf-8")
    errors = 0
    t0 = time.perf_counter()

    def _process_batch(batch):
        """Embed one batch: interleave pred and gt texts, then compute cosine pairs."""
        all_texts = []
        for qid, pred in batch:
            gt_text = samples[qid]["output"]
            all_texts.append(pred)
            all_texts.append(gt_text)

        embeddings = _embed_batch(embed_client, embed_model, all_texts)

        results = []
        for j, (qid, _) in enumerate(batch):
            pred_emb = embeddings[j * 2]
            gt_emb = embeddings[j * 2 + 1]
            sim = cosine_similarity(pred_emb, gt_emb)
            results.append({
                "question_id": qid,
                "role": samples[qid]["role"],
                "embedding_similarity": round(sim, 4),
            })
        return results

    pbar = tqdm(total=len(tasks), desc="embedding", unit="pair",
                file=sys.stderr, dynamic_ncols=True)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_process_batch, b): b for b in batches}
        for future in as_completed(futures):
            batch = futures[future]
            try:
                records = future.result()
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                pbar.update(len(records))
            except Exception as e:
                errors += len(batch)
                tqdm.write(f"  ERROR embedding batch ({len(batch)} pairs): {e}",
                           file=sys.stderr)
                pbar.update(len(batch))
            elapsed = time.perf_counter() - t0
            done_n = pbar.n
            pbar.set_postfix_str(
                f"{done_n/elapsed:.1f} pair/s, err={errors}")

    pbar.close()
    f.close()
    elapsed = time.perf_counter() - t0
    scored = len(tasks) - errors
    print(f"  Embedding done: {scored} scored, {errors} errors, "
          f"{elapsed:.1f}s ({len(tasks)/elapsed:.1f} pair/s)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge + Embedding scoring for dialogue predictions",
    )
    parser.add_argument(
        "--predictions_dir", type=str, required=True,
        help="Directory containing predictions.jsonl (output of predict_prompt.py or predict_hypernet.py)",
    )
    parser.add_argument(
        "--persona_data", type=str, required=True,
        help="Path to the dialogue JSON whose profile_text serves as persona "
             "ground truth for the judge (typically m6_phase_tree/all_dialogues.json)",
    )
    parser.add_argument(
        "--rubric", type=str, default=None,
        help="Path to scoring rubric markdown (default: persona_rubric.md next to this script)",
    )
    parser.add_argument(
        "--judge_model", type=str, default=None,
        help="Judge label + output file (default: JUDGE_MODEL env or gpt-4.1). "
             "Non-gpt-4.1 models write to judge_scores_<model>.jsonl.",
    )
    parser.add_argument(
        "--api_model", type=str, default=None,
        help="Model name sent to the judge API (default: same as --judge_model). "
             "Use when the upstream model differs from the recorded judge_model.",
    )
    parser.add_argument(
        "--judge_scores_file", type=str, default=None,
        help="Override judge output filename (default: auto from --judge_model)",
    )
    parser.add_argument(
        "--skip_embedding", action="store_true",
        help="Only run the LLM judge pass; skip embedding similarity",
    )
    parser.add_argument(
        "--disable_thinking", dest="disable_thinking", action="store_true",
        default=None,
        help="Disable GLM chain-of-thought thinking (extra_body thinking.type=disabled). "
             "Defaults to ON for glm-* models / JUDGE_DISABLE_THINKING=1.",
    )
    parser.add_argument(
        "--enable_thinking", dest="disable_thinking", action="store_false",
        help="Keep GLM thinking enabled even for glm-* models.",
    )
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Parallel workers for the LLM judge pass (and embedding "
                             "pass when --embed_workers is not set)")
    parser.add_argument("--embed_workers", type=int, default=None,
                        help="Parallel workers for the embedding pass; defaults to --num_workers")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Only score first N predictions (for debugging)")
    parser.add_argument(
        "--sample_ids_file", type=str, default=None,
        help="JSON list of question_ids to score (fixed subsample manifest)",
    )
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Max retries per judge API call on failure")
    parser.add_argument("--rate_limit_sleep", type=float, default=0.1,
                        help="Sleep seconds between API calls for rate-limit protection")
    parser.add_argument("--concurrent_passes", dest="concurrent_passes",
                        action="store_true", default=True,
                        help="Run the LLM-judge pass and the embedding pass in parallel "
                             "(they hit independent endpoints). Default: enabled.")
    parser.add_argument("--sequential_passes", dest="concurrent_passes",
                        action="store_false",
                        help="Run passes sequentially (Pass 1 → Pass 2). Useful if both "
                             "passes share a single API endpoint with tight rate limits.")
    args = parser.parse_args()

    # --- Load persona ground-truth data ---
    persona_list = load_json(args.persona_data)
    samples = {s["question_id"]: s for s in persona_list}
    print(f"Loaded {len(samples)} persona ground-truth samples from {args.persona_data}",
          flush=True)

    # --- Load predictions ---
    pred_path = os.path.join(args.predictions_dir, "predictions.jsonl")
    if not os.path.exists(pred_path):
        print(f"ERROR: {pred_path} not found. Run predict_prompt.py or predict_hypernet.py first.", flush=True)
        return
    pred_list = load_jsonl(pred_path)
    if args.num_samples is not None:
        pred_list = pred_list[:args.num_samples]
        print(f"  ⚠ Debug mode: limited to first {args.num_samples} predictions",
              flush=True)
    predictions = {p["question_id"]: p["prediction"] for p in pred_list}
    print(f"Loaded {len(predictions)} predictions from {pred_path}", flush=True)

    if args.sample_ids_file:
        with open(args.sample_ids_file, "r", encoding="utf-8") as f:
            allowed_ids = set(json.load(f))
        before = len(predictions)
        predictions = {k: v for k, v in predictions.items() if k in allowed_ids}
        print(
            f"  Subsample filter: {len(predictions)}/{before} predictions "
            f"(manifest={len(allowed_ids)} ids from {args.sample_ids_file})",
            flush=True,
        )

    # --- API clients ---
    judge_model = args.judge_model or get_env("JUDGE_MODEL") or "gpt-4.1"
    api_model = args.api_model or get_env("JUDGE_API_MODEL") or judge_model
    judge_api_key = get_env("JUDGE_API_KEY") or get_env("OPENAI_API_KEY")
    judge_base_url = get_env("JUDGE_BASE_URL") or get_env("OPENAI_BASE_URL")
    judge_client = OpenAI(api_key=judge_api_key, base_url=judge_base_url)

    embed_model = get_env("EMBED_MODEL") or "text-embedding-3-small"
    embed_api_key = get_env("EMBED_API_KEY") or judge_api_key
    embed_base_url = get_env("EMBED_BASE_URL") or judge_base_url
    embed_client = OpenAI(api_key=embed_api_key, base_url=embed_base_url)

    rubric_path = args.rubric or _default_rubric_path()
    rubric_text = load_rubric(rubric_path)

    embed_workers = args.embed_workers if args.embed_workers is not None else args.num_workers
    disable_thinking = should_disable_thinking(api_model, args.disable_thinking)

    print(f"\n{'─' * 50}", flush=True)
    print(f"  Judge model : {judge_model}", flush=True)
    if api_model != judge_model:
        print(f"  API model   : {api_model}", flush=True)
    print(f"  Judge file  : {args.judge_scores_file or judge_scores_filename(judge_model)}", flush=True)
    print(f"  Thinking    : {'OFF' if disable_thinking else 'ON'}", flush=True)
    print(f"  Embed model : {embed_model}", flush=True)
    print(f"  Rubric      : {rubric_path}", flush=True)
    print(f"  Workers     : judge={args.num_workers}, embed={embed_workers}", flush=True)
    print(f"  Pass mode   : {'CONCURRENT (judge ∥ embed)' if args.concurrent_passes else 'SEQUENTIAL'}", flush=True)
    print(f"  Max retries : {args.max_retries}", flush=True)
    print(f"  Rate limit  : {args.rate_limit_sleep}s/call", flush=True)
    print(f"  Predictions : {len(predictions)}", flush=True)
    print(f"  Persona ref : {len(samples)}", flush=True)
    print(f"{'─' * 50}", flush=True)

    judge_fname = args.judge_scores_file or judge_scores_filename(judge_model)
    judge_path = os.path.join(args.predictions_dir, judge_fname)
    embed_path = os.path.join(args.predictions_dir, "embedding_scores.jsonl")

    def _judge_pass():
        print(f"\n▶ LLM-as-Judge ({judge_model}) → {judge_fname}", flush=True)
        run_llm_judge(
            judge_client, judge_model, samples, predictions,
            judge_path, rubric_text, args.num_workers,
            max_retries=args.max_retries,
            rate_limit_sleep=args.rate_limit_sleep,
            disable_thinking=disable_thinking,
            api_model=api_model,
            record_model=judge_model,
        )

    def _embed_pass():
        print(f"\n▶ Embedding Similarity ({embed_model})", flush=True)
        run_embedding_scoring(
            embed_client, embed_model, samples, predictions,
            embed_path, embed_workers,
        )

    if args.skip_embedding:
        _judge_pass()
    elif args.concurrent_passes:
        # Both passes target independent API endpoints (JUDGE_BASE_URL vs
        # EMBED_BASE_URL), so we can saturate both simultaneously and roughly
        # halve wall-clock time. The two writers append to *separate* JSONL
        # files, so there is no file-handle contention.
        t_judge = threading.Thread(target=_judge_pass, name="judge-pass", daemon=False)
        t_embed = threading.Thread(target=_embed_pass, name="embed-pass", daemon=False)
        t_judge.start()
        t_embed.start()
        t_judge.join()
        t_embed.join()
    else:
        _judge_pass()
        _embed_pass()

    # --- Quick summary ---
    judge_items = load_jsonl(judge_path)
    if judge_items:
        char_scores = [r["character_score"] for r in judge_items]
        sem_scores = [r["semantic_score"] for r in judge_items]
        print(f"\n{'─' * 50}", flush=True)
        print(f"  Quick summary ({len(judge_items)} samples):", flush=True)
        print(f"    Character score: {sum(char_scores)/len(char_scores):.3f} avg", flush=True)
        print(f"    Semantic score:  {sum(sem_scores)/len(sem_scores):.3f} avg", flush=True)
    embed_items = load_jsonl(embed_path)
    if embed_items:
        emb_scores = [r["embedding_similarity"] for r in embed_items]
        print(f"    Embedding sim:   {sum(emb_scores)/len(emb_scores):.4f} avg", flush=True)
    print(f"{'─' * 50}", flush=True)
    print(f"\n✓ All scoring done. Results in: {args.predictions_dir}/", flush=True)


if __name__ == "__main__":
    main()
