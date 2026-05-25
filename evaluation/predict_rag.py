"""Dialogue-continuation prediction with retrieval-augmented prompting.

Two modes are supported via ``--mode``:

* ``rag``  — Retrieval-Augmented Generation.  The prompt contains only
  the retrieved demonstrations (no profile text), matching the standard
  RAG baseline.
* ``pag``  — Profile-Augmented Generation.  The prompt contains BOTH
  the raw character profile AND the retrieved demonstrations.  This is
  the stronger retrieval baseline.

Pool construction
-----------------
The retrieval pool is built from ``phase_tree_data/processed/<DATASET>/m1_context_only/
train.json`` (all training-split dialogues, all characters).  Using the
train split exclusively guarantees no test-time leakage.  Embeddings are
produced via an OpenAI-compatible endpoint configured by
``RETRIEVAL_EMBED_*`` env vars (falling back to ``EMBED_*`` / ``JUDGE_*``).
By default this points to a local Qwen3-Embedding-4B for best Chinese
retrieval quality.  Cached on disk under
``phase_tree_data/processed/<DATASET>/_retrieval_cache/``.

Inference
---------
Inference uses vLLM (continuous batching).  A small HF fallback is
provided for environments without vLLM.

Usage::

    # RAG on RAIDEN random_test
    python evaluation/predict_rag.py \\
        --data        phase_tree_data/processed/RAIDEN/m1_context_only/random_test.json \\
        --pool        phase_tree_data/processed/RAIDEN/m1_context_only/train.json \\
        --output_dir  results/RAIDEN/comparison/main/rag/random_test \\
        --mode rag --top_k 5

    # PAG on Friends ood_test
    python evaluation/predict_rag.py \\
        --data        phase_tree_data/processed/Friends/m1_context_only/ood_test.json \\
        --profile_data phase_tree_data/processed/Friends/m2_raw_profile/all_dialogues.json \\
        --pool        phase_tree_data/processed/Friends/m1_context_only/train.json \\
        --output_dir  results/Friends/comparison/main/pag/ood_test \\
        --mode pag --top_k 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env (RETRIEVAL_EMBED_* for retrieval, EMBED_* used by judge.py)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass

from retrieval import (RetrievalPool, LocalEmbedClient,  # noqa: E402
                       format_demonstrations)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

RAG_PROMPT = """\
Below is a multi-turn dialogue. Your task is to predict the single line that {character} would most likely say next.

To help you, here are retrieved examples of similar dialogue moments from past conversations:

{demos}

Use these examples to infer {character}'s tone, speaking style, and likely intent. Keep the reply short and natural, matching the tone and length of the other lines. Output only that one line, no explanation.

Dialogue context:
{context}

{character}:"""


PAG_PROMPT = """\
Below is a multi-turn dialogue. Your task is to predict the single line that {character} would most likely say next.

Character profile for {character}:
{profile}

Here are retrieved examples of similar dialogue moments from past conversations:

{demos}

Use both the profile and the examples to infer {character}'s tone, speaking style, and likely intent. Keep the reply short and natural, matching the tone and length of the other lines. Output only that one line, no explanation.

Dialogue context:
{context}

{character}:"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_done_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["question_id"])
    return ids


def load_profile_lookup(path: str) -> dict[str, str]:
    """Load a {question_id: profile_text} mapping from a processed dataset."""
    out: dict[str, str] = {}
    for s in load_data(path):
        qid = s.get("question_id")
        if qid is None:
            continue
        out[qid] = (s.get("profile_text") or "").strip()
    return out


# ---------------------------------------------------------------------------
# Embedding API client (matches judge.py env-var convention)
# ---------------------------------------------------------------------------

def _make_embed_client():
    """Build an OpenAI-compatible embedding client from env vars.

    Priority: ``RETRIEVAL_EMBED_*`` → ``EMBED_*`` → ``JUDGE_*`` fallback.
    This allows using a local Qwen3-Embedding-4B for retrieval while
    ``judge.py`` continues to use the OpenAI embedding endpoint for scoring.
    """
    from openai import OpenAI

    def _ge(k: str) -> str | None:
        v = os.environ.get(k)
        return v if (v and v.strip()) else None

    embed_model = (_ge("RETRIEVAL_EMBED_MODEL")
                   or _ge("EMBED_MODEL")
                   or "text-embedding-3-small")
    embed_api_key = (_ge("RETRIEVAL_EMBED_API_KEY")
                     or _ge("EMBED_API_KEY")
                     or _ge("JUDGE_API_KEY")
                     or "EMPTY")
    embed_base_url = (_ge("RETRIEVAL_EMBED_BASE_URL")
                      or _ge("EMBED_BASE_URL")
                      or _ge("JUDGE_BASE_URL"))
    client = OpenAI(api_key=embed_api_key, base_url=embed_base_url)
    return client, embed_model


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(sample: dict, demos_text: str, mode: str,
                 profile_lookup: dict[str, str] | None) -> str:
    character = sample["role"]
    context = sample["input"]
    if mode == "rag":
        return RAG_PROMPT.format(
            character=character, context=context, demos=demos_text)
    if mode == "pag":
        if profile_lookup is None:
            raise ValueError("--profile_data required for --mode pag")
        profile = profile_lookup.get(sample["question_id"], "").strip()
        if not profile:
            profile = "(profile not available)"
        return PAG_PROMPT.format(
            character=character, context=context,
            profile=profile, demos=demos_text)
    raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Retrieval pre-computation
# ---------------------------------------------------------------------------

def precompute_retrievals(pool: RetrievalPool,
                          queries: list[dict],
                          top_k: int,
                          exclude_same_scene: bool = True,
                          ) -> tuple[dict[str, str], dict]:
    """Encode all queries in batch, retrieve top-K, format demonstrations.

    Parameters
    ----------
    exclude_same_scene : bool
        When True (default) the retrieval excludes any pool sample whose
        scene fingerprint matches the query's, on top of the usual
        self-id filter.  Requires the pool to be built with
        ``scene_window > 0``.  Setting False reproduces the legacy
        behavior (used for ablations / regression checks).

    Returns
    -------
    demos_lookup : dict[str, str]
        ``{question_id: demonstrations_text}``.
    stats : dict
        Diagnostics: ``n_queries``, ``n_with_scene_match``,
        ``mean_excluded_per_query``, ``max_excluded_per_query``.  Useful
        for the meta.json audit trail.
    """
    print(f"\n  Encoding {len(queries)} queries ...", flush=True)
    q_embs = pool.encode_queries_batch(queries)

    desc = (f"retrieve (scene-filter on, window={pool.scene_window})"
            if exclude_same_scene and pool.scene_window > 0
            else "retrieve")
    print(f"  Retrieving top-{top_k} for each query ...", flush=True)
    out: dict[str, str] = {}
    n_with_scene_match = 0
    excluded_counts: list[int] = []
    for q, emb in tqdm(zip(queries, q_embs), total=len(queries),
                       desc=desc, unit="query"):
        excl: set[str] = {q["question_id"]}
        if exclude_same_scene and pool.scene_window > 0:
            scene_qs = pool.scene_qids_for(q)
            scene_qs.discard(q["question_id"])
            if scene_qs:
                n_with_scene_match += 1
                excluded_counts.append(len(scene_qs))
                excl |= scene_qs
        hits = pool.query_top_k(emb, k=top_k, exclude_qids=excl)
        out[q["question_id"]] = format_demonstrations(hits)

    stats = {
        "n_queries": len(queries),
        "exclude_same_scene": bool(exclude_same_scene
                                    and pool.scene_window > 0),
        "scene_window": pool.scene_window,
        "n_queries_with_same_scene_pool_sample": n_with_scene_match,
        "mean_pool_samples_excluded": (
            round(sum(excluded_counts) / len(excluded_counts), 2)
            if excluded_counts else 0.0),
        "max_pool_samples_excluded": (max(excluded_counts)
                                      if excluded_counts else 0),
    }
    return out, stats


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------

def run_vllm(args, remaining: list[dict], pred_path: str,
             demos_lookup: dict[str, str],
             profile_lookup: dict[str, str] | None) -> float:
    import torch
    from vllm import LLM, SamplingParams

    n_gpus = torch.cuda.device_count()
    tp = min(n_gpus, args.tensor_parallel) if args.tensor_parallel else n_gpus
    print(f"  vLLM: tensor_parallel={tp}, max_model_len={args.max_model_len}",
          flush=True)

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=tp,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=args.seed,
    )
    if getattr(args, "gpu_memory_utilization", None):
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    prompts: list[str] = []
    for s in remaining:
        demos = demos_lookup.get(s["question_id"], "(no retrieval)")
        raw = build_prompt(s, demos, args.mode, profile_lookup)
        messages = [{"role": "user", "content": raw}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompts.append(text)

    print(f"\n  Generating {len(prompts)} predictions ...", flush=True)
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    elapsed = time.perf_counter() - t0

    with open(pred_path, "a", encoding="utf-8") as f:
        for s, out in zip(remaining, outputs):
            pred = out.outputs[0].text.strip()
            record = {
                "question_id": s["question_id"],
                "role": s["role"],
                "prediction": pred,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    speed = len(remaining) / elapsed if elapsed > 0 else 0
    print(f"\n✓ Predictions done: {len(remaining)} samples in {elapsed:.1f}s "
          f"({speed:.1f} samples/s)", flush=True)
    print(f"  Output: {pred_path}", flush=True)

    del llm
    torch.cuda.empty_cache()
    import gc; gc.collect()
    return elapsed


# ---------------------------------------------------------------------------
# HuggingFace fallback
# ---------------------------------------------------------------------------

def run_hf(args, remaining: list[dict], pred_path: str,
           demos_lookup: dict[str, str],
           profile_lookup: dict[str, str] | None) -> float:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"  Loading model: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    f = open(pred_path, "a", encoding="utf-8")
    t0 = time.perf_counter()
    n_done = 0

    pbar = tqdm(total=len(remaining), desc="predict", unit="sample",
                file=sys.stderr, dynamic_ncols=True)

    for i in range(0, len(remaining), args.batch_size):
        batch = remaining[i: i + args.batch_size]
        prompts = []
        for s in batch:
            demos = demos_lookup.get(s["question_id"], "(no retrieval)")
            prompts.append(build_prompt(s, demos, args.mode, profile_lookup))

        messages_batch = [[{"role": "user", "content": p}] for p in prompts]
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=args.max_model_len,
        ).to(model.device)

        gen_kwargs: dict = dict(
            max_new_tokens=args.max_tokens,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
        if args.temperature > 0:
            gen_kwargs["temperature"] = args.temperature

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        for s, ids in zip(batch, output_ids):
            new_ids = ids[inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            record = {
                "question_id": s["question_id"],
                "role": s["role"],
                "prediction": text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()

        n_done += len(batch)
        pbar.update(len(batch))
        elapsed = time.perf_counter() - t0
        speed = n_done / elapsed
        pbar.set_postfix_str(f"{speed:.1f} samples/s")

    pbar.close()
    f.close()
    elapsed = time.perf_counter() - t0
    print(f"\n✓ Predictions done: {n_done} samples in {elapsed:.1f}s "
          f"({n_done/elapsed:.1f} samples/s)", flush=True)
    print(f"  Output: {pred_path}", flush=True)

    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    return elapsed


# ---------------------------------------------------------------------------
# Token statistics
# ---------------------------------------------------------------------------

def _arr_stats(values: list[int | float]) -> dict:
    import numpy as np
    arr = np.array(values)
    return {
        "mean": round(float(arr.mean()), 1),
        "std": round(float(arr.std()), 1),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "median": round(float(np.median(arr)), 1),
        "total": int(arr.sum()),
    }


_tokenizer_cache: dict = {}


def _get_tokenizer(model_path: str):
    if model_path not in _tokenizer_cache:
        from transformers import AutoTokenizer
        _tokenizer_cache[model_path] = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
    return _tokenizer_cache[model_path]


def _load_cached_token_stats(output_dir: str, expected_n: int) -> dict | None:
    meta_path = os.path.join(output_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        ts = meta.get("token_stats")
        if ts and ts.get("num_samples") == expected_n:
            return ts
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _compute_prediction_token_stats(pred_path: str, model_path: str) -> dict | None:
    if not os.path.exists(pred_path):
        return None
    predictions = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line).get("prediction", ""))
    if not predictions:
        return None
    tok = _get_tokenizer(model_path)
    pred_lens = [len(tok.encode(p)) for p in predictions]
    stats = _arr_stats(pred_lens)
    stats["num_predictions"] = len(predictions)
    return stats


def _compute_token_stats(samples: list[dict], model_path: str,
                         mode: str, demos_lookup: dict[str, str],
                         profile_lookup: dict[str, str] | None) -> dict:
    tok = _get_tokenizer(model_path)

    profile_lens, context_lens, output_lens = [], [], []
    demos_lens, prompt_lens = [], []

    for s in samples:
        profile_text = (s.get("profile_text") or "").strip()
        if not profile_text and profile_lookup is not None:
            profile_text = profile_lookup.get(s["question_id"], "")
        profile_lens.append(len(tok.encode(profile_text)))
        context_lens.append(len(tok.encode(s.get("input", ""))))
        output_lens.append(len(tok.encode(s.get("output", ""))))

        demos_text = demos_lookup.get(s["question_id"], "")
        demos_lens.append(len(tok.encode(demos_text)))

        raw_prompt = build_prompt(s, demos_text, mode, profile_lookup)
        messages = [{"role": "user", "content": raw_prompt}]
        full_prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        prompt_lens.append(len(tok.encode(full_prompt)))

    return {
        "profile_tokens": _arr_stats(profile_lens),
        "context_tokens": _arr_stats(context_lens),
        "output_tokens": _arr_stats(output_lens),
        "demos_tokens": _arr_stats(demos_lens),
        "prompt_tokens": _arr_stats(prompt_lens),
        "tokenizer": model_path,
        "num_samples": len(samples),
    }


# ---------------------------------------------------------------------------
# Meta serialization (mirrors predict_prompt.py)
# ---------------------------------------------------------------------------

def _save_meta(args, total_samples: int, predicted_samples: int,
               pool_size: int, embed_model: str,
               retrieval_seconds: float | None,
               latency_stats: dict | None,
               token_stats: dict | None = None,
               leak_stats: dict | None = None) -> None:
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass

    meta = {
        "method": args.mode,
        "model": args.model,
        "backend": args.backend,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "tensor_parallel": args.tensor_parallel,
        "total_samples": total_samples,
        "predicted_this_run": predicted_samples,
        "data_path": args.data,
        "pool_path": args.pool,
        "pool_size": pool_size,
        "top_k": args.top_k,
        "embed_model": embed_model,
        "profile_data": args.profile_data,
        "exclude_same_scene": (
            False if getattr(args, "no_exclude_scene", False)
            else True),
        "scene_window": getattr(args, "scene_window", 0),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_hash": git_hash,
    }
    if retrieval_seconds is not None:
        meta["retrieval_seconds"] = round(retrieval_seconds, 2)
    if latency_stats:
        meta["latency"] = latency_stats
    if token_stats:
        meta["token_stats"] = token_stats
    if leak_stats:
        meta["leakage_filter"] = leak_stats

    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  Meta saved: {meta_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG / PAG dialogue-continuation prediction",
    )
    parser.add_argument("--data", required=True,
                        help="Path to the test/eval JSON (any m1-m6 file works; "
                             "only role/input/output/question_id are read)")
    parser.add_argument("--pool", required=True,
                        help="Path to the pool JSON (typically "
                             "<dataset>/m1_context_only/train.json)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for predictions.jsonl + meta.json")
    parser.add_argument("--mode", required=True, choices=["rag", "pag"],
                        help="rag = retrieved-only; pag = retrieved + raw profile")
    parser.add_argument("--profile_data", default=None,
                        help="Path to raw-profile JSON (required for --mode pag, "
                             "typically <dataset>/m2_raw_profile/all_dialogues.json)")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of demonstrations to retrieve per query "
                             "(default 3; matches the RAIDEN runs and keeps "
                             "prompts within the same length budget across "
                             "datasets for an apples-to-apples comparison).")
    parser.add_argument("--cache_dir", default=None,
                        help="Embedding cache dir (default: <dataset>/_retrieval_cache/)")

    parser.add_argument("--model", default=None,
                        help="LLM path (default: PHASE-Tree/models/Qwen2.5-7B-Instruct)")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    parser.add_argument("--device", default="cuda",
                        help="Device for HF backend")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for HF backend (vLLM auto-batches)")
    parser.add_argument("--tensor_parallel", type=int, default=0,
                        help="Tensor parallel size for vLLM (0 = auto)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=None,
                        help="vLLM gpu_memory_utilization (default: vLLM default 0.9)")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--max_model_len", type=int, default=16384,
                        help="vLLM context window. RAG/PAG prompts are "
                             "~1.5–2× longer than plain prompt baselines, so "
                             "8192 is the recommended floor.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Only process first N samples (debug)")

    parser.add_argument("--embed_model", default=None,
                        help="Local embedding model path (e.g. /dev/shm/"
                             "Qwen3-Embedding-4B). If set, loads in-process "
                             "on GPU, encodes, then frees memory before LLM. "
                             "If unset, falls back to RETRIEVAL_EMBED_* API.")
    parser.add_argument("--embed_batch_size", type=int, default=256,
                        help="Outer scheduling batch (texts handed to one "
                             "_encode_batch call). The actual per-forward "
                             "tensor size is controlled by "
                             "--embed_st_batch_size.")
    parser.add_argument("--embed_st_batch_size", type=int, default=8,
                        help="SentenceTransformer's *internal* mini-batch. "
                             "Long-context dialogue queries can have huge "
                             "attention/activation tensors so 4-8 is "
                             "recommended on 80GB cards (was hard-coded to "
                             "32 previously, which OOM'd at 70+ GB/process).")
    parser.add_argument("--embed_devices", default=None,
                        help="Comma-separated CUDA device list for "
                             "SentenceTransformer encoding. Example: "
                             "'cuda:0,cuda:1,cuda:2,cuda:3'. If set with "
                             ">1 device, encoding runs across all of them "
                             "via SentenceTransformer.encode_multi_process. "
                             "Default: single device on the visible GPU.")
    parser.add_argument("--embed_workers", type=int, default=8,
                        help="Pool/query encoding parallelism")

    # ---- Same-scene leakage filter --------------------------------------
    # Utterance-level random splits (used by all 8 datasets here) routinely
    # put different utterances of the same scene into different splits.
    # When that happens, retrieving a "later" pool sample exposes the
    # query's ground-truth output as part of the pool sample's *input*
    # context — measured at 77.3% on TheOffice random_test, vs 1.6% on
    # ood_test.  We mitigate by hashing each sample's first
    # ``--scene_window`` non-empty context lines and excluding pool samples
    # that share a fingerprint with the query.  Defaults are conservative
    # and reproduce-friendly: filter ON, window = 3 lines.
    #
    # Validation on TheOffice (offline check; see commit message):
    #   window=2: blocks 76.8% of GT-leak paths on random_test, hits 0.5%
    #             of ood_test (potentially false-positive).
    #   window=3: same 76.8% block rate, ood_test false-positive rate 0%.
    #   window=4: block rate drops to 66.6% (fingerprint becomes too
    #             specific and fails to group same-scene utterances).
    parser.add_argument("--scene_window", type=int, default=3,
                        help="First N non-empty context lines used as the "
                             "scene fingerprint (0 = disable filter; "
                             "default 3 was validated on TheOffice and "
                             "yields 0%% false-positive on ood_test).")
    parser.add_argument("--no_exclude_scene", action="store_true",
                        help="Disable the same-scene retrieval filter "
                             "(reproduces the legacy / leaky behavior; "
                             "useful only for ablations).")
    args = parser.parse_args()

    if args.model is None:
        args.model = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-7B-Instruct")
    if args.mode == "pag" and not args.profile_data:
        parser.error("--profile_data is required for --mode pag")

    # ---- Load data ------------------------------------------------------
    samples = load_data(args.data)
    if args.num_samples is not None:
        samples = samples[:args.num_samples]
        print(f"  ⚠ Debug mode: limited to first {args.num_samples} samples",
              flush=True)
    pool_samples = load_data(args.pool)

    profile_lookup: dict[str, str] | None = None
    if args.mode == "pag":
        profile_lookup = load_profile_lookup(args.profile_data)
        print(f"  Profile lookup: {len(profile_lookup)} entries from "
              f"{args.profile_data}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    done_ids = load_done_ids(pred_path)
    remaining = [s for s in samples if s["question_id"] not in done_ids]

    n_roles = len(set(s["role"] for s in samples))
    print(f"\n{'─' * 50}", flush=True)
    print(f"  Method      : {args.mode.upper()}", flush=True)
    print(f"  Data        : {args.data}", flush=True)
    print(f"  Pool        : {args.pool}  ({len(pool_samples)} samples)", flush=True)
    print(f"  Top-K       : {args.top_k}", flush=True)
    print(f"  Samples     : {len(samples)} ({n_roles} characters)", flush=True)
    _embed_tag = args.embed_model or "API"
    print(f"  Embed       : {_embed_tag}", flush=True)
    print(f"  Backend     : {args.backend}", flush=True)
    print(f"  Output      : {args.output_dir}/", flush=True)
    print(f"  Progress    : {len(done_ids)}/{len(samples)} done, "
          f"{len(remaining)} remaining", flush=True)
    print(f"{'─' * 50}", flush=True)

    if not remaining:
        print(f"\n✓ All {len(samples)} predictions already done.", flush=True)
        cached_ts = _load_cached_token_stats(args.output_dir, len(samples))
        if not cached_ts:
            cached_ts = None
        pred_ts = _compute_prediction_token_stats(pred_path, args.model)
        if cached_ts and pred_ts:
            cached_ts["prediction_tokens"] = pred_ts
        _save_meta(args, len(samples), 0, len(pool_samples),
                   embed_model="(unused)", retrieval_seconds=None,
                   latency_stats=None, token_stats=cached_ts)
        return

    # ---- Build / load retrieval pool ------------------------------------
    local_embed = args.embed_model is not None
    if local_embed:
        # Parse the optional comma-separated device list for multi-GPU
        # encoding; fall back to single-GPU on the visible CUDA card.
        if args.embed_devices:
            _devs = [d.strip() for d in args.embed_devices.split(",")
                     if d.strip()]
            # Allow plain ints "0,1,2" as a convenience.
            _devs = [d if d.startswith("cuda") or d == "cpu"
                     else f"cuda:{d}" for d in _devs]
        else:
            _devs = ["cuda"]
        embed_client = LocalEmbedClient(
            args.embed_model,
            devices=_devs,
            st_batch_size=args.embed_st_batch_size,
        )
        embed_model = os.path.basename(args.embed_model)
    else:
        embed_client, embed_model = _make_embed_client()

    cache_dir = args.cache_dir or os.path.join(
        os.path.dirname(args.pool), "..", "_retrieval_cache")
    cache_dir = os.path.normpath(cache_dir)
    cache_path = os.path.join(
        cache_dir, f"pool_{embed_model.replace('/', '_')}.npz")

    exclude_same_scene = not args.no_exclude_scene
    effective_scene_window = args.scene_window if exclude_same_scene else 0
    pool = RetrievalPool(
        samples=pool_samples,
        embed_client=embed_client,
        embed_model=embed_model,
        batch_size=args.embed_batch_size,
        num_workers=args.embed_workers if not local_embed else 1,
        cache_path=cache_path,
        scene_window=effective_scene_window,
    )

    t_retr_0 = time.perf_counter()
    pool.build_or_load()

    # ---- Pre-compute retrievals for the remaining queries ---------------
    demos_lookup, leak_stats = precompute_retrievals(
        pool, remaining, args.top_k,
        exclude_same_scene=exclude_same_scene,
    )
    retrieval_seconds = time.perf_counter() - t_retr_0
    print(f"  Same-scene filter: enabled={leak_stats['exclude_same_scene']}, "
          f"window={leak_stats['scene_window']}, "
          f"queries with collision={leak_stats['n_queries_with_same_scene_pool_sample']}/"
          f"{leak_stats['n_queries']}, "
          f"mean excluded={leak_stats['mean_pool_samples_excluded']}, "
          f"max excluded={leak_stats['max_pool_samples_excluded']}",
          flush=True)

    if local_embed:
        embed_client.unload()

    # ---- Inference ------------------------------------------------------
    if args.backend == "vllm":
        try:
            import vllm as _vllm  # noqa: F401
        except Exception as e:
            print(f"\n⚠ vLLM unavailable ({e}), falling back to HF backend.",
                  flush=True)
            args.backend = "hf"

    if args.backend == "vllm":
        infer_elapsed = run_vllm(args, remaining, pred_path,
                                 demos_lookup, profile_lookup)
    else:
        infer_elapsed = run_hf(args, remaining, pred_path,
                               demos_lookup, profile_lookup)

    latency_stats = {
        "retrieval_seconds": round(retrieval_seconds, 2),
        "inference_seconds": round(infer_elapsed, 2),
        "total_seconds": round(retrieval_seconds + infer_elapsed, 2),
        "num_predicted": len(remaining),
        "mean_ms_per_sample": round(infer_elapsed / len(remaining) * 1000, 1),
        "samples_per_second": round(len(remaining) / infer_elapsed, 2)
                              if infer_elapsed > 0 else 0,
    }
    print(f"\n  Computing token statistics ...", flush=True)
    token_stats = _compute_token_stats(
        remaining, args.model, args.mode, demos_lookup, profile_lookup)
    pred_token_stats = _compute_prediction_token_stats(pred_path, args.model)
    if pred_token_stats:
        token_stats["prediction_tokens"] = pred_token_stats
    print(f"  Token stats: profile={token_stats['profile_tokens']['mean']:.1f}, "
          f"context={token_stats['context_tokens']['mean']:.1f}, "
          f"output_gt={token_stats['output_tokens']['mean']:.1f}, "
          f"demos={token_stats['demos_tokens']['mean']:.1f}, "
          f"prompt={token_stats['prompt_tokens']['mean']:.1f} (mean tokens)",
          flush=True)

    _save_meta(args, len(samples), len(remaining), len(pool_samples),
               embed_model=embed_model, retrieval_seconds=retrieval_seconds,
               latency_stats=latency_stats, token_stats=token_stats,
               leak_stats=leak_stats)


if __name__ == "__main__":
    main()
