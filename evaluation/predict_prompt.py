"""Dialogue-continuation prediction with vLLM (fast) or HuggingFace (fallback).

Two inference backends are supported:

* ``vllm`` (default) — continuous batching with automatic tensor parallelism.
  Much faster than HF, especially for large sample counts.
* ``hf`` — HuggingFace ``model.generate()`` with left-padded batching.
  Simpler dependency chain; useful when vLLM is unavailable.

Each sample's ``profile_text`` is injected into the prompt so the model can
produce persona-aware responses.  Two prompt modes are supported:

  * **profile** (default): injects the per-sample ``profile_text``.
  * **baseline**: context-only prompt without any profile.

Predictions are written incrementally to ``predictions.jsonl`` under the
specified output directory.  Existing predictions are skipped on re-run
(resume support).

Usage examples::

    # vLLM backend — predict on random_test split
    python evaluation/predict_prompt.py \\
        --data  phase_tree_data/processed/RAIDEN/m6_phase_tree/random_test.json \\
        --output_dir results/RAIDEN/prompt/main/m6_phase_tree/random_test

    # Explicit HF backend — ood_test split
    python evaluation/predict_prompt.py \\
        --data  phase_tree_data/processed/RAIDEN/m2_raw_profile/ood_test.json \\
        --output_dir results/RAIDEN/prompt/main/m2_raw_profile/ood_test \\
        --backend hf --batch_size 8

    # Baseline prompt (no profile)
    python evaluation/predict_prompt.py \\
        --data  phase_tree_data/processed/RAIDEN/m1_context_only/random_test.json \\
        --output_dir results/RAIDEN/prompt/main/m1_context_only/random_test \\
        --prompt_mode baseline
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

BASELINE_PROMPT = """\
Below is a multi-turn dialogue. Predict the single line that {character} would most likely say next.
Keep the reply short and natural, matching the tone and length of the other lines. Output only that one line, no explanation.

Dialogue context:
{context}

{character}:"""

PROFILE_PROMPT = """\
Below is a multi-turn dialogue. Predict the single line that {character} would most likely say next.

Character profile for {character}:
{profile}

Keep the reply short and natural, matching the tone and length of the other lines. Output only that one line, no explanation.

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
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["question_id"])
    return ids


def build_prompt(sample: dict, prompt_mode: str) -> str:
    character = sample["role"]
    context = sample["input"]
    if prompt_mode == "profile":
        return PROFILE_PROMPT.format(
            character=character,
            context=context,
            profile=sample.get("profile_text", "").strip(),
        )
    return BASELINE_PROMPT.format(character=character, context=context)


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------

def run_vllm(args, remaining: list[dict], pred_path: str) -> float:
    """Run vLLM inference. Returns total elapsed seconds."""
    import torch
    from vllm import LLM, SamplingParams

    n_gpus = torch.cuda.device_count()
    tp = min(n_gpus, args.tensor_parallel) if args.tensor_parallel else n_gpus
    print(f"  vLLM: tensor_parallel={tp}, max_model_len={args.max_model_len}", flush=True)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=tp,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    prompts = []
    for s in remaining:
        raw = build_prompt(s, args.prompt_mode)
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
    return elapsed


# ---------------------------------------------------------------------------
# HuggingFace backend
# ---------------------------------------------------------------------------

def run_hf(args, remaining: list[dict], pred_path: str) -> float:
    """Run HuggingFace inference. Returns total elapsed seconds."""
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
    print(f"  Model loaded on {args.device}", flush=True)

    f = open(pred_path, "a", encoding="utf-8")
    t0 = time.perf_counter()
    n_done = 0

    pbar = tqdm(total=len(remaining), desc="predict", unit="sample",
                file=sys.stderr, dynamic_ncols=True)

    for i in range(0, len(remaining), args.batch_size):
        batch = remaining[i: i + args.batch_size]
        prompts = [build_prompt(s, args.prompt_mode) for s in batch]

        messages_batch = [[{"role": "user", "content": p}] for p in prompts]
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=4096,
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
    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run dialogue-continuation predictions",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to the processed dialogue JSON file",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory for prediction outputs (predictions.jsonl)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to the model (default: PHASE-Tree/models/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--backend", type=str, default="vllm", choices=["vllm", "hf"],
        help="Inference backend: 'vllm' (fast, default) or 'hf' (fallback)",
    )
    parser.add_argument("--prompt_mode", type=str, default="profile",
                        choices=["profile", "baseline"],
                        help="Prompt mode: 'profile' injects profile_text, "
                             "'baseline' uses context only")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for HF backend (ignored by vLLM)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for HF backend (vLLM handles batching automatically)")
    parser.add_argument("--tensor_parallel", type=int, default=0,
                        help="Tensor parallel size for vLLM (0 = auto-detect GPU count)")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Max generation tokens (256 covers 95%+ of all datasets)")
    parser.add_argument("--max_model_len", type=int, default=16384,
                        help="vLLM context window (default 16384; Qwen2.5-7B "
                             "supports up to 32K). Long-term datasets like "
                             "Friends m6 have ~4K-token profiles, so 4096 was "
                             "too tight and produced empty outputs.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Only process first N samples (for debugging)")
    args = parser.parse_args()

    if args.model is None:
        args.model = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-7B-Instruct")

    samples = load_data(args.data)
    if args.num_samples is not None:
        samples = samples[:args.num_samples]
        print(f"  ⚠ Debug mode: limited to first {args.num_samples} samples",
              flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    done_ids = load_done_ids(pred_path)
    remaining = [s for s in samples if s["question_id"] not in done_ids]

    n_roles = len(set(s["role"] for s in samples))
    print(f"\n{'─' * 50}", flush=True)
    print(f"  Data        : {args.data}", flush=True)
    print(f"  Samples     : {len(samples)} ({n_roles} characters)", flush=True)
    print(f"  Backend     : {args.backend}", flush=True)
    print(f"  Prompt mode : {args.prompt_mode}", flush=True)
    print(f"  Seed        : {args.seed}", flush=True)
    if args.backend == "hf":
        print(f"  Batch size  : {args.batch_size}", flush=True)
    print(f"  Output      : {args.output_dir}/", flush=True)
    print(f"  Progress    : {len(done_ids)}/{len(samples)} done, "
          f"{len(remaining)} remaining", flush=True)
    print(f"{'─' * 50}", flush=True)

    # --- Compute token statistics (skip if meta.json already has valid stats) ---
    token_stats = _load_cached_token_stats(args.output_dir, len(samples))
    if token_stats:
        print(f"\n  Token stats (cached): profile={token_stats['profile_tokens']['mean']:.1f}, "
              f"context={token_stats['context_tokens']['mean']:.1f}, "
              f"output_gt={token_stats['output_tokens']['mean']:.1f}, "
              f"prompt={token_stats['prompt_tokens']['mean']:.1f} (mean tokens)",
              flush=True)
    else:
        print(f"\n  Computing token statistics ...", flush=True)
        token_stats = _compute_token_stats(samples, args.model, args.prompt_mode)
        print(f"  Token stats: profile={token_stats['profile_tokens']['mean']:.1f}, "
              f"context={token_stats['context_tokens']['mean']:.1f}, "
              f"output_gt={token_stats['output_tokens']['mean']:.1f}, "
              f"prompt={token_stats['prompt_tokens']['mean']:.1f} (mean tokens)",
              flush=True)

    if not remaining:
        print(f"\n✓ All {len(samples)} predictions already done.", flush=True)
        pred_token_stats = _compute_prediction_token_stats(pred_path, args.model)
        if pred_token_stats:
            token_stats["prediction_tokens"] = pred_token_stats
        _save_meta(args, len(samples), 0, token_stats)
        return

    if args.backend == "vllm":
        try:
            import vllm as _vllm  # noqa: F401
        except Exception as e:
            print(f"\n⚠ vLLM unavailable ({e}), falling back to HF backend.",
                  flush=True)
            args.backend = "hf"

    if args.backend == "vllm":
        infer_elapsed = run_vllm(args, remaining, pred_path)
    else:
        infer_elapsed = run_hf(args, remaining, pred_path)

    # Compute actual prediction token stats from predictions.jsonl
    pred_token_stats = _compute_prediction_token_stats(pred_path, args.model)
    if pred_token_stats:
        token_stats["prediction_tokens"] = pred_token_stats

    # Record latency
    latency_stats = {
        "total_seconds": round(infer_elapsed, 2),
        "num_predicted": len(remaining),
        "mean_ms_per_sample": round(infer_elapsed / len(remaining) * 1000, 1),
        "samples_per_second": round(len(remaining) / infer_elapsed, 2)
                              if infer_elapsed > 0 else 0,
    }

    _save_meta(args, len(samples), len(remaining), token_stats, latency_stats)


def _arr_stats(values: list[int | float]) -> dict:
    """Compute descriptive stats for a list of numbers."""
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


def _load_cached_token_stats(output_dir: str, expected_n: int) -> dict | None:
    """Load token_stats from existing meta.json if sample count matches."""
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


_tokenizer_cache: dict = {}


def _get_tokenizer(model_path: str):
    """Return a cached tokenizer instance (loaded at most once per model)."""
    if model_path not in _tokenizer_cache:
        from transformers import AutoTokenizer
        _tokenizer_cache[model_path] = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
    return _tokenizer_cache[model_path]


def _compute_token_stats(samples: list[dict], model_path: str,
                          prompt_mode: str) -> dict:
    """Compute exact token counts using the model's tokenizer."""
    tokenizer = _get_tokenizer(model_path)

    profile_lens = []
    context_lens = []
    output_lens = []
    prompt_lens = []

    for s in samples:
        profile_text = s.get("profile_text", "")
        context_text = s.get("input", "")
        output_text = s.get("output", "")

        profile_lens.append(len(tokenizer.encode(profile_text)))
        context_lens.append(len(tokenizer.encode(context_text)))
        output_lens.append(len(tokenizer.encode(output_text)))

        raw_prompt = build_prompt(s, prompt_mode)
        messages = [{"role": "user", "content": raw_prompt}]
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_lens.append(len(tokenizer.encode(full_prompt)))

    return {
        "profile_tokens": _arr_stats(profile_lens),
        "context_tokens": _arr_stats(context_lens),
        "output_tokens": _arr_stats(output_lens),
        "prompt_tokens": _arr_stats(prompt_lens),
        "tokenizer": model_path,
        "num_samples": len(samples),
    }


def _compute_prediction_token_stats(pred_path: str, model_path: str) -> dict | None:
    """Compute token lengths of actual model predictions from predictions.jsonl."""
    if not os.path.exists(pred_path):
        return None

    predictions = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line)["prediction"])

    if not predictions:
        return None

    tokenizer = _get_tokenizer(model_path)

    pred_lens = [len(tokenizer.encode(p)) for p in predictions]
    stats = _arr_stats(pred_lens)
    stats["num_predictions"] = len(predictions)
    return stats


def _save_meta(args, total_samples: int, predicted_samples: int,
               token_stats: dict | None = None,
               latency_stats: dict | None = None):
    """Write meta.json alongside predictions for reproducibility."""
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass

    meta = {
        "model": args.model,
        "backend": args.backend,
        "prompt_mode": args.prompt_mode,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "tensor_parallel": args.tensor_parallel,
        "total_samples": total_samples,
        "predicted_this_run": predicted_samples,
        "data_path": args.data,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_hash": git_hash,
    }
    if token_stats:
        meta["token_stats"] = token_stats
    if latency_stats:
        meta["latency"] = latency_stats

    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  Meta saved: {meta_path}", flush=True)


# ---------------------------------------------------------------------------
# Multi-task mode: load vLLM once, run multiple tasks
# ---------------------------------------------------------------------------

def multi_main():
    """Run multiple tasks on a single GPU with vLLM loaded only once.

    Usage::

        python evaluation/predict_prompt.py --multi \\
            --tasks tasks.json \\
            --model /dev/shm/Qwen2.5-7B-Instruct

    ``tasks.json`` is a JSON list::

        [
          {"data": "phase_tree_data/.../m2_raw_profile/random_test.json",
           "output_dir": "results/.../m2_raw_profile/random_test",
           "prompt_mode": "profile"},
          {"data": "phase_tree_data/.../m1_context_only/ood_test.json",
           "output_dir": "results/.../m1_context_only/ood_test",
           "prompt_mode": "baseline"}
        ]
    """
    parser = argparse.ArgumentParser(
        description="Multi-task predictions: vLLM loaded once per GPU",
    )
    parser.add_argument("--multi", action="store_true")
    parser.add_argument("--tasks", required=True,
                        help="JSON file listing tasks [{data, output_dir, "
                             "prompt_mode?}, ...]")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["vllm", "hf"])
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--max_model_len", type=int, default=16384,
                        help="vLLM context window. See single-task help for details.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor_parallel", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    if args.model is None:
        args.model = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-7B-Instruct")

    with open(args.tasks) as f:
        tasks = json.load(f)
    assert isinstance(tasks, list) and tasks, "--tasks must be a non-empty JSON list"

    # Collect task data
    task_data = []
    total_remaining = 0
    for tc in tasks:
        samples = load_data(tc["data"])
        if args.num_samples is not None:
            samples = samples[:args.num_samples]
        os.makedirs(tc["output_dir"], exist_ok=True)
        pred_path = os.path.join(tc["output_dir"], "predictions.jsonl")
        done_ids = load_done_ids(pred_path)
        remaining = [s for s in samples if s["question_id"] not in done_ids]
        total_remaining += len(remaining)
        task_data.append({
            "config": tc,
            "samples": samples,
            "remaining": remaining,
            "pred_path": pred_path,
            "prompt_mode": tc.get("prompt_mode", "profile"),
        })

    print(f"\n{'=' * 60}", flush=True)
    print(f"  Mode      : Multi-task (vLLM loaded ONCE)", flush=True)
    print(f"  Tasks     : {len(tasks)}", flush=True)
    for i, td in enumerate(task_data):
        tc = td["config"]
        label = os.path.basename(os.path.dirname(tc["data"]))
        split = os.path.splitext(os.path.basename(tc["data"]))[0]
        print(f"    [{i}] {label}/{split} ({td['prompt_mode']})  "
              f"({len(td['remaining'])}/{len(td['samples'])} remaining)",
              flush=True)
    print(f"  Model     : {args.model}", flush=True)
    print(f"  Backend   : {args.backend}", flush=True)
    print(f"  Total     : {total_remaining} samples remaining", flush=True)
    print(f"{'=' * 60}", flush=True)

    if total_remaining == 0:
        print("\nAll tasks already completed.", flush=True)
        return

    if args.backend == "vllm":
        _multi_vllm(args, task_data)
    else:
        _multi_hf(args, task_data)


def _multi_vllm(args, task_data: list[dict]):
    """Run multiple tasks with a single vLLM instance."""
    import torch
    from vllm import LLM, SamplingParams

    n_gpus = torch.cuda.device_count()
    tp = min(n_gpus, args.tensor_parallel) if args.tensor_parallel else n_gpus
    print(f"\n  Loading vLLM: model={args.model}, tp={tp}", flush=True)
    t_load = time.perf_counter()

    llm = LLM(
        model=args.model,
        tensor_parallel_size=tp,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    print(f"  vLLM loaded in {time.perf_counter() - t_load:.1f}s "
          f"(max_model_len={args.max_model_len})", flush=True)

    for task_idx, td in enumerate(task_data):
        remaining = td["remaining"]
        tc = td["config"]
        label = os.path.basename(os.path.dirname(tc["data"]))
        split = os.path.splitext(os.path.basename(tc["data"]))[0]

        if not remaining:
            print(f"\n  [{task_idx}] {label}/{split}: already complete", flush=True)
            continue

        print(f"\n  [{task_idx}] {label}/{split}: {len(remaining)} samples ...",
              flush=True)

        prompts = []
        for s in remaining:
            raw = build_prompt(s, td["prompt_mode"])
            messages = [{"role": "user", "content": raw}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            prompts.append(text)

        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        elapsed = time.perf_counter() - t0

        with open(td["pred_path"], "a", encoding="utf-8") as f:
            for s, out in zip(remaining, outputs):
                record = {
                    "question_id": s["question_id"],
                    "role": s["role"],
                    "prediction": out.outputs[0].text.strip(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        speed = len(remaining) / elapsed if elapsed > 0 else 0
        print(f"    Done: {len(remaining)} samples in {elapsed:.1f}s "
              f"({speed:.1f} samples/s)", flush=True)

        _save_meta_for_task(args, td, elapsed)

    del llm
    import torch
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print(f"\n  Multi-task vLLM pipeline complete.", flush=True)


def _multi_hf(args, task_data: list[dict]):
    """Run multiple tasks with a single HF model instance."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"\n  Loading HF model: {args.model}", flush=True)
    t_load = time.perf_counter()
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
    print(f"  Model loaded in {time.perf_counter() - t_load:.1f}s", flush=True)

    for task_idx, td in enumerate(task_data):
        remaining = td["remaining"]
        tc = td["config"]
        label = os.path.basename(os.path.dirname(tc["data"]))
        split = os.path.splitext(os.path.basename(tc["data"]))[0]

        if not remaining:
            print(f"\n  [{task_idx}] {label}/{split}: already complete", flush=True)
            continue

        print(f"\n  [{task_idx}] {label}/{split}: {len(remaining)} samples ...",
              flush=True)

        f = open(td["pred_path"], "a", encoding="utf-8")
        t0 = time.perf_counter()
        n_done = 0

        pbar = tqdm(total=len(remaining), desc=f"  {label}/{split}",
                     unit="sample", file=sys.stderr, dynamic_ncols=True)

        for i in range(0, len(remaining), args.batch_size):
            batch = remaining[i: i + args.batch_size]
            prompts = [build_prompt(s, td["prompt_mode"]) for s in batch]

            messages_batch = [[{"role": "user", "content": p}] for p in prompts]
            texts = [
                tokenizer.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True,
                )
                for m in messages_batch
            ]
            inputs = tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=4096,
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
            elapsed_so_far = time.perf_counter() - t0
            pbar.set_postfix_str(f"{n_done / elapsed_so_far:.1f} samples/s")

        pbar.close()
        f.close()
        elapsed = time.perf_counter() - t0
        speed = n_done / elapsed if elapsed > 0 else 0
        print(f"    Done: {n_done} samples in {elapsed:.1f}s "
              f"({speed:.1f} samples/s)", flush=True)

        _save_meta_for_task(args, td, elapsed)

    print(f"\n  Multi-task HF pipeline complete.", flush=True)


def _save_meta_for_task(args, td: dict, infer_elapsed: float):
    """Save meta.json for one task in multi-task mode."""
    tc = td["config"]
    remaining = td["remaining"]
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass

    meta = {
        "model": args.model,
        "backend": args.backend,
        "prompt_mode": td["prompt_mode"],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "tensor_parallel": args.tensor_parallel,
        "total_samples": len(td["samples"]),
        "predicted_this_run": len(remaining),
        "data_path": tc["data"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_hash": git_hash,
        "mode": "multi_task",
    }
    if remaining and infer_elapsed > 0:
        meta["latency"] = {
            "total_seconds": round(infer_elapsed, 2),
            "num_predicted": len(remaining),
            "mean_ms_per_sample": round(infer_elapsed / len(remaining) * 1000, 1),
            "samples_per_second": round(len(remaining) / infer_elapsed, 2),
        }

    meta_path = os.path.join(tc["output_dir"], "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    if "--multi" in sys.argv:
        multi_main()
    else:
        main()
