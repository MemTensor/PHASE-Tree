"""Inference for MT-LoRA (single shared LoRA adapter).

Loads the base LLM together with the adapter trained by
``train_mt_lora.py`` and generates predictions on a test split using the
**baseline** (context-only, no profile) prompt — exactly mirroring the
training distribution.

Backend
-------
We use vLLM with its native LoRA support (``enable_lora=True`` +
``LoRARequest``).  This keeps inference fast (≈ same throughput as
``predict_prompt.py``), and there is no per-request adapter switching
since MT-LoRA is one shared adapter.

Usage
-----
::

    python evaluation/predict_mt_lora.py \\
        --data         phase_tree_data/processed/RAIDEN/m1_context_only/random_test.json \\
        --lora_path    phase_tree_models/mt_lora/RAIDEN \\
        --output_dir   results/RAIDEN/comparison/main/mt_lora/random_test
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


BASELINE_PROMPT = """\
Below is a multi-turn dialogue. Predict the single line that {character} would most likely say next.
Keep the reply short and natural, matching the tone and length of the other lines. Output only that one line, no explanation.

Dialogue context:
{context}

{character}:"""


def build_prompt(sample: dict) -> str:
    return BASELINE_PROMPT.format(
        character=sample["role"],
        context=sample["input"],
    )


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


def _read_lora_rank(adapter_dir: str, default: int) -> int:
    """Read ``r`` from ``<adapter_dir>/adapter_config.json`` if present.

    Falls back to ``default`` on any failure.  This keeps vLLM's
    ``max_lora_rank`` aligned with the actual adapter, avoiding silent
    rank mismatches when users override training rank.
    """
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return default
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        r = cfg.get("r") or cfg.get("lora_r") or cfg.get("rank")
        return int(r) if r is not None else default
    except Exception:
        return default


def _vllm_generate(args, remaining: list[dict], pred_path: str,
                   base_model: str) -> float:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    detected_rank = _read_lora_rank(args.lora_path, args.max_lora_rank)
    effective_rank = max(detected_rank, args.max_lora_rank)
    if detected_rank > args.max_lora_rank:
        print(f"  ⚠ adapter_config.json reports r={detected_rank}, raising "
              f"max_lora_rank from {args.max_lora_rank} to {detected_rank}",
              flush=True)
    print(f"  Loading vLLM (base={base_model}, lora={args.lora_path}, "
          f"max_lora_rank={effective_rank})", flush=True)
    llm = LLM(
        model=base_model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=effective_rank,
    )
    tokenizer = llm.get_tokenizer()

    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    lora_request = LoRARequest(
        lora_name="mt_lora", lora_int_id=1, lora_path=args.lora_path,
    )

    prompts: list[str] = []
    qids: list[str] = []
    roles: list[str] = []
    for s in remaining:
        msgs = [{"role": "user", "content": build_prompt(s)}]
        chat = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(chat)
        qids.append(s["question_id"])
        roles.append(s["role"])

    print(f"  Generating {len(prompts)} predictions …", flush=True)
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling, lora_request=lora_request)
    elapsed = time.perf_counter() - t0

    with open(pred_path, "a", encoding="utf-8") as f:
        for qid, role, out in zip(qids, roles, outputs):
            text = out.outputs[0].text.strip()
            f.write(json.dumps({
                "question_id": qid,
                "role": role,
                "prediction": text,
            }, ensure_ascii=False) + "\n")
    print(f"\n✓ Predictions done: {len(prompts)} samples in {elapsed:.1f}s "
          f"({len(prompts)/elapsed:.2f} samples/s)", flush=True)
    print(f"  Output: {pred_path}", flush=True)

    import torch
    del llm
    torch.cuda.empty_cache()
    import gc; gc.collect()
    return elapsed


def _hf_generate(args, remaining: list[dict], pred_path: str,
                 base_model: str) -> float:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading HF model (base + LoRA) …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map=args.device,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    f = open(pred_path, "a", encoding="utf-8")
    t0 = time.perf_counter()
    pbar = tqdm(total=len(remaining), desc="mt_lora-predict",
                unit="sample", file=sys.stderr, dynamic_ncols=True)

    gen_kwargs = dict(
        max_new_tokens=args.max_tokens,
        do_sample=args.temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
    )
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature

    for s in remaining:
        msgs = [{"role": "user", "content": build_prompt(s)}]
        chat = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(chat, return_tensors="pt", truncation=True,
                        max_length=args.max_model_len).to(model.device)
        with torch.no_grad():
            output = model.generate(**ids, **gen_kwargs)
        new_ids = output[0, ids["input_ids"].shape[1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        f.write(json.dumps({
            "question_id": s["question_id"],
            "role": s["role"],
            "prediction": text,
        }, ensure_ascii=False) + "\n")
        f.flush()
        pbar.update(1)

    pbar.close()
    f.close()
    elapsed = time.perf_counter() - t0
    print(f"\n✓ Predictions done in {elapsed:.1f}s", flush=True)

    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    return elapsed


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


def _compute_token_stats(samples: list[dict], model_path: str) -> dict:
    tok = _get_tokenizer(model_path)

    profile_lens, context_lens, output_lens, prompt_lens = [], [], [], []

    for s in samples:
        profile_lens.append(len(tok.encode(s.get("profile_text", ""))))
        context_lens.append(len(tok.encode(s.get("input", ""))))
        output_lens.append(len(tok.encode(s.get("output", ""))))

        raw_prompt = build_prompt(s)
        messages = [{"role": "user", "content": raw_prompt}]
        full_prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        prompt_lens.append(len(tok.encode(full_prompt)))

    return {
        "profile_tokens": _arr_stats(profile_lens),
        "context_tokens": _arr_stats(context_lens),
        "output_tokens": _arr_stats(output_lens),
        "prompt_tokens": _arr_stats(prompt_lens),
        "tokenizer": model_path,
        "num_samples": len(samples),
    }


def _save_meta(args, base_model: str, total_samples: int,
               predicted_samples: int, latency: dict | None,
               token_stats: dict | None = None) -> None:
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass
    train_meta_path = os.path.join(args.lora_path, "train_meta.json")
    train_meta = None
    if os.path.exists(train_meta_path):
        try:
            with open(train_meta_path, "r", encoding="utf-8") as f:
                train_meta = json.load(f)
        except Exception:
            train_meta = None

    meta = {
        "method": "mt_lora",
        "model": base_model,
        "lora_path": args.lora_path,
        "backend": args.backend,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "max_lora_rank": args.max_lora_rank,
        "seed": args.seed,
        "total_samples": total_samples,
        "predicted_this_run": predicted_samples,
        "data_path": args.data,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_hash": git_hash,
    }
    if train_meta is not None:
        meta["train_meta"] = train_meta
    if latency is not None:
        meta["latency"] = latency
    if token_stats:
        meta["token_stats"] = token_stats

    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MT-LoRA inference (single shared adapter)")
    parser.add_argument("--data", required=True)
    parser.add_argument("--lora_path", required=True,
                        help="Directory produced by train_mt_lora.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default=None,
                        help="Base LLM path (default: read from "
                             "lora_path/adapter_config.json or fall back to "
                             "PHASE-Tree/models/Qwen2.5-7B-Instruct)")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--device", default="cuda",
                        help="(hf backend only) torch device")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--max_lora_rank", type=int, default=16)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    if args.model is None:
        cfg_path = os.path.join(args.lora_path, "adapter_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    args.model = json.load(f).get("base_model_name_or_path") \
                        or os.path.join(PROJECT_ROOT, "models", "Qwen2.5-7B-Instruct")
            except Exception:
                args.model = os.path.join(PROJECT_ROOT, "models",
                                          "Qwen2.5-7B-Instruct")
        else:
            args.model = os.path.join(PROJECT_ROOT, "models",
                                      "Qwen2.5-7B-Instruct")

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
    print(f"  Method      : MT-LoRA", flush=True)
    print(f"  Base model  : {args.model}", flush=True)
    print(f"  LoRA        : {args.lora_path}", flush=True)
    print(f"  Data        : {args.data}", flush=True)
    print(f"  Samples     : {len(samples)} ({n_roles} characters)", flush=True)
    print(f"  Backend     : {args.backend}", flush=True)
    print(f"  Output      : {args.output_dir}/", flush=True)
    print(f"  Progress    : {len(done_ids)}/{len(samples)} done, "
          f"{len(remaining)} remaining", flush=True)
    print(f"{'─' * 50}", flush=True)

    token_stats = _load_cached_token_stats(args.output_dir, len(samples))
    if token_stats:
        print(f"\n  Token stats (cached): profile={token_stats.get('profile_tokens',{}).get('mean','-')}, "
              f"context={token_stats['context_tokens']['mean']:.1f}, "
              f"output_gt={token_stats['output_tokens']['mean']:.1f}, "
              f"prompt={token_stats['prompt_tokens']['mean']:.1f} (mean tokens)",
              flush=True)
    else:
        print(f"\n  Computing token statistics ...", flush=True)
        token_stats = _compute_token_stats(samples, args.model)
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
        _save_meta(args, args.model, len(samples), 0, latency=None,
                   token_stats=token_stats)
        return

    if args.backend == "vllm":
        elapsed = _vllm_generate(args, remaining, pred_path, args.model)
    else:
        elapsed = _hf_generate(args, remaining, pred_path, args.model)

    pred_token_stats = _compute_prediction_token_stats(pred_path, args.model)
    if pred_token_stats:
        token_stats["prediction_tokens"] = pred_token_stats

    latency = {
        "total_seconds": round(elapsed, 2),
        "num_predicted": len(remaining),
        "mean_ms_per_sample": round(elapsed / len(remaining) * 1000, 1),
        "samples_per_second": round(len(remaining) / elapsed, 2)
            if elapsed > 0 else 0,
    }
    _save_meta(args, args.model, len(samples), len(remaining), latency,
               token_stats=token_stats)


if __name__ == "__main__":
    main()
