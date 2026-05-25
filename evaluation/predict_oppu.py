"""Inference for OPPU (One-PEFT-Per-User; per-character LoRAs).

For each test sample, look up the LoRA adapter trained for that
character by ``train_oppu.py`` and use it to generate the prediction.

Backend strategy
----------------
We use vLLM in multi-LoRA mode (``enable_lora=True`` with
``max_loras=1`` and re-issued ``LoRARequest``s per role group).  Per-role
adapter swaps inside vLLM are essentially free once the engine is
initialised, so we batch all samples of one role into a single
``llm.generate`` call.

Roles without an adapter (e.g. unseen at training time, or whose
training was skipped) are handled by ``--missing_strategy``:

  * ``base`` — fall back to plain base-model inference (no LoRA), and
    record the role in ``meta.json`` under ``missing_roles``.
  * ``skip`` — drop those samples entirely.

Usage
-----
::

    python evaluation/predict_oppu.py \\
        --data         phase_tree_data/processed/RAIDEN/m1_context_only/random_test.json \\
        --oppu_root    phase_tree_models/oppu/RAIDEN \\
        --output_dir   results/RAIDEN/comparison/main/oppu/random_test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


BASELINE_PROMPT = """\
Below is a multi-turn dialogue. Predict the single line that {character} would most likely say next.
Keep the reply short and natural, matching the tone and length of the other lines. Output only that one line, no explanation.

Dialogue context:
{context}

{character}:"""


def slugify(role: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", role.strip())
    return s.strip("_") or "role"


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


def discover_adapters(oppu_root: str) -> dict[str, str]:
    """Map role-string → adapter directory.

    Looks at ``role_index.json`` first (authoritative), then falls back
    to scanning subdirectories of oppu_root that contain an
    ``adapter_model.{safetensors,bin}`` file.
    """
    by_role: dict[str, str] = {}
    idx = os.path.join(oppu_root, "role_index.json")
    if os.path.exists(idx):
        try:
            with open(idx, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in data.get("roles", []):
                if r.get("status") in ("trained", "skipped_existing"):
                    p = r.get("path") or os.path.join(oppu_root, r["slug"])
                    if os.path.exists(os.path.join(p, "adapter_model.safetensors")) \
                       or os.path.exists(os.path.join(p, "adapter_model.bin")):
                        by_role[r["role"]] = p
            if by_role:
                return by_role
        except Exception as e:
            print(f"  ⚠ failed to read role_index.json: {e}; falling back to dir scan",
                  flush=True)

    for entry in os.scandir(oppu_root):
        if entry.is_dir():
            p = entry.path
            if os.path.exists(os.path.join(p, "adapter_model.safetensors")) \
               or os.path.exists(os.path.join(p, "adapter_model.bin")):
                by_role[entry.name] = p
    return by_role


def _read_lora_rank(adapter_dir: str, default: int) -> int:
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


def _vllm_generate(args, by_role_groups: dict[str, list[dict]],
                   role_to_adapter: dict[str, str],
                   pred_path: str, base_model: str
                   ) -> tuple[float, set[str]]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    detected = max(
        (_read_lora_rank(p, args.max_lora_rank)
         for p in role_to_adapter.values()),
        default=args.max_lora_rank,
    )
    effective_rank = max(detected, args.max_lora_rank)
    if detected > args.max_lora_rank:
        print(f"  ⚠ per-role adapters report r up to {detected}, raising "
              f"max_lora_rank from {args.max_lora_rank} to {detected}",
              flush=True)
    print(f"  Loading vLLM (base={base_model}, multi-LoRA enabled, "
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

    f_out = open(pred_path, "a", encoding="utf-8")
    t0 = time.perf_counter()
    missing_roles: set[str] = set()
    n_total = sum(len(v) for v in by_role_groups.values())
    n_done = 0
    next_lora_id = 1

    for role, group in by_role_groups.items():
        prompts = []
        qids = []
        for s in group:
            msgs = [{"role": "user", "content": build_prompt(s)}]
            chat = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(chat)
            qids.append(s["question_id"])

        adapter_path = role_to_adapter.get(role)
        if adapter_path is None:
            missing_roles.add(role)
            if args.missing_strategy == "skip":
                print(f"  ⚠ no adapter for {role!r}; skipping {len(group)} samples",
                      flush=True)
                continue
            print(f"  ⚠ no adapter for {role!r}; falling back to base model "
                  f"({len(group)} samples)", flush=True)
            outputs = llm.generate(prompts, sampling)
        else:
            slug = slugify(role)
            lora_request = LoRARequest(
                lora_name=slug, lora_int_id=next_lora_id, lora_path=adapter_path,
            )
            next_lora_id += 1
            print(f"  → role={role!r} ({len(group)} samples) using "
                  f"{os.path.basename(adapter_path)}", flush=True)
            outputs = llm.generate(prompts, sampling, lora_request=lora_request)

        for qid, out in zip(qids, outputs):
            text = out.outputs[0].text.strip()
            f_out.write(json.dumps({
                "question_id": qid,
                "role": role,
                "prediction": text,
            }, ensure_ascii=False) + "\n")
        f_out.flush()
        n_done += len(prompts)
        elapsed = time.perf_counter() - t0
        print(f"     done {n_done}/{n_total}  ({n_done/elapsed:.2f} samples/s)",
              flush=True)

    f_out.close()
    elapsed = time.perf_counter() - t0
    print(f"\n✓ Predictions done: {n_done} samples in {elapsed:.1f}s",
          flush=True)
    print(f"  Output: {pred_path}", flush=True)

    del llm
    import torch
    torch.cuda.empty_cache()
    import gc; gc.collect()
    return elapsed, missing_roles


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
               predicted_samples: int, missing_roles: set[str] | None,
               role_to_adapter: dict[str, str], latency: dict | None,
               token_stats: dict | None = None) -> None:
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass

    meta = {
        "method": "oppu",
        "model": base_model,
        "oppu_root": args.oppu_root,
        "missing_strategy": args.missing_strategy,
        "num_adapters_available": len(role_to_adapter),
        "backend": "vllm",
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
    if missing_roles:
        meta["missing_roles"] = sorted(missing_roles)
    if latency is not None:
        meta["latency"] = latency
    if token_stats:
        meta["token_stats"] = token_stats

    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OPPU inference (per-character LoRAs)")
    parser.add_argument("--data", required=True)
    parser.add_argument("--oppu_root", required=True,
                        help="Parent directory containing per-role adapters "
                             "(output of train_oppu.py)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default=None,
                        help="Base LLM (default: read from any per-role "
                             "adapter_config.json or fall back to "
                             "PHASE-Tree/models/Qwen2.5-7B-Instruct)")
    parser.add_argument("--missing_strategy", choices=["base", "skip"],
                        default="base",
                        help="What to do for roles without an adapter")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--max_lora_rank", type=int, default=16)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    role_to_adapter = discover_adapters(args.oppu_root)
    if not role_to_adapter:
        raise RuntimeError(
            f"No trained adapters found under {args.oppu_root}. "
            "Did train_oppu.py run successfully?")

    if args.model is None:
        sample_adapter = next(iter(role_to_adapter.values()))
        cfg_path = os.path.join(sample_adapter, "adapter_config.json")
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

    test_roles = set(s["role"] for s in samples)
    missing_at_planning = test_roles - set(role_to_adapter.keys())

    print(f"\n{'─' * 50}", flush=True)
    print(f"  Method            : OPPU (per-character LoRAs)", flush=True)
    print(f"  Base model        : {args.model}", flush=True)
    print(f"  OPPU root         : {args.oppu_root}", flush=True)
    print(f"  Adapters discovered: {len(role_to_adapter)}", flush=True)
    print(f"  Test roles        : {len(test_roles)} "
          f"({len(test_roles - missing_at_planning)} matched, "
          f"{len(missing_at_planning)} missing)", flush=True)
    if missing_at_planning:
        print(f"  Missing roles     : "
              f"{sorted(missing_at_planning)[:5]}{'…' if len(missing_at_planning)>5 else ''}",
              flush=True)
    print(f"  Data              : {args.data}", flush=True)
    print(f"  Samples           : {len(samples)}", flush=True)
    print(f"  Backend           : vllm (multi-LoRA, grouped by role)",
          flush=True)
    print(f"  Output            : {args.output_dir}/", flush=True)
    print(f"  Progress          : {len(done_ids)}/{len(samples)} done, "
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
        _save_meta(args, args.model, len(samples), 0,
                   missing_at_planning, role_to_adapter, latency=None,
                   token_stats=token_stats)
        return

    by_role: dict[str, list[dict]] = defaultdict(list)
    for s in remaining:
        if args.missing_strategy == "skip" and s["role"] not in role_to_adapter:
            continue
        by_role[s["role"]].append(s)
    if not by_role:
        print(f"\n⚠ Nothing to predict (missing_strategy='skip' filtered all).",
              flush=True)
        _save_meta(args, args.model, len(samples), 0,
                   test_roles - set(role_to_adapter.keys()),
                   role_to_adapter, latency=None)
        return

    elapsed, missing = _vllm_generate(
        args, by_role, role_to_adapter, pred_path, args.model)

    pred_token_stats = _compute_prediction_token_stats(pred_path, args.model)
    if pred_token_stats:
        token_stats["prediction_tokens"] = pred_token_stats

    n_predicted = sum(len(g) for g in by_role.values())
    latency = {
        "total_seconds": round(elapsed, 2),
        "num_predicted": n_predicted,
        "mean_ms_per_sample": round(elapsed / n_predicted * 1000, 1)
            if n_predicted else 0.0,
        "samples_per_second": round(n_predicted / elapsed, 2)
            if elapsed > 0 else 0,
    }
    _save_meta(args, args.model, len(samples), n_predicted,
               missing | missing_at_planning, role_to_adapter, latency,
               token_stats=token_stats)


if __name__ == "__main__":
    main()
