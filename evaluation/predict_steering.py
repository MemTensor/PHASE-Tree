"""Activation-Steering character adaptation at decoding time.

Idea (training-free, complements CFG):

  • Pre-computed via :mod:`steering_calibrate`, each character has a
    persona vector v_role ∈ ℝ^d distilled from (with-profile − without-
    profile) hidden-state deltas at one layer.
  • At inference we *do not* show the model the profile.  Instead, we
    register a forward-hook on the chosen layer that adds
    ``α · v_role`` to the residual stream at the prompt's last token
    position (and at every newly generated position).
  • The model therefore "feels" the persona without paying its prompt-
    length cost — useful for very long-context regimes.

This is the classical "Activation Addition" / Persona-Vector recipe.

Why HF backend
--------------
Forward hooks need direct access to the PyTorch module graph; vLLM
fuses kernels and hides this.  HF is the only sane choice.

Usage
-----
::

    python evaluation/predict_steering.py \\
        --data        LongEvoRoleBench/processed/RAIDEN/m1_context_only/random_test.json \\
        --vectors     results/RAIDEN/comparison/main/steering/persona_vectors.pt \\
        --output_dir  results/RAIDEN/comparison/main/steering/random_test \\
        --alpha 4.0

The data file's ``profile_text`` is *not* injected into the prompt — we
deliberately use the baseline (context-only) variant so that the only
character adaptation signal comes from the steering vector.

Hyperparameters
---------------
``--alpha`` (default 4.0): steering strength.  The persona-vector
magnitudes are typically O(1)–O(10), so values in 1–10 are sensible.
Too large → hallucinated character traits and degenerate text; too
small → indistinguishable from baseline.
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


# ---------------------------------------------------------------------------
# Steering hook: add α·v to every token's residual at the chosen layer.
# We re-arm the hook per-sample (different roles → different vectors).
# ---------------------------------------------------------------------------

class SteeringHook:
    """Forward hook that adds a fixed vector to the residual stream output
    of a specific decoder layer.  ``set(vector, alpha)`` (re)configures it
    between samples; ``__call__`` is the hook invocation.
    """

    def __init__(self) -> None:
        self.vector = None
        self.alpha = 0.0
        self.handle = None

    def set(self, vector, alpha: float) -> None:
        self.vector = vector
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        if self.vector is None or self.alpha == 0.0:
            return output
        if isinstance(output, tuple):
            hidden = output[0]
            v = self.vector.to(device=hidden.device, dtype=hidden.dtype)
            hidden = hidden + self.alpha * v
            return (hidden,) + output[1:]
        else:
            v = self.vector.to(device=output.device, dtype=output.dtype)
            return output + self.alpha * v


def get_decoder_layer(model, layer_idx: int):
    cfg = type(model).__name__.lower()
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise RuntimeError(
        f"Cannot locate decoder layers for model class {cfg}; "
        "extend get_decoder_layer().")


# ---------------------------------------------------------------------------
# Inference loop — batched generation with role-grouped persona vectors
# ---------------------------------------------------------------------------
#
# Each forward pass through `target` adds the SAME α·v to every position of
# every row in the active batch.  Because different roles need different v's,
# we group `remaining` by role first, then process each group in mini-batches
# of size `args.batch_size`.  Within a group the hook vector is constant, so
# the batched math (broadcasting [d] across [B, T, d]) is mathematically
# identical to running each sample on its own — no gradient flows here, so
# there is no batch-norm-style cross-sample contamination either.
#
# Trade-offs picked:
#   - resume support is preserved (we only iterate `remaining`);
#   - output order changes (grouped by role) — judge.py keys by question_id;
#   - skipped roles (no vector) still go through the same code path with
#     hook disabled, so their throughput matches the rest of the batch.
# ---------------------------------------------------------------------------

def run_steering(args, remaining: list[dict], pred_path: str,
                 vectors: dict, calib_meta: dict) -> tuple[float, set[str]]:
    import torch
    from collections import defaultdict
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

    attn_kwargs = {}
    try:
        from flash_attn import flash_attn_func  # noqa: F401
        attn_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
        **attn_kwargs,
    )
    model.eval()

    layer = calib_meta["layer"]
    target = get_decoder_layer(model, layer)
    hook = SteeringHook()
    hook.handle = target.register_forward_hook(hook)
    fa_tag = " +FA2" if attn_kwargs else ""
    print(f"  Model loaded{fa_tag}; hook at layer {layer}; alpha={args.alpha}",
          flush=True)

    role_groups: dict[str, list[dict]] = defaultdict(list)
    for s in remaining:
        role_groups[s["role"]].append(s)
    print(f"  Grouped {len(remaining)} samples into {len(role_groups)} role(s); "
          f"batch_size={args.batch_size}", flush=True)

    f = open(pred_path, "a", encoding="utf-8")
    t0 = time.perf_counter()
    n_done = 0
    skipped_roles: set[str] = set()
    pbar = tqdm(total=len(remaining), desc="steering-predict", unit="sample",
                file=sys.stderr, dynamic_ncols=True)

    gen_kwargs: dict = dict(
        max_new_tokens=args.max_tokens,
        do_sample=args.temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
    )
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature

    for role, group in role_groups.items():
        v = vectors.get(role)
        if v is None:
            skipped_roles.add(role)
            hook.set(None, 0.0)
        else:
            hook.set(v, args.alpha)

        for i in range(0, len(group), args.batch_size):
            mini = group[i: i + args.batch_size]
            prompts = [build_prompt(s) for s in mini]
            messages_batch = [[{"role": "user", "content": p}] for p in prompts]
            texts = [
                tokenizer.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True)
                for m in messages_batch
            ]
            inputs = tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=args.max_model_len,
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs,
                )

            prompt_len = inputs["input_ids"].shape[1]
            for s, ids in zip(mini, output_ids):
                new_ids = ids[prompt_len:]
                text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                f.write(json.dumps({
                    "question_id": s["question_id"],
                    "role": role,
                    "prediction": text,
                }, ensure_ascii=False) + "\n")
            f.flush()

            n_done += len(mini)
            pbar.update(len(mini))
            elapsed = time.perf_counter() - t0
            speed = n_done / elapsed if elapsed > 0 else 0.0
            pbar.set_postfix_str(f"{speed:.2f} samples/s")

    pbar.close()
    if hook.handle is not None:
        hook.handle.remove()
    f.close()
    elapsed = time.perf_counter() - t0
    print(f"\n✓ Predictions done: {n_done} samples in {elapsed:.1f}s "
          f"({n_done/elapsed:.2f} samples/s)", flush=True)
    if skipped_roles:
        print(f"  ⚠ {len(skipped_roles)} role(s) had no persona vector "
              f"(decoded as plain baseline): {sorted(skipped_roles)[:5]}…",
              flush=True)
    print(f"  Output: {pred_path}", flush=True)

    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    return elapsed, skipped_roles


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


def _save_meta(args, total_samples: int, predicted_samples: int,
               calib_meta: dict, latency_stats: dict | None,
               skipped_roles: set[str] | None,
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
        "method": "steering",
        "model": args.model,
        "backend": "hf",
        "alpha": args.alpha,
        "layer": calib_meta["layer"],
        "calibration_num_per_role": calib_meta.get("num_per_role"),
        "calibration_d_model": calib_meta.get("d_model"),
        "calibration_n_layers": calib_meta.get("n_layers"),
        "vectors_path": args.vectors,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "total_samples": total_samples,
        "predicted_this_run": predicted_samples,
        "data_path": args.data,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_hash": git_hash,
    }
    if latency_stats:
        meta["latency"] = latency_stats
    if skipped_roles:
        meta["skipped_roles"] = sorted(skipped_roles)
    if token_stats:
        meta["token_stats"] = token_stats

    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  Meta saved: {meta_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Activation-steering dialogue prediction (persona vectors)")
    parser.add_argument("--data", required=True,
                        help="Test JSON; profile_text NOT used at inference")
    parser.add_argument("--vectors", required=True,
                        help="Path to persona_vectors.pt produced by "
                             "steering_calibrate.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default=None,
                        help="LLM path (default: PHASE-Tree/models/Qwen2.5-7B-Instruct)")
    parser.add_argument("--alpha", type=float, default=4.0,
                        help="Steering strength (recommend 1–10)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Mini-batch size for HF generate. Hook adds the "
                             "same role-vector to every row, so we group by "
                             "role and batch within each group.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    if args.model is None:
        args.model = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-7B-Instruct")

    import torch
    print(f"  Loading persona vectors: {args.vectors}", flush=True)
    bundle = torch.load(args.vectors, map_location="cpu")
    vectors: dict = bundle["vectors"]
    calib_meta = {k: v for k, v in bundle.items() if k != "vectors"}
    if calib_meta.get("model") and calib_meta["model"] != args.model:
        print(f"  ⚠ vectors were calibrated on {calib_meta['model']!r} but "
              f"inference uses {args.model!r}; expect degraded behaviour.",
              flush=True)

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
    n_with_vec = len(set(s["role"] for s in samples) & set(vectors.keys()))
    print(f"\n{'─' * 50}", flush=True)
    print(f"  Method      : Steering (α={args.alpha}, layer={calib_meta['layer']})",
          flush=True)
    print(f"  Data        : {args.data}", flush=True)
    print(f"  Samples     : {len(samples)} ({n_roles} characters; "
          f"{n_with_vec} have persona vectors)", flush=True)
    print(f"  Backend     : hf (role-grouped batched, batch_size={args.batch_size})",
          flush=True)
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
        _save_meta(args, len(samples), 0, calib_meta,
                   latency_stats=None, skipped_roles=None,
                   token_stats=token_stats)
        return

    elapsed, skipped = run_steering(args, remaining, pred_path,
                                    vectors, calib_meta)

    pred_token_stats = _compute_prediction_token_stats(pred_path, args.model)
    if pred_token_stats:
        token_stats["prediction_tokens"] = pred_token_stats

    latency_stats = {
        "total_seconds": round(elapsed, 2),
        "num_predicted": len(remaining),
        "mean_ms_per_sample": round(elapsed / len(remaining) * 1000, 1),
        "samples_per_second": round(len(remaining) / elapsed, 2)
                              if elapsed > 0 else 0,
    }
    _save_meta(args, len(samples), len(remaining), calib_meta,
               latency_stats, skipped, token_stats=token_stats)


if __name__ == "__main__":
    main()
