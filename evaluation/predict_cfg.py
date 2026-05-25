"""Classifier-Free-Guidance (CFG) character adaptation at decoding time.

CFG is a training-free technique borrowed from text-to-image diffusion
models and recently popularized for LLMs (Sanchez et al., 2023; Liu et
al., NeurIPS 2024).  At every decoding step we run TWO parallel forward
passes:

  * **conditional**   prompt = profile + dialogue context
  * **unconditional** prompt = dialogue context only (baseline)

The next-token logits are combined as:

    logits_final = uncond + γ * (cond − uncond)         (γ ≥ 1)

Setting γ = 1 recovers vanilla profile-injection; γ > 1 amplifies the
profile's effect; γ → ∞ is "argmax of the difference".  Typical values
are 1.5–3.0.

Implementation
--------------
We ship a custom :class:`BatchedCFGLogitsProcessor` that mirrors the math
of HuggingFace's ``UnbatchedClassifierFreeGuidanceLogitsProcessor`` but
batches the unconditional pass across ``B`` samples in a single call to
``model.forward``.  The cond and uncond batches each maintain their own
KV-cache (cond inside ``model.generate``; uncond inside the processor),
so total memory is roughly 2× a normal batched generate.  Throughput
scales near-linearly with ``--batch_size`` until VRAM saturates.

Why HF backend (not vLLM)
-------------------------
vLLM does not currently support paired conditional/unconditional decoding
through its ``logits_processors`` API: each prompt has its own KV-cache
and there is no public hook to feed an unconditional context per request.
HF + a batched CFG processor is the cleanest correct implementation.

Usage::

    python evaluation/predict_cfg.py \\
        --data        phase_tree_data/processed/RAIDEN/m6_phase_tree/random_test.json \\
        --output_dir  results/RAIDEN/comparison/main/cfg/random_test \\
        --guidance_scale 1.5

Pass any m1–m6 file as ``--data``.  Whatever ``profile_text`` is in the
file becomes the *conditional* prompt; the *unconditional* prompt always
re-builds the baseline (context-only) variant for the same sample.
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


# ---------------------------------------------------------------------------
# Prompt templates (mirror predict_prompt.py for apples-to-apples comparison)
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


def build_cond_prompt(sample: dict) -> str:
    return PROFILE_PROMPT.format(
        character=sample["role"],
        context=sample["input"],
        profile=(sample.get("profile_text") or "").strip(),
    )


def build_uncond_prompt(sample: dict) -> str:
    return BASELINE_PROMPT.format(
        character=sample["role"],
        context=sample["input"],
    )


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


# ---------------------------------------------------------------------------
# Batched CFG logits processor — drop-in replacement for HF's "unbatched"
# version that accepts a [B, L_uncond] unconditional batch instead of [1, L].
# ---------------------------------------------------------------------------
#
# At every decoding step ``model.generate`` calls ``__call__(input_ids,
# scores)``.  ``input_ids`` is the cond batch's running sequence
# ``[B, T_cond]`` and ``scores`` are its next-token logits ``[B, V]``.
#
# We maintain a parallel uncond state:
#   * first call:  forward the full uncond batch ``[B, L_uncond]``, cache KV;
#   * later calls: feed ``input_ids[:, -1:]`` (cond's just-chosen token)
#                  through the cached KV — **a single token per row** —
#                  giving us next-token uncond logits in O(B·d) per step.
#
# The combined log-prob is HF's standard CFG form:
#
#     out = γ · log_softmax(cond) + (1 − γ) · log_softmax(uncond)
#
# Memory: cond KV-cache + uncond KV-cache + one extra forward per step.
# For a 7B model in bf16 with B=4 and 8K context, this still fits
# comfortably on an 80GB GPU; halve B if you OOM at >16K.
# ---------------------------------------------------------------------------

def _make_batched_cfg_processor():
    """Return a class-bound LogitsProcessor.  Imported lazily so the file
    can still be parsed without ``transformers`` installed."""
    import torch
    from transformers import LogitsProcessor

    class BatchedCFGLogitsProcessor(LogitsProcessor):
        def __init__(self, model, uncond_input_ids, uncond_attention_mask,
                     guidance_scale: float):
            self.model = model
            self.guidance_scale = float(guidance_scale)
            self.uncond_input_ids = uncond_input_ids
            self.uncond_attention_mask = uncond_attention_mask
            self.past_key_values = None
            self.first_pass = True

        @torch.no_grad()
        def __call__(self, input_ids, scores):
            if self.guidance_scale == 1.0:
                return scores

            if self.first_pass:
                out = self.model(
                    input_ids=self.uncond_input_ids,
                    attention_mask=self.uncond_attention_mask,
                    use_cache=True,
                )
                self.past_key_values = out.past_key_values
                uncond_logits = out.logits[:, -1, :]  # [B, V]
                self.first_pass = False
            else:
                # cond's just-generated token (HF appends it to input_ids
                # before the next logits-processor call).
                new_token = input_ids[:, -1:]  # [B, 1]
                self.uncond_attention_mask = torch.cat(
                    [
                        self.uncond_attention_mask,
                        torch.ones_like(
                            new_token,
                            dtype=self.uncond_attention_mask.dtype,
                            device=self.uncond_attention_mask.device,
                        ),
                    ],
                    dim=-1,
                )
                out = self.model(
                    input_ids=new_token,
                    attention_mask=self.uncond_attention_mask,
                    past_key_values=self.past_key_values,
                    use_cache=True,
                )
                self.past_key_values = out.past_key_values
                uncond_logits = out.logits[:, -1, :]  # [B, V]

            cond_logp = torch.nn.functional.log_softmax(scores, dim=-1)
            uncond_logp = torch.nn.functional.log_softmax(uncond_logits, dim=-1)
            return self.guidance_scale * (cond_logp - uncond_logp) + uncond_logp

    return BatchedCFGLogitsProcessor


# ---------------------------------------------------------------------------
# CFG inference (HF backend, batched)
# ---------------------------------------------------------------------------

def run_cfg(args, remaining: list[dict], pred_path: str) -> float:
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
    fa_tag = " +FA2" if attn_kwargs else ""
    print(f"  Model loaded{fa_tag}; guidance_scale={args.guidance_scale}; "
          f"batch_size={args.batch_size}", flush=True)

    BatchedCFG = _make_batched_cfg_processor()

    f = open(pred_path, "a", encoding="utf-8")
    t0 = time.perf_counter()
    n_done = 0
    pbar = tqdm(total=len(remaining), desc="cfg-predict", unit="sample",
                file=sys.stderr, dynamic_ncols=True)

    gen_kwargs: dict = dict(
        max_new_tokens=args.max_tokens,
        do_sample=args.temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
    )
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature

    for i in range(0, len(remaining), args.batch_size):
        mini = remaining[i: i + args.batch_size]
        cond_texts = [build_cond_prompt(s) for s in mini]
        uncond_texts = [build_uncond_prompt(s) for s in mini]

        cond_chats = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": t}],
                tokenize=False, add_generation_prompt=True)
            for t in cond_texts
        ]
        uncond_chats = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": t}],
                tokenize=False, add_generation_prompt=True)
            for t in uncond_texts
        ]

        cond_inputs = tokenizer(
            cond_chats, return_tensors="pt", padding=True,
            truncation=True, max_length=args.max_model_len,
        ).to(model.device)
        uncond_inputs = tokenizer(
            uncond_chats, return_tensors="pt", padding=True,
            truncation=True, max_length=args.max_model_len,
        ).to(model.device)

        cfg = BatchedCFG(
            model=model,
            uncond_input_ids=uncond_inputs["input_ids"],
            uncond_attention_mask=uncond_inputs["attention_mask"],
            guidance_scale=args.guidance_scale,
        )

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=cond_inputs["input_ids"],
                attention_mask=cond_inputs["attention_mask"],
                logits_processor=[cfg],
                **gen_kwargs,
            )

        prompt_len = cond_inputs["input_ids"].shape[1]
        for s, ids in zip(mini, output_ids):
            new_ids = ids[prompt_len:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            f.write(json.dumps({
                "question_id": s["question_id"],
                "role": s["role"],
                "prediction": text,
            }, ensure_ascii=False) + "\n")
        f.flush()

        n_done += len(mini)
        pbar.update(len(mini))
        elapsed = time.perf_counter() - t0
        speed = n_done / elapsed if elapsed > 0 else 0.0
        pbar.set_postfix_str(f"{speed:.2f} samples/s")

    pbar.close()
    f.close()
    elapsed = time.perf_counter() - t0
    print(f"\n✓ Predictions done: {n_done} samples in {elapsed:.1f}s "
          f"({n_done/elapsed:.2f} samples/s)", flush=True)
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


def _compute_token_stats(samples: list[dict], model_path: str) -> dict:
    tok = _get_tokenizer(model_path)

    context_lens, output_lens, profile_lens = [], [], []
    cond_prompt_lens, uncond_prompt_lens = [], []

    for s in samples:
        context_lens.append(len(tok.encode(s.get("input", ""))))
        output_lens.append(len(tok.encode(s.get("output", ""))))
        profile_lens.append(len(tok.encode(s.get("profile_text", ""))))

        cond = build_cond_prompt(s)
        uncond = build_uncond_prompt(s)
        cond_msg = [{"role": "user", "content": cond}]
        uncond_msg = [{"role": "user", "content": uncond}]
        cond_prompt_lens.append(len(tok.encode(tok.apply_chat_template(
            cond_msg, tokenize=False, add_generation_prompt=True))))
        uncond_prompt_lens.append(len(tok.encode(tok.apply_chat_template(
            uncond_msg, tokenize=False, add_generation_prompt=True))))

    return {
        "context_tokens": _arr_stats(context_lens),
        "output_tokens": _arr_stats(output_lens),
        "profile_tokens": _arr_stats(profile_lens),
        "cond_prompt_tokens": _arr_stats(cond_prompt_lens),
        "uncond_prompt_tokens": _arr_stats(uncond_prompt_lens),
        "tokenizer": model_path,
        "num_samples": len(samples),
    }


# ---------------------------------------------------------------------------
# Meta serialization
# ---------------------------------------------------------------------------

def _save_meta(args, total_samples: int, predicted_samples: int,
               latency_stats: dict | None,
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
        "method": "cfg",
        "model": args.model,
        "backend": "hf",
        "guidance_scale": args.guidance_scale,
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
    if token_stats:
        meta["token_stats"] = token_stats

    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  Meta saved: {meta_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classifier-Free-Guidance dialogue prediction",
    )
    parser.add_argument("--data", required=True,
                        help="Path to a processed JSON containing profile_text "
                             "(typically m6_phase_tree/<split>.json)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default=None,
                        help="LLM path (default: PHASE-Tree/models/Qwen2.5-7B-Instruct)")
    parser.add_argument("--guidance_scale", type=float, default=1.5,
                        help="CFG guidance scale γ (1.0 = vanilla profile injection; "
                             "1.5 mild, 2.0 moderate, 3.0 aggressive)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batched CFG processor size. Each step does one "
                             "extra forward pass over the uncond batch, so "
                             "memory is ~2× a normal batched generate; halve "
                             "this if you OOM.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=None)
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
    print(f"  Method      : CFG (γ={args.guidance_scale})", flush=True)
    print(f"  Data        : {args.data}", flush=True)
    print(f"  Samples     : {len(samples)} ({n_roles} characters)", flush=True)
    print(f"  Backend     : hf (BatchedCFGLogitsProcessor, batch_size={args.batch_size})",
          flush=True)
    print(f"  Output      : {args.output_dir}/", flush=True)
    print(f"  Progress    : {len(done_ids)}/{len(samples)} done, "
          f"{len(remaining)} remaining", flush=True)
    print(f"{'─' * 50}", flush=True)

    token_stats = _load_cached_token_stats(args.output_dir, len(samples))
    if token_stats:
        print(f"\n  Token stats (cached): context={token_stats['context_tokens']['mean']:.1f}, "
              f"profile={token_stats['profile_tokens']['mean']:.1f}, "
              f"cond_prompt={token_stats['cond_prompt_tokens']['mean']:.1f} (mean tokens)",
              flush=True)
    else:
        print(f"\n  Computing token statistics ...", flush=True)
        token_stats = _compute_token_stats(samples, args.model)
        print(f"  Token stats: context={token_stats['context_tokens']['mean']:.1f}, "
              f"profile={token_stats['profile_tokens']['mean']:.1f}, "
              f"cond_prompt={token_stats['cond_prompt_tokens']['mean']:.1f}, "
              f"uncond_prompt={token_stats['uncond_prompt_tokens']['mean']:.1f} (mean tokens)",
              flush=True)

    if not remaining:
        print(f"\n✓ All {len(samples)} predictions already done.", flush=True)
        pred_token_stats = _compute_prediction_token_stats(pred_path, args.model)
        if pred_token_stats:
            token_stats["prediction_tokens"] = pred_token_stats
        _save_meta(args, len(samples), 0, latency_stats=None,
                   token_stats=token_stats)
        return

    elapsed = run_cfg(args, remaining, pred_path)

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
    _save_meta(args, len(samples), len(remaining), latency_stats,
               token_stats=token_stats)


if __name__ == "__main__":
    main()
