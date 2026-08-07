"""Train one LoRA per character (One-PEFT-Per-User; OPPU).

Reference: Tan et al., "Personalized Pieces: Efficient Personalized
Large Language Models through Collaboration of LoRA Experts", EMNLP'24.

Each character gets its OWN LoRA adapter, trained only on that
character's training dialogues with the baseline (context-only) prompt.
This gives a strong upper bound on what per-user fine-tuning can
achieve, at the cost of (#characters) separate training runs.

Implementation
--------------
We thinly wrap :mod:`train_mt_lora` so the training loop / collator /
loss masking are *identical* — the only difference is that we pass
``--filter_role`` to keep only one character per run, and we sweep over
all characters present in the training file.

Output layout
-------------
``--output_dir/<role>/`` for each character, where each subfolder is a
standard PEFT adapter directory plus ``train_meta.json``.

Usage
-----
::

    python evaluation/train_oppu.py \\
        --train_data LongEvoRoleBench/processed/RAIDEN/m1_context_only/train.json \\
        --output_dir phase_tree_models/oppu/RAIDEN \\
        --epochs 5 --lr 2e-4

Skip already-trained roles via ``--skip_existing`` (default ON).  Pass
``--only_roles "Alice,Bob"`` to train just a subset (e.g. for parallel
launches across multiple machines).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import subprocess
import time
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def slugify(role: str) -> str:
    """Filesystem-safe role identifier (preserves human readability)."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", role.strip())
    return s.strip("_") or "role"


def list_roles(train_data: str, min_samples: int = 5) -> list[tuple[str, int]]:
    with open(train_data, "r", encoding="utf-8") as f:
        data = json.load(f)
    cnt = Counter(s["role"] for s in data)
    return [(r, n) for r, n in cnt.most_common() if n >= min_samples]


def train_one_role(args, role: str, role_dir: str) -> int:
    """Spawn a subprocess running train_mt_lora.py with --filter_role.

    Subprocessing is the simplest way to release all GPU memory between
    roles and avoid PEFT-state contamination across consecutive trainings
    in one Python interpreter.
    """
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "evaluation", "train_mt_lora.py"),
        "--train_data", args.train_data,
        "--output_dir", role_dir,
        "--filter_role", role,
        "--model", args.model,
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--target_modules", *args.target_modules,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--batch_size", str(args.batch_size),
        "--grad_accum", str(args.grad_accum),
        "--warmup_ratio", str(args.warmup_ratio),
        "--weight_decay", str(args.weight_decay),
        "--max_seq_len", str(args.max_seq_len),
        "--seed", str(args.seed),
        "--logging_steps", str(args.logging_steps),
    ]
    if not args.gradient_checkpointing:
        cmd.append("--no_gradient_checkpointing")

    env = os.environ.copy()
    env["TRANSFORMERS_VERBOSITY"] = env.get("TRANSFORMERS_VERBOSITY", "warning")
    print(f"  $ {' '.join(cmd[:6])} … --filter_role '{role}'", flush=True)
    return subprocess.call(cmd, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train per-character LoRAs (OPPU baseline).")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output_dir", required=True,
                        help="Parent dir; per-role adapters land in <dir>/<role>/")
    parser.add_argument("--model", default=None)

    # Forwarded to train_mt_lora.py
    parser.add_argument("--lora_r", type=int, default=8,
                        help="OPPU paper uses small r=8 (less data per role)")
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--epochs", type=int, default=5,
                        help="More epochs because per-role data is smaller")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        default=True)
    parser.add_argument("--no_gradient_checkpointing",
                        dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)

    # OPPU-specific orchestration
    parser.add_argument("--min_samples_per_role", type=int, default=10,
                        help="Skip characters with fewer training dialogues")
    parser.add_argument("--only_roles", default=None,
                        help="Comma-separated subset (e.g. 'Alice,Bob')")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing",
                        dest="skip_existing", action="store_false")
    args = parser.parse_args()

    if args.model is None:
        args.model = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-7B-Instruct")

    os.makedirs(args.output_dir, exist_ok=True)

    roles = list_roles(args.train_data, min_samples=args.min_samples_per_role)
    if args.only_roles:
        wanted = {r.strip() for r in args.only_roles.split(",") if r.strip()}
        roles = [(r, n) for r, n in roles if r in wanted]

    if not roles:
        raise RuntimeError("No qualifying roles to train.")

    print(f"\n{'═' * 60}", flush=True)
    print(f"  OPPU training — {len(roles)} characters", flush=True)
    print(f"  Train data : {args.train_data}", flush=True)
    print(f"  Output     : {args.output_dir}/<role>/", flush=True)
    print(f"  Min samples: {args.min_samples_per_role}", flush=True)
    print(f"{'═' * 60}\n", flush=True)

    role_index: list[dict] = []
    started = time.perf_counter()
    n_done = n_skipped = n_failed = 0

    for i, (role, n_samples) in enumerate(roles):
        slug = slugify(role)
        role_dir = os.path.join(args.output_dir, slug)
        adapter_path = os.path.join(role_dir, "adapter_model.safetensors")
        bin_adapter = os.path.join(role_dir, "adapter_model.bin")

        print(f"\n[{i+1}/{len(roles)}] role={role!r} (slug={slug}, "
              f"n_samples={n_samples})", flush=True)

        if args.skip_existing and (os.path.exists(adapter_path)
                                    or os.path.exists(bin_adapter)):
            print(f"  ↳ adapter already exists → skipping", flush=True)
            role_index.append({
                "role": role, "slug": slug, "path": role_dir,
                "n_samples": n_samples, "status": "skipped_existing",
            })
            n_skipped += 1
            continue

        os.makedirs(role_dir, exist_ok=True)
        rc = train_one_role(args, role, role_dir)
        if rc != 0:
            print(f"  ✗ training failed (rc={rc})", flush=True)
            role_index.append({
                "role": role, "slug": slug, "path": role_dir,
                "n_samples": n_samples, "status": f"failed_rc{rc}",
            })
            n_failed += 1
        else:
            role_index.append({
                "role": role, "slug": slug, "path": role_dir,
                "n_samples": n_samples, "status": "trained",
            })
            n_done += 1

    elapsed = time.perf_counter() - started
    index_path = os.path.join(args.output_dir, "role_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "train_data": args.train_data,
            "model": args.model,
            "min_samples_per_role": args.min_samples_per_role,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "epochs": args.epochs,
            "lr": args.lr,
            "roles": role_index,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'═' * 60}", flush=True)
    print(f"  OPPU training complete", flush=True)
    print(f"  Trained        : {n_done}", flush=True)
    print(f"  Skipped        : {n_skipped}", flush=True)
    print(f"  Failed         : {n_failed}", flush=True)
    print(f"  Total wall     : {elapsed/60:.1f} min", flush=True)
    print(f"  Index          : {index_path}", flush=True)
    print(f"{'═' * 60}\n", flush=True)


if __name__ == "__main__":
    main()
