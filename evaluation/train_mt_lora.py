"""Train a single Multi-Task LoRA shared across all characters (MT-LoRA).

Standard shared-adapter baseline.  We fine-tune *one* LoRA adapter on
the union of every character's training dialogues, with the **baseline
prompt** (no profile injection).  The resulting adapter captures
dataset-wide stylistic priors but cannot distinguish individual
characters beyond what is implied by the role name in the prompt itself.

This is intentionally weaker than OPPU (per-character LoRA) and serves
as the "shared backbone, no character adaptation" reference point.

Inputs
------
A processed JSON file (``LongEvoRoleBench/processed/<DATASET>/m1_context_only/train.json``
or any ``m*_*/train.json`` — we ignore everything except ``role``,
``input``, ``output``).

Output
------
``--output_dir`` will contain a standard PEFT adapter directory
(``adapter_config.json``, ``adapter_model.safetensors``, ...) ready to
be loaded by ``predict_mt_lora.py`` or any vLLM ``LoRARequest``.

Usage
-----
::

    # Single GPU
    python evaluation/train_mt_lora.py \\
        --train_data LongEvoRoleBench/processed/RAIDEN/m1_context_only/train.json \\
        --output_dir phase_tree_models/mt_lora/RAIDEN \\
        --epochs 3 --lr 1e-4

    # Multi-GPU
    accelerate launch evaluation/train_mt_lora.py \\
        --train_data LongEvoRoleBench/processed/RAIDEN/m1_context_only/train.json \\
        --output_dir phase_tree_models/mt_lora/RAIDEN

Hyper-parameters
----------------
LoRA defaults (r=16, α=32, dropout=0.05, q/k/v/o + gate/up/down) follow
the OPPU paper.  Override via CLI flags.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


BASELINE_PROMPT = """\
Below is a multi-turn dialogue. Predict the single line that {character} would most likely say next.
Keep the reply short and natural, matching the tone and length of the other lines. Output only that one line, no explanation.

Dialogue context:
{context}

{character}:"""


def load_samples(path: str, filter_role: str | None = None) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if filter_role is not None:
        data = [s for s in data if s["role"] == filter_role]
    return data


def build_dataset(samples, tokenizer, max_seq_len: int):
    """Tokenize each sample as a chat (user prompt + assistant response).

    Labels are masked (-100) on every token belonging to the user prompt,
    so the LM-loss is computed only over the assistant turn — the
    standard "completion-only" SFT recipe.
    """
    from datasets import Dataset

    def encode(s):
        prompt = BASELINE_PROMPT.format(character=s["role"], context=s["input"])
        target = (s.get("output") or "").strip()
        if not target:
            return None
        prompt_msgs = [{"role": "user", "content": prompt}]
        full_msgs = prompt_msgs + [{"role": "assistant", "content": target}]

        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False)

        prompt_ids = tokenizer(
            prompt_text, add_special_tokens=False, truncation=True,
            max_length=max_seq_len,
        )["input_ids"]
        full_ids = tokenizer(
            full_text, add_special_tokens=False, truncation=True,
            max_length=max_seq_len,
        )["input_ids"]

        if len(full_ids) <= len(prompt_ids):
            return None

        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        if all(x == -100 for x in labels):
            return None

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }

    rows = []
    for s in samples:
        out = encode(s)
        if out is not None:
            rows.append(out)
    print(f"  Encoded {len(rows)}/{len(samples)} samples (skipped "
          f"{len(samples) - len(rows)} empty/over-long)", flush=True)
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MT-LoRA (single shared LoRA over all characters).")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--filter_role", default=None,
                        help="(used by train_oppu.py) restrict to one role")

    # LoRA hyper-parameters (OPPU defaults)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"])

    # Optimisation
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        default=True)
    parser.add_argument("--no_gradient_checkpointing",
                        dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=1)
    args = parser.parse_args()

    if args.model is None:
        args.model = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-7B-Instruct")

    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Trainer,
                              TrainingArguments, set_seed)

    set_seed(args.seed)

    print(f"  Loading tokenizer / data…", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    samples = load_samples(args.train_data, filter_role=args.filter_role)
    if not samples:
        raise RuntimeError(
            f"No samples to train on (filter_role={args.filter_role!r})")
    n_roles = len(set(s["role"] for s in samples))
    print(f"  Train data : {args.train_data}", flush=True)
    print(f"  Samples    : {len(samples)} ({n_roles} characters)", flush=True)

    train_ds = build_dataset(samples, tokenizer, args.max_seq_len)

    print(f"  Loading base model: {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, label_pad_token_id=-100,
    )

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False}
            if args.gradient_checkpointing else None,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    print(f"\n{'─' * 50}", flush=True)
    print(f"  Training MT-LoRA on {len(train_ds)} examples", flush=True)
    print(f"  Output     : {args.output_dir}/", flush=True)
    print(f"{'─' * 50}\n", flush=True)
    trainer.train()

    print(f"\n  Saving adapter → {args.output_dir}", flush=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    meta = {
        "method": "mt_lora",
        "model": args.model,
        "train_data": args.train_data,
        "filter_role": args.filter_role,
        "num_samples": len(train_ds),
        "num_roles": n_roles,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(args.output_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"✓ Done.", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
