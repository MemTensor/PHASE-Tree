"""PHASE-Tree hypermod / LoRA SFT training entry point.

Orchestrates the full training pipeline:

1. Parses a YAML config + CLI overrides into :class:`TrainingArguments`.
2. Loads the frozen base LLM and tokenizer.
3. Initialises the hypermod (optionally warm-started from a checkpoint).
4. Builds hierarchical multi-task dataloaders from the configured
   PHASE-Tree role-play datasets.
5. Runs the training loop via :func:`sft_trainer.train`.

Usage::

    python train_custom_sft.py src/configs/phase_tree_hyper_lora.yaml \\
        --lr=5e-6 --epochs=40000
"""

import gc
import os
import random
import string
import subprocess
import sys
import time
from copy import deepcopy
from glob import glob
from pathlib import Path

# Ensure PHASE-Tree's own src/ takes precedence over any stale editable install.
_PHASE_TREE_SRC = str(Path(__file__).resolve().parents[1])
if _PHASE_TREE_SRC not in sys.path:
    sys.path.insert(0, _PHASE_TREE_SRC)

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from datasets import disable_caching
from transformers import get_scheduler, set_seed

from hyper_llm_modulator.configs import ArgumentParser, TrainingArguments
from hyper_llm_modulator.data import create_dataloaders
from hyper_llm_modulator.hyper_modulator import create_hypermod
from hyper_llm_modulator.sft_trainer import train, load_resume_state, find_latest_checkpoint
from hyper_llm_modulator.utils import (
    add_full_stop,
    create_logger,
    get_layers,
    get_metadata,
    get_model_and_tokenizer,
    get_num_params,
    get_peft_config,
    get_pooling_fn,
    get_tokenizer,
    save_yaml,
)
from hyper_llm_modulator.utils.model_loading import get_emb_model_and_fns

MODEL_INPUT_KEYS = ["input_ids", "attention_mask"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _configure_gradient_checkpointing(model, logger):
    """Enable gradient checkpointing and disable caches across wrapped models."""

    def _set_use_cache_false(module):
        if module is None:
            return
        if hasattr(module, "config") and getattr(module.config, "use_cache", True):
            module.config.use_cache = False
        if hasattr(module, "generation_config") and getattr(module.generation_config, "use_cache", True):
            module.generation_config.use_cache = False

    def _enable_input_grads(module):
        if module is None:
            return
        if hasattr(module, "enable_input_require_grads"):
            try:
                module.enable_input_require_grads()
            except Exception:
                pass

    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    for attr in [None, "model", "base_model", "module"]:
        target = model if attr is None else getattr(model, attr, None)
        _set_use_cache_false(target)
        _enable_input_grads(target)

    logger.info("Gradient checkpointing enabled; caches disabled for wrapped model hierarchy")


def _load_model(args, peft_config, device, logger):
    """Load LLM + tokenizer, enable grad ckpt, compute layer_indices."""
    model, tokenizer = get_model_and_tokenizer(
        args.model_dir,
        train=True,
        requires_grad="fullfinetune" in args.exp_setup,
        peft_config=peft_config,
        model_kwargs={"output_hidden_states": True, "output_attentions": False},
        device=device,
    )
    if args.gradient_checkpointing:
        _configure_gradient_checkpointing(model, logger)

    layer_indices = torch.tensor(range(len(get_layers(model))), dtype=torch.long, device=device)

    assert tokenizer.chat_template is not None, "Only chat models are supported"
    logger.debug(f"Model config: {model.config}")
    logger.debug(f"is_intx_model: True")
    logger.debug(f"Tokenizer: {tokenizer}")
    logger.debug(f"layer_indices: {layer_indices}")

    return model, tokenizer, layer_indices


def _setup_hypermod(args, peft_type, device, model, layer_indices, task_emb_size, logger):
    """Create hypermod (if hypernet) or activate adapter/train mode.

    `task_emb_size` is the hidden size of whatever model produces task embeddings
    (explicit emb model in Path A, or the LLM itself in Path B). Pass `None` when
    `use_one_hot_task_emb=True`.
    """
    use_hypernet = args.use_hypernet
    hypermod = None

    if use_hypernet:
        effective_size = None if args.use_one_hot_task_emb else task_emb_size
        hypermod = create_hypermod(args, peft_type, device, model, layer_indices, effective_size)
        logger.debug(f"Hypermod: {hypermod}")
        model.add_module("hypermod", hypermod)
    elif "lora" in args.exp_setup:
        model.set_adapter("default")
    else:
        model.train()

    return hypermod


def _warm_start_hypermod(args, hypermod, logger):
    """Load pretrained hypermod weights if --init_hypermod_from is set."""
    init_path = getattr(args, "init_hypermod_from", None)
    if hypermod is None or not init_path:
        return

    if not os.path.isfile(init_path):
        raise FileNotFoundError(f"--init_hypermod_from points to a missing file: {init_path}")

    logger.info("=" * 70)
    logger.info(f"WARM-START: loading pretrained hypermod weights from {init_path}")
    state = torch.load(init_path, map_location="cpu", weights_only=False)
    info = hypermod.load_state_dict(state, strict=False)
    logger.info(
        f"  loaded {len(state)} tensors | "
        f"missing={len(info.missing_keys)} | unexpected={len(info.unexpected_keys)}"
    )
    if info.missing_keys:
        logger.info(f"  missing keys (first 10): {info.missing_keys[:10]}")
    if info.unexpected_keys:
        logger.info(f"  unexpected keys (first 10): {info.unexpected_keys[:10]}")
    if getattr(args, "init_hypermod_strict", False) and (info.missing_keys or info.unexpected_keys):
        raise RuntimeError(
            f"init_hypermod_strict=True but state_dict load reported "
            f"{len(info.missing_keys)} missing / {len(info.unexpected_keys)} unexpected keys."
        )
    logger.info("WARM-START: done. Optimizer/scheduler/step-counter are NOT restored.")
    logger.info("=" * 70)
    del state
    gc.collect()


def _log_trainable_params(model, logger):
    logger.debug("Trainable model parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            logger.debug(f"  {name}, dtype:{p.dtype}")

    num_total, num_trainable = get_num_params(model)
    logger.info(
        f"trainable params: {num_trainable:,d} "
        f"|| all params: {num_total:,d} "
        f"|| trainable%: {100 * num_trainable / num_total:.4f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args, logger):
    args.train_ds_names = args.train_ds_names[: args.n_train_ds]
    args.use_hypernet = use_hypernet = "hyper" in args.exp_setup
    train_metadata = get_metadata(args.train_ds_names, args.use_per_task_emb)
    val_metadata = get_metadata(args.eval_ds_info, args.use_per_task_emb)

    save_dir = args.save_dir
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    save_yaml(vars(args), f"{save_dir}/args.yaml")
    set_seed(args.seed)

    # ── PEFT config ──────────────────────────────────────────────────────
    peft_config = None
    peft_type = args.exp_setup.split("_")[-1]
    if peft_type == "lora":
        peft_config = get_peft_config(args.model_dir, peft_type, target_modules=args.target_modules)
        peft_config.save_pretrained(save_dir)
        logger.debug(f"peft_config:\n{peft_config}")
    else:
        logger.warning(f"peft_type: {peft_type}. Doing normal full-finetuning without any PEFT.")

    # ── Accelerator ──────────────────────────────────────────────────────
    plugin = GradientAccumulationPlugin(num_steps=args.grad_accum_steps, sync_with_dataloader=False)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_plugin=plugin,
        split_batches=True,
        log_with="wandb",
    )

    def clear_mem():
        torch.cuda.empty_cache()
        accelerator.free_memory()
        gc.collect()

    wandb_dir = f"{os.environ['HOME']}/.wandb/logs/{os.environ['WANDB_PROJECT']}/"
    os.makedirs(wandb_dir, exist_ok=True)
    accelerator.init_trackers(
        os.getenv("WANDB_PROJECT"),
        config=vars(args),
        init_kwargs=dict(wandb={"group": args.run_name, "name": args.run_name, "dir": wandb_dir, "notes": args.notes}),
    )
    device = accelerator.device

    # ── Embedding model / data / LLM ─────────────────────────────────────
    # Two code paths depending on whether an explicit embedding model is used:
    #   Path A: explicit emb model → load emb first for dataloaders → free → load LLM
    #   Path B: use LLM itself as emb model → load LLM first → dataloaders after
    use_explicit_emb = use_hypernet and not args.use_one_hot_task_emb and args.emb_model
    task_emb_size = None  # cached hidden_size of whatever model produces task embeddings

    if use_explicit_emb:
        # ── Path A ───────────────────────────────────────────────────────
        emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn = get_emb_model_and_fns(
            args.emb_model, device, getattr(args, "user_profile_format", None)
        )
        emb_model.eval()
        # Cache hidden_size BEFORE freeing the emb model so we don't have to reload
        # ~8GB of weights again just to read a single integer for hypermod construction.
        task_emb_size = emb_model.config.hidden_size
        temp_tokenizer = get_tokenizer(args.model_dir, train=True, peft_config=peft_config)

        data_loaders = create_dataloaders(
            args, train_metadata, val_metadata, use_hypernet, device,
            temp_tokenizer, True, emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn,
        )
        train_dataloader = data_loaders["train"]
        val_dataloaders = {k: v for k, v in data_loaders.items() if "train" not in k}

        del emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn
        clear_mem()
        logger.info("Cleaned up explicit embedding model to free memory before loading main model")

        model, tokenizer, layer_indices = _load_model(args, peft_config, device, logger)
    else:
        # ── Path B ───────────────────────────────────────────────────────
        model, tokenizer, layer_indices = _load_model(args, peft_config, device, logger)

        emb_model = emb_tokenizer = task_desc_format_fn = pooling_fn = None
        if use_hypernet and not args.use_one_hot_task_emb:
            emb_model = model
            emb_tokenizer = deepcopy(tokenizer)
            task_desc_format_fn = add_full_stop
            pooling_fn = get_pooling_fn("last_token")
            emb_model.eval()
            task_emb_size = model.config.hidden_size

        data_loaders = create_dataloaders(
            args, train_metadata, val_metadata, use_hypernet, device,
            tokenizer, True, emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn,
        )
        train_dataloader = data_loaders["train"]
        val_dataloaders = {k: v for k, v in data_loaders.items() if "train" not in k}

    # ── Hypermod setup + warm-start ──────────────────────────────────────
    hypermod = _setup_hypermod(args, peft_type, device, model, layer_indices, task_emb_size, logger)
    _warm_start_hypermod(args, hypermod, logger)

    if getattr(args, "freeze_heads", False) and hypermod is not None:
        frozen_count = 0
        for name, p in hypermod.named_parameters():
            if name.startswith("heads."):
                p.requires_grad = False
                frozen_count += 1
        logger.info(
            f"FREEZE_HEADS: froze {frozen_count} parameter tensors in hypermod.heads "
            f"({sum(p.numel() for p in hypermod.heads.parameters()):,d} params)"
        )

    _log_trainable_params(model, logger)

    # ── Optimizer / Scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    model, hypermod, optimizer = accelerator.prepare(model, hypermod, optimizer)
    train_dataloader = accelerator.prepare(train_dataloader)
    for k, v in val_dataloaders.items():
        val_dataloaders[k] = accelerator.prepare(v)

    num_training_steps = args.epochs * len(train_dataloader)
    num_warmup_steps = args.warmup_frac * num_training_steps
    scheduler = get_scheduler(
        "linear", optimizer,
        num_warmup_steps=int(num_warmup_steps / args.grad_accum_steps),
        num_training_steps=int(num_training_steps / args.grad_accum_steps),
    )
    scheduler = accelerator.prepare(scheduler)
    inp_dropout = getattr(peft_config, f"{peft_type.lower()}_dropout", 0.0)

    # ── Resume from previous run (if requested) ──────────────────────────
    resume_step = 0
    resume_run = getattr(args, "resume_from_run", None)
    if resume_run:
        logger.info(f"RESUME: attempting to resume from {resume_run}")
        rstate = load_resume_state(resume_run)
        if rstate is None:
            logger.warning(f"RESUME: no checkpoint found in {resume_run}, training from scratch")
        else:
            resume_step = rstate["curstep"]
            hypermod_path = rstate["hypermod_path"]
            logger.info(f"RESUME: restoring hypermod from {hypermod_path} (step {resume_step})")
            hstate = torch.load(hypermod_path, map_location="cpu", weights_only=False)
            hypermod.load_state_dict(hstate, strict=False)
            del hstate
            gc.collect()

            if rstate.get("optimizer") is not None:
                try:
                    optimizer.load_state_dict(rstate["optimizer"])
                    logger.info("RESUME: optimizer state restored")
                except Exception as e:
                    logger.warning(f"RESUME: failed to restore optimizer state: {e}")
            if rstate.get("scheduler") is not None:
                try:
                    scheduler.load_state_dict(rstate["scheduler"])
                    logger.info("RESUME: scheduler state restored")
                except Exception as e:
                    logger.warning(f"RESUME: failed to restore scheduler state: {e}")
            logger.info(f"RESUME: will skip {resume_step} completed steps and continue training")
            del rstate

    # ── Train ────────────────────────────────────────────────────────────
    train(
        args, save_dir, inp_dropout, accelerator, model, layer_indices,
        hypermod, train_dataloader, val_dataloaders, optimizer, num_training_steps, scheduler,
        resume_step=resume_step,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("WANDB_PROJECT", "hypermod_sft")
    os.environ.setdefault("WANDB_WATCH", "all")
    os.environ.setdefault("WANDB_CONSOLE", "off")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    disable_caching()

    parser = ArgumentParser((TrainingArguments,))
    args = parser.parse()
    assert (
        args.use_per_task_emb + args.use_inp_as_desc + args.use_per_sample_desc + args.use_default_desc
    ) <= 1, "only one or none of use_per_task_emb, use_inp_as_desc, use_per_sample_desc can be used"
    assert (
        args.use_per_task_emb or not args.use_one_hot_task_emb
    ), "one_hot_task_emb can only be used with use_per_task_emb"

    resume_run = getattr(args, "resume_from_run", None)
    if resume_run and os.path.isdir(resume_run):
        args.save_dir = resume_run
        args.run_name = os.path.basename(resume_run)
    else:
        uid = "".join(random.choices(string.ascii_letters + string.digits, k=8))
        args.run_name = time.strftime("%Y%m%d-%H%M%S") + f"_{uid}"
        args.save_dir = f"phase_tree_models/sft/{args.exp_setup}/{args.run_name}"

    logger = create_logger(args.save_dir, debug=args.debug)
    logger.debug(f"CMD: {' '.join(sys.argv)}")
    logger.debug(f"args: {args}")
    logger.debug(f"Is CUDA available: {torch.cuda.is_available()}")
    logger.debug(f"CUDA Index: {torch.cuda.current_device()}")
    logger.debug(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    main(args, logger)

    subprocess.run("wandb sync --no-include-online --clean", shell=True)
    subprocess.run("wandb artifact cache cleanup 10GB", shell=True)
