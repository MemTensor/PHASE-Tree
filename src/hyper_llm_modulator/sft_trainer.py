"""SFT training loop and validation for the hypermod + frozen LLM.

Implements the core training loop that:

1. Samples batches from a hierarchical multi-task dataloader.
2. Encodes profile embeddings through the hypermod to produce LoRA
   weights on the fly.
3. Injects the generated LoRA via forward hooks and computes
   teacher-forced cross-entropy loss on the target tokens.
4. Periodically validates, checkpoints, and logs to W&B.

Also provides checkpoint management (save/load/resume) and an
early-stopping utility.
"""

from collections import defaultdict
from contextlib import contextmanager
from glob import glob
import json
import logging
import os
from functools import partial
import shutil
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import wandb
from peft import PeftModel
from transformers.modeling_utils import unwrap_model
import yaml

from hyper_llm_modulator.hooks import add_lora_hooks, remove_hook_handles_
from hyper_llm_modulator.hyper_modulator import save_hypermod_checkpoint
from hyper_llm_modulator.utils import save_lora_from_peft_model, log_scalar
import re

logger = logging.getLogger()

MODEL_INPUT_KEYS = ["input_ids", "attention_mask"]


# Ref: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    """Patience-based early stopping for validation loss."""

    def __init__(self, patience, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# taken from https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
@contextmanager
def evaluating(*models):
    """Temporarily switch to evaluation mode."""
    is_training = [model.training if model is not None else False for model in models]
    try:
        for model in models:
            if model is not None:
                model.eval()
        yield models
    finally:
        for model, training in zip(models, is_training):
            if model is not None:
                model.train(training)


def neftune_post_forward_hook(module, input, output):
    """
    Implements the NEFTune forward pass for the model using forward hooks. Note this works only for
    torch.nn.Embedding layers. This method is slightly adapted from the original source code
    that can be found here: https://github.com/neelsjain/NEFTune

    Simply add it to your model as follows:
    ```python
    model = ...
    model.embed_tokens.neftune_noise_alpha = 0.1
    model.embed_tokens.register_forward_hook(neftune_post_forward_hook)
    ```

    Args:
        module (`torch.nn.Module`):
            The embedding module where the hook is attached. Note that you need to set
            `module.neftune_noise_alpha` to the desired noise alpha value.
        input (`torch.Tensor`):
            The input tensor to the model.
        output (`torch.Tensor`):
            The output tensor of the model (i.e. the embeddings).
    """
    if module.training:
        dims = output.size(1) * output.size(2)  # Keep as scalar, not tensor
        mag_norm = module.neftune_noise_alpha / torch.sqrt(torch.tensor(dims, dtype=torch.float32))
        # Clamp mag_norm to prevent excessive noise
        mag_norm = torch.clamp(mag_norm, max=1.0)
        output = output + torch.zeros_like(output).uniform_(-mag_norm.item(), mag_norm.item())
    return output


def trl_activate_neftune(model, neftune_noise_alpha):
    r"""
    Activates the neftune as presented in this code: https://github.com/neelsjain/NEFTune and paper: https://arxiv.org/abs/2310.05914
    Since in transformers Trainer we do have an `_activate_neftune` method, we need to rename this method to avoid conflicts.
    """
    unwrapped_model = unwrap_model(model)
    if isinstance(unwrapped_model, PeftModel):
        embeddings = unwrapped_model.base_model.model.get_input_embeddings()
    else:
        embeddings = unwrapped_model.get_input_embeddings()

    embeddings.neftune_noise_alpha = neftune_noise_alpha
    hook_handle = embeddings.register_forward_hook(neftune_post_forward_hook)
    return hook_handle


def get_loss_batch(
    batch,
    model,
    target_modules,
    inp_dropout,
    layer_indices,
    use_hypernet,
    hypermod,
    equally_weight_sample,
    l2_reg_generated_w=0,
    label_smoothing=0,
    return_per_token_acc=False,
    return_entropy=False,
):
    out = dict()
    out["generated_w_l2_loss"] = torch.zeros(1, device=model.device)
    bs = batch["input_ids"].shape[0]
    hook_handles = []

    if use_hypernet:
        # Task embeddings are pre-computed; online embedding of the input
        # prompt (hyperdecoder-style training) is not yet supported.
        encoder_out = hypermod.task_encoder(batch["task_embs"])
        encoded_task_emb = encoder_out["encoded_task_emb"]
        # generated lora weights only once for all samples
        # then hook the generated loras to the model
        factorized_delta_w, hook_handles = generate_and_hook_delta_w(
            target_modules=target_modules,
            inp_dropout=inp_dropout,
            model=model,
            layer_indices=layer_indices,
            hypermod=hypermod,
            encoded_task_emb=encoded_task_emb,
            bs=bs,
            training=model.training,
        )
        if l2_reg_generated_w:
            for A, B in factorized_delta_w.values():
                # Standard L2 regularization: sum of squared weights
                out["generated_w_l2_loss"] += (A.pow(2).mean() + B.pow(2).mean()) * l2_reg_generated_w
    outputs = model(**{k: batch[k] for k in MODEL_INPUT_KEYS})
    out["sft_loss"] = compute_loss(
        batch["labels"],
        outputs.logits,
        equally_weight_sample=equally_weight_sample,
        label_smoothing=label_smoothing,
    )
    if return_per_token_acc or return_entropy:
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        indices = torch.where(shift_labels != -100)
    if return_per_token_acc:
        # only compute acc when batch["labels"] != -100
        out["per_token_acc"] = (shift_logits.argmax(-1) == shift_labels)[indices].float().mean()
    if return_entropy:
        logits = shift_logits[indices]
        prob = torch.nn.functional.softmax(logits, dim=-1)
        # Prevent log(0) by adding small epsilon and using clamp
        eps = 1e-8
        prob = torch.clamp(prob, min=eps)
        out["entropy"] = -torch.sum(prob * torch.log(prob), dim=-1).mean()
    return out, hook_handles


def save_resume_state(save_dir, optimizer, scheduler, curstep):
    """Persist optimizer / scheduler / step so training can resume later."""
    state = {
        "curstep": curstep,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    path = os.path.join(save_dir, "checkpoints", f"it_{curstep}", "resume_state.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    latest_link = os.path.join(save_dir, "resume_state_latest.pt")
    torch.save(state, latest_link)
    logger.info(f"Saved resume state at step {curstep} → {path}")
    return path


def find_latest_checkpoint(run_dir):
    """Return (step, checkpoint_dir) for the highest-step checkpoint in *run_dir*."""
    ckpt_root = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_root):
        return None, None
    steps = []
    for name in os.listdir(ckpt_root):
        m = re.match(r"it_(\d+)$", name)
        if m and os.path.isfile(os.path.join(ckpt_root, name, "hypermod.pt")):
            steps.append(int(m.group(1)))
    if not steps:
        return None, None
    best = max(steps)
    return best, os.path.join(ckpt_root, f"it_{best}")


def load_resume_state(run_dir, device="cpu"):
    """Load the latest resume state from a previous run directory.

    Returns a dict with keys *curstep*, *optimizer*, *scheduler*,
    *hypermod_path*, or ``None`` if no resumable state is found.
    """
    step, ckpt_dir = find_latest_checkpoint(run_dir)
    if step is None:
        return None

    hypermod_path = os.path.join(ckpt_dir, "hypermod.pt")
    resume_pt = os.path.join(ckpt_dir, "resume_state.pt")
    if not os.path.isfile(resume_pt):
        resume_pt = os.path.join(run_dir, "resume_state_latest.pt")
    if not os.path.isfile(resume_pt):
        logger.warning(
            f"Found hypermod checkpoint at step {step} but no resume_state.pt — "
            f"will resume hypermod weights only (optimizer/scheduler reset)."
        )
        return {"curstep": step, "optimizer": None, "scheduler": None, "hypermod_path": hypermod_path}

    state = torch.load(resume_pt, map_location=device, weights_only=False)
    state["hypermod_path"] = hypermod_path
    if state.get("curstep") is None:
        state["curstep"] = step
    logger.info(f"Loaded resume state from step {state['curstep']} ({resume_pt})")
    return state


def train(
    args,
    save_dir,
    inp_dropout,
    accelerator,
    model,
    layer_indices,
    hypermod,
    train_dataloader,
    val_dataloaders,
    optimizer,
    num_training_steps,
    scheduler,
    resume_step=0,
):
    train_start_time = time.perf_counter()
    train_phase_seconds = 0.0
    validation_phase_seconds = 0.0

    def _timed_validate(dataloaders, step):
        nonlocal validation_phase_seconds
        val_start_time = time.perf_counter()
        result = validate(model, hypermod, dataloaders, _get_loss_batch, step)
        validation_phase_seconds += time.perf_counter() - val_start_time
        return result
    model.train()
    if args.use_hypernet:
        hypermod.train()
        wandb.watch(hypermod, log="all", log_freq=1000)

    _log_train_vals = partial(
        log_train_vals,
        len_train_dataloader=len(train_dataloader),
        scheduler=scheduler,
    )

    _get_loss_batch = partial(
        get_loss_batch,
        model=model,
        target_modules=args.target_modules,
        inp_dropout=inp_dropout,
        layer_indices=layer_indices,
        use_hypernet=args.use_hypernet,
        hypermod=hypermod,
        equally_weight_sample=args.equally_weight_sample,
    )
    _get_loss_batch_train = partial(
        _get_loss_batch,
        label_smoothing=args.label_smoothing,
        l2_reg_generated_w=args.l2_reg_generated_w,
    )

    neftune_hook_handle = trl_activate_neftune(model, args.neftune_noise_alpha)

    # Initialize best validation loss tracking (generalized to any monitored split)
    best_val_loss = float('inf')
    best_checkpoint_path = None
    best_step = 0
    monitored_dataset_name = None  # which val split to monitor for best
    
    # Track top-k checkpoints with lowest validation loss
    top_k_limit = max(0, int(getattr(args, "top_k_checkpoints", 0)))
    top_k_checkpoints = []  # list of dicts: {val_loss, step, checkpoint_path}
    top_k_info_path = os.path.join(save_dir, "top_k_checkpoints.yaml")
    
    def _maybe_update_top_k(val_loss: float, step: int, checkpoint_path: str):
        nonlocal top_k_checkpoints
        if top_k_limit <= 0:
            return
        # Add/update entry for this step
        top_k_checkpoints.append({
            "val_loss": float(val_loss),
            "step": int(step),
            "checkpoint_path": checkpoint_path,
        })
        # Keep only top-k by lowest val loss
        top_k_checkpoints = sorted(top_k_checkpoints, key=lambda x: x["val_loss"])[: top_k_limit]
        # Persist metadata to disk for debugging/resume
        try:
            with open(top_k_info_path, "w") as f:
                yaml.safe_dump({
                    "k": top_k_limit,
                    "monitored_dataset": monitored_dataset_name,
                    "top_k": top_k_checkpoints,
                }, f, sort_keys=False)
        except Exception as e:
            logger.warning(f"Failed to write top-k checkpoint info: {e}")
    
    # Initialize early stopping for hypernet training
    hypernet_early_stopper = None
    if args.use_early_stopping and args.use_hypernet:
        hypernet_early_stopper = EarlyStopper(
            patience=args.early_stopping_patience, 
            min_delta=args.early_stopping_min_delta
        )
        logger.info(f"Early stopping enabled for hypernet training with patience={args.early_stopping_patience}")

    # validate before training (unless skip_val is True)
    val_info = {}
    if not args.skip_val:
        if args.also_val_on_train:
            val_info = _timed_validate({"train": train_dataloader}, 0)
        val_info = _timed_validate(val_dataloaders, 0)

    # On hypernet training, always save an initial checkpoint at step 0
    if args.use_hypernet:
        cp_path = save_hypermod_checkpoint(save_dir, hypermod, curstep=0)

        # Decide which validation split to monitor for "best" if we have validation info
        if val_info:
            if "val/unseen" in val_info:
                monitored_dataset_name = "val/unseen"
            elif "val/seen" in val_info:
                monitored_dataset_name = "val/seen"
            else:
                monitored_dataset_name = next(iter(val_info.keys()))

            current_val_loss = val_info[monitored_dataset_name]["sft_loss"]
            best_val_loss = current_val_loss
            best_checkpoint_path = cp_path
            best_step = 0
            logger.info(
                f"Monitoring {monitored_dataset_name}/sft_loss. Initial best: {best_val_loss:.4f} at step {best_step}"
            )
            if hypernet_early_stopper is not None:
                hypernet_early_stopper.early_stop(current_val_loss)
            # Track in top-k list
            _maybe_update_top_k(current_val_loss, best_step, cp_path)
    elif "mt_lora" in args.exp_setup:
        lora_dir = save_lora_checkpoint(save_dir, model, args.model_dir, curstep=0)
    elif "val/seen" in val_info:
        # normal LoRA training
        stopper = EarlyStopper(patience=3, min_delta=0)
        stopper.early_stop(val_info["val/seen"]["sft_loss"])

    curstep = 1
    grad_norm = 0
    avg_losses = defaultdict(list)
    early_stop = False
    nan_detected = False

    if resume_step > 0:
        logger.info(f"RESUME: will fast-forward from step 1 to step {resume_step} (skipping completed steps)")

    for epoch_idx in (pbar := tqdm(range(args.epochs), total=num_training_steps)):
        batch_idx = 0
        total_batches_in_epoch = len(train_dataloader)

        for batch in train_dataloader:
            batch_idx += 1
            is_end_of_epoch = (batch_idx == total_batches_in_epoch)

            if resume_step > 0 and curstep <= resume_step:
                pbar.update(1)
                pbar.set_description(f"skip {curstep}/{resume_step}")
                curstep += 1
                continue

            ##########################################
            # Training
            ##########################################
            train_step_start = time.perf_counter()
            with accelerator.accumulate(model), accelerator.autocast():
                batch_loss, hook_handles = _get_loss_batch_train(batch)
                try:
                    loss = batch_loss["sft_loss"] + batch_loss["generated_w_l2_loss"]

                    # Check for NaN and provide debugging info
                    if (
                        torch.isnan(loss)
                        or torch.isnan(batch_loss["sft_loss"])
                        or torch.isnan(batch_loss["generated_w_l2_loss"])
                    ):
                        logger.error(f"NaN detected at step {curstep}!")
                        try:
                            logger.error(f"sft_loss: {float(batch_loss['sft_loss'].item())}")
                        except Exception:
                            logger.error("sft_loss: nan")
                        try:
                            logger.error(f"l2_loss: {float(batch_loss['generated_w_l2_loss'].item())}")
                        except Exception:
                            logger.error("l2_loss: nan")
                        try:
                            logger.error(f"total_loss: {float(loss.item())}")
                        except Exception:
                            logger.error("total_loss: nan")
                        # Gracefully stop training loop to evaluate best/last checkpoint
                        nan_detected = True
                        early_stop = True
                        break

                    avg_losses["train/sft_loss"].append(batch_loss["sft_loss"].item())
                    avg_losses["train/generated_w_l2_loss"].append(batch_loss["generated_w_l2_loss"].item())
                    avg_losses["train/total_loss"].append(loss.item())

                    # NOTE: zero_grad MUST come AFTER step() (not before backward), otherwise
                    # AcceleratedOptimizer.zero_grad — which only fires on sync steps — will wipe
                    # the accumulated gradients from earlier micro-batches in the same accumulation
                    # window, silently making `grad_accum_steps` a no-op.
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                finally:
                    remove_hook_handles_(hook_handles)
            train_phase_seconds += time.perf_counter() - train_step_start

            pbar.update(1)
            pbar.set_description(f"loss: {loss.item():.4f}")

            ##########################################
            # Logging and Validation
            ##########################################
            if (curstep % args.logging_freq == 0) or (curstep == num_training_steps):
                _log_train_vals(grad_norm, avg_losses, curstep)
                # reset avg_losses
                avg_losses = defaultdict(list)

            # Determine when to validate and save checkpoints based on use_hierarchical_sampler
            should_validate = False
            if args.use_hierarchical_sampler:
                # Original behavior: validate every val_freq steps
                should_validate = (curstep % args.val_freq == 0) or (curstep == num_training_steps)
            else:
                # New behavior: validate at end of each epoch
                should_validate = is_end_of_epoch or (curstep == num_training_steps)
            
            if should_validate:
                if args.also_val_on_train:
                    val_info = _timed_validate({"train": train_dataloader}, curstep)

                val_info = _timed_validate(val_dataloaders, curstep)
                
                if args.use_hypernet:
                    cp_path = save_hypermod_checkpoint(save_dir, hypermod, curstep)
                    save_resume_state(save_dir, optimizer, scheduler, curstep)

                    # Decide monitored split if not decided yet
                    if monitored_dataset_name is None and val_info:
                        if "val/unseen" in val_info:
                            monitored_dataset_name = "val/unseen"
                        elif "val/seen" in val_info:
                            monitored_dataset_name = "val/seen"
                        else:
                            monitored_dataset_name = next(iter(val_info.keys()))
                        logger.info(f"Monitoring {monitored_dataset_name}/sft_loss for best checkpoint selection")

                    # Track best validation loss for the monitored split
                    if monitored_dataset_name is not None and monitored_dataset_name in val_info:
                        current_val_loss = val_info[monitored_dataset_name]["sft_loss"]
                        if current_val_loss < best_val_loss:
                            best_val_loss = current_val_loss
                            best_checkpoint_path = cp_path
                            best_step = curstep
                            logger.info(
                                f"New best {monitored_dataset_name}/sft_loss: {best_val_loss:.4f} at step {best_step}"
                            )
                            # Save the best checkpoint as the final hypermod
                            shutil.copy(cp_path, f"{save_dir}/hypermod.pt")
                            # Persist best checkpoint metadata
                            try:
                                best_info_path = os.path.join(save_dir, "best_checkpoint_info.yaml")
                                with open(best_info_path, "w") as f:
                                    yaml.safe_dump(
                                        {
                                            "best_checkpoint_path": best_checkpoint_path,
                                            "best_step": int(best_step),
                                            "best_val_loss": float(best_val_loss),
                                            "monitored_dataset": monitored_dataset_name,
                                        },
                                        f,
                                        sort_keys=False,
                                    )
                                logger.debug(f"Wrote best checkpoint info to {best_info_path}")
                            except Exception as e:
                                logger.warning(f"Failed to write best checkpoint info: {e}")
                        # Track in top-k list
                        _maybe_update_top_k(current_val_loss, curstep, cp_path)

                        # Early stopping check on monitored split
                        if hypernet_early_stopper is not None:
                            if hypernet_early_stopper.early_stop(current_val_loss):
                                logger.info(
                                    f"Early stopping triggered! No improvement in {monitored_dataset_name}/sft_loss for {args.early_stopping_patience} validation checks."
                                )
                                logger.info(
                                    f"Best {monitored_dataset_name}/sft_loss: {best_val_loss:.4f} at step {best_step}"
                                )
                                early_stop = True
                                break
                
                elif "mt_lora" in args.exp_setup:
                    lora_dir = save_lora_checkpoint(save_dir, model, args.model_dir, curstep)
                elif "val/seen" in val_info:
                    if stopper.early_stop(val_info["val/seen"]["sft_loss"]):
                        logger.info("Early stopping")
                        early_stop = True
                        break

                # read early stop signal from the watcher
                if os.path.isfile(f"{save_dir}/earlystop_info.yaml"):
                    early_stop = True
                    break

            curstep += 1
        if early_stop:
            break

    total_runtime_seconds = time.perf_counter() - train_start_time
    _persist_training_timing(
        save_dir,
        total_runtime_seconds,
        train_phase_seconds,
        validation_phase_seconds,
    )

    if args.use_hypernet:
        last_cp_path = save_hypermod_checkpoint(save_dir, hypermod, curstep)

        # Use the best checkpoint if we found one, otherwise use the last one
        final_hypermod_path = f"{save_dir}/hypermod.pt"

        if best_checkpoint_path is None:
            best_info_path = os.path.join(save_dir, "best_checkpoint_info.yaml")
            if os.path.isfile(best_info_path):
                try:
                    with open(best_info_path, "r") as f:
                        persisted = yaml.safe_load(f) or {}
                    candidate = persisted.get("best_checkpoint_path")
                    if candidate and os.path.isfile(candidate):
                        best_checkpoint_path = candidate
                        best_step = persisted.get("best_step", best_step)
                        best_val_loss = persisted.get("best_val_loss", best_val_loss)
                        monitored_dataset_name = persisted.get(
                            "monitored_dataset", monitored_dataset_name
                        )
                        logger.info(
                            "Recovered best checkpoint from persisted metadata: %s",
                            best_checkpoint_path,
                        )
                    else:
                        logger.warning(
                            "Persisted best checkpoint metadata found but checkpoint is missing."
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to load persisted best checkpoint metadata from {best_info_path}: {e}"
                    )

        if best_checkpoint_path is not None:
            if monitored_dataset_name is None:
                logger.info("Using best checkpoint but monitored split name is unknown")
            else:
                logger.info(
                    f"Using best checkpoint from step {best_step} with {monitored_dataset_name}/sft_loss: {best_val_loss:.4f}"
                )
            if os.path.abspath(best_checkpoint_path) != os.path.abspath(final_hypermod_path):
                shutil.copy(best_checkpoint_path, final_hypermod_path)
        else:
            if nan_detected:
                logger.info("NaN encountered during training; using last checkpoint for evaluation")
            else:
                logger.info("No val/unseen dataset found, using last checkpoint")
            if os.path.abspath(last_cp_path) != os.path.abspath(final_hypermod_path):
                shutil.copy(last_cp_path, final_hypermod_path)

        # PHASE-Tree training stops here; generative evaluation is performed
        # separately via `evaluation/predict_hypernet.py` + `judge.py`.
    elif "mt_lora" in args.exp_setup:
        lora_dir = save_lora_checkpoint(save_dir, model, args.model_dir, curstep)
        if not os.path.isfile(f"{save_dir}/adapter_model.safetensors"):
            shutil.copy(f"{lora_dir}/adapter_model.safetensors", f"{save_dir}/adapter_model.safetensors")
        if not os.path.isfile(f"{save_dir}/config.json"):
            shutil.copy(f"{lora_dir}/config.json", f"{save_dir}/config.json")
    elif "mt_fullfinetune" in args.exp_setup:
        model.save_pretrained(save_dir)
    else:
        lora_dir = save_lora_checkpoint(save_dir, model, args.model_dir, curstep)
        shutil.copy(f"{lora_dir}/adapter_model.safetensors", f"{save_dir}/adapter_model.safetensors")

    if args.keep_only_best:
        # Keep the last checkpoint and any top-k checkpoints
        cp_dirs = sorted(glob(f"{save_dir}/checkpoints/it_*"), key=os.path.getmtime)
        # Determine steps to keep
        keep_steps = set()
        # Always keep the highest (last) step
        if cp_dirs:
            try:
                last_step = int(os.path.basename(cp_dirs[-1]).split("it_")[-1])
                keep_steps.add(last_step)
            except Exception:
                pass
        # Keep top-k steps
        for entry in top_k_checkpoints:
            keep_steps.add(int(entry.get("step", -1)))
        # Remove all others
        for cp_dir in cp_dirs:
            try:
                step = int(os.path.basename(cp_dir).split("it_")[-1])
            except Exception:
                step = -1
            if step not in keep_steps:
                shutil.rmtree(cp_dir)

    wandb.unwatch(hypermod)
    accelerator.end_training()
    neftune_hook_handle.remove()
    model.eval()
    if args.use_hypernet:
        hypermod.eval()


def validate(model, hypermod, val_dataloaders, _get_loss_batch, curstep):
    logger.info(f"Validating at step {curstep}")
    with torch.no_grad(), evaluating(model, hypermod):
        out = dict()
        for val_dataloader_name, val_dataloader in val_dataloaders.items():
            if val_dataloader is None:
                continue
            val_info = defaultdict(list)
            for val_batch in val_dataloader:
                if val_batch is None:
                    break
                batch_loss, hook_handles = _get_loss_batch(
                    val_batch, return_per_token_acc=True, return_entropy=True
                )
                try:
                    val_info["sft_loss"].append(batch_loss["sft_loss"].item())
                    val_info["per_token_acc"].append(batch_loss["per_token_acc"].item())
                    val_info["entropy"].append(batch_loss["entropy"].item())
                finally:
                    remove_hook_handles_(hook_handles)
            for k, v in val_info.items():
                val_info[k] = np.mean(v)
                log_scalar(f"{val_dataloader_name}/{k}", val_info[k], curstep)
            out[val_dataloader_name] = val_info
    return out


def _persist_training_timing(
    save_dir: str,
    total_elapsed_seconds: float,
    train_phase_seconds: float,
    validation_phase_seconds: float,
) -> None:
    """Persist wall clock, training, and validation runtimes (seconds)."""
    if total_elapsed_seconds is None:
        return

    timing_path = os.path.join(save_dir, "timing_stats.json")
    train_only_seconds = float(train_phase_seconds or 0.0)
    validation_seconds = float(validation_phase_seconds or 0.0)
    total_seconds = float(total_elapsed_seconds)
    overhead_seconds = max(total_seconds - train_only_seconds - validation_seconds, 0.0)

    payload = {
        "total_runtime_seconds": total_seconds,
        "training_time_seconds": train_only_seconds,
        "validation_time_seconds": validation_seconds,
    }
    if overhead_seconds > 0.0:
        payload["other_overhead_seconds"] = overhead_seconds
    try:
        if os.path.exists(timing_path):
            with open(timing_path, "r", encoding="utf-8") as fh:
                existing = json.load(fh) or {}
        else:
            existing = {}
        existing.update(payload)
        with open(timing_path, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=4)
        logger.info(
            "Recorded training timing details to %s (total=%.2fs, train=%.2fs, val=%.2fs)",
            timing_path,
            total_seconds,
            train_only_seconds,
            validation_seconds,
        )
    except Exception as exc:
        logger.warning(f"Failed to persist training timing info: {exc}")


def save_lora_checkpoint(save_dir, model, model_dir, curstep):
    lora_dir = f"{save_dir}/checkpoints/it_{curstep}/"
    save_lora_from_peft_model(model, model_dir, lora_dir)
    if os.path.exists(f"{save_dir}/adapter_config.json"):
        shutil.copy(f"{save_dir}/adapter_config.json", f"{lora_dir}/adapter_config.json")
    return lora_dir


def log_train_vals(grad_norm, avg_losses, curstep, len_train_dataloader, scheduler):
    wandb.log(
        {
            "train/total_loss": np.mean(avg_losses["train/total_loss"]),
            "train/sft_loss": np.mean(avg_losses["train/sft_loss"]),
            "train/generated_w_l2_loss": np.mean(avg_losses["train/generated_w_l2_loss"]),
            "train/learning_rate": scheduler.get_last_lr()[0],
            "train/epoch": curstep / len_train_dataloader,
            "train/global_step": curstep,
            "train/grad_norm": grad_norm,
        },
        step=curstep,
    )
    logger.info(
        f"train/total_loss: {np.mean(avg_losses['train/total_loss']):.4f} "
        f"|| train/sft_loss: {np.mean(avg_losses['train/sft_loss']):.4f} "
        f"|| train/generated_w_l2_loss: {np.mean(avg_losses['train/generated_w_l2_loss']):.4f} "
    )


def compute_loss(labels, logits, equally_weight_sample, label_smoothing):
    bs = logits.shape[0]
    vocab_size = logits.shape[-1]
    # based on HG Transformers
    # modified to weight each example equally
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    max_seq_len = shift_labels.shape[1]
    seq_len = torch.where(shift_labels != -100, 1, 0).sum(-1, keepdim=True)
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Ensure tensors are on the same device
    if equally_weight_sample:
        # weight each sample equally
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smoothing)
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss.view(bs, max_seq_len)
        # Prevent division by zero: only compute loss for samples with valid tokens
        valid_samples = seq_len.squeeze(-1) > 0
        if valid_samples.any():
            loss = (loss[valid_samples] / seq_len[valid_samples]).sum(-1).mean()
        else:
            # If no valid samples, return zero loss
            loss = torch.zeros(1, device=loss.device, requires_grad=True)
    else:
        # weight each token equally
        loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        loss = loss_fct(shift_logits, shift_labels)
    return loss


def generate_and_hook_delta_w(
    target_modules,
    inp_dropout,
    model,
    layer_indices,
    hypermod,
    encoded_task_emb,
    bs,
    training,
):
    hook_handles = []
    factorized_delta_w = dict()
    for target_module in target_modules:
        factorized_delta_w[target_module] = hypermod.get_delta_weights(
            layer_indices.repeat_interleave(bs),
            target_module,
            encoded_task_emb.tile(layer_indices.shape[0], 1),
            factorized=True,
        )
        lora_A, lora_B = factorized_delta_w[target_module]
        for layer_index in layer_indices:
            start_indices, end_indices = layer_index * bs, (layer_index + 1) * bs
            handles = add_lora_hooks(
                model,
                [target_module],
                [layer_index],
                lora_A[start_indices:end_indices].transpose(-1, -2),  # [bs, in_features, r]
                lora_B[start_indices:end_indices].transpose(-1, -2),  # [bs, r, out_features]
                hypermod.scaling,
                inp_dropout,
                training,
            )
            hook_handles += handles
    return factorized_delta_w, hook_handles
