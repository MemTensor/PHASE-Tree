#!/usr/bin/env python3
"""Compute SFT validation loss on PHASE-Tree task test sets for saved hypermod checkpoints.

This mirrors training-time validation in ``sft_trainer.validate`` + ``get_loss_batch`` (teacher-forced
CE on labels). For downstream generative evaluation (vLLM + judge), see ``evaluation/`` scripts.

By default, **all** eval tasks from merged ``eval_ds_info`` are used (typically every
``*_random_test`` and ``*_ood_test`` in ``phase_tree_hyper_lora.yaml``). After **each** 10k
checkpoint, results are written under ``<run_dir>/eval_ckpt_val_loss/`` (``it_<step>.json`` +
append to ``metrics.csv``) so progress is not lost if the job stops mid-run.

Typical use::

    cd PHASE-Tree
    PYTHONPATH=src:$PYTHONPATH python src/scripts/eval_hypermod_ckpt_val_loss.py \\
        --run_dir phase_tree_models/sft/hyper_lora/<run_id> \\
        --step_start 10000 --step_stride 10000

Optional: ``--eval_suffix random_test`` to restrict to task names ending with that suffix.

If the run's ``args.yaml`` has empty ``eval_ds_info``, ``--base_config`` (default:
``src/configs/phase_tree_hyper_lora.yaml``) supplies the full eval task list.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import numbers
import os
import sys
from argparse import Namespace
from copy import deepcopy
from functools import partial
from glob import glob
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch
from datasets import disable_caching
from peft import get_peft_config as _peft_get_peft_config, PeftConfig

from hyper_llm_modulator.data import create_dataloaders
from hyper_llm_modulator.sft_trainer import get_loss_batch, validate
from hyper_llm_modulator.hyper_modulator import create_hypermod
from hyper_llm_modulator.utils import get_layers, get_metadata, get_tokenizer
from hyper_llm_modulator.utils import get_peft_config as _custom_get_peft_config
from hyper_llm_modulator.utils.model_loading import get_emb_model_and_fns, get_model_and_tokenizer
from hyper_llm_modulator.utils.task_metadata import TASKS_DIRECTORY

logger = logging.getLogger(__name__)


class _DeviceDataLoader:
    """Wraps a DataLoader so every tensor in each batch is moved to *device*."""

    def __init__(self, dataloader, device: torch.device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for batch in self.dataloader:
            if batch is None:
                yield batch
                continue
            yield {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

    def __len__(self):
        return len(self.dataloader)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_train_ds_names(names: list[str]) -> list[str]:
    """Map short names (e.g. RAIDEN) to tasks/RAIDEN_train when the short task dir is missing."""
    resolved: list[str] = []
    for n in names:
        if (TASKS_DIRECTORY / n).is_dir():
            resolved.append(n)
            continue
        cand = f"{n}_train"
        if (TASKS_DIRECTORY / cand).is_dir():
            resolved.append(cand)
            continue
        resolved.append(n)
    return resolved


def _rewrite_paths_for_shm(payload: dict) -> dict:
    """Optional speed-up: remap model paths to a tmpfs RAM disk when available.

    If a model directory exists under ``/dev/shm/phase/models/<name>`` or
    ``/dev/shm/<name>`` (where ``<name>`` is the basename of the configured
    path), the corresponding ``model_dir`` / ``emb_model`` entry is rewritten
    to that location for faster reads.  Falls back to the original path when
    no RAM-disk copy exists.  Set ``PHASE_EVAL_SKIP_SHM=1`` to disable.
    """
    if os.environ.get("PHASE_EVAL_SKIP_SHM", "").strip().lower() in ("1", "true", "yes"):
        return payload

    def _prefer(path: str | None) -> str | None:
        if not path or not isinstance(path, str):
            return path
        path = path.strip()
        if not path or path.startswith("http"):
            return path
        base = os.path.basename(os.path.normpath(path))
        for c in (f"/dev/shm/phase/models/{base}", f"/dev/shm/{base}"):
            if os.path.isdir(c):
                if c != path:
                    logger.info("RAM-disk model path (faster load): %s (was %s)", c, path)
                return c
        return path

    out = dict(payload)
    if out.get("model_dir"):
        out["model_dir"] = _prefer(out["model_dir"])
    if out.get("emb_model"):
        out["emb_model"] = _prefer(out["emb_model"])
    return out


def _load_run_args(run_dir: str, base_config: str | None) -> Namespace:
    run_path = os.path.join(run_dir, "args.yaml")
    with open(run_path, encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not payload.get("eval_ds_info") and base_config:
        with open(base_config, encoding="utf-8") as fh:
            base = yaml.safe_load(fh) or {}
        ev = base.get("eval_ds_info")
        if ev:
            logger.info("Filled eval_ds_info from %s (%d tasks)", base_config, len(ev))
            payload["eval_ds_info"] = ev
    if payload.get("train_ds_names"):
        payload["train_ds_names"] = _resolve_train_ds_names(list(payload["train_ds_names"]))
    payload = _rewrite_paths_for_shm(payload)
    return Namespace(**payload)


def _filter_eval_ds_info(args: Namespace, suffix: str | None) -> Namespace:
    """suffix e.g. 'random_test' keeps only task names ending with _random_test."""
    if not suffix:
        return args
    ev = args.eval_ds_info
    if isinstance(ev, list):
        filtered = [x for x in ev if str(x).endswith(suffix)]
        if not filtered:
            raise ValueError(f"No eval_ds_info entries end with _{suffix}: {ev}")
        args = Namespace(**{**vars(args), "eval_ds_info": filtered})
        logger.info("Filtered eval_ds_info to %d tasks (*_%s)", len(filtered), suffix)
    return args


def _discover_steps(run_dir: str, step_start: int, step_end: int | None, step_stride: int) -> list[int]:
    root = os.path.join(run_dir, "checkpoints")
    steps: list[int] = []
    for d in sorted(glob(os.path.join(root, "it_*"))):
        name = os.path.basename(d)
        if not name.startswith("it_"):
            continue
        try:
            s = int(name.split("it_")[-1])
        except ValueError:
            continue
        if s < step_start:
            continue
        if step_end is not None and s > step_end:
            continue
        if (s - step_start) % step_stride != 0:
            continue
        if os.path.isfile(os.path.join(d, "hypermod.pt")):
            steps.append(s)
    return sorted(steps)


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def _val_info_to_jsonable(val_info: dict[str, dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for split_name, metrics in val_info.items():
        row: dict[str, float] = {}
        for k, v in metrics.items():
            if isinstance(v, numbers.Real) and not isinstance(v, bool):
                row[k] = float(v)
        out[split_name] = row
    return out


def _save_per_step_results(
    run_dir: str,
    step: int,
    val_info: dict[str, dict],
    metrics_csv_fieldnames: list[str],
    *,
    write_metrics_csv: bool = True,
) -> None:
    """Write it_<step>.json; optionally append rows to eval_ckpt_val_loss/metrics.csv."""
    out_dir = os.path.join(run_dir, "eval_ckpt_val_loss")
    os.makedirs(out_dir, exist_ok=True)

    splits_data = _val_info_to_jsonable(val_info)

    avg_row: dict[str, float] = {}
    for metric_key in ("sft_loss", "per_token_acc", "entropy"):
        vals = [m[metric_key] for m in splits_data.values() if metric_key in m]
        if vals:
            avg_row[metric_key] = sum(vals) / len(vals)
    splits_data["__average__"] = avg_row

    payload = {"run_dir": run_dir, "step": step, "splits": splits_data}
    json_path = os.path.join(out_dir, f"it_{step}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote %s  (avg sft_loss=%.4f)", json_path, avg_row.get("sft_loss", float("nan")))

    if not write_metrics_csv:
        return

    csv_path = os.path.join(out_dir, "metrics.csv")
    rows: list[dict] = []
    for split_name, metrics in val_info.items():
        rows.append(
            {
                "step": step,
                "split": split_name,
                "sft_loss": metrics.get("sft_loss"),
                "per_token_acc": metrics.get("per_token_acc"),
                "entropy": metrics.get("entropy"),
            }
        )
    rows.append(
        {
            "step": step,
            "split": "__average__",
            "sft_loss": avg_row.get("sft_loss"),
            "per_token_acc": avg_row.get("per_token_acc"),
            "entropy": avg_row.get("entropy"),
        }
    )
    write_header = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=metrics_csv_fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)
        fh.flush()
    logger.info("Appended %d row(s) to %s", len(rows), csv_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    disable_caching()

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", action="append", required=True, help="Training run directory (repeatable).")
    p.add_argument(
        "--base_config",
        default=str(_REPO_ROOT / "src" / "configs" / "phase_tree_hyper_lora.yaml"),
        help="YAML to take eval_ds_info from when run args.yaml has an empty list.",
    )
    p.add_argument("--step_start", type=int, default=10_000)
    p.add_argument("--step_end", type=int, default=None, help="Inclusive max step (default: no limit).")
    p.add_argument("--step_stride", type=int, default=10_000)
    p.add_argument(
        "--eval_suffix",
        default="",
        help="If set (e.g. random_test), keep only eval_ds_info names ending with this suffix. "
        "Default: empty = all eval tasks (random_test + ood_test, etc.).",
    )
    p.add_argument(
        "--no_per_step_save",
        action="store_true",
        help="Do not write <run_dir>/eval_ckpt_val_loss/it_<step>.json or metrics.csv after each step.",
    )
    p.add_argument(
        "--no_metrics_csv",
        action="store_true",
        help="Only write it_<step>.json (no metrics.csv). Use when many workers share the same run_dir in parallel.",
    )
    p.add_argument("--out_csv", default=None, help="Optional combined CSV (all runs/steps) at end of job.")
    p.add_argument("--device", default="cuda", help="cuda | cuda:0 | cpu")
    args_ns = p.parse_args()

    device = torch.device(args_ns.device if torch.cuda.is_available() else "cpu")
    out_csv = args_ns.out_csv
    metrics_csv_cols = ["step", "split", "sft_loss", "per_token_acc", "entropy"]
    write_metrics_csv = not args_ns.no_metrics_csv

    rows: list[dict] = []

    for run_dir in args_ns.run_dir:
        run_dir = os.path.abspath(run_dir)
        raw_args = _load_run_args(run_dir, args_ns.base_config)
        raw_args = _filter_eval_ds_info(raw_args, (args_ns.eval_suffix or "").strip() or None)

        # Discover checkpoint steps early so we fail fast on misconfigured runs.
        steps = _discover_steps(run_dir, args_ns.step_start, args_ns.step_end, args_ns.step_stride)
        if not steps:
            logger.warning("No checkpoints found for %s with start=%s stride=%s", run_dir, args_ns.step_start, args_ns.step_stride)
            continue
        logger.info("Run %s — will evaluate steps: %s", run_dir, steps)

        peft_type = raw_args.exp_setup.split("_")[-1]
        use_hypernet = getattr(raw_args, "use_hypernet", "hyper" in getattr(raw_args, "exp_setup", ""))
        use_explicit_emb = use_hypernet and not raw_args.use_one_hot_task_emb and raw_args.emb_model

        peft_config = _custom_get_peft_config(
            raw_args.model_dir, peft_type, target_modules=raw_args.target_modules
        )

        # Build one dataloader per eval dataset so we can report per-task loss.
        logger.info("Building per-task val dataloaders (train split skipped, %d tasks)…",
                     len(raw_args.eval_ds_info))

        if use_explicit_emb:
            temp_tokenizer = get_tokenizer(raw_args.model_dir, train=True, peft_config=peft_config)
            emb_model_obj, emb_tokenizer_obj, tdf, pf = get_emb_model_and_fns(
                raw_args.emb_model, device
            )
            emb_model_obj.eval()
            task_emb_size = emb_model_obj.config.hidden_size
        else:
            from hyper_llm_modulator.utils import add_full_stop, get_pooling_fn
            model_tmp, temp_tokenizer = get_model_and_tokenizer(
                raw_args.model_dir, train=False, requires_grad=False,
                peft_config=peft_config,
                model_kwargs={"output_hidden_states": True, "output_attentions": False},
                device=device,
            )
            emb_model_obj = model_tmp
            emb_tokenizer_obj = deepcopy(temp_tokenizer)
            emb_model_obj.eval()
            task_emb_size = emb_model_obj.config.hidden_size
            tdf, pf = add_full_stop, get_pooling_fn("last_token")

        empty_train_meta = get_metadata([], raw_args.use_per_task_emb)
        val_dataloaders: dict = {}
        for ds_name in raw_args.eval_ds_info:
            single_dl_args = Namespace(**{**vars(raw_args), "train_ds_names": [], "eval_ds_info": [ds_name]})
            single_val_meta = get_metadata([ds_name], raw_args.use_per_task_emb)
            dl = create_dataloaders(
                single_dl_args, empty_train_meta, single_val_meta, use_hypernet, device,
                temp_tokenizer, True, emb_model_obj, emb_tokenizer_obj, tdf, pf,
            )
            for v in dl.values():
                if v is not None:
                    val_dataloaders[ds_name] = _DeviceDataLoader(v, device)
                    break
            logger.info("  ✓ %s (%d batches)", ds_name, len(val_dataloaders.get(ds_name, [])))

        if use_explicit_emb:
            del emb_model_obj, emb_tokenizer_obj, temp_tokenizer, tdf, pf
        else:
            del model_tmp, emb_model_obj, emb_tokenizer_obj, temp_tokenizer, tdf, pf
        gc.collect()
        torch.cuda.empty_cache()

        if not val_dataloaders:
            raise RuntimeError(f"No validation dataloaders for {run_dir}; check eval_ds_info / train_ds_names.")
        logger.info("Val dataloaders ready (%d tasks): %s", len(val_dataloaders), list(val_dataloaders.keys()))

        # Load the base LLM and hypermod architecture once per run; only the
        # hypermod state_dict is reloaded for each evaluated checkpoint step.
        logger.info("Loading LLM for validation…")
        model, tokenizer = get_model_and_tokenizer(
            raw_args.model_dir, train=False, requires_grad=False,
            peft_config=peft_config,
            model_kwargs={"output_hidden_states": True, "output_attentions": False},
            device=device,
        )
        layer_indices = torch.tensor(range(len(get_layers(model))), dtype=torch.long, device=device)

        hypermod = create_hypermod(
            raw_args, peft_type, device, model, layer_indices, task_emb_size, from_scratch=False
        )

        # Read LoRA dropout directly from adapter_config.json so we do not
        # need to re-instantiate the PEFT config just to access this field.
        with open(os.path.join(run_dir, "adapter_config.json"), encoding="utf-8") as fh:
            inp_dropout = json.load(fh).get("lora_dropout", 0.0)

        # Reset metrics CSV for this fresh evaluation run.
        if not args_ns.no_per_step_save and write_metrics_csv:
            out_dir = os.path.join(run_dir, "eval_ckpt_val_loss")
            os.makedirs(out_dir, exist_ok=True)
            metrics_csv_path = os.path.join(out_dir, "metrics.csv")
            if os.path.isfile(metrics_csv_path):
                os.remove(metrics_csv_path)
                logger.info("Reset %s for this evaluation run", metrics_csv_path)

        # Evaluate each checkpoint step; only the hypermod weights are reloaded.
        for step in steps:
            logger.info("=== step %d ===", step)
            ckpt_path = os.path.join(run_dir, "checkpoints", f"it_{step}", "hypermod.pt")
            if not os.path.isfile(ckpt_path):
                logger.warning("Missing checkpoint: %s — skipping", ckpt_path)
                continue

            state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
            info = hypermod.load_state_dict(state_dict, strict=False)
            logger.info("Loaded hypermod weights for step %d: %s", step, info)
            hypermod.eval()
            del state_dict

            _get_loss_batch = partial(
                get_loss_batch,
                model=model,
                target_modules=raw_args.target_modules,
                inp_dropout=inp_dropout,
                layer_indices=layer_indices,
                use_hypernet=use_hypernet,
                hypermod=hypermod,
                equally_weight_sample=raw_args.equally_weight_sample,
                l2_reg_generated_w=getattr(raw_args, "l2_reg_generated_w", 0.0),
                label_smoothing=getattr(raw_args, "label_smoothing", 0.0),
            )

            val_info = validate(model, hypermod, val_dataloaders, _get_loss_batch, step)

            if not args_ns.no_per_step_save:
                _save_per_step_results(
                    run_dir, step, val_info, metrics_csv_cols, write_metrics_csv=write_metrics_csv
                )
            for split_name, metrics in val_info.items():
                row = {
                    "run_dir": run_dir,
                    "step": step,
                    "split": split_name,
                    "sft_loss": metrics.get("sft_loss"),
                    "per_token_acc": metrics.get("per_token_acc"),
                    "entropy": metrics.get("entropy"),
                }
                rows.append(row)
                logger.info("%s step=%d sft_loss=%s", split_name, step, metrics.get("sft_loss"))

        # Release per-run resources before processing the next run directory.
        del model, tokenizer, hypermod, layer_indices
        gc.collect()
        torch.cuda.empty_cache()

    if rows and out_csv:
        fieldnames = ["run_dir", "step", "split", "sft_loss", "per_token_acc", "entropy"]
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        logger.info("Wrote combined %s (%d rows)", out_csv, len(rows))
    elif rows:
        logger.info("Done (%d rows in memory); per-step files under each run_dir/eval_ckpt_val_loss/", len(rows))
    else:
        logger.error("No evaluation rows produced (no matching checkpoints or empty runs).")


if __name__ == "__main__":
    main()
