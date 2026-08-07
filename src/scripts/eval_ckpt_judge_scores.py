#!/usr/bin/env python3
"""Evaluate hypermod checkpoints via full inference + LLM-as-Judge scoring.

Samples a fixed subset (default 5%) of each dataset's m6_phase_tree test data,
then for each checkpoint step:
  1. Loads the checkpoint's hypermod weights (architecture + vLLM stay resident)
  2. Encodes profile embeddings through the (updated) task_encoder
  3. Generates per-profile LoRA adapters
  4. Runs vLLM inference
  5. Scores predictions with LLM-as-Judge + embedding similarity

Efficiency: the embedding model is loaded once (profile embeddings are reused
across checkpoints), and vLLM is loaded once (only LoRA adapters are swapped).
Only the hypermod state_dict (~1 GB) is reloaded per checkpoint step.

Results are saved under ``<run_dir>/eval_ckpt_judge_scores/``.

Usage::

    cd PHASE-Tree
    PYTHONPATH=src:$PYTHONPATH python src/scripts/eval_ckpt_judge_scores.py \\
        --run_dir phase_tree_models/sft/hyper_lora/<your_run_id>

    # Custom steps and sample fraction
    PYTHONPATH=src:$PYTHONPATH python src/scripts/eval_ckpt_judge_scores.py \\
        --run_dir phase_tree_models/sft/hyper_lora/<your_run_id> \\
        --steps 5000 10000 15000 \\
        --sample_frac 0.10 --seed 42

    # Skip prediction (only judge already-generated predictions)
    PYTHONPATH=src:$PYTHONPATH python src/scripts/eval_ckpt_judge_scores.py \\
        --run_dir phase_tree_models/sft/hyper_lora/<your_run_id> \\
        --skip_predict

    # Skip judge (only generate predictions)
    PYTHONPATH=src:$PYTHONPATH python src/scripts/eval_ckpt_judge_scores.py \\
        --run_dir phase_tree_models/sft/hyper_lora/<your_run_id> \\
        --skip_judge
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

ALL_DATASETS = [
    "RAIDEN", "CharacterEval", "HPD", "SimsConv",
    "ChatHaruhi", "Friends", "StarTrek_TNG", "TheOffice",
]

BASELINE_PROMPT = """\
Below is a multi-turn dialogue. Predict the single line that {character} would most likely say next.
Keep the reply short and natural, matching the tone and length of the other lines. Output only that one line, no explanation.

Dialogue context:
{context}

{character}:"""


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def sample_subset(
    data_dir: str,
    datasets: list[str],
    split: str,
    sample_frac: float,
    seed: int,
    out_dir: str,
) -> dict[str, dict]:
    """Sample ``sample_frac`` of each dataset's m6_phase_tree split.

    Returns {dataset: {"path": ..., "n_total": ..., "n_sampled": ...}}.
    """
    os.makedirs(out_dir, exist_ok=True)
    info: dict[str, dict] = {}
    for ds in datasets:
        src = os.path.join(data_dir, ds, "m6_phase_tree", f"{split}.json")
        if not os.path.isfile(src):
            print(f"  [SKIP] {src} not found", flush=True)
            continue
        with open(src, encoding="utf-8") as f:
            all_data = json.load(f)
        rng = random.Random(seed)
        n = max(1, round(len(all_data) * sample_frac))
        subset = rng.sample(all_data, min(n, len(all_data)))
        out_path = os.path.join(out_dir, f"{ds}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=1)
        info[ds] = {"path": out_path, "n_total": len(all_data), "n_sampled": len(subset)}
        print(f"  {ds}: {len(subset)}/{len(all_data)} sampled → {out_path}", flush=True)
    return info


def build_prompt(sample: dict) -> str:
    return BASELINE_PROMPT.format(
        character=sample["role"],
        context=sample["input"],
    )


def load_jsonl(path: str) -> list[dict]:
    items = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Model resolution: optional RAM-disk override for faster weight loads
# ---------------------------------------------------------------------------

def _prefer_shm(path: str) -> str:
    """Return a tmpfs (``/dev/shm``) copy of *path* if one exists, else *path*.

    This is a transparent speed-up: when the same model directory has been
    pre-copied to ``/dev/shm/phase/models/<name>`` or ``/dev/shm/<name>`` it
    will be used in place of the disk-backed path.  Falls back silently when
    no RAM-disk copy is present.
    """
    if not path:
        return path
    base = os.path.basename(os.path.normpath(path))
    for cand in (f"/dev/shm/phase/models/{base}", f"/dev/shm/{base}"):
        if os.path.isdir(cand):
            return cand
    return path


def _get_ramdisk_lora_dir(fallback: str) -> str:
    for prefix in ("/dev/shm/lora_ckpt_eval", "/tmp/lora_ckpt_eval"):
        try:
            d = tempfile.mkdtemp(prefix="lora_", dir=os.path.dirname(prefix))
            return d
        except OSError:
            pass
    os.makedirs(fallback, exist_ok=True)
    return fallback


# ---------------------------------------------------------------------------
# Lightweight hypermod loader (adapted from predict_phase_tree.py)
# ---------------------------------------------------------------------------

def _find_checkpoint_base_dir(checkpoint_path: str) -> str:
    search = os.path.dirname(os.path.abspath(checkpoint_path))
    for _ in range(5):
        if os.path.isfile(os.path.join(search, "args.yaml")):
            return search
        parent = os.path.dirname(search)
        if parent == search:
            break
        search = parent
    return os.path.dirname(os.path.abspath(checkpoint_path))


def load_hypermod_architecture(
    run_dir: str,
    model_dir_override: str | None,
    emb_model_override: str | None,
    device: str,
) -> tuple:
    """Build HyperModulator + load embedding model WITHOUT loading the base LLM.

    Returns ``(ckpt_args, hypermod, peft_cfg, model_dir, emb_model,
    emb_tokenizer, pooling_fn, n_layers)``.
    """
    import yaml
    from peft import LoraConfig
    from transformers import AutoConfig

    from hyper_llm_modulator.hyper_modulator import (
        HyperModulator,
        get_lora_module_names,
        zero_lora_param_dict,
    )
    from hyper_llm_modulator.utils.model_loading import get_emb_model_and_fns

    with open(os.path.join(run_dir, "args.yaml"), encoding="utf-8") as f:
        raw_args = yaml.safe_load(f)

    _DEFAULTS = {
        "pred_z_score": True, "shared_AB_head": False,
        "autoreg_gen": False, "learnable_pos_emb": False,
        "use_conv_fusion": False, "conv_fusion_type": "1d",
        "conv_fusion_kernel_size": 3, "conv_fusion_num_layers": 2,
        "conv_fusion_channels": 64, "conv_fusion_dropout": 0.1,
        "use_attention_fusion": False, "attention_fusion_type": "self",
        "attention_num_heads": 8, "attention_dropout": 0.1,
        "attention_num_layers": 2, "factorized": False,
        "delta_w_scaling": 10000,
    }
    for k, v in _DEFAULTS.items():
        raw_args.setdefault(k, v)

    emb_name = emb_model_override or _prefer_shm(raw_args.get("emb_model", ""))
    model_dir = model_dir_override or _prefer_shm(raw_args["model_dir"])

    ckpt_args = argparse.Namespace(**raw_args)
    ckpt_args.model_dir = model_dir

    with open(os.path.join(run_dir, "adapter_config.json"), encoding="utf-8") as f:
        adapter_dict = json.load(f)
    peft_cfg = LoraConfig(**{
        k: v for k, v in adapter_dict.items()
        if k in LoraConfig.__dataclass_fields__
    })

    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    n_layers = model_config.num_hidden_layers

    hidden = model_config.hidden_size
    n_heads = model_config.num_attention_heads
    n_kv_heads = getattr(model_config, "num_key_value_heads", n_heads)
    head_dim = hidden // n_heads
    dim_map = {
        "q_proj": (hidden, n_heads * head_dim),
        "k_proj": (hidden, n_kv_heads * head_dim),
        "v_proj": (hidden, n_kv_heads * head_dim),
        "o_proj": (n_heads * head_dim, hidden),
        "gate_proj": (hidden, model_config.intermediate_size),
        "up_proj": (hidden, model_config.intermediate_size),
        "down_proj": (model_config.intermediate_size, hidden),
    }
    in_features, out_features = {}, {}
    for mod in peft_cfg.target_modules:
        in_features[mod] = dim_map[mod][0]
        out_features[mod] = dim_map[mod][1]

    layer_indices = list(range(n_layers))
    module_names = {m: [[] for _ in layer_indices] for m in peft_cfg.target_modules}
    for li in layer_indices:
        for mod in peft_cfg.target_modules:
            prefix = f"base_model.model.model.layers.{li}.self_attn.{mod}"
            module_names[mod][li] = [
                f"{prefix}.lora_A.default.weight",
                f"{prefix}.lora_B.default.weight",
            ]

    te_path = os.path.join(run_dir, "checkpoints", "it_5000", "hypermod.pt")
    if not os.path.isfile(te_path):
        te_path = os.path.join(run_dir, "hypermod.pt")
    state_dict = torch.load(te_path, map_location=device, weights_only=False)
    te_key = "task_encoder.mlp.0.weight"
    task_emb_size = state_dict[te_key].shape[1] if te_key in state_dict else None

    class _MockModel:
        pass
    mock = _MockModel()
    mock.config = model_config
    mock.peft_config = {"default": peft_cfg}
    mock.device = torch.device(device)

    import hyper_llm_modulator.hyper_modulator as _hmod
    _orig_giof = _hmod.get_in_out_features
    _orig_gipw = _hmod.get_init_peft_weights

    def _fast_in_out(model, peft_config=None):
        return in_features, out_features

    r = peft_cfg.r

    def _fast_init_peft_weights(model, peft_config=None):
        result = {}
        for mod in peft_cfg.target_modules:
            a = torch.nn.Linear(in_features[mod], r, bias=False, device=device)
            b = torch.nn.Linear(r, out_features[mod], bias=False, device=device)
            torch.nn.init.zeros_(b.weight)
            result[mod] = {"lora_A": a, "lora_B": b}
        return result

    _hmod.get_in_out_features = _fast_in_out
    _hmod.get_init_peft_weights = _fast_init_peft_weights

    try:
        hypermod = HyperModulator(
            mock,
            training_task=ckpt_args.training_task,
            pred_z_score=ckpt_args.pred_z_score,
            output_space=peft_cfg.peft_type.lower(),
            module_names=module_names,
            match_lora_init=False,
            task_emb_size=task_emb_size,
            shared_AB_head=ckpt_args.shared_AB_head,
            autoreg_gen=ckpt_args.autoreg_gen,
            learnable_pos_emb=ckpt_args.learnable_pos_emb,
            zero_init_head=False,
            latent_size=ckpt_args.hypernet_latent_size,
            head_in_size=ckpt_args.head_in_size,
            head_use_bias=True,
            factorized=getattr(ckpt_args, "factorized", False),
            delta_w_scaling=getattr(ckpt_args, "delta_w_scaling", 10000),
            use_conv_fusion=ckpt_args.use_conv_fusion,
            conv_fusion_type=ckpt_args.conv_fusion_type,
            conv_fusion_kernel_size=ckpt_args.conv_fusion_kernel_size,
            conv_fusion_num_layers=ckpt_args.conv_fusion_num_layers,
            conv_fusion_channels=ckpt_args.conv_fusion_channels,
            conv_fusion_dropout=ckpt_args.conv_fusion_dropout,
            use_attention_fusion=getattr(ckpt_args, "use_attention_fusion", False),
            attention_fusion_type=getattr(ckpt_args, "attention_fusion_type", "self"),
            attention_num_heads=getattr(ckpt_args, "attention_num_heads", 8),
            attention_dropout=getattr(ckpt_args, "attention_dropout", 0.1),
            attention_num_layers=getattr(ckpt_args, "attention_num_layers", 2),
        )
    finally:
        _hmod.get_in_out_features = _orig_giof
        _hmod.get_init_peft_weights = _orig_gipw

    info = hypermod.load_state_dict(state_dict, strict=False)
    print(f"  Loaded initial hypermod state dict: {info}", flush=True)
    hypermod.eval().to(device)
    del state_dict

    emb_model, emb_tokenizer, _, pooling_fn = get_emb_model_and_fns(emb_name, device)

    return (ckpt_args, hypermod, peft_cfg, model_dir,
            emb_model, emb_tokenizer, pooling_fn, n_layers)


# ---------------------------------------------------------------------------
# Collect scores from judge output
# ---------------------------------------------------------------------------

def collect_scores(result_dir: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    judge_path = os.path.join(result_dir, "judge_scores.jsonl")
    if os.path.isfile(judge_path):
        items = load_jsonl(judge_path)
        if items:
            scores["character"] = sum(r["character_score"] for r in items) / len(items)
            scores["semantic"] = sum(r["semantic_score"] for r in items) / len(items)
            scores["n_judge"] = len(items)
    embed_path = os.path.join(result_dir, "embedding_scores.jsonl")
    if os.path.isfile(embed_path):
        items = load_jsonl(embed_path)
        if items:
            scores["embedding"] = sum(r["embedding_similarity"] for r in items) / len(items)
            scores["n_embed"] = len(items)
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run_dir", required=True, help="Training run directory.")
    parser.add_argument(
        "--steps", type=int, nargs="+",
        default=[5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000],
        help="Checkpoint steps to evaluate (default: 5000..40000 by 5000).",
    )
    parser.add_argument("--sample_frac", type=float, default=0.05,
                        help="Fraction of each dataset to sample (default: 0.05).")
    parser.add_argument("--split", default="random_test",
                        help="Test split to sample from (default: random_test).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to evaluate (default: all 8).")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID.")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--emb_batch_size", type=int, default=64)
    parser.add_argument("--skip_predict", action="store_true",
                        help="Skip prediction, only run judge on existing predictions.")
    parser.add_argument("--skip_judge", action="store_true",
                        help="Skip judge, only generate predictions.")
    parser.add_argument("--judge_workers", type=int, default=10)
    parser.add_argument("--model_override", default=None)
    parser.add_argument("--emb_model_override", default=None)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    eval_dir = os.path.join(run_dir, "eval_ckpt_judge_scores")
    subset_dir = os.path.join(eval_dir, "subset")
    data_dir = os.path.join(_REPO_ROOT, "LongEvoRoleBench", "processed")
    datasets = args.datasets or ALL_DATASETS

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Validate checkpoint steps exist
    valid_steps = []
    for step in sorted(args.steps):
        ckpt_path = os.path.join(run_dir, "checkpoints", f"it_{step}", "hypermod.pt")
        if os.path.isfile(ckpt_path):
            valid_steps.append(step)
        else:
            print(f"  [WARN] Checkpoint not found: {ckpt_path}", flush=True)
    if not valid_steps:
        print("ERROR: No valid checkpoints found.", flush=True)
        sys.exit(1)

    # ==================================================================
    # STEP 0: Sample 5% subset (reuse if subset/ already exists)
    # ==================================================================
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Checkpoint Judge Evaluation", flush=True)
    print(f"  Run dir    : {run_dir}", flush=True)
    print(f"  Steps      : {valid_steps}", flush=True)
    print(f"  Datasets   : {datasets}", flush=True)
    print(f"  Sample frac: {args.sample_frac}", flush=True)
    print(f"  Split      : {args.split}", flush=True)
    print(f"  Seed       : {args.seed}", flush=True)
    print(f"{'=' * 60}", flush=True)

    meta_path = os.path.join(subset_dir, "_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            old_meta = json.load(f)
        if (old_meta.get("seed") == args.seed
                and old_meta.get("sample_frac") == args.sample_frac
                and old_meta.get("split") == args.split):
            print("\n  Reusing existing subset (same seed/frac/split).", flush=True)
            ds_info = {}
            for ds in datasets:
                p = os.path.join(subset_dir, f"{ds}.json")
                if os.path.isfile(p):
                    with open(p, encoding="utf-8") as f:
                        n = len(json.load(f))
                    ds_info[ds] = {"path": p, "n_sampled": n}
                    print(f"    {ds}: {n} samples", flush=True)
        else:
            print("\n  Subset params changed, re-sampling ...", flush=True)
            ds_info = sample_subset(data_dir, datasets, args.split,
                                    args.sample_frac, args.seed, subset_dir)
    else:
        print(f"\n  Sampling {args.sample_frac*100:.0f}% of each dataset ...", flush=True)
        ds_info = sample_subset(data_dir, datasets, args.split,
                                args.sample_frac, args.seed, subset_dir)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "sample_frac": args.sample_frac,
                    "split": args.split, "datasets": list(ds_info.keys()),
                    "timestamp": datetime.now().isoformat()}, f, indent=2)

    active_datasets = [ds for ds in datasets if ds in ds_info]
    if not active_datasets:
        print("ERROR: No datasets available after sampling.", flush=True)
        sys.exit(1)

    # Load all sampled data
    all_samples: dict[str, list[dict]] = {}
    for ds in active_datasets:
        with open(ds_info[ds]["path"], encoding="utf-8") as f:
            all_samples[ds] = json.load(f)

    total_samples = sum(len(v) for v in all_samples.values())
    all_profile_texts = list(set(
        s.get("profile_text", "")
        for samples in all_samples.values()
        for s in samples
    ))
    print(f"\n  Total: {total_samples} samples, "
          f"{len(all_profile_texts)} unique profiles across "
          f"{len(active_datasets)} datasets", flush=True)

    if args.skip_predict:
        print("\n  [skip_predict] Jumping to judge step ...", flush=True)
    else:
        # ==============================================================
        # STEP 1: Load emb model → embed all profiles → free emb model
        # ==============================================================
        print(f"\n{'─' * 60}", flush=True)
        print("  STEP 1: Embed all profiles (emb model loaded ONCE)", flush=True)
        print(f"{'─' * 60}", flush=True)

        from hyper_llm_modulator.utils.utils import embed_texts

        (
            ckpt_args, hypermod, peft_cfg, model_dir,
            emb_model, emb_tokenizer, pooling_fn, n_layers,
        ) = load_hypermod_architecture(
            run_dir, args.model_override, args.emb_model_override, device,
        )

        lora_rank = peft_cfg.r
        identity_fn = lambda x: x  # noqa: E731
        layer_indices_t = torch.tensor(range(n_layers), dtype=torch.long, device=device)

        non_empty = [pt for pt in all_profile_texts if pt]
        non_empty_pos = {pt: i for i, pt in enumerate(non_empty)}
        hidden_size = emb_model.config.hidden_size

        t_emb = time.perf_counter()
        with torch.no_grad():
            raw_embs = None
            if non_empty:
                raw_embs = embed_texts(
                    non_empty, emb_model, emb_tokenizer,
                    identity_fn, pooling_fn, device,
                    batch_size=args.emb_batch_size,
                )
                if raw_embs.device != torch.device(device):
                    raw_embs = raw_embs.to(device)
        emb_elapsed = time.perf_counter() - t_emb
        print(f"  Embedded {len(non_empty)} profiles in {emb_elapsed:.1f}s", flush=True)

        del emb_model, emb_tokenizer, pooling_fn, identity_fn
        gc.collect()
        torch.cuda.empty_cache()
        print("  Freed emb_model.", flush=True)

        # ==============================================================
        # STEP 2: Load vLLM (base model loaded ONCE)
        # ==============================================================
        print(f"\n{'─' * 60}", flush=True)
        print("  STEP 2: Load vLLM (LLM loaded ONCE)", flush=True)
        print(f"{'─' * 60}", flush=True)

        from hyper_llm_modulator.hyper_modulator import save_lora
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        import vllm.transformers_utils.tokenizer as _vllm_tok
        import vllm.transformers_utils.tokenizer_group.tokenizer_group as _vllm_tg
        _orig_fn = _vllm_tok.get_lora_tokenizer
        def _safe_get_lora_tokenizer(lora_request, *a, **kw):
            try:
                return _orig_fn(lora_request, *a, **kw)
            except Exception:
                return None
        _vllm_tok.get_lora_tokenizer = _safe_get_lora_tokenizer
        _vllm_tok.get_lora_tokenizer_async = _vllm_tok.make_async(_safe_get_lora_tokenizer)
        _vllm_tg.get_lora_tokenizer = _safe_get_lora_tokenizer
        _vllm_tg.get_lora_tokenizer_async = _vllm_tok.make_async(_safe_get_lora_tokenizer)

        inference_model_dir = _prefer_shm(model_dir)
        print(f"  Model: {inference_model_dir}", flush=True)
        t_vllm = time.perf_counter()

        llm = LLM(
            model=inference_model_dir,
            tensor_parallel_size=1,
            max_model_len=4096,
            enable_lora=True,
            max_lora_rank=lora_rank,
            max_loras=16,
            trust_remote_code=True,
            dtype="bfloat16",
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        tokenizer = llm.get_tokenizer()
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        print(f"  vLLM loaded in {time.perf_counter() - t_vllm:.1f}s", flush=True)

        # ==============================================================
        # STEP 3: For each checkpoint → encode → LoRA → infer
        # ==============================================================
        lora_id_counter = 1

        for step in valid_steps:
            ckpt_path = os.path.join(
                run_dir, "checkpoints", f"it_{step}", "hypermod.pt")
            step_dir = os.path.join(eval_dir, f"it_{step}")

            # Check if this step is already complete
            all_done = all(
                os.path.isfile(os.path.join(step_dir, ds, "predictions.jsonl"))
                for ds in active_datasets
            )
            if all_done:
                print(f"\n  === step {step} === (predictions exist, skipping)",
                      flush=True)
                continue

            print(f"\n{'─' * 60}", flush=True)
            print(f"  === step {step} ===", flush=True)
            print(f"{'─' * 60}", flush=True)

            # Load hypermod weights
            state_dict = torch.load(
                ckpt_path, map_location=device, weights_only=False)
            info = hypermod.load_state_dict(state_dict, strict=False)
            print(f"  Loaded hypermod weights: {info}", flush=True)
            hypermod.eval()
            del state_dict

            # Encode profiles through task_encoder
            t_enc = time.perf_counter()
            with torch.no_grad():
                encoded_lookup: dict[str, torch.Tensor] = {}
                for pt in all_profile_texts:
                    if pt and raw_embs is not None:
                        task_emb = raw_embs[non_empty_pos[pt]:non_empty_pos[pt] + 1]
                    else:
                        task_emb = torch.zeros(1, hidden_size, device=device)
                    enc_out = hypermod.task_encoder(task_emb)
                    encoded_lookup[pt] = enc_out["encoded_task_emb"].detach()
            print(f"  Encoded {len(all_profile_texts)} profiles in "
                  f"{time.perf_counter() - t_enc:.1f}s", flush=True)

            # Per-dataset LoRA generation + inference
            for ds in active_datasets:
                ds_dir = os.path.join(step_dir, ds)
                pred_path = os.path.join(ds_dir, "predictions.jsonl")
                os.makedirs(ds_dir, exist_ok=True)

                if os.path.isfile(pred_path):
                    existing = load_jsonl(pred_path)
                    done_ids = {r["question_id"] for r in existing}
                else:
                    done_ids = set()

                samples = all_samples[ds]
                remaining = [s for s in samples if s["question_id"] not in done_ids]
                if not remaining:
                    print(f"  {ds}: already complete ({len(samples)} predictions)",
                          flush=True)
                    continue

                profiles = list(set(
                    s.get("profile_text", "") for s in remaining))
                print(f"  {ds}: {len(remaining)} samples, "
                      f"{len(profiles)} profiles ...", flush=True)

                lora_base_dir = _get_ramdisk_lora_dir(
                    os.path.join(ds_dir, "_temp_loras"))

                profile_to_lora: dict[str, str] = {}
                with torch.no_grad():
                    for pt in profiles:
                        ph = hashlib.sha256(pt.encode()).hexdigest()[:16]
                        lora_dir = os.path.join(lora_base_dir, f"lora_{ph}")
                        if (os.path.isdir(lora_dir) and os.path.isfile(
                                os.path.join(lora_dir, "adapter_model.safetensors"))):
                            profile_to_lora[pt] = lora_dir
                            continue
                        lora_sd = hypermod.gen_lora(
                            layer_indices_t, encoded_lookup[pt])
                        lora_sd = {
                            k.replace(".default.weight", ".weight"): v
                            for k, v in lora_sd.items()
                        }
                        save_lora(lora_sd, hypermod.peft_config, lora_dir)
                        profile_to_lora[pt] = lora_dir

                # Group by LoRA and run vLLM inference
                lora_groups: dict[str, list[tuple[int, str]]] = {}
                for i, s in enumerate(remaining):
                    pt = s.get("profile_text", "")
                    lora_d = profile_to_lora[pt]
                    raw = build_prompt(s)
                    messages = [{"role": "user", "content": raw}]
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                    lora_groups.setdefault(lora_d, []).append((i, text))

                lora_requests: dict[str, LoRARequest] = {}
                for lora_d in lora_groups:
                    lora_requests[lora_d] = LoRARequest(
                        f"lora_{lora_id_counter}", lora_id_counter, lora_d)
                    lora_id_counter += 1

                results: dict[int, str] = {}
                t0 = time.perf_counter()
                for lora_d, group in lora_groups.items():
                    indices, prompts = zip(*group)
                    lr = lora_requests[lora_d]
                    outputs = llm.generate(
                        list(prompts), sampling_params, lora_request=lr)
                    for idx, out in zip(indices, outputs):
                        results[idx] = out.outputs[0].text.strip()
                infer_elapsed = time.perf_counter() - t0

                with open(pred_path, "a", encoding="utf-8") as f:
                    for i, s in enumerate(remaining):
                        record = {
                            "question_id": s["question_id"],
                            "role": s["role"],
                            "prediction": results[i],
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                print(f"    {len(remaining)} predictions in {infer_elapsed:.1f}s",
                      flush=True)

                # Cleanup temp LoRAs
                if os.path.exists(lora_base_dir):
                    shutil.rmtree(lora_base_dir, ignore_errors=True)
                afs_tmp = os.path.join(ds_dir, "_temp_loras")
                if os.path.exists(afs_tmp):
                    shutil.rmtree(afs_tmp, ignore_errors=True)

        # Cleanup GPU
        del llm, hypermod, layer_indices_t, encoded_lookup, raw_embs
        gc.collect()
        torch.cuda.empty_cache()
        print("\n  Freed vLLM + hypermod.", flush=True)

    # ==================================================================
    # STEP 4: Judge scoring (API-based, no GPU needed)
    # ==================================================================
    if args.skip_judge:
        print("\n  [skip_judge] Skipping judge step.", flush=True)
    else:
        print(f"\n{'═' * 60}", flush=True)
        print("  STEP 4: LLM-as-Judge scoring", flush=True)
        print(f"{'═' * 60}", flush=True)

        judge_script = str(_REPO_ROOT / "evaluation" / "judge.py")
        python_bin = sys.executable

        for step in valid_steps:
            print(f"\n  --- step {step} ---", flush=True)
            judge_procs = []
            for ds in active_datasets:
                ds_dir = os.path.join(eval_dir, f"it_{step}", ds)
                pred_path = os.path.join(ds_dir, "predictions.jsonl")
                if not os.path.isfile(pred_path):
                    print(f"    {ds}: no predictions, skipping", flush=True)
                    continue
                judge_out = os.path.join(ds_dir, "judge_scores.jsonl")
                if os.path.isfile(judge_out) and len(load_jsonl(judge_out)) > 0:
                    print(f"    {ds}: judge scores exist, skipping", flush=True)
                    continue

                persona_data = ds_info[ds]["path"]
                cmd = [
                    python_bin, judge_script,
                    "--predictions_dir", ds_dir,
                    "--persona_data", persona_data,
                    "--num_workers", str(args.judge_workers),
                ]
                env = os.environ.copy()
                env["PYTHONPATH"] = (
                    f"{_REPO_ROOT}/src:{_REPO_ROOT}:{env.get('PYTHONPATH', '')}")
                print(f"    {ds}: launching judge ...", flush=True)
                proc = subprocess.Popen(cmd, env=env)
                judge_procs.append((ds, proc))

            for ds, proc in judge_procs:
                rc = proc.wait()
                status = "OK" if rc == 0 else f"FAIL (exit {rc})"
                print(f"    {ds}: judge {status}", flush=True)

    # ==================================================================
    # STEP 5: Aggregate scores → summary
    # ==================================================================
    print(f"\n{'═' * 60}", flush=True)
    print("  STEP 5: Aggregating scores", flush=True)
    print(f"{'═' * 60}", flush=True)

    summary: dict[str, dict] = {}
    csv_rows: list[dict] = []

    for step in valid_steps:
        step_key = f"it_{step}"
        step_scores: dict[str, dict] = {}
        all_char, all_sem, all_emb = [], [], []

        for ds in active_datasets:
            ds_dir = os.path.join(eval_dir, step_key, ds)
            scores = collect_scores(ds_dir)
            step_scores[ds] = scores
            if "character" in scores:
                all_char.extend(
                    [r["character_score"] for r in load_jsonl(
                        os.path.join(ds_dir, "judge_scores.jsonl"))])
            if "semantic" in scores:
                all_sem.extend(
                    [r["semantic_score"] for r in load_jsonl(
                        os.path.join(ds_dir, "judge_scores.jsonl"))])
            if "embedding" in scores:
                all_emb.extend(
                    [r["embedding_similarity"] for r in load_jsonl(
                        os.path.join(ds_dir, "embedding_scores.jsonl"))])

            csv_rows.append({
                "step": step, "dataset": ds,
                "character": scores.get("character"),
                "semantic": scores.get("semantic"),
                "embedding": scores.get("embedding"),
                "n": scores.get("n_judge"),
            })

        avg_scores = {}
        if all_char:
            avg_scores["character"] = sum(all_char) / len(all_char)
        if all_sem:
            avg_scores["semantic"] = sum(all_sem) / len(all_sem)
        if all_emb:
            avg_scores["embedding"] = sum(all_emb) / len(all_emb)
        avg_scores["n_total"] = len(all_char)

        step_scores["__average__"] = avg_scores
        summary[step_key] = step_scores

        csv_rows.append({
            "step": step, "dataset": "__average__",
            "character": avg_scores.get("character"),
            "semantic": avg_scores.get("semantic"),
            "embedding": avg_scores.get("embedding"),
            "n": avg_scores.get("n_total"),
        })

        # Save per-step scores
        step_scores_path = os.path.join(eval_dir, step_key, "scores.json")
        with open(step_scores_path, "w", encoding="utf-8") as f:
            json.dump(step_scores, f, indent=2)

    # Write summary
    summary_path = os.path.join(eval_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(eval_dir, "summary.csv")
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "step", "dataset", "character", "semantic", "embedding", "n"])
            w.writeheader()
            w.writerows(csv_rows)

    # Print summary table
    print(f"\n{'═' * 70}", flush=True)
    print("  SUMMARY: Average scores per checkpoint", flush=True)
    print(f"{'═' * 70}", flush=True)
    print(f"  {'Step':>8}  {'Character':>10}  {'Semantic':>10}  "
          f"{'Embedding':>10}  {'N':>6}", flush=True)
    print(f"  {'─' * 8}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 6}",
          flush=True)
    for step in valid_steps:
        avg = summary.get(f"it_{step}", {}).get("__average__", {})
        c = f"{avg['character']:.3f}" if "character" in avg else "   —"
        s = f"{avg['semantic']:.3f}" if "semantic" in avg else "   —"
        e = f"{avg['embedding']:.4f}" if "embedding" in avg else "   —"
        n = str(avg.get("n_total", "—"))
        print(f"  {step:>8}  {c:>10}  {s:>10}  {e:>10}  {n:>6}", flush=True)
    print(f"{'═' * 70}", flush=True)

    # Per-dataset breakdown
    print(f"\n  Per-dataset Character scores:", flush=True)
    header = f"  {'Step':>8}"
    for ds in active_datasets:
        header += f"  {ds[:8]:>8}"
    print(header, flush=True)
    for step in valid_steps:
        row = f"  {step:>8}"
        for ds in active_datasets:
            sc = summary.get(f"it_{step}", {}).get(ds, {})
            val = f"{sc['character']:.3f}" if "character" in sc else "   —"
            row += f"  {val:>8}"
        print(row, flush=True)

    print(f"\n  Results: {eval_dir}/", flush=True)
    print(f"  Summary: {summary_path}", flush=True)
    print(f"  CSV:     {csv_path}", flush=True)


if __name__ == "__main__":
    main()
