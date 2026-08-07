"""Dialogue-continuation prediction via P2P hypernetwork-generated LoRA.

For each sample the script:

1. Looks up the ``profile_text`` field.
2. Embeds the profile text using the hypernetwork's embedding model.
3. Generates (or reuses) a LoRA adapter from the hypernetwork.
4. Runs vLLM inference with the per-sample LoRA applied.

The prompt does **not** contain the profile text — personalization is
entirely encoded in the LoRA adapter.  The prompt uses the same
*baseline* template as ``m1_context_only``.

Usage::

    # Single method + split
    python evaluation/predict_hypernet.py \\
        --data LongEvoRoleBench/processed/RAIDEN/m6_phase_tree/random_test.json \\
        --output_dir results/RAIDEN/hypernet_p2p/main/m6_phase_tree/random_test \\
        --checkpoint phase_tree_models/p2p_pretrained/hypermod.pt

    # With per-character LoRA saving
    python evaluation/predict_hypernet.py \\
        --data LongEvoRoleBench/processed/RAIDEN/m2_raw_profile/random_test.json \\
        --output_dir results/RAIDEN/hypernet_p2p/main/m2_raw_profile/random_test \\
        --checkpoint phase_tree_models/p2p_pretrained/hypermod.pt \\
        --save_loras_dir results/RAIDEN/hypernet_p2p/generated_loras/m2_raw_profile

    # Multi-process parallelism: run different methods on different GPUs
    python evaluation/predict_hypernet.py --gpu 0 --data .../m2_raw_profile/random_test.json ... &
    python evaluation/predict_hypernet.py --gpu 1 --data .../m6_phase_tree/random_test.json ... &
    wait

    # Chunked pipeline for large datasets (auto-enabled when >1000 profiles)
    # Disk usage bounded by chunk_size * ~6MB instead of n_profiles * ~6MB
    python evaluation/predict_hypernet.py \\
        --chunk_size 500 \\
        --data LongEvoRoleBench/processed/ChatHaruhi/m6_phase_tree/random_test.json \\
        --output_dir results/ChatHaruhi/hypernet_p2p/main/m6_phase_tree/random_test \\
        --checkpoint phase_tree_models/p2p_pretrained/hypermod.pt
"""

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_SRC = str(PROJECT_ROOT / "src")
if _LOCAL_SRC not in sys.path:
    sys.path.insert(0, _LOCAL_SRC)

BASELINE_PROMPT = """\
Below is a multi-turn dialogue. Predict the single line that {character} would most likely say next.
Keep the reply short and natural, matching the tone and length of the other lines. Output only that one line, no explanation.

Dialogue context:
{context}

{character}:"""


# -----------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------

def load_data(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_done_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["question_id"])
    return ids


def build_prompt(sample: dict) -> str:
    return BASELINE_PROMPT.format(
        character=sample["role"],
        context=sample["input"],
    )


# -----------------------------------------------------------------------
# LoRA generation
# -----------------------------------------------------------------------

def _find_checkpoint_base_dir(checkpoint_path: str) -> str:
    """Walk up from *checkpoint_path* to find the directory with args.yaml.

    ``load_hypermod_checkpoint`` uses a naive ``split("checkpoint")`` to
    locate the base directory, which breaks when any ancestor path also
    contains the substring "checkpoint" (e.g. a hypothetical
    ``phase_tree_models/checkpoint_foo/hypermod.pt``). This helper avoids
    that problem by walking up to the first directory containing
    ``args.yaml``.
    """
    search = os.path.dirname(os.path.abspath(checkpoint_path))
    for _ in range(5):
        if os.path.isfile(os.path.join(search, "args.yaml")):
            return search
        parent = os.path.dirname(search)
        if parent == search:
            break
        search = parent
    return os.path.dirname(os.path.abspath(checkpoint_path))


def _resolve_local_emb_model(checkpoint_path: str) -> str | None:
    """Try to find a local copy of the embedding model specified in args.yaml.

    Searches ``PROJECT_ROOT/models/`` and ``PROJECT_ROOT/../models/`` for a
    directory whose name matches the HuggingFace repo basename.
    """
    import yaml

    base_dir = _find_checkpoint_base_dir(checkpoint_path)
    args_path = os.path.join(base_dir, "args.yaml")
    if not os.path.isfile(args_path):
        return None
    with open(args_path) as f:
        raw = yaml.safe_load(f)
    emb_name = raw.get("emb_model", "")
    if not emb_name or "/" not in emb_name:
        return None
    local_name = emb_name.split("/")[-1]
    for search_dir in [PROJECT_ROOT / "models", PROJECT_ROOT.parent / "models"]:
        candidate = search_dir / local_name
        if candidate.is_dir():
            return str(candidate)
    return None


def _get_ramdisk_lora_dir(fallback_dir: str) -> str:
    """Return a per-process temp LoRA directory on /dev/shm (ramdisk).

    Writing LoRAs to ramdisk avoids frequent small-file I/O on AFS which
    can degrade performance for other users sharing the filesystem.
    If /dev/shm is unavailable or has <2GB free, falls back to *fallback_dir*
    (typically on AFS).
    """
    ramdisk = "/dev/shm"
    min_free_gb = 2.0
    try:
        st = os.statvfs(ramdisk)
        free_gb = (st.f_bavail * st.f_frsize) / (1024 ** 3)
        if free_gb < min_free_gb:
            print(f"  /dev/shm has only {free_gb:.1f}GB free "
                  f"(need {min_free_gb}GB), falling back to AFS", flush=True)
            os.makedirs(fallback_dir, exist_ok=True)
            return fallback_dir
    except OSError:
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir

    ramdisk_dir = os.path.join(ramdisk, f"lora_tmp_{os.getpid()}")
    os.makedirs(ramdisk_dir, exist_ok=True)
    print(f"  Using ramdisk for temp LoRAs: {ramdisk_dir}", flush=True)
    return ramdisk_dir


def _make_safe_ckpt_dir(
    checkpoint_path: str,
    emb_model_override: str | None,
    model_dir_override: str | None = None,
) -> tuple[str, str]:
    """Create a temp directory with symlinks that ``load_hypermod_checkpoint``
    can safely resolve, avoiding both the "checkpoint"-in-path bug and
    parallel race conditions when overriding ``emb_model`` / ``model_dir``.

    Returns ``(temp_hypermod_path, temp_dir_to_cleanup)``.
    """
    import yaml

    base_dir = _find_checkpoint_base_dir(checkpoint_path)
    abs_ckpt = os.path.abspath(checkpoint_path)

    tmp = tempfile.mkdtemp(prefix="hmod_")
    os.symlink(abs_ckpt, os.path.join(tmp, "hypermod.pt"))
    os.symlink(
        os.path.join(base_dir, "adapter_config.json"),
        os.path.join(tmp, "adapter_config.json"),
    )

    args_src = os.path.join(base_dir, "args.yaml")
    need_rewrite = bool(emb_model_override or model_dir_override)

    if need_rewrite:
        with open(args_src) as f:
            raw = yaml.safe_load(f)
        if emb_model_override and raw.get("emb_model") and raw["emb_model"] != emb_model_override:
            print(f"  Overriding emb_model: {raw['emb_model']} "
                  f"-> {emb_model_override}", flush=True)
            raw["emb_model"] = emb_model_override
        if model_dir_override and raw.get("model_dir") and raw["model_dir"] != model_dir_override:
            print(f"  Overriding model_dir: {raw['model_dir']} "
                  f"-> {model_dir_override}", flush=True)
            raw["model_dir"] = model_dir_override
        with open(os.path.join(tmp, "args.yaml"), "w") as f:
            yaml.dump(raw, f, default_flow_style=False)
    else:
        os.symlink(args_src, os.path.join(tmp, "args.yaml"))

    return os.path.join(tmp, "hypermod.pt"), tmp


def _load_hypermod_lightweight(
    checkpoint_path: str,
    emb_model_override: str | None,
    model_dir_override: str | None,
    device: str,
) -> tuple:
    """Load hypermodulator + embedding model WITHOUT loading the 14GB base model.

    Derives model architecture info from ``AutoConfig`` (config only, no
    weights) and constructs the ``HyperModulator`` by loading its state dict
    directly.  This saves ~10 minutes of base-model loading per task.

    Returns ``(ckpt_args, hypermod, peft_config, model_dir, emb_model,
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

    base_dir = _find_checkpoint_base_dir(checkpoint_path)
    with open(os.path.join(base_dir, "args.yaml")) as f:
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

    emb_name = emb_model_override or raw_args.get("emb_model", "")
    model_dir = model_dir_override or raw_args["model_dir"]

    if emb_model_override and raw_args.get("emb_model") != emb_model_override:
        print(f"  Overriding emb_model: {raw_args.get('emb_model')} "
              f"-> {emb_model_override}", flush=True)
    if model_dir_override and raw_args.get("model_dir") != model_dir_override:
        print(f"  Overriding model_dir: {raw_args.get('model_dir')} "
              f"-> {model_dir_override}", flush=True)

    ckpt_args = argparse.Namespace(**raw_args)
    ckpt_args.model_dir = model_dir

    adapter_path = os.path.join(base_dir, "adapter_config.json")
    with open(adapter_path) as f:
        adapter_dict = json.load(f)
    peft_cfg = LoraConfig(**{
        k: v for k, v in adapter_dict.items()
        if k in LoraConfig.__dataclass_fields__
    })

    print("  Loading model config (no weights) ...", flush=True)
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    n_layers = model_config.num_hidden_layers

    # Derive in/out features from config instead of loading the model
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
    in_features = {}
    out_features = {}
    for mod in peft_cfg.target_modules:
        if mod in dim_map:
            in_features[mod] = dim_map[mod][0]
            out_features[mod] = dim_map[mod][1]
        else:
            raise ValueError(f"Unknown target module '{mod}' — add it to dim_map")

    # Build module_names the same way get_lora_module_names does, but
    # synthetically from the known Qwen layer pattern.
    layer_indices = list(range(n_layers))
    module_names = {m: [[] for _ in layer_indices] for m in peft_cfg.target_modules}
    for li in layer_indices:
        for mod in peft_cfg.target_modules:
            prefix = f"base_model.model.model.layers.{li}.self_attn.{mod}"
            module_names[mod][li] = [
                f"{prefix}.lora_A.default.weight",
                f"{prefix}.lora_B.default.weight",
            ]

    print("  Loading hypermodulator weights ...", flush=True)
    state_dict = torch.load(
        os.path.join(base_dir, "hypermod.pt"),
        map_location=device, weights_only=False,
    )
    te_key = "task_encoder.mlp.0.weight"
    task_emb_size = state_dict[te_key].shape[1] if te_key in state_dict else None

    # Build a minimal mock object so HyperModulator.__init__ can read
    # .config, .peft_config, .device, and get_in_out_features works.
    class _MockModel:
        pass
    mock = _MockModel()
    mock.config = model_config
    mock.peft_config = {"default": peft_cfg}
    mock.device = torch.device(device)

    # Monkey-patch functions that normally inspect the loaded PeftModel
    import hyper_llm_modulator.hyper_modulator as _hmod
    _orig_giof = _hmod.get_in_out_features
    _orig_gipw = _hmod.get_init_peft_weights

    def _fast_in_out(model, peft_config=None):
        return in_features, out_features

    def _fast_init_peft_weights(model, peft_config=None):
        r = peft_cfg.r
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
    print(f"  Loaded hypermod state dict: {info}", flush=True)
    hypermod.eval().to(device)

    print("  Loading embedding model ...", flush=True)
    emb_model, emb_tokenizer, _, pooling_fn = get_emb_model_and_fns(emb_name, device)

    return (ckpt_args, hypermod, peft_cfg, model_dir,
            emb_model, emb_tokenizer, pooling_fn, n_layers)


@torch.no_grad()
def generate_loras(
    checkpoint_path: str,
    samples: list[dict],
    lora_base_dir: str,
    emb_model_override: str | None = None,
    model_dir_override: str | None = None,
    device: str = "cuda",
    emb_batch_size: int = 64,
) -> tuple[dict[str, str], dict]:
    """Generate LoRA adapters for all unique profile texts.

    Uses a lightweight loader that skips loading the 14GB base model,
    cutting load time from ~15 min to ~3 min.

    Returns
    -------
    profile_to_lora : dict[str, str]
        Mapping from profile_text -> LoRA directory path.
    gen_stats : dict
        Generation timing, model_dir, lora_rank.
    """
    from hyper_llm_modulator.hyper_modulator import save_lora
    from hyper_llm_modulator.utils.utils import embed_texts

    print("\n  Loading hypernetwork (lightweight, no base model) ...", flush=True)
    (
        ckpt_args, hypermod, peft_cfg, model_dir,
        emb_model, emb_tokenizer, pooling_fn, n_layers,
    ) = _load_hypermod_lightweight(
        checkpoint_path, emb_model_override, model_dir_override, device,
    )

    identity_fn = lambda x: x  # noqa: E731

    layer_indices = torch.tensor(
        range(n_layers), dtype=torch.long, device=device,
    )

    profile_to_indices: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        pt = s.get("profile_text", "")
        profile_to_indices.setdefault(pt, []).append(i)

    n_unique = len(profile_to_indices)
    print(f"  {len(samples)} samples -> {n_unique} unique profiles", flush=True)

    # ------------------------------------------------------------------
    # Phase 1: Partition into cached vs uncached
    # ------------------------------------------------------------------
    os.makedirs(lora_base_dir, exist_ok=True)
    profile_to_lora: dict[str, str] = {}
    uncached_profiles: list[str] = []
    uncached_lora_dirs: list[str] = []

    for profile_text in profile_to_indices:
        profile_hash = hashlib.sha256(profile_text.encode()).hexdigest()[:16]
        lora_dir = os.path.join(lora_base_dir, f"lora_{profile_hash}")

        if os.path.isdir(lora_dir) and os.path.isfile(
            os.path.join(lora_dir, "adapter_model.safetensors")
        ):
            profile_to_lora[profile_text] = lora_dir
        else:
            uncached_profiles.append(profile_text)
            uncached_lora_dirs.append(lora_dir)

    n_cached = n_unique - len(uncached_profiles)
    if n_cached:
        print(f"  {n_cached}/{n_unique} LoRAs cached, "
              f"{len(uncached_profiles)} to generate", flush=True)

    peft_r = peft_cfg.r
    emb_elapsed = 0.0
    gen_seconds: list[float] = []

    if uncached_profiles:
        # --------------------------------------------------------------
        # Phase 2: Batch-embed all uncached profiles
        # --------------------------------------------------------------
        t_emb = time.perf_counter()

        non_empty_texts = [pt for pt in uncached_profiles if pt]
        non_empty_indices = [i for i, pt in enumerate(uncached_profiles) if pt]

        hidden_size = emb_model.config.hidden_size
        zero_emb = torch.zeros(1, hidden_size, device=device)
        emb_lookup: list[torch.Tensor] = [zero_emb] * len(uncached_profiles)

        if non_empty_texts:
            all_embs = embed_texts(
                non_empty_texts, emb_model, emb_tokenizer,
                identity_fn, pooling_fn, device,
                batch_size=emb_batch_size,
            )
            if all_embs.device != torch.device(device):
                all_embs = all_embs.to(device)
            for dest_idx, src_idx in zip(non_empty_indices,
                                         range(len(non_empty_texts))):
                emb_lookup[dest_idx] = all_embs[src_idx : src_idx + 1]

        emb_elapsed = time.perf_counter() - t_emb
        print(f"  Embedded {len(non_empty_texts)} profiles in {emb_elapsed:.1f}s "
              f"(batch_size={emb_batch_size})", flush=True)

        # --------------------------------------------------------------
        # Phase 3: Generate LoRAs sequentially (gen_lora supports batch=1)
        # --------------------------------------------------------------
        for idx, (profile_text, lora_dir) in enumerate(
            zip(uncached_profiles, uncached_lora_dirs)
        ):
            task_emb = emb_lookup[idx]
            encoder_out = hypermod.task_encoder(task_emb)
            encoded = encoder_out["encoded_task_emb"].detach()

            t0 = time.perf_counter()
            lora_sd = hypermod.gen_lora(layer_indices, encoded)
            gen_seconds.append(time.perf_counter() - t0)

            # vLLM expects "lora_A.weight", not "lora_A.default.weight"
            lora_sd = {
                k.replace(".default.weight", ".weight"): v
                for k, v in lora_sd.items()
            }
            save_lora(lora_sd, hypermod.peft_config, lora_dir)
            profile_to_lora[profile_text] = lora_dir

            n_done = idx + 1
            if n_done % 50 == 0 or n_done == len(uncached_profiles):
                print(f"    LoRA {n_done}/{len(uncached_profiles)} generated",
                      flush=True)

    del hypermod, emb_model, emb_tokenizer
    del layer_indices, identity_fn, pooling_fn
    torch.cuda.empty_cache()
    gc.collect()

    total_gen = sum(gen_seconds) if gen_seconds else 0
    gen_stats = {
        "num_unique_profiles": n_unique,
        "num_cached": n_cached,
        "num_generated": len(gen_seconds),
        "emb_seconds": round(emb_elapsed, 2),
        "emb_batch_size": emb_batch_size,
        "gen_lora_seconds": round(total_gen, 2),
        "mean_gen_ms": round(total_gen / len(gen_seconds) * 1000, 2)
        if gen_seconds else 0,
        "total_seconds": round(emb_elapsed + total_gen, 2),
        "model_dir": model_dir,
        "lora_rank": peft_r,
    }
    print(
        f"  Done: {n_unique} LoRAs ({gen_stats['num_generated']} newly generated) "
        f"— embed {emb_elapsed:.1f}s + gen {total_gen:.1f}s",
        flush=True,
    )
    return profile_to_lora, gen_stats


# -----------------------------------------------------------------------
# vLLM inference with per-group LoRA
# -----------------------------------------------------------------------

def run_vllm_with_lora(
    model_dir: str,
    lora_rank: int,
    samples: list[dict],
    profile_to_lora: dict[str, str],
    pred_path: str,
    max_tokens: int = 256,
    temperature: float = 0.3,
    seed: int = 42,
    tensor_parallel: int = 0,
    gpu_memory_utilization: float = 0.9,
) -> float:
    """Run vLLM inference, grouping samples by their LoRA.

    Returns elapsed seconds.
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # vLLM 0.5.4 only catches OSError in get_lora_tokenizer, but
    # LoRA dirs without a full model config raise ValueError/TypeError.
    # Must patch both the module attr AND the local binding in tokenizer_group.
    import vllm.transformers_utils.tokenizer as _vllm_tok
    import vllm.transformers_utils.tokenizer_group.tokenizer_group as _vllm_tg
    _orig_fn = _vllm_tok.get_lora_tokenizer

    def _safe_get_lora_tokenizer(lora_request, *a, **kw):
        try:
            return _orig_fn(lora_request, *a, **kw)
        except Exception:
            return None

    _vllm_tok.get_lora_tokenizer = _safe_get_lora_tokenizer
    _vllm_tok.get_lora_tokenizer_async = _vllm_tok.make_async(
        _safe_get_lora_tokenizer)
    _vllm_tg.get_lora_tokenizer = _safe_get_lora_tokenizer
    _vllm_tg.get_lora_tokenizer_async = _vllm_tok.make_async(
        _safe_get_lora_tokenizer)

    n_gpus = torch.cuda.device_count()
    tp = min(n_gpus, tensor_parallel) if tensor_parallel else n_gpus
    unique_loras = set(profile_to_lora.values())
    max_loras = min(len(unique_loras), 16)
    print(
        f"\n  vLLM: model={model_dir}, tp={tp}, "
        f"lora_rank={lora_rank}, max_loras={max_loras}",
        flush=True,
    )

    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tp,
        max_model_len=4096,
        enable_lora=True,
        max_lora_rank=lora_rank,
        max_loras=max(max_loras, 1),
        trust_remote_code=True,
        dtype="bfloat16",
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )

    lora_dir_to_id: dict[str, int] = {}
    lora_requests: dict[str, LoRARequest] = {}
    next_id = 1
    for lora_dir in unique_loras:
        lora_dir_to_id[lora_dir] = next_id
        lora_requests[lora_dir] = LoRARequest(
            f"lora_{next_id}", next_id, lora_dir,
        )
        next_id += 1

    lora_groups: dict[str, list[tuple[int, str]]] = {}
    for i, s in enumerate(samples):
        pt = s.get("profile_text", "")
        lora_dir = profile_to_lora[pt]
        raw = build_prompt(s)
        messages = [{"role": "user", "content": raw}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        lora_groups.setdefault(lora_dir, []).append((i, text))

    results: dict[int, str] = {}
    t0 = time.perf_counter()
    n_done = 0
    n_groups = len(lora_groups)

    for g_idx, (lora_dir, group) in enumerate(lora_groups.items()):
        indices, prompts = zip(*group)
        lr = lora_requests[lora_dir]
        outputs = llm.generate(list(prompts), sampling_params, lora_request=lr)
        for idx, out in zip(indices, outputs):
            results[idx] = out.outputs[0].text.strip()
        n_done += len(group)
        if (g_idx + 1) % 20 == 0 or (g_idx + 1) == n_groups:
            elapsed = time.perf_counter() - t0
            print(
                f"    Group {g_idx + 1}/{n_groups} done  "
                f"({n_done}/{len(samples)} samples, {elapsed:.1f}s)",
                flush=True,
            )

    elapsed = time.perf_counter() - t0

    with open(pred_path, "a", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            record = {
                "question_id": s["question_id"],
                "role": s["role"],
                "prediction": results[i],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    speed = len(samples) / elapsed if elapsed > 0 else 0
    print(
        f"\n  Inference done: {len(samples)} samples in {elapsed:.1f}s "
        f"({speed:.1f} samples/s)",
        flush=True,
    )

    del llm
    torch.cuda.empty_cache()
    gc.collect()
    return elapsed


# -----------------------------------------------------------------------
# Chunked pipeline (for large-scale LoRA generation)
# -----------------------------------------------------------------------

@torch.no_grad()
def run_chunked_pipeline(
    checkpoint_path: str,
    samples: list[dict],
    lora_base_dir: str,
    pred_path: str,
    inference_model_dir: str,
    chunk_size: int = 500,
    emb_model_override: str | None = None,
    device: str = "cuda",
    emb_batch_size: int = 64,
    max_tokens: int = 256,
    temperature: float = 0.3,
    seed: int = 42,
    tensor_parallel: int = 0,
    gpu_memory_utilization: float = 0.9,
) -> tuple[dict, float]:
    """Chunked pipeline: pre-compute embeddings, then interleave gen_lora + vLLM.

    Keeps hypermod (~1GB) in VRAM alongside vLLM so that LoRAs are generated
    on-the-fly per chunk and deleted after inference.  Peak disk usage is
    bounded by ``chunk_size * ~6 MB`` instead of ``n_profiles * ~6 MB``.

    Returns ``(gen_stats, infer_elapsed)``.
    """
    from hyper_llm_modulator.hyper_modulator import (
        load_hypermod_checkpoint,
        save_lora,
    )
    from hyper_llm_modulator.utils import get_layers
    from hyper_llm_modulator.utils.utils import embed_texts
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # ==================================================================
    # Phase 1: Load models → batch embed → encode → free heavy models
    # ==================================================================
    safe_path, tmp_dir = _make_safe_ckpt_dir(checkpoint_path, emb_model_override)
    try:
        print("\n  Loading hypernetwork checkpoint ...", flush=True)
        (
            ckpt_args, hypermod, model, tokenizer,
            emb_model, emb_tokenizer, _, pooling_fn,
        ) = load_hypermod_checkpoint(safe_path, device)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    identity_fn = lambda x: x  # noqa: E731
    layer_indices = torch.tensor(
        range(len(get_layers(model))), dtype=torch.long, device=device,
    )
    peft_config = hypermod.peft_config
    model_dir = ckpt_args.model_dir
    lora_rank = peft_config.r

    # Group samples by profile
    profile_to_sample_idx: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        pt = s.get("profile_text", "")
        profile_to_sample_idx.setdefault(pt, []).append(i)

    all_profiles = list(profile_to_sample_idx.keys())
    n_unique = len(all_profiles)

    # Partition cached vs uncached
    os.makedirs(lora_base_dir, exist_ok=True)
    profile_hash: dict[str, str] = {}
    profile_lora_dir: dict[str, str] = {}
    cached_set: set[str] = set()

    for pt in all_profiles:
        h = hashlib.sha256(pt.encode()).hexdigest()[:16]
        profile_hash[pt] = h
        ld = os.path.join(lora_base_dir, f"lora_{h}")
        profile_lora_dir[pt] = ld
        if os.path.isdir(ld) and os.path.isfile(
            os.path.join(ld, "adapter_model.safetensors")
        ):
            cached_set.add(pt)

    uncached = [pt for pt in all_profiles if pt not in cached_set]
    n_cached = len(cached_set)
    print(f"  {len(samples)} samples -> {n_unique} unique profiles "
          f"({n_cached} cached, {len(uncached)} to generate)", flush=True)

    # Batch embed + task_encoder for all uncached profiles
    t_emb = time.perf_counter()
    encoded_lookup: dict[str, torch.Tensor] = {}

    if uncached:
        non_empty = [pt for pt in uncached if pt]
        non_empty_pos = {pt: i for i, pt in enumerate(non_empty)}
        hidden_size = emb_model.config.hidden_size

        all_embs = None
        if non_empty:
            all_embs = embed_texts(
                non_empty, emb_model, emb_tokenizer,
                identity_fn, pooling_fn, device,
                batch_size=emb_batch_size,
            )
            if all_embs.device != torch.device(device):
                all_embs = all_embs.to(device)

        for pt in uncached:
            if pt and all_embs is not None:
                task_emb = all_embs[non_empty_pos[pt] : non_empty_pos[pt] + 1]
            else:
                task_emb = torch.zeros(1, hidden_size, device=device)
            enc_out = hypermod.task_encoder(task_emb)
            encoded_lookup[pt] = enc_out["encoded_task_emb"].detach()

        del all_embs

    emb_elapsed = time.perf_counter() - t_emb

    # Free heavy models — keep only hypermod (~1GB) + encoded_lookup (~20MB)
    del model, tokenizer, emb_model, emb_tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  Embedded + encoded {len(uncached)} profiles in {emb_elapsed:.1f}s, "
          f"freed base_model + emb_model (keeping hypermod ~1GB)", flush=True)

    # ==================================================================
    # Phase 2: Initialize vLLM (once)
    # ==================================================================
    n_gpus = torch.cuda.device_count()
    tp = min(n_gpus, tensor_parallel) if tensor_parallel else n_gpus
    max_loras_vllm = min(chunk_size, 16)
    print(f"\n  vLLM: model={inference_model_dir}, tp={tp}, "
          f"lora_rank={lora_rank}, max_loras={max_loras_vllm}", flush=True)

    llm = LLM(
        model=inference_model_dir,
        tensor_parallel_size=tp,
        max_model_len=4096,
        enable_lora=True,
        max_lora_rank=lora_rank,
        max_loras=max(max_loras_vllm, 1),
        trust_remote_code=True,
        dtype="bfloat16",
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    vllm_tok = llm.get_tokenizer()
    sp = SamplingParams(
        temperature=temperature, max_tokens=max_tokens, seed=seed,
    )

    # ==================================================================
    # Phase 3: Chunked generate + infer loop
    # ==================================================================
    results: dict[int, str] = {}
    gen_seconds: list[float] = []
    n_chunks = (n_unique + chunk_size - 1) // chunk_size
    t_infer_total = 0.0
    lora_id_counter = 1
    peak_chunk_uncached = min(chunk_size, len(uncached))

    print(f"\n  Chunked pipeline: {n_unique} profiles in {n_chunks} chunks "
          f"(chunk_size={chunk_size}, "
          f"peak_disk~{peak_chunk_uncached * 6.1:.0f}MB)", flush=True)

    for c_idx in range(n_chunks):
        c_profiles = all_profiles[
            c_idx * chunk_size : (c_idx + 1) * chunk_size
        ]

        # 3a: Generate LoRAs for uncached profiles in this chunk
        temp_dirs: list[str] = []
        for pt in c_profiles:
            if pt in cached_set:
                continue
            ld = profile_lora_dir[pt]
            t0 = time.perf_counter()
            lora_sd = hypermod.gen_lora(layer_indices, encoded_lookup[pt])
            gen_seconds.append(time.perf_counter() - t0)
            lora_sd = {
                k.replace(".default.weight", ".weight"): v
                for k, v in lora_sd.items()
            }
            save_lora(lora_sd, peft_config, ld)
            temp_dirs.append(ld)

        # 3b: Build LoRA requests + group samples for this chunk
        chunk_lora_reqs: dict[str, LoRARequest] = {}
        for pt in c_profiles:
            ld = profile_lora_dir[pt]
            if ld not in chunk_lora_reqs:
                chunk_lora_reqs[ld] = LoRARequest(
                    f"lora_{lora_id_counter}", lora_id_counter, ld,
                )
                lora_id_counter += 1

        lora_groups: dict[str, list[tuple[int, str]]] = {}
        for pt in c_profiles:
            ld = profile_lora_dir[pt]
            for si in profile_to_sample_idx[pt]:
                raw = build_prompt(samples[si])
                msgs = [{"role": "user", "content": raw}]
                text = vllm_tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
                lora_groups.setdefault(ld, []).append((si, text))

        # 3c: vLLM inference for this chunk
        t_chunk = time.perf_counter()
        for ld, group in lora_groups.items():
            indices, prompts = zip(*group)
            outputs = llm.generate(
                list(prompts), sp, lora_request=chunk_lora_reqs[ld],
            )
            for idx, out in zip(indices, outputs):
                results[idx] = out.outputs[0].text.strip()
        t_infer_total += time.perf_counter() - t_chunk

        # 3d: Delete temporary LoRA files for this chunk
        for td in temp_dirs:
            shutil.rmtree(td, ignore_errors=True)

        print(f"    Chunk {c_idx + 1}/{n_chunks}: "
              f"{len(c_profiles)} profiles ({len(temp_dirs)} generated), "
              f"{len(results)}/{len(samples)} samples done", flush=True)

    # Write all predictions in original order
    with open(pred_path, "a", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            if i in results:
                record = {
                    "question_id": s["question_id"],
                    "role": s["role"],
                    "prediction": results[i],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    speed = len(samples) / t_infer_total if t_infer_total > 0 else 0
    print(f"\n  Chunked pipeline done: {len(samples)} samples, "
          f"infer {t_infer_total:.1f}s ({speed:.1f} samples/s)", flush=True)

    # Cleanup
    del llm, hypermod, layer_indices, encoded_lookup
    torch.cuda.empty_cache()
    gc.collect()

    # Clean up empty temp dir
    if os.path.isdir(lora_base_dir) and not os.listdir(lora_base_dir):
        shutil.rmtree(lora_base_dir, ignore_errors=True)

    total_gen = sum(gen_seconds) if gen_seconds else 0
    gen_stats = {
        "num_unique_profiles": n_unique,
        "num_cached": n_cached,
        "num_generated": len(gen_seconds),
        "emb_seconds": round(emb_elapsed, 2),
        "emb_batch_size": emb_batch_size,
        "gen_lora_seconds": round(total_gen, 2),
        "mean_gen_ms": round(total_gen / len(gen_seconds) * 1000, 2)
        if gen_seconds else 0,
        "total_seconds": round(emb_elapsed + total_gen, 2),
        "model_dir": model_dir,
        "lora_rank": lora_rank,
        "chunked": True,
        "chunk_size": chunk_size,
        "num_chunks": n_chunks,
        "peak_disk_mb": round(peak_chunk_uncached * 6.1, 1),
    }
    return gen_stats, t_infer_total


# -----------------------------------------------------------------------
# Token statistics (aligned with predict_prompt.py)
# -----------------------------------------------------------------------

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
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tok.padding_side = "left"
        tok.truncation_side = "left"
        _tokenizer_cache[model_path] = tok
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


def _compute_token_stats(samples: list[dict], model_path: str) -> dict:
    tokenizer = _get_tokenizer(model_path)

    profile_lens = []
    context_lens = []
    output_lens = []
    prompt_lens = []

    for s in samples:
        profile_text = s.get("profile_text", "")
        context_text = s.get("input", "")
        output_text = s.get("output", "")

        profile_lens.append(len(tokenizer.encode(profile_text)))
        context_lens.append(len(tokenizer.encode(context_text)))
        output_lens.append(len(tokenizer.encode(output_text)))

        raw_prompt = build_prompt(s)
        messages = [{"role": "user", "content": raw_prompt}]
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_lens.append(len(tokenizer.encode(full_prompt)))

    return {
        "profile_tokens": _arr_stats(profile_lens),
        "context_tokens": _arr_stats(context_lens),
        "output_tokens": _arr_stats(output_lens),
        "prompt_tokens": _arr_stats(prompt_lens),
        "tokenizer": model_path,
        "num_samples": len(samples),
    }


def _compute_prediction_token_stats(
    pred_path: str, model_path: str,
) -> dict | None:
    if not os.path.exists(pred_path):
        return None

    predictions = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line)["prediction"])

    if not predictions:
        return None

    tokenizer = _get_tokenizer(model_path)
    pred_lens = [len(tokenizer.encode(p)) for p in predictions]
    stats = _arr_stats(pred_lens)
    stats["num_predictions"] = len(predictions)
    return stats


def _save_meta(
    args,
    total_samples: int,
    predicted_samples: int,
    inference_model_dir: str,
    token_stats: dict | None = None,
    gen_stats: dict | None = None,
    infer_elapsed: float | None = None,
):
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass

    meta = {
        "mode": "hypernet_lora",
        "model": inference_model_dir,
        "checkpoint": args.checkpoint,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "tensor_parallel": args.tensor_parallel,
        "total_samples": total_samples,
        "predicted_this_run": predicted_samples,
        "data_path": args.data,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_hash": git_hash,
    }
    if gen_stats:
        meta["lora_generation"] = gen_stats
    if infer_elapsed is not None and predicted_samples > 0:
        meta["inference"] = {
            "model_dir": inference_model_dir,
            "total_seconds": round(infer_elapsed, 2),
            "num_predicted": predicted_samples,
            "mean_ms_per_sample": round(infer_elapsed / predicted_samples * 1000, 1),
            "samples_per_second": round(predicted_samples / infer_elapsed, 2)
            if infer_elapsed > 0 else 0,
        }
    if token_stats:
        meta["token_stats"] = token_stats

    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  Meta saved: {meta_path}", flush=True)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Predict with P2P hypernetwork-generated LoRA",
    )
    parser.add_argument("--data", required=True,
                        help="Path to processed dialogue JSON")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for predictions")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to hypermod.pt")
    parser.add_argument("--emb_model_override", default=None,
                        help="Override the embedding model path "
                             "(use local path if HF Hub is unavailable)")
    parser.add_argument("--model_override", default=None,
                        help="Override the base LLM path for vLLM inference "
                             "(e.g. /dev/shm/Qwen2.5-7B-Instruct)")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor_parallel", type=int, default=0,
                        help="TP size for vLLM (0 = auto)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default=None,
                        help="Comma-separated GPU IDs (e.g. '0' or '0,1'). "
                             "Sets CUDA_VISIBLE_DEVICES for multi-process "
                             "parallelism across methods/splits")
    parser.add_argument("--emb_batch_size", type=int, default=64,
                        help="Batch size for embedding model (higher = faster "
                             "but more VRAM)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="Fraction of GPU memory vLLM is allowed to use. "
                             "Lower (e.g. 0.7) when sharing a GPU or when the "
                             "embedding model/hypermod hasn't fully released "
                             "memory before vLLM loads.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit samples for debugging")
    parser.add_argument("--save_loras_dir", default=None,
                        help="If set, copy generated per-character LoRAs here "
                             "for persistent storage")
    parser.add_argument("--chunk_size", type=int, default=0,
                        help="Chunked pipeline: generate+infer LoRAs in "
                             "chunks of this size, deleting after each chunk "
                             "to bound disk usage.  0 = auto (enable when "
                             ">1000 unique profiles), -1 = force disable")

    args = parser.parse_args()

    # Pin to specific GPUs before any CUDA init
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"  CUDA_VISIBLE_DEVICES={args.gpu}", flush=True)

    # Auto-detect local embedding model if not overridden
    if args.emb_model_override is None:
        local_emb = _resolve_local_emb_model(args.checkpoint)
        if local_emb:
            args.emb_model_override = local_emb
            print(f"  Auto-detected local emb_model: {local_emb}", flush=True)

    # Early-resolve inference model path (for token stats before LoRA gen)
    import yaml as _yaml
    _ckpt_base = _find_checkpoint_base_dir(args.checkpoint)
    with open(os.path.join(_ckpt_base, "args.yaml")) as _f:
        _ckpt_cfg = _yaml.safe_load(_f)
    inference_model_dir = args.model_override or _ckpt_cfg.get("model_dir", "")
    emb_model_display = (args.emb_model_override
                         or _ckpt_cfg.get("emb_model", "N/A"))

    samples = load_data(args.data)
    if args.num_samples is not None:
        samples = samples[: args.num_samples]
        print(f"  Warning: Debug mode — limited to {args.num_samples} samples",
              flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    done_ids = load_done_ids(pred_path)
    remaining = [s for s in samples if s["question_id"] not in done_ids]

    n_roles = len(set(s["role"] for s in samples))
    n_unique_profiles = len(set(s.get("profile_text", "") for s in samples))
    n_gpus = torch.cuda.device_count()
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Mode        : Hypernetwork LoRA (P2P)", flush=True)
    print(f"  Data        : {args.data}", flush=True)
    print(f"  Checkpoint  : {args.checkpoint}", flush=True)
    print(f"  Model       : {inference_model_dir}", flush=True)
    print(f"  Emb model   : {emb_model_display}", flush=True)
    print(f"  GPU         : {n_gpus} visible"
          f"{f' (CUDA_VISIBLE_DEVICES={args.gpu})' if args.gpu else ''}",
          flush=True)
    print(f"  Samples     : {len(samples)} ({n_roles} characters, "
          f"{n_unique_profiles} unique profiles)", flush=True)
    print(f"  Seed        : {args.seed}", flush=True)
    print(f"  Progress    : {len(done_ids)}/{len(samples)} done, "
          f"{len(remaining)} remaining", flush=True)
    print(f"  Output      : {args.output_dir}/", flush=True)
    if args.chunk_size != -1:
        print(f"  Chunk size  : "
              f"{'auto' if args.chunk_size == 0 else args.chunk_size}",
              flush=True)
    print(f"{'=' * 60}", flush=True)

    # --- Token statistics ---
    token_stats = _load_cached_token_stats(args.output_dir, len(samples))
    if token_stats:
        print(f"\n  Token stats (cached): "
              f"profile={token_stats['profile_tokens']['mean']:.1f}, "
              f"context={token_stats['context_tokens']['mean']:.1f}, "
              f"output_gt={token_stats['output_tokens']['mean']:.1f}, "
              f"prompt={token_stats['prompt_tokens']['mean']:.1f} (mean tokens)",
              flush=True)
    else:
        print(f"\n  Computing token statistics ...", flush=True)
        token_stats = _compute_token_stats(samples, inference_model_dir)
        print(f"  Token stats: "
              f"profile={token_stats['profile_tokens']['mean']:.1f}, "
              f"context={token_stats['context_tokens']['mean']:.1f}, "
              f"output_gt={token_stats['output_tokens']['mean']:.1f}, "
              f"prompt={token_stats['prompt_tokens']['mean']:.1f} (mean tokens)",
              flush=True)

    if not remaining:
        print(f"\nAll {len(samples)} predictions already done.", flush=True)
        pred_token_stats = _compute_prediction_token_stats(
            pred_path, inference_model_dir,
        )
        if pred_token_stats:
            token_stats["prediction_tokens"] = pred_token_stats
        _save_meta(args, len(samples), 0, inference_model_dir,
                   token_stats=token_stats)
        return

    # --- Determine chunking strategy ---
    n_remaining_profiles = len(set(
        s.get("profile_text", "") for s in remaining
    ))
    chunk_size = args.chunk_size
    if chunk_size == 0:  # auto
        if n_remaining_profiles > 1000:
            chunk_size = 500
            print(f"\n  Auto-enabled chunked pipeline (chunk_size={chunk_size}) "
                  f"for {n_remaining_profiles} unique profiles "
                  f"(~{n_remaining_profiles * 6.1 / 1024:.1f}GB LoRAs)",
                  flush=True)
    use_chunked = (
        chunk_size > 0
        and not args.save_loras_dir
    )

    _afs_temp_loras = os.path.join(args.output_dir, "_temp_loras")

    if use_chunked:
        # ---- Chunked pipeline: generate + infer in interleaved chunks ----
        lora_base_dir = _get_ramdisk_lora_dir(_afs_temp_loras)
        gen_stats, infer_elapsed = run_chunked_pipeline(
            checkpoint_path=args.checkpoint,
            samples=remaining,
            lora_base_dir=lora_base_dir,
            pred_path=pred_path,
            inference_model_dir=inference_model_dir,
            chunk_size=chunk_size,
            emb_model_override=args.emb_model_override,
            device=args.device,
            emb_batch_size=args.emb_batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
            tensor_parallel=args.tensor_parallel,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    else:
        # ---- Standard two-phase pipeline ----
        lora_base_dir = _get_ramdisk_lora_dir(_afs_temp_loras)

        if args.save_loras_dir and os.path.isdir(args.save_loras_dir):
            existing = [
                d for d in os.listdir(args.save_loras_dir)
                if d.startswith("lora_") and os.path.isdir(
                    os.path.join(args.save_loras_dir, d)
                )
            ]
            if existing:
                print(f"  Found {len(existing)} cached LoRAs in "
                      f"{args.save_loras_dir}", flush=True)
                lora_base_dir = args.save_loras_dir

        profile_to_lora, gen_stats = generate_loras(
            args.checkpoint, remaining, lora_base_dir,
            emb_model_override=args.emb_model_override,
            model_dir_override=args.model_override,
            device=args.device,
            emb_batch_size=args.emb_batch_size,
        )

        if (
            args.save_loras_dir
            and lora_base_dir != args.save_loras_dir
            and gen_stats["num_unique_profiles"] <= 50
        ):
            os.makedirs(args.save_loras_dir, exist_ok=True)
            for pt, lora_dir in profile_to_lora.items():
                dst = os.path.join(
                    args.save_loras_dir, os.path.basename(lora_dir),
                )
                if not os.path.exists(dst):
                    shutil.copytree(lora_dir, dst)
                profile_to_lora[pt] = dst
            print(f"  Per-character LoRAs saved to {args.save_loras_dir}",
                  flush=True)

        infer_elapsed = run_vllm_with_lora(
            model_dir=inference_model_dir,
            lora_rank=gen_stats["lora_rank"],
            samples=remaining,
            profile_to_lora=profile_to_lora,
            pred_path=pred_path,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
            tensor_parallel=args.tensor_parallel,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        for tmp in [lora_base_dir, _afs_temp_loras]:
            if os.path.exists(tmp):
                shutil.rmtree(tmp, ignore_errors=True)
        print(f"  Temp LoRAs cleaned up (ramdisk + AFS)", flush=True)

    pred_token_stats = _compute_prediction_token_stats(
        pred_path, inference_model_dir,
    )
    if pred_token_stats:
        token_stats["prediction_tokens"] = pred_token_stats

    _save_meta(args, len(samples), len(remaining), inference_model_dir,
               token_stats=token_stats, gen_stats=gen_stats,
               infer_elapsed=infer_elapsed)


# -----------------------------------------------------------------------
# Multi-task mode: load models once per GPU, run multiple tasks
# -----------------------------------------------------------------------

def multi_main():
    """Run multiple tasks on a single GPU with models loaded only once.

    Phase 1: Load hypermod + emb model ONCE → generate LoRAs for ALL tasks.
    Phase 2: Load vLLM ONCE → infer ALL tasks, swapping LoRA adapters.

    Usage::

        python evaluation/predict_hypernet.py --multi \\
            --tasks tasks_gpu0.json \\
            --checkpoint phase_tree_models/p2p_pretrained/hypermod.pt \\
            --emb_model_override /dev/shm/Qwen3-Embedding-4B \\
            --model_override /dev/shm/Qwen2.5-7B-Instruct

    ``tasks_gpu0.json`` is a JSON list::

        [
          {"data": "LongEvoRoleBench/.../m2_raw_profile/random_test.json",
           "output_dir": "results/.../m2_raw_profile/random_test",
           "save_loras_dir": "results/.../generated_loras/m2_raw_profile"},
          {"data": "LongEvoRoleBench/.../m2_raw_profile/ood_test.json",
           "output_dir": "results/.../m2_raw_profile/ood_test",
           "save_loras_dir": "results/.../generated_loras/m2_raw_profile"}
        ]
    """
    parser = argparse.ArgumentParser(
        description="Multi-task P2P hypernetwork LoRA: models loaded once per GPU",
    )
    parser.add_argument("--multi", action="store_true")
    parser.add_argument("--tasks", required=True,
                        help="JSON file listing tasks [{data, output_dir, "
                             "save_loras_dir?}, ...]")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--emb_model_override", default=None)
    parser.add_argument("--model_override", default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--emb_batch_size", type=int, default=64)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="Fraction of GPU memory vLLM is allowed to use")
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.tasks) as f:
        tasks = json.load(f)
    assert isinstance(tasks, list) and tasks, "--tasks must be a non-empty JSON list"

    # Auto-detect local embedding model
    if args.emb_model_override is None:
        local_emb = _resolve_local_emb_model(args.checkpoint)
        if local_emb:
            args.emb_model_override = local_emb
            print(f"  Auto-detected local emb_model: {local_emb}", flush=True)

    import yaml as _yaml
    _ckpt_base = _find_checkpoint_base_dir(args.checkpoint)
    with open(os.path.join(_ckpt_base, "args.yaml")) as _f:
        _ckpt_cfg = _yaml.safe_load(_f)
    inference_model_dir = args.model_override or _ckpt_cfg.get("model_dir", "")

    # ------------------------------------------------------------------
    # Load all tasks, identify remaining work
    # ------------------------------------------------------------------
    task_data = []
    merged_remaining = []
    total_all = 0

    for tc in tasks:
        samples = load_data(tc["data"])
        if args.num_samples is not None:
            samples = samples[:args.num_samples]
        os.makedirs(tc["output_dir"], exist_ok=True)
        pred_path = os.path.join(tc["output_dir"], "predictions.jsonl")
        done_ids = load_done_ids(pred_path)
        remaining = [s for s in samples if s["question_id"] not in done_ids]
        total_all += len(samples)
        task_data.append({
            "config": tc,
            "samples": samples,
            "remaining": remaining,
            "pred_path": pred_path,
            "done_ids": done_ids,
        })
        merged_remaining.extend(remaining)

    n_unique_profiles = len(set(
        s.get("profile_text", "") for s in merged_remaining
    ))

    print(f"\n{'=' * 60}", flush=True)
    print(f"  Mode       : Multi-task Hypernetwork LoRA (models loaded ONCE)",
          flush=True)
    print(f"  Tasks      : {len(tasks)}", flush=True)
    for i, tc in enumerate(tasks):
        td = task_data[i]
        label = os.path.basename(os.path.dirname(tc["data"]))
        split = os.path.splitext(os.path.basename(tc["data"]))[0]
        print(f"    [{i}] {label}/{split}  "
              f"({len(td['remaining'])}/{len(td['samples'])} remaining)",
              flush=True)
    print(f"  Checkpoint : {args.checkpoint}", flush=True)
    print(f"  Model      : {inference_model_dir}", flush=True)
    print(f"  Emb model  : {args.emb_model_override or 'N/A'}", flush=True)
    print(f"  Total      : {len(merged_remaining)}/{total_all} samples remaining "
          f"({n_unique_profiles} unique profiles)", flush=True)
    print(f"{'=' * 60}", flush=True)

    if not merged_remaining:
        print("\nAll tasks already completed.", flush=True)
        return

    t_total_start = time.perf_counter()

    # ==================================================================
    # PHASE 1: Generate LoRAs for ALL tasks (hypermod loaded once)
    # ==================================================================
    print(f"\n{'─' * 60}", flush=True)
    print("  PHASE 1: LoRA generation (hypermod loaded ONCE)", flush=True)
    print(f"{'─' * 60}", flush=True)

    _afs_shared_temp = os.path.join(
        os.path.dirname(tasks[0]["output_dir"]), "_shared_temp_loras",
    )
    lora_base_dir = _get_ramdisk_lora_dir(_afs_shared_temp)
    profile_to_lora, gen_stats = generate_loras(
        args.checkpoint, merged_remaining, lora_base_dir,
        emb_model_override=args.emb_model_override,
        model_dir_override=args.model_override,
        device=args.device,
        emb_batch_size=args.emb_batch_size,
    )
    lora_rank = gen_stats["lora_rank"]

    # Handle per-character LoRA persistence
    for td in task_data:
        tc = td["config"]
        save_dir = tc.get("save_loras_dir")
        if not save_dir:
            continue
        os.makedirs(save_dir, exist_ok=True)
        for s in td["remaining"]:
            pt = s.get("profile_text", "")
            if pt in profile_to_lora:
                src = profile_to_lora[pt]
                dst = os.path.join(save_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copytree(src, dst)
                profile_to_lora[pt] = dst
        n_saved = len([
            d for d in os.listdir(save_dir)
            if d.startswith("lora_") and os.path.isdir(os.path.join(save_dir, d))
        ])
        print(f"  Per-character LoRAs: {save_dir} ({n_saved} total)", flush=True)

    # ==================================================================
    # PHASE 2: vLLM inference for ALL tasks (LLM loaded once)
    # ==================================================================
    print(f"\n{'─' * 60}", flush=True)
    print("  PHASE 2: vLLM inference (LLM loaded ONCE)", flush=True)
    print(f"{'─' * 60}", flush=True)

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
    _vllm_tok.get_lora_tokenizer_async = _vllm_tok.make_async(
        _safe_get_lora_tokenizer)
    _vllm_tg.get_lora_tokenizer = _safe_get_lora_tokenizer
    _vllm_tg.get_lora_tokenizer_async = _vllm_tok.make_async(
        _safe_get_lora_tokenizer)

    n_gpus = torch.cuda.device_count()
    tp = min(n_gpus, args.tensor_parallel) if args.tensor_parallel else n_gpus
    all_unique_loras = set(profile_to_lora.values())
    max_loras = min(len(all_unique_loras), 16)

    print(f"\n  Loading vLLM: model={inference_model_dir}, tp={tp}, "
          f"lora_rank={lora_rank}, max_loras={max_loras}", flush=True)
    t_vllm_load = time.perf_counter()

    llm = LLM(
        model=inference_model_dir,
        tensor_parallel_size=tp,
        max_model_len=4096,
        enable_lora=True,
        max_lora_rank=lora_rank,
        max_loras=max(max_loras, 1),
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
    print(f"  vLLM loaded in {time.perf_counter() - t_vllm_load:.1f}s "
          f"(gpu_mem_util={args.gpu_memory_utilization})", flush=True)

    # ── Run inference per task ──────────────────────────────────────
    for task_idx, td in enumerate(task_data):
        remaining = td["remaining"]
        if not remaining:
            tc = td["config"]
            label = os.path.basename(os.path.dirname(tc["data"]))
            split = os.path.splitext(os.path.basename(tc["data"]))[0]
            print(f"\n  [{task_idx}] {label}/{split}: already complete", flush=True)
            continue

        tc = td["config"]
        label = os.path.basename(os.path.dirname(tc["data"]))
        split = os.path.splitext(os.path.basename(tc["data"]))[0]
        pred_path = td["pred_path"]

        print(f"\n  [{task_idx}] {label}/{split}: {len(remaining)} samples ...",
              flush=True)

        # Build LoRA request registry for this task
        task_lora_dirs = set()
        for s in remaining:
            pt = s.get("profile_text", "")
            task_lora_dirs.add(profile_to_lora[pt])

        lora_dir_to_id: dict[str, int] = {}
        lora_requests: dict[str, LoRARequest] = {}
        next_id = 1
        for lora_dir in task_lora_dirs:
            lora_dir_to_id[lora_dir] = next_id
            lora_requests[lora_dir] = LoRARequest(
                f"lora_{next_id}", next_id, lora_dir,
            )
            next_id += 1

        # Group by LoRA
        lora_groups: dict[str, list[tuple[int, str]]] = {}
        for i, s in enumerate(remaining):
            pt = s.get("profile_text", "")
            lora_dir = profile_to_lora[pt]
            raw = build_prompt(s)
            messages = [{"role": "user", "content": raw}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            lora_groups.setdefault(lora_dir, []).append((i, text))

        results: dict[int, str] = {}
        t0 = time.perf_counter()
        n_done = 0
        n_groups = len(lora_groups)

        for g_idx, (lora_dir, group) in enumerate(lora_groups.items()):
            indices, prompts = zip(*group)
            lr = lora_requests[lora_dir]
            outputs = llm.generate(list(prompts), sampling_params, lora_request=lr)
            for idx, out in zip(indices, outputs):
                results[idx] = out.outputs[0].text.strip()
            n_done += len(group)
            if (g_idx + 1) % 20 == 0 or (g_idx + 1) == n_groups:
                elapsed = time.perf_counter() - t0
                print(
                    f"    Group {g_idx + 1}/{n_groups}  "
                    f"({n_done}/{len(remaining)} samples, {elapsed:.1f}s)",
                    flush=True,
                )

        elapsed = time.perf_counter() - t0
        with open(pred_path, "a", encoding="utf-8") as f:
            for i, s in enumerate(remaining):
                record = {
                    "question_id": s["question_id"],
                    "role": s["role"],
                    "prediction": results[i],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        speed = len(remaining) / elapsed if elapsed > 0 else 0
        print(f"    Done: {len(remaining)} samples in {elapsed:.1f}s "
              f"({speed:.1f} samples/s)", flush=True)

        # Save meta for this task
        meta = {
            "mode": "hypernet_lora_multi",
            "model": inference_model_dir,
            "checkpoint": args.checkpoint,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "tensor_parallel": args.tensor_parallel,
            "total_samples": len(td["samples"]),
            "predicted_this_run": len(remaining),
            "data_path": tc["data"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "lora_generation": gen_stats,
            "inference": {
                "model_dir": inference_model_dir,
                "total_seconds": round(elapsed, 2),
                "num_predicted": len(remaining),
                "mean_ms_per_sample": round(elapsed / len(remaining) * 1000, 1)
                if remaining else 0,
                "samples_per_second": round(speed, 2),
            },
        }
        meta_path = os.path.join(tc["output_dir"], "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # Cleanup
    del llm
    torch.cuda.empty_cache()
    gc.collect()

    for tmp in [
        lora_base_dir,
        os.path.join(os.path.dirname(tasks[0]["output_dir"]),
                     "_shared_temp_loras"),
    ]:
        if os.path.exists(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n  Temp LoRAs cleaned up (ramdisk + AFS)", flush=True)

    total_elapsed = time.perf_counter() - t_total_start
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Multi-task pipeline complete: {len(tasks)} tasks in "
          f"{total_elapsed:.1f}s", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    if "--multi" in sys.argv:
        multi_main()
    else:
        main()
