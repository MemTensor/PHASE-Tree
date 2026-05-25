"""Utility package for the hyper_llm_modulator.

Re-exports commonly used functions from sub-modules so that downstream
code can import directly from ``hyper_llm_modulator.utils``.
"""

import json
import logging
import os
from math import sqrt

import numpy as np
import torch
import yaml

from .model_loading import (
    get_model,
    get_model_and_tokenizer,
    get_tokenizer,
    get_peft_config,
)
from .task_metadata import get_metadata_for_task, get_metadata
from .preprocessing import (
    get_preprocessing_fn,
    add_full_stop,
    get_prompt_formatting_fn,
    apply_sfr_template,
)
from .lora_formatting import (
    convert_qkv_gate_up_lora_to_splits_vllm,
    lora_state_dict_to_tensor_dict,
    get_lora_module_names,
    save_lora_from_peft_model,
    get_target_lora_dirs,
    lora_tensor_dict_to_state_dict,
    get_mean_lora,
    get_std_lora,
)
from .pooling import get_pooling_fn
from .utils import embed_texts, embed_tokens, get_inp_tokenize_fn, log_scalar


def repeat_iterator(iterable):
    # infinitely repeat the iterator
    while True:
        yield from iterable


def get_layers(model):
    if hasattr(model, "model"):
        return get_layers(model.model)
    return model.layers


def get_num_params(model):
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    return total_params, trainable_params


def compute_scaling_factor(peft_config):
    if (peft_config is None) or (not hasattr(peft_config, "lora_alpha")):
        return 1.0
    scaling = peft_config.lora_alpha / peft_config.r

    if getattr(peft_config, "use_rslora", False):
        scaling *= sqrt(peft_config.r)

    return scaling


def create_logger(log_dir, debug=False):
    """Create a global logger that logs INFO level messages to stdout and DEBUG ones to debug.log"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger()
    log_formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream_level = logging.DEBUG if debug else logging.INFO
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(stream_level)
    logger.addHandler(stream_handler)

    log_path = f"{log_dir}/debug.log"
    debug_handler = logging.FileHandler(log_path, delay=True)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(log_formatter)
    logger.addHandler(debug_handler)
    logger.setLevel(logging.DEBUG)
    logger.info(f"Logging to: {log_path}")
    return logger


def save_yaml(data, path):
    with open(path, "w") as file:
        yaml.dump(data, file)


def save_json(data, path):
    with open(path, "w") as file:
        json.dump(data, file, indent=4)


def generate_simplex_points(n_points, dimension):
    # Generate points from a Dirichlet distribution
    points = np.random.dirichlet(np.ones(dimension), size=n_points)

    return torch.tensor(points, dtype=torch.float)


def get_end_points(dimension):
    return torch.eye(dimension, dtype=torch.float)
