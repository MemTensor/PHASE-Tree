"""Training argument dataclass and YAML-aware argument parser.

Adapted from the HuggingFace alignment-handbook configuration pattern.
Provides :class:`TrainingArguments` — a single dataclass that captures all
hyperparameters for hypermod SFT training — and :class:`ArgumentParser` that
merges a YAML config with CLI overrides into a typed namespace.

Reference
    https://github.com/huggingface/alignment-handbook/blob/main/src/alignment/configs.py
"""

import dataclasses
import os
import sys
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, NewType, Optional, Tuple

from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


DataClassType = NewType("DataClassType", Any)


class ArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, other_args: Optional[List[str]] = None) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys
                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type == bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    if base_type == dict:
                        inputs[arg] = yaml.load(val, Loader=yaml.FullLoader)

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")
                else:
                    raise ValueError(f"Argument provided not found in dataclass: {arg}")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self) -> DataClassType | Tuple[DataClassType]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1].split("=")[-1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(os.path.abspath(sys.argv[1].split("=")[-1]), sys.argv[2:])
        # parse --config for the yaml path and other command line args
        elif any([arg.startswith("--config") for arg in sys.argv]):
            yaml_arg = [arg for arg in sys.argv[1:] if arg.startswith("--config") and arg.endswith(".yaml")][0]
            other_args = [arg for arg in sys.argv[1:] if arg != yaml_arg]
            output = self.parse_yaml_and_args(os.path.abspath(yaml_arg.split("=")[-1]), other_args)
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output


@dataclass
class TrainingArguments:
    config: str = field(default=None, metadata={"help": "The config file."})
    training_task: Literal["sft", "recon"] = field(default="sft", metadata={"help": "SFT vs reconstruction training."})
    model_dir: str = field(default=None, metadata={"help": "The model directory."})
    emb_model: str = field(default="", metadata={"help": "The embedding model."})
    
    # API-based embedding configuration
    use_api_embedding: bool = field(
        default=False, 
        metadata={"help": "Whether to use API-based embedding instead of local embedding models."}
    )
    openai_api_key: str = field(
        default="", 
        metadata={"help": "OpenAI API key for embedding generation. Can also be set via OPENAI_API_KEY environment variable."}
    )
    openai_embedding_model: str = field(
        default="text-embedding-3-large", 
        metadata={"help": "OpenAI embedding model to use (e.g., text-embedding-3-large, text-embedding-ada-002)."}
    )
    openai_api_base: str = field(
        default="", 
        metadata={"help": "Custom OpenAI API base URL. Leave empty to use default OpenAI API."}
    )
    vllm_api_base: str = field(
        default="http://localhost:8000", 
        metadata={"help": "Base URL for vLLM API server (e.g., http://localhost:8000)."}
    )
    vllm_embedding_model: str = field(
        default="", 
        metadata={"help": "Model name for vLLM embedding API. Should match the model served by vLLM."}
    )
    vllm_api_key: str = field(
        default="", 
        metadata={"help": "API key for vLLM embedding API. Can also be set via environment variable."}
    )
    api_embedding_batch_size: int = field(
        default=100, 
        metadata={"help": "Batch size for API-based embedding generation."}
    )
    api_embedding_timeout: int = field(
        default=60, 
        metadata={"help": "Timeout in seconds for API embedding requests."}
    )
    api_embedding_max_retries: int = field(
        default=3, 
        metadata={"help": "Maximum number of retries for failed API embedding requests."}
    )
    max_tokens_per_profile: int = field(
        default=26500,
        metadata={"help": "Maximum number of tokens per individual text for API-based embedding generation. Texts exceeding this limit will be truncated."}
    )
    
    exp_setup: Literal["lora", "vera", "hyper_lora", "hyper_vera"] = field(
        default=None, metadata={"help": "The finetuning setup."}
    )
    sft_mode: Literal["causal_lm", "completion"] = field(
        default=None,
        metadata={
            "help": "causal_lm trains on both prompts and responses while completion trains with only responses"
        },
    )
    equally_weight_sample: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to equally weight the samples in the dataset, "
                "useful for training on multiple datasets with different average prompt lengths"
            )
        },
    )
    train_ds_names: List[str] = field(default=None, metadata={"help": "The list of dataset names"})
    n_train_ds: int = field(default=None, metadata={"help": "The number of training datasets."})
    n_descs_per_ds: int = field(default=None, metadata={"help": "The number of descriptions per dataset."})
    train_data_proportion: float = field(
        default=1.0,
        metadata={"help": "Proportion of the training data to use during training. Must be in (0, 1]."},
    )
    inp_max_len: int = field(default=1024, metadata={"help": "The maximum input length."})
    target_modules: List[str] = field(default=None, metadata={"help": "The target modules for training."})
    shared_AB_head: bool = field(
        default=False,
        metadata={"help": "Whether to share the A and B heads in the HyperLoRA model."},
    )
    autoreg_gen: bool = field(
        default=False,
        metadata={"help": "Whether to use autoregressive generation in the HyperLoRA model."},
    )
    learnable_pos_emb: bool = field(
        default=False,
        metadata={
            "help": "Whether to use learnable positional embeddings in the HyperLoRA model. "
            "Can only be used when autoreg_gen is True."
        },
    )
    learnable_AB_offset: bool = field(
        default=False,
        metadata={"help": "Whether to use learnable A and B offsets in the HyperLoRA model."},
    )
    hypernet_latent_size: int = field(
        default=128,
        metadata={"help": "The latent size of the hypernet in the HyperLoRA model."},
    )
    head_in_size: int = field(
        default=512,
        metadata={"help": "The size of the input to each head in the HyperLoRA model."},
    )
    head_use_bias: bool = field(
        default=False,
        metadata={"help": "Whether to use bias in the heads of the HyperLoRA model."},
    )
    use_per_task_emb: bool = field(
        default=True,
        metadata={"help": "Whether to use per dataset embeddings in the HyperLoRA model."},
    )
    use_one_hot_task_emb: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use one hot task embeddings. Enabling this will ignore task descriptions provided."
                "Can only be used when use_per_task_emb is True."
            )
        },
    )
    use_inp_as_desc: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use input as description. Enabling this will ignore task descriptions provided."
                "Can only be used when use_per_task_emb is False."
            )
        },
    )
    use_per_sample_desc: bool = field(default=False, metadata={"help": "Whether to use per sample descriptions."})
    use_default_desc: bool = field(default=False, metadata={"help": "Use built-in default task descriptions instead of any dataset-specific ones."})
    n_points_per_task: int = field(default=1, metadata={"help": "The number of points per task."})
    use_hierarchical_sampler: bool = field(default=False, metadata={"help": "Whether to use hierarchical sampling."})
    also_val_on_train: bool = field(default=False, metadata={"help": "Whether to validate on training data."})

    lr: float = field(default=1e-4, metadata={"help": "The learning rate."})
    l2_reg_generated_w: float = field(default=1e-3, metadata={"help": "L2 regularization of the generated weights"})
    weight_decay: float = field(default=1e-3, metadata={"help": "The weight decay."})
    label_smoothing: float = field(default=0.1, metadata={"help": "The label smoothing factor."})
    grad_accum_steps: int = field(default=1, metadata={"help": "The number of gradient accumulation steps."})
    epochs: int = field(default=20, metadata={"help": "The number of epochs."})
    batch_size: int = field(default=8, metadata={"help": "The batch size."})
    val_batch_size: int = field(default=64, metadata={"help": "The evaluation batch size."})
    warmup_frac: float = field(default=0.2, metadata={"help": "The fraction of warmup steps."})
    neftune_noise_alpha: float = field(default=5, metadata={"help": "The noise alpha for NEFTune."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "The maximum gradient norm."})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Whether to enable gradient checkpointing to save memory."})
    logging_freq: int = field(default=100, metadata={"help": "The wandb logging frequency."})
    val_freq: int = field(default=10000, metadata={"help": "The validation frequency."})
    model_watch_freq: int = field(default=5000, metadata={"help": "The model watching frequency."})
    save_freq: int = field(default=10**100, metadata={"help": "The saving and gradient/weight logging frequency."})
    seed: int = field(default=42, metadata={"help": "The random seed."})
    debug: bool = field(default=False, metadata={"help": "Whether to run in debug mode."})
    notes: str = field(default=None, metadata={"help": "wandb note."})
    keep_only_best: bool = field(default=False, metadata={"help": "Whether or not to delete intermediate checkpoints."})
    # Track top-k checkpoints by validation loss
    top_k_checkpoints: int = field(
        default=3,
        metadata={"help": "Number of top checkpoints (lowest validation loss) to keep track of and evaluate. Set to 0 to disable."},
    )
    
    # Early stopping parameters
    use_early_stopping: bool = field(default=True, metadata={"help": "Whether to use early stopping based on validation loss."})
    early_stopping_patience: int = field(default=5, metadata={"help": "Number of validation checks with no improvement after which training will be stopped."})
    early_stopping_min_delta: float = field(default=0.0, metadata={"help": "Minimum change in validation loss to qualify as an improvement."})

    skip_eval: bool = field(default=False, metadata={"help": "Whether to skip evaluation."})
    skip_val: bool = field(default=False, metadata={"help": "Whether to skip validation during training."})
    eval_ds_info: dict = field(default=None, metadata={"help": "The datasets and their infomation for evaluation"})
    additional_eval_descs: List[str] = field(default=None, metadata={"help": "Additional evaluation descriptions."})
    save_to_base_model_dir: bool = field(
        default=False,
        metadata={"help": "Whether to save eval results to the base model directory (Used with normal LoRA only)."},
    )
    n_tasks_per_batch: int = field(
        default=8,
        metadata={"help": ("Number of tasks to sample per batch. Use lower number in case of OOM.")},
    )
    dataset_sampling_strategy: str = field(
        default="sqrt_size",
        metadata={
            "help": (
                "Strategy for sampling datasets in hierarchical mode to address dataset imbalance. "
                "Options: 'uniform' (equal probability), 'size' (proportional to size), "
                "'sqrt_size' (proportional to sqrt of size), 'inv_size' (inversely proportional to size)."
            )
        },
    )
    mt_lora_path: Optional[str] = field(default=None, metadata={"help": ("Path to the multi-task LoRA model.")})
    encoder_type: Literal["linear", "discrete", "vq", "softmax"] = field(
        default="linear", metadata={"help": ("Encoder type.")}
    )

    ## fusion args
    use_conv_fusion: bool = field(
        default=False, 
        metadata={"help": "Whether to use convolutional layers to fuse information across layer depths and types."}
    )
    conv_fusion_type: Literal["1d", "2d", "3d"] = field(
        default="1d", 
        metadata={"help": "Type of convolution for unified depth+type fusion: '1d' treats (depth,type) as sequence, '2d' treats depth and type as spatial dimensions, '3d' uses specialized depth+type convolution."}
    )
    conv_fusion_kernel_size: int = field(
        default=3, 
        metadata={"help": "Kernel size for convolutional fusion layers."}
    )
    conv_fusion_num_layers: int = field(
        default=2, 
        metadata={"help": "Number of convolutional layers for fusion."}
    )
    conv_fusion_channels: int = field(
        default=64, 
        metadata={"help": "Number of channels in convolutional fusion layers."}
    )
    conv_fusion_dropout: float = field(
        default=0.1, 
        metadata={"help": "Dropout rate for convolutional fusion layers."}
    )
    
    ## attention fusion args
    use_attention_fusion: bool = field(
        default=False, 
        metadata={"help": "Whether to use attention layers to fuse information across layer depths and types."}
    )
    attention_fusion_type: Literal["self", "cross", "hierarchical"] = field(
        default="self", 
        metadata={"help": "Type of attention fusion: 'self' uses self-attention across all positions, 'cross' uses cross-attention between depth and type, 'hierarchical' combines both."}
    )
    attention_num_heads: int = field(
        default=8, 
        metadata={"help": "Number of attention heads for attention fusion."}
    )
    attention_dropout: float = field(
        default=0.1, 
        metadata={"help": "Dropout rate for attention fusion layers."}
    )
    attention_num_layers: int = field(
        default=2, 
        metadata={"help": "Number of attention layers for hierarchical attention fusion."}
    )

    ## reconstruction training args
    n_embs_per_sampled_task: Optional[int] = field(
        default=None, metadata={"help": ("Number of embeddings to sample per task.")}
    )
    pred_z_score: bool = field(default=True, metadata={"help": ("Whether to predict z-scores.")})
    factorized: bool = field(default=False, metadata={"help": ("Whether to use factorized outputs.")})
    delta_w_scaling: float = field(default=10000, metadata={"help": ("Delta w scaling.")})
    
    
    # Cleanup args
    delete_generated_loras_after_eval: bool = field(
        default=True,
        metadata={"help": "Whether to delete all generated LoRA parameters after evaluation to save disk space."}
    )

    # Warm-start from a pretrained hypermod checkpoint (PHASE-Tree extension)
    init_hypermod_from: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to a pretrained hypermod.pt file (e.g. "
                "PHASE-Tree/phase_tree_models/phase_tree_pretrained/hypermod.pt). "
                "If set, hypermod weights are loaded with strict=False after the "
                "module is constructed, before optimizer creation. "
                "Optimizer state, scheduler state, and step counter are NOT loaded — "
                "this is a 'fine-tune from weights' workflow, not a full training resume."
            )
        },
    )
    init_hypermod_strict: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, fail loudly when init_hypermod_from has missing or "
                "unexpected keys. Default False allows architecture drift "
                "(e.g. extra modules added after the checkpoint was saved)."
            )
        },
    )
    freeze_heads: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, freeze the hypermod output heads (the Linear layers "
                "that map latent features to LoRA A/B weights). Only the "
                "encoder, mixer, and MLP trunk remain trainable (~20%% of "
                "hypermod params). Useful when warm-starting from a pretrained "
                "checkpoint to preserve its LoRA generation capability while "
                "adapting the feature extraction layers."
            )
        },
    )
    resume_from_run: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to a previous run directory (e.g. phase_tree_models_full/sft/hyper_lora/<run_id>) "
                "to resume training from. Loads the latest hypermod checkpoint, "
                "restores optimizer & scheduler state, and skips completed steps. "
                "The run directory must contain checkpoints/ and a resume_state.pt file "
                "(saved automatically during training)."
            )
        },
    )
