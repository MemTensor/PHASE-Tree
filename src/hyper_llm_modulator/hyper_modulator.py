"""Hypernetwork architecture for generating per-task LoRA weights.

The *hyper-modulator* (hypermod) takes a profile/task embedding as input
and outputs low-rank adapter matrices (LoRA A and B) for every target
module in the frozen base LLM.  The architecture comprises:

* A task encoder (linear or MLP) that maps the embedding to a latent
  representation.
* An optional mixer that blends multiple task embeddings.
* Per-layer output heads that project the latent into LoRA weight
  tensors for ``q_proj``, ``v_proj``, and optionally other modules.

This module also contains utilities for saving/loading hypermod
checkpoints and for constructing the hypermod from a configuration.
"""

import argparse
import logging
import math
import os
from copy import deepcopy
from types import MethodType
from math import sqrt
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_config, load_peft_weights, PeftConfig, PeftModel
from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists
from safetensors.torch import save_file
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import yaml

from hyper_llm_modulator.utils import (
    get_layers,
    get_lora_module_names,
    lora_state_dict_to_tensor_dict,
    get_model_and_tokenizer,
    get_pooling_fn,
    add_full_stop,
    get_target_lora_dirs,
    lora_tensor_dict_to_state_dict,
    get_mean_lora,
    get_std_lora,
)
from hyper_llm_modulator.utils.model_loading import get_emb_model_and_fns

from functools import partial, reduce
from operator import mul

cumprod = partial(reduce, mul)

logger = logging.getLogger()

# taken from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    x = (x * cos) + (rotate_half(x) * sin)
    return x


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class TaskEncoder(nn.Module):
    def __init__(self, task_emb_size: int, encoded_task_emb_size: int):
        super().__init__()
        self.encoded_task_emb_size = encoded_task_emb_size
        self.mlp = nn.Sequential(
            nn.Linear(task_emb_size, encoded_task_emb_size),
            nn.LayerNorm(encoded_task_emb_size),
        )

    def get_one_hot_task_emb(
        self, num_tasks: int, task_idx: torch.Tensor
    ) -> torch.Tensor:
        return torch.eye(num_tasks, device=task_idx.device)[task_idx]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"encoded_task_emb": self.mlp(x)}


class DiscreteOneHotTaskEncoder(nn.Module):
    def __init__(self, task_emb_size: int, n_classes: int, n_embs: int):
        super().__init__()
        self.n_classes = n_classes
        self.n_embs = n_embs
        self.encoded_task_emb_size = n_classes * n_embs
        self.mlp = nn.Sequential(
            nn.Linear(task_emb_size, self.encoded_task_emb_size * 4),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(self.encoded_task_emb_size * 4, self.encoded_task_emb_size),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.mlp(x)
        logits = logits.reshape(*logits.shape[:-1], self.n_embs, self.n_classes)
        if self.training:
            dist = torch.distributions.one_hot_categorical.OneHotCategorical(
                logits=logits
            )
            one_hot = dist.sample() + dist.probs - dist.probs.detach()
        else:
            one_hot = torch.eye(self.n_classes, device=x.device)[
                torch.argmax(logits, dim=-1)
            ]
        return {"encoded_task_emb": torch.flatten(one_hot, start_dim=-2, end_dim=-1)}


# similar to https://github.com/karpathy/deep-vector-quantization/blob/c3c026a1ccea369bc892ad6dde5e6d6cd5a508a4/dvq/model/quantize.py#L77
class SoftmaxTaskEncoder(nn.Module):
    def __init__(self, task_emb_size: int, code_dim: int, n_embs: int):
        super().__init__()
        self.n_embs = n_embs
        self.code_dim = code_dim
        self.mlp = nn.Sequential(
            nn.Linear(task_emb_size, code_dim * 4),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(code_dim * 4, n_embs),
        )
        self.embed = nn.Parameter(torch.randn(n_embs, code_dim), requires_grad=True)
        self.ln = nn.LayerNorm(code_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.mlp(x).reshape(-1, self.n_embs)  # [bs, n_embs]
        probs = F.softmax(logits, dim=-1)  # [bs, n_embs]
        emb = probs.unsqueeze(-1) * self.embed.unsqueeze(0)  # [bs, n_embs, code_dim]
        emb = emb.sum(1)  # [bs, code_dim]
        emb = self.ln(emb)
        return {"encoded_task_emb": emb}


# based on https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
class VQTaskEncoder(nn.Module):
    def __init__(
        self, task_emb_size: int, code_dim: int, n_embs: int, decay: float = 0.99
    ):
        super().__init__()
        self.n_embs = n_embs
        self.code_dim = code_dim
        self.decay = decay
        self.mlp = nn.Sequential(
            nn.Linear(task_emb_size, code_dim * 4),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(code_dim * 4, code_dim),
        )
        self.ln = nn.LayerNorm(code_dim)
        self.register_buffer("embed", torch.randn(code_dim, n_embs))
        self.register_buffer("cluster_size", torch.zeros(n_embs))
        self.register_buffer("embed_avg", self.embed.clone())

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z_e = self.mlp(x).reshape(-1, self.code_dim)  # [bs, code_dim]

        dist = (
            z_e.pow(2).sum(1, keepdim=True)
            - 2 * z_e @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        indices = torch.argmin(dist, dim=-1)  # [bs]
        one_hot = F.one_hot(indices, self.n_embs).to(z_e.dtype)  # [bs, n_embs]
        indices = indices.view(*x.shape[:-1])  # [bs]
        quantize = self.embed_code(indices)
        if self.training:
            # exponential moving average update for the codebook
            embed_onehot_sum = one_hot.sum(0)  # [n_embs]
            embed_sum = z_e.transpose(0, 1) @ one_hot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + 1e-5) / (n + self.n_embs * 1e-5) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        commitment_cost = 0.01
        diff = commitment_cost * (quantize.detach() - z_e).pow(2).mean()
        quantize = self.ln(z_e + (quantize - z_e).detach())
        return {"encoded_task_emb": quantize, "loss": diff}


class HyperConv3D(nn.Module):
    """3D convolution for unified depth+type fusion.
    
    Applies convolution across both layer depth and layer type dimensions simultaneously.
    """
    def __init__(self, input_dim: int, num_layers: int, num_types: int, 
                 kernel_size: int, hidden_channels: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_types = num_types
        
        # Layer normalization - only normalize over feature dimension for flexibility
        self.norm = nn.LayerNorm(input_dim)
        
        # Convolutional layers - treating depth and type as spatial dimensions
        self.conv_depth = nn.Conv2d(input_dim, hidden_channels, kernel_size, 1, "same", bias=False)
        self.conv_type = nn.Conv2d(input_dim, hidden_channels, kernel_size, 1, "same", bias=False)
        self.conv_depth2 = nn.Conv2d(hidden_channels, input_dim, kernel_size, 1, "same", bias=False)
        self.conv_type2 = nn.Conv2d(hidden_channels, input_dim, kernel_size, 1, "same", bias=False)
        
        # Bias parameter - per-feature bias for flexibility across different layer counts
        self.bias = nn.Parameter(torch.empty(input_dim))
        nn.init.uniform_(self.bias, -1.0 / self.bias.numel(), 1.0 / self.bias.numel())
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_layers, num_types, input_dim]
        Returns:
            fused_x: [batch_size, num_layers, num_types, input_dim]
        """
        batch_size = x.shape[0]
        original_x = x
        
        # Apply layer norm
        x = self.norm(x)
        
        # Process depth dimension first
        x_depth = x.permute(0, 3, 1, 2)  # [batch, input_dim, num_layers, num_types]
        x_depth = self.conv_depth(x_depth)  # [batch, hidden_channels, num_layers, num_types]
        x_depth = F.silu(x_depth)
        x_depth = self.conv_depth2(x_depth)  # [batch, input_dim, num_layers, num_types]
        x_depth = x_depth.permute(0, 2, 3, 1)  # [batch, num_layers, num_types, input_dim]
        
        # Process type dimension
        x_type = x.permute(0, 3, 2, 1)  # [batch, input_dim, num_types, num_layers]
        x_type = self.conv_type(x_type)  # [batch, hidden_channels, num_types, num_layers]
        x_type = F.silu(x_type)
        x_type = self.conv_type2(x_type)  # [batch, input_dim, num_types, num_layers]
        x_type = x_type.permute(0, 3, 2, 1)  # [batch, num_layers, num_types, input_dim]
        
        # Combine and add bias
        x_fused = (x_depth + x_type + self.bias) / 3
        
        # Residual connection
        return original_x + x_fused


class HyperConv1D(nn.Module):
    """1D convolution for unified depth+type fusion.
    
    Treats (depth, type) as a flattened sequence dimension.
    """
    def __init__(self, input_dim: int, num_layers: int, num_types: int,
                 kernel_size: int, num_conv_layers: int, hidden_channels: int, dropout: float):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_types = num_types
        self.seq_len = num_layers * num_types  # Store for reference, but calculate dynamically
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_channels)
        
        # 1D Convolutional layers
        conv_layers = []
        for i in range(num_conv_layers):
            conv_layers.extend([
                nn.Conv1d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=True
                ),
                nn.BatchNorm1d(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_layers, num_types, input_dim]
        Returns:
            fused_x: [batch_size, num_layers, num_types, input_dim]
        """
        batch_size, num_layers, num_types, input_dim = x.shape
        original_x = x
        
        # Calculate actual sequence length from input tensor dimensions
        actual_seq_len = num_layers * num_types
        
        # Flatten spatial dimensions: [batch_size, seq_len, input_dim]
        x = x.view(batch_size, actual_seq_len, input_dim)
        
        # Project to hidden channels
        x = self.input_proj(x)  # [batch_size, seq_len, hidden_channels]
        
        # Transpose for conv1d: [batch_size, hidden_channels, seq_len]
        x = x.transpose(1, 2)
        
        # Apply conv layers
        x = self.conv_layers(x)
        
        # Transpose back: [batch_size, seq_len, hidden_channels]
        x = x.transpose(1, 2)
        
        # Project back to input dimension
        x = self.output_proj(x)  # [batch_size, seq_len, input_dim]
        
        # Reshape back: [batch_size, num_layers, num_types, input_dim]
        x = x.view(batch_size, num_layers, num_types, input_dim)
        
        # Residual connection and layer norm
        x_fused = self.layer_norm(original_x + x)
        
        return x_fused


class HyperConv2D(nn.Module):
    """2D convolution for unified depth+type fusion.
    
    Treats depth and type as two spatial dimensions.
    """
    def __init__(self, input_dim: int, num_layers: int, num_types: int,
                 kernel_size: int, num_conv_layers: int, hidden_channels: int, dropout: float):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_types = num_types
        
        # Input projection to create channels
        self.input_proj = nn.Linear(input_dim, hidden_channels)
        
        # 2D Convolutional layers
        conv_layers = []
        in_channels = hidden_channels
        
        for i in range(num_conv_layers):
            out_channels = hidden_channels
            conv_layers.extend([
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=True
                ),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(),
                nn.Dropout2d(dropout)
            ])
            in_channels = out_channels
            
        # Final conv to reduce back to single channel equivalent
        conv_layers.append(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=1,
                kernel_size=1,
                bias=True
            )
        )
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Output projection back to original dimension
        self.output_proj = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_layers, num_types, input_dim]
        Returns:
            fused_x: [batch_size, num_layers, num_types, input_dim]
        """
        batch_size, num_layers, num_types, input_dim = x.shape
        original_x = x
        
        # Project to hidden channels: [batch_size, num_layers, num_types, hidden_channels]
        x = self.input_proj(x)
        
        # Treat feature dimension as channels: [batch_size, hidden_channels, num_layers, num_types]
        x = x.permute(0, 3, 1, 2)
        
        # Apply 2D convolutions
        x = self.conv_layers(x)  # [batch_size, 1, num_layers, num_types]
        
        # Reshape back: [batch_size, num_layers, num_types, 1]
        x = x.permute(0, 2, 3, 1)
        
        # Expand back to original feature dimension
        x = x.expand(-1, -1, -1, input_dim)
        
        # Apply output projection
        x = self.output_proj(x)
        
        # Residual connection and layer norm
        x_fused = self.layer_norm(original_x + x)
        
        return x_fused


class HyperAttentionFusion(nn.Module):
    """Self-attention based fusion for unified depth+type aggregation.
    
    Uses multi-head self-attention to capture dependencies across both 
    layer depth and layer type dimensions simultaneously.
    """
    def __init__(self, input_dim: int, num_layers: int, num_types: int,
                 num_heads: int = 8, dropout: float = 0.1, use_positional_encoding: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_types = num_types
        self.num_heads = num_heads
        self.seq_len = num_layers * num_types
        self.use_positional_encoding = use_positional_encoding
        
        # Ensure input_dim is divisible by num_heads
        assert input_dim % num_heads == 0, f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, input_dim)
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Positional encodings for depth and type
        if use_positional_encoding:
            self.depth_pos_encoding = nn.Parameter(torch.randn(num_layers, input_dim // 2))
            self.type_pos_encoding = nn.Parameter(torch.randn(num_types, input_dim // 2))
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_layers, num_types, input_dim]
        Returns:
            fused_x: [batch_size, num_layers, num_types, input_dim]
        """
        batch_size, num_layers, num_types, input_dim = x.shape
        original_x = x
        
        # Calculate actual sequence length from input tensor dimensions
        actual_seq_len = num_layers * num_types
        
        # Flatten spatial dimensions: [batch_size, seq_len, input_dim]
        x = x.view(batch_size, actual_seq_len, input_dim)
        
        # Add positional encodings if enabled
        if self.use_positional_encoding:
            # Create positional encodings for each (depth, type) combination
            pos_encodings = []
            for depth_idx in range(num_layers):
                for type_idx in range(num_types):
                    # Handle case where input dimensions exceed initialization dimensions
                    depth_pos_idx = min(depth_idx, self.depth_pos_encoding.size(0) - 1)
                    type_pos_idx = min(type_idx, self.type_pos_encoding.size(0) - 1)
                    
                    depth_pos = self.depth_pos_encoding[depth_pos_idx]  # [input_dim // 2]
                    type_pos = self.type_pos_encoding[type_pos_idx]     # [input_dim // 2]
                    combined_pos = torch.cat([depth_pos, type_pos], dim=0)  # [input_dim]
                    pos_encodings.append(combined_pos)
            
            pos_encodings = torch.stack(pos_encodings, dim=0)  # [actual_seq_len, input_dim]
            pos_encodings = pos_encodings.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, actual_seq_len, input_dim]
            x = x + pos_encodings
        
        # Input projection
        x = self.input_proj(x)
        
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attention(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # Feed-forward network with residual connection
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        # Reshape back: [batch_size, num_layers, num_types, input_dim]
        x = x.view(batch_size, num_layers, num_types, input_dim)
        
        return x


class HyperCrossAttentionFusion(nn.Module):
    """Cross-attention based fusion between depth and type dimensions.
    
    Uses cross-attention to model interactions between layer depths and types,
    allowing each dimension to attend to the other.
    """
    def __init__(self, input_dim: int, num_layers: int, num_types: int,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_types = num_types
        self.num_heads = num_heads
        
        # Ensure input_dim is divisible by num_heads
        assert input_dim % num_heads == 0, f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
        
        # Cross-attention modules
        # Depth attending to type
        self.depth_to_type_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Type attending to depth
        self.type_to_depth_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalizations
        self.norm_depth = nn.LayerNorm(input_dim)
        self.norm_type = nn.LayerNorm(input_dim)
        self.norm_output = nn.LayerNorm(input_dim)
        
        # Output projection
        self.output_proj = nn.Linear(input_dim * 2, input_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_layers, num_types, input_dim]
        Returns:
            fused_x: [batch_size, num_layers, num_types, input_dim]
        """
        batch_size, num_layers, num_types, input_dim = x.shape
        original_x = x
        
        # Aggregate across type dimension for depth queries: [batch_size, num_layers, input_dim]
        depth_features = x.mean(dim=2)
        
        # Aggregate across depth dimension for type queries: [batch_size, num_types, input_dim]
        type_features = x.mean(dim=1)
        
        # Cross-attention: depth attending to type
        depth_norm = self.norm_depth(depth_features)
        type_norm = self.norm_type(type_features)
        
        # Depth queries attend to type keys/values
        depth_attended, _ = self.depth_to_type_attention(
            depth_norm, type_norm, type_norm
        )  # [batch_size, num_layers, input_dim]
        
        # Type queries attend to depth keys/values  
        type_attended, _ = self.type_to_depth_attention(
            type_norm, depth_norm, depth_norm
        )  # [batch_size, num_types, input_dim]
        
        # Broadcast attended features back to original shape
        depth_attended = depth_attended.unsqueeze(2).expand(-1, -1, num_types, -1)  # [batch_size, num_layers, num_types, input_dim]
        type_attended = type_attended.unsqueeze(1).expand(-1, num_layers, -1, -1)   # [batch_size, num_layers, num_types, input_dim]
        
        # Combine attended features
        combined = torch.cat([depth_attended, type_attended], dim=-1)  # [batch_size, num_layers, num_types, input_dim * 2]
        
        # Project back to original dimension
        fused = self.output_proj(combined)  # [batch_size, num_layers, num_types, input_dim]
        
        # Residual connection and normalization
        fused = self.norm_output(original_x + fused)
        
        return fused


class HyperHierarchicalAttentionFusion(nn.Module):
    """Hierarchical attention fusion combining self-attention and cross-attention.
    
    First applies self-attention within each dimension, then cross-attention between dimensions.
    """
    def __init__(self, input_dim: int, num_layers: int, num_types: int,
                 num_heads: int = 8, dropout: float = 0.1, num_attention_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_types = num_types
        self.num_attention_layers = num_attention_layers
        
        # Multiple layers of self-attention
        self.self_attention_layers = nn.ModuleList([
            HyperAttentionFusion(
                input_dim=input_dim,
                num_layers=num_layers,
                num_types=num_types,
                num_heads=num_heads,
                dropout=dropout,
                use_positional_encoding=(i == 0)  # Only first layer uses positional encoding
            ) for i in range(num_attention_layers)
        ])
        
        # Cross-attention layer
        self.cross_attention = HyperCrossAttentionFusion(
            input_dim=input_dim,
            num_layers=num_layers,
            num_types=num_types,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_layers, num_types, input_dim]
        Returns:
            fused_x: [batch_size, num_layers, num_types, input_dim]
        """
        # Apply multiple self-attention layers
        for self_attn_layer in self.self_attention_layers:
            x = self_attn_layer(x)
        
        # Apply cross-attention
        x = self.cross_attention(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x


class MLPResidualBlock(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, pre_layer_norm, post_dropout
    ):
        super().__init__()
        layers = []
        if pre_layer_norm:
            layers.append(nn.LayerNorm(input_size))
        layers += [
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_size, output_size),
            nn.SiLU(),
        ]
        if post_dropout:
            layers.append(nn.Dropout(0.05))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        mlp_output = self.mlp(x)
        # Only add residual connection if input and output dimensions match
        if x.shape == mlp_output.shape:
            return x + mlp_output
        else:
            # If dimensions don't match, return MLP output without residual connection
            return mlp_output


def zero_lora_param_dict(target_modules, n_layers, rank, in_features, out_features):
    return nn.ParameterDict(
        {
            "A": nn.ParameterDict(
                {
                    m: nn.Parameter(
                        torch.zeros(n_layers, rank, in_features[m]), requires_grad=False
                    )
                    for m in target_modules
                }
            ),
            "B": nn.ParameterDict(
                {
                    m: nn.Parameter(
                        torch.zeros(n_layers, out_features[m], rank),
                        requires_grad=False,
                    )
                    for m in target_modules
                }
            ),
        }
    )


def lora_tensor_dict_to_param_dict(lora_tensor_dict, requires_grad):
    return nn.ParameterDict(
        {
            "A": nn.ParameterDict(
                {
                    k: nn.Parameter(v, requires_grad)
                    for k, v in lora_tensor_dict["A"].items()
                }
            ),
            "B": nn.ParameterDict(
                {
                    k: nn.Parameter(v, requires_grad)
                    for k, v in lora_tensor_dict["B"].items()
                }
            ),
        }
    )


class HyperModulator(nn.Module):
    def __init__(
        self,
        model: PeftModel,
        output_space: str,
        module_names: list[str],
        training_task: Literal["sft", "recon"] = "sft",
        pred_z_score: Optional[bool] = None,
        mean_recon_target: Optional[dict[str, torch.Tensor]] = None,
        std_recon_target: Optional[dict[str, torch.Tensor]] = None,
        match_lora_init: bool = False,
        task_emb_size: Optional[int] = None,
        shared_AB_head: bool = False,
        autoreg_gen: bool = False,
        learnable_pos_emb: bool = False,
        encoder_type: Literal["linear", "discrete", "vq", "softmax"] = "linear",
        AB_offset: Optional[dict[str, dict[str, torch.Tensor]]] = None,
        learnable_AB_offset: bool = False,
        zero_init_head: bool = False,
        latent_size: int = 128,
        head_in_size: int = 512,
        head_use_bias: bool = False,
        factorized: bool = False,
        delta_w_scaling: float = 10000,
        dtype: torch.dtype = torch.float32,
        # Fusion parameters
        use_conv_fusion: bool = False,
        conv_fusion_type: Literal["1d", "2d", "3d"] = "1d",
        conv_fusion_kernel_size: int = 3,
        conv_fusion_num_layers: int = 2,
        conv_fusion_channels: int = 64,
        conv_fusion_dropout: float = 0.1,
        # Attention fusion parameters
        use_attention_fusion: bool = False,
        attention_fusion_type: Literal["self", "cross", "hierarchical"] = "self",
        attention_num_heads: int = 8,
        attention_dropout: float = 0.1,
        attention_num_layers: int = 2,
    ):
        assert output_space == "lora", f"Invalid output space: {output_space}"
        assert (not shared_AB_head) or output_space == "lora", (
            "shared_AB_head is only supported for lora output space"
        )
        assert (not autoreg_gen) or output_space == "lora", (
            "autoreg_gen is only supported for shared_AB_head and lora output space"
        )
        assert (not learnable_pos_emb) or autoreg_gen, (
            "learnable_pos_emb is only supported when autoreg_gen is True"
        )
        assert not (training_task == "recon" and AB_offset), (
            "AB_offset is not supported for recon training"
        )
        assert not (use_conv_fusion and use_attention_fusion), (
            "Cannot use both convolution and attention fusion simultaneously"
        )

        super().__init__()
        self.model_config = model_config = model.config
        self.peft_config = peft_config = model.peft_config["default"]
        self.training_task = training_task
        self.pred_z_score = pred_z_score
        self.module_names = module_names
        self.shared_AB_head = shared_AB_head
        self.autoreg_gen = autoreg_gen
        self.learnable_pos_emb = learnable_pos_emb
        self.scaling = (
            peft_config.lora_alpha / peft_config.r if output_space == "lora" else 1
        )
        if getattr(peft_config, "use_rslora", False):
            self.scaling *= math.sqrt(peft_config.r)
        self.output_space = output_space
        self.max_num_layers = model_config.num_hidden_layers

        self.target_modules = peft_config.target_modules
        self.module_to_int = {m: i for i, m in enumerate(self.target_modules)}
        self.task_to_int = dict()

        self.factorized = factorized
        self.delta_w_scaling = delta_w_scaling
        self.device = device = model.device
        self.dtype = dtype

        # Fusion parameters
        self.use_conv_fusion = use_conv_fusion
        self.conv_fusion_type = conv_fusion_type
        self.conv_fusion_kernel_size = conv_fusion_kernel_size
        self.conv_fusion_num_layers = conv_fusion_num_layers
        self.conv_fusion_channels = conv_fusion_channels
        self.conv_fusion_dropout = conv_fusion_dropout
        
        # Attention fusion parameters
        self.use_attention_fusion = use_attention_fusion
        self.attention_fusion_type = attention_fusion_type
        self.attention_num_heads = attention_num_heads
        self.attention_dropout = attention_dropout
        self.attention_num_layers = attention_num_layers

        self.in_features, self.out_features = get_in_out_features(model, peft_config)
        n_layers = self.max_num_layers
        self.AB_offset = zero_lora_param_dict(
            self.target_modules,
            n_layers,
            peft_config.r,
            self.in_features,
            self.out_features,
        )
        self.mean_recon_target = zero_lora_param_dict(
            self.target_modules,
            n_layers,
            peft_config.r,
            self.in_features,
            self.out_features,
        )
        self.std_recon_target = zero_lora_param_dict(
            self.target_modules,
            n_layers,
            peft_config.r,
            self.in_features,
            self.out_features,
        )
        if AB_offset is not None:
            self.AB_offset = lora_tensor_dict_to_param_dict(
                AB_offset, requires_grad=learnable_AB_offset
            )
        if mean_recon_target is not None:
            self.mean_recon_target = lora_tensor_dict_to_param_dict(
                mean_recon_target, requires_grad=False
            )
        if std_recon_target is not None:
            self.std_recon_target = lora_tensor_dict_to_param_dict(
                std_recon_target, requires_grad=False
            )

        # the input and output features for each target module (i.e. layer type)

        encoded_task_emb_size = 0

        if task_emb_size is not None:
            encoded_task_emb_size = latent_size // 2
            self.task_encoder = {
                "linear": TaskEncoder(task_emb_size, encoded_task_emb_size),
                "discrete": DiscreteOneHotTaskEncoder(
                    task_emb_size, n_classes=64, n_embs=1
                ),
                "vq": VQTaskEncoder(
                    task_emb_size, code_dim=encoded_task_emb_size, n_embs=32
                ),
                "softmax": SoftmaxTaskEncoder(
                    task_emb_size, code_dim=encoded_task_emb_size, n_embs=32
                ),
            }[encoder_type].to(device)
        if encoder_type == "discrete":
            depth_emb_size = self.max_num_layers
            type_emb_size = len(self.target_modules)
            self.layer_depth_encoder = lambda x: F.one_hot(
                x, num_classes=self.max_num_layers
            ).to(dtype)
            self.layer_type_encoder = lambda x: F.one_hot(
                x, num_classes=len(self.target_modules)
            ).to(dtype)
        else:
            depth_emb_size = latent_size // 4
            type_emb_size = latent_size // 4
            self.layer_depth_encoder = nn.Sequential(
                nn.Embedding(self.max_num_layers, depth_emb_size),
                nn.LayerNorm(depth_emb_size),
            )
            self.layer_type_encoder = nn.Sequential(
                nn.Embedding(len(self.target_modules), type_emb_size),
                nn.LayerNorm(type_emb_size),
            )

        mlp_inp_size = depth_emb_size + type_emb_size + encoded_task_emb_size

        self.mixer = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear(mlp_inp_size, mlp_inp_size * 4),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(mlp_inp_size * 4, mlp_inp_size),
            nn.SiLU(),
            nn.Dropout(0.05),
        )
        # MLP1 configuration - always uses the same input size
        # When fusion is used, we pass only the fused features to MLP1
        # When fusion is not used, we pass the mixed embeddings to MLP1
        mlp1_input_size = mlp_inp_size
            
        self.mlp1 = MLPResidualBlock(
            mlp1_input_size,
            mlp1_input_size * 4,  # Hidden size proportional to input
            mlp_inp_size,  # Output back to original size
            pre_layer_norm=True,
            post_dropout=True,
        )
        self.mlp2 = MLPResidualBlock(
            mlp_inp_size,
            mlp_inp_size * 4,
            mlp_inp_size,
            pre_layer_norm=True,
            post_dropout=True,
        )

        self.mlp3 = nn.Sequential(
            nn.LayerNorm(mlp_inp_size),
            nn.Linear(mlp_inp_size, mlp_inp_size * 4),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(mlp_inp_size * 4, head_in_size),
            nn.SiLU(),
        )

        if autoreg_gen:
            if learnable_pos_emb:
                self.pos_emb = nn.Sequential(
                    nn.Embedding(peft_config.r, mlp_inp_size),
                    nn.LayerNorm(mlp_inp_size),
                )
            else:
                self.rotary_emb = LlamaRotaryEmbedding(
                    dim=mlp_inp_size,
                    max_position_embeddings=64,  # model_config.hidden_size,
                    device=device,
                )

        # multiple heads, each for a different target module
        if shared_AB_head:
            self.out_emb = nn.ParameterDict(
                (
                    m,
                    nn.ParameterDict(
                        dict(
                            # same init as the embedding layer
                            A=nn.Parameter(
                                torch.normal(0, 1, size=(mlp_inp_size,)),
                                requires_grad=True,
                            ),
                            B=nn.Parameter(
                                torch.normal(0, 1, size=(mlp_inp_size,)),
                                requires_grad=True,
                            ),
                        )
                    ),
                )
                for m in self.target_modules
            )
            self.out_emb_norm = nn.LayerNorm(mlp_inp_size)

        heads = []
        for module in self.target_modules:
            in_features = self.in_features[module]
            out_features = self.out_features[module]
            if not shared_AB_head:
                output_size = (
                    peft_config.r + out_features
                    if output_space == "vera"
                    else (peft_config.r * (in_features + out_features))
                )
                if autoreg_gen:
                    output_size = in_features + out_features

            else:
                output_size = peft_config.r * max(in_features, out_features)
                if autoreg_gen:
                    output_size = max(in_features, out_features)

            logger.debug(f"head_in_size: {head_in_size}, output_size: {output_size}")
            logger.debug(f"head_use_bias: {head_use_bias}")
            layer = nn.Linear(
                head_in_size, output_size, bias=head_use_bias, device=device
            )
            if zero_init_head:
                logger.debug(f"zeroing out head weights for {module}")
                nn.init.zeros_(layer.weight)

            heads.append((module, layer))
        self.heads = nn.ModuleDict(heads)
        peft_weights = get_init_peft_weights(model, peft_config)
        logger.debug(f"peft_weights:\n{peft_weights}")

        self.split_shapes = dict()
        # we use the trainable peft weights of the base peft model
        # to initialize the bias of the heads
        for module_name, head in self.heads.items():
            if output_space == "lora":
                A = peft_weights[module_name]["lora_A"].weight.clone().flatten()
                B = peft_weights[module_name]["lora_B"].weight.clone().flatten()
                init_bias = [A, B]

            elif output_space == "vera":
                # d init with a constant value (0.1) and b with zeros
                # for vera, we're using the outputs of the hypernet as masking + scaling values
                d = peft_weights[module_name]["vera_lambda_d"].clone().flatten()  # 0.1
                b = peft_weights[module_name]["vera_lambda_b"].clone().flatten()  # 0
                init_bias = [d, b]

                # we just need one copy of A and B
                vera_A_B_learnable = True
                self.vera_A = nn.Parameter(
                    peft_weights[module_name]["vera_A"],
                    requires_grad=vera_A_B_learnable,
                )
                self.vera_B = nn.Parameter(
                    peft_weights[module_name]["vera_B"],
                    requires_grad=vera_A_B_learnable,
                )

            with torch.no_grad():
                if match_lora_init and head.bias is not None:
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)
                    if (not shared_AB_head) and (not autoreg_gen):
                        # based on https://proceedings.mlr.press/v205/beck23a/beck23a.pdf (bias-HyperInit)
                        # this encourages the hnet to initially use some shared values of the output
                        # also an easy way to match initialization of the corresponding method!
                        head.bias.copy_(torch.cat(init_bias))

                    # match the gradient scale to the full architecture
                    elif shared_AB_head:
                        if not autoreg_gen:
                            head.bias.copy_(init_bias[0] / sqrt(2))
                        else:
                            size = head.bias.shape[0]
                            head.bias.copy_(
                                init_bias[0][:size] / sqrt(2) / sqrt(peft_config.r)
                            )
                    else:
                        size = head.bias.shape[0]
                        head.bias.copy_(init_bias[0][:size] / sqrt(peft_config.r))

                # NOTE: how we slice the weights have to be consistent with the forward pass
                self.split_shapes[module_name] = [len(x) for x in init_bias]
                logger.debug(f"split_shapes: {self.split_shapes}")

        self.heads.to(device).to(dtype)
        del peft_weights

        # Initialize fusion modules if enabled
        if self.use_conv_fusion:
            if self.conv_fusion_type == "1d":
                self.fusion_module = HyperConv1D(
                    input_dim=mlp_inp_size,
                    num_layers=self.max_num_layers,
                    num_types=len(self.target_modules),
                    kernel_size=conv_fusion_kernel_size,
                    num_conv_layers=conv_fusion_num_layers,
                    hidden_channels=conv_fusion_channels,
                    dropout=conv_fusion_dropout,
                )
            elif self.conv_fusion_type == "2d":
                self.fusion_module = HyperConv2D(
                    input_dim=mlp_inp_size,
                    num_layers=self.max_num_layers,
                    num_types=len(self.target_modules),
                    kernel_size=conv_fusion_kernel_size,
                    num_conv_layers=conv_fusion_num_layers,
                    hidden_channels=conv_fusion_channels,
                    dropout=conv_fusion_dropout,
                )
            elif self.conv_fusion_type == "3d":
                self.fusion_module = HyperConv3D(
                    input_dim=mlp_inp_size,
                    num_layers=self.max_num_layers,
                    num_types=len(self.target_modules),
                    kernel_size=conv_fusion_kernel_size,
                    hidden_channels=conv_fusion_channels,
                )
            else:
                raise ValueError(f"Unknown conv_fusion_type: {conv_fusion_type}")
            self.fusion_module.to(device).to(dtype)
            
        elif self.use_attention_fusion:
            if self.attention_fusion_type == "self":
                self.fusion_module = HyperAttentionFusion(
                    input_dim=mlp_inp_size,
                    num_layers=self.max_num_layers,
                    num_types=len(self.target_modules),
                    num_heads=attention_num_heads,
                    dropout=attention_dropout,
                    use_positional_encoding=True,
                )
            elif self.attention_fusion_type == "cross":
                self.fusion_module = HyperCrossAttentionFusion(
                    input_dim=mlp_inp_size,
                    num_layers=self.max_num_layers,
                    num_types=len(self.target_modules),
                    num_heads=attention_num_heads,
                    dropout=attention_dropout,
                )
            elif self.attention_fusion_type == "hierarchical":
                self.fusion_module = HyperHierarchicalAttentionFusion(
                    input_dim=mlp_inp_size,
                    num_layers=self.max_num_layers,
                    num_types=len(self.target_modules),
                    num_heads=attention_num_heads,
                    dropout=attention_dropout,
                    num_attention_layers=attention_num_layers,
                )
            else:
                raise ValueError(f"Unknown attention_fusion_type: {attention_fusion_type}")
            self.fusion_module.to(device).to(dtype)

    def _embed_layer_depth(
        self, depth_indices: list[int] | int | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(depth_indices, int):
            depth_indices = torch.tensor(
                [depth_indices], dtype=torch.long, device=self.device
            )
        elif isinstance(depth_indices, list):
            depth_indices = torch.tensor(
                depth_indices, dtype=torch.long, device=self.device
            )

        return self.layer_depth_encoder(depth_indices)  # [bs (or 1), max_num_layers]

    def _embed_layer_type(self, layer_type: str) -> torch.Tensor:
        module_idx = self.module_to_int[layer_type]
        module_idx = torch.tensor([module_idx], dtype=torch.long, device=self.device)
        # we only forward one layer type at at time
        # so the shape is always [1, num_target_modules]
        return self.layer_type_encoder(module_idx)

    def get_one_hot_task_emb(
        self, num_tasks: int, task_idx: torch.Tensor
    ) -> torch.Tensor:
        return self.task_encoder.get_one_hot_task_emb(num_tasks, task_idx)

    def _hypernet_forward_all_types(self, layer_indices, encoded_task_emb):
        """Forward pass for all layer types simultaneously."""
        
        num_layers = len(layer_indices)
        num_types = len(self.target_modules)
        
        # Create all combinations of layer indices and types
        all_embeddings = []
        
        for layer_idx in layer_indices:
            for type_idx, layer_type in enumerate(self.target_modules):
                # Get embeddings for this specific (layer, type) combination
                depth_emb = self._embed_layer_depth([layer_idx])  # [1, depth_emb_size]
                type_emb = self._embed_layer_type(layer_type)  # [1, type_emb_size]
                
                if encoded_task_emb is None:
                    encoded_task_emb_expanded = torch.empty(0, device=self.device)
                else:
                    encoded_task_emb_expanded = encoded_task_emb[:1]  # [1, task_emb_size]
                
                # Concatenate embeddings
                cat_emb = torch.cat([encoded_task_emb_expanded, depth_emb, type_emb], dim=-1)
                all_embeddings.append(cat_emb)
        
        # Stack all embeddings: [num_layers * num_types, emb_dim]
        all_emb = torch.cat(all_embeddings, dim=0)
        
        # Apply mixer
        mlp_inp = self.mixer(all_emb)  # [num_layers * num_types, emb_dim]
        
        if self.use_conv_fusion or self.use_attention_fusion:
            # FUSION APPROACH: Use only the fused features
            
            # Reshape for fusion: [1, num_layers, num_types, emb_dim]
            mlp_inp_3d = mlp_inp.view(1, num_layers, num_types, mlp_inp.shape[-1])
            
            # Apply fusion (convolution or attention)
            mlp_inp_3d = self.fusion_module(mlp_inp_3d)  # [1, num_layers, num_types, emb_dim]
            
            # Flatten back: [num_layers * num_types, emb_dim]
            fused_features = mlp_inp_3d.view(-1, mlp_inp_3d.shape[-1])
            
            # Use only the fused features
            mlp1_input = fused_features  # [num_layers * num_types, emb_dim]
            
        else:
            # ORIGINAL APPROACH: Skip fusion, use only mixed embeddings
            mlp1_input = mlp_inp  # [num_layers * num_types, emb_dim]
        
        # Apply MLP layers
        mlp_out = self.mlp1(mlp1_input)  # [num_layers * num_types, emb_dim]
        
        # Reshape back to separate layers and types: [num_layers, num_types, emb_dim]
        mlp_out_3d = mlp_out.view(num_layers, num_types, mlp_out.shape[-1])
        
        return mlp_out_3d

    def _hypernet_forward(self, layer_indices, layer_type, encoded_task_emb):
        """Forward pass for a single layer type (backward compatibility)."""
        
        # Get the type index
        type_idx = self.module_to_int[layer_type]
        
        # Forward all types
        all_outputs = self._hypernet_forward_all_types(layer_indices, encoded_task_emb)
        
        # Extract outputs for the specific layer type: [num_layers, emb_dim]
        mlp_out = all_outputs[:, type_idx, :]  # [num_layers, emb_dim]
        
        bs = len(layer_indices)
        head = self.heads[layer_type]
        if not self.shared_AB_head:
            # head outputs both A and B
            if not self.autoreg_gen:
                # head outputs all the ranks at once
                head_out = head(self.mlp3(self.mlp2(mlp_out)))
                splitted_out = torch.split(
                    head_out, self.split_shapes[layer_type], dim=-1
                )
            else:
                # head outputs one rank at a time
                head_in = mlp_out.unsqueeze(1).expand(-1, self.peft_config.r, -1)
                if self.learnable_pos_emb:
                    head_in = self.mlp2(
                        head_in
                        + self.pos_emb(
                            torch.arange(self.peft_config.r, device=self.device)
                        ).unsqueeze(0)
                    )
                else:
                    cos, sin = self.rotary_emb(
                        head_in,
                        torch.arange(self.peft_config.r, device=self.device).unsqueeze(
                            0
                        ),
                    )  # [1, r, 2048]
                    head_in = self.mlp2(apply_rotary_pos_emb(head_in, cos, sin))
                head_out = head(self.mlp3(head_in))

                # [bs, self.peft_config.r, self.in_features[layer_type]]
                # [bs, self.peft_config.r, self.out_features[layer_type]]
                A, B = torch.split(
                    head_out,
                    (self.in_features[layer_type], self.out_features[layer_type]),
                    dim=-1,
                )
                A = A.reshape(bs, -1)
                B = B.reshape(bs, -1)
                splitted_out = [A, B]

        else:
            # head outputs either A or B
            splitted_out = []
            for out_emb, num_features in zip(
                self.out_emb[layer_type].values(),
                [self.in_features[layer_type], self.out_features[layer_type]],
            ):
                head_in = self.mlp2(mlp_out + self.out_emb_norm(out_emb))
                if not self.autoreg_gen:
                    # head outputs all the ranks at once
                    head_out = head(self.mlp3(head_in))
                    head_out = head_out.view(bs, self.peft_config.r, -1)
                    head_out = head_out[..., :num_features]
                    head_out = head_out.reshape(bs, self.peft_config.r * num_features)
                else:
                    # head outputs one rank at a time
                    head_in = head_in.unsqueeze(1).expand(-1, self.peft_config.r, -1)
                    if self.learnable_pos_emb:
                        pos_emb = self.mlp2(
                            self.pos_emb(
                                torch.arange(self.peft_config.r, device=self.device)
                            ).unsqueeze(0)
                        )
                        head_in = head_in + pos_emb
                    else:
                        cos, sin = self.rotary_emb(
                            head_in,
                            torch.arange(
                                self.peft_config.r, device=self.device
                            ).unsqueeze(0),
                        )  # [1, r, 2048]
                        head_in = self.mlp2(apply_rotary_pos_emb(head_in, cos, sin))

                    head_out = head(self.mlp3(head_in))
                    head_out = head_out[..., :num_features]
                    head_out = head_out.reshape(bs, self.peft_config.r * num_features)

                splitted_out.append(head_out)

        return splitted_out

    def get_delta_weights(
        self,
        layer_indices: torch.Tensor,
        layer_type: str,
        encoded_task_emb: torch.Tensor = None,
        factorized: Optional[bool] = None,
    ) -> torch.Tensor:
        if factorized is None:
            factorized = self.factorized
        bs = len(layer_indices)

        splitted_out = self._hypernet_forward(
            layer_indices,
            layer_type,
            encoded_task_emb,
        )

        if self.output_space == "lora":
            A, B = splitted_out
            A = A.reshape(
                bs, self.peft_config.r, self.in_features[layer_type]
            )  # .transpose(-1, -2)
            B = B.reshape(
                bs, self.peft_config.r, self.out_features[layer_type]
            ).transpose(-1, -2)
            if self.training_task == "sft":
                A = A + self.AB_offset["A"][layer_type][layer_indices]
                B = B + self.AB_offset["B"][layer_type][layer_indices]
            if factorized:
                return A, B
            deltaW = torch.bmm(B, A)

        elif self.output_space == "vera":
            raise NotImplementedError("Vera output space is deprecated")
        # the deltaW should have shape [bs, (len(layer_indices)), out_features, in_features]
        return deltaW

    @torch.no_grad()
    def gen_lora_all_types(self, layer_indices, encoded_task_emb):
        """Generate LoRA weights for all layer types in a single pass."""
        assert encoded_task_emb.shape[0] == 1, (
            "Only one task at a time is supported for now"
        )
        
        # Forward pass for all layer types simultaneously
        all_outputs = self._hypernet_forward_all_types(layer_indices, encoded_task_emb)
        # all_outputs shape: [num_layers, num_types, emb_dim]
        
        lora_A, lora_B = dict(), dict()
        
        for type_idx, target_module in enumerate(self.target_modules):
            # Extract outputs for this layer type: [num_layers, emb_dim]
            mlp_out = all_outputs[:, type_idx, :]
            
            head = self.heads[target_module]
            bs = len(layer_indices)
            
            if not self.shared_AB_head:
                # head outputs both A and B
                if not self.autoreg_gen:
                    # head outputs all the ranks at once
                    head_out = head(self.mlp3(self.mlp2(mlp_out)))
                    splitted_out = torch.split(
                        head_out, self.split_shapes[target_module], dim=-1
                    )
                else:
                    # head outputs one rank at a time
                    head_in = mlp_out.unsqueeze(1).expand(-1, self.peft_config.r, -1)
                    if self.learnable_pos_emb:
                        head_in = self.mlp2(
                            head_in
                            + self.pos_emb(
                                torch.arange(self.peft_config.r, device=self.device)
                            ).unsqueeze(0)
                        )
                    else:
                        cos, sin = self.rotary_emb(
                            head_in,
                            torch.arange(self.peft_config.r, device=self.device).unsqueeze(
                                0
                            ),
                        )  # [1, r, 2048]
                        head_in = self.mlp2(apply_rotary_pos_emb(head_in, cos, sin))
                    head_out = head(self.mlp3(head_in))

                    # [bs, self.peft_config.r, self.in_features[target_module]]
                    # [bs, self.peft_config.r, self.out_features[target_module]]
                    A, B = torch.split(
                        head_out,
                        (self.in_features[target_module], self.out_features[target_module]),
                        dim=-1,
                    )
                    A = A.reshape(bs, -1)
                    B = B.reshape(bs, -1)
                    splitted_out = [A, B]
            else:
                # head outputs either A or B
                splitted_out = []
                for out_emb, num_features in zip(
                    self.out_emb[target_module].values(),
                    [self.in_features[target_module], self.out_features[target_module]],
                ):
                    head_in = self.mlp2(mlp_out + self.out_emb_norm(out_emb))
                    if not self.autoreg_gen:
                        # head outputs all the ranks at once
                        head_out = head(self.mlp3(head_in))
                        head_out = head_out.view(bs, self.peft_config.r, -1)
                        head_out = head_out[..., :num_features]
                        head_out = head_out.reshape(bs, self.peft_config.r * num_features)
                    else:
                        # head outputs one rank at a time
                        head_in = head_in.unsqueeze(1).expand(-1, self.peft_config.r, -1)
                        if self.learnable_pos_emb:
                            pos_emb = self.mlp2(
                                self.pos_emb(
                                    torch.arange(self.peft_config.r, device=self.device)
                                ).unsqueeze(0)
                            )
                            head_in = head_in + pos_emb
                        else:
                            cos, sin = self.rotary_emb(
                                head_in,
                                torch.arange(
                                    self.peft_config.r, device=self.device
                                ).unsqueeze(0),
                            )  # [1, r, 2048]
                            head_in = self.mlp2(apply_rotary_pos_emb(head_in, cos, sin))

                        head_out = head(self.mlp3(head_in))
                        head_out = head_out[..., :num_features]
                        head_out = head_out.reshape(bs, self.peft_config.r * num_features)

                    splitted_out.append(head_out)
            
            # Convert to factorized form
            if self.output_space == "lora":
                A, B = splitted_out
                A = A.reshape(
                    bs, self.peft_config.r, self.in_features[target_module]
                )
                B = B.reshape(
                    bs, self.peft_config.r, self.out_features[target_module]
                ).transpose(-1, -2)
                
                if self.training_task == "sft":
                    A = A + self.AB_offset["A"][target_module][layer_indices]
                    B = B + self.AB_offset["B"][target_module][layer_indices]
                
                lora_A[target_module] = A
                lora_B[target_module] = B

        # save deltaW to lora state dict format
        lora_state_dict = dict()
        for target_module in self.target_modules:
            for layer_idx in layer_indices:
                for module_name in self.module_names[target_module][layer_idx]:
                    if "lora_A" in module_name:
                        lora_state_dict[module_name] = (
                            lora_A[target_module][layer_idx].cpu().contiguous()
                        )
                    elif "lora_B" in module_name:
                        lora_state_dict[module_name] = (
                            lora_B[target_module][layer_idx].cpu().contiguous()
                        )
                    else:
                        raise ValueError(f"Unexpected module name: {module_name}")
        if self.training_task == "recon":
            lora_state_dict = self.convert_to_raw_scale(lora_state_dict, layer_indices)
        return lora_state_dict
    
    @torch.no_grad()
    def gen_lora(self, layer_indices, encoded_task_emb):
        """Generate LoRA weights - uses the efficient all-types method."""
        return self.gen_lora_all_types(layer_indices, encoded_task_emb)

    def convert_to_raw_scale(self, lora_sd, layer_indices):
        if self.factorized:
            if self.pred_z_score:
                mean_recon_target = lora_tensor_dict_to_state_dict(
                    self.mean_recon_target,
                    self.module_names,
                    self.peft_config.target_modules,
                    layer_indices,
                )
                std_recon_target = lora_tensor_dict_to_state_dict(
                    self.std_recon_target,
                    self.module_names,
                    self.peft_config.target_modules,
                    layer_indices,
                )
                for module in lora_sd:
                    lora_sd[module] = (
                        lora_sd[module] * (std_recon_target[module] + 1e-10)
                        + mean_recon_target[module]
                    )

            if self.AB_offset:
                offset = lora_tensor_dict_to_state_dict(
                    self.AB_offset,
                    self.module_names,
                    self.peft_config.target_modules,
                    layer_indices,
                )
                for module in lora_sd:
                    lora_sd[module] += offset[module]
        else:
            if self.pred_z_score:
                raise NotImplementedError(
                    "pred_z_score not implemented for non-factorized LORA"
                )
            # we don't support using AB_offset with recon training
            for module in lora_sd:
                lora_sd[module] = lora_sd[module] / sqrt(self.delta_w_scaling)
        return lora_sd


def get_in_out_features(
    model: PeftModel,
    peft_config: PeftConfig = None,
) -> tuple[dict[str, int], dict[str, int]]:
    if peft_config is None:
        peft_config = model.peft_config["default"]
    in_features = dict()
    out_features = dict()
    for module_name, module in model.named_modules():
        if not check_target_module_exists(peft_config, module_name):
            continue
        if not isinstance(module, BaseTunerLayer):
            continue
        # support just Linear layer for now
        # all modules should be a leave module that is Linear layer
        assert isinstance(module.base_layer, nn.Linear), (
            "all modules should be a leave module that is Linear layer"
        )

        # this should always pass
        name = module_name.split(".")[-1]
        assert name in peft_config.target_modules, (
            f"Module {name} not in target modules"
        )

        if name not in in_features:
            in_features[name] = module.in_features
            out_features[name] = module.out_features
        else:
            # assumes each module has the same input and output features
            assert in_features[name] == module.in_features
            assert out_features[name] == module.out_features

    return in_features, out_features


def get_init_peft_weights(model: PeftModel, peft_config: PeftConfig = None):
    if peft_config is None:
        peft_config = model.peft_config["default"]
    peft_weights = {module_name: dict() for module_name in peft_config.target_modules}
    adapter_name = "default"
    for module_name, module in model.named_modules():
        if not check_target_module_exists(peft_config, module_name):
            continue
        if not isinstance(module, BaseTunerLayer):
            continue
        # support just Linear layer for now
        # all modules should be a leave module that is Linear layer
        assert isinstance(module.base_layer, nn.Linear), (
            "all modules should be a leave module that is Linear layer"
        )

        # this should always pass
        name = module_name.split(".")[-1]
        assert name in peft_config.target_modules

        for submodule_name, submodule in module.named_modules():
            if not isinstance(submodule, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
                continue

            if adapter_name not in submodule:
                continue

            if submodule_name not in peft_weights[name]:
                peft_weights[name][submodule_name] = submodule[adapter_name]
            else:
                smod1 = peft_weights[name][submodule_name]
                smod2 = submodule[adapter_name]
                assert type(smod1) == type(smod2)

    return peft_weights


def save_lora(lora_state_dict, adapter_config, lora_dir):
    logger.debug(f"lora_dir: {lora_dir}")
    os.makedirs(lora_dir, exist_ok=True)
    adapter_config.save_pretrained(lora_dir)
    save_file(lora_state_dict, f"{lora_dir}/adapter_model.safetensors")


def create_hypermod(
    args, peft_type, device, model, layer_indices, task_emb_size, from_scratch=True
):
    assert args.training_task in ["sft", "recon"], (
        f"Invalid training task: {args.training_task}"
    )
    module_names = get_lora_module_names(model, args.target_modules, layer_indices)

    mt_lora_sd = mt_lora_td = mean_recon_target = std_recon_target = None
    if from_scratch:
        if args.mt_lora_path:
            mt_lora_sd = load_peft_weights(args.mt_lora_path)
            mt_lora_td = lora_state_dict_to_tensor_dict(
                mt_lora_sd, args.target_modules, layer_indices, device=device
            )

        if args.training_task == "recon":
            lora_paths = get_target_lora_dirs(args.train_ds_names, args.model_dir)
            target_loras = {
                task: load_peft_weights(path) for task, path in lora_paths.items()
            }
            if args.mt_lora_path:
                target_loras = {
                    task: {k: v - mt_lora_sd[k] for k, v in lora.items()}
                    for task, lora in target_loras.items()
                }
            if args.pred_z_score:
                mean_recon_target = get_mean_lora(list(target_loras.values()))
                mean_recon_target = lora_state_dict_to_tensor_dict(
                    mean_recon_target, args.target_modules, layer_indices, device
                )
                std_recon_target = get_std_lora(list(target_loras.values()))
                std_recon_target = lora_state_dict_to_tensor_dict(
                    std_recon_target, args.target_modules, layer_indices, device
                )

    hypermod = HyperModulator(
        model,
        training_task=args.training_task,
        pred_z_score=args.pred_z_score,
        mean_recon_target=mean_recon_target,
        std_recon_target=std_recon_target,
        output_space=peft_type,
        module_names=module_names,
        match_lora_init=args.mt_lora_path is None and args.training_task == "sft",
        task_emb_size=len(args.train_ds_names)
        if args.use_one_hot_task_emb
        else task_emb_size,
        shared_AB_head=args.shared_AB_head,
        autoreg_gen=args.autoreg_gen,
        learnable_pos_emb=args.learnable_pos_emb,
        AB_offset=mt_lora_td,
        learnable_AB_offset=args.learnable_AB_offset,
        zero_init_head=args.training_task == "sft",
        latent_size=args.hypernet_latent_size,
        head_in_size=args.head_in_size,
        head_use_bias=(args.training_task == "sft" and args.mt_lora_path is None)
        or (args.training_task == "recon" and not args.pred_z_score),
        factorized=getattr(args, "factorized", False),
        delta_w_scaling=getattr(args, "delta_w_scaling", 10000),
        # Fusion parameters
        use_conv_fusion=args.use_conv_fusion,
        conv_fusion_type=args.conv_fusion_type,
        conv_fusion_kernel_size=args.conv_fusion_kernel_size,
        conv_fusion_num_layers=args.conv_fusion_num_layers,
        conv_fusion_channels=args.conv_fusion_channels,
        conv_fusion_dropout=args.conv_fusion_dropout,
        # Attention fusion parameters
        use_attention_fusion=getattr(args, "use_attention_fusion", False),
        attention_fusion_type=getattr(args, "attention_fusion_type", "self"),
        attention_num_heads=getattr(args, "attention_num_heads", 8),
        attention_dropout=getattr(args, "attention_dropout", 0.1),
        attention_num_layers=getattr(args, "attention_num_layers", 2),
    ).to(device)

    return hypermod


def save_hypermod_checkpoint(save_dir, hypermod, curstep):
    save_path = f"{save_dir}/checkpoints/it_{curstep}/hypermod.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(hypermod.state_dict(), save_path)
    return save_path


def load_hypermod_checkpoint(checkpoint_path, device, skip_emb_model=False):
    """Load a hypermod checkpoint.

    Args:
        checkpoint_path: Path to the hypermod.pt checkpoint file.
        device: Device to load tensors onto.
        skip_emb_model: If True, skip loading the local embedding model and
            infer ``task_emb_size`` from the checkpoint weights instead.  Useful
            when the embedding model is unavailable locally and API-based
            embedding will be used at runtime.
    """
    base_hypermod_dir = os.path.dirname(checkpoint_path)
    if "checkpoint" in base_hypermod_dir:
        base_hypermod_dir = base_hypermod_dir.split("checkpoint")[0]

    args = argparse.Namespace(
        **yaml.safe_load(open(f"{base_hypermod_dir}/args.yaml", "r"))
    )
    peft_config = get_peft_config(
        PeftConfig.from_json_file(f"{base_hypermod_dir}/adapter_config.json")
    )
    peft_type = peft_config.peft_type.lower()
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model, tokenizer = get_model_and_tokenizer(
        args.model_dir,
        train=False,
        requires_grad=False,
        peft_config=peft_config,
        model_kwargs={"output_hidden_states": True, "output_attentions": False},
        device=device,
    )
    # train to output delta_w for all layers
    layer_indices = torch.tensor(
        range(len(get_layers(model))), dtype=torch.long, device=device
    )

    task_emb_size = emb_model = emb_tokenizer = task_desc_format_fn = pooling_fn = None
    if not args.use_one_hot_task_emb:
        if skip_emb_model:
            # Infer task_emb_size from the task encoder's first linear layer
            te_weight_key = "task_encoder.mlp.0.weight"
            if te_weight_key in state_dict:
                task_emb_size = state_dict[te_weight_key].shape[1]
                logger.info(f"Inferred task_emb_size={task_emb_size} from checkpoint")
            else:
                raise RuntimeError(
                    f"Cannot infer task_emb_size: key '{te_weight_key}' not found "
                    f"in checkpoint. Available keys: {[k for k in state_dict if 'task_encoder' in k]}"
                )
            # Set up task_desc_format_fn without the local model
            from hyper_llm_modulator.utils.preprocessing import create_profile_extraction_template_fn
            if args.emb_model and "qwen" in args.emb_model.lower():
                task_desc_format_fn = partial(create_profile_extraction_template_fn)
            else:
                task_desc_format_fn = add_full_stop
        else:
            emb_model = model
            emb_tokenizer = deepcopy(tokenizer)
            task_desc_format_fn = add_full_stop
            pooling_fn = get_pooling_fn("last_token")

            if args.emb_model:
                emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn = (
                    get_emb_model_and_fns(args.emb_model, device)
                )
                logger.debug(f"emb_model: {emb_model}")
            emb_model.eval()
            task_emb_size = emb_model.config.hidden_size

    hypermod = create_hypermod(
        args, peft_type, device, model, layer_indices, task_emb_size, from_scratch=False
    )
    info = hypermod.load_state_dict(state_dict, strict=False)
    print(f"Loaded hypermod state dict: {info}")
    hypermod.eval().to(device)
    return (
        args,
        hypermod,
        model,
        tokenizer,
        emb_model,
        emb_tokenizer,
        task_desc_format_fn,
        pooling_fn,
    )
