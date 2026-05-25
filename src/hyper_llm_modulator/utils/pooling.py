"""Pooling strategies for extracting fixed-size embeddings from transformer outputs.

Supports ``last_token`` (causal LM style) and ``cls`` (encoder style)
pooling over the hidden states.
"""

from typing import Literal

import torch


def get_pooling_fn(pooling_type: Literal["last_token", "cls"]):
    if pooling_type == "last_token":
        return last_token_pool
    elif pooling_type == "cls":
        return cls_pool
    else:
        raise ValueError(f"Invalid pooling type: {pooling_type}")


def cls_pool(outputs: dict[str, torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
    right_padding = attention_mask[:, 0].sum() == attention_mask.shape[0]
    assert right_padding, f'tokenizer.padding_side should be "right"'
    return outputs["last_hidden_state"][:, 0].detach()


def last_token_pool(outputs: dict[str, torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden_states = (
        outputs["hidden_states"][-1].detach() if "hidden_states" in outputs else outputs["last_hidden_state"].detach()
    )
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
