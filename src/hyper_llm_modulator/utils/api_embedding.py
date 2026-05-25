"""
API-based embedding utilities for text embedding generation.

This module provides functions to generate embeddings using:
1. OpenAI API - for official OpenAI embedding models
2. vLLM API - for locally served embedding models via vLLM (using OpenAI-compatible API)

The functions follow the same interface as the local embedding functions
to ensure compatibility with the existing codebase.
"""

import os
import time
import logging
from typing import List, Optional, Union
from math import sqrt

import torch
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Suppress verbose HTTP request logs from OpenAI client and related libraries
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Optional imports for API clients
try:
    import openai
    # Check OpenAI version and use appropriate client
    try:
        # Test for new OpenAI client (v1.0+) by checking if OpenAI class exists
        # Don't create a client at import time to avoid API key requirements
        _ = openai.OpenAI
        OPENAI_V1 = True
    except AttributeError:
        # Old OpenAI client (< v1.0)
        OPENAI_V1 = False
    OPENAI_AVAILABLE = True
    openai_client = None  # Don't create client at import time
except ImportError:
    OPENAI_AVAILABLE = False
    OPENAI_V1 = False
    openai_client = None
    logger.warning("OpenAI package not available. Install with: pip install openai")


def create_batches(texts: List[str], max_texts_per_batch: int = 100) -> List[List[str]]:
    """
    Create batches that respect batch size limits.
    
    Args:
        texts: List of texts to batch
        max_texts_per_batch: Maximum number of texts per batch
        
    Returns:
        List of text batches
    """
    batches = []
    current_batch = []
    
    for text in texts:
        # Check if adding this text would exceed batch size limit
        if len(current_batch) >= max_texts_per_batch:
            # Finalize current batch and start a new one
            batches.append(current_batch)
            current_batch = [text]
        else:
            # Add to current batch
            current_batch.append(text)
    
    # Don't forget the last batch
    if current_batch:
        batches.append(current_batch)
    
    return batches


class APIEmbeddingError(Exception):
    """Custom exception for API embedding errors."""
    pass


# Cache of (model_name -> tokenizer | None) used for *accurate* token counting
# during max_token_per_profile truncation.  None marks a model whose tokenizer
# could not be loaded locally (e.g. HF id with no offline cache); for those we
# fall back to the legacy chars/3.7 estimator.
_TOKEN_COUNT_TOKENIZER_CACHE: dict = {}


def _get_token_counting_tokenizer(model_name: str):
    """Lazily load the local tokenizer for `model_name` to count tokens exactly.

    For embedding models served via vLLM we point `model_name` at the local
    weight directory (e.g. ``models/Qwen3-Embedding-4B``), so the tokenizer
    files are present on disk and load instantly.  Returns None on failure;
    callers must then fall back to the chars/3.7 heuristic.
    """
    if model_name in _TOKEN_COUNT_TOKENIZER_CACHE:
        return _TOKEN_COUNT_TOKENIZER_CACHE[model_name]
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        _TOKEN_COUNT_TOKENIZER_CACHE[model_name] = tok
        logger.info(
            f"Loaded tokenizer for accurate token counting: {model_name}"
        )
        return tok
    except Exception as e:
        logger.warning(
            f"Could not load tokenizer for {model_name} ({type(e).__name__}: {e}); "
            "falling back to chars/3.7 estimation."
        )
        _TOKEN_COUNT_TOKENIZER_CACHE[model_name] = None
        return None


def embed_texts_unified_api(
    texts: List[str],
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    batch_size: int = 100,
    max_retries: int = 3,
    timeout: int = 60,
    device: Union[str, torch.device] = "cpu",
    task_desc_format_fn=None,
    max_token_per_profile: Optional[int] = None,
    **kwargs
) -> torch.Tensor:
    """
    Generate embeddings using OpenAI-compatible API (works for both OpenAI and vLLM).
    
    Args:
        texts: List of text strings to embed
        model: Model name (e.g., "text-embedding-3-large" for OpenAI or model name for vLLM)
        api_key: API key (if None, uses OPENAI_API_KEY env var for OpenAI)
        api_base: Custom API base URL (if None, uses default OpenAI API)
        batch_size: Maximum number of texts to process per API call
        max_retries: Maximum number of retries for failed requests
        timeout: Request timeout in seconds
        device: Device to place the resulting tensor on
        task_desc_format_fn: Optional function to format text descriptions
        max_token_per_profile: Maximum tokens per text (estimated as char_count // 3.7)
        **kwargs: Additional arguments (for compatibility with local embedding functions)
        
    Returns:
        torch.Tensor: Normalized embeddings with shape (len(texts), embedding_dim)
        
    Raises:
        APIEmbeddingError: If OpenAI package is not available or API calls fail
    """
    if not OPENAI_AVAILABLE:
        raise APIEmbeddingError("OpenAI package not available. Install with: pip install openai")
    
    if not model:
        raise APIEmbeddingError("Model name is required")
    
    # Initialize client
    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    elif not api_base and not os.getenv("OPENAI_API_KEY"):
        # Only require API key if using default OpenAI API
        raise APIEmbeddingError("API key not provided. Set via api_key parameter or OPENAI_API_KEY environment variable.")
    
    if api_base:
        # Ensure the base URL includes /v1 for OpenAI-compatible APIs
        if not api_base.endswith('/v1'):
            api_base = api_base.rstrip('/') + '/v1'
        client_kwargs["base_url"] = api_base
    
    if OPENAI_V1:
        client = openai.OpenAI(**client_kwargs)
    else:
        # Use old client configuration
        if api_key:
            openai.api_key = api_key
        elif not openai.api_key and not os.getenv("OPENAI_API_KEY"):
            raise APIEmbeddingError("API key not provided. Set via api_key parameter or OPENAI_API_KEY environment variable.")
        
        if api_base:
            # Ensure the base URL includes /v1 for OpenAI-compatible APIs
            if not api_base.endswith('/v1'):
                api_base = api_base.rstrip('/') + '/v1'
            openai.api_base = api_base
        client = None
    
    # Apply text formatting if provided
    if task_desc_format_fn:
        formatted_texts = [task_desc_format_fn(text) for text in texts]
    else:
        formatted_texts = texts
    
    # Truncate texts if max_token_per_profile is specified.
    # Prefer EXACT token counting via the embedding model's local tokenizer
    # (essential for CJK profiles where chars/3.7 grossly under-counts);
    # fall back to the legacy chars/3.7 estimate only when the tokenizer
    # cannot be loaded locally.
    if max_token_per_profile:
        tokenizer = _get_token_counting_tokenizer(model)
        truncated_texts = []
        n_truncated = 0

        if tokenizer is not None:
            for text in formatted_texts:
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) > max_token_per_profile:
                    truncated_ids = ids[:max_token_per_profile]
                    truncated_text = tokenizer.decode(
                        truncated_ids, skip_special_tokens=True
                    ) + "..."
                    truncated_texts.append(truncated_text)
                    n_truncated += 1
                    logger.debug(
                        f"Truncated text from {len(ids)} tokens to "
                        f"{max_token_per_profile} tokens (exact)"
                    )
                else:
                    truncated_texts.append(text)
        else:
            for text in formatted_texts:
                estimated_tokens = len(text) / 3.7
                if estimated_tokens > max_token_per_profile:
                    max_chars = int(max_token_per_profile * 3.7)
                    truncated_text = text[:max_chars] + "..."
                    truncated_texts.append(truncated_text)
                    n_truncated += 1
                    logger.debug(
                        f"Truncated text from {len(text)} chars "
                        f"(~{estimated_tokens:.0f} tokens) to {max_chars} chars "
                        f"(~{max_token_per_profile} tokens, estimated)"
                    )
                else:
                    truncated_texts.append(text)

        if n_truncated > 0:
            mode = "exact" if tokenizer is not None else "estimated"
            logger.info(
                f"Truncated {n_truncated}/{len(formatted_texts)} texts to "
                f"<= {max_token_per_profile} tokens ({mode})"
            )
        formatted_texts = truncated_texts
    
    # Create batches
    text_batches = create_batches(
        formatted_texts, 
        max_texts_per_batch=batch_size
    )
    
    embeddings = []
    total_tokens = 0
    
    api_type = "vLLM" if api_base and "localhost" in api_base else "OpenAI"
    logger.info(f"Generating embeddings for {len(formatted_texts)} texts using {api_type} API (model: {model})")
    logger.info(f"Created {len(text_batches)} batches (max {batch_size} texts per batch)")
    
    for batch_idx, batch_texts in enumerate(tqdm(text_batches, desc="API embedding batches")):
        # Log batch info for debugging
        logger.debug(f"Processing batch {batch_idx + 1}/{len(text_batches)}: {len(batch_texts)} texts")
        
        # Retry logic for API calls
        for attempt in range(max_retries):
            try:
                if OPENAI_V1:
                    # New API
                    response = client.embeddings.create(
                        input=batch_texts,
                        model=model,
                        timeout=timeout
                    )
                    # Extract embeddings from response
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                    
                    # Track token usage
                    if hasattr(response, 'usage') and response.usage:
                        total_tokens += response.usage.total_tokens
                else:
                    # Old API
                    response = openai.Embedding.create(
                        input=batch_texts,
                        model=model,
                        timeout=timeout
                    )
                    # Extract embeddings from response
                    batch_embeddings = [item['embedding'] for item in response['data']]
                    embeddings.extend(batch_embeddings)
                    
                    # Track token usage
                    if 'usage' in response:
                        total_tokens += response['usage'].get('total_tokens', 0)
                
                break  # Success, exit retry loop
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"API call failed after {max_retries} attempts: {e}")
                    raise APIEmbeddingError(f"API call failed: {e}")
                else:
                    logger.warning(f"API call attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
    
    logger.info(f"Generated {len(embeddings)} embeddings using {total_tokens} tokens")
    
    # Convert to tensor and normalize
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
    
    # Normalize embeddings (consistent with local embedding functions)
    embeddings_tensor = torch.nn.functional.normalize(embeddings_tensor) * sqrt(embeddings_tensor.shape[-1])
    
    return embeddings_tensor


def embed_texts_api(
    texts: List[str],
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
    task_desc_format_fn=None,
    max_token_per_profile:int = 26000,
    **kwargs
) -> torch.Tensor:
    """
    Generate embeddings using API (unified function for both OpenAI and vLLM).
    
    Args:
        texts: List of text strings to embed
        model: Model name (e.g., "text-embedding-3-large" for OpenAI or model name for vLLM)
        api_key: API key for authentication
        api_base: API base URL (None for OpenAI, custom URL for vLLM)
        device: Device to place the resulting tensor on
        task_desc_format_fn: Optional function to format text descriptions
        max_token_per_profile: Maximum tokens per text (estimated as char_count // 3.7)
        **kwargs: Additional arguments passed to the unified API function
        
    Returns:
        torch.Tensor: Normalized embeddings with shape (len(texts), embedding_dim)
    """
    return embed_texts_unified_api(
        texts=texts,
        model=model,
        api_key=api_key,
        api_base=api_base,
        device=device,
        task_desc_format_fn=task_desc_format_fn,
        max_token_per_profile=max_token_per_profile,
        **kwargs
    )


def create_api_embedding_kwargs(args):
    """
    Create keyword arguments for API embedding functions from training arguments.
    Automatically determines API type based on provided parameters.
    
    Args:
        args: Training arguments object containing API configuration
        
    Returns:
        dict: Keyword arguments for API embedding functions
    """
    kwargs = {
        "batch_size": args.api_embedding_batch_size,
        "max_retries": args.api_embedding_max_retries,
        "timeout": args.api_embedding_timeout,
    }
    
    # Add max_token_per_profile if it exists in args
    if hasattr(args, 'max_tokens_per_profile'):
        kwargs["max_token_per_profile"] = args.max_tokens_per_profile
    
    # Auto-detect API type based on which parameters are provided
    has_openai_params = bool(args.openai_embedding_model or args.openai_api_key or args.openai_api_base)
    has_vllm_params = bool(args.vllm_embedding_model or args.vllm_api_base != "http://localhost:8000")
    
    # Prefer vLLM if both are specified (since vLLM params are more explicit)
    if has_vllm_params:
        kwargs.update({
            "model": args.vllm_embedding_model,
            "api_key": args.vllm_api_key or None,
            "api_base": args.vllm_api_base,
        })
    elif has_openai_params:
        kwargs.update({
            "model": args.openai_embedding_model,
            "api_key": args.openai_api_key or os.getenv("OPENAI_API_KEY"),
            "api_base": args.openai_api_base or None,
        })
    else:
        # Default to OpenAI if no specific parameters are provided
        kwargs.update({
            "model": args.openai_embedding_model or "text-embedding-3-large",
            "api_key": args.openai_api_key or os.getenv("OPENAI_API_KEY"),
            "api_base": args.openai_api_base or None,
        })
    
    return kwargs 