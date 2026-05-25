"""Prompt-formatting helpers for PHASE-Tree training.

PHASE-Tree's role-play datasets (RAIDEN / Friends / HPD / TheOffice / TNG)
are pre-processed offline (see ``preprocessing/preprocess_dialogues_*.py``)
into JSON files with explicit ``input`` / ``output`` / ``profile_text``
fields. At training time the raw HuggingFace records can be passed through
the metadata template (``user_prompt_template`` etc.) without any extra
per-dataset rewriting, so :func:`get_preprocessing_fn` simply returns the
identity transformation.

The remaining helpers in this module are shared between the dataloader and
the embedding model:

- :func:`add_full_stop`, :func:`apply_sfr_template`,
  :func:`apply_profile_extraction_template`, :func:`format_profile_text`,
  :func:`create_profile_extraction_template_fn` shape the text fed to the
  embedding model that produces the hypernet input.
- :func:`get_prompt_formatting_fn` turns each example into either causal-LM
  text or a (prompt, response) pair, respecting the tokenizer's chat
  template.
- :func:`_prepare_chat_template` lets us pin a tokenizer to a local
  ``chat_template.jinja`` shipped under ``models/chat_templates/``.
"""

from pathlib import Path
from typing import Callable, Literal

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
# Chat templates live under ``models/chat_templates/`` (no top-level symlink).
CHAT_TEMPLATE_BASE_DIR = REPO_ROOT / "models" / "chat_templates"


def get_preprocessing_fn(ds_name):
    """Return a per-example transformation for ``ds_name``.

    PHASE-Tree datasets already store the fields expected by the metadata
    templates, so no rewriting is needed and we simply pass examples
    through unchanged.
    """

    del ds_name  # kept for API stability with the previous signature
    return lambda example: example


def add_full_stop(s):
    s = s.strip()
    if s[-1].isalpha():
        s += "."
    return s


def apply_sfr_template(query: str) -> str:
    # from https://github.com/microsoft/unilm/blob/9c0f1ff7ca53431fe47d2637dfe253643d94185b/e5/utils.py#L106
    task_description = "Retrieve semantically similar text."
    return f"Instruct: {task_description}\nQuery: {query}"


def apply_profile_extraction_template(user_info_text: str, task_description: str | None = None) -> str:
    """Wrap profile text with an extraction instruction prompt.

    Used to format the hypernet input fed into the embedding model so that
    it focuses on character-level features rather than the raw text string.
    """
    base_instruction = (
        "Extract and represent key character features that reflect the character's unique traits, "
        "preferences, and behavioral patterns from the provided character profile. Focus on personality, "
        "speaking style, relationships, behavioral tendencies, and other attributes that can "
        "inform character-specific response generation."
    )

    if task_description:
        task_aware_instruction = (
            f"{base_instruction} Pay special attention to features relevant for the following task: {task_description}"
        )
        return f"# Instruct:\n{task_aware_instruction}\n\n# Character Profile:\n{user_info_text.strip()}"

    return f"# Instruct:\n{base_instruction}\n\n# Character Profile:\n{user_info_text.strip()}"


def format_profile_text(profile_text, *_args, **_kwargs) -> str:
    """Return the per-sample profile string used as hypernetwork input.

    PHASE-Tree's role-play datasets carry a single ``profile_text`` field
    per sample, which is the only user-level signal used by the
    hypernetwork. The function therefore reduces to string coercion and
    accepts any additional positional/keyword arguments for backward
    compatibility with older call sites.
    """
    if profile_text is None:
        return ""
    return profile_text if isinstance(profile_text, str) else str(profile_text)


def create_profile_extraction_template_fn(task_description: str | None = None, *_args, **_kwargs):
    """Build a callable that wraps a per-sample profile string with the extraction prompt.

    Extra positional/keyword arguments are accepted (and ignored) so older
    call sites threading a ``user_profile_format`` argument keep working.
    """

    def profile_extraction_template_fn(profile_text) -> str:
        return apply_profile_extraction_template(format_profile_text(profile_text), task_description)

    return profile_extraction_template_fn


def _iter_template_paths(candidate: str):
    """Yield possible local paths for a chat template candidate."""
    if not isinstance(candidate, str):
        return

    candidate = candidate.strip()
    if not candidate:
        return

    seen: set[str] = set()
    candidate_path = Path(candidate)

    def _yield(path: Path):
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        yield path

    if candidate_path.is_absolute():
        if candidate_path.is_dir():
            yield from _yield(candidate_path / "chat_template.jinja")
        yield from _yield(candidate_path)
        return

    normalized = Path(candidate.strip("/"))
    for base_dir in (CHAT_TEMPLATE_BASE_DIR, REPO_ROOT):
        potential = base_dir / normalized
        if potential.is_dir():
            yield from _yield(potential / "chat_template.jinja")
        yield from _yield(potential)


def _find_local_chat_template(metadata: dict, apply_chat_template_fn):
    """Locate a chat template file for the provided tokenizer metadata."""
    tokenizer = getattr(apply_chat_template_fn, "__self__", None)
    if tokenizer is None:
        return None

    candidates = []
    for key in ("chat_template_path", "chat_template_name", "chat_template"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    name_or_path = getattr(tokenizer, "name_or_path", None)
    if isinstance(name_or_path, str) and name_or_path.strip():
        model_id = name_or_path.strip()
        candidates.append(model_id)
        base_name = Path(model_id).name
        if base_name != model_id:
            candidates.append(base_name)

    for candidate in candidates:
        for path in _iter_template_paths(candidate):
            if path.is_file():
                return path

    return None


def _prepare_chat_template(metadata: dict, apply_chat_template_fn):
    """Ensure the tokenizer behind ``apply_chat_template_fn`` uses the local chat template."""
    tokenizer = getattr(apply_chat_template_fn, "__self__", None)
    if tokenizer is None:
        return apply_chat_template_fn

    template_path = _find_local_chat_template(metadata, apply_chat_template_fn)
    if template_path is None:
        return apply_chat_template_fn

    cached_path = getattr(tokenizer, "_local_chat_template_path", None)
    if cached_path == str(template_path):
        return tokenizer.apply_chat_template

    try:
        template_text = template_path.read_text(encoding="utf-8")
    except OSError:
        return apply_chat_template_fn

    normalized_template = template_text.replace("    ", "").replace("\n", "")
    if getattr(tokenizer, "chat_template", None) != normalized_template:
        tokenizer.chat_template = normalized_template
    tokenizer._local_chat_template_path = str(template_path)
    return tokenizer.apply_chat_template


def get_prompt_formatting_fn(
    metadata,
    sft_mode: Literal["causal_lm", "completion"],
    apply_chat_template_fn: Callable,
    is_intx_model: bool,
):
    assert sft_mode in ["causal_lm", "completion"], f"Invalid training task: {sft_mode}"

    if is_intx_model:
        apply_chat_template_fn = _prepare_chat_template(metadata, apply_chat_template_fn)

    def f(example):
        output_texts = dict(text=[]) if sft_mode == "causal_lm" else dict(prompt=[], response=[])
        df = pd.DataFrame(dict(example))
        for _, inp_txt in df.iterrows():
            if sft_mode == "causal_lm":
                text = metadata["text_template"].format(**inp_txt)
                output_texts["text"].append(text)
            elif sft_mode == "completion":
                prompt = metadata["user_prompt_template"].format(**inp_txt)
                output_texts["prompt"].append(prompt)
                output_texts["response"].append(str(inp_txt[metadata["response_field"]]))
        return output_texts

    def f_intx(example):
        output_texts = dict(text=[]) if sft_mode == "causal_lm" else dict(prompt=[], response=[])
        df = pd.DataFrame(dict(example))
        for _, inp_txt in df.iterrows():
            # NOTE: we assume specific chat_template here
            # that the chat_template should not have a default system_message
            # and it skips the system header if system_message is not provided
            # that is, using apply_chat_template to response_chat would not add the system_message
            prompt_chat = [
                {"role": "system", "content": metadata["system_message"].format(**inp_txt)},
                {"role": "user", "content": metadata["user_prompt_template"].format(**inp_txt)},
            ]
            response_chat = [
                {
                    "role": "assistant",
                    "content": metadata["assistant_prefill"].format(**inp_txt)
                    + str(inp_txt[metadata["response_field"]]),
                }
            ]
            if "assistant_postfill" in metadata:
                response_chat[0]["content"] += metadata["assistant_postfill"].format(**inp_txt)
            if sft_mode == "causal_lm":
                text = apply_chat_template_fn(prompt_chat + response_chat, tokenize=False, add_generation_prompt=False)
                output_texts["text"].append(text)
            elif sft_mode == "completion":
                prompt = apply_chat_template_fn(prompt_chat, tokenize=False, add_generation_prompt=False)
                response = apply_chat_template_fn(response_chat, tokenize=False, add_generation_prompt=False)
                output_texts["prompt"].append(prompt)
                output_texts["response"].append(response)
        return output_texts

    return f if not is_intx_model else f_intx
