"""Per-character persona-vector calibration for activation steering.

Implements the standard *contrastive activation* recipe (Anthropic
"Persona Vectors" 2024; Turner et al. "Activation Addition" 2023):

    1. For each character, sample K dialogue contexts from the *training*
       split.
    2. For every sample, build two prompts:
         conditional   = profile + context
         unconditional = context only
    3. Forward both through the model, capture the residual stream at one
       chosen layer at the *last* token position.
    4. Average the (cond − uncond) deltas over the K samples → that
       character's persona vector v_role ∈ ℝ^d.
    5. Save {role: v_role} to a single .pt file for downstream use by
       ``predict_steering.py``.

Layer choice
------------
We follow Anthropic and pick a middle-late layer (~60–70 % depth).  For
Qwen2.5-7B (28 transformer layers, indices 0–27) this defaults to layer
18 (≈64 %).  Override with ``--layer``.

Pool selection
--------------
We read the *training* split of the m2_raw_profile variant so the
profile embeddings the persona vector is built from match what the LLM
sees at evaluation time.  No test leakage: the test split is never
touched here.

Usage
-----
::

    python evaluation/steering_calibrate.py \\
        --train_data LongEvoRoleBench/processed/RAIDEN/m2_raw_profile/train.json \\
        --output     results/RAIDEN/comparison/main/steering/persona_vectors.pt \\
        --layer 18 --num_per_role 50

The output ``.pt`` is a dict ``{role: torch.FloatTensor[d_model]}``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Mirror predict_cfg.py — keep templates identical so calibration prompts
# match what predict_steering.py will see at inference time.
BASELINE_PROMPT = """\
Below is a multi-turn dialogue. Predict the single line that {character} would most likely say next.
Keep the reply short and natural, matching the tone and length of the other lines. Output only that one line, no explanation.

Dialogue context:
{context}

{character}:"""

PROFILE_PROMPT = """\
Below is a multi-turn dialogue. Predict the single line that {character} would most likely say next.

Character profile for {character}:
{profile}

Keep the reply short and natural, matching the tone and length of the other lines. Output only that one line, no explanation.

Dialogue context:
{context}

{character}:"""


def build_prompts(sample: dict) -> tuple[str, str]:
    cond = PROFILE_PROMPT.format(
        character=sample["role"],
        context=sample["input"],
        profile=(sample.get("profile_text") or "").strip(),
    )
    uncond = BASELINE_PROMPT.format(
        character=sample["role"],
        context=sample["input"],
    )
    return cond, uncond


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate per-character persona vectors for activation "
                    "steering.")
    parser.add_argument("--train_data", required=True,
                        help="Train-split JSON containing profile_text "
                             "(e.g. m2_raw_profile/train.json)")
    parser.add_argument("--output", required=True,
                        help="Path to write persona_vectors.pt")
    parser.add_argument("--model", default=None,
                        help="LLM path (default: PHASE-Tree/models/Qwen2.5-7B-Instruct)")
    parser.add_argument("--layer", type=int, default=18,
                        help="Transformer layer index to extract residual stream "
                             "from (default 18 for Qwen2.5-7B's 28 layers)")
    parser.add_argument("--num_per_role", type=int, default=50,
                        help="How many training samples per role to average over")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.model is None:
        args.model = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-7B-Instruct")

    random.seed(args.seed)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading model: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    if not (0 <= args.layer < n_layers):
        raise ValueError(f"layer {args.layer} out of range [0, {n_layers})")
    print(f"  Model: {n_layers} layers × {d_model} dim, "
          f"using layer {args.layer} (~{100*args.layer/n_layers:.0f}% depth)",
          flush=True)

    print(f"  Loading train data: {args.train_data}", flush=True)
    with open(args.train_data, "r", encoding="utf-8") as f:
        train = json.load(f)

    by_role: dict[str, list[dict]] = defaultdict(list)
    for s in train:
        if (s.get("profile_text") or "").strip():
            by_role[s["role"]].append(s)
    roles = sorted(by_role.keys())
    print(f"  Found {len(roles)} roles in train split (with profile_text)",
          flush=True)

    @torch.no_grad()
    def last_token_hidden(prompt: str) -> torch.Tensor:
        msgs = [{"role": "user", "content": prompt}]
        chat = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(
            chat, return_tensors="pt", truncation=True,
            max_length=args.max_model_len,
        ).to(model.device)
        out = model(**ids, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[args.layer + 1]  # +1 because index 0 = embeddings
        return h[0, -1, :].detach().float().cpu()

    persona_vectors: dict[str, torch.Tensor] = {}
    for role in roles:
        pool = by_role[role]
        random.shuffle(pool)
        chosen = pool[:args.num_per_role]
        deltas = []
        for s in tqdm(chosen, desc=f"calibrate[{role}]", file=sys.stderr,
                      dynamic_ncols=True):
            cond_p, uncond_p = build_prompts(s)
            h_cond = last_token_hidden(cond_p)
            h_uncond = last_token_hidden(uncond_p)
            deltas.append(h_cond - h_uncond)
        v = torch.stack(deltas, dim=0).mean(dim=0)
        persona_vectors[role] = v
        print(f"  ✓ {role}: ‖v‖={v.norm():.3f}, n={len(chosen)}", flush=True)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "vectors": persona_vectors,
        "layer": args.layer,
        "num_per_role": args.num_per_role,
        "model": args.model,
        "d_model": d_model,
        "n_layers": n_layers,
    }, args.output)
    print(f"\n✓ Saved persona vectors for {len(persona_vectors)} roles → {args.output}",
          flush=True)


if __name__ == "__main__":
    main()
