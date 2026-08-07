"""Preprocess CharacterEval multi-turn dialogues and perform 3-way split.

This script runs two consecutive steps:

**Step 1 — Dialogue conversion.**  Each multi-turn dialogue is converted
into a single training/evaluation sample:

* ``input``  — all turns preceding the last target-character utterance.
* ``output`` — the last target-character utterance itself.

Stage directions inside parentheses (e.g. ``（笑）``, ``(sighs)``) are
stripped from both input and output.  The result is saved as the unsplit
``all_dialogues.json``.

**Step 2 — Character-level 3-way split.**  Profile embeddings are
computed via the embedding API (configured in ``.env``), then K-Means
clustering selects outlier clusters for the OOD set.  All dialogues of
a given character are assigned to exactly one split:

* ``train.json``        — characters from majority clusters.
* ``random_test.json``  — diverse sample from remaining characters.
* ``ood_test.json``     — characters from small, isolated clusters.

Input
    ``LongEvoRoleBench/raw_data/CharacterEval/data/test_data.jsonl``
    ``LongEvoRoleBench/processed/CharacterEval/intermediate/raw_profile_texts.json``

Output
    ``LongEvoRoleBench/processed/CharacterEval/intermediate/all_dialogues.json``
    ``LongEvoRoleBench/processed/CharacterEval/intermediate/character_embeddings.npz``
    ``LongEvoRoleBench/processed/CharacterEval/intermediate/{train,random_test,ood_test}.json``

Dependencies
    * ``split_utils`` — shared embedding + 3-way split logic
      (requires ``openai``, ``python-dotenv``, ``scikit-learn``).

Usage::

    python preprocess_dialogues_charactereval.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from split_utils import compute_character_embeddings, three_way_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "LongEvoRoleBench" / "raw_data"
PROCESSED_DIR = PROJECT_ROOT / "LongEvoRoleBench" / "processed"

# ---------------------------------------------------------------------------
# Step 1: dialogue parsing
# ---------------------------------------------------------------------------

_RE_PAREN = re.compile(r"（[^）]*）|\([^)]*\)")


def _strip_parens(text: str) -> str:
    """Remove all parenthesized stage directions from *text*."""
    cleaned = _RE_PAREN.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_dialogue_lines(context: str) -> list[tuple[str, str]]:
    """Parse a raw dialogue context into a list of (speaker, content)."""
    turns: list[tuple[str, str]] = []
    for line in context.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        sep = line.find("：")
        if sep == -1:
            if turns:
                turns[-1] = (turns[-1][0], turns[-1][1] + line)
            continue
        speaker = line[:sep].strip()
        content = _strip_parens(line[sep + 1:])
        if not content:
            continue
        turns.append((speaker, content))
    return turns


def build_single_sample(item: dict) -> dict | None:
    """One dialogue -> one sample.

    ``input``  = all turns before the last target-character turn.
    ``output`` = the last target-character turn's content.
    Returns *None* when no usable target turn exists.
    """
    role = item["role"]
    turns = parse_dialogue_lines(item["context"])

    last_target_idx = None
    for i in range(len(turns) - 1, -1, -1):
        if turns[i][0] == role:
            last_target_idx = i
            break

    if last_target_idx is None or last_target_idx == 0:
        return None

    history_lines = [f"{s}：{c}" for s, c in turns[:last_target_idx]]
    return {
        "user_id": f"CharacterEval_{role}",
        "question_id": f"CharacterEval_{item['id']}",
        "novel_name": item["novel_name"],
        "role": role,
        "profile_text": "",
        "input": "\n".join(history_lines),
        "output": turns[last_target_idx][1],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Step 1: all_dialogues.json ----
    src = RAW_DATA_DIR / "CharacterEval" / "data" / "test_data.jsonl"
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} dialogue items from {src}")

    all_samples: list[dict] = []
    skipped = 0
    for item in data:
        sample = build_single_sample(item)
        if sample:
            all_samples.append(sample)
        else:
            skipped += 1

    out_dir = PROCESSED_DIR / "CharacterEval" / "intermediate"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_path = out_dir / "all_dialogues.json"
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    roles = sorted({s["role"] for s in all_samples})
    print(f"\nall_dialogues.json: {len(all_samples)} samples "
          f"({skipped} skipped), {len(roles)} roles")
    print(f"Saved -> {all_path}")

    # ---- Step 2: embeddings ----
    prof_path = out_dir / "raw_profile_texts.json"
    if not prof_path.exists():
        raise FileNotFoundError(
            f"{prof_path} not found. "
            f"Run  python src/tree_pipeline/raw_profiles_to_text.py --dataset CharacterEval  first."
        )
    with open(prof_path, "r", encoding="utf-8") as f:
        profile_texts: dict[str, str] = json.load(f)

    chars_with_prof = [r for r in roles if r in profile_texts]
    print(f"\nCharacters with profile: {len(chars_with_prof)}/{len(roles)}")

    chars, emb = compute_character_embeddings(
        chars_with_prof, profile_texts, env_path=PROJECT_ROOT / ".env")

    emb_path = out_dir / "character_embeddings.npz"
    np.savez(emb_path, user_ids=np.array(chars), embeddings=emb)
    print(f"Saved embeddings -> {emb_path}")

    # ---- Step 3: 3-way split ----
    n_ood = max(5, len(chars) // 7)
    n_rt = max(5, len(chars) // 7)
    print(f"\nTarget split: train≈{len(chars)-n_ood-n_rt}, "
          f"random_test≈{n_rt}, ood_test≈{n_ood}")

    train_c, rt_c, ood_c = three_way_split(chars, emb, n_rt, n_ood)

    print(f"\nFinal split: train={len(train_c)}, "
          f"random_test={len(rt_c)}, ood_test={len(ood_c)}")
    print(f"  Train chars:       {train_c}")
    print(f"  Random test chars: {rt_c}")
    print(f"  OOD test chars:    {ood_c}")

    role_to_samples: dict[str, list[dict]] = defaultdict(list)
    for s in all_samples:
        role_to_samples[s["role"]].append(s)

    splits = {
        "train": [s for c in train_c for s in role_to_samples[c]],
        "random_test": [s for c in rt_c for s in role_to_samples[c]],
        "ood_test": [s for c in ood_c for s in role_to_samples[c]],
    }

    for name, samples in splits.items():
        path = out_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        n_roles = len({s["role"] for s in samples})
        print(f"  {name}.json: {len(samples)} samples, {n_roles} chars")

    # Verify zero overlap
    r_train = {s["role"] for s in splits["train"]}
    r_rt = {s["role"] for s in splits["random_test"]}
    r_ood = {s["role"] for s in splits["ood_test"]}
    assert not (r_train & r_rt), "Train / Random test role overlap!"
    assert not (r_train & r_ood), "Train / OOD test role overlap!"
    assert not (r_rt & r_ood), "Random test / OOD test role overlap!"
    print("\n✓ No role overlap between any splits")


if __name__ == "__main__":
    main()
