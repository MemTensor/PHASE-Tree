"""Preprocess RAIDEN multi-turn dialogues and perform 3-way split.

**Step 1 — Dialogue conversion.**  Each entry in ``dialogue.json`` is
converted into a single training/evaluation sample:

* ``input``  — all messages in the ``messages`` list, formatted as
  ``角色名：content`` (assistant) / ``用户：content`` (user).
* ``output`` — the ``reference`` field (the target assistant response).

The result is saved as ``all_dialogues.json``.

**Step 2 — Character-level 3-way split.**  Profile embeddings are
computed via the embedding API (configured in ``.env``), then K-Means
clustering selects outlier clusters for the OOD set.  All dialogues of
a given character are assigned to exactly one split:

* ``train.json``        — characters from majority clusters.
* ``random_test.json``  — diverse sample from remaining characters.
* ``ood_test.json``     — characters from small, isolated clusters.

Input
    ``LongEvoRoleBench/raw_data/RAIDEN/release_data/dialogue.json``
    ``LongEvoRoleBench/processed/RAIDEN/intermediate/raw_profile_texts.json``

Output
    ``LongEvoRoleBench/processed/RAIDEN/intermediate/all_dialogues.json``
    ``LongEvoRoleBench/processed/RAIDEN/intermediate/character_embeddings.npz``
    ``LongEvoRoleBench/processed/RAIDEN/intermediate/{train,random_test,ood_test}.json``

Dependencies
    * ``split_utils`` — shared embedding + 3-way split logic
      (requires ``openai``, ``python-dotenv``, ``scikit-learn``).

Usage::

    python preprocess_dialogues_raiden.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from split_utils import compute_character_embeddings, three_way_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "LongEvoRoleBench" / "raw_data"
PROCESSED_DIR = PROJECT_ROOT / "LongEvoRoleBench" / "processed"

# ---------------------------------------------------------------------------
# Step 1: dialogue conversion
# ---------------------------------------------------------------------------

def build_single_sample(dialogue_id: str, entry: dict) -> dict | None:
    """Convert one RAIDEN dialogue entry into a training sample.

    ``input``  = all messages formatted as ``speaker：content``.
    ``output`` = the ``reference`` (next assistant turn).
    Returns *None* when the entry is unusable.
    """
    npc_name = entry.get("npc_name", "")
    messages = entry.get("messages", [])
    reference = entry.get("reference", "")

    if not npc_name or not reference or not messages:
        return None

    history_lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        text = msg.get("text", "").strip()
        if not text:
            continue
        if role == "assistant":
            history_lines.append(f"{npc_name}：{text}")
        else:
            history_lines.append(f"用户：{text}")

    if not history_lines:
        return None

    return {
        "user_id": f"RAIDEN_{npc_name}",
        "question_id": f"RAIDEN_{dialogue_id}",
        "role": npc_name,
        "profile_text": "",
        "input": "\n".join(history_lines),
        "output": reference,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Step 1: all_dialogues.json ----
    src = RAW_DATA_DIR / "RAIDEN" / "release_data" / "dialogue.json"
    with open(src, "r", encoding="utf-8") as f:
        data: dict = json.load(f)
    print(f"Loaded {len(data)} dialogue entries from {src}")

    all_samples: list[dict] = []
    skipped = 0
    for did, entry in data.items():
        sample = build_single_sample(did, entry)
        if sample:
            all_samples.append(sample)
        else:
            skipped += 1

    out_dir = PROCESSED_DIR / "RAIDEN" / "intermediate"
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
            f"Run  python src/tree_pipeline/raw_profiles_to_text.py --dataset RAIDEN  first."
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
    n_ood = max(4, len(chars) // 7)
    n_rt = max(4, len(chars) // 7)
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
