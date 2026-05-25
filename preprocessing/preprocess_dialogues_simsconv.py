"""Preprocess SimsConv multi-turn dialogues and perform 3-way split.

**Step 1 — Dialogue conversion.**  Each entry in ``Instructed.jsonl`` is
converted into a single training/evaluation sample.

Two versions are produced:

* ``all_dialogues.json`` (clean): only ``speaking`` turns — used for
  constructing final training samples.
* ``all_dialogues_full.json`` (full): both ``speaking`` and ``thinking``
  turns — used as input for ``src/tree_pipeline/update_session_moment.py`` so the LLM has
  richer context for accurate session/moment state annotation.

**Step 2 — Character-level 3-way split.**  Profile embeddings are
computed via the embedding API, then K-Means clustering selects outlier
clusters for the OOD set.  The same split is applied to both versions.

Input
    ``phase_tree_data/raw_data/SimsConv/Instructed.jsonl``
    ``phase_tree_data/processed/SimsConv/intermediate/raw_profile_texts.json``

Output
    ``phase_tree_data/processed/SimsConv/intermediate/all_dialogues.json``
    ``phase_tree_data/processed/SimsConv/intermediate/all_dialogues_full.json``
    ``phase_tree_data/processed/SimsConv/intermediate/character_embeddings.npz``
    ``phase_tree_data/processed/SimsConv/intermediate/{train,random_test,ood_test}.json``
    ``phase_tree_data/processed/SimsConv/intermediate/{train,random_test,ood_test}_full.json``

Dependencies
    * ``split_utils`` — shared embedding + 3-way split logic
      (requires ``openai``, ``python-dotenv``, ``scikit-learn``).

Usage::

    python preprocess_dialogues_simsconv.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from split_utils import compute_character_embeddings, three_way_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "phase_tree_data" / "raw_data"
PROCESSED_DIR = PROJECT_ROOT / "phase_tree_data" / "processed"

_NAME_PATTERNS = [
    re.compile(r"^You are ([^,]+),"),
    re.compile(r"^You're ([^,]+),"),
    re.compile(r"^You go by the name of ([^.]+)\."),
]

_TURN_RE = re.compile(r"^(.+?)\s*\((speaking|thinking)\):\s*(.*)", re.DOTALL)

# ---------------------------------------------------------------------------
# Narration / stage-direction cleaning
# ---------------------------------------------------------------------------
# SimsConv uses novel-style formatting where narration is interleaved with
# speech via double-quote boundaries:
#   speech," Narrator action. "more speech
# After cleaning, only the spoken text should remain.

# Mid-narration with continuation (handles optional ' before "):
#   speech[,.?!](')?" narration-text "continuation
# Uses lazy [^"\n]+? so it stops at the first space-" (the continuation opener).
_NARR_MID_RE = re.compile(r"""([,.\?!])('?)" [^"\n]{3,}? \"""")

# Trailing narration at end of string:
#   speech[,.?!](')?" narration$
# Requires a space right after " to avoid matching orphan close-quotes like ."\n
_NARR_TRAIL_RE = re.compile(r"""([,.\?!])('?)" [^"\n]{3,}$""")

# Initial narration before the opening " of actual speech.
# \b word boundaries prevent false positives (e.g. "lit" inside "responsibility").
_NARR_INIT_VERBS = re.compile(
    r"\b(?:said|says|replied|replies|asked|asks|began|begins|countered|counters|"
    r"murmured|murmurs|mused|muses|exclaimed|exclaims|remarked|remarks|"
    r"chuckled|chuckles|sighed|sighs|whispered|whispers|responded|responds|"
    r"continued|continues|paused|pauses|grinned|grins|smiled|smiles|"
    r"laughed|laughs|spoke|speaks|added|adds|noted|notes|conceded|concedes|"
    r"suggested|suggests|offered|offers|chimed|chimes|shrugged|shrugs|nodded|nods|"
    r"motioned|motions|called|calls|concluded|concludes|quipped|quips|"
    r"declared|declares|announced|announces|admitted|admits|observed|observes|"
    r"leaned|leans|stood|stands|sat|sits|walked|walks|turned|turns|"
    r"interrupted|interrupts|interjected|recovered|recovers|switched|switches|"
    r"cuts|cut|rumbles|rumbled|sparkling|sparkled|lights|lit|"
    r"trailed|trails|directed|directs|running|ran|"
    r"rises|rose|risen|blushed|blushes|retorted|retorts|"
    r"overheard|overhears|giggled|giggles|waved|waves|"
    r"gestured|gestures|pointed|points|glanced|glances|"
    r"muttered|mutters|cried|cries|shouted|shouts|"
    r"snapped|snaps|growled|growls|hissed|hisses|"
    r"stammered|stammers|stuttered|stutters|pleaded|pleads|"
    r"boomed|booms|drawled|drawls|purred|purrs|"
    r"hollered|hollers|bellowed|bellows|barked|barks|"
    r"teased|teases|joked|jokes|mocked|mocks|"
    r"snarled|snarls|sneered|sneers|scoffed|scoffs|"
    r"cheered|cheers|clapped|claps|hugged|hugs)\b", re.IGNORECASE,
)
_NARR_INIT_POSSESSIVE = re.compile(
    r"[A-Z]\w+(?:'s|'s)\s+(?:voice|eyes|tone|gaze|laughter|words|face|"
    r"melodic|whisper|expression|smile|sigh|hand|hands|mind|cheeks|"
    r"deep\s+voice|soft\s+voice|lips|brow|shoulders|head|heart)",
    re.IGNORECASE,
)
_NARR_INIT_RE = re.compile(r'^([^"]+?)[,.]?\s*"')


def clean_narration(text: str) -> str:
    """Remove novel-style stage directions embedded in dialogue text.

    SimsConv raw data sometimes contains narration like::

        I came to bring you this," Cassidy said, holding up an object. "It's a keepsake

    This function strips those narration segments, keeping only the speech,
    then removes any remaining orphan double-quote characters that are
    artifacts of the novel formatting.
    """
    if '"' not in text:
        return text

    # --- Phase 1: strip narration text ------------------------------------

    # 1a. Mid-sentence narration with continuation (loop for chained cases):
    prev = None
    while prev != text:
        prev = text
        text = _NARR_MID_RE.sub(lambda m: m.group(1) + m.group(2) + " ", text)

    # 1b. Trailing narration at end (no continuation):
    text = _NARR_TRAIL_RE.sub(lambda m: m.group(1) + m.group(2), text)

    # 1c. Initial narration before first opening ":
    #     Narration prefixes are short (< 150 chars).  The length guard avoids
    #     false positives where a long speech paragraph happens to contain a
    #     verb from the list (e.g. "Rose" matching the verb "rose").
    m = _NARR_INIT_RE.match(text)
    if m:
        prefix = m.group(1)
        if len(prefix) < 150 and (
            _NARR_INIT_VERBS.search(prefix) or _NARR_INIT_POSSESSIVE.search(prefix)
        ):
            text = text[m.end():]

    # --- Phase 2: strip orphan " characters -------------------------------
    # After narration removal, any remaining " are novel-formatting artifacts
    # (paragraph-break quotes, inline phrase quotes, trailing close-quotes).
    text = text.replace('"', '')

    text = re.sub(r"  +", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Step 1: dialogue conversion
# ---------------------------------------------------------------------------

def extract_protagonist(instruction: str) -> str:
    """Extract the protagonist name from the instruction preamble."""
    for pat in _NAME_PATTERNS:
        m = pat.search(instruction)
        if m:
            return m.group(1).strip()
    return ""


def parse_turns(output_text: str) -> list[dict]:
    """Parse the ``output`` field into a list of structured turns."""
    raw_turns = output_text.split("<|eot|>")
    turns = []
    for raw in raw_turns:
        raw = raw.strip()
        if not raw:
            continue
        m = _TURN_RE.match(raw)
        if m:
            turns.append({
                "speaker": m.group(1).strip(),
                "action": m.group(2),
                "content": m.group(3).strip(),
            })
    return turns


def _flatten_content(text: str) -> str:
    """Collapse internal newlines in a single turn's content into spaces.

    SimsConv speaking turns can span multiple paragraphs.  Since ``\\n`` is
    used as the turn delimiter in the formatted input, internal newlines must
    be removed so that each line corresponds to exactly one turn.
    """
    return re.sub(r"\s*\n\s*", " ", text).strip()


def build_samples(
    dialogue_id: str,
    protagonist: str,
    turns: list[dict],
) -> tuple[dict | None, dict | None]:
    """Build one clean sample and one full sample from parsed turns.

    * **clean**: only ``speaking`` turns, all speakers keep their
      original names.
    * **full**: ``speaking`` + ``thinking``, thinking turns formatted as
      ``Speaker (thinking): content``.

    Returns ``(clean_sample, full_sample)``.  Either can be ``None`` when
    no usable output turn exists.
    """
    speaking_turns = [t for t in turns if t["action"] == "speaking"]
    if not speaking_turns:
        return None, None

    # Find the last speaking turn by the protagonist
    last_protag_idx = None
    for i in range(len(speaking_turns) - 1, -1, -1):
        if speaking_turns[i]["speaker"] == protagonist:
            last_protag_idx = i
            break

    if last_protag_idx is None or last_protag_idx == 0:
        return None, None

    output_text = _flatten_content(
        clean_narration(speaking_turns[last_protag_idx]["content"])
    )

    # --- Clean version: speaking turns before the output ---
    clean_lines = []
    for t in speaking_turns[:last_protag_idx]:
        content = _flatten_content(clean_narration(t["content"]))
        clean_lines.append(f"{t['speaker']}: {content}")

    # --- Full version: all turns up to (and including) the last speaking turn ---
    # We include all turns (speaking + thinking) that occur before the
    # protagonist's last speaking turn in the original turn order.
    orig_last_idx = turns.index(speaking_turns[last_protag_idx])
    full_lines = []
    for t in turns[:orig_last_idx]:
        content = _flatten_content(clean_narration(t["content"]))
        if t["action"] == "thinking":
            full_lines.append(f"{t['speaker']} (thinking): {content}")
        else:
            full_lines.append(f"{t['speaker']}: {content}")

    if not clean_lines:
        return None, None

    base = {
        "user_id": f"SimsConv_{protagonist}",
        "question_id": f"SimsConv_{protagonist}_{dialogue_id}",
        "role": protagonist,
        "profile_text": "",
    }

    clean_sample = {**base, "input": "\n".join(clean_lines), "output": output_text}
    full_sample = {**base, "input": "\n".join(full_lines), "output": output_text}
    return clean_sample, full_sample


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Step 1: dialogue conversion ----
    src = RAW_DATA_DIR / "SimsConv" / "Instructed.jsonl"
    with open(src, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Loaded {len(lines)} entries from {src}")

    clean_samples: list[dict] = []
    full_samples: list[dict] = []
    skipped = 0

    for i, line in enumerate(lines):
        entry = json.loads(line)
        protagonist = extract_protagonist(entry["instruction"])
        if not protagonist:
            skipped += 1
            continue

        turns = parse_turns(entry["output"])
        dialogue_id = entry.get("source", f"dialogue_{i}")

        clean, full = build_samples(dialogue_id, protagonist, turns)
        if clean and full:
            clean_samples.append(clean)
            full_samples.append(full)
        else:
            skipped += 1

    out_dir = PROCESSED_DIR / "SimsConv" / "intermediate"
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_path = out_dir / "all_dialogues.json"
    with open(clean_path, "w", encoding="utf-8") as f:
        json.dump(clean_samples, f, ensure_ascii=False, indent=2)

    full_path = out_dir / "all_dialogues_full.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(full_samples, f, ensure_ascii=False, indent=2)

    roles = sorted({s["role"] for s in clean_samples})
    print(f"\nall_dialogues.json (clean): {len(clean_samples)} samples "
          f"({skipped} skipped), {len(roles)} roles")
    print(f"all_dialogues_full.json (full): {len(full_samples)} samples")
    print(f"Saved -> {clean_path}")
    print(f"Saved -> {full_path}")

    # ---- Step 2: embeddings ----
    prof_path = out_dir / "raw_profile_texts.json"
    if not prof_path.exists():
        raise FileNotFoundError(
            f"{prof_path} not found. "
            f"Run  python src/tree_pipeline/raw_profiles_to_text.py --dataset SimsConv  first."
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
    print(f"\nTarget split: train≈{len(chars) - n_ood - n_rt}, "
          f"random_test≈{n_rt}, ood_test≈{n_ood}")

    train_c, rt_c, ood_c = three_way_split(chars, emb, n_rt, n_ood)

    print(f"\nFinal split: train={len(train_c)}, "
          f"random_test={len(rt_c)}, ood_test={len(ood_c)}")
    print(f"  Train chars:       {train_c}")
    print(f"  Random test chars: {rt_c}")
    print(f"  OOD test chars:    {ood_c}")

    # Build role->samples maps for both versions
    role_clean: dict[str, list[dict]] = defaultdict(list)
    role_full: dict[str, list[dict]] = defaultdict(list)
    for s in clean_samples:
        role_clean[s["role"]].append(s)
    for s in full_samples:
        role_full[s["role"]].append(s)

    split_chars = {
        "train": train_c,
        "random_test": rt_c,
        "ood_test": ood_c,
    }

    for name, char_list in split_chars.items():
        # Clean version
        samples = [s for c in char_list for s in role_clean[c]]
        path = out_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        n_roles = len({s["role"] for s in samples})
        print(f"  {name}.json: {len(samples)} samples, {n_roles} chars")

        # Full version
        full_s = [s for c in char_list for s in role_full[c]]
        path_full = out_dir / f"{name}_full.json"
        with open(path_full, "w", encoding="utf-8") as f:
            json.dump(full_s, f, ensure_ascii=False, indent=2)
        print(f"  {name}_full.json: {len(full_s)} samples")

    # Verify zero overlap
    r_train = {s["role"] for s in [s for c in train_c for s in role_clean[c]]}
    r_rt = {s["role"] for s in [s for c in rt_c for s in role_clean[c]]}
    r_ood = {s["role"] for s in [s for c in ood_c for s in role_clean[c]]}
    assert not (r_train & r_rt), "Train / Random test role overlap!"
    assert not (r_train & r_ood), "Train / OOD test role overlap!"
    assert not (r_rt & r_ood), "Random test / OOD test role overlap!"
    print("\n✓ No role overlap between any splits")


if __name__ == "__main__":
    main()
