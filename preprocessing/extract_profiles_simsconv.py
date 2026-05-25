"""Extract raw character profiles from the SimsConv dataset.

Parse ``Instructed.jsonl``, extract each character's free-form English
narrative profile, deduplicate by character name, and save as
``raw_profiles.json`` for downstream consumption by
``src/tree_pipeline/profiles_to_trees.py``.

The instruction field contains three parts separated by markers:

1. **Profile** (before "Respond and answer like …")
2. **Scene status** (between "The status of you is as follows:" and
   "The interactions are as follows:")
3. **Dialogue** (in the ``output`` field)

Three instruction preamble patterns have been observed:

* ``You are <Name>, a …``
* ``You're <Name>, a …``
* ``You go by the name of <Name>. …``

Input
    ``phase_tree_data/raw_data/SimsConv/Instructed.jsonl``

Output
    ``phase_tree_data/processed/SimsConv/intermediate/raw_profiles.json``

Usage::

    python extract_profiles_simsconv.py
"""

import json
import re
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "phase_tree_data" / "raw_data"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "phase_tree_data" / "processed"

_NAME_PATTERNS = [
    re.compile(r"^You are ([^,]+),"),
    re.compile(r"^You're ([^,]+),"),
    re.compile(r"^You go by the name of ([^.]+)\."),
]


def extract_name(instruction: str) -> str:
    """Extract the protagonist's name from the instruction preamble."""
    for pat in _NAME_PATTERNS:
        m = pat.search(instruction)
        if m:
            return m.group(1).strip()
    return ""


def extract_profile_text(instruction: str) -> str:
    """Return the profile paragraph (everything before the role-play marker)."""
    marker = "Respond and answer like"
    idx = instruction.find(marker)
    return instruction[:idx].strip() if idx > 0 else instruction.strip()


def parse_basic_fields(profile_text: str) -> dict:
    """Regex-extract reliable demographic facts from the narrative.

    Fields ``age`` and ``gender`` are always present; set to ``None``
    when extraction fails so that the output schema is uniform.
    """
    fields: dict[str, str | None] = {"age": None, "gender": None}

    # Age: "28-year-old", "28 year-old", "male of 35 years"
    m = re.search(r"(\d{1,3})[- ]year[- ]old", profile_text, re.IGNORECASE)
    if not m:
        m = re.search(r"\b(?:male|female|man|woman)\s+of\s+(\d{1,3})\s+years\b",
                       profile_text, re.IGNORECASE)
    if m:
        fields["age"] = m.group(1)

    # Gender: "28-year-old male", "a male of 35 years", "28-year-old jovial guy"
    m = re.search(
        r"\b(\d{1,3})[- ]year[- ]old\s+(?:\w+\s+)*(male|female|man|woman|guy|gal)\b",
        profile_text,
        re.IGNORECASE,
    )
    if not m:
        m = re.search(r"\ba\s+(male|female|man|woman)\s+of\s+\d{1,3}\s+years\b",
                       profile_text, re.IGNORECASE)
    if m:
        g = m.group(len(m.groups())).lower()
        fields["gender"] = "Male" if g in ("male", "man", "guy") else "Female"

    return fields


def main() -> None:
    src = RAW_DATA_DIR / "SimsConv" / "Instructed.jsonl"
    with open(src, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Loaded {len(lines)} entries from {src}")

    profiles: dict[str, dict] = {}
    skipped = 0

    for line in lines:
        entry = json.loads(line)
        name = extract_name(entry["instruction"])
        if not name:
            skipped += 1
            continue

        if name in profiles:
            continue

        profile_text = extract_profile_text(entry["instruction"])
        fields = parse_basic_fields(profile_text)
        fields["name"] = name
        fields["full_profile_text"] = profile_text
        profiles[name] = fields

    out_dir = PROCESSED_DIR / "SimsConv" / "intermediate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw_profiles.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(profiles)} profiles -> {out_path}")
    if skipped:
        print(f"Skipped {skipped} entries (name not extracted)")

    for name, prof in list(profiles.items())[:3]:
        print(f"\n--- {name} ---")
        print(f"  age={prof.get('age')}, gender={prof.get('gender')}")
        txt = prof["full_profile_text"]
        print(f"  profile: {txt[:150]}...")


if __name__ == "__main__":
    main()
