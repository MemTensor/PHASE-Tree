"""Extract raw character profiles from the CharacterEval dataset.

Copy the original CharacterEval profile JSON to the intermediate
directory **without any transformation**, preserving the raw data as-is
for reproducibility and downstream consumption by other preprocessing
scripts.

Input
    ``phase_tree_data/raw_data/CharacterEval/data/character_profiles.json``

Output
    ``phase_tree_data/processed/CharacterEval/intermediate/raw_profiles.json``

Usage::

    python extract_profiles_charactereval.py

Note
    This is the first step of the CharacterEval preprocessing pipeline.
    Subsequent scripts (``src/tree_pipeline/profiles_to_trees.py``,
    ``src/tree_pipeline/raw_profiles_to_text.py``) consume the output
    produced here.
"""

import json
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "phase_tree_data" / "raw_data"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "phase_tree_data" / "processed"


def main() -> None:
    src = RAW_DATA_DIR / "CharacterEval" / "data" / "character_profiles.json"
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} characters from {src}")

    out_dir = PROCESSED_DIR / "CharacterEval" / "intermediate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw_profiles.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} profiles -> {out_path}")


if __name__ == "__main__":
    main()
