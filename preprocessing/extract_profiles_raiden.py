"""Extract raw character profiles from the RAIDEN dataset.

Parse ``npc.json``, structure each character's ``npc_setting`` text
blob into a key-value dict, and save as ``raw_profiles.json`` for
downstream consumption by ``src/tree_pipeline/profiles_to_trees.py``.

Input
    ``phase_tree_data/raw_data/RAIDEN/release_data/npc.json``

Output
    ``phase_tree_data/processed/RAIDEN/intermediate/raw_profiles.json``

Usage::

    python extract_profiles_raiden.py
"""

import json
import re
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "phase_tree_data" / "raw_data"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "phase_tree_data" / "processed"

_KEY_RE = re.compile(
    r"^([\u4e00-\u9fffa-zA-Z][^:：\n]{0,35})\s*[:：]\s*(.*)"
)


def parse_npc_setting(text: str) -> dict:
    """Parse the npc_setting text blob into a {field: value} dict.

    The npc_setting field uses a loose ``key: value`` format separated
    by newlines.  Keys start with a CJK character or Latin letter and
    are followed by a colon (half- or full-width).  Values may span
    multiple lines.
    """
    result: dict[str, str] = {}
    cur_key: str | None = None
    cur_val = ""

    for raw_line in text.split("\n"):
        line = raw_line.strip().rstrip(",").rstrip("，").strip()
        if not line:
            continue

        m = _KEY_RE.match(line)
        if m:
            if cur_key is not None:
                result[cur_key] = cur_val.strip()
            cur_key = m.group(1).strip()
            cur_val = m.group(2).strip()
        else:
            if cur_key is not None:
                cur_val += " " + line

    if cur_key is not None:
        result[cur_key] = cur_val.strip()

    return result


def main() -> None:
    src = RAW_DATA_DIR / "RAIDEN" / "release_data" / "npc.json"
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} characters from {src}")

    profiles: dict[str, dict] = {}
    for char_name, char_data in data.items():
        parsed = parse_npc_setting(char_data.get("npc_setting", ""))
        profiles[char_name] = parsed

    out_dir = PROCESSED_DIR / "RAIDEN" / "intermediate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw_profiles.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(profiles)} profiles -> {out_path}")


if __name__ == "__main__":
    main()
