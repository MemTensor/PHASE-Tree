"""Post-process updated attribute trees to fix first-person perspective leaks.

The LLM sometimes generates session/moment content using first-person pronouns
("my", "me", "I" in English; "我", "我的" in Chinese) instead of third-person.
This script fixes those occurrences in-place within the *_phase_updated_trees.json files.

Usage::

    python fix_perspective.py --dataset SimsConv --split all --dry_run
    python fix_perspective.py --dataset SimsConv --split all
    python fix_perspective.py --dataset RAIDEN --split all
    python fix_perspective.py --dataset CharacterEval --split all
    python fix_perspective.py --dataset ChatHaruhi --split all
"""

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "LongEvoRoleBench" / "processed"

# ── English pronoun replacement ──────────────────────────────────────────

_PRONOUN_MAP = {
    "Male":   {"my": "his",  "me": "him",  "myself": "himself",  "mine": "his",  "I": None},
    "Female": {"my": "her",  "me": "her",  "myself": "herself",  "mine": "hers", "I": None},
    "Other":  {"my": "their","me": "them", "myself": "themselves","mine": "theirs","I": None},
}

_EN_PRONOUNS_RE = re.compile(
    r"""
    (?<![A-Za-z])       # not preceded by a letter (avoid "Miami", "admit")
    (I|my|me|myself|mine)
    (?![A-Za-z])        # not followed by a letter (avoid "myself-" false positive is fine)
    """,
    re.VERBOSE,
)


def _fix_english(text: str, gender: str, char_name: str) -> str:
    """Replace first-person pronouns with third-person equivalents."""
    pmap = _PRONOUN_MAP.get(gender, _PRONOUN_MAP["Other"])
    first_name = char_name.split()[0]
    # strip title-like prefixes
    if first_name in ("Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sgt.", "Sgt",
                      "Sergeant", "Captain", "Commander", "Sir", "Lady"):
        parts = char_name.split()
        first_name = parts[1] if len(parts) > 1 else parts[0]
    # strip nickname quotes
    first_name = first_name.strip("'\"")

    def _replace(m: re.Match) -> str:
        word = m.group(1)
        if word == "I":
            return first_name
        key = word.lower()
        replacement = pmap.get(key, word)
        if replacement is None:
            return first_name
        # preserve original capitalisation for sentence-start
        if word[0].isupper() and key != "i":
            return replacement.capitalize()
        return replacement

    return _EN_PRONOUNS_RE.sub(_replace, text)


# ── Chinese pronoun replacement ──────────────────────────────────────────

_ZH_SELF_COMPOUNDS = frozenset({
    "自我怀疑", "自我改变", "自我投资", "自我身份", "自我保护",
    "自我反省", "自我提升", "自我认知", "自我介绍", "自我牺牲",
    "自我调节", "自我管理", "自我实现", "自我成长", "自我价值",
    "自我表达", "自我意识", "自我评价", "自我肯定", "自我否定",
    "自我安慰", "自我激励", "自我约束", "自我修复", "自我完善",
    "自我控制", "自我欺骗", "自我毁灭", "自我救赎", "自我定位",
    "持自我", "坚持自我",
})

_ZH_VERB_BEFORE = (
    "对|找|知道|以为|认为|觉得|见到|看到|看|见|邀请|期待|夸|夸奖"
    "|称|问|询问|请|让|帮|关心|喜欢|欢迎|尊重|了解|理解|注意"
    "|崇拜|羡慕|感谢|提醒|告诉|告知|鼓励|安慰|说服|嘲笑|批评"
    "|赞|赞扬|佩服|推荐|带|害|要|给|向|和|与|跟|服|望|待|享"
    "|到|说|过|听|有|去|可|误认为|质疑|是|适合|陪|约|记|写"
    "|发现|记得|怀疑|把|建议|好奇|听闻|认出|模仿|满足|为|如|管"
    "|信|闻|出|遇见|遇|称呼|关注|注|叫做|称为|成为|当做"
    "|了"
)

_ZH_MAIN_RE = re.compile(
    rf"(?:{_ZH_VERB_BEFORE})我(?!们)",
)

_ZH_MY_RE = re.compile(
    r"(?<![《\u300a])我的(?![》\u300b])",
)


_ZH_QUOTE_RE = re.compile(r'["\u201c][^"\u201d]*?我[^"\u201d]*?["\u201d]')


def _fix_chinese(text: str) -> str:
    """Replace first-person references with '其' in Chinese text."""
    # Protect quoted speech: replace 我 inside quotes with placeholder
    quoted_spans = []
    for m in _ZH_QUOTE_RE.finditer(text):
        quoted_spans.append((m.start(), m.end()))

    if quoted_spans:
        chars = list(text)
        for s, e in quoted_spans:
            for i in range(s, e):
                if chars[i] == "我":
                    chars[i] = "\ufffd"  # placeholder
        text = "".join(chars)

    for comp in _ZH_SELF_COMPOUNDS:
        if comp in text:
            placeholder = comp.replace("自我", "自__SELF__")
            text = text.replace(comp, placeholder)

    text = _ZH_MY_RE.sub("其", text)
    text = _ZH_MAIN_RE.sub(lambda m: m.group(0).replace("我", "其"), text)
    # Sentence-initial 我 (start of string or after comma/period/semicolon)
    text = re.sub(r"^我(?!们)", "其", text)
    text = re.sub(r"(?<=[，。；,;])我(?!们)", "其", text)

    text = text.replace("__SELF__", "我")
    text = text.replace("\ufffd", "我")  # restore quoted 我
    return text


# ── Tree-level processing ───────────────────────────────────────────────

def _is_english_tree(tree: dict) -> bool:
    name = tree.get("identity", {}).get("name") or ""
    return sum(c < "\u0080" for c in name) > len(name) * 0.5


def _fix_value(val, is_en: bool, gender: str, char_name: str):
    """Recursively fix first-person in a session/moment value."""
    if isinstance(val, str):
        if is_en:
            return _fix_english(val, gender, char_name)
        else:
            return _fix_chinese(val)
    elif isinstance(val, list):
        return [_fix_value(v, is_en, gender, char_name) for v in val]
    elif isinstance(val, dict):
        return {k: _fix_value(v, is_en, gender, char_name) for k, v in val.items()}
    return val


def fix_tree(tree: dict, gender_map: dict | None = None) -> tuple[dict, int]:
    """Fix perspective in one tree. Returns (fixed_tree, num_changes)."""
    is_en = _is_english_tree(tree)
    char_name = tree.get("identity", {}).get("name", "")

    if is_en and gender_map:
        gender = gender_map.get(char_name, "Other")
    elif is_en:
        gender = tree.get("identity", {}).get("gender") or "Other"
    else:
        gender = ""

    changes = 0
    for layer_key in ("session", "moment"):
        layer = tree.get(layer_key)
        if not layer or not isinstance(layer, dict):
            continue
        for field_key, field_val in layer.items():
            original = json.dumps(field_val, ensure_ascii=False)
            fixed_val = _fix_value(field_val, is_en, gender, char_name)
            fixed_str = json.dumps(fixed_val, ensure_ascii=False)
            if fixed_str != original:
                layer[field_key] = fixed_val
                changes += 1

    return tree, changes


def process_dataset(dataset: str, splits: list[str], dry_run: bool):
    intermediate = PROCESSED_DIR / dataset / "intermediate"

    # Build gender map for SimsConv from raw_profiles
    gender_map = None
    raw_profiles_path = intermediate / "raw_profiles.json"
    if raw_profiles_path.exists():
        with open(raw_profiles_path, encoding="utf-8") as f:
            raw = json.load(f)
        gender_map = {}
        for name, prof in raw.items():
            g = prof.get("gender")
            if g and g.lower() in ("male", "m"):
                gender_map[name] = "Male"
            elif g and g.lower() in ("female", "f"):
                gender_map[name] = "Female"
            else:
                gender_map[name] = "Other"

    for split in splits:
        path = intermediate / f"{split}_phase_updated_trees.json"
        if not path.exists():
            print(f"[{dataset}/{split}] File not found: {path}, skipping")
            continue

        with open(path, encoding="utf-8") as f:
            trees = json.load(f)

        total_changes = 0
        affected_samples = 0

        for qid in trees:
            tree, n = fix_tree(trees[qid], gender_map)
            if n > 0:
                trees[qid] = tree
                total_changes += n
                affected_samples += 1

        print(f"[{dataset}/{split}] {len(trees)} trees, "
              f"{affected_samples} samples fixed, "
              f"{total_changes} field-level changes")

        if affected_samples > 0 and not dry_run:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(trees, f, ensure_ascii=False, indent=2)
            print(f"  -> saved to {path}")
        elif dry_run and affected_samples > 0:
            print(f"  -> [DRY RUN] would save to {path}")

            # Show some examples
            count = 0
            with open(path, encoding="utf-8") as f:
                orig_trees = json.load(f)
            for qid in orig_trees:
                orig = json.dumps(orig_trees[qid].get("session", {}), ensure_ascii=False)
                fixed = json.dumps(trees[qid].get("session", {}), ensure_ascii=False)
                if orig != fixed:
                    print(f"\n  Example [{qid}]:")
                    print(f"    BEFORE: {orig[:200]}")
                    print(f"    AFTER:  {fixed[:200]}")
                    count += 1
                    if count >= 3:
                        break


def main():
    parser = argparse.ArgumentParser(
        description="Fix first-person perspective in updated attribute trees"
    )
    parser.add_argument("--dataset", required=True,
                        choices=["CharacterEval", "RAIDEN", "ChatHaruhi", "SimsConv", "Friends", "TheOffice", "StarTrek_TNG", "HPD"],
                        help="Dataset to fix")
    parser.add_argument("--split", required=True,
                        choices=["train", "random_test", "ood_test", "all"],
                        help="Which split to process")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only report changes without writing files")
    args = parser.parse_args()

    splits = (["train", "random_test", "ood_test"]
              if args.split == "all" else [args.split])

    process_dataset(args.dataset, splits, args.dry_run)

    if args.split == "all":
        merged_path = (PROCESSED_DIR / args.dataset / "intermediate"
                       / "all_dialogues_phase_updated_trees.json")
        if merged_path.exists():
            print(f"\nProcessing merged file: {merged_path.name}")
            process_dataset(args.dataset, ["all_dialogues"], args.dry_run)


if __name__ == "__main__":
    main()
