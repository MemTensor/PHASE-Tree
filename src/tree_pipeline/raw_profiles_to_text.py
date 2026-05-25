"""Flatten raw character profiles into continuous natural-language text.

Read the structured JSON profiles produced by the extraction step and
convert each character into a single flowing paragraph using
deterministic, rule-based concatenation (no LLM involved).  The output
serves as a human-readable baseline profile and as input for embedding
computation in downstream scripts.

The output file is named ``raw_profile_texts.json`` to distinguish it
from attribute-tree-based profile flattening used at a later stage.

Input
    ``phase_tree_data/processed/<dataset>/intermediate/raw_profiles.json``

Output
    ``phase_tree_data/processed/<dataset>/intermediate/raw_profile_texts.json``

Usage::

    python raw_profiles_to_text.py --dataset CharacterEval
"""

import argparse
import json
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "phase_tree_data" / "processed"

NAME_KEYS = {"姓名", "name", "character_name", "角色名", "npc_name"}

INTRO_ORDER = [
    "性别", "物种", "年龄", "工作", "昵称", "生日", "生肖",
    "星座", "身高", "体重", "居住地", "学历", "智商", "情商",
    "恋爱状态",
]
INTRO_SET = set(INTRO_ORDER)

SKIP_VALUES = {
    "未知", "无", "暂无", "不详",
    "N/A", "n/a", "none", "None", "null", "",
}

DIRECT_FIELDS = {
    "人物经历", "角色经历", "角色背景", "评价", "角色评价",
    "形象描述", "其他",
}

PREFIX_MAP = {
    "人物性格": "性格",
    "性格": "性格",
    "爱好": "爱好",
    "喜欢的事情/东西": "喜欢",
    "不喜欢的事情/东西": "不喜欢",
    "特长": "擅长",
    "经典台词": "经典台词有",
    "口头禅": "口头禅是",
}

NO_PREFIX_FIELDS = {"人物关系", "家庭成员"}

# ---------------------------------------------------------------------------
# ChatHaruhi-specific: bilingual natural-language templates
# ---------------------------------------------------------------------------

def _fmt_list(val, lang: str) -> str:
    """Format a value (string or list) into a joined string."""
    if isinstance(val, list):
        items = [str(v).strip() for v in val if v]
        sep = "; " if lang == "en" else "、"
        return sep.join(items)
    return str(val).strip() if val else ""


def _cap(s: str) -> str:
    """Capitalize the first letter of a string."""
    return s[0].upper() + s[1:] if s else s


def _zh_prefixed(prefix: str, val: str) -> str:
    """Prepend a Chinese connector, skipping it if the value already starts
    with the same word or a close synonym to avoid duplication."""
    synonyms: dict[str, set[str]] = {
        "信奉": {"信奉", "重视", "崇尚", "相信", "坚持"},
        "追求": {"追求", "渴望", "希望", "想要", "期望"},
        "擅长": {"擅长", "善于", "精通", "精于", "熟练"},
        "习惯": {"习惯", "喜欢", "爱", "常常", "总是"},
    }
    skip_words = synonyms.get(prefix, {prefix})
    for w in skip_words:
        if val.startswith(w):
            return val
    return f"{prefix}{val}"


def _chatharuhi_profile_to_text_en(name: str, p: dict) -> str:
    """Build a flowing English paragraph from ChatHaruhi profile fields."""
    sents: list[str] = []

    intro_parts: list[str] = [name]
    if p.get("gender"):
        intro_parts.append(p["gender"].strip().lower())
    if p.get("age"):
        intro_parts.append(p["age"].strip())
    if p.get("occupation"):
        intro_parts.append(p["occupation"].strip())
    sents.append(", ".join(intro_parts) + ".")

    for key in ("personality", "values_and_beliefs", "emotional_patterns",
                "goals_and_motivations", "speaking_style"):
        if p.get(key):
            sents.append(_cap(p[key].strip()) + ".")
    if p.get("catchphrases"):
        quotes = _fmt_list(p["catchphrases"], "en")
        sents.append(f"Catchphrases include {quotes}.")
    for key in ("behavioral_traits", "expertise_and_skills", "quirks"):
        if p.get(key):
            sents.append(_cap(p[key].strip()) + ".")
    if p.get("background"):
        sents.append(_cap(p["background"].strip()))
    if p.get("relationships"):
        sents.append(_cap(p["relationships"].strip()) + ".")
    if p.get("hobbies"):
        sents.append(_cap(p["hobbies"].strip()) + ".")

    text = " ".join(sents)
    text = text.replace("..", ".").replace(". .", ".")
    return text


def _chatharuhi_profile_to_text_zh(name: str, p: dict) -> str:
    """Build a flowing Chinese paragraph from ChatHaruhi profile fields."""
    intro_parts: list[str] = [name]
    if p.get("gender"):
        intro_parts.append(p["gender"].strip())
    if p.get("age"):
        age = p["age"].strip()
        if age.isdigit():
            age += "岁"
        intro_parts.append(age)
    if p.get("occupation"):
        intro_parts.append(p["occupation"].strip())
    intro = "，".join(intro_parts) + "。"

    sents: list[str] = []
    if p.get("personality"):
        sents.append(f"性格{p['personality'].strip()}")
    if p.get("values_and_beliefs"):
        sents.append(_zh_prefixed("信奉", p["values_and_beliefs"].strip()))
    if p.get("emotional_patterns"):
        sents.append(f"情绪上{p['emotional_patterns'].strip()}")
    if p.get("goals_and_motivations"):
        sents.append(_zh_prefixed("追求", p["goals_and_motivations"].strip()))
    if p.get("speaking_style"):
        sents.append(f"说话风格{p['speaking_style'].strip()}")
    if p.get("catchphrases"):
        quotes = _fmt_list(p["catchphrases"], "zh")
        sents.append(f"口头禅有{quotes}")
    if p.get("behavioral_traits"):
        sents.append(f"行为上{p['behavioral_traits'].strip()}")
    if p.get("expertise_and_skills"):
        sents.append(_zh_prefixed("擅长", p["expertise_and_skills"].strip()))
    if p.get("quirks"):
        sents.append(_zh_prefixed("习惯", p["quirks"].strip()))
    if p.get("background"):
        sents.append(p["background"].strip())
    if p.get("relationships"):
        sents.append(p["relationships"].strip())
    if p.get("hobbies"):
        sents.append(f"爱好{p['hobbies'].strip()}")

    body = "。".join(sents)
    if body:
        body += "。"
    text = intro + body
    text = text.replace("。。", "。")
    return text


def _is_chatharuhi_profile(profile: dict) -> bool:
    """Detect ChatHaruhi-style profiles by checking for English field names."""
    return "_lang" in profile or "personality" in profile


def chatharuhi_profile_to_text(char_key: str, profile: dict) -> str:
    """Dispatch to EN or ZH template based on _lang field."""
    name = (profile.get("name") or char_key).strip()
    lang = profile.get("_lang", "zh")
    if lang == "en":
        return _chatharuhi_profile_to_text_en(name, profile)
    return _chatharuhi_profile_to_text_zh(name, profile)


# ---------------------------------------------------------------------------
# SimsConv-specific: profile is already a narrative paragraph
# ---------------------------------------------------------------------------

def _is_simsconv_profile(profile: dict) -> bool:
    """Detect SimsConv profiles by the ``full_profile_text`` field."""
    return "full_profile_text" in profile


def simsconv_profile_to_text(char_key: str, profile: dict) -> str:
    """SimsConv profiles are already natural-language paragraphs."""
    return profile.get("full_profile_text", char_key)


# ---------------------------------------------------------------------------
# Original logic for CharacterEval / RAIDEN (Chinese field names)
# ---------------------------------------------------------------------------


def _flatten(val) -> str:
    """Recursively flatten a value into a plain string."""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, list):
        parts = [_flatten(x) for x in val]
        return "、".join(p for p in parts if p)
    if isinstance(val, dict):
        parts = []
        for k, v in val.items():
            flat = _flatten(v)
            if not flat or flat in SKIP_VALUES:
                continue
            if isinstance(v, list) or "、" in flat:
                parts.append(f"{k}包括{flat}")
            else:
                parts.append(f"{k}为{flat}")
        return "，".join(parts)
    return ""


def _intro_clause(key: str, raw_val) -> Optional[str]:
    """Turn an intro-group field into a short natural-language clause."""
    val = _flatten(raw_val)
    if not val or val in SKIP_VALUES:
        return None

    transforms = {
        "性别": lambda v: v,
        "物种": lambda v: None if v == "人类" else v,
        "年龄": lambda v: (
            v if "岁" in v
            else f"{v}岁" if v.strip().isdigit()
            else f"年龄{v}"
        ),
        "工作": lambda v: v,
        "昵称": lambda v: f"又称{v}",
        "生日": lambda v: f"出生于{v}",
        "生肖": lambda v: f"属{v}",
        "星座": lambda v: v,
        "身高": lambda v: f"身高{v}",
        "体重": lambda v: f"体重{v}",
        "居住地": lambda v: f"居住于{v}",
        "学历": lambda v: v,
        "智商": lambda v: f"智商{v}",
        "情商": lambda v: f"情商{v}",
        "恋爱状态": lambda v: v,
    }

    fn = transforms.get(key)
    return fn(val) if fn else val


def profile_to_text(char_key: str, profile: dict) -> str:
    """Convert one character's profile dict into a continuous paragraph."""
    if _is_simsconv_profile(profile):
        return simsconv_profile_to_text(char_key, profile)
    if _is_chatharuhi_profile(profile):
        return chatharuhi_profile_to_text(char_key, profile)

    name = None
    for k in NAME_KEYS:
        if k in profile:
            name = _flatten(profile[k])
            break
    name = name or char_key

    intro_parts = [name]
    for key in INTRO_ORDER:
        if key in profile:
            clause = _intro_clause(key, profile[key])
            if clause:
                intro_parts.append(clause)
    intro = "，".join(intro_parts) + "。"

    used = NAME_KEYS | INTRO_SET
    sentences = []
    for key, val in profile.items():
        if key in used:
            continue
        flat = _flatten(val)
        if not flat or flat in SKIP_VALUES:
            continue

        if key in DIRECT_FIELDS:
            sentences.append(flat)
        elif key in PREFIX_MAP:
            prefix = PREFIX_MAP[key]
            if prefix and flat.startswith(prefix):
                sentences.append(flat)
            else:
                sentences.append(f"{prefix}{flat}")
        elif key in NO_PREFIX_FIELDS:
            sentences.append(flat)
        else:
            sentences.append(f"{key}{flat}")

    body = "。".join(sentences)
    if body:
        body += "。"
    text = intro + body
    text = text.replace("。。", "。")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert raw profiles to continuous natural-language text",
    )
    ap.add_argument(
        "--dataset", type=str, required=True, choices=["CharacterEval", "RAIDEN", "ChatHaruhi", "SimsConv", "Friends", "TheOffice", "StarTrek_TNG", "HPD"],
        help="Dataset to process",
    )
    args = ap.parse_args()

    src = PROCESSED_DIR / args.dataset / "intermediate" / "raw_profiles.json"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found.")

    with open(src, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    results = {}
    for key, prof in profiles.items():
        results[key] = profile_to_text(key, prof)

    out = PROCESSED_DIR / args.dataset / "intermediate" / "raw_profile_texts.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. {len(results)} profile texts -> {out}")


if __name__ == "__main__":
    main()
