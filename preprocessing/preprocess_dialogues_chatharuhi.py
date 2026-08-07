"""Preprocess ChatHaruhi multi-turn dialogues and perform 3-way split.

**Step 1 — Dialogue conversion.**  Each JSONL entry becomes one
training / evaluation sample:

* For simple entries (no ``more_dialogues``):
  ``input`` = ``<user_role>：user_question``,
  ``output`` = ``agent_response``.
  The original ``user_role`` name from the raw data is preserved
  (falls back to ``用户`` when empty).

* For multi-turn entries:
  All turns are reconstructed; everything up to (but excluding) the
  last agent turn becomes ``input``; the last agent turn is ``output``.

**Step 2 — Character-level 3-way split** via profile embeddings and
K-Means clustering (identical approach to the RAIDEN pipeline).

Input
    ``LongEvoRoleBench/raw_data/ChatHaruhi/Haruhi_54K_v1.jsonl``
    ``LongEvoRoleBench/processed/ChatHaruhi/intermediate/raw_profiles.json``
    ``LongEvoRoleBench/processed/ChatHaruhi/intermediate/raw_profile_texts.json``

Output
    ``LongEvoRoleBench/processed/ChatHaruhi/intermediate/all_dialogues.json``
    ``LongEvoRoleBench/processed/ChatHaruhi/intermediate/character_embeddings.npz``
    ``LongEvoRoleBench/processed/ChatHaruhi/intermediate/{train,random_test,ood_test}.json``

Dependencies
    * ``split_utils`` — shared embedding + 3-way split logic
      (requires ``openai``, ``python-dotenv``, ``scikit-learn``).

Usage::

    python preprocess_dialogues_chatharuhi.py
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
# Name normalisation  (mirrors extract_profiles_chatharuhi.py)
# ---------------------------------------------------------------------------

NORMALIZE_MAP = {
    "哈利": "Harry", "罗恩": "Ron", "赫敏": "Hermione",
    "邓布利多": "Dumbledore", "斯内普": "Snape", "小天狼星": "Sirius",
    "麦格教授": "Professor McGonagall",
    "教授麦格": "Professor McGonagall",
    "教授麦格教": "Professor McGonagall",
    "教授麦格教嗔地说道": "Professor McGonagall",
    "教授麦格教嗔怒地说道": "Professor McGonagall",
    "教授麦格教授": "Professor McGonagall",
    "教授麦格教练": "Professor McGonagall",
    "McGonagall教授": "Professor McGonagall",
    "教授特里劳妮": "Professor Trelawney",
    "Professor Snape": "Snape",
    "萧峰": "乔峰", "萧峰见": "乔峰",
}

NOISE_NAMES = frozenset({
    "", "旁白", "Narrator", "观众", "学生", "Student", "百姓A",
    "保安", "经理", "警察", "顾客", "囚犯", "士兵", "访客",
    "老人", "老僧", "陌生人", "女子", "年轻书生", "大哥", "和尚",
    "那人", "蒙面人", "黑衣女郎", "青衣少女", "沙溢吕秀才", "大聪明",
})


def normalize_name(name: str) -> str:
    cleaned = re.sub(r"（[^）]*）", "", name).strip()
    cleaned = re.sub(r"\([^)]*\)", "", cleaned).strip()
    return NORMALIZE_MAP.get(cleaned, cleaned)


def is_noise(name: str) -> bool:
    if name in NOISE_NAMES or len(name) > 15:
        return True
    if re.match(r"^('''|Apologies|对不起|抱歉)", name):
        return True
    return False


# ---------------------------------------------------------------------------
# Language detection (mirrors extract_profiles_chatharuhi.py)
# ---------------------------------------------------------------------------

_ASCII_RE = re.compile(r"[a-zA-Z]")


def _entry_lang(entry: dict) -> str:
    """Detect the language of a single dialogue entry from its response."""
    resp = entry.get("agent_response", "")
    if not resp:
        return "zh"
    ratio = len(_ASCII_RE.findall(resp)) / len(resp)
    return "en" if ratio > 0.5 else "zh"


# ---------------------------------------------------------------------------
# Step 1: dialogue conversion
# ---------------------------------------------------------------------------

_TURN_RE = re.compile(r"^(.+?)\s*[:：]\s*(.*)", re.DOTALL)

_NOISE_PUNCT_RE = re.compile(r'[「」"]')


def _clean_text(text: str) -> str:
    """Strip CJK bracket quotes ``「」`` and double quotes ``"`` from text."""
    return _NOISE_PUNCT_RE.sub("", text).strip()


def build_single_sample(
    idx: int, entry: dict, agent_role: str, lang: str = "zh",
) -> dict | None:
    """Convert one ChatHaruhi entry into a training sample.

    Uses the original ``user_role`` name from the raw data rather than
    a generic label.  Falls back to ``User`` / ``用户`` (depending on
    *lang*) only when ``user_role`` is empty.
    CJK bracket quotes ``「」`` are stripped from all text.
    """
    user_question = _clean_text(entry.get("user_question", ""))
    agent_response = _clean_text(entry.get("agent_response", ""))
    if not user_question or not agent_response:
        return None

    raw_user_role = entry.get("user_role", "").strip()
    fallback = "User" if lang == "en" else "用户"
    user_label = raw_user_role if raw_user_role else fallback

    more = entry.get("more_dialogues", [])

    if not more:
        return {
            "user_id": f"ChatHaruhi_{agent_role}",
            "question_id": f"ChatHaruhi_{idx:06d}",
            "role": agent_role,
            "profile_text": "",
            "input": f"{user_label}：{user_question}",
            "output": agent_response,
        }

    all_turns: list[tuple[str, str]] = [
        (user_label, user_question),
        (agent_role, agent_response),
    ]
    for turn_str in more:
        m = _TURN_RE.match(turn_str)
        if not m:
            continue
        raw_speaker = m.group(1).strip()
        text = _clean_text(m.group(2).strip())
        norm_speaker = normalize_name(raw_speaker)
        speaker = agent_role if norm_speaker == agent_role else raw_speaker
        all_turns.append((speaker, text))

    last_agent_idx = -1
    for i in range(len(all_turns) - 1, -1, -1):
        if all_turns[i][0] == agent_role:
            last_agent_idx = i
            break

    if last_agent_idx <= 0:
        return {
            "user_id": f"ChatHaruhi_{agent_role}",
            "question_id": f"ChatHaruhi_{idx:06d}",
            "role": agent_role,
            "profile_text": "",
            "input": f"{user_label}：{user_question}",
            "output": agent_response,
        }

    context = "\n".join(f"{s}：{t}" for s, t in all_turns[:last_agent_idx])
    return {
        "user_id": f"ChatHaruhi_{agent_role}",
        "question_id": f"ChatHaruhi_{idx:06d}",
        "role": agent_role,
        "profile_text": "",
        "input": context,
        "output": all_turns[last_agent_idx][1],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    src = RAW_DATA_DIR / "ChatHaruhi" / "Haruhi_54K_v1.jsonl"
    raw: list[dict] = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw.append(json.loads(line))
    print(f"Loaded {len(raw)} entries from {src}")

    prof_dir = PROCESSED_DIR / "ChatHaruhi" / "intermediate"
    prof_path = prof_dir / "raw_profiles.json"
    if not prof_path.exists():
        raise FileNotFoundError(
            f"{prof_path} not found.  "
            "Run  python extract_profiles_chatharuhi.py  first."
        )
    with open(prof_path, "r", encoding="utf-8") as f:
        profiles_data = json.load(f)
    known_chars: set[str] = set(profiles_data.keys())
    char_langs: dict[str, str] = {
        k: v.get("_lang", "zh") for k, v in profiles_data.items()
    }
    print(f"Known characters (from profiles): {len(known_chars)}")

    # ---- Step 1: build samples (only for characters with profiles) ----
    all_samples: list[dict] = []
    skipped = 0
    lang_filtered = 0
    for idx, entry in enumerate(raw):
        norm = normalize_name(entry["agent_role"])
        if is_noise(norm) or norm not in known_chars:
            skipped += 1
            continue
        if _entry_lang(entry) != char_langs.get(norm, "zh"):
            lang_filtered += 1
            continue
        sample = build_single_sample(idx, entry, norm, char_langs.get(norm, "zh"))
        if sample:
            all_samples.append(sample)
        else:
            skipped += 1

    out_dir = prof_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_path = out_dir / "all_dialogues.json"
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    roles = sorted({s["role"] for s in all_samples})
    print(f"\nall_dialogues.json: {len(all_samples)} samples "
          f"({skipped} skipped, {lang_filtered} lang-filtered), "
          f"{len(roles)} chars")
    print(f"Saved -> {all_path}")

    # ---- Step 2: embeddings ----
    txt_path = out_dir / "raw_profile_texts.json"
    if not txt_path.exists():
        raise FileNotFoundError(
            f"{txt_path} not found.  "
            "Run  python src/tree_pipeline/raw_profiles_to_text.py --dataset ChatHaruhi  first."
        )
    with open(txt_path, "r", encoding="utf-8") as f:
        profile_texts: dict[str, str] = json.load(f)

    chars_with_prof = [r for r in roles if r in profile_texts]
    print(f"\nCharacters with profile text: "
          f"{len(chars_with_prof)}/{len(roles)}")

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

    r_train = {s["role"] for s in splits["train"]}
    r_rt = {s["role"] for s in splits["random_test"]}
    r_ood = {s["role"] for s in splits["ood_test"]}
    assert not (r_train & r_rt), "Train / Random test role overlap!"
    assert not (r_train & r_ood), "Train / OOD test role overlap!"
    assert not (r_rt & r_ood), "Random test / OOD test role overlap!"
    print("\n✓ No role overlap between any splits")


if __name__ == "__main__":
    main()
