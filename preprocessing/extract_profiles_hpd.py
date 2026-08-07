"""Extract initial character profiles for HPD's 6 main characters using LLM.

HPD (Harry Potter Dialogue) provides per-session character attributes, but
these are mostly physical/external (looks, belongings, spells, lineage).
Inner personality traits are largely marked "None".

This script:

1. Loads **Book 1** sessions and collects the HPD-provided `attributes`
   for each main character (the structured external info).
2. Collects Book 1 `dialogue` entries featuring each character as
   evidence of speaking style, personality, and behaviour.
3. Uses LLM with both the structured attributes AND dialogue excerpts
   to synthesise a comprehensive character profile in our standard schema.
4. Saves profiles to the unified format.

Input
    ``LongEvoRoleBench/raw_data/HPD/EN_all.json``

Output
    ``LongEvoRoleBench/processed/HPD/intermediate/raw_profiles.json``

Dependencies
    * ``openai``, ``python-dotenv`` — LLM API access.

Usage::

    python extract_profiles_hpd.py
    python extract_profiles_hpd.py --n_sample 40
"""

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "LongEvoRoleBench" / "raw_data"
PROCESSED_DIR = PROJECT_ROOT / "LongEvoRoleBench" / "processed"
load_dotenv(PROJECT_ROOT / ".env")

RANDOM_SEED = 42

MAIN_CHARACTERS = [
    "Harry",
    "Ron",
    "Hermione",
    "Dumbledore",
    "Hagrid",
    "Snape",
]

PROFILE_FIELDS = [
    "name", "gender", "age", "occupation",
    "personality", "values_and_beliefs", "emotional_patterns",
    "speaking_style", "catchphrases",
    "behavioral_traits", "expertise_and_skills", "quirks",
    "background", "relationships", "hobbies",
    "goals_and_motivations",
]

# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a character profiling specialist. You will be given:
1. Structured character attributes from the Harry Potter books (Book 1 only).
2. Dialogue excerpts from Book 1 (Harry Potter and the Philosopher's Stone) \
featuring a specific character.

Your task is to combine the structured attributes with evidence from the \
dialogues to produce a comprehensive character profile in JSON.

## Critical rules
1. Use the provided structured attributes as factual ground truth for \
fields they cover (name, gender, age, appearance, belongings, etc.).
2. For personality, speaking style, values, emotions, and behaviour, \
analyse ONLY the provided Book 1 dialogues. Do NOT use your own knowledge \
about later books or films.
3. The profile should reflect who this character is at the **beginning** \
of the series (Book 1 / age ~11 for students).
4. For any field where NEITHER the structured attributes NOR the dialogues \
provide evidence, set its value to null. Never guess or fabricate.
5. **Language: ALL value strings MUST be written in English.**
6. Each fact goes in exactly ONE field — no duplication across fields.
7. Keep values concise. Prefer comma-separated descriptors over long prose.

## Field definitions

### Basic info
- **name**: The character's full name if inferable, otherwise first name.
- **gender**: Gender if clearly inferable, otherwise null.
- **age**: Age or age range if inferable, otherwise null.
- **occupation**: Job title, role, or social position at Hogwarts or \
in the wizarding world.

### Inner world
- **personality**: Innate character traits, temperament. Do NOT include \
values/beliefs, emotional patterns, or hobbies here.
- **values_and_beliefs**: Core worldview, moral principles, things they \
stand for. Distinct from personality traits.
- **emotional_patterns**: Typical emotional reactions, triggers, emotional \
coping style.
- **goals_and_motivations**: What drives the character, their ambitions.

### Speech & behaviour
- **speaking_style**: A DESCRIPTION of HOW the character talks — tone, \
register, verbal habits, sentence patterns, dialect features. No actual \
quotes here.
- **catchphrases**: A JSON **array** of actual iconic lines or recurring \
expressions quoted verbatim from the dialogues. 2-5 examples if available. \
MUST be an array of strings.
- **behavioral_traits**: Recurring action patterns, social conduct.
- **expertise_and_skills**: Specific abilities, magical talents, or \
knowledge domains.
- **quirks**: Distinctive mannerisms, unusual habits, idiosyncrasies.

### Context
- **background**: 1-2 sentence summary of the character's backstory \
visible from Book 1. Do NOT include occupation, relationships, or \
personality here.
- **relationships**: Key interpersonal connections. Format each as \
"role is Name" (e.g. "best friend is Ron, mentor is Dumbledore").
- **hobbies**: Interests, specific likes and dislikes.

## Output — valid JSON only, no markdown fences
{
  "name": "...",
  "gender": "..." or null,
  "age": "..." or null,
  "occupation": "..." or null,
  "personality": "..." or null,
  "values_and_beliefs": "..." or null,
  "emotional_patterns": "..." or null,
  "goals_and_motivations": "..." or null,
  "speaking_style": "..." or null,
  "catchphrases": ["...", "..."] or null,
  "behavioral_traits": "..." or null,
  "expertise_and_skills": "..." or null,
  "quirks": "..." or null,
  "background": "..." or null,
  "relationships": "..." or null,
  "hobbies": "..." or null
}"""


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_hpd_data() -> dict:
    path = RAW_DATA_DIR / "HPD" / "EN_all.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_book_sessions(data: dict, book_num: int) -> list[tuple[str, dict]]:
    """Return (session_id, session_dict) pairs for a given book."""
    prefix = f"Book{book_num}-"
    return [
        (sid, sess) for sid, sess in data.items()
        if sess.get("position", "").startswith(prefix)
    ]


def normalize_speaker(name: str) -> str:
    """Strip whitespace from speaker names."""
    return name.strip()


def sessions_for_character(
    sessions: list[tuple[str, dict]], char_name: str
) -> list[tuple[str, dict]]:
    """Filter sessions where char_name appears in speakers list."""
    result = []
    for sid, sess in sessions:
        speakers = [normalize_speaker(s) for s in sess.get("speakers", [])]
        if char_name in speakers:
            result.append((sid, sess))
    return result


def get_latest_attributes(
    sessions: list[tuple[str, dict]], char_name: str
) -> dict | None:
    """Get the most complete attributes dict for a character from Book1 sessions."""
    best = None
    best_filled = 0
    for sid, sess in sessions:
        attrs = sess.get("attributes", {})
        for key, val in attrs.items():
            if normalize_speaker(key) == char_name:
                filled = sum(
                    1 for v in val.values()
                    if v and str(v).strip() not in ("None", "", "none")
                )
                if filled > best_filled:
                    best = val
                    best_filled = filled
    return best


def format_session_dialogue(sess: dict, max_lines: int = 20) -> str:
    """Format a session's dialogue field into text."""
    dialogue = sess.get("dialogue", [])
    lines = []
    for line in dialogue[:max_lines]:
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def format_hpd_attributes(attrs: dict) -> str:
    """Format HPD structured attributes into readable text."""
    if not attrs:
        return "(No structured attributes available)"
    skip_vals = {"None", "none", "", " "}
    parts = []
    for key, val in attrs.items():
        if val and str(val).strip() not in skip_vals:
            parts.append(f"- {key}: {val}")
    return "\n".join(parts) if parts else "(All attributes are None/empty)"


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


def _normalise_profile(raw: dict) -> dict:
    skip = {"未知", "无", "暂无", "不详", "unknown", "Unknown", "N/A", "n/a",
            "none", "None", ""}
    out: dict = {}
    for key in PROFILE_FIELDS:
        val = raw.get(key)
        if key == "catchphrases":
            if isinstance(val, list):
                val = [str(v).strip() for v in val if v]
            elif isinstance(val, str) and val.strip() and val.strip() not in skip:
                val = [val.strip()]
            else:
                val = None
            out[key] = val if val else None
        else:
            if isinstance(val, list):
                val = ", ".join(str(v).strip() for v in val if v)
            if val is None or (isinstance(val, str) and val.strip() in skip):
                out[key] = None
            else:
                out[key] = val.strip() if isinstance(val, str) else val
    return out


def generate_profile(
    client: OpenAI, model: str, char_name: str,
    char_sessions: list[tuple[str, dict]],
    hpd_attrs: dict | None,
    n_sample: int = 30, max_retries: int = 3,
) -> dict:
    rng = random.Random(RANDOM_SEED)
    sampled = rng.sample(
        char_sessions, min(n_sample, len(char_sessions))
    )

    parts = []
    for i, (sid, sess) in enumerate(sampled, 1):
        dialogue_text = format_session_dialogue(sess)
        if dialogue_text:
            pos = sess.get("position", sid)
            parts.append(f"[Session {i}] ({pos})\n{dialogue_text}")
    dialogues_text = "\n\n".join(parts)

    attrs_text = format_hpd_attributes(hpd_attrs)

    user_prompt = (
        f"Character name: {char_name}\n\n"
        f"## Structured attributes from Harry Potter Book 1\n"
        f"{attrs_text}\n\n"
        f"## Dialogue excerpts from Book 1 ({len(sampled)} sessions)\n\n"
        f"{dialogues_text}\n\n"
        f"Based on the structured attributes AND dialogue evidence above, "
        f"produce the JSON profile for {char_name} as they appear in Book 1."
    )

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=2500,
            )
            raw = _extract_json(resp.choices[0].message.content)
            return _normalise_profile(raw)
        except Exception as e:
            print(f"    [{char_name} attempt {attempt}] {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)

    return {k: None for k in PROFILE_FIELDS}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract initial character profiles for HPD "
                    "(6 main Harry Potter characters)",
    )
    ap.add_argument("--n_sample", type=int, default=30,
                    help="Max sessions sampled per character for LLM analysis")
    ap.add_argument("--max_workers", type=int, default=6)
    args = ap.parse_args()

    print("Loading HPD data...")
    data = load_hpd_data()
    print(f"Total sessions: {len(data)}")

    book1_sessions = get_book_sessions(data, 1)
    print(f"Book 1 sessions: {len(book1_sessions)}")

    char_sessions: dict[str, list[tuple[str, dict]]] = {}
    char_attrs: dict[str, dict | None] = {}
    for char in MAIN_CHARACTERS:
        cs = sessions_for_character(book1_sessions, char)
        char_sessions[char] = cs
        attrs = get_latest_attributes(cs, char)
        char_attrs[char] = attrs
        n_filled = (
            sum(1 for v in attrs.values()
                if v and str(v).strip() not in ("None", "", "none"))
            if attrs else 0
        )
        print(f"  {char}: {len(cs)} sessions, "
              f"attributes filled: {n_filled}")

    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )
    model = os.getenv("LLM_MODEL", "gpt-4.1")

    print(f"\nGenerating profiles via LLM ({model}, "
          f"{args.max_workers} workers) ...")
    profiles: dict[str, dict] = {}
    t0 = time.time()

    def _gen(name: str):
        prof = generate_profile(
            client, model, name,
            char_sessions[name], char_attrs[name],
            args.n_sample,
        )
        prof["_lang"] = "en"
        return name, prof

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {pool.submit(_gen, n): n for n in MAIN_CHARACTERS}
        for i, fut in enumerate(as_completed(futs), 1):
            name, prof = fut.result()
            profiles[name] = prof
            elapsed = time.time() - t0
            print(f"  [{i}/{len(MAIN_CHARACTERS)}] {name} done ({elapsed:.1f}s)")

    out_dir = PROCESSED_DIR / "HPD" / "intermediate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw_profiles.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {len(profiles)} profiles saved -> {out_path} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
