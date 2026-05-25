"""Extract initial character profiles for Star Trek TNG's 6 main characters using LLM.

Star Trek TNG provides only dialogue transcripts — no explicit character profiles.
This script:

1. Loads **Season 1** dialogue data (to capture *initial* personality only,
   so that later-season character evolution can be demonstrated by the
   dynamic attribute tree).
2. For each of the 6 main characters, collects all their utterances
   from Season 1 scenes and samples representative scenes.
3. Uses LLM with an English prompt to synthesise a structured JSON
   character profile per character.
4. Saves profiles using the same unified field schema as Friends/TheOffice.

Input
    ``phase_tree_data/raw_data/StarTrek/star_trek_tng_season_01.json``

Output
    ``phase_tree_data/processed/StarTrek_TNG/intermediate/raw_profiles.json``

Dependencies
    * ``openai``, ``python-dotenv`` — LLM API access.

Usage::

    python extract_profiles_tng.py
    python extract_profiles_tng.py --seasons 1 2 --n_sample 40
"""

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "phase_tree_data" / "raw_data"
PROCESSED_DIR = PROJECT_ROOT / "phase_tree_data" / "processed"
load_dotenv(PROJECT_ROOT / ".env")

RANDOM_SEED = 42

MAIN_CHARACTERS = [
    "Picard",
    "Riker",
    "Data",
    "Worf",
    "Troi",
    "Laforge",
]

PROFILE_FIELDS = [
    "name", "gender", "age", "occupation",
    "personality", "values_and_beliefs", "emotional_patterns",
    "speaking_style", "catchphrases",
    "behavioral_traits", "expertise_and_skills", "quirks",
    "background", "relationships", "hobbies",
    "goals_and_motivations",
]

_SYSTEM = """\
You are a character profiling specialist. You will be given dialogue \
excerpts from Season 1 of the TV show "Star Trek: The Next Generation" \
featuring a specific character. Your task is to analyse their speech \
patterns, behaviour, and relationships **as they appear in these early \
episodes only** to produce a comprehensive character profile in JSON.

## Critical rules
1. Use ONLY evidence from the provided dialogues. Do NOT use your own \
knowledge about the character's development in later seasons or any \
external source. The profile should reflect who this character is at \
the **beginning** of the series.
2. For any field where the dialogues provide NO evidence, set its value \
to null. Never guess or fabricate.
3. **Language: ALL value strings MUST be written in English.**
4. Each fact goes in exactly ONE field — no duplication across fields.
5. Keep values concise. Prefer comma-separated descriptors over long prose.

## Field definitions

### Basic info
- **name**: The character's full name if inferable, otherwise the name used.
- **gender**: Gender if clearly inferable, otherwise null.
- **age**: Age or age range if inferable, otherwise null.
- **occupation**: Job title, rank, role, or position aboard the ship.

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
register, verbal habits, sentence patterns. No actual quotes here.
- **catchphrases**: A JSON **array** of actual iconic lines or recurring \
expressions quoted verbatim from the dialogues. 2-5 examples if available. \
MUST be an array of strings.
- **behavioral_traits**: Recurring action patterns, social conduct.
- **expertise_and_skills**: Specific abilities or knowledge domains.
- **quirks**: Distinctive mannerisms, unusual habits, idiosyncrasies.

### Context
- **background**: 1-2 sentence summary of the most pivotal life events \
visible from the dialogues. Do NOT include occupation, relationships, or \
personality here.
- **relationships**: Key interpersonal connections. Format each as \
"role is Name" (e.g. "captain is Picard, friend is Geordi").
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

def load_season(season_num: int) -> dict:
    """Load a single season JSON file and return its parsed content."""
    path = RAW_DATA_DIR / "StarTrek" / f"star_trek_tng_season_{season_num:02d}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_scenes(season_nums: list[int]) -> list[dict]:
    """Return a flat list of scenes across the requested seasons."""
    scenes = []
    for sn in season_nums:
        sdata = load_season(sn)
        for ep in sdata["episodes"]:
            for sc in ep["scenes"]:
                scenes.append({
                    "scene_id": sc["scene_id"],
                    "utterances": sc["utterances"],
                })
    return scenes


def scenes_for_character(scenes: list[dict], char_name: str) -> list[dict]:
    """Keep only scenes where *char_name* speaks at least once."""
    result = []
    for sc in scenes:
        if any(char_name in u.get("speakers", []) for u in sc["utterances"]):
            result.append(sc)
    return result


def format_scene(scene: dict, max_utts: int = 30) -> str:
    """Format a scene's utterances into readable dialogue text."""
    lines = []
    for u in scene["utterances"][:max_utts]:
        speaker = ", ".join(u["speakers"])
        text = u.get("transcript", "")
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Parse JSON from LLM output, tolerating markdown fences and trailing commas."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


def _normalise_profile(raw: dict) -> dict:
    """Ensure all expected fields exist; convert empty / 'unknown' to null."""
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
    char_scenes: list[dict], n_sample: int = 30, max_retries: int = 3,
) -> dict:
    """Call the LLM to generate a character profile from sampled scene excerpts."""
    rng = random.Random(RANDOM_SEED)
    sampled = rng.sample(char_scenes, min(n_sample, len(char_scenes)))

    parts = []
    for i, sc in enumerate(sampled, 1):
        parts.append(f"[Scene {i}] ({sc['scene_id']})\n{format_scene(sc)}")
    dialogues_text = "\n\n".join(parts)

    user_prompt = (
        f"Character name: {char_name}\n\n"
        f"Below are {len(sampled)} scene excerpts from Season 1 of "
        f"Star Trek: The Next Generation featuring this character:\n\n"
        f"{dialogues_text}\n\n"
        f"Analyse the dialogues and produce the JSON profile."
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
        description="Extract initial character profiles for Star Trek TNG (6 main characters)",
    )
    ap.add_argument("--seasons", type=int, nargs="+", default=[1],
                    help="Which seasons to sample dialogues from (default: [1])")
    ap.add_argument("--n_sample", type=int, default=30,
                    help="Max scenes sampled per character for LLM analysis")
    ap.add_argument("--max_workers", type=int, default=6)
    args = ap.parse_args()

    print(f"Loading scenes from season(s): {args.seasons}")
    scenes = collect_scenes(args.seasons)
    print(f"Total scenes loaded: {len(scenes)}")

    char_scenes: dict[str, list[dict]] = {}
    for char in MAIN_CHARACTERS:
        cs = scenes_for_character(scenes, char)
        char_scenes[char] = cs
        print(f"  {char}: {len(cs)} scenes")

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
            client, model, name, char_scenes[name], args.n_sample,
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

    out_dir = PROCESSED_DIR / "StarTrek_TNG" / "intermediate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw_profiles.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {len(profiles)} profiles saved -> {out_path} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
