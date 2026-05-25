"""Convert raw character profiles into standardised attribute trees via LLM.

For datasets that ship structured character profile data (e.g.
CharacterEval), this script sends each profile to an LLM which
semantically analyses every field and organises the information into
a four-layer attribute tree (identity / persona / session / moment).
API calls run concurrently via ``ThreadPoolExecutor`` to maximise
throughput.

Datasets **without** pre-existing profile files (e.g. Friends) require
a dialogue-based construction path and should NOT use this script.

Input
    ``phase_tree_data/processed/<dataset>/intermediate/raw_profiles.json``

Output
    ``phase_tree_data/processed/<dataset>/intermediate/attribute_trees.json``

Dependencies
    * ``openai``, ``python-dotenv`` — LLM API access (configured in
      ``.env`` at the project root).

Usage::

    # Process all characters (default 8 workers)
    python profiles_to_trees.py --dataset CharacterEval

    # Limit to 5 characters for quick testing
    python profiles_to_trees.py --dataset CharacterEval --max_chars 5

    # Custom concurrency
    python profiles_to_trees.py --dataset CharacterEval --workers 16
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED_DIR = PROJECT_ROOT / "phase_tree_data" / "processed"

# ---------------------------------------------------------------------------
# System prompt (English, universal across all datasets)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a character attribute extraction specialist. Your task is to read a \
fictional character's raw profile data (which may contain irregular, redundant, \
or heterogeneous fields) and extract it into a standardized attribute tree JSON.

## CRITICAL RULES — READ BEFORE ANYTHING ELSE

1. **USE ONLY THE PROVIDED DATA.** Extract information EXCLUSIVELY from the raw \
profile JSON given to you. Do NOT add, infer, or supplement any information from \
your own training knowledge, world knowledge, or any external source. If something \
is not explicitly stated or clearly implied in the provided JSON, you must NOT \
include it.

2. **NULL FOR MISSING FIELDS.** If the raw profile contains NO information for a \
field, set its value to null. Do NOT fabricate, guess, or generate plausible content. \
A null is always better than a hallucination.

3. **NO CHARACTER NAME IN VALUES.** The name is stored in identity.name. Do NOT \
mention the character's name inside any other value field. \
Bad: "John is loyal and brave" — Good: "Loyal and brave"

4. **SEMANTIC ANALYSIS — DO NOT BLINDLY COPY SOURCE FIELDS.** The source profile \
may group heterogeneous information under a single field. You MUST analyze each \
individual fact and route it to the ONE target field whose definition it best \
matches. Never dump an entire source field into a single target field without \
checking whether parts of it belong elsewhere. \
Example: a source field labeled "personality" might contain "diligent at work, \
likes candy, good at disguise". You must split: "diligent at work" -> \
behavioral_tendencies, "likes candy" -> hobbies, "good at disguise" -> \
behavioral_tendencies. \
Example: a source field labeled "experience" might contain "was a teacher, went \
to prison for 6 years". You must split: "teacher" -> occupation, "went to prison \
for 6 years" -> backstory.

5. **STRICT NO-DUPLICATION — ZERO TOLERANCE.** Each fact appears in exactly ONE \
field. Before writing a value, check all other fields you have already written. \
- Job titles go ONLY in occupation — not in behavioral_tendencies, not in backstory. \
- Personality traits go ONLY in personality — not restated in hobbies or backstory. \
- Relationship details go ONLY in relationships — not in backstory. \
- Demographic facts go ONLY in demographics — not in backstory. \
If a fact has already been placed in one field, omit it entirely from all others.

6. **SKIP TRIVIALLY OBVIOUS INFO.** If the character is a human in a real-world \
setting, do NOT write "human" in demographics — it carries no distinguishing value. \
Only include species/race when it is non-human or otherwise distinctive (e.g. elf, \
vampire, robot). Skip any data that is universally default.

7. **OUTPUT LANGUAGE.** Write all value strings in the SAME language as the input \
raw profile data. Field keys (identity, persona, etc.) are always in English.

8. **PARAPHRASING TOLERANCE.** You may lightly reorganize, add up to half a \
sentence of connective phrasing, or omit up to half a sentence of redundant \
detail for readability. But do NOT heavily rewrite or embellish beyond the source.

9. **NO COLONS IN VALUES — USE NATURAL LANGUAGE.** All value strings must be \
written in flowing natural language. Do NOT use "key: value" or "label: content" \
formatting with colons inside any value string. Instead, use natural-language \
connectors such as "is" / "includes" (English) or "是" / "为" / "包括" (Chinese). \
Bad: "age: 36, residence: Beijing" → Good: "36 years old, lives in Beijing" \
Bad: "ex-wife: Alice, daughter: Emma" → Good: "ex-wife is Alice, daughter is Emma" \
Bad: "residence: null, age: null, height: null" (enumerating unknown sub-items) \
Good: null (set the entire field to null when nothing is known) \
This rule applies to ALL fields without exception.

## Attribute Tree Structure

The tree has four layers: identity, persona, session, moment. \
You only fill identity and persona. Session and moment use defaults.

### identity layer (immutable facts)
- **name**: The character's formal full name only. No nicknames or aliases.
- **gender**: Gender. If not explicitly stated and cannot be clearly inferred, \
use null.
- **backstory**: A VERY BRIEF summary of the 1-2 most pivotal life turning points. \
**HARD LIMIT: 1 sentence, maximum 40 Chinese characters or 30 English words.** \
If the source has a rich story, ruthlessly compress — keep only the single most \
defining event. STRICTLY EXCLUDE: occupation/job titles (go in occupation), \
personality traits (go in personality), relationship names (go in relationships), \
demographic facts (go in demographics). When the source embeds such info, strip \
it out completely and keep only the core event. \
Bad (too long): "Imprisoned for six years, exploited after release, forced to carry \
out missions with daughter held hostage, eventually shot and killed." \
Good (compressed): "Imprisoned for six years, exploited after release, eventually \
killed during a mission." \
Bad (contains job title): "Served as a law professor, then became the party secretary, \
ultimately succumbed to corruption." \
Good (event only): "Entered politics but succumbed to corruption and fell into crime." \
Bad (contains job): "Stayed at school as a teacher, moved into the apartment due to \
dorm policy disagreements." \
Good (event only): "Moved into the apartment due to school dorm policy disagreements." \
Bad (too detailed, multiple events crammed in): "Orphaned and exiled by clan as a \
child, became the top alchemist, betrayed by disciple, awakened in a ring, took a \
new disciple, regained a body, broke through to the highest rank." \
Good (compressed): "Exiled by clan as a child, betrayed by disciple, later awakened \
and gained a new life."

### persona layer (slowly-changing traits)

Each field has a text value and a resistance level.

- **speaking_style** (resistance: core): A description of HOW the character speaks, \
NOT the quotes themselves. Analyze source fields about catchphrases, iconic quotes, \
speech style, verbal habits, or dialect features, then SUMMARIZE the speaking \
pattern in descriptive terms. \
Bad: "To be or not to be" (this is a quote, not a style description) \
Good: "Philosophical and dramatic, often uses rhetorical questions" \
Bad: "Fake brand phones are the best" (raw quote) \
Good: "Humorous and self-deprecating, uses slang and street-smart expressions" \
If no speech-related information exists in the raw data, set value to null.

- **personality** (resistance: core): Innate character traits, temperament, moral \
outlook, and emotional tendencies. \
Source fields: personality, temperament, emotional traits, moral outlook, values, \
beliefs (or equivalent fields in other languages). \
What belongs here: innate traits (e.g. "stubborn", "loyal", "introverted"), \
emotional tendencies (e.g. "inner moral conflict"), moral outlook and core beliefs \
(e.g. "values loyalty above all", "pragmatic worldview"). \
What does NOT belong here: hobbies and specific likes/dislikes go in hobbies; \
work habits (e.g. "diligent at work") go in behavioral_tendencies. \
Analyze each descriptor individually. \
If no such information exists in the raw data, set value to null.

- **behavioral_tendencies** (resistance: moderate): Recurring action patterns, \
work style, social conduct, talents, specific skills, combat techniques. This \
describes HOW the character habitually acts — not what happened to them. \
What belongs here: work ethic (e.g. "diligent and responsible"), social manner \
(e.g. "warm and approachable"), explicitly described skills or abilities. \
What does NOT belong here: \
(a) Life events or plot points (e.g. "was imprisoned", "was forced to do X", \
"leveled up to rank X") — those are backstory. \
(b) Job titles (e.g. "killer", "teacher") — those go in occupation. \
(c) Abilities inferred from a job title without explicit description. \
(d) Power levels, cultivation ranks, or progression milestones — those are \
events/achievements, not behavioral patterns. \
If nothing qualifies, set value to null.

- **hobbies** (resistance: low): Interests, hobbies, specific likes and dislikes. \
Source fields: hobbies, likes, dislikes, favorite things, interests \
(or equivalent fields in other languages). \
What belongs here: concrete interests (e.g. "likes reading", "enjoys candy"), \
specific favorite items or activities, and explicit dislikes (e.g. "hates crowds"). \
What does NOT belong here: innate personality traits (go in personality), \
professional skills (go in behavioral_tendencies). \
If no such information exists in the raw data, set value to null.

- **relationships** (resistance: low): Key interpersonal relationships ONLY. \
Each item must STRICTLY follow the pattern "ROLE is NAME" (English) or \
"ROLE为NAME" (Chinese) — nothing else. NO colons. NO plot events, activities, \
emotional descriptions, or narrative context. \
Bad: "was exploited by Boss Zhang" (event, not a relationship) \
Bad: "pursued a romance with Alice" (narrative, not a relationship entry) \
Bad: "the two share a deep master-disciple bond" (literary description) \
Bad: "had a complicated relationship with Bob and sacrificed for him" (events) \
Bad: "once loved Alice, eventually married someone else" (events, not role+name) \
Good: "wife is Alice, daughter is Emma, students include Bob and Charlie" \
Good: "mentor is Gandalf, best friend is Sam" \
If the source mentions events or emotions alongside relationships, strip them \
and keep ONLY "role + name". Keep only the most important relationships. \
If no such information exists in the raw data, set value to null.

- **occupation** (resistance: low): Job title, professional role, social role. \
Keep it concise. All job titles and role labels belong here and ONLY here. \
If no such information exists in the raw data, set value to null.

- **demographics** (resistance: low): Basic demographic info — age, height, \
weight, species/race (non-human only), residence, education, blood type, zodiac, \
etc. Write in natural language without colons. \
Bad: "age: 36, residence: Beijing" → Good: "36 years old, lives in Beijing" \
Bad: "residence: null, age: null, height: null" (listing unknown sub-items) \
Good: null (if ALL demographic items are unknown, set the entire value to null) \
Do NOT include gender (already in identity.gender), name, occupation, or \
relationships. Only include sub-items that have actual known values.

## resistance assignment (fixed)
- core: speaking_style, personality
- moderate: behavioral_tendencies
- low: hobbies, relationships, occupation, demographics

## Additional constraints
- Ignore meta-production info such as actor, voice actor, broadcast info, etc.
- Respect the character's worldview: fantasy/game abilities belong in \
behavioral_tendencies.
- Analyze THEN fill: read each source field, decide which target field each \
individual fact belongs to based on its semantic meaning. A single source field \
may need to be split across multiple target fields.

## Pre-output self-check (MANDATORY)

Before outputting, verify each field against these checks: \
1. speaking_style: Is this a DESCRIPTION of how they speak, or is it raw quotes? \
If it contains actual quotes or catchphrases verbatim, REWRITE as a style description. \
2. behavioral_tendencies: Does every fact here describe a HABITUAL pattern or \
skill? If any fact is a life event, plot point, power level, cultivation rank, \
job title, or inferred ability, REMOVE it. If nothing remains, set value to null. \
3. relationships: Does EVERY item follow the strict "ROLE is NAME" pattern? \
If any item contains narrative verbs (e.g. "once", "pursued", "helped", "became", \
"sacrificed", "protected", "experienced", "developed", "established") or emotional \
descriptions, REMOVE those phrases entirely — keep only "role + name". \
If an item is purely an event with no clear role+name, DELETE it. \
4. backstory: (a) Is it within 40 Chinese characters / 30 English words? If not, \
COMPRESS further — keep only the single most pivotal event. \
(b) Does it contain ANY job title, role name, or occupation keyword (e.g. teacher, \
secretary, prosecutor, manager, boss, professor, killer, alchemist, clan leader, \
sect master, guard — or their equivalents in the source language)? If yes, REMOVE \
that part and rephrase using only the event. \
5. For ALL fields: Is any fact duplicated across two or more fields? If yes, keep \
it in the most specific field and remove from others. \
6. For ALL fields: Does the value string contain any colon (: or ：) used as a \
key-value separator? If yes, REWRITE using natural-language connectors ("is", \
"includes", or their equivalents in the source language). \
7. For demographics and relationships: Is the value a list of "label: null" items? \
If ALL sub-items are null/unknown, set the entire field value to null.

## Output format

Output ONLY a valid JSON object. No extra text, explanations, or markdown fences.

{
  "identity": {
    "name": "...",
    "gender": "...",
    "backstory": "..." or null
  },
  "persona": {
    "speaking_style":        {"value": "..." or null, "resistance": "core"},
    "personality":           {"value": "..." or null, "resistance": "core"},
    "behavioral_tendencies": {"value": "..." or null, "resistance": "moderate"},
    "hobbies":               {"value": "..." or null, "resistance": "low"},
    "relationships":         {"value": "..." or null, "resistance": "low"},
    "occupation":            {"value": "..." or null, "resistance": "low"},
    "demographics":          {"value": "..." or null, "resistance": "low"}
  }
}"""


def build_user_prompt(char_key: str, raw_profile: dict) -> str:
    """Build the user prompt with the raw profile data."""
    profile_json = json.dumps(raw_profile, ensure_ascii=False, indent=2)
    return (
        f"Below is the raw profile data for the character \"{char_key}\" in JSON format.\n"
        f"Extract and organize it into a standardized attribute tree following the "
        f"system instructions.\n\n"
        f"REMINDER:\n"
        f"- Use ONLY the data below; do NOT use your own knowledge about this character.\n"
        f"- Analyze each individual fact and place it in the SINGLE best-matching field.\n"
        f"- A single source field may need to be SPLIT across multiple target fields.\n"
        f"- NEVER duplicate any fact across fields.\n"
        f"- Work habits and social conduct go in behavioral_tendencies, NOT personality.\n"
        f"- Hobbies, likes, dislikes go in hobbies, NOT personality.\n"
        f"- Job titles go ONLY in occupation.\n"
        f"- Do NOT include the character's name in any value string.\n\n"
        f"```json\n{profile_json}\n```"
    )


NAME_FIELD_CANDIDATES = ["name", "姓名", "中文名", "角色名", "character_name", "npc_name"]


def make_character_id(char_key: str, raw_profile: dict) -> str:
    """Generate a normalized character_id from the profile."""
    name = char_key
    for key in NAME_FIELD_CANDIDATES:
        if key in raw_profile:
            name = raw_profile[key]
            break
    if isinstance(name, list):
        name = name[0]
    cid = name.strip().lower().replace(" ", "_").replace("·", "_")
    cid = re.sub(r"[^\w\u4e00-\u9fff]", "", cid)
    return cid or char_key


def extract_json_from_response(text: str) -> dict:
    """Robustly extract JSON from LLM response, handling markdown fences and trailing commas."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


LONG_TERM_DATASETS = {"Friends", "StarTrek_TNG", "TheOffice", "HPD"}


def build_full_tree(char_key: str, raw_profile: dict, extracted: dict,
                    *, dialogue_mode: str = "short_term") -> dict:
    """Assemble the complete attribute tree with metadata and defaults."""
    character_id = make_character_id(char_key, raw_profile)
    tree = {
        "tree_id": f"{character_id}::p00",
        "character_id": character_id,
        "dialogue_mode": dialogue_mode,
        "identity": extracted.get("identity", {}),
        "persona": {
            "version": "p00",
            **extracted.get("persona", {}),
        },
        "session": {
            "learned_info": [],
            "attitude_shifts": {},
            "commitments": [],
            "stance_changes": [],
        },
        "moment": {
            "emotion": "neutral",
            "emotion_intensity": 3,
            "scene_context": None,
        },
    }
    return tree


def process_character(
    client: OpenAI,
    model: str,
    char_key: str,
    raw_profile: dict,
    max_retries: int = 3,
    dialogue_mode: str = "short_term",
) -> dict | None:
    """Call LLM to extract attribute tree for one character."""
    user_prompt = build_user_prompt(char_key, raw_profile)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=2000,
            )
            content = resp.choices[0].message.content
            extracted = extract_json_from_response(content)

            if "identity" not in extracted or "persona" not in extracted:
                raise ValueError("Missing required top-level keys")

            return build_full_tree(char_key, raw_profile, extracted,
                                   dialogue_mode=dialogue_mode)

        except json.JSONDecodeError as e:
            print(f"    [attempt {attempt}] JSON parse error: {e}")
        except Exception as e:
            print(f"    [attempt {attempt}] Error: {e}")

        if attempt < max_retries:
            time.sleep(2 * attempt)

    return None


def _worker(client: OpenAI, model: str, char_key: str, raw_profile: dict,
            dialogue_mode: str = "short_term") -> tuple:
    """Thread worker that returns ``(char_key, tree_or_None)``."""
    tree = process_character(client, model, char_key, raw_profile,
                             dialogue_mode=dialogue_mode)
    return char_key, tree


def main():
    parser = argparse.ArgumentParser(
        description="Build attribute trees from raw profiles using LLM"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["CharacterEval", "RAIDEN", "ChatHaruhi", "SimsConv", "Friends", "TheOffice", "StarTrek_TNG", "HPD"],
        help="Dataset to process",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=0,
        help="Max characters to process (0 = all). Useful for testing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent API workers (default: 8)",
    )
    args = parser.parse_args()

    model = os.getenv("LLM_MODEL", "gpt-4.1")
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")

    if not api_key:
        raise RuntimeError("LLM_API_KEY not set in .env")

    client = OpenAI(api_key=api_key, base_url=base_url)

    processed_dir = PROCESSED_DIR / args.dataset / "intermediate"
    raw_profiles_path = processed_dir / "raw_profiles.json"
    if not raw_profiles_path.exists():
        raise FileNotFoundError(
            f"{raw_profiles_path} not found. "
            f"Run the corresponding extract_profiles script first."
        )

    with open(raw_profiles_path, "r", encoding="utf-8") as f:
        raw_profiles = json.load(f)

    char_keys = list(raw_profiles.keys())
    if args.max_chars > 0:
        char_keys = char_keys[: args.max_chars]

    total = len(char_keys)
    print(f"Processing {total} characters with model={model}, workers={args.workers}")
    print(f"API base: {base_url}")
    print("-" * 60)

    def _progress(done_n, total_n, n_failed, elapsed_s, width=30):
        pct = done_n / total_n if total_n else 1
        filled = int(width * pct)
        bar = "█" * filled + "░" * (width - filled)
        rate = done_n / elapsed_s if elapsed_s > 0 else 0
        eta = (total_n - done_n) / rate if rate > 0 else 0
        fail_str = f" | {n_failed} failed" if n_failed else ""
        return (f"\r{bar} {pct:5.1%} | {done_n}/{total_n} | "
                f"{rate:.1f} it/s | ETA {eta:.0f}s{fail_str}")

    trees = {}
    failed = []
    done = 0
    t0 = time.time()

    mode = "long_term" if args.dataset in LONG_TERM_DATASETS else "short_term"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_worker, client, model, ck, raw_profiles[ck], mode): ck
            for ck in char_keys
        }
        for future in as_completed(futures):
            char_key, tree = future.result()
            done += 1
            if tree:
                trees[char_key] = tree
            else:
                failed.append(char_key)
                sys.stderr.write(f"\n[FAILED] {char_key}\n")

            sys.stdout.write(_progress(done, total, len(failed),
                                       time.time() - t0))
            sys.stdout.flush()

    out_path = processed_dir / "attribute_trees.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trees, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {len(trees)} trees saved -> {out_path} "
          f"({elapsed:.1f}s, {len(failed)} failed)")


if __name__ == "__main__":
    main()
