"""Extract scene-level evolution session summaries for long-term datasets.

For each scene in ``all_dialogues.json``, reconstructs the full dialogue
and asks an LLM to evaluate how the scene affected each main character
present.  Results with significance ``"low"`` are discarded; the rest
are appended to per-character session archive files.

Input
    ``LongEvoRoleBench/processed/<dataset>/intermediate/all_dialogues.json``
    ``LongEvoRoleBench/processed/<dataset>/intermediate/attribute_trees.json``

Output
    ``LongEvoRoleBench/processed/<dataset>/intermediate/evolution/<Char>_session_archive.json``

Usage::

    # Process all scenes
    python extract_evolution_sessions.py --dataset Friends

    # Test on one episode only
    python extract_evolution_sessions.py --dataset Friends --test_episode S01E01

    # Resume from checkpoint
    python extract_evolution_sessions.py --dataset Friends --resume
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED_DIR = PROJECT_ROOT / "LongEvoRoleBench" / "processed"
CHECKPOINT_EVERY = 30
SCENE_BATCH_SIZE = 12  # number of scenes processed concurrently per batch

MAIN_CHARACTERS: list[str] = []  # populated at runtime from attribute_trees.json
SYSTEM_PROMPT: str = ""  # populated at runtime via _build_system_prompt(dataset)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Friendly show names per dataset (fallback to dataset key if not listed).
SHOW_NAMES: dict[str, str] = {
    "Friends": "Friends",
    "TheOffice": "The Office",
    "StarTrek_TNG": "Star Trek: The Next Generation",
    "HPD": "Harry Potter",
}


def _build_system_prompt(dataset: str) -> str:
    show = SHOW_NAMES.get(dataset, dataset)
    return SYSTEM_PROMPT_TEMPLATE.replace("{show}", show)


SYSTEM_PROMPT_TEMPLATE = """\
You are a character development analyst for the TV show "{show}". Given a \
scene's full dialogue and a specific character's current persona, evaluate \
what happened in this scene FROM THAT CHARACTER'S PERSPECTIVE and assess its \
significance for their long-term character development.

## CRITICAL RULES
1. Analyze ONLY from the specified character's perspective.
2. Use ONLY evidence from the provided dialogue. Do NOT use external knowledge \
about the show's future plot.
3. Write in THIRD PERSON. Never use "I", "my", "me".
4. ALL output must be in English.
5. If the character is barely involved or the scene has no meaningful impact \
on them, set significance to "low".

## Significance levels
- **high**: ALWAYS use "high" when the scene EXPLICITLY shows any of:
  * a relationship STATUS change (breaking up, getting together, engagement, \
marriage, divorce, learning of a pregnancy/birth, gaining/losing a child, \
losing a pet, reconciliation)
  * an occupation/career change (hired, fired, quit, promoted, audition won)
  * a major life revelation (parent's affair, learning a hidden truth, \
coming out, moving)
  * a turning-point decision the character verbally commits to
- **medium**: Notable interactions that reveal or slightly shift character \
traits (e.g., a meaningful conversation, minor conflict, learning something new) \
without producing a status change.
- **low**: The character is barely present, only makes small talk, or nothing \
in the scene has any meaningful impact on their development.

IMPORTANT: When in doubt between high and medium for a relationship/job/family \
status change, choose **high**. Status transitions MUST be high so they can \
trigger a persona update.

## Output — valid JSON only, no markdown fences
{
  "summary": "One sentence describing what this scene meant for the character (third person)",
  "significance": "high/medium/low",
  "affected_fields": ["persona fields potentially affected, e.g. personality, relationships, occupation, behavioral_tendencies"]
}

If significance is "low", summary should be brief and affected_fields can be \
an empty list."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def persona_to_summary(tree: dict) -> str:
    """Flatten a persona tree into a concise text summary for the LLM."""
    parts = []
    identity = tree.get("identity", {})
    if identity.get("name"):
        parts.append(f"Name: {identity['name']}")
    if identity.get("backstory"):
        parts.append(f"Backstory: {identity['backstory']}")

    labels = {
        "personality": "Personality",
        "speaking_style": "Speaking style",
        "behavioral_tendencies": "Behavioral tendencies",
        "relationships": "Relationships",
        "occupation": "Occupation",
    }
    persona = tree.get("persona", {})
    for field, label in labels.items():
        val = persona.get(field, {})
        if isinstance(val, dict):
            val = val.get("value")
        if val:
            parts.append(f"{label}: {val}")
    return "\n".join(parts)


def reconstruct_scene_dialogue(scene_samples: list[dict]) -> str:
    """Build the most complete dialogue text from scene samples.

    Takes the last sample's input (which contains all prior utterances)
    and appends its own role: output to get the fullest reconstruction.
    """
    last = scene_samples[-1]
    return f"{last['input']}\n{last['role']}: {last['output']}"


def identify_present_characters(scene_dialogue: str) -> list[str]:
    """Return main characters who speak in the scene dialogue."""
    return [c for c in MAIN_CHARACTERS if f"{c}:" in scene_dialogue]


def _parse_chapter(position: str | None) -> int:
    """Extract chapter number from HPD _position like 'Book1-chapter2' → 2."""
    if not position:
        return 0
    m = re.search(r"chapter(\d+)", position, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _parse_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


# ---------------------------------------------------------------------------
# Async LLM call
# ---------------------------------------------------------------------------

async def extract_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    character: str,
    persona_summary: str,
    scene_dialogue: str,
    scene_id: str,
    season: int,
    episode: int,
    max_retries: int = 3,
) -> dict | None:
    user_prompt = (
        f"## Character being analyzed: {character}\n\n"
        f"## Character's current persona (for reference):\n{persona_summary}\n\n"
        f"## Scene dialogue:\n{scene_dialogue}"
    )

    async with semaphore:
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=500,
                )
                result = _parse_json(resp.choices[0].message.content)

                sig = result.get("significance", "low").lower()
                if sig not in ("high", "medium", "low"):
                    sig = "low"

                return {
                    "scene_id": scene_id,
                    "season": season,
                    "episode": episode,
                    "summary": result.get("summary", ""),
                    "significance": sig,
                    "affected_fields": result.get("affected_fields", []),
                    "status": "active",
                }
            except json.JSONDecodeError as e:
                sys.stderr.write(
                    f"\n    [{character}@{scene_id}] JSON error "
                    f"(attempt {attempt}): {e}\n")
            except Exception as e:
                sys.stderr.write(
                    f"\n    [{character}@{scene_id}] Error "
                    f"(attempt {attempt}): {e}\n")

            if attempt < max_retries:
                await asyncio.sleep(2 * attempt)

    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def process_scenes(
    client: AsyncOpenAI,
    model: str,
    trees: dict,
    scenes: dict[str, list[dict]],
    scene_order: list[str],
    workers: int,
    evolution_dir: Path,
    resume: bool,
):
    archives: dict[str, list[dict]] = {}
    for char in MAIN_CHARACTERS:
        fpath = evolution_dir / f"{char.replace(' ', '_')}_session_archive.json"
        if resume and fpath.exists():
            with open(fpath, encoding="utf-8") as f:
                archives[char] = json.load(f)
        else:
            archives[char] = []

    processed_scenes: set[str] = set()
    if resume:
        for entries in archives.values():
            for e in entries:
                processed_scenes.add(e["scene_id"])

    semaphore = asyncio.Semaphore(workers)
    total = len(scene_order)
    new_entries = 0
    t0 = time.time()

    pending = [sid for sid in scene_order if sid not in processed_scenes]
    done = total - len(pending)
    last_checkpoint = done

    for batch_start in range(0, len(pending), SCENE_BATCH_SIZE):
        batch = pending[batch_start: batch_start + SCENE_BATCH_SIZE]

        batch_tasks = []
        batch_meta: list[tuple[str, str]] = []  # (char, scene_id)
        for scene_id in batch:
            samples = scenes[scene_id]
            dialogue = reconstruct_scene_dialogue(samples)
            present = identify_present_characters(dialogue)
            if not present:
                continue
            season = samples[0].get("_season") or samples[0].get("_book")
            episode = samples[0].get("_episode") or _parse_chapter(samples[0].get("_position"))
            for char in present:
                batch_tasks.append(
                    extract_one(
                        client, semaphore, model, char,
                        persona_to_summary(trees.get(char, {})),
                        dialogue, scene_id, season, episode,
                    )
                )
                batch_meta.append((char, scene_id))

        if batch_tasks:
            results = await asyncio.gather(*batch_tasks)
            for (char, _scene_id), res in zip(batch_meta, results):
                if res and res["significance"] != "low":
                    archives[char].append(res)
                    new_entries += 1

        done += len(batch)
        elapsed = time.time() - t0
        rate = (done - (total - len(pending))) / elapsed if elapsed > 0 else 0
        remaining = total - done
        eta = remaining / rate if rate > 0 else 0
        pct = done / total
        filled = int(30 * pct)
        bar = "█" * filled + "░" * (30 - filled)
        sys.stdout.write(
            f"\r  {bar} {pct:5.1%} | {done}/{total} scenes | "
            f"{rate:.2f} sc/s | ETA {eta:.0f}s | +{new_entries} entries")
        sys.stdout.flush()

        if done - last_checkpoint >= CHECKPOINT_EVERY:
            _save_archives(archives, evolution_dir)
            last_checkpoint = done

    _save_archives(archives, evolution_dir)
    print()
    for char in MAIN_CHARACTERS:
        active = sum(1 for e in archives[char] if e["status"] == "active")
        high = sum(1 for e in archives[char] if e["significance"] == "high")
        med = sum(1 for e in archives[char] if e["significance"] == "medium")
        print(f"  {char}: {len(archives[char])} entries "
              f"(high={high}, medium={med}, active={active})")


def _save_archives(archives: dict[str, list], evolution_dir: Path):
    for char in MAIN_CHARACTERS:
        fpath = evolution_dir / f"{char.replace(' ', '_')}_session_archive.json"
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(archives[char], f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

async def async_main(args):
    model = os.getenv("LLM_MODEL", "gpt-4.1")
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    if not api_key:
        raise RuntimeError("LLM_API_KEY not set in .env")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    intermediate_dir = PROCESSED_DIR / args.dataset / "intermediate"
    evolution_dir = intermediate_dir / "evolution"
    evolution_dir.mkdir(parents=True, exist_ok=True)

    with open(intermediate_dir / "all_dialogues.json", encoding="utf-8") as f:
        all_samples = json.load(f)
    print(f"Loaded {len(all_samples)} samples")

    with open(intermediate_dir / "attribute_trees.json", encoding="utf-8") as f:
        trees = json.load(f)
    print(f"Loaded {len(trees)} attribute trees")

    global MAIN_CHARACTERS, SYSTEM_PROMPT
    MAIN_CHARACTERS = list(trees.keys())
    SYSTEM_PROMPT = _build_system_prompt(args.dataset)
    print(f"Main characters: {MAIN_CHARACTERS}")
    print(f"Show name in prompt: {SHOW_NAMES.get(args.dataset, args.dataset)}")

    scenes: dict[str, list[dict]] = defaultdict(list)
    scene_order: list[str] = []
    for s in all_samples:
        sid = s.get("_scene_id") or s.get("_session_id")
        if not sid:
            continue
        if sid not in scenes:
            scene_order.append(sid)
        scenes[sid].append(s)

    if args.test_episode:
        m = re.match(r"S(\d+)E(\d+)", args.test_episode.upper())
        mb = re.match(r"B(\d+)C(\d+)", args.test_episode.upper())
        if m:
            ts, te = int(m.group(1)), int(m.group(2))
            scene_order = [
                sid for sid in scene_order
                if (scenes[sid][0].get("_season") or scenes[sid][0].get("_book")) == ts
                and (scenes[sid][0].get("_episode") or _parse_chapter(scenes[sid][0].get("_position"))) == te
            ]
            print(f"Test mode: S{ts:02d}E{te:02d} → {len(scene_order)} scenes")
        elif mb:
            ts, te = int(mb.group(1)), int(mb.group(2))
            scene_order = [
                sid for sid in scene_order
                if scenes[sid][0].get("_book") == ts
                and _parse_chapter(scenes[sid][0].get("_position")) == te
            ]
            print(f"Test mode: Book{ts} Chapter{te} → {len(scene_order)} scenes")
        else:
            raise ValueError(f"Invalid episode format: {args.test_episode} "
                             f"(expected S01E01 or B1C2)")

    print(f"\nProcessing {len(scene_order)} scenes "
          f"(model={model}, workers={args.workers})")
    print(f"API base: {base_url}")
    print("-" * 60)

    await process_scenes(
        client, model, trees, scenes, scene_order,
        workers=args.workers,
        evolution_dir=evolution_dir,
        resume=args.resume,
    )
    print("\nDone.")


def main():
    ap = argparse.ArgumentParser(
        description="Extract scene-level evolution sessions "
                    "for long-term datasets")
    ap.add_argument("--dataset", type=str, required=True,
                    choices=["Friends", "StarTrek_TNG", "TheOffice", "HPD"],
                    help="Dataset to process")
    ap.add_argument("--test_episode", type=str, default=None,
                    help="Process only one episode (e.g. S01E01)")
    ap.add_argument("--workers", type=int, default=16,
                    help="Max concurrent API requests (default: 16)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from checkpoint")
    args = ap.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
