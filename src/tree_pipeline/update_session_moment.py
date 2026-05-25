"""Annotate attribute trees with dialogue-derived session and moment states.

For each dialogue sample the script:

1. Loads the character's **frozen** attribute tree (persona unchanged).
2. Sends the dialogue history to an LLM which analyses what the
   character learned, felt, and committed to during the conversation.
3. Merges the LLM-returned session and moment fields into a copy of
   the original tree, producing a per-sample updated attribute tree.

Concurrency is handled via ``asyncio`` + ``AsyncOpenAI`` with
incremental checkpointing so that long runs can be safely resumed.

Input
    ``phase_tree_data/processed/<dataset>/intermediate/attribute_trees.json``
    ``phase_tree_data/processed/<dataset>/intermediate/<split>.json``

Output
    ``phase_tree_data/processed/<dataset>/intermediate/<split>_phase_updated_trees.json``

Dependencies
    * ``openai``, ``python-dotenv`` — LLM API access (configured in
      ``.env`` at the project root).

Usage::

    # Process all samples in a single split (default 16 workers)
    python update_session_moment.py --dataset CharacterEval --split ood_test

    # Process every split in one command
    python update_session_moment.py --dataset CharacterEval --split all

    # Custom concurrency
    python update_session_moment.py --dataset CharacterEval --split train --workers 32

    # Quick test with limited samples
    python update_session_moment.py --dataset CharacterEval --split ood_test --max_samples 5

    # Resume from a previous checkpoint (skips already-processed samples)
    python update_session_moment.py --dataset CharacterEval --split ood_test --resume
"""

import argparse
import asyncio
import copy
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED_DIR = PROJECT_ROOT / "phase_tree_data" / "processed"

CHECKPOINT_EVERY = 50

# ---------------------------------------------------------------------------
# System prompt for session/moment update
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a character state analyst. Given a character's persona profile and a \
dialogue history, your task is to analyze WHAT CHANGED for the character during \
this conversation and output updated session and moment fields.

## CRITICAL RULES

1. **PERSONA IS FROZEN.** Do NOT modify or comment on the persona layer. Your \
job is ONLY to fill session and moment based on the dialogue.

2. **ANALYZE FROM THE CHARACTER'S PERSPECTIVE.** The "role" field tells you which \
character you are analyzing. Only track what THIS character learned, felt, or \
committed to — not other speakers.

3. **USE ONLY DIALOGUE EVIDENCE.** Every field you fill must be grounded in \
something explicitly said or clearly implied in the dialogue. Do NOT fabricate \
events or emotions not supported by the text.

4. **BE CONCISE.** Each learned_info item should be one short sentence. \
attitude_shifts values should be brief descriptions. Emotion should be a single \
word or short phrase.

5. **NULL/EMPTY FOR MISSING INFO.** If the dialogue is too short or simple to \
extract meaningful changes, return empty lists/objects. Do NOT invent content.

6. **OUTPUT LANGUAGE.** Write all values in the SAME language as the character's \
persona profile. The required language will be explicitly specified in the user \
message — follow it strictly, regardless of what language the dialogue is in.

7. **THIRD-PERSON PERSPECTIVE.** You MUST write ALL session and moment content \
in the THIRD PERSON. Never use first-person pronouns ("I", "my", "me", "myself", \
"mine", "我", "我的") to refer to the character being analyzed. Use the \
character's name or third-person pronouns (he/she/his/her/him, 他/她/其) instead.
  - WRONG: "User admires my dedication"  →  RIGHT: "User admires her dedication"
  - WRONG: "用户对我的厨艺感兴趣"  →  RIGHT: "用户对其厨艺感兴趣"

## Fields to fill

### session (cumulative within this dialogue)
- **learned_info** (list of strings): New information the character learned \
during this conversation.
  - Chinese example: ["对方对出身背景感兴趣", "用户提到了海贼王的梦想"]
  - English example: ["The user is interested in magical creatures", "Harry mentioned Voldemort's return"]
- **attitude_shifts** (object, key=person, value=description): How the \
character's attitude toward specific people changed.
  - Chinese example: {"高育良": "尊敬→怀疑", "小明": "陌生→友好"}
  - English example: {"Harry": "neutral→protective", "Malfoy": "distrust→hostility"}
- **commitments** (list of strings): Promises or decisions the character made.
  - Chinese example: ["答应帮忙调查此事", "决定暂时保密"]
  - English example: ["Agreed to help investigate", "Promised to keep the secret"]
- **stance_changes** (list of strings): Shifts in the character's position or \
viewpoint.
  - Chinese example: ["从怀疑转为确信有人通风报信"]
  - English example: ["Shifted from skepticism to certainty about the betrayal"]

### moment (state at the END of this dialogue)
- **emotion** (string): The character's dominant emotion at the conversation's \
end. Use a concise label.
  - Chinese example: "愤怒", "焦虑", "释然", "兴奋"
  - English example: "angry", "anxious", "relieved", "excited"
- **emotion_intensity** (int 1-10): How strong the emotion is.
- **scene_context** (string or null): Brief description of the scene/situation.

## Output format

Output ONLY a valid JSON object with exactly two keys: "session" and "moment". \
No extra text, explanations, or markdown fences.

{
  "session": {
    "learned_info": [...],
    "attitude_shifts": {...},
    "commitments": [...],
    "stance_changes": [...]
  },
  "moment": {
    "emotion": "...",
    "emotion_intensity": N,
    "scene_context": "..." or null
  }
}"""


def build_user_prompt(role: str, persona_summary: str,
                      dialogue_input: str, *, use_english: bool) -> str:
    lang_instruction = (
        "**You MUST write ALL output values in English.**"
        if use_english else
        "**你必须用中文撰写所有输出内容。**"
    )
    return (
        f"## Output language requirement\n{lang_instruction}\n\n"
        f"## Character being analyzed: {role}\n\n"
        f"## Character persona (frozen, for reference only):\n"
        f"{persona_summary}\n\n"
        f"## Dialogue history (analyze this to update session & moment):\n"
        f"{dialogue_input}"
    )


def _is_english_tree(tree: dict) -> bool:
    name = tree.get("identity", {}).get("name") or ""
    return sum(c < "\u0080" for c in name) > len(name) * 0.5


def persona_to_summary(tree: dict) -> str:
    en = _is_english_tree(tree)
    parts = []
    identity = tree.get("identity", {})
    if identity.get("name"):
        label = "Name" if en else "名字"
        parts.append(f"{label}: {identity['name']}")
    if identity.get("gender"):
        label = "Gender" if en else "性别"
        parts.append(f"{label}: {identity['gender']}")
    if identity.get("backstory"):
        label = "Backstory" if en else "背景"
        parts.append(f"{label}: {identity['backstory']}")

    _LABELS_EN = {
        "personality": "Personality",
        "speaking_style": "Speaking style",
        "behavioral_tendencies": "Behavioral tendencies",
        "hobbies": "Hobbies",
        "relationships": "Relationships",
        "occupation": "Occupation",
        "demographics": "Demographics",
    }
    _LABELS_ZH = {
        "personality": "性格",
        "speaking_style": "说话风格",
        "behavioral_tendencies": "行为倾向",
        "hobbies": "爱好",
        "relationships": "关系",
        "occupation": "职业",
        "demographics": "人口特征",
    }
    labels = _LABELS_EN if en else _LABELS_ZH

    persona = tree.get("persona", {})
    for field in labels:
        val = persona.get(field, {}).get("value")
        if val:
            parts.append(f"{labels[field]}: {val}")

    return "\n".join(parts)


def extract_json_from_response(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


def validate_update(update: dict) -> dict:
    session = update.get("session", {})
    moment = update.get("moment", {})

    clean_session = {
        "learned_info": session.get("learned_info", []) or [],
        "attitude_shifts": session.get("attitude_shifts", {}) or {},
        "commitments": session.get("commitments", []) or [],
        "stance_changes": session.get("stance_changes", []) or [],
    }
    for key in ["learned_info", "commitments", "stance_changes"]:
        clean_session[key] = [str(x) for x in clean_session[key] if x]

    clean_moment = {
        "emotion": moment.get("emotion", "neutral") or "neutral",
        "emotion_intensity": int(moment.get("emotion_intensity", 3) or 3),
        "scene_context": moment.get("scene_context"),
    }
    clean_moment["emotion_intensity"] = max(1, min(10, clean_moment["emotion_intensity"]))

    return {"session": clean_session, "moment": clean_moment}


def apply_update(base_tree: dict, update: dict) -> dict:
    updated = copy.deepcopy(base_tree)
    updated["session"] = update["session"]
    updated["moment"] = update["moment"]
    return updated


# ---------------------------------------------------------------------------
# Async processing
# ---------------------------------------------------------------------------

async def process_one_sample(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    role: str,
    tree: dict,
    dialogue_input: str,
    max_retries: int = 3,
) -> dict | None:
    persona_summary = persona_to_summary(tree)
    en = _is_english_tree(tree)
    user_prompt = build_user_prompt(role, persona_summary, dialogue_input,
                                    use_english=en)

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
                    max_tokens=1500,
                )
                content = resp.choices[0].message.content
                raw_update = extract_json_from_response(content)

                if "session" not in raw_update or "moment" not in raw_update:
                    raise ValueError("Missing 'session' or 'moment' in response")

                return validate_update(raw_update)

            except json.JSONDecodeError as e:
                print(f"    [attempt {attempt}] JSON parse error: {e}")
            except Exception as e:
                print(f"    [attempt {attempt}] Error: {e}")

            if attempt < max_retries:
                await asyncio.sleep(2 * attempt)

    return None


async def process_sample_task(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    sample: dict,
    tree: dict,
) -> tuple:
    role = sample["role"]
    qid = sample["question_id"]
    update = await process_one_sample(
        client, semaphore, model, role, tree, sample["input"]
    )
    if update:
        updated_tree = apply_update(tree, update)
        return qid, updated_tree
    return qid, None


async def run_split(
    client: AsyncOpenAI,
    model: str,
    all_trees: dict,
    samples: list[dict],
    out_path: Path,
    workers: int,
    resume: bool,
):
    """Process all samples in a split with async concurrency."""
    # Load existing results if resuming
    existing = {}
    if resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"  Resuming: {len(existing)} samples already processed")

    # Filter out already-processed samples
    todo = [s for s in samples if s["question_id"] not in existing]
    if not todo:
        print(f"  All {len(samples)} samples already done, skipping.")
        return existing

    total = len(todo)
    print(f"  Processing {total} samples (workers={workers})...")

    semaphore = asyncio.Semaphore(workers)
    results = dict(existing)
    failed = []
    done = 0
    t0 = time.time()

    # Create all tasks
    tasks = [
        asyncio.create_task(
            process_sample_task(client, semaphore, model, s, all_trees[s["role"]])
        )
        for s in todo
    ]

    def _progress_bar(done_n: int, total_n: int, n_failed: int,
                      elapsed_s: float, width: int = 30) -> str:
        pct = done_n / total_n if total_n else 1
        filled = int(width * pct)
        bar = "█" * filled + "░" * (width - filled)
        rate = done_n / elapsed_s if elapsed_s > 0 else 0
        eta = (total_n - done_n) / rate if rate > 0 else 0
        fail_str = f" | {n_failed} failed" if n_failed else ""
        return (f"\r  {bar} {pct:5.1%} | {done_n}/{total_n} | "
                f"{rate:.1f} it/s | ETA {eta:.0f}s{fail_str}")

    for coro in asyncio.as_completed(tasks):
        qid, updated_tree = await coro
        done += 1

        if updated_tree:
            results[qid] = updated_tree
        else:
            failed.append(qid)
            sys.stderr.write(f"\n  [FAILED] {qid}\n")

        elapsed = time.time() - t0
        sys.stdout.write(_progress_bar(done, total, len(failed), elapsed))
        sys.stdout.flush()

        if done % CHECKPOINT_EVERY == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Done: {len(results)} saved -> {out_path} "
          f"({elapsed:.1f}s, {len(failed)} failed)")

    return results


async def async_main(args):
    model = os.getenv("LLM_MODEL", "gpt-4.1")
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")

    if not api_key:
        raise RuntimeError("LLM_API_KEY not set in .env")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    intermediate_dir = PROCESSED_DIR / args.dataset / "intermediate"

    # Load attribute trees
    trees_path = intermediate_dir / "attribute_trees.json"
    with open(trees_path, "r", encoding="utf-8") as f:
        all_trees = json.load(f)
    print(f"Loaded {len(all_trees)} attribute trees")

    # Determine which splits to process
    if args.split == "all":
        splits = ["train", "random_test", "ood_test"]
    else:
        splits = [args.split]

    for split_name in splits:
        suffix = "_full" if args.use_full else ""
        split_path = intermediate_dir / f"{split_name}{suffix}.json"
        if not split_path.exists():
            print(f"\n[{split_name}] File not found: {split_path}, skipping")
            continue

        with open(split_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        # Filter to samples with matching trees
        valid = [s for s in samples if s["role"] in all_trees]
        skipped = len(samples) - len(valid)

        print(f"\n{'='*60}")
        print(f"[{split_name}] {len(valid)} samples "
              f"({skipped} skipped, no tree)")
        print(f"{'='*60}")

        if args.max_samples > 0:
            valid = valid[:args.max_samples]

        out_path = intermediate_dir / f"{split_name}_phase_updated_trees.json"
        await run_split(
            client, model, all_trees, valid, out_path,
            workers=args.workers, resume=args.resume,
        )

    if args.split == "all":
        _merge_to_all_dialogues(intermediate_dir)


def _merge_to_all_dialogues(intermediate_dir: Path) -> None:
    """Merge train/random_test/ood_test phase_updated_trees into all_dialogues."""
    merged: dict = {}
    for split in ["train", "random_test", "ood_test"]:
        p = intermediate_dir / f"{split}_phase_updated_trees.json"
        if not p.exists():
            print(f"  [merge] {split} file not found, skipping")
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged.update(data)
        print(f"  [merge] {split}: {len(data)} entries")

    out_path = intermediate_dir / "all_dialogues_phase_updated_trees.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\n  Merged {len(merged)} entries -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Update attribute tree session/moment from dialogue context"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["CharacterEval", "RAIDEN", "ChatHaruhi", "SimsConv", "Friends", "TheOffice", "StarTrek_TNG", "HPD"],
                        help="Dataset to process")
    parser.add_argument("--split", type=str, required=True,
                        choices=["train", "random_test", "ood_test", "all"],
                        help="Which split to process ('all' = train+random_test+ood_test, then merge into all_dialogues)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples per split (0 = all)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Max concurrent API requests (default: 16)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (skip already-processed)")
    parser.add_argument("--use_full", action="store_true",
                        help="Read from {split}_full.json instead of {split}.json "
                             "(for datasets with thinking turns, e.g. SimsConv)")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
