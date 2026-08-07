"""Naive-Rewrite baseline: LLM rewrites raw profile + context into a coherent paragraph.

For each dialogue sample the script:

1. Looks up the character's raw profile text (one per character).
2. Combines it with the dialogue context (``input`` field).
3. Sends both to an LLM which produces a coherent, condensed character
   introduction — **without any structured / time-scale concepts**.
4. Saves the per-sample rewritten text so that ``inject_profiles.py``
   can consume it via ``--method naive_rewrite``.

The prompt intentionally avoids mentioning attribute trees, session,
moment, or any time-scale vocabulary, ensuring a fair "unstructured
LLM rewrite" baseline.

Concurrency is handled via ``asyncio`` + ``AsyncOpenAI`` with
incremental checkpointing so that long runs can be safely resumed.

Input
    ``LongEvoRoleBench/processed/<dataset>/intermediate/raw_profile_texts.json``
    ``LongEvoRoleBench/processed/<dataset>/intermediate/<split>.json``

Output
    ``LongEvoRoleBench/processed/<dataset>/intermediate/<split>_naive_rewrite_profile_texts.json``

Dependencies
    * ``openai``, ``python-dotenv`` — LLM API access (configured in
      ``.env`` at the project root).

Usage::

    python naive_rewrite.py --dataset RAIDEN --split ood_test
    python naive_rewrite.py --dataset RAIDEN --split all
    python naive_rewrite.py --dataset RAIDEN --split train --workers 32
    python naive_rewrite.py --dataset RAIDEN --split ood_test --max_samples 5
    python naive_rewrite.py --dataset RAIDEN --split ood_test --resume
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED_DIR = PROJECT_ROOT / "LongEvoRoleBench" / "processed"

CHECKPOINT_EVERY = 50

# ---------------------------------------------------------------------------
# Prompts — Chinese / English
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_ZH = """\
你是一个角色介绍撰写助手。给定一个角色的人物设定和当前对话上下文，请基于这两部分\
整理一段连贯的角色介绍，作为之后生成该角色回复的参考。

## 要求

1. **综合 profile 与对话上下文**：既要保留角色的核心设定（身份、性格、背景经历、\
关系等），也要反映对话中体现出的当前状态和情境信息。
2. **忠于原始信息**：只使用所提供的人物设定和对话中出现的信息，不要凭自身知识\
补充或编造 profile 中没有的事实。
3. **输出一段连续的自然语言段落**，不要使用 JSON、表格、列表、小标题等结构化格式。
4. **使用第三人称**描述角色，不要使用"我""你"等第一/二人称。
5. **不要对输出字数做限制**，根据 profile 和对话内容的丰富程度自然决定长度。\
如果信息较多可以写得详细一些，信息较少则简短即可。
6. **直接输出角色介绍段落**，不要添加额外的引导语、标题或总结语。
7. **使用中文输出。**"""

SYSTEM_PROMPT_EN = """\
You are a character profile writer. Given a character's profile/setting and \
the current dialogue context, synthesize a coherent character introduction \
paragraph that can serve as reference for generating the character's future \
replies.

## Requirements

1. **Integrate both the profile and the dialogue context**: retain the \
character's core setting (identity, personality, backstory, relationships, \
etc.) while also reflecting the current situation and state shown in the \
dialogue.
2. **Stay faithful to the source material**: use only information from the \
provided profile and dialogue. Do not add facts, events, or details from \
your own knowledge that are not present in the inputs.
3. **Output a single continuous natural-language paragraph** — no JSON, \
tables, bullet lists, headings, or any structured format.
4. **Use third person** to describe the character. Do not use "I" or "you".
5. **No word-count constraint** — let the length be determined naturally by \
how much information the profile and dialogue provide. Write more when there \
is richer information; keep it short when information is sparse.
6. **Output the character introduction paragraph directly** — do not add \
any leading phrases, titles, or closing summaries.
7. **Write in English.**"""


def _is_chinese(text: str) -> bool:
    """Heuristic: if >30% of non-whitespace chars are CJK, treat as Chinese."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return True
    cjk = sum(1 for c in chars if "\u4e00" <= c <= "\u9fff")
    return cjk / len(chars) > 0.3


def build_user_prompt(profile_text: str, dialogue_input: str,
                      *, use_chinese: bool) -> str:
    if use_chinese:
        return (
            f"## 角色人物设定\n{profile_text}\n\n"
            f"## 当前对话上下文\n{dialogue_input}"
        )
    return (
        f"## Character Profile\n{profile_text}\n\n"
        f"## Current Dialogue Context\n{dialogue_input}"
    )


# ---------------------------------------------------------------------------
# Async processing
# ---------------------------------------------------------------------------

async def rewrite_one_sample(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    profile_text: str,
    dialogue_input: str,
    use_chinese: bool,
    max_retries: int = 3,
) -> str | None:
    system = SYSTEM_PROMPT_ZH if use_chinese else SYSTEM_PROMPT_EN
    user = build_user_prompt(profile_text, dialogue_input,
                             use_chinese=use_chinese)

    async with semaphore:
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=4096,
                )
                content = resp.choices[0].message.content.strip()
                if content:
                    return content
                raise ValueError("Empty response from LLM")
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
    profile_text: str,
    use_chinese: bool,
) -> tuple[str, str | None]:
    qid = sample["question_id"]
    result = await rewrite_one_sample(
        client, semaphore, model, profile_text,
        sample["input"], use_chinese,
    )
    return qid, result


async def run_split(
    client: AsyncOpenAI,
    model: str,
    raw_profiles: dict[str, str],
    samples: list[dict],
    out_path: Path,
    workers: int,
    resume: bool,
):
    existing: dict[str, str] = {}
    if resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"  Resuming: {len(existing)} samples already processed")

    todo = [s for s in samples if s["question_id"] not in existing]
    if not todo:
        print(f"  All {len(samples)} samples already done, skipping.")
        return existing

    total = len(todo)
    print(f"  Processing {total} samples (workers={workers})...")

    semaphore = asyncio.Semaphore(workers)
    results = dict(existing)
    failed: list[str] = []
    done = 0
    t0 = time.time()

    tasks = []
    for s in todo:
        profile = raw_profiles.get(s["role"], "")
        use_zh = _is_chinese(profile)
        tasks.append(
            asyncio.create_task(
                process_sample_task(client, semaphore, model, s, profile, use_zh)
            )
        )

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
        qid, text = await coro
        done += 1

        if text:
            results[qid] = text
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(args):
    model = os.getenv("LLM_MODEL", "gpt-4.1")
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")

    if not api_key:
        raise RuntimeError("LLM_API_KEY not set in .env")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    intermediate_dir = PROCESSED_DIR / args.dataset / "intermediate"

    raw_path = intermediate_dir / "raw_profile_texts.json"
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_profiles: dict[str, str] = json.load(f)
    print(f"Loaded {len(raw_profiles)} raw profile texts")

    if args.split == "all":
        splits = ["train", "random_test", "ood_test"]
    else:
        splits = [args.split]

    for split_name in splits:
        split_path = intermediate_dir / f"{split_name}.json"
        if not split_path.exists():
            print(f"\n[{split_name}] File not found: {split_path}, skipping")
            continue

        with open(split_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        valid = [s for s in samples if s["role"] in raw_profiles]
        skipped = len(samples) - len(valid)

        print(f"\n{'='*60}")
        print(f"[{split_name}] {len(valid)} samples "
              f"({skipped} skipped, no profile)")
        print(f"{'='*60}")

        if args.max_samples > 0:
            valid = valid[:args.max_samples]

        out_path = intermediate_dir / f"{split_name}_naive_rewrite_profile_texts.json"
        await run_split(
            client, model, raw_profiles, valid, out_path,
            workers=args.workers, resume=args.resume,
        )

    if args.split == "all":
        _merge_to_all_dialogues(intermediate_dir)


def _merge_to_all_dialogues(intermediate_dir: Path) -> None:
    """Merge train/random_test/ood_test naive rewrite results into all_dialogues."""
    merged: dict[str, str] = {}
    for split in ["train", "random_test", "ood_test"]:
        p = intermediate_dir / f"{split}_naive_rewrite_profile_texts.json"
        if not p.exists():
            print(f"  [merge] {split} file not found, skipping")
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged.update(data)
        print(f"  [merge] {split}: {len(data)} entries")

    out_path = intermediate_dir / "all_dialogues_naive_rewrite_profile_texts.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\n  Merged {len(merged)} entries -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Naive-Rewrite: LLM rewrites raw profile + context "
                    "into a coherent character introduction paragraph"
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
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
