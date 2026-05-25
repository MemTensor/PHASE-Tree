"""Extract per-dialogue session & moment for the long-term track.

Mirrors :mod:`update_session_moment` (short-term) but uses the **time-evolved
persona snapshot** as the LLM context, instead of the frozen ``p00`` tree.

For each dialogue sample (``role`` + ``input`` + ``output``) the script:

1. Resolves the speaking character and the dialogue's episode tag.
2. Looks up the persona snapshot **from the previous episode** (so the
   persona used to interpret a dialogue at S05E08 is the snapshot saved at
   the end of S05E07 — preventing temporal leakage of updates that were
   themselves derived from S05E08's sessions).
3. Prompts the LLM with that prev-episode persona + the full dialogue to
   produce a per-dialogue ``session`` and ``moment`` (same schema as the
   short-term flow).
4. Stores the LLM output keyed by ``question_id``, along with metadata
   pointing to which snapshot was used.

Inputs
------
``phase_tree_data/processed/<dataset>/intermediate/<split>.json``
``phase_tree_data/processed/<dataset>/intermediate/attribute_trees.json``
``phase_tree_data/processed/<dataset>/intermediate/evolution/persona_snapshots.json``

Outputs
-------
``phase_tree_data/processed/<dataset>/intermediate/<split>_long_term_moments.json``

Schema (per ``question_id``)::

    {
      "session": {learned_info, attitude_shifts, commitments, stance_changes},
      "moment":  {emotion, emotion_intensity, scene_context},
      "_persona_from_episode": "S05E07",
      "_persona_version":      "p32"
    }

Concurrency: ``asyncio`` + ``AsyncOpenAI``; checkpoint every 50 samples;
``--resume`` skips already-processed question IDs.

Usage::

    python extract_long_term_moments.py --dataset Friends --split all
    python extract_long_term_moments.py --dataset Friends --split train --workers 32
    python extract_long_term_moments.py --dataset Friends --split ood_test --resume
    python extract_long_term_moments.py --dataset Friends --split ood_test --max_samples 10
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

# Re-use the prompt + helpers from the short-term script so the per-dialogue
# session/moment schema is identical between the two tracks.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tree_pipeline.update_session_moment import (  # type: ignore
    SYSTEM_PROMPT,
    build_user_prompt,
    extract_json_from_response,
    persona_to_summary,
    validate_update,
    _is_english_tree,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED_DIR = PROJECT_ROOT / "phase_tree_data" / "processed"

CHECKPOINT_EVERY = 100


# ---------------------------------------------------------------------------
# Snapshot resolution (persona at T-1)
# ---------------------------------------------------------------------------

def _ep_ord(season: int, episode: int) -> int:
    """Comparable integer for ``(season, episode)``.  Friends has < 30 eps
    per season so season*100 + episode is collision-free.
    """
    return int(season) * 100 + int(episode)


def _ep_ord_tag(ep_tag: str) -> int:
    m = re.match(r"S(\d+)E(\d+)", ep_tag)
    return _ep_ord(int(m.group(1)), int(m.group(2))) if m else -1


def _resolve_persona_at_T_minus_1(
    snapshots: dict,
    base_tree: dict,
    char: str,
    season: int,
    episode: int,
) -> tuple[dict, str | None, str]:
    """Return ``(tree_at_t_minus_1, source_episode_tag, source_version)``.

    ``snapshots`` is the loaded ``persona_snapshots.json`` keyed by
    ``character`` → ``ep_tag`` → ``{version, tree}``.  The function picks the
    snapshot from the **strictly previous** episode (or the latest episode
    with ``ord < ord(season, episode)``).  If no prior snapshot exists (e.g.
    the dialogue is from the very first appearance episode), falls back to
    the initial p00 tree from ``attribute_trees.json``.
    """
    cur_ord = _ep_ord(season, episode)
    char_snaps = snapshots.get(char, {})

    chosen_ep: str | None = None
    chosen_ord = -1
    for ep_tag in char_snaps:
        o = _ep_ord_tag(ep_tag)
        if o < 0 or o >= cur_ord:
            continue
        if o > chosen_ord:
            chosen_ord = o
            chosen_ep = ep_tag

    if chosen_ep is None:
        # No prior snapshot — fall back to the initial tree.
        return copy.deepcopy(base_tree), None, base_tree.get(
            "persona", {}).get("version", "p00")

    snap = char_snaps[chosen_ep]
    tree = copy.deepcopy(snap["tree"])
    return tree, chosen_ep, snap.get("version", tree.get("persona", {}).get(
        "version", "?"))


# ---------------------------------------------------------------------------
# Async LLM call
# ---------------------------------------------------------------------------

async def process_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    role: str,
    persona_tree: dict,
    dialogue_input: str,
    dialogue_output: str,
    max_retries: int = 3,
) -> dict | None:
    """Call the LLM once for a single sample's session/moment extraction."""
    persona_summary = persona_to_summary(persona_tree)
    en = _is_english_tree(persona_tree)

    # Build the dialogue payload by appending the speaker's reply so the LLM
    # sees the FULL exchange (mirrors short-term semantics).
    full_dialogue = (
        f"{dialogue_input}\n{role}: {dialogue_output}".strip()
        if dialogue_output
        else dialogue_input
    )

    user_prompt = build_user_prompt(
        role, persona_summary, full_dialogue, use_english=en,
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
                    max_tokens=1500,
                    timeout=120.0,
                )
                raw = extract_json_from_response(
                    resp.choices[0].message.content)
                if "session" not in raw or "moment" not in raw:
                    raise ValueError("missing session/moment")
                return validate_update(raw)
            except json.JSONDecodeError as e:
                sys.stderr.write(
                    f"\n  [{role}] JSON err attempt {attempt}: {e}\n")
            except Exception as e:
                sys.stderr.write(
                    f"\n  [{role}] err attempt {attempt}: {e}\n")
            if attempt < max_retries:
                await asyncio.sleep(2 * attempt)
    return None


async def process_sample(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    sample: dict,
    snapshots: dict,
    base_trees: dict,
) -> tuple[str, dict | None]:
    role = sample["role"]
    qid = sample["question_id"]
    season = sample.get("_season")
    episode = sample.get("_episode")

    base_tree = base_trees.get(role)
    if base_tree is None:
        return qid, None

    if season is None or episode is None:
        # Without timeline metadata fall back to the frozen p00 tree —
        # behaviour matches short-term in that pathological case.
        persona_tree, src_ep, src_ver = base_tree, None, base_tree.get(
            "persona", {}).get("version", "p00")
    else:
        persona_tree, src_ep, src_ver = _resolve_persona_at_T_minus_1(
            snapshots, base_tree, role, season, episode,
        )

    update = await process_one(
        client, semaphore, model, role, persona_tree,
        sample.get("input", ""), sample.get("output", ""),
    )
    if not update:
        return qid, None
    update["_persona_from_episode"] = src_ep
    update["_persona_version"] = src_ver
    return qid, update


# ---------------------------------------------------------------------------
# Split runner
# ---------------------------------------------------------------------------

async def run_split(
    client: AsyncOpenAI,
    model: str,
    samples: list[dict],
    snapshots: dict,
    base_trees: dict,
    out_path: Path,
    workers: int,
    resume: bool,
) -> dict:
    existing: dict = {}
    if resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"  Resuming: {len(existing)} samples already processed")

    todo = [
        s for s in samples
        if s["role"] in base_trees
        and s["question_id"] not in existing
    ]
    if not todo:
        print(f"  All {len(samples)} samples already done.")
        return existing

    print(f"  Processing {len(todo)} samples (workers={workers})...")

    semaphore = asyncio.Semaphore(workers)
    results = dict(existing)
    failed: list[str] = []
    done = 0
    t0 = time.time()

    tasks = [
        asyncio.create_task(
            process_sample(client, semaphore, model, s, snapshots, base_trees)
        )
        for s in todo
    ]

    for coro in asyncio.as_completed(tasks):
        qid, update = await coro
        done += 1
        if update is not None:
            results[qid] = update
        else:
            failed.append(qid)
            sys.stderr.write(f"\n  [FAILED] {qid}\n")

        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(todo) - done) / rate if rate > 0 else 0
        bar_w = 30
        pct = done / len(todo)
        bar = "█" * int(bar_w * pct) + "░" * (bar_w - int(bar_w * pct))
        sys.stdout.write(
            f"\r  {bar} {pct:5.1%} | {done}/{len(todo)} | "
            f"{rate:.1f} it/s | ETA {eta:.0f}s"
            f"{' | ' + str(len(failed)) + ' failed' if failed else ''}"
        )
        sys.stdout.flush()

        if done % CHECKPOINT_EVERY == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    elapsed = time.time() - t0
    print(f"\n  Saved {len(results)} -> {out_path}  "
          f"({elapsed:.1f}s; {len(failed)} failed)")
    return results


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

    base_trees = json.loads(
        (intermediate_dir / "attribute_trees.json").read_text())
    print(f"Loaded {len(base_trees)} base attribute trees")

    snap_path = evolution_dir / "persona_snapshots.json"
    if not snap_path.exists():
        raise FileNotFoundError(
            f"persona_snapshots.json not found at {snap_path} — run "
            f"evolve_persona.py first.")
    snapshots = json.loads(snap_path.read_text())
    print(f"Loaded persona snapshots for {len(snapshots)} characters")

    # Split files may have been generated from all_dialogues.json with
    # temporal metadata (``_season``, ``_episode``, ``_scene_id``) stripped
    # out.  Reload it and build a question_id -> metadata index so we can
    # always resolve the right snapshot.
    #
    # HPD uses ``_book`` + ``_position`` ("Book1-chapter2") instead of
    # ``_season`` + ``_episode``.  Normalise here so the downstream
    # snapshot lookup (keyed by S##E## tags) works without further
    # branching.
    all_dialogues_path = intermediate_dir / "all_dialogues.json"
    qid_meta: dict[str, dict] = {}
    if all_dialogues_path.exists():
        with open(all_dialogues_path, "r", encoding="utf-8") as f:
            all_samples = json.load(f)
        for s in all_samples:
            qid = s.get("question_id")
            if qid:
                season = s.get("_season")
                episode = s.get("_episode")
                if season is None:
                    season = s.get("_book")
                if episode is None:
                    pos = s.get("_position")
                    if pos:
                        m = re.search(r"chapter(\d+)", pos, re.IGNORECASE)
                        if m:
                            episode = int(m.group(1))
                scene_id = s.get("_scene_id") or s.get("_session_id")
                qid_meta[qid] = {
                    "_season": season,
                    "_episode": episode,
                    "_scene_id": scene_id,
                }
        print(f"Indexed {len(qid_meta)} dialogue metadata entries from "
              f"all_dialogues.json")

    splits = (
        ["train", "random_test", "ood_test"]
        if args.split == "all" else [args.split]
    )

    for split_name in splits:
        suffix = "_full" if args.use_full else ""
        split_path = intermediate_dir / f"{split_name}{suffix}.json"
        if not split_path.exists():
            print(f"\n[{split_name}] Not found: {split_path}, skipping")
            continue

        with open(split_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        # Backfill missing temporal metadata from the all_dialogues index.
        for s in samples:
            qid = s.get("question_id")
            if qid and qid in qid_meta:
                meta = qid_meta[qid]
                for k, v in meta.items():
                    if s.get(k) is None and v is not None:
                        s[k] = v

        valid = [
            s for s in samples
            if s.get("role") in base_trees
            and s.get("_season") is not None
            and s.get("_episode") is not None
        ]
        skipped = len(samples) - len(valid)

        print(f"\n{'='*60}")
        print(f"[{split_name}] {len(valid)} samples ({skipped} skipped)")
        print("=" * 60)

        if args.max_samples > 0:
            valid = valid[:args.max_samples]

        out_path = intermediate_dir / f"{split_name}_long_term_moments.json"
        await run_split(
            client, model, valid, snapshots, base_trees, out_path,
            workers=args.workers, resume=args.resume,
        )

    if args.split == "all":
        _merge_to_all_dialogues(intermediate_dir)


def _merge_to_all_dialogues(intermediate_dir: Path) -> None:
    merged: dict = {}
    for split in ["train", "random_test", "ood_test"]:
        p = intermediate_dir / f"{split}_long_term_moments.json"
        if not p.exists():
            print(f"  [merge] {split} file not found, skipping")
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged.update(data)
        print(f"  [merge] {split}: {len(data)} entries")

    out_path = intermediate_dir / "all_dialogues_long_term_moments.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\n  Merged {len(merged)} entries -> {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Extract per-dialogue session/moment using time-evolved "
                    "persona snapshots (long-term track).")
    ap.add_argument("--dataset", type=str, required=True,
                    choices=["Friends", "TheOffice", "StarTrek_TNG", "HPD"])
    ap.add_argument("--split", type=str, required=True,
                    choices=["train", "random_test", "ood_test", "all"])
    ap.add_argument("--max_samples", type=int, default=0,
                    help="Limit per split (0 = all)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--use_full", action="store_true",
                    help="Read from <split>_full.json instead of <split>.json")
    args = ap.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
