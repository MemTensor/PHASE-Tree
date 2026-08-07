"""Preprocess Friends multi-party dialogues into PHASE-Tree samples with temporal split.

**Step 1 — Dialogue conversion.**  For each scene across all 10 seasons,
construct training/evaluation samples where each main character's utterance
is a potential ``output``.  The ``input`` is all preceding dialogue in that
scene with all speakers' real names preserved.

**Step 2 — Temporal 3-way split.**  Unlike short-term datasets which use
clustering, Friends uses a *temporal* split to support long-term evolution:

* ``train``        — Seasons 1–7 (random 85% of qualifying samples)
* ``random_test``  — Seasons 1–7 (random 15% holdout, same epoch)
* ``ood_test``     — Seasons 8–10 (later time period — for testing
                     dynamic attribute tree evolution)

Input
    ``LongEvoRoleBench/raw_data/Friends/json/friends_season_*.json``

Output
    ``LongEvoRoleBench/processed/Friends/intermediate/all_dialogues.json``
    ``LongEvoRoleBench/processed/Friends/intermediate/{train,random_test,ood_test}.json``

Usage::

    python preprocess_dialogues_friends.py
    python preprocess_dialogues_friends.py --train_seasons 1 2 3 4 5 6 7 \\
                                           --test_seasons 8 9 10
"""

import argparse
import json
import random
import hashlib
from collections import defaultdict, Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "LongEvoRoleBench" / "raw_data"
PROCESSED_DIR = PROJECT_ROOT / "LongEvoRoleBench" / "processed"

RANDOM_SEED = 42

MAIN_CHARACTERS = [
    "Monica Geller",
    "Ross Geller",
    "Rachel Green",
    "Chandler Bing",
    "Joey Tribbiani",
    "Phoebe Buffay",
]
MAIN_SET = set(MAIN_CHARACTERS)

NOISE_SPEAKERS = frozenset({
    "#ALL#", "All", "ALL", "Everyone",
})

MIN_CONTEXT_UTTS = 3
MIN_OUTPUT_WORDS = 3
RANDOM_TEST_RATIO = 0.15


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_season(season_num: int) -> dict:
    path = RAW_DATA_DIR / "Friends" / "json" / f"friends_season_{season_num:02d}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

def _stable_id(scene_id: str, utt_idx: int) -> str:
    """Generate a deterministic question_id from scene and position."""
    raw = f"Friends_{scene_id}_{utt_idx}"
    short_hash = hashlib.md5(raw.encode()).hexdigest()[:12]
    return f"Friends_{short_hash}"


def build_samples_from_scene(
    scene: dict, season: int, episode: int,
) -> list[dict]:
    """Build (context, target) samples from one scene.

    For each main character utterance (from index >= MIN_CONTEXT_UTTS),
    create one sample:
      - input:  all prior utterances with real speaker names
      - output: the target utterance
      - role:   the target character
    """
    utts = scene["utterances"]
    samples = []

    for i, utt in enumerate(utts):
        speakers = utt.get("speakers", [])
        if not speakers or len(speakers) != 1:
            continue
        speaker = speakers[0]
        if speaker not in MAIN_SET or speaker in NOISE_SPEAKERS:
            continue

        text = utt.get("transcript", "").strip()
        if not text or len(text.split()) < MIN_OUTPUT_WORDS:
            continue

        if i < MIN_CONTEXT_UTTS:
            continue

        context_lines = []
        for cu in utts[:i]:
            cu_speakers = cu.get("speakers", [])
            cu_text = cu.get("transcript", "").strip()
            if not cu_text or not cu_speakers:
                continue
            cu_speaker = cu_speakers[0] if len(cu_speakers) == 1 else ", ".join(cu_speakers)
            if cu_speaker in NOISE_SPEAKERS:
                cu_speaker = "Everyone"
            context_lines.append(f"{cu_speaker}: {cu_text}")

        if not context_lines:
            continue

        scene_id = scene.get("scene_id", f"s{season:02d}_e{episode:02d}")
        qid = _stable_id(scene_id, i)

        samples.append({
            "user_id": f"Friends_{speaker.replace(' ', '_')}",
            "question_id": qid,
            "role": speaker,
            "profile_text": "",
            "input": "\n".join(context_lines),
            "output": text,
            "_season": season,
            "_episode": episode,
            "_scene_id": scene_id,
        })

    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Preprocess Friends dialogues into PHASE-Tree samples "
                    "with temporal train/test split",
    )
    ap.add_argument("--train_seasons", type=int, nargs="+",
                    default=list(range(1, 8)),
                    help="Seasons for train + random_test (default: 1-7)")
    ap.add_argument("--test_seasons", type=int, nargs="+",
                    default=[8, 9, 10],
                    help="Seasons for ood_test (default: 8-10)")
    ap.add_argument("--random_test_ratio", type=float,
                    default=RANDOM_TEST_RATIO,
                    help="Fraction of train-epoch samples held out as random_test")
    args = ap.parse_args()

    all_seasons = sorted(set(args.train_seasons + args.test_seasons))
    print(f"Train seasons: {args.train_seasons}")
    print(f"OOD test seasons: {args.test_seasons}")

    all_samples: list[dict] = []
    season_counts: dict[int, int] = {}

    for sn in all_seasons:
        sdata = load_season(sn)
        sn_samples = []
        for ep in sdata["episodes"]:
            ep_id = int(ep["episode_id"].split("_e")[-1])
            for sc in ep["scenes"]:
                sn_samples.extend(build_samples_from_scene(sc, sn, ep_id))
        season_counts[sn] = len(sn_samples)
        all_samples.extend(sn_samples)
        print(f"  Season {sn:2d}: {len(sn_samples):>5} samples")

    print(f"\nTotal samples: {len(all_samples)}")

    char_counts = Counter(s["role"] for s in all_samples)
    print("\nPer-character breakdown:")
    for c in MAIN_CHARACTERS:
        print(f"  {c}: {char_counts.get(c, 0)}")

    out_dir = PROCESSED_DIR / "Friends" / "intermediate"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _clean(s: dict) -> dict:
        return {k: v for k, v in s.items() if not k.startswith("_")}

    # Save all_dialogues (keep temporal metadata for phase-tree evolution)
    all_path = out_dir / "all_dialogues.json"
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"\nSaved all_dialogues.json: {len(all_samples)} samples -> {all_path}")

    # Temporal split
    train_epoch = [s for s in all_samples if s["_season"] in args.train_seasons]
    ood_epoch = [s for s in all_samples if s["_season"] in args.test_seasons]

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(train_epoch)
    n_rt = int(len(train_epoch) * args.random_test_ratio)
    random_test_samples = train_epoch[:n_rt]
    train_samples = train_epoch[n_rt:]

    splits = {
        "train": [_clean(s) for s in train_samples],
        "random_test": [_clean(s) for s in random_test_samples],
        "ood_test": [_clean(s) for s in ood_epoch],
    }

    print(f"\nSplit results:")
    for name, samples in splits.items():
        path = out_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        n_chars = len({s["role"] for s in samples})
        char_dist = Counter(s["role"] for s in samples)
        print(f"  {name:>15}: {len(samples):>5} samples, {n_chars} chars")
        for c in MAIN_CHARACTERS:
            print(f"    {c}: {char_dist.get(c, 0)}")

    # Verify all 6 characters appear in each split
    for name, samples in splits.items():
        chars_in = {s["role"] for s in samples}
        missing = MAIN_SET - chars_in
        if missing:
            print(f"  WARNING: {name} is missing characters: {missing}")

    print("\nDone.")


if __name__ == "__main__":
    main()
