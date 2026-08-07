"""Preprocess HPD (Harry Potter Dialogue) into PHASE-Tree samples with temporal split.

**Step 1 — Dialogue conversion.**  For each session across all 7 books,
construct training/evaluation samples where each main character's utterance
is a potential ``output``.  The ``input`` is all preceding dialogue in that
session with speaker names preserved.

HPD dialogue lines are already formatted as "Speaker: text". This script
parses them, normalizes speaker names (strip whitespace), and builds
(context, target) samples. Compound speakers (e.g. "Harry, Ron") are kept
as-is in context but never selected as output targets.

**Step 2 — Temporal 3-way split.**

* ``train``        — Book 1–5 (random 80% of qualifying samples)
* ``random_test``  — Book 1–5 (random 20% holdout, same epoch)
* ``ood_test``     — Book 6–7 (later time period — for testing dynamic
                     attribute tree evolution; Dumbledore's death in Book 6
                     and full war in Book 7 represent massive character shifts)

Input
    ``LongEvoRoleBench/raw_data/HPD/EN_all.json``

Output
    ``LongEvoRoleBench/processed/HPD/intermediate/all_dialogues.json``
    ``LongEvoRoleBench/processed/HPD/intermediate/{train,random_test,ood_test}.json``

Usage::

    python preprocess_dialogues_hpd.py
    python preprocess_dialogues_hpd.py --train_books 1 2 3 4 5 \\
                                       --test_books 6 7
"""

import argparse
import json
import random
import hashlib
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "LongEvoRoleBench" / "raw_data"
PROCESSED_DIR = PROJECT_ROOT / "LongEvoRoleBench" / "processed"

RANDOM_SEED = 42

MAIN_CHARACTERS = [
    "Harry",
    "Ron",
    "Hermione",
    "Dumbledore",
    "Hagrid",
    "Snape",
]
MAIN_SET = set(MAIN_CHARACTERS)

MIN_CONTEXT_LINES = 3
MIN_OUTPUT_WORDS = 3
RANDOM_TEST_RATIO = 0.20


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hpd_data() -> dict:
    path = RAW_DATA_DIR / "HPD" / "EN_all.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_speaker(name: str) -> str:
    return name.strip()


def parse_dialogue_line(line: str) -> tuple[str, str]:
    """Parse 'Speaker: text' format. Returns (speaker, text).
    For compound speakers like 'Harry, Ron: text', speaker will contain comma.
    """
    line = line.strip()
    colon_idx = line.find(":")
    if colon_idx == -1:
        return ("", line)
    speaker = line[:colon_idx].strip()
    text = line[colon_idx + 1:].strip()
    return (speaker, text)


# Patterns that indicate a pure narrative stub (not real dialogue)
_NARRATIVE_STUBS = {
    "he said.", "she said.", "he said", "she said",
    "said he.", "said she.",
}

# Regex to detect third-person narrative fragments appended to dialogue.
# Matches patterns like "Harry thought this..." or "Ron noticed that..."
# that clearly describe a character in third person mid-dialogue.
_NARRATIVE_TAIL_RE = re.compile(
    r"\s+"
    r"(?:Harry|Ron|Hermione|Dumbledore|Hagrid|Snape|He|She)"
    r"\s+(?:thought|noticed|stared|felt|saw|looked|remembered|realized|wondered)"
    r"\s+.{10,}",
)


def clean_dialogue_lines(
    dialogue: list[str],
) -> list[tuple[str, str]]:
    """Parse and clean raw HPD dialogue lines.

    1. Normalize speaker names (strip whitespace).
    2. Drop lines that are pure narrative stubs ("he said.").
    3. Strip third-person narrative tails appended to real dialogue.
    4. Drop lines with empty text after cleaning.
    """
    cleaned: list[tuple[str, str]] = []
    for line in dialogue:
        speaker, text = parse_dialogue_line(line)
        speaker = normalize_speaker(speaker)

        if not speaker or not text:
            continue

        if text.lower().strip('"""\'') in _NARRATIVE_STUBS:
            continue

        m = _NARRATIVE_TAIL_RE.search(text)
        if m:
            text = text[:m.start()].rstrip(" ,;")

        text = text.strip()
        if not text:
            continue

        cleaned.append((speaker, text))
    return cleaned


def is_single_main_char(speaker: str) -> str | None:
    """Check if speaker is exactly one main character (after normalization).
    Returns the character name or None."""
    normalized = normalize_speaker(speaker)
    if normalized in MAIN_SET:
        return normalized
    return None


def get_book_num(position: str) -> int | None:
    """Extract book number from position string like 'Book3-chapter12'."""
    if not position or not position.startswith("Book"):
        return None
    try:
        book_part = position.split("-")[0]
        return int(book_part.replace("Book", ""))
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

def _stable_id(session_id: str, line_idx: int) -> str:
    raw = f"HPD_{session_id}_{line_idx}"
    short_hash = hashlib.md5(raw.encode()).hexdigest()[:12]
    return f"HPD_{short_hash}"


def build_samples_from_session(
    session_id: str, sess: dict,
) -> list[dict]:
    """Build (context, target) samples from a single HPD session."""
    dialogue = sess.get("dialogue", [])
    if not dialogue or len(dialogue) < MIN_CONTEXT_LINES + 1:
        return []

    position = sess.get("position", "")
    book_num = get_book_num(position)
    if book_num is None:
        return []

    parsed = clean_dialogue_lines(dialogue)

    samples = []
    for i, (speaker, text) in enumerate(parsed):
        char = is_single_main_char(speaker)
        if char is None:
            continue

        if not text or len(text.split()) < MIN_OUTPUT_WORDS:
            continue

        if i < MIN_CONTEXT_LINES:
            continue

        context_lines = []
        for prev_speaker, prev_text in parsed[:i]:
            if prev_text:
                context_lines.append(f"{prev_speaker}: {prev_text}")

        if not context_lines:
            continue

        qid = _stable_id(session_id, i)

        samples.append({
            "user_id": f"HPD_{char}",
            "question_id": qid,
            "role": char,
            "profile_text": "",
            "input": "\n".join(context_lines),
            "output": text,
            "_book": book_num,
            "_position": position,
            "_session_id": session_id,
        })

    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Preprocess HPD dialogues into PHASE-Tree samples "
                    "with temporal train/test split",
    )
    ap.add_argument("--train_books", type=int, nargs="+",
                    default=list(range(1, 6)),
                    help="Books for train + random_test (default: 1-5)")
    ap.add_argument("--test_books", type=int, nargs="+",
                    default=[6, 7],
                    help="Books for ood_test (default: 6-7)")
    ap.add_argument("--random_test_ratio", type=float,
                    default=RANDOM_TEST_RATIO,
                    help="Fraction of train-epoch samples held out as random_test")
    args = ap.parse_args()

    print("Loading HPD data...")
    data = load_hpd_data()
    print(f"Total sessions in dataset: {len(data)}")

    train_book_set = set(args.train_books)
    test_book_set = set(args.test_books)
    all_books = sorted(train_book_set | test_book_set)
    print(f"Train books: {sorted(train_book_set)}")
    print(f"OOD test books: {sorted(test_book_set)}")

    all_samples: list[dict] = []
    book_counts: dict[int, int] = {}

    for sid, sess in data.items():
        position = sess.get("position", "")
        book_num = get_book_num(position)
        if book_num is None or book_num not in (train_book_set | test_book_set):
            continue

        session_samples = build_samples_from_session(sid, sess)
        all_samples.extend(session_samples)
        book_counts[book_num] = book_counts.get(book_num, 0) + len(session_samples)

    for b in all_books:
        print(f"  Book {b}: {book_counts.get(b, 0):>5} samples")

    print(f"\nTotal samples: {len(all_samples)}")

    char_counts = Counter(s["role"] for s in all_samples)
    print("\nPer-character breakdown:")
    for c in MAIN_CHARACTERS:
        print(f"  {c}: {char_counts.get(c, 0)}")

    out_dir = PROCESSED_DIR / "HPD" / "intermediate"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _clean(s: dict) -> dict:
        return {k: v for k, v in s.items() if not k.startswith("_")}

    # Save all_dialogues (keep temporal metadata for phase-tree evolution)
    all_path = out_dir / "all_dialogues.json"
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"\nSaved all_dialogues.json: {len(all_samples)} samples -> {all_path}")

    # Temporal split
    train_epoch = [s for s in all_samples if s["_book"] in train_book_set]
    ood_epoch = [s for s in all_samples if s["_book"] in test_book_set]

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

    for name, samples in splits.items():
        chars_in = {s["role"] for s in samples}
        missing = MAIN_SET - chars_in
        if missing:
            print(f"  WARNING: {name} is missing characters: {missing}")

    print("\nDone.")


if __name__ == "__main__":
    main()
