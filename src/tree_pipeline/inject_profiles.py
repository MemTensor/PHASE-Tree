"""Inject profile text into dialogue JSON files to produce final datasets.

Read intermediate dialogue splits (``profile_text`` still empty) and a
profile-text source file, then fill in ``profile_text`` for every sample.
The completed files are written to method-specific sub-directories under
the dataset's processed directory.

Six experimental methods are supported:

* ``context_only``   — no profile text (lower bound).
  Output dir: ``m1_context_only/``
* ``raw_profile``    — raw natural-language profile per *character*.
  Source: ``intermediate/raw_profile_texts.json``
  Output dir: ``m2_raw_profile/``
* ``naive_rewrite``  — LLM-rewritten profile per *sample*.
  Source: ``intermediate/{split}_naive_rewrite_profile_texts.json``
  Output dir: ``m3_naive_rewrite/``
* ``static_tree``    — static attribute tree per *character*.
  Source: ``intermediate/static_tree_profile_texts.json``
  Output dir: ``m4_static_tree/``
* ``dynamic_tree``   — dynamic attribute tree (long-term only) per *sample*.
  Source: ``intermediate/{split}_long_term_dynamic_profile_texts.json``
  Output dir: ``m5_dynamic_tree/``
* ``phase_tree``     — PHASE attribute tree (with session/moment) per *sample*.
  Source: ``intermediate/{split}_phase_profile_texts.json``
  Output dir: ``m6_phase_tree/``

Output naming: ``phase_tree_data/processed/<dataset>/<method_dir>/<split>.json``

Usage::

    python inject_profiles.py --dataset RAIDEN --method context_only
    python inject_profiles.py --dataset RAIDEN --method raw_profile
    python inject_profiles.py --dataset RAIDEN --method naive_rewrite
    python inject_profiles.py --dataset RAIDEN --method static_tree
    python inject_profiles.py --dataset RAIDEN --method phase_tree
    python inject_profiles.py --dataset RAIDEN --method phase_tree --split train
"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "phase_tree_data" / "processed"

METHODS = (
    "context_only", "raw_profile", "naive_rewrite",
    "static_tree", "dynamic_tree", "phase_tree",
    "long_term_dynamic_tree", "long_term_phase_tree",
)

METHOD_DIR_MAP = {
    "context_only": "m1_context_only",
    "raw_profile": "m2_raw_profile",
    "naive_rewrite": "m3_naive_rewrite",
    "static_tree": "m4_static_tree",
    "dynamic_tree": "m5_dynamic_tree",
    "phase_tree": "m6_phase_tree",
    # Legacy aliases — same output dirs, kept for backward compatibility
    "long_term_dynamic_tree": "m5_dynamic_tree",
    "long_term_phase_tree": "m6_phase_tree",
}

SPLITS = ["all_dialogues", "train", "random_test", "ood_test"]


def load_profile_texts(
    intermediate_dir: Path, method: str, split_name: str,
) -> dict[str, str] | None:
    """Load the profile-text mapping for the given method and split.

    Returns *None* for ``context_only`` (no profile needed).
    """
    if method == "context_only":
        return None
    if method == "raw_profile":
        path = intermediate_dir / "raw_profile_texts.json"
    elif method == "static_tree":
        path = intermediate_dir / "static_tree_profile_texts.json"
    elif method == "naive_rewrite":
        path = intermediate_dir / f"{split_name}_naive_rewrite_profile_texts.json"
    elif method == "phase_tree":
        path = intermediate_dir / f"{split_name}_phase_profile_texts.json"
    elif method == "long_term_phase_tree":
        path = intermediate_dir / f"{split_name}_long_term_phase_profile_texts.json"
    elif method in ("dynamic_tree", "long_term_dynamic_tree"):
        path = intermediate_dir / f"{split_name}_long_term_dynamic_profile_texts.json"
    else:
        raise ValueError(f"Unknown method: {method}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _match_key(method: str) -> str:
    """Determine which sample field to use for profile lookup."""
    if method in ("raw_profile", "static_tree"):
        return "role"
    return "question_id"


def inject(
    samples: list[dict],
    profiles: dict[str, str] | None,
    match_key: str,
) -> tuple[list[dict], int]:
    """Fill ``profile_text`` for each sample.  Returns (new_list, n_missing)."""
    out = []
    missing = 0
    for s in samples:
        if profiles is None:
            text = ""
        else:
            key = s[match_key]
            text = profiles.get(key, "")
            if not text:
                missing += 1
        out.append({**s, "profile_text": text})
    return out, missing


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inject profile texts into dialogue splits",
    )
    ap.add_argument("--dataset", type=str, required=True,
                    choices=["CharacterEval", "RAIDEN", "ChatHaruhi", "SimsConv", "Friends", "TheOffice", "StarTrek_TNG", "HPD"],
                    help="Dataset to process")
    ap.add_argument("--method", type=str, required=True,
                    choices=list(METHODS),
                    help="Experimental method (determines profile source and output dir)")
    ap.add_argument("--split", type=str, default="all",
                    choices=SPLITS + ["all"],
                    help="Which split to process (default: all)")
    args = ap.parse_args()

    intermediate_dir = PROCESSED_DIR / args.dataset / "intermediate"
    out_dir = PROCESSED_DIR / args.dataset / METHOD_DIR_MAP[args.method]
    out_dir.mkdir(parents=True, exist_ok=True)

    mk = _match_key(args.method)
    splits = SPLITS if args.split == "all" else [args.split]

    for split_name in splits:
        src_path = intermediate_dir / f"{split_name}.json"
        if not src_path.exists():
            print(f"  {split_name}: SKIPPED (file not found)")
            continue

        profiles = load_profile_texts(intermediate_dir, args.method, split_name)
        n_profiles = len(profiles) if profiles is not None else 0
        print(f"[{split_name}] Loaded {n_profiles} profile texts "
              f"(method={args.method}, match_key={mk})")

        with open(src_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        injected, n_missing = inject(samples, profiles, mk)

        out_path = out_dir / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(injected, f, ensure_ascii=False, indent=2)

        print(f"  {split_name}: {len(injected)} samples -> {out_path}"
              + (f"  ({n_missing} missing profiles)" if n_missing else ""))

    print("\nDone.")


if __name__ == "__main__":
    main()
