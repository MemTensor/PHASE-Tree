#!/usr/bin/env python3
"""Extract GPT-4.1 judge scores for the human-eval sample set.

Reads ``judge_scores.jsonl`` from each sample's result directory and writes
``results/human_eval/gpt_judge_scores.json`` for comparison with
``human_scores_{A,B,C}.json``.

Usage:
    python evaluation/extract_gpt_judge_scores.py
    python evaluation/extract_gpt_judge_scores.py --he-dir results/human_eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_judge_lookup(path: Path) -> dict[str, dict]:
    """Return ``{question_id: row}`` from a ``judge_scores.jsonl`` file."""
    lookup: dict[str, dict] = {}
    if not path.is_file():
        return lookup
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("question_id")
            if qid:
                lookup[qid] = row
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--he-dir",
        type=Path,
        default=ROOT / "results" / "human_eval",
        help="Directory with samples_to_score.json",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Root of evaluation result trees",
    )
    args = parser.parse_args()

    samples_path = args.he_dir / "samples_to_score.json"
    out_path = args.he_dir / "gpt_judge_scores.json"
    samples = json.loads(samples_path.read_text(encoding="utf-8"))

    cache: dict[str, dict[str, dict]] = {}
    records: list[dict] = []
    missing: list[str] = []

    for sample in samples:
        sid = sample["sample_id"]
        qid = sample["question_id"]
        judge_path = (
            args.results_dir
            / sample["dataset"]
            / sample["track"]
            / "main"
            / sample["method"]
            / sample["split"]
            / "judge_scores.jsonl"
        )
        key = str(judge_path)
        if key not in cache:
            cache[key] = load_judge_lookup(judge_path)

        row = cache[key].get(qid)
        if row is None:
            missing.append(sid)
            continue

        records.append(
            {
                "sample_id": sid,
                "question_id": qid,
                "character_score": row["character_score"],
                "semantic_score": row["semantic_score"],
                "reasoning": row.get("reasoning", ""),
            }
        )

    payload = {
        "judge_model": "gpt-4.1",
        "judge_scores_file": "judge_scores.jsonl",
        "backbone": "Qwen2.5-7B-Instruct",
        "n_samples_expected": len(samples),
        "n_samples_found": len(records),
        "missing_sample_ids": missing,
        "scores": records,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}: {len(records)}/{len(samples)} scores")
    if missing:
        print(f"missing ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")


if __name__ == "__main__":
    main()
