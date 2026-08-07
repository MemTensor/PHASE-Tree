#!/usr/bin/env python3
"""Balanced sampling for human rescoring of Qwen2.5-7B backbone results.

For each (dataset, track, method) combination, sample items so every method is
represented. Uses a fixed seed and stable ``sample_id`` values so later raters
score the same set.

Usage:
    python evaluation/sample_for_human_eval.py
    python evaluation/sample_for_human_eval.py --target 200 --out results/human_eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASETS = [
    "CharacterEval",
    "ChatHaruhi",
    "Friends",
    "HPD",
    "RAIDEN",
    "SimsConv",
    "StarTrek_TNG",
    "TheOffice",
]

# Datasets that have extra methods
EXTENDED_DATASETS = {"Friends", "HPD", "StarTrek_TNG", "TheOffice"}

# Track -> methods available. Some methods only appear on extended datasets.
TRACK_METHODS = {
    "comparison": {
        "core": ["cfg", "mt_lora", "pag", "rag", "steering"],
        "extended_only": ["oppu"],
    },
    "prompt": {
        "core": ["m1_context_only", "m2_raw_profile", "m3_naive_rewrite", "m4_static_tree", "m6_phase_tree"],
        "extended_only": ["m5_dynamic_tree"],
    },
    "phase_tree": {
        "core": ["m2_raw_profile", "m3_naive_rewrite", "m4_static_tree", "m6_phase_tree"],
        "extended_only": ["m5_dynamic_tree"],
    },
    "hypernet_p2p": {
        "core": ["m2_raw_profile", "m3_naive_rewrite", "m4_static_tree", "m6_phase_tree"],
        "extended_only": ["m5_dynamic_tree"],
    },
}

SPLITS = ["random_test", "ood_test"]
DEFAULT_SEED = 20260710


def make_sample_id(dataset, track, method, split, question_id) -> str:
    """Deterministic short id so we can compare with future model reruns."""
    raw = f"{dataset}|{track}|{method}|{split}|{question_id}"
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    return f"HE-{h}"


def load_profile_lookup(data_dir: Path, dataset: str) -> dict:
    """Load ``{question_id: profile fields}`` from ``m6_phase_tree`` dialogues."""
    src = data_dir / dataset / "m6_phase_tree" / "all_dialogues.json"
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        item["question_id"]: {
            "profile_text": item["profile_text"],
            "input": item["input"],
            "output": item["output"],
            "role": item.get("role", ""),
        }
        for item in data
    }


def load_predictions(pred_path: Path) -> dict:
    preds = {}
    if not pred_path.exists():
        return preds
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = obj.get("question_id")
            if qid:
                preds[qid] = obj
    return preds


def enumerate_combos(results_dir: Path):
    """Return existing (dataset, track, method, split, pred_path) tuples."""
    combos = []
    for dataset in DATASETS:
        for track, cfg in TRACK_METHODS.items():
            methods = list(cfg["core"])
            if dataset in EXTENDED_DATASETS:
                methods.extend(cfg["extended_only"])
            for method in methods:
                for split in SPLITS:
                    pred_path = (
                        results_dir / dataset / track / "main" / method / split / "predictions.jsonl"
                    )
                    if pred_path.exists():
                        combos.append((dataset, track, method, split, pred_path))
    return combos


def plan_quotas(target_total: int = 200):
    """Allocate quotas per (track, method) globally, then split across datasets.

    Strategy:
        - Every (track, method) present in the results tree gets a quota.
        - Extended-only methods (fewer datasets) receive a smaller quota
          so the per-dataset load stays roughly even.
        - Any remaining budget is round-robin distributed to full-coverage
          methods.
    """
    method_keys = []
    for track, cfg in TRACK_METHODS.items():
        for method in cfg["core"]:
            method_keys.append((track, method, "full"))
        for method in cfg["extended_only"]:
            method_keys.append((track, method, "extended"))

    quotas = {}
    for track, method, kind in method_keys:
        quotas[(track, method)] = 8 if kind == "full" else 4

    total = sum(quotas.values())
    remaining = target_total - total
    keys_full = [k for k in quotas if k in [(t, m) for t, m, kind in method_keys if kind == "full"]]

    i = 0
    while remaining > 0:
        quotas[keys_full[i % len(keys_full)]] += 1
        remaining -= 1
        i += 1

    return quotas


def sample(results_dir: Path, data_dir: Path, out_dir: Path, target_total: int, seed: int) -> None:
    random.seed(seed)
    combos = enumerate_combos(results_dir)
    quotas = plan_quotas(target_total)

    print(f"[info] enumerated {len(combos)} (dataset,track,method,split) folders")
    print(f"[info] total quota target: {sum(quotas.values())}")

    profile_cache = {}
    prediction_cache = {}

    def get_profiles(ds):
        if ds not in profile_cache:
            profile_cache[ds] = load_profile_lookup(data_dir, ds)
        return profile_cache[ds]

    def get_preds(pred_path):
        key = str(pred_path)
        if key not in prediction_cache:
            prediction_cache[key] = load_predictions(pred_path)
        return prediction_cache[key]

    grouped = defaultdict(list)
    for c in combos:
        dataset, track, method, split, pred_path = c
        grouped[(track, method)].append(c)

    selected = []
    seen_qid_per_method = defaultdict(set)

    for (track, method), quota in quotas.items():
        candidates = grouped.get((track, method), [])
        if not candidates:
            print(f"[warn] no candidates for {track}/{method}")
            continue

        random.shuffle(candidates)

        allocation_per_combo = defaultdict(int)
        i = 0
        for _ in range(quota):
            allocation_per_combo[candidates[i % len(candidates)]] += 1
            i += 1

        for combo, n in allocation_per_combo.items():
            dataset, tr, mth, split, pred_path = combo
            preds = get_preds(pred_path)
            if not preds:
                continue
            profiles = get_profiles(dataset)
            valid_qids = [
                qid for qid in preds.keys()
                if qid in profiles and qid not in seen_qid_per_method[(track, method)]
            ]
            if len(valid_qids) < n:
                extra = [qid for qid in preds.keys() if qid in profiles]
                valid_qids = list(set(extra))
            random.shuffle(valid_qids)
            for qid in valid_qids[:n]:
                seen_qid_per_method[(track, method)].add(qid)
                p = profiles[qid]
                pred = preds[qid].get("prediction", "")
                sample_id = make_sample_id(dataset, track, method, split, qid)
                selected.append({
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "track": track,
                    "method": method,
                    "split": split,
                    "question_id": qid,
                    "role": p["role"],
                    "profile_text": p["profile_text"],
                    "input": p["input"],
                    "ground_truth": p["output"],
                    "prediction": pred,
                })

    while len(selected) < target_total:
        (track, method), quota = random.choice(list(quotas.items()))
        candidates = grouped.get((track, method), [])
        if not candidates:
            continue
        combo = random.choice(candidates)
        dataset, tr, mth, split, pred_path = combo
        preds = get_preds(pred_path)
        profiles = get_profiles(dataset)
        available = [qid for qid in preds if qid in profiles and qid not in seen_qid_per_method[(track, method)]]
        if not available:
            continue
        qid = random.choice(available)
        seen_qid_per_method[(track, method)].add(qid)
        p = profiles[qid]
        sample_id = make_sample_id(dataset, track, method, split, qid)
        selected.append({
            "sample_id": sample_id,
            "dataset": dataset,
            "track": track,
            "method": method,
            "split": split,
            "question_id": qid,
            "role": p["role"],
            "profile_text": p["profile_text"],
            "input": p["input"],
            "ground_truth": p["output"],
            "prediction": preds[qid].get("prediction", ""),
        })

    selected = selected[:target_total]

    method_counts = defaultdict(int)
    dataset_counts = defaultdict(int)
    for s in selected:
        method_counts[(s["track"], s["method"])] += 1
        dataset_counts[s["dataset"]] += 1

    print("\n[info] per-method counts:")
    for k, v in sorted(method_counts.items()):
        print(f"  {k[0]}/{k[1]:<20s} -> {v}")
    print("\n[info] per-dataset counts:")
    for k, v in sorted(dataset_counts.items()):
        print(f"  {k:<18s} -> {v}")
    print(f"\n[info] total selected: {len(selected)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "samples_to_score.json"
    out_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "LongEvoRoleBench" / "processed")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "human_eval")
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    sample(args.results_dir, args.data_dir, args.out, args.target, args.seed)


if __name__ == "__main__":
    main()
