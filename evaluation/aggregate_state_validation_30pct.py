#!/usr/bin/env python3
"""Aggregate per-corpus 30% state-validation ratings into paper summaries.

Reads ``ratings30_*.json`` and ``balanced_subset_30pct.json``, then writes:
  - ``semantic30_3raters.json``
  - ``validation_30pct_3raters_SEMANTIC_summary.md``

Usage:
    python evaluation/aggregate_state_validation_30pct.py
    python evaluation/aggregate_state_validation_30pct.py --out results/state_validation
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RATING_FILES = [
    "ratings30_Friends.json",
    "ratings30_The_Office.json",
    "ratings30_Harry_Potter.json",
    "ratings30_Star_Trek.json",
]
SCORE = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
CORPORA = ["Friends", "The Office", "Harry Potter", "Star Trek"]
VALID_LABELS = frozenset({"supported", "partial", "unsupported"})


def normalize_label(value: object) -> str:
    """Parse a rater label; ``unsupported`` must be matched before ``supported``."""
    text = str(value).strip().lower()
    if text in VALID_LABELS:
        return text
    for label in ("unsupported", "partial", "supported"):
        if label in text:
            return label
    raise ValueError(f"unrecognized label: {value!r}")


def load_ratings(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("ratings") or data.get("items") or data
        if isinstance(data, dict):
            data = list(data.values())
    if not isinstance(data, list):
        raise ValueError(f"expected list in {path}")
    return data


def cohens_kappa(y1: list[int], y2: list[int]) -> float:
    n = len(y1)
    a = sum(1 for i in range(n) if y1[i] == 1 and y2[i] == 1)
    b = sum(1 for i in range(n) if y1[i] == 1 and y2[i] == 0)
    c = sum(1 for i in range(n) if y1[i] == 0 and y2[i] == 1)
    d = sum(1 for i in range(n) if y1[i] == 0 and y2[i] == 0)
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def fleiss_kappa(ratings: list[list[str]]) -> float:
    cats = ["supported", "partial", "unsupported"]
    idx = {c: i for i, c in enumerate(cats)}
    n_items = len(ratings)
    n_raters = 3
    k = len(cats)
    mat = [[0] * k for _ in range(n_items)]
    for i, row in enumerate(ratings):
        for lab in row:
            mat[i][idx[lab]] += 1
    p_i = [sum(x * (x - 1) for x in mat[i]) / (n_raters * (n_raters - 1)) for i in range(n_items)]
    p_bar = sum(p_i) / n_items
    p_j = [sum(mat[i][j] for i in range(n_items)) / (n_items * n_raters) for j in range(k)]
    pe = sum(p * p for p in p_j)
    return (p_bar - pe) / (1 - pe) if pe != 1 else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "state_validation",
        help="Directory with ratings30_*.json and balanced_subset_30pct.json",
    )
    args = parser.parse_args()
    out_dir: Path = args.out

    rows: list[dict] = []
    for name in RATING_FILES:
        path = out_dir / name
        data = load_ratings(path)
        print(f"{name}: n={len(data)}")
        for item in data:
            labels = {f"R{i}": normalize_label(item[f"R{i}"]) for i in (1, 2, 3)}
            scores = {k: SCORE[v] for k, v in labels.items()}
            mean_s = sum(scores.values()) / 3
            var_s = sum((scores[k] - mean_s) ** 2 for k in scores) / 3
            maj = Counter(labels.values()).most_common()
            consensus = (
                "partial"
                if len(maj) >= 2 and maj[0][1] == maj[1][1]
                else maj[0][0]
            )
            rows.append(
                {
                    "update_id": item["update_id"],
                    "labels": labels,
                    "scores": scores,
                    "mean_score": mean_s,
                    "var_score": var_s,
                    "consensus": consensus,
                    "note": item.get("note"),
                }
            )

    subset = json.loads((out_dir / "balanced_subset_30pct.json").read_text(encoding="utf-8"))
    uid_to_corpus = {u["update_id"]: u["corpus"] for u in subset["field_updates"]}
    for row in rows:
        row["corpus"] = uid_to_corpus.get(row["update_id"], "?")

    n = len(rows)
    n_snap = subset["n_snapshots"]

    rater_stats: dict[str, dict] = {}
    for rid in ("R1", "R2", "R3"):
        scores = [r["scores"][rid] for r in rows]
        labels = [r["labels"][rid] for r in rows]
        mu = sum(scores) / n
        var = sum((x - mu) ** 2 for x in scores) / n
        rater_stats[rid] = {
            "mean_score": round(mu, 4),
            "var_score": round(var, 4),
            "std_score": round(math.sqrt(var), 4),
            "pct_supported": round(100 * sum(1 for x in labels if x == "supported") / n, 1),
            "pct_partial": round(100 * sum(1 for x in labels if x == "partial") / n, 1),
            "pct_unsupported": round(
                100 * sum(1 for x in labels if x == "unsupported") / n, 1
            ),
            "label_counts": dict(Counter(labels)),
        }

    means = [rater_stats[r]["mean_score"] for r in ("R1", "R2", "R3")]
    mean_of_means = sum(means) / 3
    var_of_means = sum((m - mean_of_means) ** 2 for m in means) / 3
    pcts = [rater_stats[r]["pct_supported"] for r in ("R1", "R2", "R3")]
    mean_pct = sum(pcts) / 3
    var_pct = sum((p - mean_pct) ** 2 for p in pcts) / 3
    avg_item_mean = sum(r["mean_score"] for r in rows) / n
    avg_item_var = sum(r["var_score"] for r in rows) / n

    kappas: dict[str, float] = {}
    for a, b in (("R1", "R2"), ("R1", "R3"), ("R2", "R3")):
        y1 = [1 if r["labels"][a] == "supported" else 0 for r in rows]
        y2 = [1 if r["labels"][b] == "supported" else 0 for r in rows]
        kappas[f"{a}-{b}"] = round(cohens_kappa(y1, y2), 3)

    fk = fleiss_kappa([[r["labels"]["R1"], r["labels"]["R2"], r["labels"]["R3"]] for r in rows])
    cons = Counter(r["consensus"] for r in rows)
    pct_s = 100 * cons.get("supported", 0) / n
    pct_p = 100 * cons.get("partial", 0) / n
    pct_u = 100 * cons.get("unsupported", 0) / n
    pct_loose = pct_s + pct_p
    mean_k = sum(kappas.values()) / 3

    per_corpus: dict[str, dict] = {}
    for corpus in CORPORA:
        items = [r for r in rows if r["corpus"] == corpus]
        if not items:
            continue
        per_corpus[corpus] = {
            "n": len(items),
            "pct_supported_consensus": round(
                100 * sum(1 for r in items if r["consensus"] == "supported") / len(items), 1
            ),
            "mean_of_item_means": round(sum(r["mean_score"] for r in items) / len(items), 3),
        }

    clause = (
        f"({n_snap} snapshots / {n} updates on Friends/Office/HP/Star Trek; "
        f"{pct_s:.0f}% supported, mean={mean_of_means:.2f}, κ={mean_k:.2f})"
    )

    summary = {
        "task": "Judge updated template given previous template + extracted evidence",
        "stance": "Use supported, partial, and unsupported as assigned by each rater",
        "sampling_quota": subset["quota"],
        "n_snapshots": n_snap,
        "n_field_updates": n,
        "corpora": CORPORA,
        "consensus_counts": dict(cons),
        "consensus_pct_supported": round(pct_s, 1),
        "consensus_pct_partial": round(pct_p, 1),
        "consensus_pct_unsupported": round(pct_u, 1),
        "consensus_pct_supported_or_partial": round(pct_loose, 1),
        "rater_stats": rater_stats,
        "three_rater_mean_of_mean_scores": round(mean_of_means, 4),
        "three_rater_var_of_mean_scores": round(var_of_means, 6),
        "three_rater_mean_of_pct_supported": round(mean_pct, 2),
        "three_rater_var_of_pct_supported": round(var_pct, 4),
        "avg_item_mean_score": round(avg_item_mean, 4),
        "avg_item_var_across_raters": round(avg_item_var, 4),
        "cohens_kappa_strict": kappas,
        "mean_pairwise_kappa_strict": round(mean_k, 3),
        "fleiss_kappa": round(fk, 3),
        "per_corpus": per_corpus,
        "paper_clause": clause,
        "paper_clause_len": len(clause),
    }

    out_json = out_dir / "semantic30_3raters.json"
    out_json.write_text(
        json.dumps({"summary": summary, "ratings": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md = [
        "# 30% × 3 semantic raters — state template validation\n",
        "## Task\n",
        "Given **extracted evidence** and the **previous template** (`old_value`), "
        "judge whether the **updated template** (`new_value`) is reasonably supported "
        "and accurate.\n",
        "## Sampling (~30% evidence-qualified snapshots per corpus)\n",
        "| Corpus | Pool | Sampled (~30%) |",
        "|---|---:|---:|",
    ]
    for corpus, quota in subset["quota"].items():
        md.append(f"| {corpus} | {quota['pool']} | {quota['sampled']} |")
    md.append(f"\n**Snapshots:** {n_snap}  ·  **Field updates:** {n}\n")
    md.append("## Three raters\n")
    md.append("| Rater | mean score | variance | %supported | %partial | %unsupported |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for rid in ("R1", "R2", "R3"):
        s = rater_stats[rid]
        md.append(
            f"| {rid} | {s['mean_score']:.3f} | {s['var_score']:.4f} | "
            f"{s['pct_supported']}% | {s['pct_partial']}% | {s['pct_unsupported']}% |"
        )
    md.extend(
        [
            "\n## Aggregate mean / variance\n",
            f"- Mean of rater mean-scores: **{mean_of_means:.4f}**",
            f"- Variance of rater mean-scores: **{var_of_means:.6f}**",
            f"- Mean of rater %supported: **{mean_pct:.2f}%**",
            f"- Variance of rater %supported: **{var_pct:.4f}**",
            f"- Avg per-item mean score: **{avg_item_mean:.4f}**",
            f"- Avg per-item variance across raters: **{avg_item_var:.4f}**",
            f"- Mean pairwise κ (supported vs not): **{mean_k:.3f}**",
            f"- Fleiss κ: **{fk:.3f}**\n",
            "## Consensus\n",
            (
                f"- supported **{pct_s:.1f}%** · partial **{pct_p:.1f}%** · "
                f"unsupported **{pct_u:.1f}%** · supported∪partial **{pct_loose:.1f}%**\n"
            ),
            "## Paper clause\n",
            f"> {clause}\n\n(len={len(clause)})\n",
        ]
    )
    (out_dir / "validation_30pct_3raters_SEMANTIC_summary.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"CLAUSE: {clause} (len={len(clause)})")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
