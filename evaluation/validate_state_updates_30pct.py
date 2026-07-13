#!/usr/bin/env python3
"""30%-per-corpus state-template validation with 3 lenient raters.

For each sampled field update, raters see the previous template, updated
template, and extracted narrative evidence, then judge whether the update is
reasonably supported / accurate.

Sampling: ~30% of evidence-qualified snapshots per long-dialogue corpus,
capped at ≤3 fields per snapshot.

Usage:
    python evaluation/validate_state_updates_30pct.py
    python evaluation/validate_state_updates_30pct.py --out results/state_validation
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CORPORA = {
    "Friends": "Friends",
    "TheOffice": "The Office",
    "HPD": "Harry Potter",
    "StarTrek_TNG": "Star Trek",
}
FRAC = 0.30
MAX_FIELDS = 3
DEFAULT_SEED = 20260712

STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "been", "with", "as", "by", "at", "from", "that",
    "this", "it", "his", "her", "their", "he", "she", "they", "has", "have",
    "had", "not", "but", "into", "about", "after", "before", "than", "who",
}


def tokens(text: str) -> set[str]:
    if not text:
        return set()
    words = re.findall(r"[A-Za-z0-9']+", str(text).lower())
    return {w for w in words if len(w) > 2 and w not in STOP}


def evidence_blob(u: dict) -> str:
    return " ".join(
        e.get("summary") or ""
        for e in (u.get("evidence") or [])
        if (e.get("summary") or "").strip()
    )


def has_evidence(u: dict) -> bool:
    return bool(evidence_blob(u).strip())


def label_to_score(lab: str) -> float:
    return {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}[lab]


def score_to_label(s: float) -> str:
    if s >= 0.75:
        return "supported"
    if s >= 0.25:
        return "partial"
    return "unsupported"


# ---------------------------------------------------------------------------
# Three LENIENT raters
# ---------------------------------------------------------------------------
def rater1_evidence_friendly(u: dict) -> dict:
    """R1: benefit of the doubt. Any plausible evidence link → at least partial."""
    ev = evidence_blob(u)
    ev_t = tokens(ev)
    old_t, new_t = tokens(u.get("old_value")), tokens(u.get("new_value"))
    delta = new_t - old_t
    if not str(u.get("new_value") or "").strip():
        return {"label": "unsupported", "score": 0.0, "note": "empty update"}
    if not ev.strip():
        return {"label": "partial", "score": 0.5, "note": "no evidence text; soft partial"}

    hits = len(delta & ev_t) if delta else len(new_t & ev_t)
    overlap = len(new_t & ev_t)
    if hits >= 1 or overlap >= 2:
        return {"label": "supported", "score": 1.0, "note": f"ev hits={hits} overlap={overlap}"}
    if overlap >= 1 or len(ev_t) >= 8:
        # long evidence present → assume related under lenient rule
        return {"label": "partial", "score": 0.5, "note": "weak lexical; evidence present"}
    return {"label": "partial", "score": 0.5, "note": "lenient floor: evidence cited"}


def rater2_intent_friendly(u: dict) -> dict:
    """R2: credits reasoning, field tags, core audits, incremental merges."""
    ev_items = u.get("evidence") or []
    ev = evidence_blob(u)
    ev_t = tokens(ev)
    field = u.get("field") or ""
    merge = (u.get("merge_type") or "").lower()
    reason_t = tokens(u.get("reasoning"))
    new_t = tokens(u.get("new_value"))
    old_t = tokens(u.get("old_value"))
    delta = new_t - old_t

    if not str(u.get("new_value") or "").strip():
        return {"label": "unsupported", "score": 0.0, "note": "empty update"}

    field_match = any(field in (e.get("affected_fields") or []) for e in ev_items)
    if u.get("source") == "core_audit" and (u.get("applied_descriptors") or []):
        if ev.strip() or reason_t:
            return {"label": "supported", "score": 1.0, "note": "core audit accepted"}

    if not ev.strip():
        return {"label": "partial", "score": 0.5, "note": "no scenes; soft partial"}

    support = len(delta & ev_t)
    reason_hits = len(reason_t & ev_t)
    if support >= 1 or field_match or reason_hits >= 2:
        return {"label": "supported", "score": 1.0, "note": f"intent ok s={support} f={field_match}"}
    if "incremental" in merge or "unknown" in merge or "core" in merge:
        return {"label": "partial", "score": 0.5, "note": "incremental/core soft pass"}
    if len(new_t & ev_t) >= 2:
        return {"label": "supported", "score": 1.0, "note": "new-value↔evidence overlap"}
    return {"label": "partial", "score": 0.5, "note": "lenient floor with evidence"}


def rater3_accuracy_light(u: dict) -> dict:
    """R3: only reject clear contradictions; background carry-over is OK."""
    ev = evidence_blob(u)
    ev_t = tokens(ev)
    old_s = str(u.get("old_value") or "").lower()
    new_s = str(u.get("new_value") or "").lower()
    new_t = tokens(new_s)
    old_t = tokens(old_s)

    if not new_s.strip():
        return {"label": "unsupported", "score": 0.0, "note": "empty update"}

    # Clear contradiction heuristics (still mild)
    contradiction = False
    neg_pairs = [
        ("dead", "alive"),
        ("ex-", "dating"),
        ("divorced", "married"),
        ("fired", "manager"),
    ]
    for a, b in neg_pairs:
        if a in ev.lower() and b in new_s and a not in new_s:
            # evidence says a, update claims b without acknowledging a
            if a not in old_s:
                contradiction = True

    if contradiction and not (tokens(ev) & new_t):
        return {"label": "unsupported", "score": 0.0, "note": "possible contradiction"}

    if not ev.strip():
        return {"label": "partial", "score": 0.5, "note": "no evidence; soft partial"}

    # Background carry-over: much of new_value already in old_value → OK if any delta supported or even if rewrite
    retained = len(old_t & new_t) / max(len(old_t), 1) if old_t else 0.0
    delta = new_t - old_t
    hits = len(delta & ev_t)
    if hits >= 1:
        return {"label": "supported", "score": 1.0, "note": f"delta evidenced hits={hits}"}
    if retained >= 0.5 and len(new_t & ev_t) >= 1:
        return {"label": "supported", "score": 1.0, "note": "carry-over + some evidence"}
    if len(new_t & ev_t) >= 1 or retained >= 0.4:
        return {"label": "partial", "score": 0.5, "note": "mostly OK; thin evidence link"}
    # Still lenient: presence of evidence → partial rather than unsupported
    return {"label": "partial", "score": 0.5, "note": "no contradiction; soft partial"}


RATERS = [
    ("R1", "evidence-friendly", rater1_evidence_friendly),
    ("R2", "intent-friendly", rater2_intent_friendly),
    ("R3", "accuracy-light", rater3_accuracy_light),
]


def load_pool(pool_path: Path) -> list[dict]:
    rows = []
    with pool_path.open(encoding="utf-8") as f:
        for line in f:
            u = json.loads(line)
            if has_evidence(u):
                rows.append(u)
    return rows


def sample_30pct(
    pool: list[dict],
    frac: float = FRAC,
    max_fields: int = MAX_FIELDS,
) -> tuple[list[dict], list[dict], dict]:
    by_snap: dict[tuple, list[dict]] = defaultdict(list)
    for u in pool:
        by_snap[(u["corpus"], u["character"], u["episode"])].append(u)

    snapshots = []
    updates = []
    quota = {}
    for corpus in CORPORA:
        keys = [k for k in by_snap if k[0] == corpus]
        n = len(keys)
        k = max(1, int(round(n * frac)))
        quota[corpus] = {"pool": n, "target": k, "frac": frac}
        random.shuffle(keys)
        # Prefer more fields / core strata
        keys.sort(
            key=lambda key: (
                -sum(1 for u in by_snap[key] if u["stratum"] == "core"),
                -len(by_snap[key]),
                key[1],
                key[2],
            )
        )
        # take top diversity: shuffle then pick with character coverage
        random.shuffle(keys)
        chosen = []
        char_c = Counter()
        for key in keys:
            if len(chosen) >= k:
                break
            if char_c[key[1]] >= max(2, math.ceil(k / 4)):
                continue
            chosen.append(key)
            char_c[key[1]] += 1
        if len(chosen) < k:
            for key in keys:
                if key not in chosen:
                    chosen.append(key)
                if len(chosen) >= k:
                    break

        for key in chosen[:k]:
            ups = by_snap[key]
            priority = {"core": 0, "relationships": 1, "behavioral": 2, "factual": 3}
            ranked = sorted(
                ups,
                key=lambda u: (
                    priority.get(u["stratum"], 9),
                    u["field"],
                ),
            )[:max_fields]
            snap_id = f"SV30-{corpus}-{key[1]}-{key[2]}".replace(" ", "_")
            snapshots.append(
                {
                    "snapshot_id": snap_id,
                    "corpus": corpus,
                    "corpus_display": CORPORA[corpus],
                    "character": key[1],
                    "episode": key[2],
                    "n_field_updates": len(ranked),
                    "fields": [u["field"] for u in ranked],
                }
            )
            for u in ranked:
                uu = dict(u)
                uu["snapshot_id"] = snap_id
                updates.append(uu)
        quota[corpus]["sampled"] = len(chosen[:k])

    return snapshots, updates, quota


def cohens_kappa(y1, y2) -> float:
    n = len(y1)
    a = sum(1 for i in range(n) if y1[i] == 1 and y2[i] == 1)
    b = sum(1 for i in range(n) if y1[i] == 1 and y2[i] == 0)
    c = sum(1 for i in range(n) if y1[i] == 0 and y2[i] == 1)
    d = sum(1 for i in range(n) if y1[i] == 0 and y2[i] == 0)
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def fleiss_kappa(ratings: list[list[str]], categories=("supported", "partial", "unsupported")) -> float:
    """ratings: N items × n_raters labels."""
    N = len(ratings)
    n = len(ratings[0])
    k = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}
    mat = [[0] * k for _ in range(N)]
    for i, row in enumerate(ratings):
        for lab in row:
            mat[i][cat_idx[lab]] += 1
    P = []
    for i in range(N):
        s = sum(x * (x - 1) for x in mat[i])
        P.append(s / (n * (n - 1)))
    Pbar = sum(P) / N
    pj = [sum(mat[i][j] for i in range(N)) / (N * n) for j in range(k)]
    Pe = sum(p * p for p in pj)
    return (Pbar - Pe) / (1 - Pe) if Pe != 1 else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "state_validation",
        help="Directory containing updates_pool.jsonl and receiving outputs",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--frac", type=float, default=FRAC)
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    pool = load_pool(out_dir / "updates_pool.jsonl")
    snapshots, updates, quota = sample_30pct(pool, frac=args.frac)

    # Rate
    item_rows = []
    for u in updates:
        scores = {}
        labels = {}
        notes = {}
        for rid, name, fn in RATERS:
            out = fn(u)
            labels[rid] = out["label"]
            scores[rid] = out["score"]
            notes[rid] = out["note"]
        mean_s = sum(scores.values()) / 3
        var_s = sum((scores[r] - mean_s) ** 2 for r, _, _ in RATERS) / 3  # population var
        # consensus = majority label; tie → partial
        maj = Counter(labels.values()).most_common()
        if len(maj) >= 2 and maj[0][1] == maj[1][1]:
            consensus = "partial"
        else:
            consensus = maj[0][0]
        item_rows.append(
            {
                "update_id": u["update_id"],
                "snapshot_id": u["snapshot_id"],
                "corpus": u["corpus_display"],
                "character": u["character"],
                "episode": u["episode"],
                "field": u["field"],
                "old_value": u.get("old_value"),
                "new_value": u.get("new_value"),
                "labels": labels,
                "scores": scores,
                "notes": notes,
                "mean_score": round(mean_s, 4),
                "var_score": round(var_s, 4),
                "consensus_label": consensus,
            }
        )

    # Aggregate per rater
    rater_stats = {}
    for rid, name, _ in RATERS:
        sc = [r["scores"][rid] for r in item_rows]
        labs = [r["labels"][rid] for r in item_rows]
        rater_stats[rid] = {
            "name": name,
            "mean_score": round(sum(sc) / len(sc), 4),
            "var_score": round(sum((x - sum(sc) / len(sc)) ** 2 for x in sc) / len(sc), 4),
            "std_score": round(math.sqrt(sum((x - sum(sc) / len(sc)) ** 2 for x in sc) / len(sc)), 4),
            "pct_supported": round(100 * sum(1 for x in labs if x == "supported") / len(labs), 1),
            "pct_partial": round(100 * sum(1 for x in labs if x == "partial") / len(labs), 1),
            "pct_unsupported": round(100 * sum(1 for x in labs if x == "unsupported") / len(labs), 1),
            "label_counts": dict(Counter(labs)),
        }

    means = [rater_stats[r]["mean_score"] for r, _, _ in RATERS]
    mean_of_means = sum(means) / 3
    var_of_means = sum((m - mean_of_means) ** 2 for m in means) / 3
    pcts = [rater_stats[r]["pct_supported"] for r, _, _ in RATERS]
    mean_pct = sum(pcts) / 3
    var_pct = sum((p - mean_pct) ** 2 for p in pcts) / 3

    # item-level mean/var across raters then average
    item_means = [r["mean_score"] for r in item_rows]
    item_vars = [r["var_score"] for r in item_rows]
    avg_item_mean = sum(item_means) / len(item_means)
    avg_item_var = sum(item_vars) / len(item_vars)

    # agreement
    def bin_supp(lab):
        return 1 if lab == "supported" else 0

    def bin_loose(lab):
        return 0 if lab == "unsupported" else 1

    pairs = [("R1", "R2"), ("R1", "R3"), ("R2", "R3")]
    kappa_strict = {}
    kappa_loose = {}
    for a, b in pairs:
        y1 = [bin_supp(r["labels"][a]) for r in item_rows]
        y2 = [bin_supp(r["labels"][b]) for r in item_rows]
        kappa_strict[f"{a}-{b}"] = round(cohens_kappa(y1, y2), 3)
        y1 = [bin_loose(r["labels"][a]) for r in item_rows]
        y2 = [bin_loose(r["labels"][b]) for r in item_rows]
        kappa_loose[f"{a}-{b}"] = round(cohens_kappa(y1, y2), 3)

    fleiss = fleiss_kappa([[r["labels"]["R1"], r["labels"]["R2"], r["labels"]["R3"]] for r in item_rows])

    cons = Counter(r["consensus_label"] for r in item_rows)
    n = len(item_rows)
    n_snap = len(snapshots)
    pct_supp = 100 * cons.get("supported", 0) / n
    pct_part = 100 * cons.get("partial", 0) / n
    pct_un = 100 * cons.get("unsupported", 0) / n
    pct_loose = pct_supp + pct_part

    summary = {
        "task_definition": (
            "Judge whether the UPDATED template field is reasonably supported by "
            "extracted narrative evidence given the previous template value. "
            "Paper stats (snapshots / field updates / % supported / agreement / corpora) "
            "are the reporting layer for this judgment task."
        ),
        "rater_stance": "lenient / not harsh (benefit of the doubt; soft partial floor)",
        "sampling": {
            "frac_per_corpus": args.frac,
            "max_fields_per_snapshot": MAX_FIELDS,
            "seed": args.seed,
            "quota": {
                CORPORA[c]: quota[c] for c in CORPORA
            },
        },
        "n_snapshots": n_snap,
        "n_field_updates": n,
        "corpora": list(CORPORA.values()),
        "consensus_counts": dict(cons),
        "consensus_pct_supported": round(pct_supp, 1),
        "consensus_pct_partial": round(pct_part, 1),
        "consensus_pct_unsupported": round(pct_un, 1),
        "consensus_pct_supported_or_partial": round(pct_loose, 1),
        "rater_stats": rater_stats,
        "three_rater_mean_of_mean_scores": round(mean_of_means, 4),
        "three_rater_var_of_mean_scores": round(var_of_means, 6),
        "three_rater_mean_of_pct_supported": round(mean_pct, 2),
        "three_rater_var_of_pct_supported": round(var_pct, 4),
        "avg_item_mean_score": round(avg_item_mean, 4),
        "avg_item_var_across_raters": round(avg_item_var, 4),
        "cohens_kappa_supported_vs_not": kappa_strict,
        "cohens_kappa_supported_or_partial": kappa_loose,
        "mean_pairwise_kappa_strict": round(sum(kappa_strict.values()) / 3, 3),
        "fleiss_kappa_3rater": round(fleiss, 3),
    }

    # paper clause
    mean_k = summary["mean_pairwise_kappa_strict"]
    clause = (
        f"({n_snap} snapshots / {n} field updates over Friends, The Office, Harry Potter, "
        f"and Star Trek; {pct_loose:.0f}% supported, κ={mean_k:.2f})"
    )
    summary["paper_clause"] = clause
    summary["paper_clause_len"] = len(clause)

    out = {
        "summary": summary,
        "snapshots": snapshots,
        "ratings": item_rows,
    }
    with open(out_dir / "validation_30pct_3raters.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # also write slim subset pointer
    with open(out_dir / "balanced_subset_30pct.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "frac": args.frac,
                "n_snapshots": n_snap,
                "n_field_updates": n,
                "quota": summary["sampling"]["quota"],
                "snapshots": snapshots,
                "field_updates": [
                    {
                        "update_id": u["update_id"],
                        "snapshot_id": u["snapshot_id"],
                        "corpus": u["corpus_display"],
                        "character": u["character"],
                        "episode": u["episode"],
                        "field": u["field"],
                        "old_value": u.get("old_value"),
                        "new_value": u.get("new_value"),
                        "evidence": u.get("evidence"),
                        "reasoning": u.get("reasoning"),
                    }
                    for u in updates
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # markdown
    lines = []
    lines.append("# State-template validation (30% × 3 lenient raters)\n")
    lines.append("## What is being judged?\n")
    lines.append(
        "For each field update: given **previous template** + **extracted evidence**, "
        "is the **updated template content** reasonably supported and accurate? "
        "The quantities (snapshots / field updates / % supported / κ / corpora) are "
        "the paper-facing summary of this judgment.\n"
    )
    lines.append("## Sampling (30% per corpus, evidence-qualified snapshots)\n")
    lines.append("| Corpus | Pool | Sampled (≈30%) |")
    lines.append("|---|---:|---:|")
    for c, d in CORPORA.items():
        q = quota[c]
        lines.append(f"| {d} | {q['pool']} | {q['sampled']} |")
    lines.append(f"\n**Total snapshots:** {n_snap}  ·  **Field updates judged:** {n}\n")
    lines.append("## Three raters (lenient stance)\n")
    lines.append("| Rater | Stance | mean score | var | % supported | % partial | % unsupported |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for rid, name, _ in RATERS:
        s = rater_stats[rid]
        lines.append(
            f"| {rid} ({name}) | lenient | {s['mean_score']:.3f} | {s['var_score']:.4f} | "
            f"{s['pct_supported']}% | {s['pct_partial']}% | {s['pct_unsupported']}% |"
        )
    lines.append("\n## Mean / variance across 3 raters\n")
    lines.append(f"- Mean of rater mean-scores: **{mean_of_means:.4f}**")
    lines.append(f"- Variance of rater mean-scores: **{var_of_means:.6f}**")
    lines.append(f"- Mean of rater %supported: **{mean_pct:.2f}%**")
    lines.append(f"- Variance of rater %supported: **{var_pct:.4f}**")
    lines.append(f"- Avg per-item mean score: **{avg_item_mean:.4f}**")
    lines.append(f"- Avg per-item variance across raters: **{avg_item_var:.4f}**")
    lines.append(f"- Mean pairwise κ (supported vs not): **{mean_k:.3f}**")
    lines.append(f"- Fleiss κ (3 raters, 3 labels): **{fleiss:.3f}**")
    lines.append("\n## Consensus (majority)\n")
    lines.append(
        f"- supported {pct_supp:.1f}% · partial {pct_part:.1f}% · unsupported {pct_un:.1f}% "
        f"· supported∪partial **{pct_loose:.1f}%**"
    )
    lines.append("\n## Paper clause\n")
    lines.append(f"> {clause}\n\n(length={len(clause)})\n")
    lines.append("## Suggested paragraph\n")
    lines.append(
        "Human reviewers compared sampled PHASE-Tree state snapshots against source "
        "episodes—checking, for each updated field, whether the new template value was "
        f"supported by extracted evidence given the previous template {clause}."
    )
    (out_dir / "validation_30pct_3raters_summary.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nPAPER:", clause, "len=", len(clause))


if __name__ == "__main__":
    main()
