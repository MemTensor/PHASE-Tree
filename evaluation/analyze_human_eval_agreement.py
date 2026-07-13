#!/usr/bin/env python3
"""Agreement / correlation analysis for the 200-sample human evaluation.

Reports:
  1. Inter-annotator agreement among human raters A/B/C
     (Krippendorff's alpha + pairwise Pearson r / Spearman rho).
  2. Correlation between human scores and the GPT-4.1 automatic judge.
  3. Whether human ranking agrees with the automatic judge that
     PHASE-Tree (``m6_phase_tree``) beats baselines.

Usage:
    python evaluation/analyze_human_eval_agreement.py
    python evaluation/analyze_human_eval_agreement.py --he-dir results/human_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent

HUMAN_RATER_NAMES = ["A", "B", "C"]
PT_METHOD = "m6_phase_tree"
DIMS = ["character", "semantic"]


# --------------------------------------------------------------------------- #
# statistics helpers (no numpy/scipy required)
# --------------------------------------------------------------------------- #
def pearson(xs, ys):
    n = len(xs)
    if n == 0:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _rank(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs, ys):
    return pearson(_rank(xs), _rank(ys))


def krippendorff_alpha_interval(columns):
    """Krippendorff's alpha (interval metric) for R raters x N units.

    ``columns`` is a list of R equal-length score lists (one per rater).
    Units may have any number of raters (>=2 used); here all units are fully
    rated, so this reduces to the balanced case.
    """
    n_raters = len(columns)
    n_units = len(columns[0])
    units = [[columns[r][u] for r in range(n_raters)] for u in range(n_units)]

    Do_num = 0.0
    Do_den = 0
    for row in units:
        m_u = len(row)
        for i in range(m_u):
            for j in range(m_u):
                if i == j:
                    continue
                Do_num += (row[i] - row[j]) ** 2
                Do_den += 1
    Do = Do_num / Do_den if Do_den else 0.0

    flat = [v for row in units for v in row]
    m = len(flat)
    De_num = 0.0
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            De_num += (flat[i] - flat[j]) ** 2
    De_den = m * (m - 1)
    De = De_num / De_den if De_den else 0.0
    if De == 0:
        return float("nan")
    return 1 - Do / De


def fmt(x, nd=3):
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:.{nd}f}"


# --------------------------------------------------------------------------- #
def load_scores(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "scores" in data:
        data = data["scores"]
    return {r["sample_id"]: r for r in data}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--he-dir",
        type=Path,
        default=ROOT / "results" / "human_eval",
        help="Directory with samples and rater score JSON files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for MD/CSV (default: --he-dir)",
    )
    args = parser.parse_args()
    he_dir: Path = args.he_dir
    out_dir: Path = args.out_dir or he_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_path = he_dir / "samples_to_score.json"
    human_raters = [(name, he_dir / f"human_scores_{name}.json") for name in HUMAN_RATER_NAMES]
    gpt_file = he_dir / "gpt_judge_scores.json"
    out_md = out_dir / "human_vs_gpt_agreement_analysis.md"
    out_csv = out_dir / "human_vs_gpt_per_method.csv"

    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    sample_by_id = {s["sample_id"]: s for s in samples}
    order = [s["sample_id"] for s in samples]

    # rater_name -> {dim -> [scores in sample order]}
    raters: dict[str, dict[str, list[float]]] = {}
    for name, path in human_raters:
        d = load_scores(path)
        assert set(d) == set(order), f"{name}: sample_id mismatch"
        raters[name] = {
            "character": [d[sid]["character_score"] for sid in order],
            "semantic": [d[sid]["semantic_score"] for sid in order],
        }
    gpt = load_scores(gpt_file)
    assert set(gpt) == set(order), "GPT: sample_id mismatch"
    raters["GPT"] = {
        "character": [gpt[sid]["character_score"] for sid in order],
        "semantic": [gpt[sid]["semantic_score"] for sid in order],
    }

    # human consensus = per-sample mean of A/B/C
    consensus = {
        dim: [mean(raters[n][dim][i] for n, _ in human_raters) for i in range(len(order))]
        for dim in DIMS
    }

    def overall(scores_by_dim):
        return [
            (scores_by_dim["character"][i] + scores_by_dim["semantic"][i]) / 2
            for i in range(len(order))
        ]

    human_names = [n for n, _ in human_raters]

    md: list[str] = []
    md.append("# Human scores vs GPT-4.1 judge: agreement and correlation (Qwen2.5-7B, N=200)")
    md.append("")
    md.append("- Samples: `results/human_eval/samples_to_score.json`")
    md.append("- Human raters: `human_scores_A/B/C.json` (blind, 1–5 integers, two dimensions)")
    md.append("- Automatic judge: `gpt_judge_scores.json` (GPT-4.1, same 200 items)")
    md.append("- Dimensions: `character`, `semantic`; `overall = (character+semantic)/2`")
    md.append("- Statistics implemented without numpy/scipy (Pearson / Spearman / Krippendorff α).")
    md.append("")

    # ------------------------------------------------------------------ #
    # 0. overall means
    # ------------------------------------------------------------------ #
    md.append("## 0. Overall means by rater")
    md.append("")
    md.append("| Rater | character | semantic | overall |")
    md.append("|---|---:|---:|---:|")
    for n in human_names + ["GPT"]:
        c = mean(raters[n]["character"])
        s = mean(raters[n]["semantic"])
        md.append(f"| {n} | {fmt(c)} | {fmt(s)} | {fmt((c + s) / 2)} |")
    cc = mean(consensus["character"])
    cs = mean(consensus["semantic"])
    md.append(f"| **Human consensus (mean A/B/C)** | {fmt(cc)} | {fmt(cs)} | {fmt((cc + cs) / 2)} |")
    md.append("")

    # ------------------------------------------------------------------ #
    # 1. inter-annotator agreement (human A/B/C)
    # ------------------------------------------------------------------ #
    md.append("## 1. Inter-annotator agreement (human A/B/C)")
    md.append("")
    md.append("### 1.1 Krippendorff's α (interval)")
    md.append("")
    md.append("| Dimension | α (A/B/C) | α (A/B/C + GPT) |")
    md.append("|---|---:|---:|")
    for dim in DIMS:
        a3 = krippendorff_alpha_interval([raters[n][dim] for n in human_names])
        a4 = krippendorff_alpha_interval([raters[n][dim] for n in human_names + ["GPT"]])
        md.append(f"| {dim} | {fmt(a3)} | {fmt(a4)} |")
    a3o = krippendorff_alpha_interval([overall(raters[n]) for n in human_names])
    a4o = krippendorff_alpha_interval([overall(raters[n]) for n in human_names + ["GPT"]])
    md.append(f"| overall | {fmt(a3o)} | {fmt(a4o)} |")
    md.append("")
    md.append(
        "> Interpretation: α ≥ 0.80 excellent; ≥ 0.667 conventional publication floor; "
        "< 0.667 indicates disagreement but may still support descriptive claims."
    )
    md.append("")

    md.append("### 1.2 Pairwise Pearson r / Spearman ρ (human raters)")
    md.append("")
    md.append("| Pair | Dimension | Pearson r | Spearman ρ |")
    md.append("|---|---|---:|---:|")
    hpairs = [
        (human_names[0], human_names[1]),
        (human_names[0], human_names[2]),
        (human_names[1], human_names[2]),
    ]
    for a, b in hpairs:
        for dim in DIMS:
            xs, ys = raters[a][dim], raters[b][dim]
            md.append(
                f"| {a} vs {b} | {dim} | {fmt(pearson(xs, ys))} | {fmt(spearman(xs, ys))} |"
            )
    md.append("")

    # ------------------------------------------------------------------ #
    # 2. human vs GPT-4.1 correlation
    # ------------------------------------------------------------------ #
    md.append("## 2. Human vs GPT-4.1 correlation (per sample, N=200)")
    md.append("")
    md.append("| Human side | Dimension | Pearson r | Spearman ρ |")
    md.append("|---|---|---:|---:|")
    human_side = [(n, raters[n]) for n in human_names] + [("consensus", consensus)]
    for label, sc in human_side:
        for dim in DIMS:
            xs, ys = sc[dim], raters["GPT"][dim]
            md.append(
                f"| {label} | {dim} | {fmt(pearson(xs, ys))} | {fmt(spearman(xs, ys))} |"
            )
        xs, ys = overall(sc), overall(raters["GPT"])
        md.append(
            f"| {label} | overall | {fmt(pearson(xs, ys))} | {fmt(spearman(xs, ys))} |"
        )
    md.append("")

    # ------------------------------------------------------------------ #
    # 3. per (track/method) means for consensus vs GPT + method ranking corr
    # ------------------------------------------------------------------ #
    combos = defaultdict(list)  # (track, method) -> [sample idx]
    for i, sid in enumerate(order):
        s = sample_by_id[sid]
        combos[(s["track"], s["method"])].append(i)

    combo_rows = []
    for key in sorted(combos):
        idxs = combos[key]
        row = {
            "track": key[0],
            "method": key[1],
            "n": len(idxs),
            "human_overall": mean(overall(consensus)[i] for i in idxs),
            "gpt_overall": mean(overall(raters["GPT"])[i] for i in idxs),
            "human_character": mean(consensus["character"][i] for i in idxs),
            "gpt_character": mean(raters["GPT"]["character"][i] for i in idxs),
            "human_semantic": mean(consensus["semantic"][i] for i in idxs),
            "gpt_semantic": mean(raters["GPT"]["semantic"][i] for i in idxs),
        }
        combo_rows.append(row)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "track",
                "method",
                "n",
                "human_character",
                "gpt_character",
                "human_semantic",
                "gpt_semantic",
                "human_overall",
                "gpt_overall",
            ]
        )
        for r in combo_rows:
            w.writerow(
                [
                    r["track"],
                    r["method"],
                    r["n"],
                    fmt(r["human_character"]),
                    fmt(r["gpt_character"]),
                    fmt(r["human_semantic"]),
                    fmt(r["gpt_semantic"]),
                    fmt(r["human_overall"]),
                    fmt(r["gpt_overall"]),
                ]
            )

    md.append(
        "## 3. Method-level ranking consistency "
        "(human consensus vs GPT-4.1, 22 track/method combos)"
    )
    md.append("")
    h_over = [r["human_overall"] for r in combo_rows]
    g_over = [r["gpt_overall"] for r in combo_rows]
    md.append(
        f"- Combo-level overall rank correlation: Pearson r = **{fmt(pearson(h_over, g_over))}**, "
        f"Spearman ρ = **{fmt(spearman(h_over, g_over))}** (N=22 combos)"
    )
    for dim in DIMS:
        h = [r[f"human_{dim}"] for r in combo_rows]
        g = [r[f"gpt_{dim}"] for r in combo_rows]
        md.append(
            f"- Combo-level {dim} rank correlation: Pearson r = {fmt(pearson(h, g))}, "
            f"Spearman ρ = {fmt(spearman(h, g))}"
        )
    md.append("")
    md.append(f"> Per-method means: `{out_csv.name}`")
    md.append("")

    # ------------------------------------------------------------------ #
    # 4. PT vs baseline direction check
    # ------------------------------------------------------------------ #
    md.append("## 4. PT (`m6_phase_tree`) vs baseline direction check")
    md.append("")
    md.append(
        "Criterion: whether human consensus and GPT-4.1 agree on the direction "
        "“PT beats baseline” (Δ = PT − baseline; same sign ⇒ agree)."
    )
    md.append("")

    def group_mean(idxs, series):
        return mean(series[i] for i in idxs) if idxs else float("nan")

    pt_idx = [i for i, sid in enumerate(order) if sample_by_id[sid]["method"] == PT_METHOD]
    base_idx = [i for i, sid in enumerate(order) if sample_by_id[sid]["method"] != PT_METHOD]

    md.append("### 4.1 Pooled: all `m6_phase_tree` vs all other methods")
    md.append("")
    md.append(f"- PT samples = {len(pt_idx)}, baseline samples = {len(base_idx)}")
    md.append("")
    md.append("| Dimension | Judge | PT | baseline | Δ (PT−base) | PT higher? |")
    md.append("|---|---|---:|---:|---:|---|")
    for dim in DIMS + ["overall"]:
        if dim == "overall":
            h_series, g_series = overall(consensus), overall(raters["GPT"])
        else:
            h_series, g_series = consensus[dim], raters["GPT"][dim]
        for lab, series in (("Human consensus", h_series), ("GPT-4.1", g_series)):
            pt = group_mean(pt_idx, series)
            bs = group_mean(base_idx, series)
            d = pt - bs
            mark = "↑" if d > 0 else ("↓" if d < 0 else "=")
            md.append(f"| {dim} | {lab} | {fmt(pt)} | {fmt(bs)} | {fmt(d, 3)} | {mark} |")
    md.append("")

    md.append("### 4.2 Per track: PT vs other methods in the same track (pooled baseline)")
    md.append("")
    md.append(
        "| track | Dimension | Human Δ | Human dir | GPT Δ | GPT dir | Same direction? |"
    )
    md.append("|---|---|---:|---|---:|---|---|")
    tracks_with_pt = sorted(
        {
            sample_by_id[sid]["track"]
            for sid in order
            if sample_by_id[sid]["method"] == PT_METHOD
        }
    )
    consistency_flags = []
    for track in tracks_with_pt:
        pt_i = [
            i
            for i, sid in enumerate(order)
            if sample_by_id[sid]["track"] == track
            and sample_by_id[sid]["method"] == PT_METHOD
        ]
        bs_i = [
            i
            for i, sid in enumerate(order)
            if sample_by_id[sid]["track"] == track
            and sample_by_id[sid]["method"] != PT_METHOD
        ]
        for dim in DIMS + ["overall"]:
            if dim == "overall":
                h_series, g_series = overall(consensus), overall(raters["GPT"])
            else:
                h_series, g_series = consensus[dim], raters["GPT"][dim]
            hd = group_mean(pt_i, h_series) - group_mean(bs_i, h_series)
            gd = group_mean(pt_i, g_series) - group_mean(bs_i, g_series)
            same = (hd > 0) == (gd > 0)
            consistency_flags.append(same)

            def arrow(v: float) -> str:
                return "↑" if v > 0 else ("↓" if v < 0 else "=")

            md.append(
                f"| {track} | {dim} | {fmt(hd)} | {arrow(hd)} | {fmt(gd)} | {arrow(gd)} | "
                f"{'yes' if same else 'no'} |"
            )
    md.append("")
    n_same = sum(consistency_flags)
    md.append(f"- Cells with matching direction: **{n_same}/{len(consistency_flags)}**")
    md.append("")

    md.append("### 4.3 Prompt track: PT (m6) vs each baseline method")
    md.append("")
    md.append("| Baseline method | Human overall Δ | GPT overall Δ | Same direction? |")
    md.append("|---|---:|---:|---|")
    prompt_methods = sorted(
        {
            sample_by_id[sid]["method"]
            for sid in order
            if sample_by_id[sid]["track"] == "prompt"
        }
    )
    pt_prompt = [
        i
        for i, sid in enumerate(order)
        if sample_by_id[sid]["track"] == "prompt"
        and sample_by_id[sid]["method"] == PT_METHOD
    ]
    h_ov, g_ov = overall(consensus), overall(raters["GPT"])
    for m in prompt_methods:
        if m == PT_METHOD:
            continue
        b_i = [
            i
            for i, sid in enumerate(order)
            if sample_by_id[sid]["track"] == "prompt" and sample_by_id[sid]["method"] == m
        ]
        hd = group_mean(pt_prompt, h_ov) - group_mean(b_i, h_ov)
        gd = group_mean(pt_prompt, g_ov) - group_mean(b_i, g_ov)
        same = (hd > 0) == (gd > 0)
        md.append(f"| {m} | {fmt(hd)} | {fmt(gd)} | {'yes' if same else 'no'} |")
    md.append("")

    # ------------------------------------------------------------------ #
    # 4.4 PT vs best other baseline
    # ------------------------------------------------------------------ #
    md.append("### 4.4 PT vs best non-PT baseline (headline check)")
    md.append("")
    md.append(
        "For each setting, take the **non-PT method with the highest overall** as the "
        "best baseline, then compare against PT (`m6_phase_tree`): "
        "Δ = PT − best_baseline."
    )
    md.append("")

    def best_non_pt(rows, score_key):
        cands = [r for r in rows if r["method"] != PT_METHOD]
        if not cands:
            return None
        return max(cands, key=lambda r: r[score_key])

    def verdict(delta):
        if delta > 0:
            return "yes, PT better"
        if delta < 0:
            return "no, best baseline better"
        return "tie"

    md.append("#### (a) Prompt track (main paper comparison setting)")
    md.append("")
    md.append(
        "| Judge | Dimension | PT | best baseline | best method | Δ | PT better? |"
    )
    md.append("|---|---|---:|---:|---|---:|---|")
    prompt_rows = [r for r in combo_rows if r["track"] == "prompt"]
    pt_prompt_row = next(r for r in prompt_rows if r["method"] == PT_METHOD)
    for who, pfx in (("Human consensus", "human"), ("GPT-4.1", "gpt")):
        for dim in ("character", "semantic", "overall"):
            key = f"{pfx}_{dim}"
            best = best_non_pt(prompt_rows, key)
            pt_v = pt_prompt_row[key]
            bs_v = best[key]
            d = pt_v - bs_v
            md.append(
                f"| {who} | {dim} | {fmt(pt_v)} | {fmt(bs_v)} | `{best['method']}` "
                f"(n={best['n']}) | {fmt(d)} | {verdict(d)} |"
            )
    md.append("")

    md.append(
        "#### (b) Tracks that include PT: PT vs best same-track baseline (overall)"
    )
    md.append("")
    md.append(
        "| track | Judge | PT overall | best baseline | best method | Δ | PT better? |"
    )
    md.append("|---|---|---:|---:|---|---:|---|")
    for track in tracks_with_pt:
        trows = [r for r in combo_rows if r["track"] == track]
        pt_row = next(r for r in trows if r["method"] == PT_METHOD)
        for who, pfx in (("Human consensus", "human"), ("GPT-4.1", "gpt")):
            key = f"{pfx}_overall"
            best = best_non_pt(trows, key)
            d = pt_row[key] - best[key]
            md.append(
                f"| {track} | {who} | {fmt(pt_row[key])} | {fmt(best[key])} | "
                f"`{best['method']}` (n={best['n']}) | {fmt(d)} | {verdict(d)} |"
            )
    md.append("")

    md.append(
        "#### (c) All 22 track/method combos: does the best PT beat the best non-PT?"
    )
    md.append("")
    md.append(
        "| Judge | Dimension | Best PT combo | PT score | Best non-PT combo | "
        "non-PT score | Δ | PT better? |"
    )
    md.append("|---|---|---|---:|---|---:|---:|---|")
    pt_combos = [r for r in combo_rows if r["method"] == PT_METHOD]
    non_pt_combos = [r for r in combo_rows if r["method"] != PT_METHOD]
    for who, pfx in (("Human consensus", "human"), ("GPT-4.1", "gpt")):
        for dim in ("character", "semantic", "overall"):
            key = f"{pfx}_{dim}"
            best_pt = max(pt_combos, key=lambda r: r[key])
            best_np = max(non_pt_combos, key=lambda r: r[key])
            d = best_pt[key] - best_np[key]
            md.append(
                f"| {who} | {dim} | `{best_pt['track']}/{best_pt['method']}` "
                f"(n={best_pt['n']}) | {fmt(best_pt[key])} | "
                f"`{best_np['track']}/{best_np['method']}` (n={best_np['n']}) | "
                f"{fmt(best_np[key])} | {fmt(d)} | {verdict(d)} |"
            )
    md.append("")

    h_best = best_non_pt(prompt_rows, "human_overall")
    g_best = best_non_pt(prompt_rows, "gpt_overall")
    h_d = pt_prompt_row["human_overall"] - h_best["human_overall"]
    g_d = pt_prompt_row["gpt_overall"] - g_best["gpt_overall"]
    md.append("#### Takeaway")
    md.append("")
    md.append(
        f"- **prompt track / human consensus**: PT overall = {fmt(pt_prompt_row['human_overall'])}, "
        f"best baseline = `{h_best['method']}` ({fmt(h_best['human_overall'])}), "
        f"Δ = **{fmt(h_d)}** → **{verdict(h_d)}**."
    )
    md.append(
        f"- **prompt track / GPT-4.1**: PT overall = {fmt(pt_prompt_row['gpt_overall'])}, "
        f"best baseline = `{g_best['method']}` ({fmt(g_best['gpt_overall'])}), "
        f"Δ = **{fmt(g_d)}** → **{verdict(g_d)}**."
    )
    md.append("")

    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[md]  -> {out_md}")
    print(f"[csv] -> {out_csv}")


if __name__ == "__main__":
    main()
