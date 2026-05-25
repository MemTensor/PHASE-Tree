"""Aggregate scores across experiments and produce a comparison report.

Reads ``judge_scores.jsonl`` and ``embedding_scores.jsonl`` from each
experiment directory under a dataset results folder, then computes:

  * Per-experiment mean scores with 95% confidence intervals.
  * Pairwise deltas and statistical tests vs. a chosen baseline.
  * Win/tie/loss rates vs. the baseline.
  * Per-character breakdown (optional).
  * LaTeX-ready tables for paper inclusion.

Outputs:
  * ``{results_dir}/summary.json``   — machine-readable aggregate.
  * ``{results_dir}/report.md``      — formatted markdown report.
  * ``{results_dir}/tables.tex``     — LaTeX tables.
  * Stdout                           — human-readable summary.

Usage::

    python evaluation/report.py \\
        --results_dir results/RAIDEN/prompt/main \\
        --experiments m1_context_only m2_raw_profile m3_naive_rewrite m4_static_tree m6_phase_tree \\
        --splits random_test ood_test \\
        --baseline m2_raw_profile
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types that the stdlib encoder cannot serialize."""

    def default(self, o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


METHOD_LABELS = {
    "m1_context_only": "Context-Only (lower bound)",
    "m2_raw_profile": "Raw-Profile (baseline)",
    "m3_naive_rewrite": "Naive-Rewrite",
    "m4_static_tree": "Static-Tree",
    "m5_dynamic_tree": "Dynamic-Tree",
    "m6_phase_tree": "PHASE-Tree (ours)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    items = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) for a list of values."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = np.mean(values)
    if n < 2:
        return float(m), float(m), float(m)
    se = stats.sem(values)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return float(m), float(m - h), float(m + h)


def paired_test(a: list[float], b: list[float]) -> dict:
    """Wilcoxon signed-rank and paired t-test."""
    n = min(len(a), len(b))
    if n < 5:
        return {"wilcoxon_p": None, "ttest_p": None, "n": n, "significant": False}
    a, b = a[:n], b[:n]
    try:
        _, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_p = None
    _, t_p = stats.ttest_rel(a, b)
    sig = (t_p < 0.05) if t_p is not None else False
    return {
        "wilcoxon_p": round(float(w_p), 6) if w_p is not None else None,
        "ttest_p": round(float(t_p), 6),
        "n": n,
        "significant": sig,
    }


def effect_size_cohen_d(a: list[float], b: list[float]) -> float:
    """Paired Cohen's d."""
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    diff = [b[i] - a[i] for i in range(n)]
    md = np.mean(diff)
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(md / sd)


# ---------------------------------------------------------------------------
# Collect scores
# ---------------------------------------------------------------------------

def collect_experiment_scores(exp_dir: str, splits: list[str] | None = None) -> dict:
    """Return {question_id: {role, character_score, semantic_score, embedding_similarity, split}}.

    Supports two directory layouts:
      1. Split-based: <exp_dir>/<split>/judge_scores.jsonl  (new)
      2. Flat:        <exp_dir>/judge_scores.jsonl          (legacy)
    """
    judge = {}
    embed = {}

    if splits:
        for split in splits:
            split_dir = os.path.join(exp_dir, split)
            for r in load_jsonl(os.path.join(split_dir, "judge_scores.jsonl")):
                r["split"] = split
                judge[r["question_id"]] = r
            for r in load_jsonl(os.path.join(split_dir, "embedding_scores.jsonl")):
                embed[r["question_id"]] = r

    if not judge:
        for r in load_jsonl(os.path.join(exp_dir, "judge_scores.jsonl")):
            judge[r["question_id"]] = r
        for r in load_jsonl(os.path.join(exp_dir, "embedding_scores.jsonl")):
            embed[r["question_id"]] = r

    merged = {}
    for qid in judge:
        j = judge[qid]
        e = embed.get(qid, {})
        merged[qid] = {
            "role": j.get("role", ""),
            "character_score": j["character_score"],
            "semantic_score": j["semantic_score"],
            "embedding_similarity": e.get("embedding_similarity", 0.0),
            "split": j.get("split", "all"),
        }
    return merged


# ---------------------------------------------------------------------------
# Report computation
# ---------------------------------------------------------------------------

def compute_report(
    results_dir: str,
    experiment_names: list[str],
    baseline_name: str,
    splits: list[str] | None = None,
) -> dict:
    all_scores: dict[str, dict] = {}
    for name in experiment_names:
        exp_dir = os.path.join(results_dir, name)
        scores = collect_experiment_scores(exp_dir, splits)
        if scores:
            all_scores[name] = scores
        else:
            print(f"  WARNING: No scores for experiment '{name}'")

    if not all_scores:
        print("ERROR: No experiment scores found.")
        return {}

    # --- Per-experiment summary with CI ---
    exp_summary = {}
    for name, scores in all_scores.items():
        char_s = [s["character_score"] for s in scores.values()]
        sem_s = [s["semantic_score"] for s in scores.values()]
        emb_s = [s["embedding_similarity"] for s in scores.values()]

        char_m, char_lo, char_hi = mean_ci(char_s)
        sem_m, sem_lo, sem_hi = mean_ci(sem_s)
        emb_m, emb_lo, emb_hi = mean_ci(emb_s)

        exp_summary[name] = {
            "n": len(scores),
            "character": {"mean": round(char_m, 3), "ci_lo": round(char_lo, 3), "ci_hi": round(char_hi, 3)},
            "semantic": {"mean": round(sem_m, 3), "ci_lo": round(sem_lo, 3), "ci_hi": round(sem_hi, 3)},
            "embedding": {"mean": round(emb_m, 4), "ci_lo": round(emb_lo, 4), "ci_hi": round(emb_hi, 4)},
        }

    # --- Per-character breakdown ---
    per_character: dict[str, dict] = {}
    for name, scores in all_scores.items():
        role_scores = defaultdict(lambda: {"char": [], "sem": [], "emb": []})
        for s in scores.values():
            role_scores[s["role"]]["char"].append(s["character_score"])
            role_scores[s["role"]]["sem"].append(s["semantic_score"])
            role_scores[s["role"]]["emb"].append(s["embedding_similarity"])
        per_character[name] = {
            role: {
                "n": len(v["char"]),
                "character_mean": round(float(np.mean(v["char"])), 3),
                "semantic_mean": round(float(np.mean(v["sem"])), 3),
                "embedding_mean": round(float(np.mean(v["emb"])), 4),
            }
            for role, v in sorted(role_scores.items())
        }

    # --- Pairwise comparisons vs baseline ---
    comparisons = {}
    baseline_scores = all_scores.get(baseline_name, {})
    for name, scores in all_scores.items():
        if name == baseline_name:
            continue
        common = sorted(set(baseline_scores.keys()) & set(scores.keys()))
        if not common:
            continue

        bl_c = [baseline_scores[t]["character_score"] for t in common]
        bl_s = [baseline_scores[t]["semantic_score"] for t in common]
        bl_e = [baseline_scores[t]["embedding_similarity"] for t in common]
        ex_c = [scores[t]["character_score"] for t in common]
        ex_s = [scores[t]["semantic_score"] for t in common]
        ex_e = [scores[t]["embedding_similarity"] for t in common]

        def win_rate(bl, ex):
            wins = sum(1 for b, e in zip(bl, ex) if e > b)
            ties = sum(1 for b, e in zip(bl, ex) if e == b)
            n = len(bl)
            return {
                "win": wins, "tie": ties, "loss": n - wins - ties, "n": n,
                "win_pct": round(wins / n * 100, 1) if n else 0,
            }

        comparisons[name] = {
            "n_paired": len(common),
            "character": {
                "delta": round(float(np.mean(ex_c)) - float(np.mean(bl_c)), 3),
                "cohen_d": round(effect_size_cohen_d(bl_c, ex_c), 3),
                "stat": paired_test(bl_c, ex_c),
                "winrate": win_rate(bl_c, ex_c),
            },
            "semantic": {
                "delta": round(float(np.mean(ex_s)) - float(np.mean(bl_s)), 3),
                "cohen_d": round(effect_size_cohen_d(bl_s, ex_s), 3),
                "stat": paired_test(bl_s, ex_s),
                "winrate": win_rate(bl_s, ex_s),
            },
            "embedding": {
                "delta": round(float(np.mean(ex_e)) - float(np.mean(bl_e)), 4),
                "cohen_d": round(effect_size_cohen_d(bl_e, ex_e), 3),
                "stat": paired_test(bl_e, ex_e),
                "winrate": win_rate(bl_e, ex_e),
            },
        }

    # --- Per-split summary ---
    per_split: dict[str, dict] = {}
    detected_splits = set()
    for scores in all_scores.values():
        for s in scores.values():
            detected_splits.add(s.get("split", "all"))

    if len(detected_splits) > 1:
        for split_name in sorted(detected_splits):
            split_summary = {}
            for name, scores in all_scores.items():
                split_scores = [s for s in scores.values()
                                if s.get("split") == split_name]
                if not split_scores:
                    continue
                char_s = [s["character_score"] for s in split_scores]
                sem_s = [s["semantic_score"] for s in split_scores]
                emb_s = [s["embedding_similarity"] for s in split_scores]
                char_m, char_lo, char_hi = mean_ci(char_s)
                sem_m, sem_lo, sem_hi = mean_ci(sem_s)
                emb_m, emb_lo, emb_hi = mean_ci(emb_s)
                split_summary[name] = {
                    "n": len(split_scores),
                    "character": {"mean": round(char_m, 3), "ci_lo": round(char_lo, 3), "ci_hi": round(char_hi, 3)},
                    "semantic": {"mean": round(sem_m, 3), "ci_lo": round(sem_lo, 3), "ci_hi": round(sem_hi, 3)},
                    "embedding": {"mean": round(emb_m, 4), "ci_lo": round(emb_lo, 4), "ci_hi": round(emb_hi, 4)},
                }
            per_split[split_name] = split_summary

    return {
        "experiment_summary": exp_summary,
        "per_split": per_split,
        "per_character": per_character,
        "vs_baseline": comparisons,
        "baseline": baseline_name,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt_ci(metric: dict, fmt: str = ".3f") -> str:
    """Format mean ± CI as 'mean (lo–hi)'."""
    m = metric["mean"]
    lo = metric["ci_lo"]
    hi = metric["ci_hi"]
    return f"{m:{fmt}} ({lo:{fmt}}–{hi:{fmt}})"


def _sig_marker(p_val) -> str:
    if p_val is None:
        return ""
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return ""


def generate_markdown(report: dict, experiment_names: list[str],
                      per_character: bool = False) -> str:
    lines = []
    baseline_name = report["baseline"]
    summary = report.get("experiment_summary", {})
    comparisons = report.get("vs_baseline", {})

    from datetime import datetime
    lines.append("# Evaluation Report\n")
    lines.append(f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **Baseline**: {METHOD_LABELS.get(baseline_name, baseline_name)}")
    lines.append(f"- **Methods**: {len(experiment_names)}")
    total_n = sum(s.get("n", 0) for s in summary.values())
    lines.append(f"- **Total samples**: {total_n}\n")

    # --- Main results table ---
    lines.append("## Main Results (mean with 95% CI)\n")
    lines.append("| Method | N | Character ↑ | Semantic ↑ | Embedding ↑ |")
    lines.append("|--------|--:|:-----------:|:----------:|:-----------:|")
    for name in experiment_names:
        s = summary.get(name)
        if not s:
            continue
        label = METHOD_LABELS.get(name, name)
        char_str = _fmt_ci(s["character"])
        sem_str = _fmt_ci(s["semantic"])
        emb_str = _fmt_ci(s["embedding"], ".4f")
        lines.append(f"| {label} | {s['n']} | {char_str} | {sem_str} | {emb_str} |")
    lines.append("")

    # --- Delta table ---
    if comparisons:
        lines.append(f"## Δ vs Baseline ({METHOD_LABELS.get(baseline_name, baseline_name)})\n")
        lines.append("| Method | Δ Char | Δ Sem | Δ Emb | Cohen's d (Char) | p-value |")
        lines.append("|--------|-------:|------:|------:|-----------------:|--------:|")
        for name in experiment_names:
            c = comparisons.get(name)
            if not c:
                continue
            label = METHOD_LABELS.get(name, name)
            dc = c["character"]["delta"]
            ds = c["semantic"]["delta"]
            de = c["embedding"]["delta"]
            cd = c["character"]["cohen_d"]
            p = c["character"]["stat"]["ttest_p"]
            sig = _sig_marker(p)
            lines.append(f"| {label} | {dc:+.3f} | {ds:+.3f} | {de:+.4f} | {cd:.3f} | {p:.4f}{sig} |")
        lines.append("")

        # --- Win rate ---
        lines.append(f"## Win Rate vs Baseline\n")
        lines.append("| Method | Metric | Win | Tie | Loss | Win% |")
        lines.append("|--------|--------|----:|----:|-----:|-----:|")
        for name in experiment_names:
            c = comparisons.get(name)
            if not c:
                continue
            label = METHOD_LABELS.get(name, name)
            for metric_key, metric_label in [("character", "Character"), ("semantic", "Semantic"), ("embedding", "Embedding")]:
                w = c[metric_key]["winrate"]
                lines.append(f"| {label} | {metric_label} | {w['win']} | {w['tie']} | {w['loss']} | {w['win_pct']:.1f}% |")
        lines.append("")

    # --- Per-character breakdown ---
    if per_character and report.get("per_character"):
        per_char_data = report["per_character"]
        all_roles = sorted(set(
            role for exp_data in per_char_data.values() for role in exp_data
        ))
        if all_roles:
            lines.append("## Per-Character Breakdown (Character Score)\n")
            header_methods = [n for n in experiment_names if n in per_char_data]
            header_labels = [METHOD_LABELS.get(n, n) for n in header_methods]
            lines.append("| Character | " + " | ".join(header_labels) + " |")
            lines.append("|-----------|" + "|".join(["---:" for _ in header_methods]) + "|")
            for role in all_roles:
                row = f"| {role} |"
                for m in header_methods:
                    val = per_char_data.get(m, {}).get(role, {}).get("character_mean", 0)
                    row += f" {val:.3f} |"
                lines.append(row)
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def generate_latex(report: dict, experiment_names: list[str]) -> str:
    lines = []
    summary = report.get("experiment_summary", {})
    comparisons = report.get("vs_baseline", {})
    baseline_name = report["baseline"]

    # ---- Table 1: Main results with CI ----
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Main results on the evaluation set. "
                 r"Scores are reported as mean$_{\pm \text{95\% CI}}$. "
                 r"$\uparrow$ indicates higher is better. "
                 r"Best results in \textbf{bold}. "
                 r"Significance vs.\ baseline: "
                 r"\textsuperscript{*} $p<.05$, "
                 r"\textsuperscript{**} $p<.01$, "
                 r"\textsuperscript{***} $p<.001$ (paired $t$-test).}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{l c c c}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{Char.\,$\uparrow$} "
                 r"& \textbf{Sem.\,$\uparrow$} & \textbf{Emb.\,$\uparrow$} \\")
    lines.append(r"\midrule")

    char_vals = {n: summary[n]["character"]["mean"] for n in experiment_names if n in summary}
    sem_vals = {n: summary[n]["semantic"]["mean"] for n in experiment_names if n in summary}
    emb_vals = {n: summary[n]["embedding"]["mean"] for n in experiment_names if n in summary}
    best_char = max(char_vals.values()) if char_vals else 0
    best_sem = max(sem_vals.values()) if sem_vals else 0
    best_emb = max(emb_vals.values()) if emb_vals else 0

    for name in experiment_names:
        s = summary.get(name)
        if not s:
            continue
        label = METHOD_LABELS.get(name, name).replace("_", r"\_")

        c = comparisons.get(name, {})
        char_sig = _sig_marker(c.get("character", {}).get("stat", {}).get("ttest_p")) if c else ""
        sem_sig = _sig_marker(c.get("semantic", {}).get("stat", {}).get("ttest_p")) if c else ""
        emb_sig = _sig_marker(c.get("embedding", {}).get("stat", {}).get("ttest_p")) if c else ""

        char_m = s["character"]["mean"]
        sem_m = s["semantic"]["mean"]
        emb_m = s["embedding"]["mean"]
        char_ci = s["character"]["ci_hi"] - char_m
        sem_ci = s["semantic"]["ci_hi"] - sem_m
        emb_ci = s["embedding"]["ci_hi"] - emb_m

        char_str = f"{char_m:.2f}$_{{\\pm {char_ci:.2f}}}${char_sig}"
        sem_str = f"{sem_m:.2f}$_{{\\pm {sem_ci:.2f}}}${sem_sig}"
        emb_str = f"{emb_m:.3f}$_{{\\pm {emb_ci:.3f}}}${emb_sig}"

        if abs(char_m - best_char) < 1e-4:
            char_str = r"\textbf{" + char_str + "}"
        if abs(sem_m - best_sem) < 1e-4:
            sem_str = r"\textbf{" + sem_str + "}"
        if abs(emb_m - best_emb) < 1e-5:
            emb_str = r"\textbf{" + emb_str + "}"

        lines.append(f"{label} & {char_str} & {sem_str} & {emb_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # ---- Table 2: Effect sizes and significance ----
    if comparisons:
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\small")
        bl_label = METHOD_LABELS.get(baseline_name, baseline_name).replace("_", r"\_")
        lines.append(r"\caption{Pairwise comparison vs.\ " + bl_label +
                     r". $\Delta$: improvement over baseline. " +
                     r"$d$: Cohen's $d$ effect size.}")
        lines.append(r"\label{tab:pairwise}")
        lines.append(r"\begin{tabular}{l r r r r r}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Method} & \textbf{$\Delta$Char.} & \textbf{$\Delta$Sem.} "
                     r"& \textbf{$\Delta$Emb.} & \textbf{$d$ (Char.)} & \textbf{$p$-value} \\")
        lines.append(r"\midrule")

        for name in experiment_names:
            c = comparisons.get(name)
            if not c:
                continue
            label = METHOD_LABELS.get(name, name).replace("_", r"\_")
            dc = c["character"]["delta"]
            ds = c["semantic"]["delta"]
            de = c["embedding"]["delta"]
            cd = c["character"]["cohen_d"]
            p = c["character"]["stat"]["ttest_p"]
            sig = _sig_marker(p)

            p_str = f"${p:.4f}$" if p >= 0.001 else f"$<.001$"
            if sig:
                p_str += r"\textsuperscript{" + sig + "}"

            lines.append(f"{label} & {dc:+.3f} & {ds:+.3f} & {de:+.4f} "
                         f"& {cd:.3f} & {p_str} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_report(report: dict, experiment_names: list[str],
                 per_character: bool = False):
    summary = report.get("experiment_summary", {})
    comparisons = report.get("vs_baseline", {})
    baseline_name = report["baseline"]

    W = 82

    print(f"\n{'═' * W}")
    print(f"  EVALUATION REPORT")
    print(f"  Baseline: {METHOD_LABELS.get(baseline_name, baseline_name)}")
    print(f"{'═' * W}")

    # --- Main table with CI ---
    print(f"\n{'─' * W}")
    print(f"  {'Method':<28} {'N':>5}  {'Character ↑':>16}  "
          f"{'Semantic ↑':>16}  {'Embedding ↑':>16}")
    print(f"{'─' * W}")
    for name in experiment_names:
        s = summary.get(name)
        if not s:
            continue
        label = METHOD_LABELS.get(name, name)[:28]
        c_ci = f"{s['character']['mean']:.3f}±{s['character']['ci_hi'] - s['character']['mean']:.3f}"
        s_ci = f"{s['semantic']['mean']:.3f}±{s['semantic']['ci_hi'] - s['semantic']['mean']:.3f}"
        e_ci = f"{s['embedding']['mean']:.4f}±{s['embedding']['ci_hi'] - s['embedding']['mean']:.4f}"
        print(f"  {label:<28} {s['n']:>5}  {c_ci:>16}  {s_ci:>16}  {e_ci:>16}")
    print(f"{'─' * W}")

    # --- Per-split table ---
    per_split = report.get("per_split", {})
    if per_split:
        for split_name, split_data in sorted(per_split.items()):
            print(f"\n  [{split_name}]")
            print(f"  {'Method':<28} {'N':>5}  {'Character ↑':>16}  "
                  f"{'Semantic ↑':>16}  {'Embedding ↑':>16}")
            for name in experiment_names:
                s = split_data.get(name)
                if not s:
                    continue
                label = METHOD_LABELS.get(name, name)[:28]
                c_ci = f"{s['character']['mean']:.3f}±{s['character']['ci_hi'] - s['character']['mean']:.3f}"
                s_ci = f"{s['semantic']['mean']:.3f}±{s['semantic']['ci_hi'] - s['semantic']['mean']:.3f}"
                e_ci = f"{s['embedding']['mean']:.4f}±{s['embedding']['ci_hi'] - s['embedding']['mean']:.4f}"
                print(f"  {label:<28} {s['n']:>5}  {c_ci:>16}  {s_ci:>16}  {e_ci:>16}")
        print(f"{'─' * W}")

    # --- Delta table ---
    if comparisons:
        print(f"\n  Δ vs {METHOD_LABELS.get(baseline_name, baseline_name)}")
        print(f"{'─' * W}")
        print(f"  {'Method':<28} {'ΔChar':>7} {'ΔSem':>7} {'ΔEmb':>8}"
              f"  {'d(C)':>6} {'d(S)':>6} {'d(E)':>6}  {'p(C)':>10}")
        print(f"{'─' * W}")
        for name in experiment_names:
            c = comparisons.get(name)
            if not c:
                continue
            label = METHOD_LABELS.get(name, name)[:28]
            dc = c["character"]["delta"]
            ds = c["semantic"]["delta"]
            de = c["embedding"]["delta"]
            d_c = c["character"]["cohen_d"]
            d_s = c["semantic"]["cohen_d"]
            d_e = c["embedding"]["cohen_d"]
            p = c["character"]["stat"]["ttest_p"]
            sig = _sig_marker(p)
            print(f"  {label:<28} {dc:>+7.3f} {ds:>+7.3f} {de:>+8.4f}"
                  f"  {d_c:>6.3f} {d_s:>6.3f} {d_e:>6.3f}  {p:>7.4f}{sig}")
        print(f"{'─' * W}")

        # --- Win rate ---
        print(f"\n  Win / Tie / Loss vs Baseline (Character score)")
        print(f"{'─' * W}")
        print(f"  {'Method':<28} {'Win':>5} {'Tie':>5} {'Loss':>5}  {'Win%':>6}  "
              f"{'Sem Win%':>8}  {'Emb Win%':>8}")
        print(f"{'─' * W}")
        for name in experiment_names:
            c = comparisons.get(name)
            if not c:
                continue
            label = METHOD_LABELS.get(name, name)[:28]
            wc = c["character"]["winrate"]
            ws = c["semantic"]["winrate"]
            we = c["embedding"]["winrate"]
            print(f"  {label:<28} {wc['win']:>5} {wc['tie']:>5} {wc['loss']:>5}"
                  f"  {wc['win_pct']:>5.1f}%  {ws['win_pct']:>7.1f}%  {we['win_pct']:>7.1f}%")
        print(f"{'─' * W}")

    # --- Per-character breakdown ---
    if per_character and report.get("per_character"):
        per_char_data = report["per_character"]
        all_roles = sorted(set(
            role for exp_data in per_char_data.values() for role in exp_data
        ))
        if all_roles:
            print(f"\n{'─' * W}")
            print(f"  Per-Character Breakdown (Character score mean)")
            print(f"{'─' * W}")
            header_methods = [n for n in experiment_names if n in per_char_data]
            header_labels = [METHOD_LABELS.get(n, n)[:12] for n in header_methods]
            print(f"  {'Character':<16}", end="")
            for lbl in header_labels:
                print(f"  {lbl:>12}", end="")
            print()
            print(f"  {'─' * 16}", end="")
            for _ in header_methods:
                print(f"  {'─' * 12}", end="")
            print()
            for role in all_roles:
                print(f"  {role:<16}", end="")
                for m in header_methods:
                    val = per_char_data.get(m, {}).get(role, {}).get("character_mean", 0)
                    print(f"  {val:>12.3f}", end="")
                print()
            print(f"{'─' * W}")

    print()


# ---------------------------------------------------------------------------
# Token statistics (from meta.json)
# ---------------------------------------------------------------------------

def _find_meta_files(results_dir: str, name: str) -> list[str]:
    """Find all meta.json files for an experiment (flat or split-based layout)."""
    flat = os.path.join(results_dir, name, "meta.json")
    if os.path.exists(flat):
        return [flat]
    paths = []
    exp_dir = os.path.join(results_dir, name)
    if os.path.isdir(exp_dir):
        for entry in os.listdir(exp_dir):
            candidate = os.path.join(exp_dir, entry, "meta.json")
            if os.path.exists(candidate):
                paths.append(candidate)
    return paths


def _merge_token_stats(all_ts: list[dict]) -> dict:
    """Weighted-merge token_stats dicts from multiple splits."""
    if len(all_ts) == 1:
        return all_ts[0]

    KEYS = ("profile_tokens", "context_tokens", "output_tokens",
            "prompt_tokens", "prediction_tokens")
    merged: dict = {}
    total_n = 0
    for ts in all_ts:
        total_n += ts.get("num_samples", 0)

    for key in KEYS:
        parts = [(ts[key], ts.get("num_samples", 0))
                 for ts in all_ts if key in ts]
        if not parts:
            continue
        w_mean = sum(p["mean"] * n for p, n in parts) / total_n
        w_std = sum(p["std"] * n for p, n in parts) / total_n
        g_min = min(p["min"] for p, _ in parts)
        g_max = max(p["max"] for p, _ in parts)
        w_median = sum(p["median"] * n for p, n in parts) / total_n
        g_total = sum(p["total"] for p, _ in parts)
        merged[key] = {
            "mean": round(w_mean, 1), "std": round(w_std, 1),
            "min": g_min, "max": g_max,
            "median": round(w_median, 1), "total": g_total,
        }

    merged["num_samples"] = total_n
    merged["tokenizer"] = all_ts[0].get("tokenizer", "")
    return merged


def collect_token_stats(results_dir: str, experiment_names: list[str]) -> dict:
    """Read token_stats from each experiment's meta.json (merges across splits)."""
    stats = {}
    for name in experiment_names:
        meta_files = _find_meta_files(results_dir, name)
        if not meta_files:
            continue
        ts_list = []
        for mf in meta_files:
            with open(mf, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "token_stats" in meta:
                ts_list.append(meta["token_stats"])
        if ts_list:
            stats[name] = _merge_token_stats(ts_list)
    return stats


def collect_latency_stats(results_dir: str, experiment_names: list[str]) -> dict:
    """Read latency info from each experiment's meta.json (sums across splits)."""
    stats = {}
    for name in experiment_names:
        meta_files = _find_meta_files(results_dir, name)
        if not meta_files:
            continue
        total_sec = 0.0
        total_predicted = 0
        for mf in meta_files:
            with open(mf, "r", encoding="utf-8") as f:
                meta = json.load(f)
            lat = meta.get("latency")
            if lat:
                total_sec += lat.get("total_seconds", 0)
                total_predicted += lat.get("num_predicted", 0)
        if total_predicted > 0:
            stats[name] = {
                "total_seconds": round(total_sec, 2),
                "num_predicted": total_predicted,
                "mean_ms_per_sample": round(total_sec / total_predicted * 1000, 1),
                "samples_per_second": round(total_predicted / total_sec, 2)
                                      if total_sec > 0 else 0,
            }
    return stats


def print_token_stats(token_stats: dict, experiment_names: list[str]):
    """Print token statistics table to console."""
    if not token_stats:
        return
    W = 96
    print(f"\n{'─' * W}")
    print(f"  Dataset Token Statistics (exact, via tokenizer)")
    print(f"{'─' * W}")
    print(f"  {'Method':<20} {'Profile':>8} {'Context':>8} "
          f"{'GT Output':>10} {'Pred':>8} {'Prompt':>8} {'N':>6}")
    print(f"{'─' * W}")
    for name in experiment_names:
        ts = token_stats.get(name)
        if not ts:
            continue
        label = METHOD_LABELS.get(name, name)[:20]
        p = ts.get("profile_tokens", {}).get("mean", 0)
        c = ts.get("context_tokens", {}).get("mean", 0)
        o = ts.get("output_tokens", {}).get("mean", 0)
        pr_tok = ts.get("prompt_tokens") or ts.get("cond_prompt_tokens") or {}
        pr = pr_tok.get("mean", 0)
        pred = ts.get("prediction_tokens", {}).get("mean", 0)
        n = ts.get("num_samples", 0)
        pred_str = f"{pred:.1f}" if pred else "—"
        print(f"  {label:<20} {p:>8.1f} {c:>8.1f} "
              f"{o:>10.1f} {pred_str:>8} {pr:>8.1f} {n:>6}")
    print(f"{'─' * W}")
    print(f"  (All values are mean token counts per sample; "
          f"GT Output = ground truth, Pred = actual model output)")


def generate_token_stats_markdown(token_stats: dict, experiment_names: list[str]) -> str:
    """Generate markdown table for token statistics."""
    if not token_stats:
        return ""
    lines = []
    lines.append("## Token Statistics\n")
    lines.append("| Method | Profile (mean±std) | Context (mean±std) | "
                 "GT Output (mean±std) | Prediction (mean±std) | "
                 "Prompt (mean±std) | N |")
    lines.append("|--------|---:|---:|---:|---:|---:|--:|")
    for name in experiment_names:
        ts = token_stats.get(name)
        if not ts:
            continue
        label = METHOD_LABELS.get(name, name)
        p = ts.get("profile_tokens")
        c = ts.get("context_tokens")
        o = ts.get("output_tokens")
        pr = ts.get("prompt_tokens") or ts.get("cond_prompt_tokens")
        pred = ts.get("prediction_tokens")

        def _fmt_tok(tok):
            return f"{tok['mean']:.1f}±{tok['std']:.1f}" if tok else "—"

        lines.append(
            f"| {label} "
            f"| {_fmt_tok(p)} "
            f"| {_fmt_tok(c)} "
            f"| {_fmt_tok(o)} "
            f"| {_fmt_tok(pred)} "
            f"| {_fmt_tok(pr)} "
            f"| {ts.get('num_samples', 0)} |"
        )
    lines.append("")
    return "\n".join(lines)


def generate_token_stats_latex(token_stats: dict, experiment_names: list[str]) -> str:
    """Generate LaTeX table for token statistics."""
    if not token_stats:
        return ""
    lines = []
    lines.append("")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Token statistics per method. "
                 r"All values are mean token counts per sample "
                 r"(computed with the inference tokenizer). "
                 r"GT = ground-truth output; Pred = actual model prediction.}")
    lines.append(r"\label{tab:token_stats}")
    lines.append(r"\begin{tabular}{l r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{Profile} & \textbf{Context} "
                 r"& \textbf{GT Output} & \textbf{Pred} & \textbf{Prompt} \\")
    lines.append(r"\midrule")
    for name in experiment_names:
        ts = token_stats.get(name)
        if not ts:
            continue
        label = METHOD_LABELS.get(name, name).replace("_", r"\_")
        p = ts.get("profile_tokens", {}).get("mean", 0)
        c = ts.get("context_tokens", {}).get("mean", 0)
        o = ts.get("output_tokens", {}).get("mean", 0)
        pr_tok = ts.get("prompt_tokens") or ts.get("cond_prompt_tokens") or {}
        pr = pr_tok.get("mean", 0)
        pred = ts.get("prediction_tokens", {}).get("mean")
        pred_str = f"{pred:.1f}" if pred else "---"
        lines.append(f"{label} & {p:.1f} & {c:.1f} & {o:.1f} "
                     f"& {pred_str} & {pr:.1f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate cross-experiment comparison report",
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Dataset-level results directory (e.g. results/RAIDEN/prompt/main)",
    )
    parser.add_argument(
        "--experiments", nargs="+", required=True,
        help="Experiment sub-directory names to compare "
             "(e.g. m1_context_only m2_raw_profile m3_naive_rewrite m4_static_tree m6_phase_tree)",
    )
    parser.add_argument(
        "--splits", nargs="*", default=None,
        help="Splits to include (e.g. random_test ood_test). "
             "If omitted, auto-detects from directory structure.",
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Baseline experiment name for pairwise comparison "
             "(default: m2_raw_profile)",
    )
    parser.add_argument(
        "--per_character", action="store_true",
        help="Include per-character breakdown in the report",
    )
    args = parser.parse_args()

    baseline = args.baseline or "m2_raw_profile"
    splits = args.splits if args.splits else None

    report = compute_report(args.results_dir, args.experiments, baseline,
                            splits=splits)
    if not report:
        return

    # --- Collect token statistics from meta.json ---
    token_stats = collect_token_stats(args.results_dir, args.experiments)
    if token_stats:
        report["token_stats"] = token_stats

    # --- Save JSON ---
    os.makedirs(args.results_dir, exist_ok=True)
    summary_path = os.path.join(args.results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)

    # --- Save Markdown ---
    md_content = generate_markdown(report, args.experiments,
                                   per_character=args.per_character)
    if token_stats:
        md_content += "\n" + generate_token_stats_markdown(token_stats,
                                                           args.experiments)
    md_path = os.path.join(args.results_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # --- Save LaTeX ---
    tex_content = generate_latex(report, args.experiments)
    if token_stats:
        tex_content += "\n" + generate_token_stats_latex(token_stats,
                                                         args.experiments)
    tex_path = os.path.join(args.results_dir, "tables.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    # --- Console output ---
    print_report(report, args.experiments, per_character=args.per_character)
    if token_stats:
        print_token_stats(token_stats, args.experiments)

    # --- Latency summary ---
    latency_stats = collect_latency_stats(args.results_dir, args.experiments)
    if latency_stats:
        report["latency"] = latency_stats
        W = 82
        print(f"\n{'─' * W}")
        print(f"  Inference Latency")
        print(f"{'─' * W}")
        print(f"  {'Method':<20} {'Total (s)':>10} {'ms/sample':>10} "
              f"{'samples/s':>10} {'N':>6}")
        print(f"{'─' * W}")
        for name in args.experiments:
            lat = latency_stats.get(name)
            if not lat:
                continue
            label = METHOD_LABELS.get(name, name)[:20]
            print(f"  {label:<20} {lat['total_seconds']:>10.1f} "
                  f"{lat['mean_ms_per_sample']:>10.1f} "
                  f"{lat['samples_per_second']:>10.2f} "
                  f"{lat['num_predicted']:>6}")
        print(f"{'─' * W}")

        # Re-save summary.json with latency
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)

    print(f"\n  Outputs saved:")
    print(f"    • {summary_path}")
    print(f"    • {md_path}")
    print(f"    • {tex_path}")


if __name__ == "__main__":
    main()
