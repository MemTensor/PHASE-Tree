"""Generate publication-quality figures from evaluation results.

Reads ``summary.json`` produced by ``report.py`` and generates:

  * ``bar_chart.pdf``  — grouped bar chart comparing methods.
  * ``radar_plot.pdf`` — radar/spider plot for multi-dimensional comparison.
  * ``per_character_heatmap.pdf`` — heatmap of per-character scores (optional).
  * ``delta_plot.pdf`` — improvement over baseline.

All text in figures is in English and rendered in Times New Roman
(STIX font, metrically identical to Times New Roman, is used as the
rendering backend when TNR is not installed on the system).

Usage::

    # Generate all plots from a single dataset
    python evaluation/visualize.py \\
        --results_dir results/RAIDEN/prompt/main

    # Specify output format (default: pdf)
    python evaluation/visualize.py \\
        --results_dir results/RAIDEN/prompt/main --format png --dpi 300

    # Compare across multiple datasets
    python evaluation/visualize.py \\
        --results_dir results/RAIDEN/prompt/main results/CharacterEval/prompt/main \\
        --output_dir figures/
"""

import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Suppress CJK glyph warnings (character names may be Chinese)
warnings.filterwarnings("ignore", message=".*Glyph.*missing.*")

# ---------------------------------------------------------------------------
# Font configuration — strict Times New Roman
# ---------------------------------------------------------------------------
# Priority: Times New Roman > STIX (metrically identical to TNR) > DejaVu Serif
# STIX was specifically designed by STI Pub as a Times-compatible font for
# scientific publishing. It is indistinguishable from TNR in print.

_FONT_PRIORITY = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]

try:
    import matplotlib.font_manager as _fm
    _avail = {f.name for f in _fm.fontManager.ttflist}
    _SERIF_FONTS = [f for f in _FONT_PRIORITY if f in _avail]
    if not _SERIF_FONTS:
        _SERIF_FONTS = ["DejaVu Serif"]
except Exception:
    _SERIF_FONTS = _FONT_PRIORITY

STYLE_CONFIG = {
    "font.family": "serif",
    "font.serif": _SERIF_FONTS,
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
}

METHOD_LABELS = {
    "m1_context_only": "Context-Only",
    "m2_raw_profile": "Raw-Profile",
    "m3_naive_rewrite": "Naive-Rewrite",
    "m4_static_tree": "Static-Tree",
    "m5_dynamic_tree": "Dynamic-Tree",
    "m6_phase_tree": "PHASE-Tree",
}

# Colorblind-friendly academic palette (Tol muted scheme)
METHOD_COLORS = {
    "m1_context_only": "#88CCEE",
    "m2_raw_profile": "#44AA99",
    "m3_naive_rewrite": "#DDCC77",
    "m4_static_tree": "#CC6677",
    "m5_dynamic_tree": "#117733",
    "m6_phase_tree": "#882255",
}


def _label(name: str) -> str:
    return METHOD_LABELS.get(name, name)


def _color(name: str) -> str:
    return METHOD_COLORS.get(name, "#333333")


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def generate_bar_chart(summary: dict, methods: list[str], output_path: str):
    """Grouped bar chart: separate subplots for 1-5 scale metrics and 0-1 embedding."""
    score_metrics = ["character", "semantic"]
    score_labels = ["Character", "Semantic"]

    n_methods = len(methods)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5),
                                    gridspec_kw={"width_ratios": [2, 1]})

    # Left panel: Character & Semantic (1-5 scale)
    x = np.arange(len(score_metrics))
    width = 0.8 / n_methods
    for i, method in enumerate(methods):
        s = summary.get(method)
        if not s:
            continue
        values = [s[m]["mean"] for m in score_metrics]
        ci_errs = [s[m]["ci_hi"] - s[m]["mean"] for m in score_metrics]
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax1.bar(x + offset, values, width * 0.88,
                       yerr=ci_errs, capsize=3, error_kw={"linewidth": 0.8},
                       label=_label(method), color=_color(method),
                       edgecolor="white", linewidth=0.5)
        for bar_idx, (bar, val) in enumerate(zip(bars, values)):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + ci_errs[bar_idx] + 0.05,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(score_labels)
    ax1.set_ylabel("Score (1\u20135)")
    ax1.set_ylim(0, 5.5)
    ax1.legend(loc="upper left")

    # Right panel: Embedding similarity (0-1 scale)
    x2 = np.arange(1)
    for i, method in enumerate(methods):
        s = summary.get(method)
        if not s:
            continue
        val = s["embedding"]["mean"]
        ci_err = s["embedding"]["ci_hi"] - val
        offset = (i - n_methods / 2 + 0.5) * width
        ax2.bar(x2 + offset, [val], width * 0.88,
                yerr=[ci_err], capsize=3, error_kw={"linewidth": 0.8},
                color=_color(method), edgecolor="white", linewidth=0.5)
        ax2.text(x2[0] + offset, val + ci_err + 0.008,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(["Embedding"])
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_ylim(0, 1.0)

    fig.suptitle("Method Comparison", fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  \u2713 Bar chart: {output_path}")


# ---------------------------------------------------------------------------
# Radar / spider plot
# ---------------------------------------------------------------------------

def generate_radar_plot(summary: dict, methods: list[str], output_path: str):
    """Radar plot comparing methods across all metrics (normalized to 0-1)."""
    metrics = ["character", "semantic", "embedding"]
    metric_labels = ["Character\n(1\u20135)", "Semantic\n(1\u20135)", "Embedding\n(0\u20131)"]
    max_vals = {"character": 5.0, "semantic": 5.0, "embedding": 1.0}
    n = len(metrics)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for method in methods:
        s = summary.get(method)
        if not s:
            continue
        values = [s[m]["mean"] / max_vals[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=_label(method),
                color=_color(method), markersize=5)
        ax.fill(angles, values, alpha=0.08, color=_color(method))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    ax.set_title("Multi-dimensional Comparison (normalized)",
                 fontweight="bold", y=1.08)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  \u2713 Radar plot: {output_path}")


# ---------------------------------------------------------------------------
# Per-character heatmap
# ---------------------------------------------------------------------------

def generate_heatmap(per_character: dict, methods: list[str], output_path: str,
                     metric: str = "character_mean"):
    """Heatmap: characters x methods for a given metric."""
    all_roles = sorted(set(
        role for method_data in per_character.values()
        for role in method_data
    ))
    if not all_roles:
        print("  \u26a0 No per-character data, skipping heatmap.")
        return

    matrix = []
    for role in all_roles:
        row = []
        for method in methods:
            val = per_character.get(method, {}).get(role, {}).get(metric, 0)
            row.append(val)
        matrix.append(row)

    matrix = np.array(matrix)
    method_labels = [_label(m) for m in methods]

    fig, ax = plt.subplots(
        figsize=(max(6, len(methods) * 1.5), max(4, len(all_roles) * 0.5)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=1, vmax=5)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(method_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(all_roles)))
    ax.set_yticklabels(all_roles)

    for i in range(len(all_roles)):
        for j in range(len(methods)):
            val = matrix[i, j]
            color = "white" if val > 3.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Score")
    metric_name = metric.replace("_mean", "").capitalize()
    ax.set_title(f"Per-Character Scores ({metric_name})")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  \u2713 Heatmap: {output_path}")


# ---------------------------------------------------------------------------
# Delta / improvement plot
# ---------------------------------------------------------------------------

def generate_delta_plot(comparisons: dict, methods: list[str], output_path: str):
    """Horizontal bar chart of improvements over baseline."""
    plot_methods = [m for m in methods if m in comparisons]
    if not plot_methods:
        print("  \u26a0 No comparison data, skipping delta plot.")
        return

    metrics = ["character", "semantic", "embedding"]
    metric_labels = ["Character", "Semantic", "Embedding"]

    fig, axes = plt.subplots(1, 3, figsize=(12, max(3, len(plot_methods) * 0.8)),
                             sharey=True)

    y_pos = np.arange(len(plot_methods))

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        deltas = [comparisons[m][metric]["delta"] for m in plot_methods]
        colors = ["#27ae60" if d > 0 else "#c0392b" for d in deltas]

        ax.barh(y_pos, deltas, color=colors, alpha=0.8, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([_label(m) for m in plot_methods])
        ax.set_xlabel(f"\u0394 {label}")
        ax.set_title(label)

        for i, d in enumerate(deltas):
            ax.text(d + 0.01 * (1 if d >= 0 else -1), i,
                    f"{d:+.3f}", va="center", fontsize=9,
                    ha="left" if d >= 0 else "right")

    fig.suptitle("Improvement over Baseline", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  \u2713 Delta plot: {output_path}")


# ---------------------------------------------------------------------------
# Per-split comparison
# ---------------------------------------------------------------------------

def generate_split_comparison(per_split: dict, methods: list[str],
                              output_path: str):
    """Grouped bar chart: random_test vs ood_test per method for each metric."""
    splits = sorted(per_split.keys())
    if len(splits) < 2:
        print("  \u26a0 Only one split found, skipping split comparison.")
        return

    metrics = ["character", "semantic", "embedding"]
    metric_titles = ["Character Score", "Semantic Score", "Embedding Similarity"]
    y_limits = [(0, 5.5), (0, 5.5), (0, 1.0)]

    n_methods = len(methods)
    n_splits = len(splits)
    split_hatches = ["", "//"]
    split_alphas = [0.95, 0.65]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, metric, title, ylim in zip(axes, metrics, metric_titles, y_limits):
        x = np.arange(n_methods)
        width = 0.8 / n_splits

        for si, split in enumerate(splits):
            split_data = per_split.get(split, {})
            values = []
            ci_errs = []
            for method in methods:
                s = split_data.get(method, {})
                m_data = s.get(metric, {})
                val = m_data.get("mean", 0)
                err = m_data.get("ci_hi", val) - val
                values.append(val)
                ci_errs.append(err)

            offset = (si - n_splits / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width * 0.88,
                          yerr=ci_errs, capsize=2.5,
                          error_kw={"linewidth": 0.7},
                          label=split,
                          color=[_color(m) for m in methods],
                          alpha=split_alphas[si % len(split_alphas)],
                          hatch=split_hatches[si % len(split_hatches)],
                          edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([_label(m) for m in methods], rotation=25, ha="right",
                           fontsize=9)
        ax.set_title(title)
        ax.set_ylim(*ylim)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#888888",
                       alpha=split_alphas[si % len(split_alphas)],
                       hatch=split_hatches[si % len(split_hatches)],
                       edgecolor="white")
        for si in range(n_splits)
    ]
    fig.legend(handles, splits, loc="upper center",
               ncol=n_splits, bbox_to_anchor=(0.5, 1.02), fontsize=10)

    fig.suptitle("Split Comparison (random_test vs ood_test)",
                 fontweight="bold", y=1.07)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2713 Split comparison: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures from evaluation results",
    )
    parser.add_argument(
        "--results_dir", type=str, nargs="+", required=True,
        help="One or more results directories (each with summary.json)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for figures (default: <results_dir>/figures/)",
    )
    parser.add_argument("--format", type=str, default="pdf",
                        choices=["pdf", "png", "svg"],
                        help="Output format for figures")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for raster formats (png)")
    args = parser.parse_args()

    plt.rcParams.update(STYLE_CONFIG)
    plt.rcParams["savefig.dpi"] = args.dpi

    print(f"  Font: {plt.rcParams['font.serif'][0]} "
          f"(family={plt.rcParams['font.family']})")

    for results_dir in args.results_dir:
        summary_path = os.path.join(results_dir, "summary.json")
        if not os.path.exists(summary_path):
            print(f"  \u26a0 {summary_path} not found, skipping.", flush=True)
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        summary = report.get("experiment_summary", {})
        per_character = report.get("per_character", {})
        comparisons = report.get("vs_baseline", {})
        per_split = report.get("per_split", {})
        methods = list(summary.keys())

        out_dir = args.output_dir or os.path.join(results_dir, "figures")
        os.makedirs(out_dir, exist_ok=True)

        normed = os.path.normpath(results_dir)
        base = os.path.basename(normed)
        dataset_name = (os.path.basename(os.path.dirname(normed))
                        if base in ("main", "ablation") else base)
        ext = args.format

        print(f"\n  Generating figures for: {dataset_name}")
        print(f"  Output: {out_dir}/")

        generate_bar_chart(
            summary, methods,
            os.path.join(out_dir, f"bar_chart.{ext}"),
        )
        generate_radar_plot(
            summary, methods,
            os.path.join(out_dir, f"radar_plot.{ext}"),
        )
        if per_character:
            generate_heatmap(
                per_character, methods,
                os.path.join(out_dir, f"heatmap_character.{ext}"),
                metric="character_mean",
            )
        if comparisons:
            generate_delta_plot(
                comparisons, methods,
                os.path.join(out_dir, f"delta_plot.{ext}"),
            )
        if per_split and len(per_split) >= 2:
            generate_split_comparison(
                per_split, methods,
                os.path.join(out_dir, f"split_comparison.{ext}"),
            )

    print(f"\n\u2713 All figures generated.", flush=True)


if __name__ == "__main__":
    main()
