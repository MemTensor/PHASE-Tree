"""Generate the two token-efficiency figures used in the paper.

Figure 1 (main text): cost-quality Pareto scatter (Sem vs. prompt tokens),
two stacked panels (Short / Long), explicit vs implicit color-coded.

Figure 2 (appendix): horizontal stacked bar of token decomposition
(context / profile / template overhead) per method, two side-by-side panels.

Output directory:
    Defaults to ``<repo_root>/evaluation/figures``.  Override with
    ``--out-dir`` or the ``TOKEN_FIG_OUT_DIR`` environment variable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Aggregated numbers (from `summary.json` and prior token-stats aggregation)
# ---------------------------------------------------------------------------

# Avg prompt tokens (per dialogue, mean of all samples) for each method, by group.
PROMPT_TOKENS = {
    "Short": {
        "Base": 204,
        "RAG": 622,
        "PAG": 1045,
        "CFG": 831,
        "Ours_E": 471,
        "MT-LoRA": 204,
        "Steering": 204,
        "OPPU": None,
        "P2P": 204,
        "Ours_I": 204,
    },
    "Long": {
        "Base": 372,
        "RAG": 1347,
        "PAG": 1628,
        "CFG": 1024,
        "Ours_E": 1736,
        "MT-LoRA": 372,
        "Steering": 372,
        "OPPU": 372,
        "P2P": 372,
        "Ours_I": 372,
    },
}

# (profile, context, retrieved, prompt_total). 'retrieved' is the extra
# retrieved-dialogue tokens injected by RAG/PAG on top of context+profile;
# template = prompt_total - profile - context - retrieved (~85 tokens for
# non-CFG methods; doubled for CFG since it runs two forward passes).
TOKEN_DECOMP = {
    "Short": {
        "Base":     (0,   121,   0, 204),
        "RAG":      (0,   121, 418, 622),
        "PAG":      (416, 121, 423, 1045),
        "CFG":      (416, 242,   0, 831),
        "Ours_E":   (260, 121,   0, 471),
        "MT-LoRA":  (0,   121,   0, 204),
        "Steering": (0,   121,   0, 204),
        "P2P":      (0,   121,   0, 204),
        "Ours_I":   (0,   121,   0, 204),
    },
    "Long": {
        "Base":     (0,    290,   0, 372),
        "RAG":      (0,    290, 975, 1347),
        "PAG":      (275,  290, 975, 1628),
        "CFG":      (275,  580,   0, 1024),
        "Ours_E":   (1358, 290,   0, 1736),
        "MT-LoRA":  (0,    290,   0, 372),
        "Steering": (0,    290,   0, 372),
        "OPPU":     (0,    290,   0, 372),
        "P2P":      (0,    290,   0, 372),
        "Ours_I":   (0,    290,   0, 372),
    },
}

# Macro-average performance scores (from Table 1 / tab:external in the paper).
QUALITY = {
    "Char": {
        "Short": {
            "Base": 2.142, "RAG": 2.527, "PAG": 2.989, "CFG": 3.075, "Ours_E": 3.028,
            "MT-LoRA": 2.300, "Steering": 2.151, "OPPU": None, "P2P": 2.321, "Ours_I": 2.319,
        },
        "Long": {
            "Base": 2.326, "RAG": 2.405, "PAG": 2.510, "CFG": 2.389, "Ours_E": 3.003,
            "MT-LoRA": 2.269, "Steering": 2.381, "OPPU": 2.376, "P2P": 2.396, "Ours_I": 2.307,
        },
    },
    "Sem": {
        "Short": {
            "Base": 3.539, "RAG": 3.659, "PAG": 3.588, "CFG": 3.245, "Ours_E": 3.792,
            "MT-LoRA": 3.736, "Steering": 3.554, "OPPU": None, "P2P": 3.706, "Ours_I": 3.748,
        },
        "Long": {
            "Base": 3.323, "RAG": 3.289, "PAG": 2.889, "CFG": 2.429, "Ours_E": 3.697,
            "MT-LoRA": 3.428, "Steering": 2.350, "OPPU": 3.141, "P2P": 3.410, "Ours_I": 3.434,
        },
    },
    "Emb": {
        "Short": {
            "Base": 0.394, "RAG": 0.434, "PAG": 0.427, "CFG": 0.381, "Ours_E": 0.421,
            "MT-LoRA": 0.445, "Steering": 0.394, "OPPU": None, "P2P": 0.427, "Ours_I": 0.445,
        },
        "Long": {
            "Base": 0.268, "RAG": 0.273, "PAG": 0.255, "CFG": 0.225, "Ours_E": 0.314,
            "MT-LoRA": 0.283, "Steering": 0.249, "OPPU": 0.283, "P2P": 0.276, "Ours_I": 0.283,
        },
    },
}

QUALITY_METRICS = ["Char", "Sem", "Emb"]
QUALITY_FMT = {"Char": "{:.3f}", "Sem": "{:.3f}", "Emb": "{:.3f}"}

# Route assignment for color coding.
ROUTE = {
    "Base": "ref",
    "RAG": "explicit", "PAG": "explicit", "CFG": "explicit", "Ours_E": "explicit",
    "MT-LoRA": "implicit", "Steering": "implicit", "OPPU": "implicit",
    "P2P": "implicit", "Ours_I": "implicit",
}

ROUTE_COLOR = {
    "explicit": "#2E7D32",
    "implicit": "#7B1FA2",
    "ref":      "#555555",
}

OURS_METHODS = {"Ours_E", "Ours_I"}

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR = Path(os.environ.get("TOKEN_FIG_OUT_DIR", DEFAULT_OUT_DIR))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Figure 1: cost-quality Pareto scatter (main text, 1-column)
# ---------------------------------------------------------------------------

def _ours_label(name: str) -> str:
    if name in ("Ours_E", "Ours_I"):
        return "Ours"
    return name


PAIRED_ORDER = ["Base", "RAG", "PAG", "CFG", "Ours_E",
                "MT-LoRA", "Steering", "OPPU", "P2P", "Ours_I"]


def make_pareto_figure():
    """Per-horizon, 4-panel comparison: cost (left, inverted) and three
    quality metrics (Char, Sem, Emb) on the right. Two-column-wide figure
    suitable for the main text.
    """
    fig = plt.figure(figsize=(7.0, 4.4))
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1.25, 1.0, 1.0, 1.0],
        hspace=0.40, wspace=0.16,
        left=0.10, right=0.99, top=0.92, bottom=0.13,
    )

    for row, group in enumerate(["Short", "Long"]):
        ax_cost = fig.add_subplot(gs[row, 0])
        ax_q = [fig.add_subplot(gs[row, 1 + i]) for i in range(3)]

        methods = [m for m in PAIRED_ORDER
                   if PROMPT_TOKENS[group].get(m) is not None
                   and QUALITY["Sem"][group].get(m) is not None]
        y = np.arange(len(methods))[::-1]
        ylim = (y.min() - 0.6, y.max() + 0.6)

        costs = [PROMPT_TOKENS[group][m] for m in methods]

        colors, edges, edge_lw = [], [], []
        for m in methods:
            base_color = ROUTE_COLOR[ROUTE[m]]
            colors.append(base_color)
            if m in OURS_METHODS:
                edges.append("black"); edge_lw.append(1.0)
            else:
                edges.append("none"); edge_lw.append(0.0)

        ax_cost.barh(y, costs, height=0.7, color=colors,
                     edgecolor=edges, linewidth=edge_lw)
        max_cost = max(costs)
        for yi, c in zip(y, costs):
            ax_cost.text(c + max_cost * 0.02, yi, f"{c}",
                         va="center", ha="right", fontsize=7, color="#333")
        ax_cost.invert_xaxis()
        ax_cost.set_xlim(max_cost * 1.32, 0)
        ax_cost.set_xlabel("Prompt tokens $\\downarrow$", fontsize=8)
        ax_cost.tick_params(axis="x", labelsize=7)
        ax_cost.set_yticks(y)
        labels = [_ours_label(m) for m in methods]
        ax_cost.set_yticklabels(labels, fontsize=8)
        ax_cost.tick_params(axis="y", which="both", length=0, pad=2)
        for tick, m in zip(ax_cost.get_yticklabels(), methods):
            tick.set_color(ROUTE_COLOR[ROUTE[m]])
        ax_cost.set_ylim(ylim)
        ax_cost.grid(True, axis="x", linewidth=0.4, alpha=0.4)
        ax_cost.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax_cost.spines[spine].set_visible(False)

        for ax, metric in zip(ax_q, QUALITY_METRICS):
            vals = [QUALITY[metric][group][m] for m in methods]
            ax.barh(y, vals, height=0.7, color=colors,
                    edgecolor=edges, linewidth=edge_lw)
            vmin, vmax = min(vals), max(vals)
            span = vmax - vmin if vmax > vmin else max(vmax, 1) * 0.1
            x0 = vmin - span * 0.10
            x1 = vmax + span * 0.32
            for yi, v in zip(y, vals):
                ax.text(v + span * 0.04, yi, QUALITY_FMT[metric].format(v),
                        va="center", ha="left", fontsize=7, color="#333")
            ax.set_xlim(x0, x1)
            ax.set_xlabel(f"{metric} score $\\uparrow$", fontsize=8)
            ax.tick_params(axis="x", labelsize=7)
            ax.set_yticks(y)
            ax.set_yticklabels([])
            ax.tick_params(axis="y", which="both", length=0, labelleft=False)
            ax.set_ylim(ylim)
            ax.grid(True, axis="x", linewidth=0.4, alpha=0.4)
            ax.set_axisbelow(True)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)

        n_explicit = sum(1 for m in methods if ROUTE[m] in ("explicit", "ref"))
        if n_explicit < len(methods):
            divider_y = len(methods) - n_explicit - 0.5
            for a in (ax_cost, *ax_q):
                a.axhline(divider_y, color="#999", lw=0.6, ls=":")

        title = "Short-Dialogue" if group == "Short" else "Long-Dialogue"
        x_center = 0.5 * (ax_cost.get_position().x0 + ax_q[-1].get_position().x1)
        y_top = ax_cost.get_position().y1
        fig.text(x_center, y_top + 0.01, title,
                 fontsize=10, fontweight="bold", ha="center", va="bottom")

    legend_handles = [
        Patch(facecolor=ROUTE_COLOR["explicit"], edgecolor="none", label="Explicit"),
        Patch(facecolor=ROUTE_COLOR["implicit"], edgecolor="none", label="Implicit"),
        Patch(facecolor="#999", edgecolor="none", label="Base (no profile)"),
        Patch(facecolor="#ccc", edgecolor="black", lw=1.0, label="PHASE-Tree (Ours)"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=4,
        frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=8,
        columnspacing=1.6, handletextpad=0.4,
    )

    out = OUT_DIR / "token_pareto.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2: stacked horizontal bar of token decomposition (appendix)
# ---------------------------------------------------------------------------

# Display order: explicit cluster first, then implicit cluster.
EXPLICIT_ORDER = ["Base", "RAG", "PAG", "CFG", "Ours_E"]
IMPLICIT_ORDER = ["MT-LoRA", "Steering", "OPPU", "P2P", "Ours_I"]


def make_decomp_figure():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.6), sharex=False)

    for ax, group in zip(axes, ["Short", "Long"]):
        order = [m for m in (EXPLICIT_ORDER + IMPLICIT_ORDER) if m in TOKEN_DECOMP[group]]
        labels, profile, context, retrieved, overhead = [], [], [], [], []
        for m in order:
            p, c, r, total = TOKEN_DECOMP[group][m]
            t = max(0, total - p - c - r)
            labels.append(_ours_label(m))
            profile.append(p); context.append(c); retrieved.append(r); overhead.append(t)

        y = np.arange(len(order))
        h = 0.7

        ctx_color = "#90CAF9"
        prof_color = "#A5D6A7"
        retr_color = "#FFCC80"
        ovh_color = "#E0E0E0"

        context = np.array(context); profile = np.array(profile)
        retrieved = np.array(retrieved); overhead = np.array(overhead)

        ax.barh(y, context, h, color=ctx_color, edgecolor="white", linewidth=0.4,
                label="Context")
        ax.barh(y, profile, h, left=context, color=prof_color,
                edgecolor="white", linewidth=0.4, label="Profile")
        ax.barh(y, retrieved, h, left=context + profile, color=retr_color,
                edgecolor="white", linewidth=0.4, label="Retrieved")
        ax.barh(y, overhead, h, left=context + profile + retrieved,
                color=ovh_color, edgecolor="white", linewidth=0.4,
                label="Template / instruction")

        for i, m in enumerate(order):
            total = TOKEN_DECOMP[group][m][3]
            ax.text(total + max(context) * 0.04, i, f"{total}",
                    va="center", ha="left", fontsize=7.5, color="#333")

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        for tick, m in zip(ax.get_yticklabels(), order):
            tick.set_color(ROUTE_COLOR[ROUTE[m]])
        ax.invert_yaxis()

        ax.axhline(len(EXPLICIT_ORDER) - 0.5, color="#999", lw=0.6, ls=":")

        ax.set_xlabel("Tokens per prompt")
        title = "Short-Dialogue" if group == "Short" else "Long-Dialogue"
        ax.set_title(title, loc="center", fontsize=10, fontweight="bold")

        ax.grid(True, axis="x", linewidth=0.4, alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(TOKEN_DECOMP[group][m][3] for m in order) * 1.18)

    legend_handles = [
        Patch(facecolor="#90CAF9", edgecolor="white", label="Dialogue context"),
        Patch(facecolor="#A5D6A7", edgecolor="white", label="Profile (in prompt)"),
        Patch(facecolor="#FFCC80", edgecolor="white", label="Retrieved dialogue"),
        Patch(facecolor="#E0E0E0", edgecolor="white", label="Template / instruction"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=4,
        frameon=False, bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = OUT_DIR / "token_decomposition.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory to write the generated figures into "
             "(default: %(default)s; also overridable via $TOKEN_FIG_OUT_DIR).",
    )
    args = parser.parse_args()

    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating token figures into {OUT_DIR} ...")
    make_pareto_figure()
    make_decomp_figure()
    print("Done.")
