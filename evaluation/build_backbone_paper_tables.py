#!/usr/bin/env python3
"""Build paper-style result tables for backbone tags (GPT-4.1 judge, 25% subsample).

Outputs markdown tables matching the paper layout:
  Table A – Explicit comparison (Base / RAG / PAG / CFG / Ours)
  Table B – Prompt ablation (Base / RP / NR / ST / DT / PT)

Usage:
    python evaluation/build_backbone_paper_tables.py
    python evaluation/build_backbone_paper_tables.py --out results/backbone_tables
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from statistics import mean

ROOT = Path(__file__).resolve().parent.parent

BACKBONES = {
    "main": "Qwen2.5-7B",
    "qwen3_32b": "Qwen3-32B",
    "qwen3_0_6b": "Qwen3-0.6B",
    "gemma_4_e4b_it": "Gemma-4-E4B",
}

# Other backbone tags were judged on a fixed 25% subsample; `main` (7B) stores
# full-sample judge files, so we filter to the same IDs when building tables.
SUBSAMPLE_REF_TAG = "qwen3_32b"

DATASETS = [
    "RAIDEN", "CharacterEval", "SimsConv", "ChatHaruhi",
    "Friends", "TheOffice", "HPD", "StarTrek_TNG",
]

DATASET_LABELS = {
    "RAIDEN": "RAIDEN",
    "CharacterEval": "CharacterEval",
    "SimsConv": "SimsConv",
    "ChatHaruhi": "ChatHaruhi",
    "Friends": "Friends",
    "TheOffice": "The Office",
    "HPD": "Harry Potter",
    "StarTrek_TNG": "Star Trek",
}

SHORT_DS = {"RAIDEN", "CharacterEval", "SimsConv", "ChatHaruhi"}
LONG_DS = {"Friends", "TheOffice", "HPD", "StarTrek_TNG"}

PROMPT_METHODS = {
    "m1_context_only": "Base",
    "m2_raw_profile": "RP",
    "m3_naive_rewrite": "NR",
    "m4_static_tree": "ST",
    "m5_dynamic_tree": "DT",
    "m6_phase_tree": "PT",
}

COMP_METHODS = {
    "rag": "RAG",
    "pag": "PAG",
    "cfg": "CFG",
}

SPLITS = ["random_test", "ood_test"]


def has_m5(ds: str) -> bool:
    return (ROOT / f"phase_tree_data/processed/{ds}/m5_dynamic_tree").is_dir()


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def subsample_ids(ds: str, split: str) -> set[str] | None:
    """Return the fixed 25% question_id set used by other backbone tags."""
    ref = (
        ROOT / "results" / ds / "prompt" / SUBSAMPLE_REF_TAG
        / "m1_context_only" / split / "judge_scores.jsonl"
    )
    if not ref.is_file():
        return None
    return {r["question_id"] for r in load_jsonl(ref) if "question_id" in r}


def aggregate_split_dir(
    split_dir: Path, keep_ids: set[str] | None = None
) -> dict[str, list[float]] | None:
    judge_path = split_dir / "judge_scores.jsonl"
    if not judge_path.is_file():
        return None
    rows = load_jsonl(judge_path)
    if keep_ids is not None:
        rows = [r for r in rows if r.get("question_id") in keep_ids]
    if not rows:
        return None
    emb_path = split_dir / "embedding_scores.jsonl"
    emb_map = {r["question_id"]: r.get("embedding_similarity") for r in load_jsonl(emb_path)}
    char = [float(r["character_score"]) for r in rows]
    sem = [float(r["semantic_score"]) for r in rows]
    emb = [float(emb_map[r["question_id"]]) for r in rows if r["question_id"] in emb_map]
    return {"char": char, "sem": sem, "emb": emb, "n": len(rows)}


def aggregate_method(track: str, tag: str, ds: str, method: str) -> dict[str, float | None]:
    vals: dict[str, list[float]] = defaultdict(list)
    total_n = 0
    for split in SPLITS:
        split_dir = ROOT / "results" / ds / track / tag / method / split
        keep = subsample_ids(ds, split) if tag == "main" else None
        part = aggregate_split_dir(split_dir, keep_ids=keep)
        if part is None:
            continue
        total_n += part["n"]
        for k in ("char", "sem", "emb"):
            vals[k].extend(part[k])
    if total_n == 0:
        return {"char": None, "sem": None, "emb": None, "n": 0}
    out: dict[str, float | None] = {"n": total_n}
    for k in ("char", "sem", "emb"):
        out[k] = float(mean(vals[k])) if vals[k] else None
    return out


def fmt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def best_second(values: dict[str, float | None]) -> tuple[set[str], set[str]]:
    ranked = sorted(
        ((k, v) for k, v in values.items() if v is not None),
        key=lambda x: x[1],
        reverse=True,
    )
    best: set[str] = set()
    second: set[str] = set()
    if ranked:
        best.add(ranked[0][0])
        for k, v in ranked[1:]:
            if v == ranked[0][1]:
                best.add(k)
            elif not second and v < ranked[0][1]:
                second.add(k)
            elif second and v == next(iter(second), None):
                second.add(k)
            else:
                break
        if len(best) < len(ranked):
            top_val = ranked[0][1]
            for k, v in ranked:
                if k not in best and v == next(
                    (x[1] for x in ranked if x[1] < top_val), None
                ):
                    second.add(k)
    return best, second


def cell(label: str, v: float | None, best: set[str], second: set[str], digits: int = 3) -> str:
    s = fmt(v, digits)
    if v is None:
        return s
    if label in best:
        return f"**{s}**"
    if label in second:
        return f"_{s}_"
    return s


def macro_avg(rows: dict[str, dict[str, float | None]], keys: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for m in keys:
        vals = [rows[ds][m]["char"] for ds in rows if rows[ds][m]["char"] is not None]
        out[f"{m}_char"] = float(np.mean(vals)) if vals else None
        vals = [rows[ds][m]["sem"] for ds in rows if rows[ds][m]["sem"] is not None]
        out[f"{m}_sem"] = float(np.mean(vals)) if vals else None
        vals = [rows[ds][m]["emb"] for ds in rows if rows[ds][m]["emb"] is not None]
        out[f"{m}_emb"] = float(np.mean(vals)) if vals else None
    return out


def build_comparison_table(tag: str, display_name: str) -> str:
    """Table A: Base + RAG/PAG/CFG + Ours (m6 prompt)."""
    methods = ["Base", "RAG", "PAG", "CFG", "Ours"]
    rows: dict[str, dict[str, dict[str, float | None]]] = {}

    for ds in DATASETS:
        rows[ds] = {}
        rows[ds]["Base"] = aggregate_method("prompt", tag, ds, "m1_context_only")
        rows[ds]["Ours"] = aggregate_method("prompt", tag, ds, "m6_phase_tree")
        for internal, label in COMP_METHODS.items():
            rows[ds][label] = aggregate_method("comparison", tag, ds, internal)

    lines = [
        f"## {display_name} — Explicit Methods (GPT-4.1, 25% subsample)",
        "",
        "| Dataset | Metric | Base | RAG | PAG | CFG | Ours |",
        "|---------|--------|------|-----|-----|-----|------|",
    ]

    for ds in DATASETS:
        label = DATASET_LABELS[ds]
        for metric, key, digits in [("Char ↑", "char", 3), ("Sem ↑", "sem", 3), ("Emb ↑", "emb", 3)]:
            vals = {m: rows[ds][m][key] for m in methods}
            best, second = best_second(vals)
            cells = [cell(m, vals[m], best, second, digits) for m in methods]
            lines.append(f"| {label} | {metric} | " + " | ".join(cells) + " |")

    for group_name, group_ds in [("Short-Dialogue", SHORT_DS), ("Long-Dialogue", LONG_DS)]:
        sub = {ds: rows[ds] for ds in DATASETS if ds in group_ds}
        for metric, key, digits in [("Char ↑", "char", 3), ("Sem ↑", "sem", 3), ("Emb ↑", "emb", 3)]:
            vals = {
                m: float(mean([sub[ds][m][key] for ds in sub if sub[ds][m][key] is not None]))
                if any(sub[ds][m][key] is not None for ds in sub) else None
                for m in methods
            }
            best, second = best_second(vals)
            cells = [cell(m, vals[m], best, second, digits) for m in methods]
            lines.append(f"| **{group_name}** | {metric} | " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def build_ablation_table(tag: str, display_name: str) -> str:
    """Table B: Base / RP / NR / ST / DT / PT."""
    method_order = ["Base", "RP", "NR", "ST", "DT", "PT"]
    internal_order = [
        "m1_context_only", "m2_raw_profile", "m3_naive_rewrite",
        "m4_static_tree", "m5_dynamic_tree", "m6_phase_tree",
    ]
    label_map = dict(zip(internal_order, method_order))

    rows: dict[str, dict[str, dict[str, float | None]]] = {}
    for ds in DATASETS:
        rows[ds] = {}
        for internal, label in label_map.items():
            if internal == "m5_dynamic_tree" and not has_m5(ds):
                rows[ds][label] = {"char": None, "sem": None, "emb": None, "n": 0}
            else:
                rows[ds][label] = aggregate_method("prompt", tag, ds, internal)

    lines = [
        f"## {display_name} — Prompt Ablation (GPT-4.1, 25% subsample)",
        "",
        "| Dataset | Metric | Base | RP | NR | ST | DT | PT |",
        "|---------|--------|------|----|----|----|----|-----|",
    ]

    for ds in DATASETS:
        dlabel = DATASET_LABELS[ds]
        for metric, key, digits in [("Char ↑", "char", 3), ("Sem ↑", "sem", 3), ("Emb ↑", "emb", 3)]:
            vals = {m: rows[ds][m][key] for m in method_order}
            if ds in SHORT_DS and key == "char":
                vals["DT"] = None  # no DT on short datasets in paper
            best, second = best_second({k: v for k, v in vals.items() if v is not None})
            cells = []
            for m in method_order:
                v = vals[m]
                if ds in SHORT_DS and m == "DT":
                    cells.append("—")
                else:
                    cells.append(cell(m, v, best, second, digits))
            lines.append(f"| {dlabel} | {metric} | " + " | ".join(cells) + " |")

    for group_name, group_ds in [("Short-Dialogue", SHORT_DS), ("Long-Dialogue", LONG_DS)]:
        sub = {ds: rows[ds] for ds in DATASETS if ds in group_ds}
        for metric, key, digits in [("Char ↑", "char", 3), ("Sem ↑", "sem", 3), ("Emb ↑", "emb", 3)]:
            vals = {}
            for m in method_order:
                if group_name == "Short-Dialogue" and m == "DT":
                    vals[m] = None
                    continue
                vlist = [sub[ds][m][key] for ds in sub if sub[ds][m][key] is not None]
                vals[m] = float(mean(vlist)) if vlist else None
            best, second = best_second({k: v for k, v in vals.items() if v is not None})
            cells = []
            for m in method_order:
                if vals[m] is None and m == "DT" and group_name == "Short-Dialogue":
                    cells.append("—")
                else:
                    cells.append(cell(m, vals[m], best, second, digits))
            lines.append(f"| **{group_name}** | {metric} | " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/backbone_tables")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    all_md = [
        "# Backbone Result Tables (GPT-4.1 Judge, 25% Fixed Subsample)",
        "",
        "Judge: `gpt-4.1` · Metrics pooled over `random_test` + `ood_test`.",
        "Emb = mean cosine similarity when `embedding_scores.jsonl` exists (otherwise —).",
        "",
    ]

    summary = {}
    for tag, name in BACKBONES.items():
        comp = build_comparison_table(tag, name)
        abl = build_ablation_table(tag, name)
        (out_dir / f"{tag}_comparison.md").write_text(comp + "\n", encoding="utf-8")
        (out_dir / f"{tag}_ablation.md").write_text(abl + "\n", encoding="utf-8")
        all_md.extend([comp, abl, "---", ""])
        summary[tag] = {"display_name": name}

    (out_dir / "all_backbones.md").write_text("\n".join(all_md), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote tables to {out_dir}/")
    for tag in BACKBONES:
        print(f"  {tag}_comparison.md  {tag}_ablation.md")
    print(f"  all_backbones.md")


if __name__ == "__main__":
    main()
