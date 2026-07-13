#!/usr/bin/env python3
"""Build paper-style markdown tables for main (7B) results.

Tables:
  - Prompt ablation (Base / RP / NR / ST / DT / PT)
  - Comparison explicit (Base / RAG / PAG / CFG / Ours from prompt)
  - Comparison implicit (MT-LoRA / Steering / OPPU / P2P / Ours from phase_tree)

Scores are aggregated from judge JSONL files (random_test + ood_test).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent

DATASETS = [
    "RAIDEN", "CharacterEval", "SimsConv", "ChatHaruhi",
    "Friends", "TheOffice", "HPD", "StarTrek_TNG",
]
SHORT_DS = {"RAIDEN", "CharacterEval", "SimsConv", "ChatHaruhi"}
LONG_DS = {"Friends", "TheOffice", "HPD", "StarTrek_TNG"}
DS_LABEL = {
    "RAIDEN": "RAIDEN",
    "CharacterEval": "CharacterEval",
    "SimsConv": "SimsConv",
    "ChatHaruhi": "ChatHaruhi",
    "Friends": "Friends",
    "TheOffice": "The Office",
    "HPD": "Harry Potter",
    "StarTrek_TNG": "Star Trek",
}

JUDGE_LABELS = {
    "gpt-4.1": "GPT-4.1",
    "glm-5.2": "GLM-5.2",
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
}

PROMPT_METHODS = [
    ("m1_context_only", "Base"),
    ("m2_raw_profile", "RP"),
    ("m3_naive_rewrite", "NR"),
    ("m4_static_tree", "ST"),
    ("m5_dynamic_tree", "DT"),
    ("m6_phase_tree", "PT (Ours)"),
]

COMP_EXPLICIT = [
    ("m1_context_only", "Base", "prompt"),
    ("rag", "RAG", "comparison"),
    ("pag", "PAG", "comparison"),
    ("cfg", "CFG", "comparison"),
    ("m6_phase_tree", "Ours", "prompt"),
]

COMP_IMPLICIT = [
    ("mt_lora", "MT-LoRA", "comparison"),
    ("steering", "Steering", "comparison"),
    ("oppu", "OPPU", "comparison"),
    ("m2_raw_profile", "P2P", "hypernet_p2p"),
    ("m6_phase_tree", "Ours", "phase_tree"),
]

SPLITS = ["random_test", "ood_test"]
METRICS = [("char", "Char"), ("sem", "Sem"), ("emb", "Emb")]


def judge_file(judge: str) -> str:
    return "judge_scores.jsonl" if judge == "gpt-4.1" else f"judge_scores_{judge}.jsonl"


def has_m5(ds: str) -> bool:
    return (ROOT / f"phase_tree_data/processed/{ds}/m5_dynamic_tree").is_dir()


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def aggregate_track_method(ds: str, track: str, method: str, judge: str) -> dict[str, float | None]:
    vals: dict[str, list[float]] = defaultdict(list)
    seen: set[str] = set()
    total = 0
    for split in SPLITS:
        split_dir = ROOT / "results" / ds / track / "main" / method / split
        judge_path = split_dir / judge_file(judge)
        if not judge_path.is_file():
            continue
        emb_map = {
            r["question_id"]: r.get("embedding_similarity")
            for r in load_jsonl(split_dir / "embedding_scores.jsonl")
        }
        split_rows: dict[str, dict] = {}
        for row in load_jsonl(judge_path):
            qid = row.get("question_id")
            if qid:
                split_rows[qid] = row
        for qid, row in split_rows.items():
            if qid in seen:
                continue
            seen.add(qid)
            total += 1
            vals["char"].append(float(row["character_score"]))
            vals["sem"].append(float(row["semantic_score"]))
            if qid in emb_map and emb_map[qid] is not None:
                vals["emb"].append(float(emb_map[qid]))
    if total == 0:
        return {"char": None, "sem": None, "emb": None, "n": 0}
    return {
        "char": float(mean(vals["char"])) if vals["char"] else None,
        "sem": float(mean(vals["sem"])) if vals["sem"] else None,
        "emb": float(mean(vals["emb"])) if vals["emb"] else None,
        "n": total,
    }


def fmt(v: float | None, metric: str) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}"


def mark_cells(vals: list[float | None], metric: str) -> list[str]:
    ranked = sorted(
        ((i, v) for i, v in enumerate(vals) if v is not None),
        key=lambda x: x[1],
        reverse=True,
    )
    best: set[int] = set()
    second: set[int] = set()
    if ranked:
        best.add(ranked[0][0])
        top = ranked[0][1]
        for i, v in ranked[1:]:
            if v == top:
                best.add(i)
            elif not second:
                second.add(i)
            elif v == next(iter(second), None):
                second.add(i)
            else:
                break
    out = []
    for i, v in enumerate(vals):
        s = fmt(v, metric)
        if v is None:
            out.append(s)
        elif i in best:
            out.append(f"**{s}**")
        elif i in second:
            out.append(f"_{s}_")
        else:
            out.append(s)
    return out


def avg_metric(rows: dict[str, dict[str, float | None]], labels: list[str], key: str) -> list[float | None]:
    out = []
    for label in labels:
        vals = [rows[ds][label][key] for ds in rows if rows[ds][label][key] is not None]
        out.append(float(mean(vals)) if vals else None)
    return out


def build_prompt_table(judge: str) -> str:
    labels = [l for _, l in PROMPT_METHODS]
    rows: dict[str, dict[str, dict[str, float | None]]] = {}
    for ds in DATASETS:
        rows[ds] = {}
        for method, label in PROMPT_METHODS:
            if method == "m5_dynamic_tree" and not has_m5(ds):
                rows[ds][label] = {"char": None, "sem": None, "emb": None, "n": 0}
            else:
                rows[ds][label] = aggregate_track_method(ds, "prompt", method, judge)

    lines = [
        f"# Prompt Track — {JUDGE_LABELS[judge]}",
        "",
        "Backbone: Qwen2.5-7B-Instruct (`main`) · pooled `random_test` + `ood_test`",
        "",
        "| Dataset | Metric | " + " | ".join(labels) + " |",
        "|---------|--------|" + "|".join([":---:"] * len(labels)) + "|",
    ]
    for group_name, group_ds in [("Short-Dialogue", SHORT_DS), ("Long-Dialogue", LONG_DS)]:
        for ds in DATASETS:
            if ds not in group_ds:
                continue
            for key, mlabel in METRICS:
                vals = []
                for _, label in PROMPT_METHODS:
                    if label == "DT" and ds in SHORT_DS:
                        vals.append(None)
                    else:
                        vals.append(rows[ds][label][key])
                lines.append(
                    f"| {DS_LABEL[ds]} | {mlabel} ↑ | " + " | ".join(mark_cells(vals, key)) + " |"
                )
        for key, mlabel in METRICS:
            sub = {ds: rows[ds] for ds in DATASETS if ds in group_ds}
            vals = []
            for _, label in PROMPT_METHODS:
                if label == "DT" and group_name == "Short-Dialogue":
                    vals.append(None)
                else:
                    vals.append(float(mean([
                        sub[ds][label][key] for ds in sub
                        if sub[ds][label][key] is not None
                    ])) if any(sub[ds][label][key] is not None for ds in sub) else None)
            lines.append(
                f"| **{group_name}** | {mlabel} ↑ | " + " | ".join(mark_cells(vals, key)) + " |"
            )
    lines.append("")
    return "\n".join(lines)


def build_comparison_table(judge: str, spec: list[tuple[str, str, str]], title: str) -> str:
    labels = [l for _, l, _ in spec]
    rows: dict[str, dict[str, dict[str, float | None]]] = {}
    for ds in DATASETS:
        rows[ds] = {}
        for method, label, track in spec:
            rows[ds][label] = aggregate_track_method(ds, track, method, judge)

    lines = [
        f"# {title} — {JUDGE_LABELS[judge]}",
        "",
        "Backbone: Qwen2.5-7B-Instruct (`main`) · pooled `random_test` + `ood_test`",
        "",
        "| Dataset | Metric | " + " | ".join(labels) + " |",
        "|---------|--------|" + "|".join([":---:"] * len(labels)) + "|",
    ]
    for group_name, group_ds in [("Short-Dialogue", SHORT_DS), ("Long-Dialogue", LONG_DS)]:
        for ds in DATASETS:
            if ds not in group_ds:
                continue
            for key, mlabel in METRICS:
                vals = [rows[ds][label][key] for label in labels]
                lines.append(
                    f"| {DS_LABEL[ds]} | {mlabel} ↑ | " + " | ".join(mark_cells(vals, key)) + " |"
                )
        for key, mlabel in METRICS:
            sub = {ds: rows[ds] for ds in DATASETS if ds in group_ds}
            vals = avg_metric(sub, labels, key)
            lines.append(
                f"| **{group_name}** | {mlabel} ↑ | " + " | ".join(mark_cells(vals, key)) + " |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/paper_tables")
    parser.add_argument(
        "--judges",
        nargs="+",
        default=["gpt-4.1", "glm-5.2", "deepseek-v4-flash"],
    )
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    all_md = [
        "# Main Result Tables (Qwen2.5-7B-Instruct, Full Sample)",
        "",
        "Backbone tag: `main` · Metrics pooled over `random_test` + `ood_test`.",
        "Emb = mean cosine similarity from `embedding_scores.jsonl`.",
        "",
    ]
    manifest: dict[str, object] = {"backbone": "main", "judges": {}}

    for judge in args.judges:
        slug = judge.replace("/", "_")
        prompt = build_prompt_table(judge)
        explicit = build_comparison_table(
            judge, COMP_EXPLICIT, "Comparison — Explicit Textual Provision"
        )
        implicit = build_comparison_table(
            judge, COMP_IMPLICIT, "Comparison — Implicit Parametric Adaptation"
        )

        (out_dir / f"{slug}_prompt.md").write_text(prompt, encoding="utf-8")
        (out_dir / f"{slug}_comparison_explicit.md").write_text(explicit, encoding="utf-8")
        (out_dir / f"{slug}_comparison_implicit.md").write_text(implicit, encoding="utf-8")

        label = JUDGE_LABELS[judge]
        all_md.extend([
            prompt.replace(f"# Prompt Track — {label}", f"## {label} — Prompt Ablation"),
            explicit.replace(
                f"# Comparison — Explicit Textual Provision — {label}",
                f"## {label} — Explicit Textual Provision",
            ),
            implicit.replace(
                f"# Comparison — Implicit Parametric Adaptation — {label}",
                f"## {label} — Implicit Parametric Adaptation",
            ),
            "---",
            "",
        ])
        manifest["judges"][judge] = {
            "display_name": label,
            "files": [
                f"{slug}_prompt.md",
                f"{slug}_comparison_explicit.md",
                f"{slug}_comparison_implicit.md",
            ],
        }
        print(f"Wrote {slug}_*.md")

    (out_dir / "all_judges.md").write_text("\n".join(all_md), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote all_judges.md and manifest.json")
    print(f"Tables saved to {out_dir}/")


if __name__ == "__main__":
    main()
