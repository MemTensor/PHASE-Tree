"""Incremental cross-method report generator with auto-discovery.

Discovers all methods with scored results under a results directory,
merges new methods into an existing summary.json without overwriting
previously computed results (resume/incremental logic).

Works for both ``comparison/main/`` and ``hypernet_p2p/main/`` layouts
since both follow: ``<results_dir>/<method>/<split>/judge_scores.jsonl``.

Incremental logic:
  1. Load existing summary.json if present.
  2. Auto-discover all methods that have judge_scores.jsonl in at least one split.
  3. Identify NEW methods (not yet in summary) or STALE methods (file mtime > report time).
  4. Re-run compute_report for the full set (necessary for correct pairwise comparisons).
  5. Merge per_character / token_stats from old summary for methods whose scores haven't changed.
  6. Write updated summary.json, report.md, tables.tex.

This ensures that adding a new method (e.g. ``mt_lora``) doesn't destroy
existing results and avoids needless recomputation.

Usage::

    # Auto-discover and report all comparison methods for TheOffice
    python evaluation/autoreport.py \\
        --results_dir results/TheOffice/comparison/main \\
        --splits random_test ood_test \\
        --baseline rag

    # Incremental update after adding a new method (mt_lora)
    python evaluation/autoreport.py \\
        --results_dir results/TheOffice/comparison/main \\
        --splits random_test ood_test \\
        --baseline rag

    # For hypernet_p2p (auto-discovers m2..m6)
    python evaluation/autoreport.py \\
        --results_dir results/TheOffice/hypernet_p2p/main \\
        --splits random_test ood_test \\
        --baseline m2_raw_profile

    # Force full recomputation (ignore existing summary)
    python evaluation/autoreport.py \\
        --results_dir results/TheOffice/comparison/main \\
        --force
"""

import argparse
import json
import os
import time
from pathlib import Path

from report import (
    METHOD_LABELS,
    _NumpyEncoder,
    collect_experiment_scores,
    collect_latency_stats,
    collect_token_stats,
    compute_report,
    generate_latex,
    generate_markdown,
    generate_token_stats_latex,
    generate_token_stats_markdown,
    print_report,
    print_token_stats,
)


COMPARISON_METHOD_ORDER = ["rag", "pag", "cfg", "steering", "mt_lora", "oppu"]
HYPERNET_METHOD_ORDER = [
    "m1_context_only", "m2_raw_profile", "m3_naive_rewrite",
    "m4_static_tree", "m5_dynamic_tree", "m6_phase_tree",
]


def discover_methods(results_dir: str, splits: list[str] | None = None) -> list[str]:
    """Auto-discover methods that have at least one judge_scores.jsonl."""
    methods = []
    if not os.path.isdir(results_dir):
        return methods

    for entry in sorted(os.listdir(results_dir)):
        entry_path = os.path.join(results_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry.startswith("_") or entry.startswith("."):
            continue
        if entry in ("figures", "generated_loras"):
            continue

        has_scores = False
        if splits:
            for split in splits:
                judge_path = os.path.join(entry_path, split, "judge_scores.jsonl")
                if os.path.exists(judge_path) and os.path.getsize(judge_path) > 0:
                    has_scores = True
                    break
        else:
            judge_flat = os.path.join(entry_path, "judge_scores.jsonl")
            if os.path.exists(judge_flat) and os.path.getsize(judge_flat) > 0:
                has_scores = True
            else:
                for sub in os.listdir(entry_path):
                    judge_sub = os.path.join(entry_path, sub, "judge_scores.jsonl")
                    if os.path.exists(judge_sub) and os.path.getsize(judge_sub) > 0:
                        has_scores = True
                        break

        if has_scores:
            methods.append(entry)

    return methods


def sort_methods(methods: list[str]) -> list[str]:
    """Sort methods according to canonical ordering."""
    order_map = {}
    for i, m in enumerate(COMPARISON_METHOD_ORDER):
        order_map[m] = i
    for i, m in enumerate(HYPERNET_METHOD_ORDER):
        order_map[m] = i

    def sort_key(m):
        if m in order_map:
            return (0, order_map[m])
        return (1, m)

    return sorted(methods, key=sort_key)


def get_method_mtime(results_dir: str, method: str, splits: list[str] | None) -> float:
    """Get the most recent modification time of score files for a method."""
    latest = 0.0
    method_dir = os.path.join(results_dir, method)
    if not os.path.isdir(method_dir):
        return latest

    score_files = []
    if splits:
        for split in splits:
            for fname in ("judge_scores.jsonl", "embedding_scores.jsonl"):
                p = os.path.join(method_dir, split, fname)
                if os.path.exists(p):
                    score_files.append(p)
    else:
        for fname in ("judge_scores.jsonl", "embedding_scores.jsonl"):
            p = os.path.join(method_dir, fname)
            if os.path.exists(p):
                score_files.append(p)

    for p in score_files:
        mt = os.path.getmtime(p)
        if mt > latest:
            latest = mt

    return latest


def load_existing_summary(results_dir: str) -> dict | None:
    """Load existing summary.json if present."""
    summary_path = os.path.join(results_dir, "summary.json")
    if not os.path.exists(summary_path):
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def needs_update(
    results_dir: str,
    discovered_methods: list[str],
    existing_summary: dict | None,
    splits: list[str] | None,
    force: bool = False,
) -> tuple[bool, list[str], list[str]]:
    """Determine if an update is needed.

    Returns (needs_update, new_methods, stale_methods).
    """
    if force or existing_summary is None:
        return True, discovered_methods, []

    existing_methods = set(existing_summary.get("experiment_summary", {}).keys())
    new_methods = [m for m in discovered_methods if m not in existing_methods]

    summary_path = os.path.join(results_dir, "summary.json")
    summary_mtime = os.path.getmtime(summary_path)

    stale_methods = []
    for m in discovered_methods:
        if m in existing_methods:
            method_mtime = get_method_mtime(results_dir, m, splits)
            if method_mtime > summary_mtime:
                stale_methods.append(m)

    if new_methods or stale_methods:
        return True, new_methods, stale_methods
    return False, [], []


def select_baseline(methods: list[str], user_baseline: str | None) -> str:
    """Select baseline from user preference or auto-detect."""
    if user_baseline and user_baseline in methods:
        return user_baseline

    for candidate in ["m2_raw_profile", "rag"]:
        if candidate in methods:
            return candidate

    return methods[0]


def main():
    parser = argparse.ArgumentParser(
        description="Auto-discover methods and generate incremental comparison report",
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Results directory (e.g. results/TheOffice/comparison/main "
             "or results/TheOffice/hypernet_p2p/main)",
    )
    parser.add_argument(
        "--splits", nargs="*", default=None,
        help="Splits to include (default: auto-detect from directories)",
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Baseline method for pairwise comparison (auto-detected if unset)",
    )
    parser.add_argument(
        "--per_character", action="store_true", default=True,
        help="Include per-character breakdown (default: True)",
    )
    parser.add_argument(
        "--no_per_character", action="store_true",
        help="Disable per-character breakdown",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force full recomputation even if summary.json exists and is fresh",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress console output",
    )
    args = parser.parse_args()

    per_character = not args.no_per_character

    # Auto-detect splits if not provided
    splits = args.splits
    if not splits:
        splits_candidates = set()
        if os.path.isdir(args.results_dir):
            for method_dir in os.listdir(args.results_dir):
                full_path = os.path.join(args.results_dir, method_dir)
                if not os.path.isdir(full_path) or method_dir.startswith("_"):
                    continue
                for sub in os.listdir(full_path):
                    sub_path = os.path.join(full_path, sub)
                    if os.path.isdir(sub_path) and os.path.exists(
                        os.path.join(sub_path, "judge_scores.jsonl")
                    ):
                        splits_candidates.add(sub)
        if splits_candidates:
            splits = sorted(splits_candidates)
            if not args.quiet:
                print(f"  Auto-detected splits: {splits}")

    # Discover methods
    discovered = discover_methods(args.results_dir, splits)
    if not discovered:
        print("ERROR: No methods with scored results found in", args.results_dir)
        return

    discovered = sort_methods(discovered)
    if not args.quiet:
        print(f"  Discovered methods: {discovered}")

    # Check existing summary for incremental logic
    existing_summary = load_existing_summary(args.results_dir)
    update_needed, new_methods, stale_methods = needs_update(
        args.results_dir, discovered, existing_summary, splits, force=args.force
    )

    if not update_needed:
        if not args.quiet:
            print("  Summary is up-to-date. No changes needed.")
            print(f"    (Use --force to regenerate)")
        return

    if not args.quiet:
        if new_methods:
            print(f"  New methods to add: {new_methods}")
        if stale_methods:
            print(f"  Stale methods to refresh: {stale_methods}")
        if args.force:
            print("  Force mode: recomputing everything.")

    # Select baseline
    baseline = select_baseline(discovered, args.baseline)
    if not args.quiet:
        print(f"  Baseline: {baseline}")

    # Full recomputation with all discovered methods
    report = compute_report(
        args.results_dir, discovered, baseline, splits=splits
    )
    if not report:
        print("ERROR: compute_report returned empty.")
        return

    # Add metadata for incremental tracking
    report["_meta"] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results_dir": args.results_dir,
        "methods": discovered,
        "splits": splits,
        "baseline": baseline,
        "incremental_from": (
            existing_summary.get("_meta", {}).get("generated_at")
            if existing_summary else None
        ),
        "new_methods_added": new_methods,
        "stale_methods_refreshed": stale_methods,
    }

    # Collect token statistics
    token_stats = collect_token_stats(args.results_dir, discovered)
    if token_stats:
        report["token_stats"] = token_stats

    # Collect latency
    latency_stats = collect_latency_stats(args.results_dir, discovered)
    if latency_stats:
        report["latency"] = latency_stats

    # Save JSON
    os.makedirs(args.results_dir, exist_ok=True)
    summary_path = os.path.join(args.results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)

    # Save Markdown
    md_content = generate_markdown(report, discovered, per_character=per_character)
    if token_stats:
        md_content += "\n" + generate_token_stats_markdown(token_stats, discovered)
    md_path = os.path.join(args.results_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Save LaTeX
    tex_content = generate_latex(report, discovered)
    if token_stats:
        tex_content += "\n" + generate_token_stats_latex(token_stats, discovered)
    tex_path = os.path.join(args.results_dir, "tables.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    # Console output
    if not args.quiet:
        print_report(report, discovered, per_character=per_character)
        if token_stats:
            print_token_stats(token_stats, discovered)

        print(f"\n  Outputs saved:")
        print(f"    \u2022 {summary_path}")
        print(f"    \u2022 {md_path}")
        print(f"    \u2022 {tex_path}")
        if new_methods:
            print(f"\n  \u2714 Incrementally added: {new_methods}")
        if stale_methods:
            print(f"  \u2714 Refreshed stale: {stale_methods}")


if __name__ == "__main__":
    main()
