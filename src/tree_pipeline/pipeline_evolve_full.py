"""End-to-end persona-evolution pipeline (orchestrator).

Runs the LLM-based ``evolve_persona`` step plus all relationship/persona
post-processing patches in a single command.  Each step is a thin shell-out
to the existing standalone module via ``python -m tree_pipeline.<step>`` so
this orchestrator stays small and the underlying scripts remain
independently runnable.

Pipeline order (each step depends on the previous one's output of
``persona_snapshots.json``):

    1. evolve_persona                  ─ LLM episode-by-episode update
    2. decay_stale_romantic            ─ demote stale "current" romances
    3. repair_inter_main_reciprocity   ─ patch reciprocity gaps / hallucinations
    4. normalize_legacy_relationships  ─ fix bare-name / plural-`are` entries
    5. align_inverse_pair              ─ Pass 1 evidence demote + Pass 2 partner-tier align
    6. forward_fill_continuity         ─ fill 1–N episode regression gaps
    7. audit_core_traits  (optional)   ─ periodic LLM audit of core fields

Each step writes its own backup/log next to ``persona_snapshots.json`` so
the run is reversible.

Usage::

    # Full run (default — invokes evolve, then all patches)
    python -m tree_pipeline.pipeline_evolve_full --dataset Friends

    # Dry-run: every patch step prints expected changes without writing
    python -m tree_pipeline.pipeline_evolve_full --dataset Friends --dry_run

    # Skip the (slow / expensive) LLM evolve step and only run the patches
    python -m tree_pipeline.pipeline_evolve_full --dataset Friends --skip_evolve

    # Skip a specific patch by name
    python -m tree_pipeline.pipeline_evolve_full --dataset Friends \
        --skip audit_core_traits

    # Forward args to the evolve step
    python -m tree_pipeline.pipeline_evolve_full --dataset Friends \
        --workers 8 --test_episode S05E10

    # Run only patches (skip evolve) with audit on
    python -m tree_pipeline.pipeline_evolve_full --dataset Friends \
        --skip_evolve --include_audit
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
PROCESSED_DIR = PROJECT_ROOT / "phase_tree_data" / "processed"


@dataclass
class Step:
    """One sub-step in the pipeline."""

    name: str
    module: str
    description: str
    extra_args: list[str] = field(default_factory=list)
    supports_dry_run: bool = True
    enabled_by_default: bool = True
    long_running: bool = False


# Canonical step definitions.  Order matters.
STEPS: list[Step] = [
    Step(
        name="evolve_persona",
        module="tree_pipeline.evolve_persona",
        description="LLM episode-by-episode persona update",
        supports_dry_run=False,
        enabled_by_default=True,
        long_running=True,
    ),
    Step(
        name="decay_stale_romantic",
        module="tree_pipeline.decay_stale_romantic",
        description="Auto-demote stale current romantic partners",
    ),
    Step(
        name="repair_inter_main_reciprocity",
        module="tree_pipeline.repair_inter_main_reciprocity",
        description="Patch reciprocity gaps / unreciprocated hallucinations",
    ),
    Step(
        name="normalize_legacy_relationships",
        module="tree_pipeline.normalize_legacy_relationships",
        description="Normalize bare-name / plural-`are` legacy entries",
    ),
    Step(
        name="align_inverse_pair",
        module="tree_pipeline.align_inverse_pair",
        description=(
            "Pass 1 evidence demote (premature husband/fiancé) "
            "+ Pass 2 partner-tier alignment"
        ),
    ),
    Step(
        name="forward_fill_continuity",
        module="tree_pipeline.forward_fill_continuity",
        description="Fill 1-to-N episode regression gaps in main couples",
    ),
    Step(
        name="audit_core_traits",
        module="tree_pipeline.audit_core_traits",
        description=(
            "Periodic LLM audit of core fields (personality / "
            "speaking_style). Disabled by default (requires LLM calls)."
        ),
        enabled_by_default=False,
        long_running=True,
    ),
]


def _snap_path(dataset: str) -> Path:
    return (PROCESSED_DIR / dataset / "intermediate" / "evolution"
            / "persona_snapshots.json")


def _print_banner(text: str) -> None:
    bar = "═" * max(60, len(text) + 4)
    print(bar)
    print(f"  {text}")
    print(bar)


def _run_step(step: Step, base_cmd: list[str], cwd: Path,
              dry_run: bool, extra: list[str]) -> dict:
    """Run a single sub-step and return a dict with status + timing.

    Long-running (LLM) steps stream output line-by-line so progress is
    visible in the pipeline log file in real time.  Quick patch steps
    capture-and-print at completion to keep the log compact.
    """
    cmd = base_cmd[:] + extra
    if dry_run and step.supports_dry_run:
        cmd.append("--dry_run")
    started = time.perf_counter()
    print(f"\n→ Running: python -m {step.module} {' '.join(cmd[3:])}",
          flush=True)
    if step.long_running:
        # Stream stdout line-by-line so the parent log shows progress while
        # the LLM step is still running, and also collect the tail.
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        captured: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            captured.append(line)
        proc.wait()
        rc = proc.returncode
        out = "".join(captured)
    else:
        result = subprocess.run(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
        rc = result.returncode
        out = result.stdout or ""
        print(out.rstrip(), flush=True)
    elapsed = time.perf_counter() - started
    status = "ok" if rc == 0 else f"failed (rc={rc})"
    print(f"   [{step.name}] {status} in {elapsed:.1f}s", flush=True)
    return {
        "name": step.name,
        "module": step.module,
        "returncode": rc,
        "elapsed_s": round(elapsed, 2),
        "tail": out.strip().split("\n")[-12:],
    }


def _backup_snapshot(snap_path: Path) -> Path | None:
    if not snap_path.exists():
        return None
    ts = time.strftime("%Y%m%d-%H%M%S")
    backup = snap_path.with_suffix(snap_path.suffix + f".pipeline_{ts}")
    shutil.copy2(snap_path, backup)
    return backup


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end evolve + post-processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["Friends", "TheOffice", "StarTrek_TNG", "HPD"],
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Pass --dry_run to every patch step (evolve has no dry-run).",
    )
    parser.add_argument(
        "--skip_evolve", action="store_true",
        help="Skip the LLM evolve step and run patches only.",
    )
    parser.add_argument(
        "--include_audit", action="store_true",
        help="Include the (off-by-default) audit_core_traits step.",
    )
    parser.add_argument(
        "--skip", default="", metavar="step1,step2",
        help="Comma-separated list of step names to skip.",
    )
    parser.add_argument(
        "--only", default="", metavar="step1,step2",
        help="Run only these named steps (overrides everything else).",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Forwarded to evolve_persona (max concurrent API requests).",
    )
    parser.add_argument(
        "--test_episode", default=None,
        help="Forwarded to evolve_persona (cap evolution at this episode).",
    )
    parser.add_argument(
        "--audit_workers", type=int, default=None,
        help="Forwarded to audit_core_traits as --workers.",
    )
    parser.add_argument(
        "--audit_interval", type=int, default=None,
        help="Forwarded to audit_core_traits as --interval (eps).",
    )
    parser.add_argument(
        "--audit_episodes", default=None,
        help="Forwarded to audit_core_traits as --audit-episodes.",
    )
    parser.add_argument(
        "--no_backup", action="store_true",
        help="Do not snapshot persona_snapshots.json before running patches.",
    )
    args = parser.parse_args()

    skipped = {s.strip() for s in args.skip.split(",") if s.strip()}
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    valid_names = {s.name for s in STEPS}
    bad = (skipped | only) - valid_names
    if bad:
        print(f"Unknown step name(s): {sorted(bad)}", file=sys.stderr)
        print(f"Valid names: {sorted(valid_names)}", file=sys.stderr)
        return 2

    base_cmd_evolve = [sys.executable, "-m", "tree_pipeline.evolve_persona",
                       "--dataset", args.dataset]
    if args.workers is not None:
        base_cmd_evolve += ["--workers", str(args.workers)]
    if args.test_episode is not None:
        base_cmd_evolve += ["--test_episode", args.test_episode]

    plan: list[tuple[Step, list[str]]] = []
    for step in STEPS:
        if only and step.name not in only:
            continue
        if step.name in skipped:
            continue
        if step.name == "evolve_persona":
            if args.skip_evolve and not only:
                continue
            plan.append((step, base_cmd_evolve[3:]))  # extra after dataset
            continue
        if step.name == "audit_core_traits":
            if not args.include_audit and not only:
                continue
            extra = ["--dataset", args.dataset]
            if args.audit_workers is not None:
                extra += ["--workers", str(args.audit_workers)]
            if args.audit_interval is not None:
                extra += ["--interval", str(args.audit_interval)]
            if args.audit_episodes is not None:
                extra += ["--audit-episodes", args.audit_episodes]
            if args.dry_run:
                extra += ["--dry-run"]  # hyphen variant — audit uses --dry-run
            plan.append((step, extra))
            continue
        # Standard patch step: just --dataset (and --dry_run added by runner).
        plan.append((step, ["--dataset", args.dataset]))

    if not plan:
        print("No steps selected.  Check --only / --skip / --skip_evolve.",
              file=sys.stderr)
        return 1

    _print_banner(
        f"Persona-evolution pipeline   dataset={args.dataset}   "
        f"dry_run={args.dry_run}   steps={len(plan)}"
    )
    print("Plan:")
    for i, (step, _) in enumerate(plan, 1):
        marker = "  (LLM)" if step.long_running else ""
        print(f"  {i}. {step.name:<32} — {step.description}{marker}")

    snap = _snap_path(args.dataset)
    if not args.no_backup and not args.dry_run and snap.exists():
        bak = _backup_snapshot(snap)
        if bak:
            print(f"\nPipeline-level backup: {bak}")

    summaries: list[dict] = []
    for i, (step, extra) in enumerate(plan, 1):
        _print_banner(f"Step {i}/{len(plan)}: {step.name}")
        if step.name == "audit_core_traits":
            base_cmd = [sys.executable, "-m", step.module]
        elif step.name == "evolve_persona":
            base_cmd = [sys.executable, "-m", step.module, "--dataset",
                        args.dataset]
            extra = [a for a in extra if a not in {"--dataset", args.dataset}]
        else:
            base_cmd = [sys.executable, "-m", step.module]
        info = _run_step(step, base_cmd, cwd=SRC_ROOT,
                         dry_run=args.dry_run, extra=extra)
        summaries.append(info)
        if info["returncode"] != 0:
            print(f"\nStep '{step.name}' failed; halting pipeline.",
                  file=sys.stderr)
            break

    _print_banner("Pipeline summary")
    total = sum(s["elapsed_s"] for s in summaries)
    for s in summaries:
        marker = "OK" if s["returncode"] == 0 else "FAIL"
        print(f"  {marker:<4} {s['name']:<32} {s['elapsed_s']:>8.1f}s")
    print(f"  {'-' * 50}")
    print(f"       {'TOTAL':<32} {total:>8.1f}s")

    log_dir = (PROCESSED_DIR / args.dataset / "intermediate" / "evolution")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline_evolve_full_log.json"
    log_path.write_text(json.dumps({
        "dataset": args.dataset,
        "dry_run": args.dry_run,
        "skipped": sorted(skipped),
        "only": sorted(only),
        "summaries": summaries,
        "total_seconds": round(total, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, ensure_ascii=False, indent=2))
    print(f"  log -> {log_path}")
    return 0 if all(s["returncode"] == 0 for s in summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
