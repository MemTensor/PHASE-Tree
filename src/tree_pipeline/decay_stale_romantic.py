"""Legacy batch post-processor for stale current-romantic partners.

As of the v9 refactor the decay logic lives in :mod:`tree_pipeline.evolve_persona`
and runs once per episode automatically during the main evolution loop.  This
script is kept for backwards compatibility — useful when you have a
pre-existing ``persona_snapshots.json`` (produced by an older evolve build,
or after a manual edit) and want to apply the same decay heuristics in a
single batch pass.

Usage::

    python -m tree_pipeline.decay_stale_romantic --dataset Friends
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tree_pipeline.evolve_persona import (  # type: ignore
    _episode_to_int,
    _parse_relationship_entries,
    decay_relationships_value,
)

PROCESSED_DIR = PROJECT_ROOT / "phase_tree_data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Friends")
    parser.add_argument("--dry_run", action="store_true",
                        help="print summary without writing files")
    args = parser.parse_args()

    ev_dir = PROCESSED_DIR / args.dataset / "intermediate" / "evolution"
    snap_path = ev_dir / "persona_snapshots.json"
    log_path = ev_dir / "decay_log.json"

    trees_path = PROCESSED_DIR / args.dataset / "intermediate" / "attribute_trees.json"
    main_chars = list(json.loads(trees_path.read_text()).keys())

    snaps = json.loads(snap_path.read_text())
    archives = {
        c: json.loads((ev_dir / f"{c.replace(' ', '_')}_session_archive.json").read_text())
        for c in main_chars
    }

    # Pre-compute first_seen_ord across the entire snapshot history.
    first_seen_ord: dict[tuple[str, str], int] = {}
    for c in main_chars:
        eps_sorted = sorted(snaps.get(c, {}).keys(), key=_episode_to_int)
        seen: set[str] = set()
        for ep in eps_sorted:
            rel = (
                snaps[c][ep]["tree"]["persona"]["relationships"]
                .get("value", "")
            )
            for role, name, _raw in _parse_relationship_entries(rel):
                role_l = role.lower().strip()
                if role_l.startswith("ex-") or role_l.startswith("late-"):
                    continue
                key = f"{role_l}|{name.lower()}"
                if key in seen:
                    continue
                seen.add(key)
                first_seen_ord[(c, key)] = _episode_to_int(ep)

    decay_records: list[dict] = []
    new_snaps: dict = copy.deepcopy(snaps)
    for c in main_chars:
        for ep, snap in new_snaps.get(c, {}).items():
            rel_field = snap["tree"]["persona"]["relationships"]
            old_val = rel_field.get("value", "")
            new_val, decays = decay_relationships_value(
                old_val, c, ep,
                archives[c],
                None,        # current_trees not available in batch mode
                new_snaps,   # use snapshots as the inter-main lookup source
                first_seen_ord, main_chars,
            )
            if decays:
                rel_field["value"] = new_val
                decay_records.extend(decays)

    by_char: dict[str, int] = defaultdict(int)
    by_partner: dict[tuple[str, str], int] = defaultdict(int)
    for r in decay_records:
        by_char[r["character"]] += 1
        from_text = r["from_entry"]
        # Pull partner name out of "<role> is <Name> ..."
        idx = from_text.find(" is ")
        if idx >= 0:
            partner = from_text[idx + 4:].strip().split(" (")[0]
            by_partner[(r["character"], partner)] += 1

    print(f"Total decays applied: {len(decay_records)}")
    print("\nPer-character decay count:")
    for c in main_chars:
        print(f"  {c:<20} {by_char.get(c, 0):>4}")
    print("\nTop (char, partner) pairs by decay count:")
    for (c, p), n in sorted(by_partner.items(), key=lambda kv: -kv[1])[:25]:
        print(f"  {c:<20}   -> {p:<25} {n:>4}")

    if args.dry_run:
        print("\n(dry-run: no files written)")
        return

    backup_path = snap_path.with_suffix(snap_path.suffix + ".before_decay")
    snap_path.rename(backup_path)
    snap_path.write_text(json.dumps(new_snaps, ensure_ascii=False, indent=2))
    log_path.write_text(json.dumps(decay_records, ensure_ascii=False, indent=2))

    print(f"\nBackup written: {backup_path}")
    print(f"Updated:        {snap_path}")
    print(f"Decay log:      {log_path}")


if __name__ == "__main__":
    main()
