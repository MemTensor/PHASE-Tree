"""Continuity filler for inter-main couple relationships.

Detects gaps of length 1–``max_gap`` where:

* In gap episodes ``E0``, neither side has a current couple-tier role
  pointing at the other.
* In ``E0-1`` (pre-gap) and ``E0+k`` (post-gap, within ``look_ahead``)
  BOTH characters have current couple roles for each other.
* The pre-gap and post-gap roles for each side are either identical OR
  the pre-gap tier is *lower-or-equal* to the post-gap tier.  This covers
  two valid scenarios:

  1. Stable continuity (boyfriend → boyfriend across gap).
  2. Legitimate canonical escalation (fiancé → husband across gap, where
     the wedding episode comes after the gap).  In this case we fill the
     gap with the *lower-tier* (pre-gap) role.

* The cumulative archive must NOT contain a breakup between pre-gap and
  post-gap episodes (real brief breakups should not be auto-filled).

Idempotent — running the script again is a no-op once gaps are filled.

Usage::

    python -m tree_pipeline.forward_fill_continuity --dataset Friends --dry_run
    python -m tree_pipeline.forward_fill_continuity --dataset Friends
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tree_pipeline.evolve_persona import (  # type: ignore
    _episode_to_int,
    _has_breakup_between,
    _parse_relationship_entries,
    _split_relationships_paren_aware,
    _role_tier,
)

PROCESSED_DIR = PROJECT_ROOT / "LongEvoRoleBench" / "processed"


def _current_couple_role_at(rel: str, target_first: str) -> str | None:
    """Return the (lower-cased) couple role pointing at ``target_first`` in
    ``rel``, or None if no current couple role exists.
    """
    for role, name, _ in _parse_relationship_entries(rel):
        rl = role.lower().strip()
        if rl.startswith(("ex-", "former", "late-")):
            continue
        if name.split()[0].lower() != target_first.lower():
            continue
        if _role_tier(rl) > 0:
            return rl
    return None


def _all_partner_firsts(snaps_a: dict) -> set[str]:
    """Collect every distinct first-name partner that ``a`` ever points at
    via a current couple-tier role across all snapshots."""
    out: set[str] = set()
    for ep, snap in snaps_a.items():
        rel = snap["tree"]["persona"]["relationships"].get("value", "")
        for role, name, _ in _parse_relationship_entries(rel):
            rl = role.lower().strip()
            if rl.startswith(("ex-", "former", "late-")):
                continue
            if _role_tier(rl) <= 0:
                continue
            first = name.split()[0]
            if first:
                out.add(first)
    return out


def detect_gaps(
    snaps: dict,
    main_chars: list[str],
    archives: dict,
    max_gap: int = 8,
    look_ahead: int = 3,
) -> list[dict]:
    """Find continuity gaps for each (main, partner-first) couple
    relationship.  Partner can be a main *or* a non-main character —
    whoever the main currently couples with.
    """
    main_first_to_full = {c.split()[0].lower(): c for c in main_chars}
    decisions: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for a in main_chars:
        a_first = a.split()[0]
        snaps_a = snaps.get(a, {})
        eps = sorted(snaps_a.keys(), key=_episode_to_int)
        for partner_first in sorted(_all_partner_firsts(snaps_a)):
            if partner_first.lower() == a_first.lower():
                continue
            partner_full = main_first_to_full.get(partner_first.lower())
            snaps_b = snaps.get(partner_full, {}) if partner_full else {}

            a_role: dict[str, str | None] = {}
            b_role: dict[str, str | None] = {}
            for ep in eps:
                a_rel = (
                    snaps_a[ep]["tree"]["persona"]["relationships"]
                    .get("value", "")
                )
                a_role[ep] = _current_couple_role_at(a_rel, partner_first)
                if partner_full and ep in snaps_b:
                    b_rel = (
                        snaps_b[ep]["tree"]["persona"]["relationships"]
                        .get("value", "")
                    )
                    b_role[ep] = _current_couple_role_at(b_rel, a_first)
                else:
                    b_role[ep] = None  # non-main partner

            i = 0
            while i < len(eps):
                ep = eps[i]
                if a_role[ep] is None and (
                        b_role[ep] is None or partner_full is None):
                    j = i
                    while (j < len(eps)
                           and a_role[eps[j]] is None
                           and (b_role[eps[j]] is None
                                or partner_full is None)):
                        # If this is a main pair and the partner has a
                        # role at this ep, stop the gap run here.
                        if (partner_full is not None
                                and b_role[eps[j]] is not None):
                            break
                        j += 1
                    gap_eps = eps[i:j]
                    pre_idx = i - 1
                    post_idx = j
                    if pre_idx < 0 or post_idx >= len(eps):
                        i = j
                        continue
                    if len(gap_eps) == 0 or len(gap_eps) > max_gap:
                        i = j
                        continue

                    pre_a = a_role[eps[pre_idx]]
                    pre_b = (b_role[eps[pre_idx]]
                             if partner_full else None)
                    if pre_a is None:
                        i = j
                        continue
                    if partner_full is not None and pre_b is None:
                        i = j
                        continue

                    # Look ahead for resumption.
                    resumed = False
                    for k in range(post_idx,
                                   min(len(eps), post_idx + look_ahead)):
                        ar = a_role[eps[k]]
                        br = b_role[eps[k]]
                        if ar is None:
                            continue
                        if _role_tier(ar) < _role_tier(pre_a):
                            continue
                        if partner_full is not None:
                            if br is None:
                                continue
                            if _role_tier(br) < _role_tier(pre_b):
                                continue
                        resumed = True
                        break
                    if not resumed:
                        i = j
                        continue

                    # Reject if archive has a breakup in the window.
                    pre_ord = _episode_to_int(eps[pre_idx])
                    post_ord = _episode_to_int(eps[post_idx])
                    arc_a = archives.get(a, [])
                    breakup = False
                    for ar_e in arc_a:
                        eo = (int(ar_e.get("season", 0)) * 100
                              + int(ar_e.get("episode", 0)))
                        if pre_ord < eo <= post_ord:
                            if _has_breakup_between(
                                    ar_e.get("summary", ""),
                                    a_first, partner_first):
                                breakup = True
                                break
                    if breakup:
                        i = j
                        continue
                    key = (a, partner_first, gap_eps[0])
                    if key in seen:
                        i = j
                        continue
                    seen.add(key)
                    decisions.append({
                        "char_a": a,
                        "char_b": partner_full or partner_first,
                        "partner_is_main": partner_full is not None,
                        "role_a": pre_a,
                        "role_b": pre_b,
                        "gap_episodes": list(gap_eps),
                        "pre_ep": eps[pre_idx],
                        "resume_ep": eps[post_idx],
                    })
                    i = j
                else:
                    i += 1
    return decisions


def apply_gap_fill(snaps: dict, decision: dict) -> int:
    """Fill the gap episodes for the main side (and the partner side if
    they are also a main).  Returns number of side-eps written.
    """
    a = decision["char_a"]
    b = decision["char_b"]
    a_first = a.split()[0]
    b_first = b.split()[0]
    role_a = decision["role_a"]
    role_b = decision["role_b"]
    is_main_pair = decision.get("partner_is_main", False)

    fills = [(a, b_first, role_a)]
    if is_main_pair and role_b is not None:
        fills.append((b, a_first, role_b))

    written = 0
    for ep in decision["gap_episodes"]:
        for owner, target_first, role in fills:
            if ep not in snaps.get(owner, {}):
                continue
            rel_node = snaps[owner][ep]["tree"]["persona"]["relationships"]
            old_val = rel_node.get("value", "")
            existing = _current_couple_role_at(old_val, target_first)
            if existing == role:
                continue
            new_parts: list[str] = []
            for raw in _split_relationships_paren_aware(old_val):
                m = re.match(r"^\s*([\w\- ]+?)\s+is\s+", raw, re.I)
                if m:
                    rl = m.group(1).strip().lower()
                    names_in = re.findall(r"[A-Z][\w\-'.]+", raw)
                    if any(n.lower() == target_first.lower()
                           for n in names_in):
                        # Drop any prior entry pointing at the same partner
                        # (friend / ex-girlfriend / former boyfriend / etc.)
                        # so the gap-fill role is the *only* entry for them.
                        continue
                new_parts.append(raw)
            new_parts.append(f"{role} is {target_first}")
            rel_node["value"] = ", ".join(p for p in new_parts if p)
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Friends")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_gap", type=int, default=8)
    parser.add_argument("--look_ahead", type=int, default=3)
    args = parser.parse_args()

    ev_dir = PROCESSED_DIR / args.dataset / "intermediate" / "evolution"
    snap_path = ev_dir / "persona_snapshots.json"
    log_path = ev_dir / "continuity_fill_log.json"
    trees_path = (
        PROCESSED_DIR / args.dataset / "intermediate" / "attribute_trees.json"
    )
    main_chars = list(json.loads(trees_path.read_text()).keys())

    archives: dict = {}
    for c in main_chars:
        path = ev_dir / f"{c.replace(' ', '_')}_session_archive.json"
        archives[c] = json.loads(path.read_text()) if path.exists() else []

    snaps = json.loads(snap_path.read_text())
    decisions = detect_gaps(
        snaps, main_chars, archives,
        max_gap=args.max_gap,
        look_ahead=args.look_ahead,
    )

    # For inter-main pairs, dedupe symmetric A↔B/B↔A entries — keep only
    # one canonical decision per unordered pair × gap window.  Non-main
    # partner decisions are inherently one-sided so are kept as-is.
    canonical: list[dict] = []
    seen_pairs: set[tuple] = set()
    for d in decisions:
        if d.get("partner_is_main"):
            key = (
                tuple(sorted([d["char_a"], d["char_b"]])),
                d["gap_episodes"][0],
                d["gap_episodes"][-1],
            )
            flat = (key[0][0], key[0][1], f"{key[1]}~{key[2]}")
            if flat in seen_pairs:
                continue
            seen_pairs.add(flat)
        canonical.append(d)

    print(f"Continuity gaps detected: {len(canonical)}")
    for d in canonical:
        eps = d["gap_episodes"]
        a_first = d["char_a"].split()[0]
        b_label = d["char_b"].split()[0] if d.get("partner_is_main") \
            else d["char_b"]
        role_str = (f"({d['role_a']}/{d['role_b']})"
                    if d.get("partner_is_main")
                    else f"({d['role_a']})")
        print(f"  {a_first:<8} ↔ {b_label:<8} "
              f"gap {eps[0]}..{eps[-1]} ({len(eps)} ep{'s' if len(eps)>1 else ''}) "
              f"pre={d['pre_ep']} resume={d['resume_ep']} role={role_str}")

    if args.dry_run:
        print("\n(dry-run: no files written)")
        return

    backup_path = snap_path.with_suffix(snap_path.suffix + ".before_continuity")
    snap_path.rename(backup_path)
    total_written = 0
    for d in canonical:
        total_written += apply_gap_fill(snaps, d)
    snap_path.write_text(json.dumps(snaps, ensure_ascii=False, indent=2))
    log_path.write_text(json.dumps({
        "decisions": canonical,
        "side_eps_written": total_written,
    }, ensure_ascii=False, indent=2))
    print(f"\nBackup:  {backup_path}")
    print(f"Updated: {snap_path}")
    print(f"Log:     {log_path}")
    print(f"Side-episodes filled: {total_written}")


if __name__ == "__main__":
    main()
