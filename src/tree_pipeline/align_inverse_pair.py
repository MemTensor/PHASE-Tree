"""Align inter-main couple-role pairs to evidenced status (two passes).

Pass 1 — **Global evidence demotion**.  For every main M and every
``<role> is <Other-Main>`` couple-tier entry in M's snapshots, validate that
M's cumulative session archive (up to that episode) actually contains the
canonical *completed* signal:

* ``husband / wife / spouse`` requires marriage evidence.
* ``fiancé / fiancée`` requires engagement evidence.

If unsupported, demote to the highest evidenced tier (married → engaged →
dating).  This catches single-side premature-escalation bugs (Monica jumping
to ``husband`` 25 eps before the canonical wedding episode in Friends).

Pass 2 — **Inverse-pair alignment**.  After Pass 1, walk every main pair and
look for residual mismatches where A claims tier-N but B claims tier-M < N.
Demote A to tier-M.  Pass 1 has already removed the unsupported claims, so
Pass 2 only fires when one side is at a *lower* but still-evidenced tier
(typical reading: the laggard is the source of truth).

Tier hierarchy::

    1  dating       boyfriend / girlfriend / partner / lover / dating / ...
    2  engaged      fiancé / fiancée / wife-to-be / husband-to-be
    3  married      husband / wife / spouse

Usage::

    python -m tree_pipeline.align_inverse_pair --dataset Friends --dry_run
    python -m tree_pipeline.align_inverse_pair --dataset Friends
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
    _has_engagement_evidence,
    _has_marriage_evidence,
    _parse_relationship_entries,
    _split_relationships_paren_aware,
    _DATING_DEMOTION,
    _ENGAGED_DEMOTION,
    _role_tier,
    STATUS_TIER_DATING,
    STATUS_TIER_ENGAGED,
    STATUS_TIER_MARRIED,
)

PROCESSED_DIR = PROJECT_ROOT / "phase_tree_data" / "processed"


def _is_couple_role(role: str) -> bool:
    return _role_tier(role.lower().strip()) > 0


def _entries_pointing_at(rel: str, target_first: str
                         ) -> list[tuple[str, str, str]]:
    out = []
    for role, name, raw in _parse_relationship_entries(rel):
        if name.split()[0].lower() == target_first.lower():
            if not role.lower().startswith(("ex-", "former", "late-")):
                out.append((role, name, raw))
    return out


def _demote_role(role_l: str, target_tier: int) -> str | None:
    """Return demoted role for ``role_l`` so it lands at ``target_tier``."""
    cur_tier = _role_tier(role_l)
    if cur_tier <= target_tier:
        return None
    if target_tier == STATUS_TIER_ENGAGED:
        return _ENGAGED_DEMOTION.get(role_l)
    if target_tier == STATUS_TIER_DATING:
        return _DATING_DEMOTION.get(role_l)
    return None


def _archive_blob_up_to(archive: list[dict], ep_ord: int) -> str:
    return "\n".join(
        s.get("summary", "") for s in archive
        if int(s.get("season", 0)) * 100 + int(s.get("episode", 0)) <= ep_ord
    )


def detect_evidence_demotions(
    snaps: dict, main_chars: list[str], archives: dict,
) -> list[dict]:
    """Pass 1 — for each character ep, demote any tier-2/3 inter-main couple
    role that lacks corresponding completion evidence in the cumulative
    archive.
    """
    main_first_to_full = {c.split()[0].lower(): c for c in main_chars}
    decisions: list[dict] = []
    for c in main_chars:
        archive = archives.get(c, [])
        for ep, snap in snaps.get(c, {}).items():
            ep_ord = _episode_to_int(ep)
            blob = _archive_blob_up_to(archive, ep_ord)
            char_first = c.split()[0]
            rel = snap["tree"]["persona"]["relationships"].get("value", "")
            for role, name, raw in _parse_relationship_entries(rel):
                role_l = role.lower().strip()
                if role_l.startswith(("ex-", "former", "late-")):
                    continue
                tier = _role_tier(role_l)
                if tier <= STATUS_TIER_DATING:
                    continue
                partner_first = name.split()[0].lower()
                if partner_first not in main_first_to_full:
                    continue
                partner_full = main_first_to_full[partner_first]
                if partner_full == c:
                    continue
                if tier == STATUS_TIER_MARRIED:
                    if _has_marriage_evidence(blob, char_first, partner_first):
                        continue
                    if _has_engagement_evidence(blob, char_first, partner_first):
                        target_tier = STATUS_TIER_ENGAGED
                        target = _demote_role(role_l, STATUS_TIER_ENGAGED)
                        reason = "no marriage evidence; demote to engaged"
                    else:
                        target_tier = STATUS_TIER_DATING
                        target = _demote_role(role_l, STATUS_TIER_DATING)
                        reason = "no marriage/engagement evidence; demote to dating"
                else:  # tier == STATUS_TIER_ENGAGED
                    if _has_engagement_evidence(blob, char_first, partner_first):
                        continue
                    target_tier = STATUS_TIER_DATING
                    target = _demote_role(role_l, STATUS_TIER_DATING)
                    reason = "no engagement evidence; demote to dating"
                if target is None:
                    continue
                decisions.append({
                    "type": "evidence_demote",
                    "character": c,
                    "episode": ep,
                    "old_role": role_l,
                    "new_role": target,
                    "partner_full": partner_full,
                    "partner_name": name,
                    "raw_entry": raw,
                    "old_tier": tier,
                    "new_tier": target_tier,
                    "reason": reason,
                })
    return decisions


def detect_mismatches(snaps: dict, main_chars: list[str]
                      ) -> list[dict]:
    """Pass 2 — residual mismatch alignment (after Pass 1 has been applied
    in memory).  For each character ep, if a couple-tier role points at a
    main partner who claims a strictly lower tier, demote our claim.
    """
    main_first_to_full = {c.split()[0].lower(): c for c in main_chars}
    decisions: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    for c in main_chars:
        for ep, snap in snaps.get(c, {}).items():
            rel = snap["tree"]["persona"]["relationships"].get("value", "")
            for role, name, raw in _parse_relationship_entries(rel):
                role_l = role.lower().strip()
                if role_l.startswith(("ex-", "former", "late-")):
                    continue
                my_tier = _role_tier(role_l)
                if my_tier == 0:
                    continue
                partner_first = name.split()[0].lower()
                if partner_first not in main_first_to_full:
                    continue
                partner_full = main_first_to_full[partner_first]
                if partner_full == c:
                    continue
                if ep not in snaps.get(partner_full, {}):
                    continue
                partner_rel = (
                    snaps[partner_full][ep]
                    ["tree"]["persona"]["relationships"]
                    .get("value", "")
                )
                back_entries = _entries_pointing_at(
                    partner_rel, c.split()[0]
                )
                if not back_entries:
                    continue
                back_tiers = [
                    _role_tier(r.lower()) for r, _, _ in back_entries
                ]
                back_tiers = [t for t in back_tiers if t > 0]
                if not back_tiers:
                    continue
                partner_tier = max(back_tiers)
                if partner_tier >= my_tier:
                    continue
                target = _demote_role(role_l, partner_tier)
                if target is None:
                    continue
                key = (c, ep, role_l, name)
                if key in seen:
                    continue
                seen.add(key)
                decisions.append({
                    "type": "align_to_partner",
                    "character": c,
                    "episode": ep,
                    "old_role": role_l,
                    "new_role": target,
                    "partner_full": partner_full,
                    "partner_name": name,
                    "raw_entry": raw,
                    "my_tier": my_tier,
                    "partner_tier": partner_tier,
                })
    return decisions


def _archives_for(main_chars: list[str], ev_dir: Path) -> dict:
    archives: dict = {}
    for c in main_chars:
        path = ev_dir / f"{c.replace(' ', '_')}_session_archive.json"
        if path.exists():
            archives[c] = json.loads(path.read_text())
        else:
            archives[c] = []
    return archives


def apply_decision(snaps: dict, decision: dict) -> bool:
    c = decision["character"]
    ep = decision["episode"]
    old_role = decision["old_role"]
    new_role = decision["new_role"]
    partner_first = decision["partner_name"].split()[0]
    rel_node = snaps[c][ep]["tree"]["persona"]["relationships"]
    old_val = rel_node.get("value", "")
    new_parts: list[str] = []
    changed = False
    for raw in _split_relationships_paren_aware(old_val):
        m = re.match(r"^\s*([\w\- ]+?)\s+is\s+", raw, re.I)
        if not m:
            new_parts.append(raw)
            continue
        rl = m.group(1).strip().lower()
        names_in = re.findall(r"[A-Z][\w\-']+", raw)
        if rl != old_role or not any(
                n.lower() == partner_first.lower() for n in names_in):
            new_parts.append(raw)
            continue
        replacement = re.sub(
            rf"^\s*{re.escape(m.group(1).strip())}\s+is\s+",
            f"{new_role} is ",
            raw, count=1, flags=re.IGNORECASE,
        )
        new_parts.append(replacement)
        changed = True
    if changed:
        rel_node["value"] = ", ".join(new_parts)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Friends")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    ev_dir = PROCESSED_DIR / args.dataset / "intermediate" / "evolution"
    snap_path = ev_dir / "persona_snapshots.json"
    log_path = ev_dir / "inverse_pair_align_log.json"
    trees_path = (
        PROCESSED_DIR / args.dataset / "intermediate" / "attribute_trees.json"
    )
    main_chars = list(json.loads(trees_path.read_text()).keys())
    archives = _archives_for(main_chars, ev_dir)

    snaps = json.loads(snap_path.read_text())
    snaps_working = copy.deepcopy(snaps)

    # ── Pass 1 — global evidence demotion ─────────────────────────────
    pass1 = detect_evidence_demotions(snaps_working, main_chars, archives)
    by_pair_1 = defaultdict(list)
    for d in pass1:
        by_pair_1[(d["character"], d["partner_full"],
                   d["old_role"], d["new_role"])].append(d["episode"])
    print(f"Pass 1 — evidence demotions: {len(pass1)} entries")
    for (c, p, old_r, new_r), eps in by_pair_1.items():
        eps = sorted(eps, key=_episode_to_int)
        print(f"  {c:<18} '{old_r} is {p.split()[0]}' -> '{new_r}' "
              f"({len(eps)} eps {eps[0]}..{eps[-1]})")

    # Apply Pass 1 in-memory before running Pass 2.
    for d in pass1:
        apply_decision(snaps_working, d)

    # ── Pass 2 — residual mismatch alignment ──────────────────────────
    pass2 = detect_mismatches(snaps_working, main_chars)
    by_pair_2 = defaultdict(list)
    for d in pass2:
        by_pair_2[(d["character"], d["partner_full"],
                   d["old_role"], d["new_role"])].append(d["episode"])
    print(f"\nPass 2 — partner-tier alignment: {len(pass2)} entries")
    for (c, p, old_r, new_r), eps in by_pair_2.items():
        eps = sorted(eps, key=_episode_to_int)
        print(f"  {c:<18} '{old_r} is {p.split()[0]}' -> '{new_r}' "
              f"({len(eps)} eps {eps[0]}..{eps[-1]})")

    if args.dry_run:
        print("\n(dry-run: no files written)")
        return

    backup_path = snap_path.with_suffix(snap_path.suffix + ".before_align")
    snap_path.rename(backup_path)
    n_p1 = sum(1 for d in pass1 if apply_decision(snaps, d))
    # Apply Pass 2 on already-Pass1-applied snapshots (note: we run Pass 2
    # detection on snaps_working, but apply_decision is generic and can run
    # on the already-applied ``snaps``).
    n_p2 = sum(1 for d in pass2 if apply_decision(snaps, d))
    snap_path.write_text(json.dumps(snaps, ensure_ascii=False, indent=2))
    log_path.write_text(json.dumps({
        "pass1_evidence_demote": pass1,
        "pass2_align_to_partner": pass2,
        "n_applied_pass1": n_p1,
        "n_applied_pass2": n_p2,
    }, ensure_ascii=False, indent=2))
    print(f"\nBackup: {backup_path}")
    print(f"Updated: {snap_path}")
    print(f"Log:     {log_path}")
    print(f"Applied: pass1={n_p1}, pass2={n_p2}")


if __name__ == "__main__":
    main()
