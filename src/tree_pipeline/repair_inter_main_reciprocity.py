"""Repair inter-main romantic-reciprocity gaps in persona snapshots.

Two complementary passes — both operate on ``persona_snapshots.json``:

1. **Short-stay hallucination demotion**
   Detect entries of the form ``<rom_role> is <Main>`` that:
     - are *unreciprocated* by the named main character throughout the
       claim's lifespan,
     - last for at most ``--max_short_stay`` consecutive episodes (default 3),
     - and either disappear or are LLM-self-corrected to ``ex-<role>`` within
       a small post-window (≤2 episodes after last occurrence).
   Such entries are most likely LLM hallucinations (e.g. Joey's "girlfriend
   is Rachel" S08E14–E15) and are rewritten to ``ex-<role> is <Main>`` so
   they no longer claim a current relationship.

2. **Sustained-claim reciprocity propagation**
   When character A asserts a current romantic role with character B (both
   in the main cast) for ``--min_propagate_eps`` consecutive episodes
   (default 2), and B's snapshot in the same episodes has *no* romantic role
   pointing back at A, append the canonical inverse role on B's side
   (``boyfriend ↔ girlfriend``, ``husband ↔ wife``, ``fiancé ↔ fiancée``).
   For asymmetric encounter-style roles (``secret romantic encounter``,
   ``one-time encounter``, ``friend with romantic tension``) we mirror the
   exact role on B's side.

Both passes are conservative — they only modify entries they can fully
attribute to the rules above and emit a JSON log of every change.

Example::

    python -m tree_pipeline.repair_inter_main_reciprocity --dataset Friends --dry_run
    python -m tree_pipeline.repair_inter_main_reciprocity --dataset Friends
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
    PROTECTED_CURRENT,
    ROMANTIC_TRANSIENT,
    _episode_to_int,
    _parse_relationship_entries,
    _split_relationships_paren_aware,
)

PROCESSED_DIR = PROJECT_ROOT / "phase_tree_data" / "processed"

# Map "current role on A" -> "canonical inverse role on B" used by the
# reciprocity-propagation pass.  Asymmetric encounter-style roles are
# mirrored 1:1 (so we get matching context on both sides).
ROLE_INVERSE: dict[str, str] = {
    "boyfriend": "girlfriend",
    "girlfriend": "boyfriend",
    "partner": "partner",
    "lover": "lover",
    "love interest": "love interest",
    "romantic interest": "romantic interest",
    "dating": "dating",
    "date": "date",
    "husband": "wife",
    "wife": "husband",
    "spouse": "spouse",
    "fiancé": "fiancée",
    "fiancée": "fiancé",
    "fiance": "fiancee",
    "fiancee": "fiance",
    "wife-to-be": "husband-to-be",
    "husband-to-be": "wife-to-be",
    "secret romantic encounter": "secret romantic encounter",
    "one-time encounter": "one-time encounter",
    "friend with romantic tension": "friend with romantic tension",
    "on-off girlfriend": "on-off boyfriend",
    "on-off boyfriend": "on-off girlfriend",
    "secret girlfriend": "secret boyfriend",
    "secret boyfriend": "secret girlfriend",
    "crush": "crush",
}

ROMANTIC_ALL = ROMANTIC_TRANSIENT | PROTECTED_CURRENT


def _is_current_romantic(role: str) -> bool:
    role_l = role.lower().strip()
    if role_l.startswith(("ex-", "former ", "late-")):
        return False
    if role_l in ROMANTIC_ALL:
        return True
    for r in ROMANTIC_ALL:
        if role_l == r or role_l.startswith(r + " "):
            return True
    return False


def _is_ex_or_former(role: str) -> bool:
    role_l = role.lower().strip()
    return role_l.startswith(("ex-", "former ", "late-"))


def _name_match(a: str, b: str) -> bool:
    return a.split()[0].lower() == b.split()[0].lower()


def _episodes_sorted(snaps_for_char: dict) -> list[str]:
    return sorted(snaps_for_char.keys(), key=_episode_to_int)


def _current_romantic_partners(rel_value: str) -> list[tuple[str, str, str]]:
    """Return all current romantic ``(role, name, raw_entry)`` tuples."""
    return [
        (role, name, raw)
        for role, name, raw in _parse_relationship_entries(rel_value)
        if _is_current_romantic(role)
    ]


def _has_inverse_or_same_pointing_at(
    rel_value: str, target_first: str
) -> tuple[bool, list[str]]:
    """Does ``rel_value`` already list any *current* romantic role pointing
    at ``target_first``?  Returns ``(matched, matched_roles)``.
    """
    matched_roles: list[str] = []
    for role, name, _raw in _current_romantic_partners(rel_value):
        if _name_match(name, target_first):
            matched_roles.append(role)
    return (bool(matched_roles), matched_roles)


def _has_ex_pointing_at(rel_value: str, target_first: str) -> bool:
    for role, name, _raw in _parse_relationship_entries(rel_value):
        if _is_ex_or_former(role) and _name_match(name, target_first):
            return True
    return False


def _build_claim_runs(
    char_name: str,
    snaps_for_char: dict,
    main_first_to_full: dict[str, str],
) -> list[dict]:
    """Group consecutive episodes where ``char_name`` claims the same
    (role, partner_main) into 'runs'.

    Each run captures: partner_full, role, episodes (sorted), and the next
    episode's classification ('absent'/'ex'/'still_current_diff_role').
    """
    eps = _episodes_sorted(snaps_for_char)
    ep_index = {ep: i for i, ep in enumerate(eps)}
    # Per (partner_first, role) -> list of episode lists (consecutive runs)
    pair_eps: dict[tuple[str, str], list[str]] = defaultdict(list)
    pair_eps_idx: dict[tuple[str, str], list[int]] = defaultdict(list)
    for ep in eps:
        rel = (
            snaps_for_char[ep]["tree"]["persona"]["relationships"]
            .get("value", "")
        )
        for role, name, _raw in _current_romantic_partners(rel):
            partner_first = name.split()[0].lower()
            if partner_first not in main_first_to_full:
                continue
            if main_first_to_full[partner_first] == char_name:
                continue
            key = (partner_first, role.lower())
            pair_eps[key].append(ep)
            pair_eps_idx[key].append(ep_index[ep])

    runs: list[dict] = []
    for (partner_first, role), ep_list in pair_eps.items():
        idx_list = pair_eps_idx[(partner_first, role)]
        # Split into consecutive-index runs.
        if not idx_list:
            continue
        cur_run = [ep_list[0]]
        cur_idx = [idx_list[0]]
        for i in range(1, len(idx_list)):
            if idx_list[i] == cur_idx[-1] + 1:
                cur_run.append(ep_list[i])
                cur_idx.append(idx_list[i])
            else:
                runs.append({
                    "partner_first": partner_first,
                    "partner_full": main_first_to_full[partner_first],
                    "role": role,
                    "episodes": list(cur_run),
                    "ep_indices": list(cur_idx),
                })
                cur_run = [ep_list[i]]
                cur_idx = [idx_list[i]]
        runs.append({
            "partner_first": partner_first,
            "partner_full": main_first_to_full[partner_first],
            "role": role,
            "episodes": list(cur_run),
            "ep_indices": list(cur_idx),
        })
    return runs


# ---------------------------------------------------------------------------
# PASS 1: short-stay hallucination demotion
# ---------------------------------------------------------------------------

def detect_short_stay_hallucinations(
    snaps: dict,
    main_chars: list[str],
    max_short_stay: int = 3,
    self_correction_window: int = 2,
) -> list[dict]:
    """Return a list of demotion records (one per (char, episode, role, name)
    pair to demote)."""
    main_first_to_full = {c.split()[0].lower(): c for c in main_chars}

    decisions: list[dict] = []
    for c in main_chars:
        eps = _episodes_sorted(snaps.get(c, {}))
        ep_index = {ep: i for i, ep in enumerate(eps)}
        runs = _build_claim_runs(c, snaps[c], main_first_to_full)
        for run in runs:
            if len(run["episodes"]) > max_short_stay:
                continue
            partner_full = run["partner_full"]
            partner_first = run["partner_first"]
            partner_eps_idx_set = set(run["ep_indices"])
            # 1) Unreciprocated throughout the run?
            unreciprocated_throughout = True
            for ep in run["episodes"]:
                if ep not in snaps.get(partner_full, {}):
                    continue
                pr = (
                    snaps[partner_full][ep]
                    ["tree"]["persona"]["relationships"]
                    .get("value", "")
                )
                matched, _ = _has_inverse_or_same_pointing_at(
                    pr, c.split()[0]
                )
                if matched:
                    unreciprocated_throughout = False
                    break
            if not unreciprocated_throughout:
                continue
            # 2) After the run ends: either entry disappears OR turns into ex-
            last_idx = run["ep_indices"][-1]
            post_window = eps[
                last_idx + 1: last_idx + 1 + self_correction_window
            ]
            self_corrected = False
            for ep in post_window:
                rel = (
                    snaps[c][ep]["tree"]["persona"]["relationships"]
                    .get("value", "")
                )
                # Entry no longer current?
                still_current = any(
                    _name_match(name, partner_first)
                    and role.lower() == run["role"]
                    for role, name, _raw in _current_romantic_partners(rel)
                )
                if still_current:
                    continue
                # Either it's gone (absent) or marked ex-/former
                turned_ex = _has_ex_pointing_at(rel, partner_first)
                self_corrected = True
                self_correction_kind = "ex" if turned_ex else "absent"
                break

            # If the run is the last segment of the timeline, also accept it
            # (no further evidence to corroborate the relationship).
            if not post_window:
                self_corrected = True
                self_correction_kind = "end-of-timeline"

            if not self_corrected:
                continue

            decisions.append({
                "type": "short_stay_demote",
                "character": c,
                "partner_full": partner_full,
                "role": run["role"],
                "episodes": list(run["episodes"]),
                "kind": self_correction_kind,
                "reason": (
                    f"unreciprocated for {len(run['episodes'])} eps "
                    f"and self-corrected ({self_correction_kind})"
                ),
            })
    return decisions


def apply_short_stay_demotion(snaps: dict, decision: dict) -> int:
    """Mutate ``snaps`` in place to demote the entries described by
    ``decision``. Returns the number of entries actually rewritten."""
    c = decision["character"]
    role = decision["role"]
    partner_first = decision["partner_full"].split()[0]
    rewritten = 0
    for ep in decision["episodes"]:
        if ep not in snaps.get(c, {}):
            continue
        rel_node = snaps[c][ep]["tree"]["persona"]["relationships"]
        old_val = rel_node.get("value", "")
        new_parts: list[str] = []
        changed = False
        for raw in _split_relationships_paren_aware(old_val):
            m = re.match(r"\s*([\w\- ]+?)\s+is\s+", raw)
            if not m:
                new_parts.append(raw)
                continue
            r_lower = m.group(1).strip().lower()
            if r_lower != role.lower():
                new_parts.append(raw)
                continue
            # Match name
            entry_names = re.findall(r"[A-Z][\w\-']+", raw)
            if not any(n.lower() == partner_first.lower()
                       for n in entry_names):
                new_parts.append(raw)
                continue
            # Demote this entry — choose ex- or former based on convention.
            if r_lower in {"secret romantic encounter", "one-time encounter",
                           "friend with romantic tension"}:
                replacement = re.sub(
                    rf"^\s*{re.escape(m.group(1).strip())}\s+is\s+",
                    f"former {m.group(1).strip()} is ", raw, count=1,
                )
            else:
                replacement = re.sub(
                    rf"^\s*{re.escape(m.group(1).strip())}\s+is\s+",
                    f"ex-{m.group(1).strip()} is ", raw, count=1,
                )
            new_parts.append(replacement)
            changed = True
            rewritten += 1
        if changed:
            rel_node["value"] = ", ".join(new_parts)
    return rewritten


# ---------------------------------------------------------------------------
# PASS 2: sustained-claim reciprocity propagation
# ---------------------------------------------------------------------------

def detect_reciprocity_gaps(
    snaps: dict,
    main_chars: list[str],
    min_propagate_eps: int = 2,
) -> list[dict]:
    """For each main A, for each sustained-claim run of length >=
    ``min_propagate_eps`` against another main B, if B is missing any
    reciprocal current entry in those episodes, propose an addition on B's
    side.
    """
    main_first_to_full = {c.split()[0].lower(): c for c in main_chars}

    decisions: list[dict] = []
    seen_pair_run: set[tuple[str, str, str, str]] = set()
    for c in main_chars:
        runs = _build_claim_runs(c, snaps[c], main_first_to_full)
        for run in runs:
            if len(run["episodes"]) < min_propagate_eps:
                continue
            partner_full = run["partner_full"]
            partner_first = run["partner_first"]
            role_l = run["role"]
            inverse_role = ROLE_INVERSE.get(role_l)
            if inverse_role is None:
                # Unsupported role for inverse lookup — skip.
                continue
            # Collect episodes where B is missing reciprocation.
            missing_eps: list[str] = []
            for ep in run["episodes"]:
                if ep not in snaps.get(partner_full, {}):
                    continue
                pr = (
                    snaps[partner_full][ep]
                    ["tree"]["persona"]["relationships"]
                    .get("value", "")
                )
                matched, _ = _has_inverse_or_same_pointing_at(
                    pr, c.split()[0]
                )
                if matched:
                    continue
                # Don't propagate if B explicitly has A as ex- — that's
                # contradictory and the decay/short-stay pass should resolve
                # it instead.
                if _has_ex_pointing_at(pr, c.split()[0]):
                    continue
                missing_eps.append(ep)
            if not missing_eps:
                continue
            key = (
                c, partner_full, role_l, missing_eps[0] + "-" + missing_eps[-1]
            )
            if key in seen_pair_run:
                continue
            seen_pair_run.add(key)
            decisions.append({
                "type": "reciprocity_propagate",
                "source_character": c,
                "target_character": partner_full,
                "source_role": role_l,
                "inverse_role": inverse_role,
                "episodes": list(missing_eps),
                "reason": (
                    f"{c} claims '{role_l} is {partner_first.title()}' for "
                    f"{len(run['episodes'])} consecutive eps; "
                    f"{partner_full} missing reciprocation in "
                    f"{len(missing_eps)}/{len(run['episodes'])} eps"
                ),
            })
    return decisions


def apply_reciprocity_propagation(snaps: dict, decision: dict) -> int:
    """Append an inverse current-romantic entry on the target side for the
    listed episodes."""
    target_full = decision["target_character"]
    source_full = decision["source_character"]
    source_first = source_full.split()[0]
    inverse_role = decision["inverse_role"]
    appended = 0
    for ep in decision["episodes"]:
        if ep not in snaps.get(target_full, {}):
            continue
        rel_node = snaps[target_full][ep]["tree"]["persona"]["relationships"]
        old_val = rel_node.get("value", "")
        # Idempotency: if matched already, skip.
        matched, _ = _has_inverse_or_same_pointing_at(old_val, source_first)
        if matched:
            continue
        # Drop any pre-existing "friend is X" entry — it's clearly stale once
        # they are explicitly current-romantic on the other side.
        new_parts: list[str] = []
        for raw in _split_relationships_paren_aware(old_val):
            m = re.match(r"\s*([\w\- ]+?)\s+is\s+", raw)
            entry_names = re.findall(r"[A-Z][\w\-']+", raw)
            if (m and m.group(1).strip().lower() == "friend"
                    and any(n.lower() == source_first.lower()
                            for n in entry_names)):
                continue
            new_parts.append(raw)
        new_parts.append(f"{inverse_role} is {source_first}")
        rel_node["value"] = ", ".join(p for p in new_parts if p)
        appended += 1
    return appended


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Friends")
    parser.add_argument("--max_short_stay", type=int, default=3)
    parser.add_argument("--self_correction_window", type=int, default=2)
    parser.add_argument("--min_propagate_eps", type=int, default=2)
    parser.add_argument("--dry_run", action="store_true",
                        help="print summary without writing files")
    parser.add_argument("--scope", default="all",
                        help=("'all' (default) or a comma-separated list of "
                              "EP IDs to restrict the apply pass to"))
    args = parser.parse_args()

    ev_dir = PROCESSED_DIR / args.dataset / "intermediate" / "evolution"
    snap_path = ev_dir / "persona_snapshots.json"
    log_path = ev_dir / "reciprocity_repair_log.json"

    trees_path = PROCESSED_DIR / args.dataset / "intermediate" / "attribute_trees.json"
    main_chars = list(json.loads(trees_path.read_text()).keys())

    snaps = json.loads(snap_path.read_text())

    short_stay = detect_short_stay_hallucinations(
        snaps, main_chars,
        max_short_stay=args.max_short_stay,
        self_correction_window=args.self_correction_window,
    )
    propagations = detect_reciprocity_gaps(
        snaps, main_chars,
        min_propagate_eps=args.min_propagate_eps,
    )

    # Cross-filter: a Pass 2 propagation on (source, target) is meaningless
    # when Pass 1 just demoted that exact (source, target, role) claim — the
    # source side will end up listing it as ex- and the inverse role would
    # contradict itself.
    short_stay_keys = {
        (d["character"], d["partner_full"], d["role"], ep)
        for d in short_stay for ep in d["episodes"]
    }
    filtered_propagations: list[dict] = []
    for d in propagations:
        survivor_eps = [
            ep for ep in d["episodes"]
            if (d["source_character"], d["target_character"],
                d["source_role"], ep) not in short_stay_keys
        ]
        if not survivor_eps:
            continue
        d2 = dict(d)
        d2["episodes"] = survivor_eps
        filtered_propagations.append(d2)
    propagations = filtered_propagations

    print(f"Pass 1 — short-stay hallucinations: {len(short_stay)} runs")
    for d in short_stay:
        eps = d["episodes"]
        first, last = eps[0], eps[-1]
        print(f"  {d['character']:<18} {d['role']:<32} -> ex- "
              f"({len(eps)} eps {first}…{last}, "
              f"partner={d['partner_full'].split()[0]}, "
              f"kind={d['kind']})")

    print(f"\nPass 2 — reciprocity propagation: {len(propagations)} runs")
    for d in propagations:
        eps = d["episodes"]
        first, last = eps[0], eps[-1]
        print(f"  {d['target_character']:<18} += "
              f"'{d['inverse_role']} is {d['source_character'].split()[0]}' "
              f"({len(eps)} eps {first}…{last}, "
              f"because {d['source_character'].split()[0]} "
              f"already claims '{d['source_role']}')")

    # Filter by scope (optional)
    scope_eps: set[str] | None = None
    if args.scope and args.scope != "all":
        scope_eps = {e.strip() for e in args.scope.split(",") if e.strip()}
        print(f"\nApplying only to scope: {sorted(scope_eps)}")

    if args.dry_run:
        print("\n(dry-run: no files written)")
        return

    # Apply (with backup)
    backup_path = snap_path.with_suffix(snap_path.suffix + ".before_recip")
    snap_path.rename(backup_path)
    print(f"\nBackup written: {backup_path}")

    new_snaps = copy.deepcopy(snaps)
    applied_log: list[dict] = []

    # Pass 1 first (so propagation sees post-demotion state)
    for d in short_stay:
        if scope_eps is not None and not (set(d["episodes"]) & scope_eps):
            continue
        n = apply_short_stay_demotion(new_snaps, d)
        applied_log.append({**d, "n_entries_rewritten": n})
    # Pass 2
    for d in propagations:
        if scope_eps is not None and not (set(d["episodes"]) & scope_eps):
            continue
        n = apply_reciprocity_propagation(new_snaps, d)
        applied_log.append({**d, "n_entries_added": n})

    snap_path.write_text(json.dumps(new_snaps, ensure_ascii=False, indent=2))
    log_path.write_text(json.dumps(applied_log, ensure_ascii=False, indent=2))

    print(f"Updated:        {snap_path}")
    print(f"Repair log:     {log_path}")
    print(f"\nApplied {len(applied_log)} decisions total.")


if __name__ == "__main__":
    main()
