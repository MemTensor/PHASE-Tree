"""Assemble long-term ``phase_tree`` and ``dynamic_tree`` per dialogue.

Combines, per ``(dialogue, character)`` sample:

  * **identity**         — frozen, from ``attribute_trees.json``.
  * **persona@T-1**      — snapshot from the **previous** episode (resolved
                            from ``persona_snapshots.json``).
  * **session, moment**  — produced by :mod:`extract_long_term_moments`.

Two output JSON files per split (mirroring the short-term naming
convention):

  * ``<split>_long_term_phase_updated_trees.json``
        Full phase tree:  identity + persona@T-1 + session + moment.
        ``tree_id`` encodes the persona version used so downstream code can
        cite the exact snapshot (e.g. ``rachel_green::p32@S05E07``).

  * ``<split>_long_term_dynamic_trees.json``
        Slim dynamic tree:  identity + persona@T-1 only (session/moment
        cleared).  Counterpart to ``static_tree`` but with the time-evolved
        persona instead of the frozen ``p00``.

If a question_id has no matching moment entry (extraction failed or the
sample was skipped via ``--max_samples``), the assembler falls back to
empty session/moment dicts so the ``phase_tree`` remains structurally
valid.

Usage::

    python assemble_long_term_phase_tree.py --dataset Friends --split all
    python assemble_long_term_phase_tree.py --dataset Friends --split ood_test
"""

import argparse
import copy
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))

PROCESSED_DIR = PROJECT_ROOT / "LongEvoRoleBench" / "processed"

# ---------------------------------------------------------------------------
# Snapshot resolution (re-uses the same logic as extract_long_term_moments)
# ---------------------------------------------------------------------------

def _ep_ord(season: int, episode: int) -> int:
    return int(season) * 100 + int(episode)


def _ep_ord_tag(ep_tag: str) -> int:
    m = re.match(r"S(\d+)E(\d+)", ep_tag)
    return _ep_ord(int(m.group(1)), int(m.group(2))) if m else -1


def _resolve_prev_snapshot(
    snapshots: dict,
    base_tree: dict,
    char: str,
    season: int,
    episode: int,
) -> tuple[dict, str | None, str]:
    """Return ``(tree_at_T_minus_1, source_episode_tag, source_version)``.

    Selects the snapshot from the strictly-previous episode, or falls back
    to the base p00 tree for the very first appearance.
    """
    cur_ord = _ep_ord(season, episode)
    char_snaps = snapshots.get(char, {})

    chosen_ep: str | None = None
    chosen_ord = -1
    for ep_tag in char_snaps:
        o = _ep_ord_tag(ep_tag)
        if o < 0 or o >= cur_ord:
            continue
        if o > chosen_ord:
            chosen_ord = o
            chosen_ep = ep_tag

    if chosen_ep is None:
        return copy.deepcopy(base_tree), None, base_tree.get(
            "persona", {}).get("version", "p00")

    snap = char_snaps[chosen_ep]
    return copy.deepcopy(snap["tree"]), chosen_ep, snap.get(
        "version", "?")


# ---------------------------------------------------------------------------
# Tree assembly
# ---------------------------------------------------------------------------

_EMPTY_SESSION = {
    "learned_info": [],
    "attitude_shifts": {},
    "commitments": [],
    "stance_changes": [],
}
_EMPTY_MOMENT = {
    "emotion": "neutral",
    "emotion_intensity": 3,
    "scene_context": None,
}


def _normalise_persona_for_long_term(tree: dict, persona_source_ep: str | None,
                                     persona_version: str) -> dict:
    """Tag the assembled tree with long-term metadata so consumers can audit
    which snapshot was used.
    """
    out = copy.deepcopy(tree)
    out["dialogue_mode"] = "long_term"
    cid = out.get("character_id") or ""
    out["tree_id"] = (
        f"{cid}::{persona_version}@{persona_source_ep}"
        if persona_source_ep else f"{cid}::{persona_version}"
    )
    return out


def assemble_phase_tree(
    base_tree: dict,
    snapshot_tree: dict,
    session: dict,
    moment: dict,
    persona_source_ep: str | None,
    persona_version: str,
) -> dict:
    """Build the full phase tree (identity + persona@T-1 + session + moment).

    The identity layer is taken from the base tree (it is permanent), the
    persona layer is taken from the snapshot, and session/moment are filled
    from the moment-extraction output.
    """
    tree = copy.deepcopy(snapshot_tree)
    # Use the canonical identity from the base tree — snapshot identity may
    # have drifted via backstory addenda but we want consistency with
    # ``attribute_trees.json`` for downstream tooling.
    tree["identity"] = copy.deepcopy(base_tree.get("identity", {}))
    # Preserve the snapshot's backstory-addendum if present (it may diverge
    # from the base tree once evolve_persona appends life events).
    snap_backstory = snapshot_tree.get("identity", {}).get("backstory")
    if snap_backstory and snap_backstory != tree["identity"].get("backstory"):
        tree["identity"]["backstory"] = snap_backstory
    tree["session"] = session
    tree["moment"] = moment
    return _normalise_persona_for_long_term(
        tree, persona_source_ep, persona_version)


def assemble_dynamic_tree(
    base_tree: dict,
    snapshot_tree: dict,
    persona_source_ep: str | None,
    persona_version: str,
) -> dict:
    """Build the dynamic tree (identity + persona@T-1; no session/moment).

    Counterpart of the short-term ``static_tree`` but with the time-evolved
    persona substituted in.
    """
    return assemble_phase_tree(
        base_tree, snapshot_tree,
        copy.deepcopy(_EMPTY_SESSION),
        copy.deepcopy(_EMPTY_MOMENT),
        persona_source_ep, persona_version,
    ) if False else _normalise_persona_for_long_term(
        {
            **copy.deepcopy(snapshot_tree),
            "identity": copy.deepcopy(base_tree.get("identity", {})),
            "session": copy.deepcopy(_EMPTY_SESSION),
            "moment": copy.deepcopy(_EMPTY_MOMENT),
        },
        persona_source_ep, persona_version,
    )


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------

def process_split(
    intermediate_dir: Path, split_name: str,
    base_trees: dict, snapshots: dict,
    qid_meta: dict | None = None,
) -> tuple[int, int, int]:
    """Returns ``(n_phase, n_dynamic, n_skipped)``."""
    suffix_in = ""
    samples_path = intermediate_dir / f"{split_name}{suffix_in}.json"
    if not samples_path.exists():
        print(f"[{split_name}] samples not found: {samples_path}")
        return 0, 0, 0

    moments_path = (
        intermediate_dir / f"{split_name}_long_term_moments.json"
    )
    moments: dict = {}
    if moments_path.exists():
        with open(moments_path, "r", encoding="utf-8") as f:
            moments = json.load(f)
    else:
        print(f"[{split_name}] WARNING: moments file missing "
              f"({moments_path}); session/moment will be empty for all "
              f"samples.")

    with open(samples_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    qid_meta = qid_meta or {}
    phase_trees: dict[str, dict] = {}
    dynamic_trees: dict[str, dict] = {}
    skipped = 0

    for s in samples:
        role = s.get("role")
        qid = s.get("question_id")
        season = s.get("_season")
        episode = s.get("_episode")
        # Backfill from the central index when missing.
        if (season is None or episode is None) and qid in qid_meta:
            meta = qid_meta[qid]
            season = season if season is not None else meta.get("_season")
            episode = episode if episode is not None else meta.get("_episode")
        base_tree = base_trees.get(role)
        if base_tree is None or season is None or episode is None:
            skipped += 1
            continue

        snap_tree, src_ep, src_ver = _resolve_prev_snapshot(
            snapshots, base_tree, role, int(season), int(episode))

        moment_record = moments.get(qid)
        if moment_record:
            session = moment_record.get("session") or copy.deepcopy(
                _EMPTY_SESSION)
            moment = moment_record.get("moment") or copy.deepcopy(
                _EMPTY_MOMENT)
        else:
            session = copy.deepcopy(_EMPTY_SESSION)
            moment = copy.deepcopy(_EMPTY_MOMENT)

        phase_trees[qid] = assemble_phase_tree(
            base_tree, snap_tree, session, moment, src_ep, src_ver)
        dynamic_trees[qid] = assemble_dynamic_tree(
            base_tree, snap_tree, src_ep, src_ver)

    phase_path = (
        intermediate_dir / f"{split_name}_long_term_phase_updated_trees.json"
    )
    dynamic_path = (
        intermediate_dir / f"{split_name}_long_term_dynamic_trees.json"
    )
    phase_path.write_text(json.dumps(
        phase_trees, ensure_ascii=False, indent=2))
    dynamic_path.write_text(json.dumps(
        dynamic_trees, ensure_ascii=False, indent=2))
    print(f"[{split_name}] phase_tree:   {len(phase_trees)} -> {phase_path}")
    print(f"[{split_name}] dynamic_tree: {len(dynamic_trees)} -> "
          f"{dynamic_path}")
    if skipped:
        print(f"[{split_name}] skipped {skipped} samples (no metadata / "
              f"unknown role)")
    return len(phase_trees), len(dynamic_trees), skipped


def _merge_split_files(intermediate_dir: Path, suffix: str) -> None:
    merged: dict = {}
    for split in ["train", "random_test", "ood_test"]:
        p = intermediate_dir / f"{split}{suffix}.json"
        if not p.exists():
            print(f"  [merge {suffix}] {split} not found, skipping")
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged.update(data)
        print(f"  [merge {suffix}] {split}: {len(data)} entries")
    out = intermediate_dir / f"all_dialogues{suffix}.json"
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    print(f"  Merged -> {out}")


def main():
    ap = argparse.ArgumentParser(
        description="Assemble long-term phase_tree and dynamic_tree per "
                    "dialogue using the time-evolved persona snapshot.")
    ap.add_argument("--dataset", type=str, required=True,
                    choices=["Friends", "TheOffice", "StarTrek_TNG", "HPD"])
    ap.add_argument("--split", type=str, required=True,
                    choices=["train", "random_test", "ood_test", "all"])
    args = ap.parse_args()

    intermediate_dir = PROCESSED_DIR / args.dataset / "intermediate"
    evolution_dir = intermediate_dir / "evolution"

    base_trees = json.loads(
        (intermediate_dir / "attribute_trees.json").read_text())
    snap_path = evolution_dir / "persona_snapshots.json"
    if not snap_path.exists():
        raise FileNotFoundError(
            f"persona_snapshots.json not found at {snap_path}")
    snapshots = json.loads(snap_path.read_text())

    # Build a question_id -> temporal metadata index from all_dialogues.json
    # so split files (which may have stripped these fields) can still be
    # processed.
    #
    # HPD samples carry ``_book``/``_position`` ("Book1-chapter2") rather
    # than ``_season``/``_episode``.  Normalise here so the downstream
    # snapshot lookup keyed by S##E## tags works for all datasets.
    qid_meta: dict = {}
    all_dialogues_path = intermediate_dir / "all_dialogues.json"
    if all_dialogues_path.exists():
        with open(all_dialogues_path, "r", encoding="utf-8") as f:
            for s in json.load(f):
                qid = s.get("question_id")
                if not qid:
                    continue
                season = s.get("_season")
                episode = s.get("_episode")
                if season is None:
                    season = s.get("_book")
                if episode is None:
                    pos = s.get("_position")
                    if pos:
                        m = re.search(r"chapter(\d+)", pos, re.IGNORECASE)
                        if m:
                            episode = int(m.group(1))
                scene_id = s.get("_scene_id") or s.get("_session_id")
                qid_meta[qid] = {
                    "_season": season,
                    "_episode": episode,
                    "_scene_id": scene_id,
                }

    splits = (
        ["train", "random_test", "ood_test"]
        if args.split == "all" else [args.split]
    )
    for split_name in splits:
        process_split(
            intermediate_dir, split_name, base_trees, snapshots, qid_meta,
        )

    if args.split == "all":
        _merge_split_files(
            intermediate_dir, "_long_term_phase_updated_trees")
        _merge_split_files(
            intermediate_dir, "_long_term_dynamic_trees")


if __name__ == "__main__":
    main()
