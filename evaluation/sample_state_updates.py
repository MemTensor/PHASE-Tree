#!/usr/bin/env python3
"""Build the candidate pool of PHASE-Tree field updates for state validation.

Four long-dialogue corpora: Friends, TheOffice, HPD, StarTrek_TNG.

Sources
-------
- ``evolution_log.json`` (accepted LLM / auto field updates; missing for Friends)
- ``core_audit_log.json`` (accepted personality / speaking_style descriptor updates)
- Friends substitute: consecutive ``persona_snapshots.json`` version diffs

Output
------
- ``results/state_validation/updates_pool.jsonl`` — full candidate pool used by
  ``validate_state_updates_30pct.py``

Usage:
    python evaluation/sample_state_updates.py
    python evaluation/sample_state_updates.py --out results/state_validation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CORPORA = {
    "Friends": "Friends",
    "TheOffice": "The Office",
    "HPD": "Harry Potter",
    "StarTrek_TNG": "Star Trek",
}

# Seed used when regenerating the pool (must match validation scripts).
DEFAULT_SEED = 20260712

PERSONA_FIELDS = {
    "relationships",
    "occupation",
    "demographics",
    "hobbies",
    "behavioral_tendencies",
    "personality",
    "speaking_style",
    "backstory_addendum",
}

# Resolved at import; ``main()`` may override via --data-dir.
DATA = ROOT / "phase_tree_data" / "processed"


def _uid(*parts: str) -> str:
    raw = "|".join(parts)
    return "SV-" + hashlib.md5(raw.encode()).hexdigest()[:10]


def _episode_bucket(ep: str) -> str:
    """early / mid / late from season or book number."""
    m = re.match(r"S(\d+)", ep) or re.match(r"(?:Book|B)(\d+)", ep, re.I)
    if not m:
        # HPD sometimes uses plain episode labels; fall back mid
        return "mid"
    n = int(m.group(1))
    if n <= 2:
        return "early"
    if n <= 5:
        return "mid"
    return "late"


def _field_stratum(field: str) -> str:
    if field in {"personality", "speaking_style"}:
        return "core"
    if field == "relationships":
        return "relationships"
    if field == "behavioral_tendencies":
        return "behavioral"
    return "factual"  # occupation / demographics / hobbies / backstory


def load_session_archives(corpus: str) -> dict[str, dict]:
    """Map scene_id -> summary record across all character archives."""
    evo = DATA / corpus / "intermediate" / "evolution"
    by_id: dict[str, dict] = {}
    for path in evo.glob("*_session_archive.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # may be list or {sessions: [...]} or {character: [...]}
        if isinstance(data, dict):
            sessions = data.get("sessions") or data.get("archive") or None
            if sessions is None:
                # values might be lists
                for v in data.values():
                    if isinstance(v, list):
                        sessions = v
                        break
            if sessions is None:
                continue
        else:
            sessions = data
        for s in sessions:
            sid = s.get("scene_id") or s.get("session_id")
            if sid:
                by_id[sid] = s
    return by_id


def load_evolution_log(corpus: str) -> list[dict]:
    path = DATA / corpus / "intermediate" / "evolution" / "evolution_log.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("updates", [])


def load_core_audit(corpus: str) -> list[dict]:
    path = DATA / corpus / "intermediate" / "evolution" / "core_audit_log.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("audits", [])


def _persona_field_value(tree: dict, field: str):
    persona = (tree or {}).get("persona") or {}
    node = persona.get(field)
    if isinstance(node, dict):
        return node.get("value")
    return node


def reconstruct_friends_from_snapshots() -> list[dict]:
    """Diff consecutive version bumps in Friends persona_snapshots.json."""
    snap_path = DATA / "Friends" / "intermediate" / "evolution" / "persona_snapshots.json"
    with open(snap_path, encoding="utf-8") as f:
        snaps = json.load(f)

    events: list[dict] = []
    for character, episodes in snaps.items():
        # preserve chronological order
        ep_items = list(episodes.items())
        prev_version = None
        prev_tree = None
        prev_ep = None
        for ep, node in ep_items:
            version = node.get("version")
            tree = node.get("tree") or {}
            if prev_version is not None and version != prev_version:
                changes = {}
                for field in PERSONA_FIELDS:
                    if field == "backstory_addendum":
                        continue
                    old = _persona_field_value(prev_tree, field)
                    new = _persona_field_value(tree, field)
                    if old != new and new is not None:
                        changes[field] = {
                            "old_value": old,
                            "new_value": new,
                            "merge_type": "unknown_diff",
                            "consumed_session_ids": [],
                        }
                if changes:
                    events.append(
                        {
                            "character": character,
                            "episode": ep,
                            "from_version": prev_version,
                            "to_version": version,
                            "changes": changes,
                            "backstory_addendum": None,
                            "reasoning": "Reconstructed from Friends persona_snapshots version bump "
                            f"({prev_ep}/{prev_version} → {ep}/{version}).",
                            "sessions_consumed": 0,
                            "_source": "snapshot_diff",
                        }
                    )
            prev_version = version
            prev_tree = tree
            prev_ep = ep
    return events


def expand_to_field_updates(corpus: str, events: list[dict], source: str) -> list[dict]:
    rows = []
    for ev in events:
        character = ev.get("character", "")
        episode = ev.get("episode", "")
        changes = ev.get("changes") or {}
        for field, meta in changes.items():
            if not isinstance(meta, dict):
                continue
            old = meta.get("old_value")
            new = meta.get("new_value")
            if new is None:
                continue
            sid = meta.get("consumed_session_ids") or meta.get("evidence_session_ids") or []
            merge = meta.get("merge_type") or meta.get("merge_strategy") or "unknown"
            rows.append(
                {
                    "update_id": _uid(corpus, character, episode, field, str(new)[:80]),
                    "corpus": corpus,
                    "corpus_display": CORPORA[corpus],
                    "character": character,
                    "episode": episode,
                    "field": field,
                    "stratum": _field_stratum(field),
                    "time_bucket": _episode_bucket(episode),
                    "merge_type": merge,
                    "old_value": old,
                    "new_value": new,
                    "evidence_session_ids": list(sid),
                    "reasoning": ev.get("reasoning"),
                    "source": source,
                    "from_version": ev.get("from_version"),
                    "to_version": ev.get("to_version"),
                }
            )
    return rows


def expand_core_audits(corpus: str, audits: list[dict]) -> list[dict]:
    rows = []
    for a in audits:
        applied = a.get("applied") or []
        if not applied:
            continue
        character = a.get("character", "")
        episode = a.get("episode", "")
        field = a.get("field", "personality")
        old = a.get("old_value")
        new = a.get("new_value")
        evidence = []
        reasons = []
        for ap in applied:
            evidence.extend(ap.get("evidence_session_ids") or [])
            if ap.get("reasoning"):
                reasons.append(ap["reasoning"])
        rows.append(
            {
                "update_id": _uid(corpus, character, episode, field, "core", str(new)[:80]),
                "corpus": corpus,
                "corpus_display": CORPORA[corpus],
                "character": character,
                "episode": episode,
                "field": field,
                "stratum": "core",
                "time_bucket": _episode_bucket(episode),
                "merge_type": "core_audit",
                "old_value": old,
                "new_value": new,
                "evidence_session_ids": list(dict.fromkeys(evidence)),
                "reasoning": " | ".join(reasons) if reasons else a.get("reasoning"),
                "source": "core_audit",
                "from_version": None,
                "to_version": None,
                "applied_descriptors": applied,
            }
        )
    return rows


def attach_evidence(rows: list[dict], archives: dict[str, dict]) -> None:
    for r in rows:
        ev = []
        for sid in r.get("evidence_session_ids") or []:
            s = archives.get(sid)
            if s:
                ev.append(
                    {
                        "scene_id": sid,
                        "summary": s.get("summary") or s.get("text") or "",
                        "significance": s.get("significance"),
                        "affected_fields": s.get("affected_fields"),
                    }
                )
            else:
                ev.append({"scene_id": sid, "summary": "", "missing": True})
        r["evidence"] = ev


def build_pool() -> list[dict]:
    pool: list[dict] = []
    for corpus in CORPORA:
        archives = load_session_archives(corpus)
        if corpus == "Friends":
            events = reconstruct_friends_from_snapshots()
            rows = expand_to_field_updates(corpus, events, "snapshot_diff")
        else:
            events = load_evolution_log(corpus)
            # tag source
            for e in events:
                e.setdefault("_source", "evolution_log")
            rows = expand_to_field_updates(corpus, events, "evolution_log")
        core_rows = expand_core_audits(corpus, load_core_audit(corpus))
        all_rows = rows + core_rows
        attach_evidence(all_rows, archives)
        pool.extend(all_rows)
        print(
            f"{corpus}: evolve_fields={len(rows)} core={len(core_rows)} "
            f"archive_ids={len(archives)}"
        )
    return pool


def main() -> None:
    global DATA

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "phase_tree_data" / "processed",
        help="Processed PHASE-Tree data root",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "state_validation",
        help="Output directory for updates_pool.jsonl",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    DATA = args.data_dir
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    pool = build_pool()
    pool_path = out_dir / "updates_pool.jsonl"
    with pool_path.open("w", encoding="utf-8") as f:
        for row in pool:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"pool size: {len(pool)}")
    print(f"per corpus: {dict(Counter(u['corpus_display'] for u in pool))}")
    print(f"wrote {pool_path}")


if __name__ == "__main__":
    main()
