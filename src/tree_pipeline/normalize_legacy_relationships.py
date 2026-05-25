"""One-shot normaliser for legacy ``<role> are <list>`` and bare-name
relationship entries.

Two patterns observed in early-season Monica snapshots cause downstream parse
ambiguity:

1. ``close friends are Chandler, Joey, Phoebe`` — uses plural ``are`` and a
   comma list, so any consumer that splits on ``,`` then matches ``<role>
   is <Name>`` either drops Joey/Phoebe entirely or treats them as
   continuation of "close friends are Chandler".
2. ``boyfriend is Chandler, Joey, Phoebe`` — singular ``is`` with a
   continuation comma list (S05E01–S05E04 transitional artefact).  A naive
   reader could misinterpret Joey/Phoebe as additional boyfriends.

This script rewrites both patterns to canonical ``friend is <Name>`` /
``<role-singular> is <Name>`` per-entry form.  It is safe to run multiple
times (idempotent).

Usage::

    python -m tree_pipeline.normalize_legacy_relationships --dataset Friends --dry_run
    python -m tree_pipeline.normalize_legacy_relationships --dataset Friends
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tree_pipeline.evolve_persona import (  # type: ignore
    _split_relationships_paren_aware,
)

PROCESSED_DIR = PROJECT_ROOT / "phase_tree_data" / "processed"

# Plural -> singular (used to repaint "close friends are X, Y" → "close friend is X, close friend is Y").
PLURAL_TO_SINGULAR = {
    "close friends": "close friend",
    "best friends": "best friend",
    "friends": "friend",
    "ex-friends": "ex-friend",
    "ex-boyfriends": "ex-boyfriend",
    "ex-girlfriends": "ex-girlfriend",
    "co-parents": "co-parent",
    "siblings": "sibling",
    "brothers": "brother",
    "sisters": "sister",
    "parents": "parent",
    "children": "child",
    "sons": "son",
    "daughters": "daughter",
    "exes": "ex-partner",
    "roommates": "roommate",
}

_BARE_NAME = re.compile(
    r"^[A-Z][\w\-']+(?:\s+[A-Z][\w\-']+)*\s*$"
)


def _is_bare_name(s: str) -> bool:
    return bool(_BARE_NAME.match(s.strip()))


def _expand_plural_are(entry: str) -> list[str] | None:
    """If ``entry`` is ``<role> are <Name>`` (plural form), return a list with
    a single ``<singular-role> is <Name>`` rewritten entry.  Otherwise None.
    """
    m = re.match(r"^([\w\- ]+?)\s+are\s+(.+)$", entry, re.I)
    if not m:
        return None
    role = m.group(1).strip().lower()
    name_part = m.group(2).strip()
    sing = PLURAL_TO_SINGULAR.get(role)
    if sing is None:
        return None
    if not _is_bare_name(name_part):
        return None
    return [f"{sing} is {name_part}"]


# Preposition-based current-relationship synonyms: "engaged to X" → "fiancé is X"
PREPOSITION_REWRITE: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^engaged\s+to\s+(.+)$", re.I), "fiancé is {0}"),
    (re.compile(r"^married\s+to\s+(.+)$", re.I), "spouse is {0}"),
    (re.compile(r"^newly[-\s]married\s+to\s+(.+)$", re.I), "spouse is {0}"),
]


def _apply_preposition_rewrite(entry: str) -> str | None:
    for pat, tmpl in PREPOSITION_REWRITE:
        m = pat.match(entry)
        if not m:
            continue
        # First name token only — drop trailing parenthetical context.
        rest = m.group(1).strip()
        return tmpl.format(rest)
    return None


def _dedupe_after_normalise(parts: list[str]) -> tuple[list[str], list[str]]:
    """Remove duplicate (role, primary-name) entries that may result from
    rewriting e.g. ``engaged to Chandler`` into ``fiancé is Chandler`` when
    the raw value already had ``fiancé is Chandler``.
    """
    out: list[str] = []
    seen: set[tuple[str, str]] = set()
    notes: list[str] = []
    for entry in parts:
        m = re.match(r"^([\w\- ]+?)\s+is\s+([A-Z][\w\-']+)", entry)
        if not m:
            out.append(entry)
            continue
        key = (m.group(1).strip().lower(), m.group(2).strip().lower())
        if key in seen:
            notes.append(f"dedupe: dropped duplicate '{entry}'")
            continue
        seen.add(key)
        out.append(entry)
    return out, notes


def normalise_value(value: str) -> tuple[str, list[str]]:
    """Normalise a relationships string.

    Returns ``(new_value, change_log)``.  Three rewrites are applied:

    * plural ``<role> are <Name>`` → singular ``<role-singular> is <Name>``
    * bare-name continuations → ``friend is <Name>``
    * prepositional ``engaged to <Name>`` / ``married to <Name>`` → canonical
      ``fiancé is <Name>`` / ``spouse is <Name>``

    Duplicate entries that result from the rewrites (e.g. when both
    ``engaged to Chandler`` and ``fiancé is Chandler`` already coexist) are
    deduplicated in a final pass.
    """
    if not value:
        return value, []
    parts = _split_relationships_paren_aware(value)
    if not parts:
        return value, []

    new_parts: list[str] = []
    changes: list[str] = []
    for raw in parts:
        # Plural-are pattern (top-level entry, head of a list)
        expanded = _expand_plural_are(raw)
        if expanded:
            for x in expanded:
                new_parts.append(x)
                changes.append(f"plural-are: '{raw}' -> '{x}'")
            continue
        # Preposition-based current relationship
        rewritten = _apply_preposition_rewrite(raw)
        if rewritten is not None:
            new_parts.append(rewritten)
            changes.append(f"preposition: '{raw}' -> '{rewritten}'")
            continue
        # Standard "<role> is/are <X>" entry
        m = re.match(r"^([\w\- ]+?)\s+(?:is|are)\s+(.+)$", raw, re.I)
        if m:
            new_parts.append(raw)
            continue
        # Bare-name continuation
        if _is_bare_name(raw):
            replacement = f"friend is {raw}"
            new_parts.append(replacement)
            changes.append(f"bare-name: '{raw}' -> '{replacement}'")
            continue
        # Anything else — leave alone (e.g. "dated X", "dating Y")
        new_parts.append(raw)

    deduped, dedupe_notes = _dedupe_after_normalise(new_parts)
    changes.extend(dedupe_notes)
    return ", ".join(deduped), changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Friends")
    parser.add_argument("--dry_run", action="store_true",
                        help="print summary without writing files")
    args = parser.parse_args()

    ev_dir = PROCESSED_DIR / args.dataset / "intermediate" / "evolution"
    snap_path = ev_dir / "persona_snapshots.json"
    log_path = ev_dir / "legacy_relationship_normalize_log.json"

    trees_path = (
        PROCESSED_DIR / args.dataset / "intermediate" / "attribute_trees.json"
    )
    main_chars = list(json.loads(trees_path.read_text()).keys())

    snaps = json.loads(snap_path.read_text())
    new_snaps = copy.deepcopy(snaps)
    log_records: list[dict] = []
    total_changes = 0

    for c in main_chars:
        for ep, snap in new_snaps.get(c, {}).items():
            rel = snap["tree"]["persona"]["relationships"]
            old_val = rel.get("value", "")
            new_val, changes = normalise_value(old_val)
            if changes:
                rel["value"] = new_val
                total_changes += len(changes)
                log_records.append({
                    "character": c,
                    "episode": ep,
                    "before": old_val,
                    "after": new_val,
                    "changes": changes,
                })

    print(f"Total entries normalised: {total_changes}")
    print(f"Episodes affected:        {len(log_records)}")
    by_char: dict[str, int] = {}
    for r in log_records:
        by_char[r["character"]] = by_char.get(r["character"], 0) + 1
    print("\nPer-character episode count:")
    for c in main_chars:
        if c in by_char:
            print(f"  {c:<20} {by_char[c]:>4}")

    if args.dry_run:
        print("\n(dry-run: no files written)")
        # Show first 3 sample changes
        print("\nSample changes:")
        for r in log_records[:3]:
            print(f"  {r['character']} {r['episode']}:")
            for ch in r["changes"][:5]:
                print(f"    - {ch}")
        return

    backup_path = snap_path.with_suffix(snap_path.suffix + ".before_norm")
    snap_path.rename(backup_path)
    snap_path.write_text(json.dumps(new_snaps, ensure_ascii=False, indent=2))
    log_path.write_text(json.dumps(log_records, ensure_ascii=False, indent=2))

    print(f"\nBackup written: {backup_path}")
    print(f"Updated:        {snap_path}")
    print(f"Log:            {log_path}")


if __name__ == "__main__":
    main()
