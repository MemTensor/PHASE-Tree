"""Evaluate and apply persona evolution based on accumulated session archives.

For each episode boundary, collects ``"active"`` session archive entries for a
character and asks an LLM whether the accumulated experiences warrant a
persona update.  The LLM returns per-field decisions (new value, merge type,
and the specific session IDs used as evidence).  The decision then passes
through several **hard** validators before being applied:

  * **scene_id whitelist** – consumed_session_ids must come from the active set
  * **resistance thresholds** – low / moderate / core fields require different
    amounts of evidence (and high-significance counts for core)
  * **field cooldown** – minimum number of episodes between two updates of
    the same field
  * **incremental safety** – non-replacement merges must keep ≥ 80 % of the
    previous value's length

Sessions are also subject to:

  * **expiration** – medium-significance entries unused for > 8 episodes are
    marked ``expired`` (high-significance never expire)
  * **archive truncation** – the LLM only sees all high entries plus the most
    recent 20 medium entries

Input
    ``LongEvoRoleBench/processed/<dataset>/intermediate/attribute_trees.json``
    ``LongEvoRoleBench/processed/<dataset>/intermediate/evolution/<Char>_session_archive.json``
    ``LongEvoRoleBench/processed/<dataset>/intermediate/all_dialogues.json``  (episode order)

Output
    ``LongEvoRoleBench/processed/<dataset>/intermediate/evolution/persona_snapshots.json``
    ``LongEvoRoleBench/processed/<dataset>/intermediate/evolution/evolution_log.json``
    Updates session archive files in-place (consumed / expired entries marked)

Usage::

    python evolve_persona.py --dataset Friends
    python evolve_persona.py --dataset Friends --test_episode S01E05
"""

import argparse
import asyncio
import copy
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED_DIR = PROJECT_ROOT / "LongEvoRoleBench" / "processed"

MAIN_CHARACTERS: list[str] = []  # populated at runtime from attribute_trees.json

# ---------------------------------------------------------------------------
# Validation parameters
# ---------------------------------------------------------------------------

# Minimum evidence requirements per resistance level.
#
# IMPORTANT: ``moderate`` and ``core`` thresholds are evaluated against the
# **lifetime pattern archive** (Track B) — i.e. all high-sig events ever
# produced for this character up to the current episode, regardless of
# whether those events have been previously consumed by low-resistance
# updates.  A high-sig session is evidence for *both* a one-off factual
# update (relationships) AND for slow-moving pattern detection (personality)
# — consumption for the former does not invalidate it for the latter.
THRESHOLDS = {
    "low": {
        "min_episodes": 1,
        "min_high": 0,
        "min_high_or_2_medium": True,  # 1 high OR 2 medium
        "cooldown": 2,
        "evidence_track": "recent",
    },
    "moderate": {
        "min_episodes": 3,
        "min_high": 0,
        "min_high_or_2_medium": False,
        "cooldown": 3,
        "evidence_track": "lifetime",
    },
    "core": {
        # Relaxed from 24 ep / 8 high (which was effectively unreachable
        # because consumed-status filtering shrank the visible window) to
        # 16 ep / 6 high evaluated against the full lifetime arc.
        "min_episodes": 16,
        "min_high": 6,
        "min_high_or_2_medium": False,
        "cooldown": 16,
        "evidence_track": "lifetime",
    },
}

MEDIUM_EXPIRY_AGE = 8     # episodes
ARCHIVE_TRUNCATE_MEDIUM = 20  # most recent N medium entries
INCREMENTAL_MIN_RATIO = 0.80  # incremental updates must keep ≥ 80% of old length
INCREMENTAL_MIN_WORD_RECALL = 0.50  # incremental updates must preserve ≥ 50% of old's distinctive words

# ---------------------------------------------------------------------------
# Decay configuration (used by per-episode decay pass + legacy decay script)
# ---------------------------------------------------------------------------

ROMANTIC_TRANSIENT = {
    "boyfriend",
    "girlfriend",
    "partner",
    "lover",
    "romantic interest",
    "date",
    "dating",
    "love interest",
    # Stable but still subject to fresh-evidence decay
    "secret romantic encounter",
    "secret girlfriend",
    "secret boyfriend",
    "one-time encounter",
    "former one-time encounter",
    "on-off girlfriend",
    "on-off boyfriend",
    "friend with romantic tension",
    "ex-romantic interest",
    "crush",
    "affair",
    "fling",
    "hookup",
    "seeing",
    "romantic partner",
}

# Roles that, when decayed, should be expressed as "former ..." rather than
# "ex-..." — fits the natural-language semantics for one-shot encounters.
DECAY_FORMER_PREFIX = {
    "secret romantic encounter",
    "one-time encounter",
    "friend with romantic tension",
}

# Inherently *unilateral* romantic roles — they describe a feeling/encounter
# that does not require the other party to acknowledge it.  Rule B
# (inter-main reciprocity) MUST NOT fire on these; only Rule A (archive
# freshness for non-main partners) applies.
UNILATERAL_ROMANTIC_ROLES = {
    "friend with romantic tension",
    "secret romantic encounter",
    "secret girlfriend",
    "secret boyfriend",
    "one-time encounter",
    "former one-time encounter",
    "crush",
    "affair",
    "fling",
    "hookup",
}

PROTECTED_CURRENT = {
    "husband", "wife", "spouse",
    "fiancé", "fiance", "fiancée", "fiancee",
    "wife-to-be", "husband-to-be",
    "co-parent", "baby's father", "baby's mother",
}

DECAY_THRESHOLD_NON_MAIN_EP = 6   # Rule A: archive freshness for non-main partners
DECAY_RULE_B_LONG_SILENCE = 10    # Rule B2: inter-main long-silence threshold

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SHOW_NAMES: dict[str, str] = {
    "Friends": "Friends",
    "TheOffice": "The Office",
    "StarTrek_TNG": "Star Trek: The Next Generation",
    "HPD": "Harry Potter",
}

SYSTEM_PROMPT: str = ""  # populated at runtime via _build_system_prompt(dataset)

_SYSTEM_PROMPT_TEMPLATE = """\
You are a character psychologist specialising in long-term personality \
development for the TV show "{show}". Given a character's current persona \
and their recent experiences (session archive), decide whether the \
accumulated experiences warrant updating any persona field.

## Field semantics — IMPORTANT

Use each persona field for its intended purpose. Do NOT cross-pollute fields:

- **occupation** = job / profession / career (e.g. "paleontologist", \
"waitress", "office worker"). NEVER put pets, hobbies, family roles or \
romantic statuses here.
- **relationships** = social bonds (family, friends, romantic partners). \
List each person at most ONCE with their CURRENT role (e.g. if Barry is the \
ex-fiancé, do not also list him as ex-boyfriend — pick the most precise role). \
**Direction matters!** The format is ``"<the OTHER person's role to YOU> \
is <Name>"``. Examples for Monica: ``"nephew is Ben"`` (Ben is Monica's \
nephew) — NEVER ``"aunt is Ben"`` (which would imply Monica's aunt is named \
Ben). For a daughter, write ``"daughter is X"``; for a parent, ``"father \
is X"``. The role describes how the named person relates to the persona \
character, not vice versa.
- **hobbies** = leisure activities the character enjoys.
- **demographics** = age, location, life stage.
- **personality** = core inner traits (adjectives describing the person).
- **speaking_style** = how they talk.
- **behavioral_tendencies** = recurring behaviour patterns.

When an event affects multiple fields (e.g. Ross gives up Marcel → it \
affects relationships AND hobbies), update all relevant fields in the same \
decision so the persona stays internally consistent.

## Two-track evidence views

You will see TWO archive sections in the user message:

- **Track A — RECENT EVIDENCE** (active high-sig + recent medium events that \
have happened since the last persona update). This is the primary view for \
**low-resistance factual fields** (relationships, occupation, demographics, \
hobbies). These fields update on specific recent events.

- **Track B — LIFETIME PATTERN ARCHIVE** (every high-significance event ever \
produced for this character up to the current episode, in chronological \
order — including events that have already been cited for previous low-field \
updates). This is the primary view for **moderate / core fields** \
(behavioral_tendencies, personality, speaking_style).  A high-significance \
event is evidence for BOTH a one-off factual update AND for slow-moving \
pattern detection — its appearance in this section does NOT mean it is \
"reusable" for low-field updates; it is shown so you can detect long-arc \
drift in deep traits.

When proposing an update, draw evidence from the appropriate track for the \
field's resistance level.

## Three-tier resistance system

Each persona field has a **resistance** level that controls how easily it \
changes:

- **low** (factual fields like occupation, relationships, demographics, \
hobbies):
  - Use Track A.
  - Update when there is at least 1 high-significance event OR 2 medium \
events providing CLEAR FACTUAL EVIDENCE.
  - Evidence must describe an ACTUAL EVENT, not an intention or plan.
  - Example: "considering breaking up" → not enough; "broke up with Paolo" → \
yes.

- **moderate** (behavioural patterns like behavioral_tendencies):
  - Use Track B.
  - Update when consistent evidence spans **at least 3 different episodes** \
in Track B pointing in the same direction.
  - The events do not need to be recent — what matters is a stable pattern \
across the lifetime arc.

- **core** (deep traits like personality, speaking_style):
  - Use Track B.
  - Update when the lifetime arc shows CONSISTENT directional drift across \
**≥ 16 distinct episodes** with **≥ 6 high-significance events** all \
reinforcing the same refinement.
  - **Prefer INCREMENTAL refinements** (adding nuance, replacing one outdated \
descriptor with a more accurate one) over wholesale REPLACEMENT of a \
character's personality. Personalities deepen and mature; they rarely \
invert.
  - Examples of legitimate core drift in long-running narratives:
    * A sheltered character becoming visibly self-reliant after years of \
independence and major life transitions.
    * A defensive humorist softening into earnest emotional expression \
after marriage / parenthood / sustained vulnerability.
    * A career-driven perfectionist learning to balance control with \
partnership after years of co-living.
  - Do NOT update core fields on a single dramatic episode, no matter how \
high-significance. Look for the long-arc pattern.
  - When in doubt, default to INCREMENTAL append rather than REPLACEMENT.

## Core-field re-examination trigger (IMPORTANT)

Each persona field carries a ``times_updated`` counter and a \
``last_updated`` tag.  When you see a "DRIFT TRIGGER" entry in the user \
message it means the *behavioural surface* (``behavioral_tendencies``) has \
already been updated several times in a consistent direction while the \
underlying *deep trait* (``personality`` or ``speaking_style``) has not \
been revisited.

When such a trigger is present AND the core-tier threshold is MET, you \
**SHOULD explicitly check** whether any single adjective / descriptor in \
the current deep-field value has been **directly contradicted** by the \
accumulated behavioural drift documented in Track B.

- If yes → propose an INCREMENTAL refinement: keep all still-accurate \
descriptors verbatim, replace ONE outdated descriptor (e.g. "passive" → \
"selectively assertive", "anxious" → "earnest but more grounded") OR \
append a small qualifier nuance.  Cite ≥ 6 scene_ids drawn from ≥ 16 \
distinct episodes.
- If no → keep the deep field unchanged.  Drift in surface behaviour does \
NOT automatically imply drift in deep traits (a character may keep their \
core temperament while accumulating coping strategies).

This trigger is NOT a mandate to update — it is a structured nudge to \
*examine*.  Spurious or weakly-supported core changes will be rejected.

## Citing evidence in ``consumed_session_ids``

For **low** fields, cite scene IDs from Track A only (recent active events).

For **moderate** / **core** fields, cite scene IDs from Track B — these may \
include events that were already cited as evidence for earlier low-field \
updates.  That re-citation is permitted and does NOT double-consume the \
event; the system understands that one significant event can serve as \
evidence for both an immediate factual change AND the long-arc pattern.

Always cite at least 3 IDs for moderate updates and at least 6 IDs for core \
updates, drawn from at least 3 / 16 distinct episodes respectively.

## Merge strategy: CONFLICT-BASED

For each field you decide to update, classify the relationship between the \
new evidence and the existing value:

**Case 1 – NO CONFLICT (additive)**: New information is compatible with all \
existing facts. Use **incremental** merge: KEEP ALL existing content, append \
or weave in the new info.

  - Old: "best friend is Monica, friend is Ross"
  - Event: "Rachel started dating Ross"
  - New: "best friend is Monica, boyfriend is Ross"
  - merge_type: "incremental"

**Case 2 – CONFLICT (mutually exclusive)**: New information makes an existing \
specific item logically untrue. Use **replacement** merge: replace ONLY the \
conflicting item, keep everything else verbatim.

  - Old: "boyfriend is Paolo, best friend is Monica"
  - Event: "Rachel and Paolo broke up"  ← EXPLICIT contradiction
  - New: "ex-boyfriend is Paolo, best friend is Monica"
  - merge_type: "replacement"

  - Old: "waitress"
  - Event: "Rachel got hired at Bloomingdale's as a buyer"  ← EXPLICIT contradiction
  - New: "buyer at Bloomingdale's"
  - merge_type: "replacement"

## CRITICAL — Distinguish what the evidence ACTUALLY shows

Session summaries describe events from the focal character's perspective. \
Be careful NOT to over-interpret:

- **"X reveals/expresses feelings for Y"** does NOT mean X confessed to Y \
directly. The disclosure is often to a THIRD PARTY (a friend, in private). \
Update X-Y's relationship ONLY if the evidence EXPLICITLY shows mutual \
romantic interaction (a kiss, going on a date, agreeing to date, etc.).
- **"X impersonates Y in a conversation with Z"** does NOT make X and Z a \
couple. The romantic dynamic exists between Y and Z; X is a stand-in.
- **"X helps Y break up with Z"** does NOT make X and Z a couple. X is just a \
facilitating friend.
- **"X comforts Y after breakup / vulnerable moment with Y"** does NOT \
establish a new romantic relationship between X and Y. Friends comfort \
friends.
- A character's INNER FEELINGS (longing, harboring feelings, jealousy, \
attraction) do NOT count as a relationship status change. The status only \
changes when ACTIONS occur (mutual kiss, date, declaration, breakup, etc.).

If you are unsure whether an event constitutes a real relationship status \
change, do NOT update.

## ABSOLUTELY CRITICAL — NO INFERENCE FROM ABSENCE

**NEVER remove or downgrade an existing fact just because the recent sessions \
don't mention it.** Absence of mention is NOT evidence of contradiction. \
A fact stays in the persona until the dialogue EXPLICITLY shows it changed.

  WRONG REASONING:
    Old: "boyfriend is Paolo"
    Sessions don't mention Paolo → infer they broke up → change to "ex-boyfriend"
    ❌ This is forbidden. No mention ≠ contradiction.

  CORRECT REASONING:
    Old: "boyfriend is Paolo"
    Session: "Paolo cheated and Rachel told him to leave"  ← EXPLICIT
    → change to "ex-boyfriend is Paolo"
    ✅ Direct dialogue evidence of the change.

If you are unsure whether the dialogue explicitly contradicts a fact, do NOT \
change that fact.

CRITICAL: NEVER delete information that is not contradicted by the new event.

## CRITICAL — Romantic-status downgrades require an EXPLICIT breakup

Changing ``"<role> is X"`` to ``"ex-<role> is X"`` (girlfriend → ex-girlfriend, \
boyfriend → ex-boyfriend, wife → ex-wife, fiancé → ex-fiancé, etc.) is a \
**status change**.  It is allowed ONLY when the sessions you cite explicitly \
describe an actual breakup, divorce, calling-it-off, or split — not arguments, \
distance, jealousy, fear of commitment, fights, hurt feelings, or temporary \
silence.  If the sessions only show conflict or doubt, KEEP the existing role \
("girlfriend is X", not "ex-girlfriend is X") and wait for explicit breakup \
evidence before downgrading.

## DO add new partners and reconciliations as they appear

The "no inference from absence" rule above bars REMOVING facts.  It does NOT \
discourage you from APPENDING new factual relationship entries.  When a \
session explicitly introduces a new dating partner, even for a short arc, \
or shows reconciliation with an estranged friend, you SHOULD update \
``relationships`` to include them.  Examples:

- A session says "Rachel introduces her new date Russ" → add ``dating Russ``  \
or ``boyfriend is Russ`` (whichever the dialogue supports).
- A session says "Rachel went on a date with Jean-Claude Van Damme" → add \
``dated Jean-Claude Van Damme``.
- A session says "Joey starred opposite Erica on Days of Our Lives and \
revealed he isn't really Drake, ending their fling" → add ``ex-girlfriend \
is Erica``.
- A session says "Rachel and Mindy made up after years of estrangement" → \
change ``ex-best friend is Mindy`` to ``friend is Mindy``.

When in doubt about whether a new partner is "important enough", err toward \
including them (factual fields like ``relationships`` are low-resistance).  \
The validators will reject genuinely unsupported claims.

## Evidence reporting

For each field change, list the SPECIFIC scene_ids (NOT indices) of the \
sessions that support that particular change. Only cite scene_ids that \
appear in the provided session archive.

**Scene-ID format MUST be exactly as shown in the archive** (lowercase, \
e.g. ``s01_e12_c10``). Do NOT invent variants like ``S01E12_c10`` or \
``Scene 10``; copy the literal string after ``scene=``.

## Output format — valid JSON only, no markdown fences

{
  "should_update": true or false,
  "reasoning": "Brief explanation of why this update is or is not warranted",
  "changes": {
    "field_name": {
      "new_value": "complete new value of the field",
      "merge_type": "incremental" or "replacement",
      "consumed_session_ids": ["scene_id_1", "scene_id_2", ...]
    }
  },
  "backstory_addendum": "One sentence to append to identity.backstory summarising what happened, or null"
}

If should_update is false, changes must be {} and backstory_addendum must be \
null.

If you can update some fields but not others (e.g. evidence is sufficient \
for occupation but not for personality), include only the fields you can \
justify. Do NOT include weakly-supported changes."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_system_prompt(dataset: str) -> str:
    show = SHOW_NAMES.get(dataset, dataset)
    return _SYSTEM_PROMPT_TEMPLATE.replace("{show}", show)


def _parse_chapter_from_position(position: str | None) -> int | None:
    """Extract chapter number from HPD _position like 'Book1-chapter2' → 2."""
    if not position:
        return None
    m = re.search(r"chapter(\d+)", position, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _episode_to_int(ep_tag: str) -> int:
    """Convert ``S01E05`` into a comparable integer (season*100 + episode)."""
    m = re.match(r"S(\d+)E(\d+)", ep_tag)
    if not m:
        return -1
    return int(m.group(1)) * 100 + int(m.group(2))


def _format_persona_for_prompt(persona: dict) -> str:
    """Format persona fields with their resistance levels."""
    lines = []
    for field, val in persona.items():
        if field == "version":
            continue
        if isinstance(val, dict) and "value" in val:
            r = val.get("resistance", "?")
            last = val.get("last_updated_at")
            n_upd = val.get("update_count", 0)
            tag = f", last_updated={last}" if last else ""
            count_tag = f", times_updated={n_upd}" if n_upd else ""
            lines.append(
                f'- {field} (resistance={r}{tag}{count_tag}): "{val["value"]}"'
            )
        elif isinstance(val, str):
            lines.append(f'- {field}: "{val}"')
    return "\n".join(lines)


def _format_archive_for_prompt(entries: list[dict]) -> str:
    lines = []
    for e in entries:
        ep_tag = f"S{e['season']:02d}E{e['episode']:02d}"
        sig = e["significance"]
        lines.append(f"- [{ep_tag}, scene={e['scene_id']}] ({sig}) {e['summary']}")
    return "\n".join(lines)


def _truncate_archive(active: list[dict],
                      max_medium: int = ARCHIVE_TRUNCATE_MEDIUM) -> list[dict]:
    """Track A — Keep all active high entries plus the most recent N active
    medium entries.  This is the *recent evidence* view used for
    low-resistance factual fields.
    """
    high = [e for e in active if e["significance"] == "high"]
    medium = [e for e in active if e["significance"] == "medium"]
    medium_recent = sorted(
        medium, key=lambda e: (e["season"], e["episode"])
    )[-max_medium:]
    combined = sorted(high + medium_recent,
                      key=lambda e: (e["season"], e["episode"]))
    return combined


def _build_lifetime_archive(
    full_archive: list[dict],
    season: int,
    episode: int,
) -> list[dict]:
    """Track B — return ALL high-significance events ever produced for this
    character up to (and including) the given episode, regardless of whether
    they have been previously consumed by low-resistance updates or marked
    expired.  Sorted chronologically.

    Rationale: a high-sig session is evidence for both one-off factual
    updates (relationships) and slow-moving pattern detection (personality);
    consumption for the former does not invalidate it for the latter.
    """
    entries = [
        e for e in full_archive
        if e.get("significance") == "high"
        and (e["season"], e["episode"]) <= (season, episode)
    ]
    return sorted(entries, key=lambda e: (e["season"], e["episode"]))


def _evidence_summary(track_b: list[dict]) -> dict:
    """Summarise the lifetime pattern archive for the LLM:
    how many unique episodes, how many high-sig events, and how this
    compares to the moderate (3 ep) and core (16 ep / 6 high) thresholds.
    """
    eps = {(e["season"], e["episode"]) for e in track_b}
    n_high = len(track_b)
    return {
        "unique_episodes": len(eps),
        "high_events": n_high,
        "moderate_threshold_met": len(eps) >= THRESHOLDS["moderate"]["min_episodes"],
        "core_threshold_met": (
            len(eps) >= THRESHOLDS["core"]["min_episodes"]
            and n_high >= THRESHOLDS["core"]["min_high"]
        ),
    }


# Field-pair → (deep field, surface field) used by the drift trigger.
# When the surface field has been updated significantly more times than the
# deep one, the LLM is prompted to re-examine whether the deep field needs
# a refinement.
_DRIFT_PAIRS: list[tuple[str, str]] = [
    ("personality", "behavioral_tendencies"),
    ("speaking_style", "behavioral_tendencies"),
]
_DRIFT_TRIGGER_THRESHOLD = 3


def _drift_signal_block(persona: dict) -> str:
    """Produce a short text block that flags any deep field whose paired
    surface field has been updated repeatedly since its last touch.  This is
    a generic signal — applies to any character whose behavioural pattern is
    drifting consistently — and is meant to nudge the LLM into examining
    whether the deep trait label is still accurate.
    """
    lines: list[str] = []
    for deep, surface in _DRIFT_PAIRS:
        deep_field = persona.get(deep)
        surf_field = persona.get(surface)
        if not isinstance(deep_field, dict) or not isinstance(surf_field, dict):
            continue
        deep_last = deep_field.get("last_updated_at")
        surf_last = surf_field.get("last_updated_at")
        surf_count = int(surf_field.get("update_count", 0) or 0)
        if surf_count < _DRIFT_TRIGGER_THRESHOLD:
            continue
        if deep_last and surf_last and (
            _episode_to_int(surf_last) <= _episode_to_int(deep_last)
        ):
            continue
        lines.append(
            f"- DRIFT TRIGGER: '{surface}' has been updated "
            f"{surf_count} time(s) (last={surf_last or 'n/a'}) while "
            f"'{deep}' has stayed unchanged since "
            f"{deep_last or 'p00 (never updated)'}.  Examine whether the "
            f"accumulated behavioural drift now contradicts any single "
            f"descriptor in '{deep}'.  If yes, propose an INCREMENTAL "
            f"refinement (replace ONE outdated descriptor with a more "
            f"precise one); if no, leave '{deep}' unchanged."
        )
    if not lines:
        return ""
    return "\n## Deep-field drift signals\n" + "\n".join(lines) + "\n"


def _expire_old_medium(archive: list[dict], current_ep: str,
                       max_age: int = MEDIUM_EXPIRY_AGE):
    """Mark medium entries unused for too long as expired (in-place)."""
    cur_int = _episode_to_int(current_ep)
    for e in archive:
        if e.get("status") != "active":
            continue
        if e["significance"] == "high":
            continue
        ep_tag = f"S{e['season']:02d}E{e['episode']:02d}"
        if cur_int - _episode_to_int(ep_tag) > max_age:
            e["status"] = "expired"
            e["expired_at_episode"] = current_ep


def _parse_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


# ---------------------------------------------------------------------------
# Hard validation
# ---------------------------------------------------------------------------

_SCENE_RE = re.compile(r"s(\d{1,2})_?e(\d{1,2})_?c(\d{1,3})", re.IGNORECASE)

# Priority table for romantic / family role specificity.
# Higher value = more specific / more permanent role.
_ROLE_PRIORITY = {
    # current romantic
    "spouse": 100, "wife": 100, "husband": 100,
    "fiancé": 90, "fiancee": 90, "fiance": 90,
    "boyfriend": 80, "girlfriend": 80,
    "partner": 75,
    "dating": 60, "dated": 55,
    # ex / former romantic
    "ex-spouse": 70, "ex-wife": 70, "ex-husband": 70,
    "ex-fiancé": 65, "ex-fiancee": 65, "ex-fiance": 65,
    "ex-boyfriend": 50, "ex-girlfriend": 50,
    "ex-partner": 45,
    # friendship intensity
    "best friend": 40, "close friend": 35,
    "friend": 30,
    # family
    "mother": 95, "father": 95,
    "son": 90, "daughter": 90,
    "brother": 85, "sister": 85,
    "twin sister": 88, "twin brother": 88,
    "grandmother": 82, "grandfather": 82,
}

# Match leading role label like "ex-fiancé is", "best friend is",
# "former best friend is", "friend with romantic tension is", or
# "secret romantic encounter is" (1–4 word role labels, optionally with the
# ``ex-`` prefix).
_ROLE_RE = re.compile(
    r"^\s*((?:ex-)?(?:[a-zà-ÿ-]+(?:\s+[a-zà-ÿ-]+){0,3}))\s+(?:is|are)\s+",
    re.IGNORECASE)
# Match a proper-noun person name (capitalized). Allow accents and internal
# hyphens (e.g. ``Mary-Angela`` or ``Joey Tribbiani``).
_NAME_RE = re.compile(
    r"\b([A-Z][A-Za-zÀ-ÿ]+(?:[-\s][A-Z][A-Za-zÀ-ÿ]+)*)\b")


def _role_priority(role: str) -> int:
    role = role.lower().strip()
    if role in _ROLE_PRIORITY:
        return _ROLE_PRIORITY[role]
    # fall back: longer roles tend to be more specific
    return 20


# Stop-words used by _incremental_word_recall — common English fillers
# that don't carry distinctive descriptor signal.  Keep this list small so
# the check stays strict (false negatives are fine, we just want to catch
# wholesale rewrites).
_STOPWORDS_FOR_RECALL: frozenset = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at",
    "by", "for", "with", "as", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "his", "her", "their", "its", "this", "that",
    "these", "those", "from", "into", "about", "over", "under", "more",
    "most", "less", "than", "so", "such", "very", "often", "sometimes",
    "always", "never", "also", "still", "yet", "just", "only",
})


def _incremental_word_recall(old_val: str, new_val: str) -> float:
    """Fraction of distinctive words from ``old_val`` that survive in
    ``new_val`` (case-insensitive, word-boundary based, after stop-word
    removal).  An ``incremental`` merge should preserve most of the old
    descriptors, so this should be high (≥0.5).  A wholesale rewrite
    will score low (e.g. ≤0.3).

    Returns the recall value in [0, 1].  If ``old_val`` has no
    distinctive words, returns 1.0 (no check possible).
    """
    if not old_val or not new_val:
        return 1.0
    old_words = {
        w.lower() for w in re.findall(r"[A-Za-z]{3,}", old_val)
    } - _STOPWORDS_FOR_RECALL
    if not old_words:
        return 1.0
    new_words = {
        w.lower() for w in re.findall(r"[A-Za-z]{3,}", new_val)
    } - _STOPWORDS_FOR_RECALL
    return len(old_words & new_words) / len(old_words)


def _dedupe_relationships(value: str) -> str:
    """Remove duplicate person entries in a relationships string.

    Example:
        "ex-fiancé is Barry, ex-boyfriend is Paolo, ex-boyfriend is Barry"
        -> "ex-fiancé is Barry, ex-boyfriend is Paolo"  (keep more-specific role)
    """
    if not value or not isinstance(value, str):
        return value

    entries = _split_relationships_paren_aware(value)
    seen: dict[str, int] = {}  # name -> index in `result`
    result: list[str] = []

    for entry in entries:
        if not entry:
            continue
        m_role = _ROLE_RE.match(entry)
        role = m_role.group(1) if m_role else ""
        names = _NAME_RE.findall(entry)
        if not names:
            result.append(entry)
            continue
        primary = names[0]
        if primary in seen:
            old_idx = seen[primary]
            old_entry = result[old_idx]
            old_role_m = _ROLE_RE.match(old_entry)
            old_role = old_role_m.group(1) if old_role_m else ""
            if _role_priority(role) > _role_priority(old_role):
                result[old_idx] = entry
        else:
            seen[primary] = len(result)
            result.append(entry)

    return ", ".join(result)


# Roles that imply a romantic / intimate bond.
_ROMANTIC_ROLE_TERMS = (
    "boyfriend", "girlfriend", "spouse", "wife", "husband",
    "fiancé", "fiancee", "fiance", "partner",
    "ex-boyfriend", "ex-girlfriend", "ex-spouse",
    "ex-wife", "ex-husband", "ex-fiancé", "ex-fiancee", "ex-fiance",
    "ex-partner", "dated", "dating", "lover", "romantic interest",
)
# Romantic-event verbs to look for next to the partner's name.
_RV_VERB = (
    r"kiss(?:es|ed|ing)?|dat(?:e|ed|ing)|broke up with|breaking up with|"
    r"breakup with|hook(?:ed|s)? up with|sl(?:ept|eep) with|"
    r"made out with|made out|got married|got engaged|"
    r"engaged to|fianc(?:[ée]+|e) to|married to|"
    r"in love with|fell for|fell in love with|in a relationship with|"
    r"going out with|out on a date with|got together with|"
    r"romance with|romantically involved with|slept with|"
    r"started dating|began dating|begin dating|start dating|"
    r"ended (?:her|his|their|the) (?:relationship|romance) with|"
    r"ending (?:her|his|their|the) (?:relationship|romance) with"
)
# Role nouns indicating romantic relationship.
_RV_NOUN = (
    r"girlfriend|boyfriend|partner|fianc(?:[ée]+|e)|wife|husband|lover|"
    r"romantic interest|romantic partner"
)
# Romantic event nouns – usually appear in "<noun> with <partner>" form.
_RV_EVENT_NOUN = (
    r"first kiss|first date|first official date|kiss|date|"
    r"intimate moment|intimate morning|intimate encounter|"
    r"romantic moment|romantic encounter|romantic involvement|"
    r"romantic relationship|romantic dinner|fling|affair|reunion|"
    r"breakup|break[- ]up|engagement"
)


def _entry_has_romantic_role(entry: str) -> bool:
    """True if a relationships-string entry uses a romantic role term in
    its prefix (before ``is``/``are``).

    Examples:
        "ex-fiancé is Barry"             -> True
        "friend and romantic interest is Rachel" -> True
        "best friend is Monica"          -> False
        "twin sister is Ursula"          -> False
    """
    if not entry:
        return False
    el = entry.lower()
    # Use the prefix before " is " / " are " when present
    m = re.search(r"\s+(?:is|are)\s+", el)
    prefix = el[:m.start()] if m else el
    return any(re.search(rf"\b{re.escape(rt)}\b", prefix)
               for rt in _ROMANTIC_ROLE_TERMS)


def _has_romantic_context(text: str, partner_first_name: str) -> bool:
    """Strict heuristic: ``text`` describes a romantic event involving
    ``partner_first_name``. Requires the partner's name to appear within a
    short window of an explicit romantic verb or role noun.
    """
    if not text or not partner_first_name:
        return False
    tl = text.lower()
    pf = re.escape(partner_first_name.lower())
    if not re.search(rf"\b{pf}\b", tl):
        return False
    patterns = [
        # "kissed Phoebe", "dating Phoebe", "broke up with Phoebe",
        # "ended her relationship with David"
        rf"(?:{_RV_VERB})\s+{pf}\b",
        # "Phoebe kissed", "Phoebe broke up with", "Phoebe fell for",
        # "Phoebe and Joey kiss"
        rf"\b{pf}\s+(?:and\s+\w+\s+)?(?:{_RV_VERB})",
        # "Monica and Alan begin dating"
        rf"\band\s+{pf}\s+(?:\w+\s+)?(?:{_RV_VERB})",
        rf"\w+\s+and\s+{pf}\s+(?:\w+\s+)?(?:{_RV_VERB})",
        # "Phoebe's girlfriend", "girlfriend Phoebe", "boyfriend is Phoebe"
        rf"\b{pf}'s\s+(?:{_RV_NOUN})",
        rf"(?:{_RV_NOUN})\s+(?:is|are|was|were)\s+{pf}\b",
        rf"(?:{_RV_NOUN})\s+{pf}\b",
        # "first kiss with Ross", "intimate moment with Ross",
        # "romantic encounter with Rachel" — natural narrative phrasing.
        rf"(?:{_RV_EVENT_NOUN})\s+with\s+(?:\w+\s+){{0,2}}{pf}\b",
        # "Rachel's first kiss / first date / romantic moment"
        rf"\b{pf}'s\s+(?:first\s+)?(?:{_RV_EVENT_NOUN})",
        # "feelings/love/crush for/to/on Ross", "feelings about Ross"
        rf"(?:feelings|love|crush|romantic feelings|romantic interest)"
        rf"\s+(?:for|to|on|towards?|about)\s+{pf}\b",
        # "confessed/admitted/revealed his feelings to Rachel"
        rf"(?:confess(?:ed|ing|es)?|admit(?:ted|ting|s)?|reveal(?:ed|ing|s)?)"
        rf"\s+(?:her|his|their|the)?\s*(?:lingering\s+|longstanding\s+)?"
        rf"feelings\s+(?:for|to|about)\s+{pf}\b",
        # "longing for Rachel", "yearning for Ross", "pining for X"
        rf"(?:longing|yearning|pining)\s+for\s+{pf}\b",
    ]
    if any(re.search(p, tl) for p in patterns):
        return True

    # Sentence-level fallback: partner name appears in a sentence that ALSO
    # contains an unambiguously romantic marker (no other proper name
    # is dominant in that sentence). Catches narrative phrasing like
    # "Rachel reciprocated his declaration of love" or "Ross shares his
    # vision of their future together with Rachel".
    sentence_markers = (
        r"reciprocate(?:s|d)?\s+(?:his|her|their|the)\s+(?:declaration\s+of\s+)?"
        r"(?:love|affection|feelings)",
        r"(?:declaration|profession)\s+of\s+love",
        r"(?:declar(?:es|ed|ing)|profess(?:es|ed|ing)?)\s+(?:his|her|their)\s+"
        r"(?:love|affection|feelings)",
        r"\btheir relationship\b",
        r"\btheir romance\b",
        r"\btheir future together\b",
        r"future\s+together,?\s+(?:including\s+)?(?:marriage|children|kids)",
        r"\bin love together\b",
    )
    sentences = re.split(r"(?<=[.!?])\s+|;\s*", tl)
    for s in sentences:
        if not re.search(rf"\b{pf}\b", s):
            continue
        if any(re.search(m, s) for m in sentence_markers):
            return True
    return False


def _normalize_entry(entry: str) -> str:
    """Lowercase and squeeze whitespace for stable entry comparison."""
    return re.sub(r"\s+", " ", entry.strip().lower())


def _entries_set(value: str) -> set[str]:
    """Return a set of normalised entries from a relationships string."""
    if not isinstance(value, str):
        return set()
    return {
        _normalize_entry(e)
        for e in _split_relationships_paren_aware(value)
        if e.strip()
    }


def _check_inter_main_romantic(
    char_name: str,
    new_value: str,
    consumed_ids: list[str],
    all_archives: dict[str, list[dict]] | None,
    old_value: str = "",
) -> list[str]:
    """Reject inter-main-character romantic claims that lack supporting
    evidence in the OTHER main character's archive.

    Catches errors like "Phoebe's ex-boyfriend is Joey" when Joey's own
    archive contains no romantic event with Phoebe.

    Entries already present in ``old_value`` (i.e. carried over from the
    previous persona version) are NOT re-validated — they were either
    valid when introduced or have been inherited unchallenged for many
    episodes.  Only newly-introduced claims need fresh corroboration.
    """
    issues: list[str] = []
    if not all_archives:
        return issues
    char_first = char_name.split()[0]
    main_first_to_full = {n.split()[0]: n for n in all_archives.keys()}
    inherited = _entries_set(old_value)

    # Episodes covered by consumed_ids
    consumed_eps: set[tuple[int, int]] = set()
    for sid in consumed_ids:
        m = re.match(r"s(\d+)_e(\d+)_", sid)
        if m:
            consumed_eps.add((int(m.group(1)), int(m.group(2))))

    for entry in _split_relationships_paren_aware(new_value):
        if not _entry_has_romantic_role(entry):
            continue
        if _normalize_entry(entry) in inherited:
            continue  # already present in old_value — skip re-validation
        names = _NAME_RE.findall(entry)
        target = next(
            (n for n in names if n != char_first and n in main_first_to_full),
            None)
        if not target:
            continue  # not pointing to another tracked main character

        target_full = main_first_to_full[target]
        target_archive = all_archives.get(target_full, [])
        # Look in target's archive within the consumed episode window (±2 eps)
        relevant = [
            e for e in target_archive
            if any(e["season"] == s and abs(e["episode"] - ep) <= 2
                   for (s, ep) in consumed_eps)
        ]
        if any(_has_romantic_context(e.get("summary", ""), char_first)
               for e in relevant):
            continue  # corroborated, allow

        issues.append(
            f"relationships: inter-main romantic claim '{entry}' lacks "
            f"corroborating romantic context in {target_full}'s archive "
            f"(checked {len(relevant)} nearby scenes)"
        )
    return issues


def _strip_inter_main_romantic(
    char_name: str,
    value: str,
    consumed_ids: list[str],
    all_archives: dict[str, list[dict]] | None,
    old_value: str = "",
) -> str:
    """Drop entries from a relationships string that fail the inter-main
    romantic cross-check, returning the remaining concatenated string.

    Entries already present in ``old_value`` are preserved unchanged.
    """
    if not all_archives:
        return value
    char_first = char_name.split()[0]
    main_first_to_full = {n.split()[0]: n for n in all_archives.keys()}
    inherited = _entries_set(old_value)
    consumed_eps: set[tuple[int, int]] = set()
    for sid in consumed_ids:
        m = re.match(r"s(\d+)_e(\d+)_", sid)
        if m:
            consumed_eps.add((int(m.group(1)), int(m.group(2))))

    kept: list[str] = []
    for entry in _split_relationships_paren_aware(value):
        if not entry:
            continue
        if not _entry_has_romantic_role(entry):
            kept.append(entry)
            continue
        if _normalize_entry(entry) in inherited:
            kept.append(entry)
            continue
        names = _NAME_RE.findall(entry)
        target = next(
            (n for n in names if n != char_first and n in main_first_to_full),
            None)
        if not target:
            kept.append(entry)
            continue
        target_full = main_first_to_full[target]
        target_archive = all_archives.get(target_full, [])
        relevant = [
            e for e in target_archive
            if any(e["season"] == s and abs(e["episode"] - ep) <= 2
                   for (s, ep) in consumed_eps)
        ]
        if any(_has_romantic_context(e.get("summary", ""), char_first)
               for e in relevant):
            kept.append(entry)
    return ", ".join(kept)


# ---------------------------------------------------------------------------
# Detect "girlfriend → ex-girlfriend" style downgrades that lack breakup
# evidence in either character's archive.
# ---------------------------------------------------------------------------

# Roles that can only be downgraded to "ex-<role>" by an explicit breakup.
_DOWNGRADE_ROLES = (
    "girlfriend", "boyfriend", "partner", "fiancé", "fiancee", "fiance",
    "wife", "husband", "lover", "spouse",
)
_DOWNGRADE_ROLE_RE = "|".join(re.escape(r) for r in _DOWNGRADE_ROLES)

_BREAKUP_VERB_END_REL = (
    r"end(?:s|ed|ing)?\s+(?:his|her|their|the)\s+"
    r"(?:relationship|romance|engagement|marriage)"
)


def _has_breakup_between(text: str, name_a: str, name_b: str) -> bool:
    """Strict check: ``text`` describes a breakup between ``name_a`` and
    ``name_b`` (either direction).  Avoids false positives when the breakup
    is between one of them and a third party (e.g. "Ross ended his
    relationship with **Julie** for her [Rachel]").
    """
    if not text or not name_a or not name_b:
        return False
    tl = text.lower()
    a = re.escape(name_a.lower())
    b = re.escape(name_b.lower())
    if not (re.search(rf"\b{a}\b", tl) and re.search(rf"\b{b}\b", tl)):
        # Allow asymmetric: maybe only one name appears (the focal narrator
        # is implied via pronouns).
        if not re.search(rf"\b{b}\b", tl):
            return False
    patterns: list[str] = []
    for x, y in ((a, b), (b, a)):
        patterns.extend([
            # "<x> broke up with <y>" / "<x> ended (his/her/their/the)
            # relationship with <y>" — y is the object of the breakup verb.
            rf"\b{x}\s+(?:broke up|broke it off|split up|"
            rf"call(?:ed|s)?\s+(?:it\s+(?:off|quits)|the\s+relationship\s+off)|"
            rf"{_BREAKUP_VERB_END_REL}|"
            rf"divorc(?:ed|es|ing)?|dump(?:ed|s|ing)?)"
            rf"\s+(?:with\s+)?{y}\b",
            # "broke up with <y>", "ended his/her/their/the relationship with
            # <y>" — passive / pronoun-subject form, requires y as object.
            rf"(?:broke up|broke it off|split up|splitting up|"
            rf"call(?:ed|s)?\s+(?:it\s+(?:off|quits)|the\s+relationship\s+off)|"
            rf"{_BREAKUP_VERB_END_REL}|"
            rf"made\s+the\s+(?:difficult\s+)?decision\s+to\s+end\s+"
            rf"(?:his|her|their|the)\s+(?:relationship|romance))"
            rf"\s+with\s+{y}\b",
            # "<x> dumped <y>", "<x> divorced <y>"
            rf"\b{x}\s+(?:dump(?:ed|s|ing)?|divorc(?:ed|es|ing)?)\s+{y}\b",
            # Conjunction: "<x> and <y> broke up / split up / divorced"
            rf"\b{x}\s+and\s+{y}\s+(?:broke up|split up|divorc(?:ed|es|ing)?|"
            rf"are\s+no\s+longer\s+(?:together|dating|a couple))\b",
            # Possessive: "their/the breakup with <y>"
            rf"(?:their|the|her|his)\s+(?:breakup|break-up|split)\s+with\s+{y}\b",
            # "<x> walked out on <y>", "<x> left <y> for"
            rf"\b{x}\s+(?:walked out on|left)\s+{y}\b",
            # "<x> experiences/experienced (a|the) (painful|emotional|...)?
            # breakup as <y> ..." — covers Joey-Kate-style narration
            rf"\b{x}\s+experienc(?:e|ed|es|ing)\s+(?:a|the)\s+"
            rf"(?:\w+\s+)?(?:breakup|break-up)\s+as\s+{y}\b",
        ])
    if any(re.search(p, tl) for p in patterns):
        return True

    # Sentence-level fallback: BOTH names co-occur in a sentence containing a
    # strong breakup signal AND no third-party name is the breakup target.
    third_party_re = re.compile(
        r"(?:breakup|break-up|split|relationship|romance)\s+with\s+([a-zà-ÿ]+)",
        re.IGNORECASE,
    )
    sentences = re.split(r"(?<=[.!?])\s+|;\s*", tl)
    for sent in sentences:
        if not (re.search(rf"\b{a}\b", sent) and re.search(rf"\b{b}\b", sent)):
            continue
        third_party = False
        for tp in third_party_re.findall(sent):
            tpl = tp.lower()
            if tpl != name_a.lower() and tpl != name_b.lower():
                third_party = True
                break
        if third_party:
            continue
        # Skip sentences that describe leaving a venue *to be with*, *to
        # join*, *to find*, *to see*, or *to meet* the partner — these are
        # the OPPOSITE of a breakup (e.g. "decided to leave the dinner to
        # be with her boyfriend, Mike").
        toward_partner_re = re.compile(
            rf"\bto\s+(?:be\s+with|join|find|see|meet|reunite\s+with)\s+"
            rf"(?:[\w\s,]*)\b{b}\b",
        )
        if toward_partner_re.search(sent):
            continue
        toward_a_re = re.compile(
            rf"\bto\s+(?:be\s+with|join|find|see|meet|reunite\s+with)\s+"
            rf"(?:[\w\s,]*)\b{a}\b",
        )
        if toward_a_re.search(sent):
            continue
        # Sentence-level signals.  We deliberately omit bare nouns like
        # "breakup" / "split up" because they often describe a breakup that
        # one of the two named persons had with a *third* party (e.g.
        # "Monica comforted Chandler after his breakup").  Pair-level
        # patterns above already catch true two-person breakups.  Phrases
        # below carry stronger directional cues: "end of their relationship"
        # implies a shared bond ending, "decides to leave" / "is leaving
        # for" describes a partner physically leaving the focal character.
        sentence_signals = [
            r"\bend\s+of\s+(?:their|the)\s+"
            r"(?:relationship|romance|engagement|marriage)\b",
            r"\bend(?:s|ed|ing)?\s+(?:their|the)\s+"
            r"(?:relationship|romance|engagement|marriage)\b",
            r"\bdecide(?:s|d)?\s+to\s+(?:leave|move|take\s+off|go)\b",
            r"\bis\s+leaving\s+for\b",
        ]
        if any(re.search(p, sent) for p in sentence_signals):
            return True
        # "<focal>... encouraging/forgiving/letting/allowing <other> to
        # pursue|be with|date <partner>" — narrator gives romantic partner
        # up.  Requires an explicit romantic verb to avoid false positives.
        for target in (a, b):
            give_up_re = re.compile(
                rf"(?:encourag(?:ing|ed|e|es)|forgav(?:e|ing)|forgiv(?:ing|en)|"
                rf"step(?:s|ped|ping)?\s+aside\s+for)"
                rf"\s+(?:\w+\s+){{0,4}}\bto\s+(?:pursue|be\s+with|date)\s+"
                rf"(?:\w+\s+){{0,2}}\b{target}\b",
            )
            if give_up_re.search(sent):
                return True
    return False


def _has_reconciliation_between(
    text: str, name_a: str, name_b: str,
) -> bool:
    """Detect explicit reconciliation / getting-back-together between
    ``name_a`` and ``name_b`` (covers both directions).  Used to suppress
    spurious ``ex-`` downgrades after a couple gets back together.
    """
    if not text or not name_a or not name_b:
        return False
    tl = text.lower()
    a = re.escape(name_a.lower())
    b = re.escape(name_b.lower())
    if not (re.search(rf"\b{a}\b", tl) and re.search(rf"\b{b}\b", tl)):
        if not re.search(rf"\b{b}\b", tl):
            return False
    pair_patterns = []
    for x, y in ((a, b), (b, a)):
        pair_patterns.extend([
            rf"\b{x}\s+and\s+{y}\s+(?:get|got|are|have)\s+(?:back\s+together|"
            rf"reunit(?:ed|ing|e)|reconcil(?:ed|ing|e)|engaged|married|"
            rf"rekindl(?:ed|ing|e))\b",
            rf"\b{x}\s+(?:gets|got|getting)\s+back\s+together\s+with\s+{y}\b",
            rf"\b{x}\s+(?:reconcil(?:es|ed|ing)|reunit(?:es|ed|ing)|"
            rf"rekindl(?:es|ed|ing))\s+with\s+{y}\b",
            rf"\b{x}\s+propos(?:es|ed|ing)\s+to\s+{y}\b",
            rf"\b{x}\s+marri(?:es|ed|ing)\s+{y}\b",
            rf"\b{x}\s+and\s+{y}\s+(?:s)?\s*wedding\b",
        ])
    if any(re.search(p, tl) for p in pair_patterns):
        return True

    # Sentence-level fallback: both names + a *pair-anchored* reconciliation
    # marker.  We deliberately omit ambiguous bare verbs like "rekindle /
    # reunite / reconcile" because they often describe a *different* pair
    # mentioned in the same sentence (e.g. "rekindling with David ... after
    # her breakup with Mike").  Pair-anchored phrases below are far less
    # ambiguous when both target names co-occur.
    sentences = re.split(r"(?<=[.!?])\s+|;\s*", tl)
    sent_signals = [
        r"\bgot\s+back\s+together\b",
        r"\bgetting\s+back\s+together\b",
        r"\bback\s+together\s+with\b",
        r"\breaffirm(?:s|ed|ing)?\s+(?:his|her|their)\s+commitment\b",
        r"\b(?:gets|getting|got)\s+engaged\b",
        r"\bengagement\b",
        r"\bwedding\s+day\b",
        r"\bgot\s+married\b",
        r"\bmarried\s+each\s+other\b",
        r"\b(?:say|said|saying)\s+(?:i\s+do|their\s+vows)\b",
        r"\bmove(?:s|d|ing)?\s+in\s+together\b",
        r"\btheir\s+wedding\b",
    ]
    for sent in sentences:
        if not (re.search(rf"\b{a}\b", sent) and re.search(rf"\b{b}\b", sent)):
            continue
        if any(re.search(p, sent) for p in sent_signals):
            return True
    return False


def _has_engagement_evidence(text: str, name_a: str, name_b: str) -> bool:
    """Strict engagement-completion check between A and B.

    Two-stage requirement (both must hold):

    1. **Direct completion**: a pair-anchored phrase that confirms the
       engagement is *complete* (``X and Y got engaged``, ``became engaged
       to <B>``, ``their engagement was announced``).  -OR-

    2. **Event + state pair**: a *proposal-event* pair pattern (``X
       proposes to Y``, ``marriage proposal from <B>``) AND somewhere in
       the cumulative blob a *state-marker* noun (``their engagement``,
       ``got engaged``, ``newly engaged``, ``announces (their|his|her)
       engagement``).  This rejects rejected/hypothetical proposals like
       the Vegas dice-roll proposal in Friends S05E24 — Chandler proposes
       but the LLM never escalates to "their engagement" until S06E25.

    Anticipatory / hypothetical / bare-noun signals are insufficient.
    """
    if not text or not name_a or not name_b:
        return False
    a = re.escape(name_a.split()[0].lower())
    b = re.escape(name_b.split()[0].lower())
    tl = text.lower()
    if not (re.search(rf"\b{a}\b", tl) and re.search(rf"\b{b}\b", tl)):
        return False

    # Stage 1A — pair-anchored DIRECT completion phrases (need both names).
    direct_completion = [
        rf"\b{a}\s+(?:and|&)\s+{b}\s+"
        rf"(?:got|are|are\s+now|now\s+are|have\s+gotten|became)\s+engaged\b",
        rf"\b{b}\s+(?:and|&)\s+{a}\s+"
        rf"(?:got|are|are\s+now|now\s+are|have\s+gotten|became)\s+engaged\b",
        rf"\bbecame?\s+engaged\s+to\s+{b}\b",
        rf"\bbecame?\s+engaged\s+to\s+{a}\b",
        rf"\b{a}\s+pop(?:ped|s)?\s+the\s+question\s+and\s+{b}\s+"
        rf"(?:said|say)\s+yes\b",
        rf"\b{b}\s+pop(?:ped|s)?\s+the\s+question\s+and\s+{a}\s+"
        rf"(?:said|say)\s+yes\b",
        rf"\b(?:his|her)\s+engagement\s+to\s+{b}\b",
        rf"\b(?:his|her)\s+engagement\s+to\s+{a}\b",
        rf"\bengagement\s+(?:of|between)\s+{a}\s+and\s+{b}\b",
        rf"\bengagement\s+(?:of|between)\s+{b}\s+and\s+{a}\b",
    ]
    if any(re.search(p, tl) for p in direct_completion):
        return True

    # Stage 1B — event-pair pattern + global state-marker.
    sentences = re.split(r"(?<=[.!?])\s+|;\s*", tl)
    discard_markers = [
        r"\bnot\s+ready\b",
        r"\bnot\s+(?:really|actually|yet)\s+engaged\b",
        r"\breject(?:ed|ing|s)?\b",
        r"\bturn(?:ed|s|ing)?\s+down\b",
        r"\bdice\s+roll\b",
        r"\bcontingent\s+on\b",
        r"\bjok(?:e|ing|ed)\s+about\b",
        r"\bpretend(?:ed|ing|s)?\s+to\b",
        r"\bhypothetical(?:ly)?\b",
        r"\bfake\s+(?:proposal|engagement)\b",
        # Anticipatory / hypothetical proposal modifiers — common LLM
        # generations that look like proposal sentences but the proposal
        # never actually happened (or is being considered/forced).
        r"\bpressur(?:ed|ing|es)?\s+(?:by\s+\w+\s+)?to\s+propos\w*\b",
        r"\bforce[ds]?\s+to\s+propos\w*\b",
        r"\bencourag(?:e|ed|ing)\s+to\s+propos\w*\b",
        r"\btold\s+to\s+propos\w*\b",
        r"\bthink(?:ing|s)?\s+about\s+propos\w*\b",
        r"\bconsider(?:ing|s|ed)?\s+propos\w*\b",
        r"\bplan(?:ning|s|ned)?\s+to\s+propos\w*\b",
        r"\bpreparing\s+to\s+propos\w*\b",
        r"\bdetailed\s+plan\s+to\s+propos\w*\b",
        # Failed / abandoned attempts.
        r"\bdid\s+not\s+propose\b",
        r"\bdoes\s+not\s+propose\b",
        r"\bdoesn'?t\s+propose\b",
        r"\bcan'?t\s+propose\b",
        r"\bunable\s+to\s+propose\b",
    ]
    proposal_event_sent = [
        rf"\bpropos(?:es?|ed|ing)\s+to\s+{b}\b",
        rf"\bpropos(?:es?|ed|ing)\s+to\s+{a}\b",
        rf"\bmarriage\s+proposal\s+from\s+{a}\b",
        rf"\bmarriage\s+proposal\s+from\s+{b}\b",
        rf"\bpropos(?:e|al)\s+(?:to\s+)?(?:him|her)\b",
    ]
    has_event = False
    for sent in sentences:
        if not (re.search(rf"\b{a}\b", sent) and re.search(rf"\b{b}\b", sent)):
            continue
        if any(re.search(d, sent) for d in discard_markers):
            continue
        if any(re.search(p, sent) for p in proposal_event_sent):
            has_event = True
            break
    if not has_event:
        return False

    # State markers must appear in a sentence containing at least one of
    # the partners *and* not bound to a third-party name (we can't always
    # detect that perfectly, but skipping bare "engagement party /
    # announcement" — which often refers to OTHER pairs in the show —
    # eliminates the most common cross-pair false positive).
    state_markers = [
        r"\btheir\s+engagement\b",
        r"\b(?:got|just\s+got|finally\s+got|have\s+gotten)\s+engaged\b",
        r"\b(?:are|are\s+now|now\s+are)\s+engaged\b",
        r"\bbecame\s+engaged\b",
        r"\bnewly\s+engaged\b",
        r"\bannounce[ds]?\s+(?:their|his|her)\s+engagement\s+to\b",
        r"\bsaid\s+yes\s+to\s+(?:his|her|the)\s+proposal\b",
        r"\baccept(?:s|ed|ing)?\s+(?:his|her|the)\s+proposal\b",
    ]
    for sent in sentences:
        if not (re.search(rf"\b{a}\b", sent) or re.search(rf"\b{b}\b", sent)):
            continue
        if any(re.search(p, sent) for p in state_markers):
            return True
    return False


def _has_marriage_evidence(text: str, name_a: str, name_b: str) -> bool:
    """Strict marriage-completion check between A and B.

    Two-stage requirement (both must hold):

    1. **Direct completion**: a pair-anchored phrase confirming a *completed*
       wedding (``X marries Y``, ``X and Y got married``, ``becomes <B>'s
       wife``, ``his/her marriage to <B>``, ``married to <B>``, possessive
       wife/husband appositive).  -OR-

    2. **Event + state pair**: a wedding-event pair pattern (``X marries
       Y`` / ``their wedding ceremony took place`` / ``first dance as
       married``) AND somewhere in the blob a *state-marker* (``got
       married``, ``are married``, ``husband and wife``, ``honeymoon``).

    Bare references — ``their wedding``, ``wedding planning``, ``upcoming
    marriage``, ``wedding venue`` — never count on their own.
    """
    if not text or not name_a or not name_b:
        return False
    a = re.escape(name_a.split()[0].lower())
    b = re.escape(name_b.split()[0].lower())
    tl = text.lower()
    if not (re.search(rf"\b{a}\b", tl) and re.search(rf"\b{b}\b", tl)):
        return False

    # Stage 1A — pair-anchored DIRECT completion phrases (whole-blob).
    direct_completion = [
        rf"\b{a}\s+marri(?:es|ed|ing)\s+{b}\b",
        rf"\b{b}\s+marri(?:es|ed|ing)\s+{a}\b",
        rf"\b{a}\s+and\s+{b}\s+(?:got|are|are\s+now|have)\s+married\b",
        rf"\b{b}\s+and\s+{a}\s+(?:got|are|are\s+now|have)\s+married\b",
        rf"\bfirst\s+dance\s+as\s+(?:a\s+)?(?:newly\s+)?married\s+"
        rf"(?:couple|woman|man)\s+with\s+{b}\b",
        rf"\bfirst\s+dance\s+as\s+(?:a\s+)?(?:newly\s+)?married\s+"
        rf"(?:couple|woman|man)\s+with\s+{a}\b",
        rf"\bbecom(?:es?|ing|e)\s+{b}'s\s+(?:wife|husband|spouse)\b",
        rf"\bbecom(?:es?|ing|e)\s+{a}'s\s+(?:wife|husband|spouse)\b",
        rf"\b(?:his|her)\s+marriage\s+to\s+{b}\b",
        rf"\b(?:his|her)\s+marriage\s+to\s+{a}\b",
        rf"\bmarried\s+to\s+{b}\b",
        rf"\bmarried\s+to\s+{a}\b",
        rf"\b(?:his|her)\s+(?:wife|husband|spouse)\s+{b}\b",
        rf"\b(?:his|her)\s+(?:wife|husband|spouse)\s+{a}\b",
    ]
    if any(re.search(p, tl) for p in direct_completion):
        return True

    # Stage 1A' — sentence-level direct completion: same sentence has
    # both names AND a "married <B>"/"marries <B>" phrase (with the other
    # main as the implied subject via pronoun).
    sent_direct_patterns = [
        rf"\bmarri(?:es|ed|ing)\s+{a}\b",
        rf"\bmarri(?:es|ed|ing)\s+{b}\b",
        rf"\bmarri(?:es|ed|ing)\s+(?:him|her)\b",
    ]
    sent_discard = [
        r"\bdid\s+not\s+marry\b",
        r"\bdoes\s+not\s+marry\b",
        r"\bdoesn'?t\s+marry\b",
        r"\bdidn'?t\s+marry\b",
        r"\bplan(?:ning|ned|s)?\s+to\s+marry\b",
        r"\bpretend(?:ed|ing|s)?\s+to\b",
    ]
    sentences_blob = re.split(r"(?<=[.!?])\s+|;\s*", tl)
    for sent in sentences_blob:
        if not (re.search(rf"\b{a}\b", sent) and re.search(rf"\b{b}\b", sent)):
            continue
        if any(re.search(d, sent) for d in sent_discard):
            continue
        if any(re.search(p, sent) for p in sent_direct_patterns):
            return True

    # Stage 1B — wedding-event + state pair.
    sentences = re.split(r"(?<=[.!?])\s+|;\s*", tl)
    pair_event_signals = [
        r"\bhad\s+(?:their|the)\s+wedding\s+ceremony\b",
        r"\b(?:their|the)\s+wedding\s+ceremony\s+took\s+place\b",
        r"\bfirst\s+dance\s+as\s+(?:a\s+)?(?:newly\s+)?married\s+"
        r"(?:couple|woman|man)\b",
        r"\bbecom(?:es?|ing|e)\s+\w+(?:'s)?\s+(?:wife|husband|spouse)\b",
        r"\bsaid\s+(?:i\s+do|their\s+vows)\b",
        r"\bexchang(?:e|ed|ing)\s+(?:their\s+)?vows\b",
        r"\bjust\s+married\b",
        r"\btied\s+the\s+knot\b",
    ]
    discard_markers = [
        r"\bplan(?:ning|ned|s)?\s+(?:their|the|a)?\s*wedding\b",
        r"\bwedding\s+venue\b",
        r"\bwedding\s+(?:date|location|invitation|invitations|guest|"
        r"caterer|cake|dress|toast|prep|preparations?|preparation)\b",
        r"\bmoved\s+up\b",
        r"\bbeing\s+moved\s+up\b",
        r"\bfantasy\s+wedding\b",
        r"\bjok(?:e|ing|ed)\s+about\s+(?:a\s+)?wedding\b",
        r"\bdream\s+wedding\b",
        r"\bupcoming\s+marriage\b",
        r"\bupcoming\s+wedding\b",
        r"\bbefore\s+(?:their|the)\s+wedding\b",
        r"\bprior\s+to\s+(?:their|the)\s+wedding\b",
        r"\bpretend(?:ed|ing|s)?\s+to\s+be\s+(?:married|engaged)\b",
    ]
    has_event = False
    for sent in sentences:
        if not (re.search(rf"\b{a}\b", sent) and re.search(rf"\b{b}\b", sent)):
            continue
        if any(re.search(d, sent) for d in discard_markers):
            continue
        if any(re.search(s, sent) for s in pair_event_signals):
            has_event = True
            break
    if not has_event:
        return False

    state_markers_global = [
        r"\bgot\s+married\b",
        r"\bare\s+(?:now\s+)?married\b",
        r"\bare\s+now\s+(?:husband\s+and\s+wife|a\s+married\s+couple)\b",
        r"\bmarried\s+each\s+other\b",
        r"\bnewly[\s-]wed\b",
        r"\bnewlyweds\b",
        r"\bnow\s+a\s+married\s+couple\b",
        r"\bnow\s+husband\s+and\s+wife\b",
        r"\bhoneymoon\b",
        r"\bnew\s+marriage\b",
    ]
    return any(re.search(p, tl) for p in state_markers_global)


# ---------------------------------------------------------------------------
# Status-tier hierarchy (for premature-escalation guard).
# ---------------------------------------------------------------------------

STATUS_TIER_DATING = 1   # boyfriend / girlfriend / partner / lover ...
STATUS_TIER_ENGAGED = 2  # fiancé / fiancée
STATUS_TIER_MARRIED = 3  # husband / wife / spouse

_DATING_DEMOTION = {
    "fiancé": "boyfriend",
    "fiance": "boyfriend",
    "fiancée": "girlfriend",
    "fiancee": "girlfriend",
    "husband": "boyfriend",
    "wife": "girlfriend",
    "spouse": "partner",
}
_ENGAGED_DEMOTION = {
    "husband": "fiancé",
    "wife": "fiancée",
    "spouse": "fiancé(e)",
}


def _validate_status_tier_evidence(
    value: str,
    char_name: str,
    char_archive: list[dict],
    main_chars: list[str],
) -> tuple[str, list[tuple[str, str, str, str]]]:
    """For each ``husband/wife/spouse`` or ``fiancé/fiancée`` claim against
    another main character, verify that the cumulative archive text contains
    the corresponding evidence (engagement / marriage).  When evidence is
    missing, demote to the highest evidenced tier.

    Returns ``(new_value, demotions)`` where each demotion is
    ``(old_role, new_role, partner_name, reason)``.
    """
    if not value or not char_name or not main_chars:
        return value, []
    char_first = char_name.split()[0]
    main_firsts = {c.split()[0].lower(): c for c in main_chars}
    text_blob = "\n".join(s.get("summary", "") for s in char_archive if s)

    new_parts: list[str] = []
    demotions: list[tuple[str, str, str, str]] = []
    for raw in _split_relationships_paren_aware(value):
        m = _ROLE_RE.match(raw)
        if not m:
            new_parts.append(raw)
            continue
        role = m.group(1).strip()
        role_l = role.lower()
        if role_l.startswith(("ex-", "former", "late-")):
            new_parts.append(raw)
            continue
        rest = raw[m.end():]
        nm = _NAME_RE.search(rest)
        if not nm:
            new_parts.append(raw)
            continue
        name = nm.group(1).strip()
        partner_first = name.split()[0]
        if partner_first.lower() not in main_firsts:
            new_parts.append(raw)
            continue
        if main_firsts[partner_first.lower()] == char_name:
            new_parts.append(raw)
            continue

        target_role: str | None = None
        reason = ""
        if role_l in {"husband", "wife", "spouse"}:
            if not _has_marriage_evidence(text_blob, char_first, partner_first):
                if _has_engagement_evidence(text_blob, char_first, partner_first):
                    target_role = _ENGAGED_DEMOTION.get(role_l, "fiancé")
                    reason = "no marriage evidence; demote to engaged tier"
                else:
                    target_role = _DATING_DEMOTION.get(role_l, "partner")
                    reason = "no marriage or engagement evidence; demote to dating"
        elif role_l in {"fiancé", "fiance", "fiancée", "fiancee"}:
            if not _has_engagement_evidence(text_blob, char_first, partner_first):
                target_role = _DATING_DEMOTION.get(role_l, "partner")
                reason = "no engagement evidence; demote to dating"

        if target_role is None:
            new_parts.append(raw)
            continue

        new_raw = re.sub(
            rf"^\s*{re.escape(role)}\s+is\s+",
            f"{target_role} is ",
            raw,
            count=1,
            flags=re.IGNORECASE,
        )
        new_parts.append(new_raw)
        demotions.append((role_l, target_role, name, reason))

    if not demotions:
        return value, []
    return ", ".join(new_parts), demotions


# ---------------------------------------------------------------------------
# Inter-main current-couple continuity guard.
# ---------------------------------------------------------------------------

_INTER_MAIN_COUPLE_ROLES = {
    "boyfriend", "girlfriend", "partner", "lover",
    "romantic interest", "romantic partner", "love interest",
    "dating", "seeing",
    "on-off boyfriend", "on-off girlfriend",
    "fiancé", "fiance", "fiancée", "fiancee",
    "husband", "wife", "spouse",
}


def _restore_dropped_inter_main_couple(
    old_value: str,
    new_value: str,
    char_name: str,
    char_archive: list[dict],
    main_chars: list[str],
) -> tuple[str, list[tuple[str, str]]]:
    """Restore current-couple entries between ``char_name`` and another main
    if a merge silently dropped them and the archive provides no breakup
    evidence.  Mirrors :func:`_restore_dropped_core_roles` but for the
    current-romantic-couple tier (boyfriend/girlfriend/partner/...).

    Returns ``(preserved_value, restored_entries)``.
    """
    if not isinstance(old_value, str) or not isinstance(new_value, str):
        return new_value, []
    char_first = char_name.split()[0] if char_name else ""
    if not char_first:
        return new_value, []
    main_firsts = {c.split()[0].lower(): c for c in main_chars}

    old_entries = _parse_relationship_entries(old_value)
    new_entries = _parse_relationship_entries(new_value)
    new_keys = {(role.lower(), name.lower()) for role, name, _ in new_entries}
    new_names = {name.lower() for _, name, _ in new_entries}

    additions: list[str] = []
    restored: list[tuple[str, str]] = []
    for role, name, raw in old_entries:
        role_l = role.lower()
        if role_l.startswith(("ex-", "former", "late-")):
            continue
        if role_l not in _INTER_MAIN_COUPLE_ROLES:
            continue
        partner_first = name.split()[0].lower()
        if partner_first not in main_firsts:
            continue
        if main_firsts[partner_first] == char_name:
            continue
        if (role_l, name.lower()) in new_keys:
            continue
        if name.lower() in new_names:
            continue  # LLM may have re-labeled (e.g. boyfriend -> partner)
        # Check for breakup evidence — drop is allowed only if explicit.
        had_breakup = False
        for ar_e in char_archive:
            summary = ar_e.get("summary", "") or ""
            if not summary:
                continue
            if _has_breakup_between(summary, char_first, partner_first):
                had_breakup = True
                break
        if had_breakup:
            continue
        additions.append(raw)
        restored.append((role, name))

    if not restored:
        return new_value, []
    sep = ", " if new_value.strip() else ""
    preserved = new_value.rstrip().rstrip(",") + sep + ", ".join(additions)
    return preserved, restored


# ---------------------------------------------------------------------------
# Inter-main status alignment (downgrade premature-escalation when partner
# is at a strictly lower tier).
# ---------------------------------------------------------------------------

def _role_tier(role_l: str) -> int:
    if role_l in {"husband", "wife", "spouse"}:
        return STATUS_TIER_MARRIED
    if role_l in {"fiancé", "fiance", "fiancée", "fiancee",
                  "wife-to-be", "husband-to-be"}:
        return STATUS_TIER_ENGAGED
    if (role_l in {"boyfriend", "girlfriend", "partner", "lover",
                   "dating", "seeing", "love interest", "romantic interest",
                   "romantic partner", "on-off boyfriend", "on-off girlfriend"}
            or any(role_l == r or role_l.startswith(r + " ")
                   for r in ("boyfriend", "girlfriend", "partner",
                             "lover", "love interest", "dating"))):
        return STATUS_TIER_DATING
    return 0  # not a couple-tier role


def _align_inter_main_status(
    value: str,
    char_name: str,
    other_personas: dict | None,
    main_chars: list[str],
) -> tuple[str, list[tuple[str, str, str, str]]]:
    """If ``value`` claims a tier-N couple role with a main partner whose
    own current persona shows tier-M < N, demote our claim to tier-M.

    This catches the LLM running ahead on one side (e.g. Monica says
    ``husband is Chandler`` while Chandler still says ``girlfriend is
    Monica``) and aligns to the lower (more conservative) side.

    Returns ``(new_value, alignments)``.
    """
    if not value or not other_personas or not main_chars:
        return value, []
    char_first = char_name.split()[0] if char_name else ""
    main_firsts = {c.split()[0].lower(): c for c in main_chars}

    new_parts: list[str] = []
    alignments: list[tuple[str, str, str, str]] = []
    for raw in _split_relationships_paren_aware(value):
        m = _ROLE_RE.match(raw)
        if not m:
            new_parts.append(raw)
            continue
        role = m.group(1).strip()
        role_l = role.lower()
        if role_l.startswith(("ex-", "former", "late-")):
            new_parts.append(raw)
            continue
        my_tier = _role_tier(role_l)
        if my_tier == 0:
            new_parts.append(raw)
            continue
        rest = raw[m.end():]
        nm = _NAME_RE.search(rest)
        if not nm:
            new_parts.append(raw)
            continue
        partner_name = nm.group(1).strip()
        partner_first = partner_name.split()[0].lower()
        if partner_first not in main_firsts:
            new_parts.append(raw)
            continue
        partner_full = main_firsts[partner_first]
        if partner_full == char_name:
            new_parts.append(raw)
            continue
        partner_persona = other_personas.get(partner_full) or {}
        partner_rel = (
            partner_persona.get("relationships", {}).get("value", "")
            if isinstance(partner_persona, dict) else ""
        )
        if not partner_rel:
            new_parts.append(raw)
            continue
        partner_tier_for_me = 0
        for prole, pname, _ in _parse_relationship_entries(partner_rel):
            prl = prole.lower()
            if prl.startswith(("ex-", "former", "late-")):
                continue
            if pname.split()[0].lower() != char_first.lower():
                continue
            t = _role_tier(prl)
            if t > partner_tier_for_me:
                partner_tier_for_me = t
        if partner_tier_for_me == 0 or partner_tier_for_me >= my_tier:
            new_parts.append(raw)
            continue

        # Partner is at strictly lower tier — demote our claim.
        if partner_tier_for_me == STATUS_TIER_ENGAGED:
            target = _ENGAGED_DEMOTION.get(role_l, "fiancé")
        else:
            target = _DATING_DEMOTION.get(role_l, "partner")
        new_raw = re.sub(
            rf"^\s*{re.escape(role)}\s+is\s+",
            f"{target} is ",
            raw,
            count=1,
            flags=re.IGNORECASE,
        )
        new_parts.append(new_raw)
        alignments.append((
            role_l, target, partner_name,
            f"partner {partner_full} at lower tier ({partner_tier_for_me})",
        ))

    if not alignments:
        return value, []
    return ", ".join(new_parts), alignments


def _auto_ex_downgrade(
    rel_value: str,
    char_archive: list[dict],
    char_name: str,
    up_to_ep_tag: str,
) -> tuple[str | None, list[tuple[str, str]]]:
    """Scan ``rel_value`` for current romantic partners.  If the character's
    own archive contains an explicit breakup with a given partner at or
    before ``up_to_ep_tag``, rewrite the entry's role with an ``ex-`` prefix.

    Returns ``(new_value, downgrades)`` where ``downgrades`` is a list of
    ``(role, partner_name)`` tuples for the auto-applied changes.  When
    nothing changes the first element is ``None``.

    Notes:
      * Skips main characters as partners (handled by the inter-main guard).
      * Skips entries whose role already starts with ``ex-``.
      * Uses :func:`_has_breakup_between` for evidence detection so this
        stays in sync with the validator.
    """
    if not isinstance(rel_value, str) or not rel_value.strip():
        return None, []
    m = re.match(r"S(\d+)E(\d+)", up_to_ep_tag.upper())
    if not m:
        return None, []
    cur_season, cur_episode = int(m.group(1)), int(m.group(2))

    char_first = char_name.split()[0] if char_name else ""
    main_firsts = {n.split()[0] for n in MAIN_CHARACTERS}

    out_entries: list[str] = []
    downgrades: list[tuple[str, str]] = []

    for raw in _split_relationships_paren_aware(rel_value):
        if not raw:
            continue
        rm = _ROLE_RE.match(raw)
        if not rm:
            out_entries.append(raw)
            continue
        role = rm.group(1).strip()
        role_l = role.lower()
        if role_l.startswith("ex-") or role_l not in _DOWNGRADE_ROLES:
            out_entries.append(raw)
            continue
        rest = raw[rm.end():]
        nm = _NAME_RE.search(rest)
        if not nm:
            out_entries.append(raw)
            continue
        partner_name = nm.group(1).strip()
        partner_first = partner_name.split()[0]
        if partner_first in main_firsts:
            out_entries.append(raw)
            continue

        latest_breakup: tuple[int, int] | None = None
        latest_reconcile: tuple[int, int] | None = None
        for ar_e in char_archive:
            ar_season = ar_e.get("season")
            ar_episode = ar_e.get("episode")
            if (ar_season, ar_episode) > (cur_season, cur_episode):
                continue
            summary = ar_e.get("summary", "") or ""
            if not summary:
                continue
            ep_key = (ar_season, ar_episode)
            if _has_breakup_between(summary, char_first, partner_first):
                if latest_breakup is None or ep_key > latest_breakup:
                    latest_breakup = ep_key
            if _has_reconciliation_between(summary, char_first, partner_first):
                if latest_reconcile is None or ep_key > latest_reconcile:
                    latest_reconcile = ep_key
        if latest_breakup is None:
            out_entries.append(raw)
            continue
        # Suppress downgrade if a *later* reconciliation event exists.
        if latest_reconcile is not None and latest_reconcile >= latest_breakup:
            out_entries.append(raw)
            continue

        new_raw = re.sub(
            rf"\b{re.escape(role)}\b",
            f"ex-{role_l}",
            raw,
            count=1,
            flags=re.IGNORECASE,
        )
        out_entries.append(new_raw)
        downgrades.append((role_l, partner_name))

    if not downgrades:
        return None, []
    return ", ".join(out_entries), downgrades


_CORE_FAMILY_ROLES = (
    "wife", "husband", "spouse",
    "fiancé", "fiance", "fiancée", "fiancee",
    "mother", "father", "stepmother", "stepfather",
    "daughter", "son", "stepdaughter", "stepson",
    "sister", "brother", "stepsister", "stepbrother",
    "grandmother", "grandfather", "grandson", "granddaughter",
    "mother-in-law", "father-in-law",
    "daughter-in-law", "son-in-law",
    "twin", "co-parent", "adoptive mother", "adoptive father",
    "adoptive daughter", "adoptive son",
)


def _restore_dropped_core_roles(
    old_value: str,
    new_value: str,
    char_name: str,
    char_archive: list[dict],
) -> tuple[str, list[tuple[str, str]]]:
    """When a *replacement* merge silently drops core family / marital roles
    from ``old_value``, append them back to ``new_value`` unless the
    character's archive contains an explicit breakup / divorce signal
    between the character and the dropped partner.

    Returns ``(preserved_value, restored_entries)`` where ``restored_entries``
    is a list of ``(role, name)`` tuples that were re-attached.

    Family roles (parent/sibling/child) are *always* restored — those bonds
    don't end mid-show.  Marital roles (wife/husband/spouse/fiancé) are
    restored unless the archive shows a breakup with that partner.
    """
    if not isinstance(old_value, str) or not isinstance(new_value, str):
        return new_value, []
    old_entries = _parse_relationship_entries(old_value)
    new_entries = _parse_relationship_entries(new_value)
    new_keys = {(role.lower(), name.lower()) for role, name, _ in new_entries}
    new_names = {name.lower() for _, name, _ in new_entries}

    char_first = char_name.split()[0] if char_name else ""
    marital_set = {
        "wife", "husband", "spouse",
        "fiancé", "fiance", "fiancée", "fiancee",
    }

    restored: list[tuple[str, str]] = []
    additions: list[str] = []
    for role, name, raw in old_entries:
        clean = role.lower().removeprefix("ex-").strip()
        if clean not in _CORE_FAMILY_ROLES:
            continue
        # Skip if same role already in new
        if (role.lower(), name.lower()) in new_keys:
            continue
        # Skip if any entry in new still references this person (LLM may have
        # re-labelled e.g. "wife is Monica" → "spouse is Monica").
        if name.lower() in new_names:
            continue
        # For marital roles, allow drop only if archive shows a breakup.
        if clean in marital_set:
            partner_first = name.split()[0]
            had_breakup = False
            for ar_e in char_archive:
                summary = ar_e.get("summary", "") or ""
                if not summary:
                    continue
                if _has_breakup_between(summary, char_first, partner_first):
                    had_breakup = True
                    break
            if had_breakup:
                continue
        # For non-marital family roles, never drop.
        additions.append(raw)
        restored.append((role, name))

    if not restored:
        return new_value, []

    # Append at the end of new_value (preserve the LLM's restructure).
    sep = ", " if new_value.strip() else ""
    preserved = new_value.rstrip().rstrip(",") + sep + ", ".join(additions)
    return preserved, restored


def _split_relationships_paren_aware(value: str) -> list[str]:
    """Split a relationships string on commas that are at paren-depth 0.

    The previous implementation used :py:meth:`str.split` which broke entries
    like ``"girlfriend is Monica (public, deeply committed partnership)"`` —
    the inner comma was treated as an entry boundary, scattering descriptive
    context across multiple synthetic entries.  This helper preserves
    parenthetical clauses intact.
    """
    if not isinstance(value, str) or not value:
        return []
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in value:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            piece = "".join(buf).strip()
            if piece:
                parts.append(piece)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_relationship_entries(value: str) -> list[tuple[str, str, str]]:
    """Split a relationships string into ``(role, name, raw_entry)`` tuples.

    Only entries with the ``<role> is <Name>`` shape are returned; family
    pronoun forms like ``ex-wife is Carol (now married to Susan)`` work too
    because we capture the first proper-noun name token after ``is``.
    """
    out: list[tuple[str, str, str]] = []
    if not isinstance(value, str):
        return out
    for raw in _split_relationships_paren_aware(value):
        if not raw:
            continue
        m = _ROLE_RE.match(raw)
        if not m:
            continue
        role = m.group(1).strip().lower()
        rest = raw[m.end():]
        # Strip parenthetical clauses (e.g. "(with Ross)", "(unborn baby with Ross)")
        # so that descriptive context names are not mistaken for the entry's name.
        rest_no_paren = re.sub(r"\([^)]*\)", " ", rest)
        nm = _NAME_RE.search(rest_no_paren)
        if not nm:
            continue
        out.append((role, nm.group(1).strip(), raw))
    return out


# ---------------------------------------------------------------------------
# Decay helpers (Rule A + Rule B) — used by both the per-episode decay pass
# inside evolve_all_characters AND the legacy decay_stale_romantic script.
# ---------------------------------------------------------------------------

def _archive_ord(s: dict) -> int:
    """Convert an archive entry's (season, episode) to a sortable int."""
    return int(s["season"]) * 100 + int(s["episode"])


def _name_last_seen_in_archive(
    name: str, archive: list[dict], up_to_ord: int
) -> int:
    """Return the latest episode-ord (≤ up_to_ord) where ``name`` appears in
    any session summary as a whole word.  Returns -1 if never seen.
    """
    pat = re.compile(rf"\b{re.escape(name)}\b", re.I)
    last = -1
    for s in archive:
        o = _archive_ord(s)
        if o > up_to_ord:
            continue
        if pat.search(s.get("summary", "")):
            if o > last:
                last = o
    return last


def _has_strict_marriage_evidence(text: str, name_a: str, name_b: str) -> bool:
    """Conservative reconciliation guard for the decay pass.

    Only honours canonical marriage / move-in evidence — fake-out engagements
    do NOT count as protection from decay.
    """
    if not text:
        return False
    a = re.escape(name_a.split()[0])
    b = re.escape(name_b.split()[0])
    sentences = re.split(r"(?<=[.!?])\s+|;\s*", text.lower())
    pat_marriage = (
        r"\b(?:got\s+married|married\s+each\s+other|"
        r"(?:say|said|saying)\s+(?:i\s+do|their\s+vows)|"
        r"their\s+wedding\s+day|wedding\s+day|"
        r"move(?:s|d|ing)?\s+in\s+together)\b"
    )
    for sent in sentences:
        if not (re.search(rf"\b{a}\b", sent, re.I)
                and re.search(rf"\b{b}\b", sent, re.I)):
            continue
        if re.search(pat_marriage, sent, re.I):
            return True
    return False


def _is_reciprocated(other_rel: str, char_first_name: str) -> bool:
    """Does ``other_rel`` list ``char_first_name`` as a *current* romantic
    partner (any role in ROMANTIC_TRANSIENT ∪ PROTECTED_CURRENT)?
    """
    rom_roles = ROMANTIC_TRANSIENT | PROTECTED_CURRENT
    for role, name, _raw in _parse_relationship_entries(other_rel):
        role_l = role.lower()
        if role_l.startswith("ex-") or role_l.startswith("late-"):
            continue
        if role_l in rom_roles and name.lower() == char_first_name.lower():
            return True
        for r in rom_roles:
            if role_l.startswith(r) and name.lower() == char_first_name.lower():
                return True
    return False


def _is_listed_as_ex(other_rel: str, char_first_name: str) -> bool:
    """Does ``other_rel`` affirmatively list ``char_first_name`` with an
    ``ex-`` romantic role?
    """
    for role, name, _raw in _parse_relationship_entries(other_rel):
        role_l = role.lower()
        if not role_l.startswith("ex-"):
            continue
        if name.lower() != char_first_name.lower():
            continue
        base = role_l[3:].strip()
        if base in ROMANTIC_TRANSIENT or base in PROTECTED_CURRENT:
            return True
        for r in ROMANTIC_TRANSIENT | PROTECTED_CURRENT:
            if base.startswith(r):
                return True
    return False


def _other_main_full_name(
    partner: str, main_chars: list[str]
) -> str | None:
    """Resolve a partner first name to a full main-character name."""
    first = partner.split()[0].lower()
    by_first = {c.split()[0].lower(): c for c in main_chars}
    return by_first.get(first)


def _partner_role_at(snaps: dict | None, char_name: str, ep_id: str,
                     partner_first: str) -> str | None:
    """Return the (lowercase) role label by which ``char_name`` references
    ``partner_first`` in the snapshot at ``ep_id``, or None.  Skips entries
    that are already demoted (ex- / former / late-).
    """
    if snaps is None:
        return None
    snap = snaps.get(char_name, {}).get(ep_id)
    if not snap:
        return None
    rel = (snap.get("tree", {}).get("persona", {})
           .get("relationships", {}).get("value", ""))
    for role, name, _ in _parse_relationship_entries(rel):
        rl = role.lower().strip()
        if rl.startswith(("ex-", "former", "late-")):
            continue
        if name.split()[0].lower() == partner_first.lower():
            return rl
    return None


def _partner_continuous_around_ep(
    snaps: dict | None, char_name: str, ep_id: str, partner_first: str,
) -> bool:
    """True iff ``partner_first`` appears as a current couple role for
    ``char_name`` at BOTH the immediately preceding AND the immediately
    following episode (in the snapshot index).  Used by the decay logic
    to short-circuit when the relationship is clearly part of an ongoing
    arc (e.g. one filled by ``forward_fill_continuity``).
    """
    if snaps is None or char_name not in snaps:
        return False
    eps_sorted = sorted(snaps[char_name].keys(), key=_episode_to_int)
    if ep_id not in snaps[char_name]:
        return False
    idx = eps_sorted.index(ep_id)
    if idx == 0 or idx == len(eps_sorted) - 1:
        return False
    prev_ep = eps_sorted[idx - 1]
    next_ep = eps_sorted[idx + 1]
    return (
        _partner_role_at(snaps, char_name, prev_ep, partner_first) is not None
        and _partner_role_at(snaps, char_name, next_ep, partner_first) is not None
    )


def decay_relationships_value(
    value: str,
    char_name: str,
    ep_id: str,
    char_archive: list[dict],
    current_trees: dict | None,
    snaps_so_far: dict | None,
    first_seen_ord: dict,
    main_chars: list[str],
) -> tuple[str, list[dict]]:
    """Apply Rule A + Rule B decay to ``value`` (a relationships string).

    Two-source inter-main lookup: prefers ``current_trees`` (fresh state for
    the same episode) and falls back to ``snaps_so_far`` (prior episodes).
    Returns ``(new_value, decay_records)``.

    Idempotency guard: an entry is skipped if both the immediately previous
    AND immediately following snapshot for the same character carry the
    same partner as a current couple role.  This prevents oscillation when
    the pipeline is rerun after ``forward_fill_continuity`` has filled a
    legitimate gap.
    """
    if not value:
        return value, []
    cur_ord = _episode_to_int(ep_id)
    char_first = char_name.split()[0]
    decays: list[dict] = []
    entries = _parse_relationship_entries(value)

    recon_cache: dict[str, bool] = {}

    def _has_recon(partner: str) -> bool:
        if partner not in recon_cache:
            text_blocks = [
                s["summary"] for s in char_archive
                if _archive_ord(s) <= cur_ord
            ]
            joined = "\n".join(text_blocks)
            recon_cache[partner] = _has_strict_marriage_evidence(
                joined, char_first, partner.split()[0]
            )
        return recon_cache[partner]

    new_entries: list[str] = []
    for role, name, raw in entries:
        role_l = role.lower().strip()
        # Already-demoted entries — skip.  Covers ``ex-`` (canonical),
        # ``late-`` (deceased) and ``former `` / ``former-`` (encounter-style
        # demotion produced by this very pass on a prior run).
        if (role_l.startswith("ex-")
                or role_l.startswith("late-")
                or role_l.startswith("former ")
                or role_l.startswith("former-")):
            new_entries.append(raw)
            continue
        if role_l in PROTECTED_CURRENT or any(
            role_l.startswith(p + " ") for p in PROTECTED_CURRENT
        ):
            new_entries.append(raw)
            continue
        if role_l not in ROMANTIC_TRANSIENT and not any(
            role_l.startswith(r) for r in ROMANTIC_TRANSIENT
        ):
            new_entries.append(raw)
            continue

        partner_first = name.split()[0]
        if _partner_continuous_around_ep(
            snaps_so_far, char_name, ep_id, partner_first
        ):
            new_entries.append(raw)
            continue

        decay_reason: str | None = None
        other_full = _other_main_full_name(name, main_chars)

        # Unilateral romantic roles do not require the other side to
        # reciprocate — skip Rule B and fall through to Rule A.
        if role_l in UNILATERAL_ROMANTIC_ROLES:
            other_full = None

        if other_full and other_full != char_name:
            other_rel = ""
            if current_trees is not None and other_full in current_trees:
                other_rel = (
                    current_trees[other_full]
                    .get("persona", {})
                    .get("relationships", {})
                    .get("value", "")
                )
            if not other_rel and snaps_so_far is not None:
                other_eps = sorted(
                    snaps_so_far.get(other_full, {}).keys(),
                    key=_episode_to_int,
                )
                use_ep = None
                for e in reversed(other_eps):
                    if _episode_to_int(e) <= cur_ord:
                        use_ep = e
                        break
                if use_ep:
                    other_rel = (
                        snaps_so_far[other_full][use_ep]
                        ["tree"]["persona"]["relationships"]
                        .get("value", "")
                    )

            if _is_listed_as_ex(other_rel, char_first):
                decay_reason = (
                    f"rule-B1: {other_full} lists {char_first} as ex-"
                )
            else:
                key = (char_name, f"{role_l}|{name.lower()}")
                seen_at = first_seen_ord.get(key, cur_ord)
                age = cur_ord - seen_at
                if age >= DECAY_RULE_B_LONG_SILENCE:
                    if not _is_reciprocated(other_rel, char_first) and \
                       not _is_listed_as_ex(other_rel, char_first):
                        decay_reason = (
                            f"rule-B2: {other_full} no reciprocation "
                            f"in {age} eps"
                        )
        else:
            last = _name_last_seen_in_archive(name, char_archive, cur_ord)
            if last > 0 and cur_ord - last > DECAY_THRESHOLD_NON_MAIN_EP:
                decay_reason = (
                    f"rule-A: last archive mention {last} "
                    f"(>{DECAY_THRESHOLD_NON_MAIN_EP} eps ago)"
                )

        if decay_reason and _has_recon(name):
            decay_reason = None

        if decay_reason:
            # Encounter-style roles read better as "former ..." than "ex-..."
            if role_l in DECAY_FORMER_PREFIX:
                new_role = f"former {role}"
            else:
                new_role = f"ex-{role}"
            new_raw = re.sub(
                rf"^\s*{re.escape(role)}\s+is\s+",
                f"{new_role} is ",
                raw,
                count=1,
            )
            new_entries.append(new_raw)
            decays.append({
                "character": char_name,
                "episode": ep_id,
                "from_entry": raw,
                "to_entry": new_raw,
                "reason": decay_reason,
            })
        else:
            new_entries.append(raw)

    # Dedupe (prefer ex- when both forms exist for same partner+role)
    seen_by_name: dict[str, str] = {}
    deduped: list[str] = []
    for raw in new_entries:
        m = re.match(r"\s*(ex-)?([\w \-]+?)\s+is\s+(.+)", raw)
        if not m:
            deduped.append(raw)
            continue
        base_role = m.group(2).strip().lower()
        rest = m.group(3).strip()
        rest_no_paren = re.sub(r"\([^)]*\)", " ", rest)
        nm = re.search(r"\b([A-Z][\w]+(?:\s+[A-Z][\w]+)*)", rest_no_paren)
        if not nm:
            deduped.append(raw)
            continue
        name_key = nm.group(1).strip().lower()
        key = f"{base_role}::{name_key}"
        is_ex = bool(m.group(1))
        if key in seen_by_name:
            prev = seen_by_name[key]
            prev_ex = prev.lstrip().lower().startswith("ex-")
            if is_ex and not prev_ex:
                idx = deduped.index(prev)
                deduped[idx] = raw
                seen_by_name[key] = raw
        else:
            seen_by_name[key] = raw
            deduped.append(raw)

    return ", ".join(s.strip() for s in deduped if s.strip()), decays


def _detect_unwarranted_ex_downgrades(
    char_name: str,
    old_value: str,
    new_value: str,
    consumed_ids: list[str],
    char_archive: list[dict],
    all_archives: dict[str, list[dict]] | None,
) -> list[tuple[str, str, str, str]]:
    """Find ``ex-<role> is <Name>`` entries in *new_value* that:

      * have a corresponding non-ex ``<role> is <Name>`` in *old_value*, AND
      * lack any breakup keyword in the partner's name window of either
        the character's own archive (consumed scenes) or the partner's
        archive (within ±2 episodes of consumed scenes).

    Returns a list of ``(role, name, old_entry, new_entry)`` tuples.
    """
    if not isinstance(old_value, str) or not isinstance(new_value, str):
        return []

    old_entries = _parse_relationship_entries(old_value)
    new_entries = _parse_relationship_entries(new_value)

    # Index old by lowercased name to detect downgrades.
    old_by_name: dict[str, tuple[str, str]] = {}
    for role, name, raw in old_entries:
        # Only track relationships that could be downgraded.
        clean_role = role.removeprefix("ex-").strip()
        if any(r == clean_role for r in _DOWNGRADE_ROLES):
            old_by_name[name.lower()] = (role, raw)

    # Episodes covered by consumed_ids
    consumed_eps: set[tuple[int, int]] = set()
    for sid in consumed_ids:
        m = re.match(r"s(\d+)_e(\d+)_", sid)
        if m:
            consumed_eps.add((int(m.group(1)), int(m.group(2))))

    char_first = char_name.split()[0].lower() if char_name else ""
    main_first_to_full = (
        {n.split()[0]: n for n in all_archives.keys()} if all_archives else {}
    )

    flagged: list[tuple[str, str, str, str]] = []
    for role, name, raw in new_entries:
        if not role.startswith("ex-"):
            continue
        clean_role = role[3:]
        if clean_role not in _DOWNGRADE_ROLES:
            continue
        if name.lower() not in old_by_name:
            continue
        old_role, _old_raw = old_by_name[name.lower()]
        # Already an ex- in the old value? then no downgrade happened.
        if old_role.startswith("ex-"):
            continue

        # Look for breakup evidence specifically between char and partner.
        partner_first = name.split()[0]
        evidence = False

        # 1) Char's own consumed scenes
        for e in char_archive:
            if e.get("scene_id") not in consumed_ids:
                continue
            summary = e.get("summary") or ""
            if _has_breakup_between(summary, char_first, partner_first):
                evidence = True
                break

        # 2) Partner's own archive within ±2 of consumed eps
        if (not evidence) and all_archives is not None and consumed_eps:
            partner_full = main_first_to_full.get(partner_first)
            partner_archive = all_archives.get(partner_full, []) if partner_full else []
            for e in partner_archive:
                if not any(e["season"] == s and abs(e["episode"] - ep) <= 2
                           for (s, ep) in consumed_eps):
                    continue
                summary = e.get("summary") or ""
                if _has_breakup_between(summary, char_first, partner_first):
                    evidence = True
                    break

        if not evidence:
            flagged.append((role, name, _old_raw, raw))
    return flagged


def _strip_unwarranted_ex_downgrades(
    new_value: str, downgrades: list[tuple[str, str, str, str]],
) -> str:
    """Replace ``ex-<role> is <Name>`` with the previous ``<role> is <Name>``
    (taken from ``old_entry``) for each flagged downgrade.
    """
    out = new_value
    for _role, _name, old_entry, new_entry in downgrades:
        out = out.replace(new_entry, old_entry)
    return out


def _detect_borrowed_romantic_partners(
    char_name: str,
    old_value: str,
    new_value: str,
    other_personas: dict[str, dict] | None,
) -> list[tuple[str, str, str, str]]:
    """Detect entries that NEW-add a romantic partner X to ``char_name``'s
    relationships, when X is already a romantic partner of ANOTHER main
    character's persona.  Catches LLM errors like Monica saying
    "ex-boyfriend is Paolo" after only witnessing Rachel's breakup with
    Paolo (Paolo belongs to Rachel, not Monica).

    Returns a list of ``(role, name, raw_entry, owner_full_name)`` tuples.
    """
    if not other_personas:
        return []
    if not isinstance(old_value, str) or not isinstance(new_value, str):
        return []

    old_entries = _parse_relationship_entries(old_value)
    new_entries = _parse_relationship_entries(new_value)

    old_keys = {(role.lower(), name.lower()) for role, name, _ in old_entries}

    main_first_names = {n.split()[0].lower() for n in other_personas.keys()}

    flagged: list[tuple[str, str, str, str]] = []
    for role, name, raw in new_entries:
        # Only consider NEW entries (not previously present).
        if (role.lower(), name.lower()) in old_keys:
            continue
        # Only romantic roles can be "borrowed".
        clean_role = role.removeprefix("ex-").strip().lower()
        if clean_role not in _DOWNGRADE_ROLES:
            continue
        # Skip when X is itself a tracked main character (Ross-Rachel etc.).
        if name.split()[0].lower() in main_first_names:
            continue
        # Check other main personas for this name as their romantic partner.
        for other_name, persona in other_personas.items():
            if other_name == char_name:
                continue
            other_rel = (persona.get("relationships") or {}).get("value", "")
            for o_role, o_name, _ in _parse_relationship_entries(other_rel):
                if o_name.lower() != name.lower():
                    continue
                o_clean = o_role.removeprefix("ex-").strip().lower()
                if o_clean in _DOWNGRADE_ROLES:
                    flagged.append((role, name, raw, other_name))
                    break
            else:
                continue
            break
    return flagged


def _strip_borrowed_romantic_partners(
    new_value: str, borrowed: list[tuple[str, str, str, str]],
) -> str:
    """Remove flagged entries from a relationships string."""
    if not borrowed:
        return new_value
    drop = {entry for _, _, entry, _ in borrowed}
    kept = [
        e for e in _split_relationships_paren_aware(new_value)
        if e and e not in drop
    ]
    return ", ".join(kept)


# Romantic / relationship keywords that signal a partner-of relationship.
_REL_NOUN_PATTERN = (
    r"breakup|break-up|relationship|romance|date|dates|dating|kiss(?:es|ed|ing)?|"
    r"engagement|fianc[ée]+|wedding|marriage|"
    r"girlfriend|boyfriend|husband|wife|partner|"
    r"ex-(?:girlfriend|boyfriend|wife|husband|fianc[ée]+|partner)"
)


def _partner_owner_in_session(
    summary: str, partner_first: str, focal_first: str,
    main_first_names: set[str],
) -> str | None:
    """Heuristic: if the partner is bound to ANOTHER main character via a
    possessive / supporting-friend phrase in this scene summary, return that
    other character's first name (lowercase).  Otherwise None.
    """
    tl = summary.lower()
    pf = re.escape(partner_first.lower())
    if not re.search(rf"\b{pf}\b", tl):
        return None
    focal = focal_first.lower()
    for mn in main_first_names:
        if mn == focal or mn == partner_first.lower():
            continue
        m = re.escape(mn)
        patterns = [
            # "<other>'s ... <partner>"   (possessive)
            rf"\b{m}'s\s+(?:\w+\s+){{0,4}}\b{pf}\b",
            # "<other> through/after/before/despite (her|his|their)
            #  breakup/relationship/etc. with <partner>"
            rf"\b{m}\b\s+(?:\w+\s+){{0,4}}(?:through|after|before|"
            rf"despite|amid|over|about)\s+(?:her|his|their|the)\s+"
            rf"(?:{_REL_NOUN_PATTERN})\s+(?:\w+\s+){{0,3}}\bwith\s+{pf}\b",
            # "<other>'s breakup/relationship/dating with <partner>"
            rf"\b{m}'s\s+(?:\w+\s+){{0,3}}(?:{_REL_NOUN_PATTERN})\s+"
            rf"(?:\w+\s+){{0,3}}\bwith\s+{pf}\b",
            # "<other> ... breakup/relationship/etc. with <partner>"
            rf"\b{m}\s+(?:\w+\s+){{0,5}}(?:{_REL_NOUN_PATTERN})\s+"
            rf"(?:\w+\s+){{0,3}}\bwith\s+{pf}\b",
            # "<other> (and <pf>|<pf> and <other>) <breakup/dating verb>"
            rf"\b{m}\s+and\s+{pf}\b",
            rf"\b{pf}\s+and\s+{m}\b",
        ]
        if any(re.search(p, tl) for p in patterns):
            return mn
    return None


def _detect_borrowed_via_consumed_sessions(
    char_name: str,
    old_value: str,
    new_value: str,
    consumed_ids: list[str],
    char_archive: list[dict],
    main_first_names: set[str],
) -> list[tuple[str, str, str, str]]:
    """Detect NEW romantic-role entries whose partner is bound to ANOTHER
    main character within the consumed session summaries themselves.
    Catches errors like Monica adding "ex-boyfriend is Paolo" when her
    only Paolo-mention is "Monica supported Rachel through her breakup with
    Paolo".

    Returns ``(role, name, raw_entry, owner_first_name)`` tuples.
    """
    if not isinstance(old_value, str) or not isinstance(new_value, str):
        return []
    old_keys = {(role.lower(), name.lower())
                for role, name, _ in _parse_relationship_entries(old_value)}
    new_entries = _parse_relationship_entries(new_value)
    char_first = char_name.split()[0]

    # Index summaries by scene_id for fast lookup
    by_id: dict[str, str] = {
        e.get("scene_id"): (e.get("summary") or "")
        for e in char_archive
    }

    flagged: list[tuple[str, str, str, str]] = []
    for role, name, raw in new_entries:
        if (role.lower(), name.lower()) in old_keys:
            continue
        clean_role = role.removeprefix("ex-").strip().lower()
        if clean_role not in _DOWNGRADE_ROLES and clean_role not in {
            "dating", "dated"
        }:
            continue
        # Skip if X is a tracked main character (Ross-Rachel etc.).
        if name.split()[0].lower() in main_first_names:
            continue
        # Look for ownership signals across consumed scenes
        owner_votes: dict[str, int] = {}
        focal_votes = 0
        for sid in consumed_ids:
            summary = by_id.get(sid, "")
            if not summary or name.split()[0].lower() not in summary.lower():
                continue
            owner = _partner_owner_in_session(
                summary, name.split()[0], char_first, main_first_names,
            )
            if owner:
                owner_votes[owner] = owner_votes.get(owner, 0) + 1
            else:
                # Partner appears but no other-owner signal — count as
                # ambiguous toward focal char (only counts when summary
                # mentions focal char too).
                if re.search(rf"\b{re.escape(char_first.lower())}\b",
                             summary.lower()):
                    focal_votes += 1
        if owner_votes and max(owner_votes.values()) > focal_votes:
            top = max(owner_votes.items(), key=lambda kv: kv[1])[0]
            flagged.append((role, name, raw, top))
    return flagged


def _normalize_scene_id(raw: str, valid_ids: set[str]) -> str | None:
    """Coerce an LLM-cited scene_id into the canonical archive format.

    Handles common LLM formatting glitches:
      ``S01E12_c10``         -> ``s01_e12_c10``
      ``S01E22, scene=s01_e22_c02`` -> ``s01_e22_c02``
      ``Scene s01-e12-c10``  -> ``s01_e12_c10``
    Returns None if no canonical match exists in ``valid_ids``.
    """
    if not isinstance(raw, str):
        return None
    if raw in valid_ids:
        return raw
    candidates = _SCENE_RE.findall(raw)
    for s, e, c in candidates:
        cand = f"s{int(s):02d}_e{int(e):02d}_c{int(c):02d}"
        if cand in valid_ids:
            return cand
    return None


def validate_decision(
    decision: dict,
    active: list[dict],
    persona: dict,
    current_ep: str,
    char_name: str = "",
    all_archives: dict[str, list[dict]] | None = None,
    other_personas: dict[str, dict] | None = None,
    lifetime_high: list[dict] | None = None,
) -> tuple[dict, list[str]]:
    """Reject field changes that violate hard constraints.

    ``active`` is the recent-evidence Track A set.  ``lifetime_high`` is the
    optional Track B set (all high-sig events to date, including those
    previously consumed by low-resistance updates).  For ``moderate`` /
    ``core`` fields, citations from Track B are allowed since those tiers
    consume from the lifetime view.

    Returns the (possibly modified) decision and a list of rejection reasons.
    """
    rejections: list[str] = []
    if not decision.get("should_update"):
        return decision, rejections

    valid_ids_low = {e["scene_id"] for e in active}
    by_id = {e["scene_id"]: e for e in active}
    if lifetime_high:
        for e in lifetime_high:
            by_id.setdefault(e["scene_id"], e)
    valid_ids_lifetime = (
        {e["scene_id"] for e in lifetime_high}
        if lifetime_high
        else set()
    ) | valid_ids_low

    new_changes: dict[str, dict] = {}
    for field, change in (decision.get("changes") or {}).items():
        if not isinstance(change, dict):
            rejections.append(f"{field}: malformed change object")
            continue

        # 0. Tier lookup — moderate/core may cite from the lifetime archive
        field_def_pre = persona.get(field) or {}
        field_tier = (field_def_pre.get("resistance") or "low").lower()
        valid_ids_for_field = (
            valid_ids_lifetime
            if field_tier in ("moderate", "core")
            else valid_ids_low
        )

        # 1. Whitelist consumed_session_ids (with format normalization)
        raw_ids = change.get("consumed_session_ids", [])
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        ids: list[str] = []
        for sid in raw_ids:
            norm = _normalize_scene_id(sid, valid_ids_for_field)
            if norm and norm not in ids:
                ids.append(norm)
        if not ids:
            rejections.append(
                f"{field}: no valid consumed_session_ids "
                f"(LLM cited {raw_ids})")
            continue

        # 2. Field must exist in persona
        field_def = persona.get(field)
        if not isinstance(field_def, dict) or "value" not in field_def:
            rejections.append(f"{field}: field not in persona schema")
            continue

        # 3. Threshold check by resistance level
        resistance = field_def.get("resistance", "low")
        thresh = THRESHOLDS.get(resistance, THRESHOLDS["low"])

        evidence = [by_id[sid] for sid in ids]
        n_eps = len({(e["season"], e["episode"]) for e in evidence})
        n_high = sum(1 for e in evidence if e["significance"] == "high")
        n_med = sum(1 for e in evidence if e["significance"] == "medium")

        if thresh.get("min_high_or_2_medium"):
            if n_high < 1 and n_med < 2:
                rejections.append(
                    f"{field} (low): need 1 high or 2 medium, "
                    f"got {n_high}h+{n_med}m")
                continue
        else:
            if n_eps < thresh["min_episodes"]:
                rejections.append(
                    f"{field} ({resistance}): only {n_eps} episodes, "
                    f"need {thresh['min_episodes']}")
                continue
            if n_high < thresh["min_high"]:
                rejections.append(
                    f"{field} ({resistance}): only {n_high} high events, "
                    f"need {thresh['min_high']}")
                continue

        # 4. Cooldown check
        last_updated = field_def.get("last_updated_at")
        if last_updated:
            ep_diff = _episode_to_int(current_ep) - _episode_to_int(last_updated)
            if ep_diff < thresh["cooldown"]:
                rejections.append(
                    f"{field} ({resistance}): cooldown {ep_diff}<"
                    f"{thresh['cooldown']} since {last_updated}")
                continue

        merge_type = change.get("merge_type", "incremental")
        new_val = change.get("new_value", "")
        old_val = field_def.get("value", "")

        # 5. Reject empty / blank new values
        if not str(new_val).strip():
            rejections.append(f"{field}: empty new_value")
            continue

        # 6. Incremental safety: keep ≥ 80% of old length
        if merge_type != "replacement" and old_val:
            if len(new_val) < INCREMENTAL_MIN_RATIO * len(old_val):
                rejections.append(
                    f"{field}: incremental merge dropped too much "
                    f"({len(new_val)}/{len(old_val)})")
                continue

        # 6b. Incremental safety: ≥50 % of old's distinctive words must
        # survive in new_val.  Catches LLM "stealth replacement" where the
        # text has incremental length but is actually a wholesale rewrite
        # (e.g. cross-character template contamination).
        if merge_type != "replacement" and old_val:
            recall = _incremental_word_recall(old_val, new_val)
            if recall < INCREMENTAL_MIN_WORD_RECALL:
                rejections.append(
                    f"{field}: incremental merge dropped too many old "
                    f"descriptors (word-recall={recall:.2f}<"
                    f"{INCREMENTAL_MIN_WORD_RECALL}); likely stealth "
                    f"replacement"
                )
                continue

        # 7. No-op detection: reject if new value is identical to old
        if str(new_val).strip() == str(old_val).strip():
            rejections.append(f"{field}: no-op update (value unchanged)")
            continue

        # 8. Entity de-duplication for relationships field
        if field == "relationships":
            cleaned = _dedupe_relationships(new_val)
            if cleaned != new_val:
                rejections.append(
                    f"{field}: deduped (was '{new_val[:80]}...')")
                new_val = cleaned
                # Re-check no-op after dedup
                if str(new_val).strip() == str(old_val).strip():
                    rejections.append(
                        f"{field}: no-op after dedup, skipping")
                    continue

            # 8b. Critical-role preservation for ANY merge:
            # If a merge silently drops a *core family/marital* role from
            # the previous value, append it back unless the character's
            # archive provides explicit evidence the bond ended (breakup,
            # divorce, death).  This catches LLMs that re-state the field
            # and accidentally omit "wife is X" / "husband is X" — happens
            # both with replacement merges (drops in re-statement) and
            # incremental merges (where the LLM rewrote so much that core
            # roles got squeezed out).
            if (
                old_val
                and char_name
                and all_archives is not None
            ):
                preserved, restored = _restore_dropped_core_roles(
                    old_val, new_val, char_name,
                    all_archives.get(char_name, []),
                )
                if restored:
                    pretty = ", ".join(
                        f"{role} {nm}" for role, nm in restored
                    )
                    rejections.append(
                        f"{field}: replacement dropped core role(s) "
                        f"[{pretty}] — restored from previous value"
                    )
                    new_val = preserved
                    if str(new_val).strip() == str(old_val).strip():
                        rejections.append(
                            f"{field}: no-op after core-role restore, skipping"
                        )
                        continue

            # 9. ex- downgrade requires explicit breakup evidence
            if char_name:
                char_archive = (
                    all_archives.get(char_name, []) if all_archives else []
                )
                downgrades = _detect_unwarranted_ex_downgrades(
                    char_name, old_val, new_val, ids,
                    char_archive, all_archives,
                )
                if downgrades:
                    sanitized = _strip_unwarranted_ex_downgrades(
                        new_val, downgrades,
                    )
                    for role, name, _, _ in downgrades:
                        rejections.append(
                            f"{field}: unwarranted '{role} is {name}' "
                            f"downgrade — no breakup evidence in either archive"
                        )
                    new_val = sanitized
                    if str(new_val).strip() == str(old_val).strip():
                        rejections.append(
                            f"{field}: no-op after ex-strip, skipping")
                        continue

            # 9b. Borrowed-partner cross-check (persona-state level)
            if char_name and other_personas:
                borrowed = _detect_borrowed_romantic_partners(
                    char_name, old_val, new_val, other_personas,
                )
                if borrowed:
                    for role, name, _, owner in borrowed:
                        rejections.append(
                            f"{field}: borrowed '{role} is {name}' from "
                            f"{owner} — that partner already belongs to "
                            f"another main character"
                        )
                    sanitized = _strip_borrowed_romantic_partners(
                        new_val, borrowed,
                    )
                    new_val = sanitized
                    if str(new_val).strip() == str(old_val).strip():
                        rejections.append(
                            f"{field}: no-op after borrowed-partner strip, "
                            f"skipping")
                        continue

            # 9c. Borrowed-partner cross-check (session-summary level)
            if char_name and all_archives is not None and other_personas is not None:
                main_firsts = {n.split()[0].lower() for n in all_archives.keys()}
                char_archive_full = all_archives.get(char_name, [])
                borrowed_sess = _detect_borrowed_via_consumed_sessions(
                    char_name, old_val, new_val, ids,
                    char_archive_full, main_firsts,
                )
                if borrowed_sess:
                    for role, name, _, owner in borrowed_sess:
                        rejections.append(
                            f"{field}: borrowed '{role} is {name}' — session "
                            f"evidence binds them to {owner.title()}, not "
                            f"{char_name.split()[0]}"
                        )
                    sanitized = _strip_borrowed_romantic_partners(
                        new_val, borrowed_sess,
                    )
                    new_val = sanitized
                    if str(new_val).strip() == str(old_val).strip():
                        rejections.append(
                            f"{field}: no-op after session-borrowed strip, "
                            f"skipping")
                        continue

            # 10. Inter-main-character romantic-claim cross-check
            # Skip entries already present in ``old_val`` to avoid undoing a
            # core-role restore (8b) — those entries were carried forward
            # from the previous persona version and were either validated
            # when first introduced or have been inherited unchallenged.
            if char_name and all_archives is not None:
                inter_issues = _check_inter_main_romantic(
                    char_name, new_val, ids, all_archives,
                    old_value=old_val,
                )
                if inter_issues:
                    rejections.extend(inter_issues)
                    sanitized = _strip_inter_main_romantic(
                        char_name, new_val, ids, all_archives,
                        old_value=old_val,
                    )
                    if sanitized.strip() == old_val.strip():
                        rejections.append(
                            f"{field}: no-op after inter-main strip, skipping")
                        continue
                    new_val = sanitized

            # 11. Status-tier evidence guard:
            # ``husband/wife/spouse`` and ``fiancé/fiancée`` claims against
            # another main require explicit marriage / engagement evidence in
            # the cumulative archive.  Without it, demote to the highest
            # evidenced tier (engaged → dating, married → engaged → dating).
            # Catches LLMs that prematurely escalate status before the
            # canonical wedding / proposal episode.
            if char_name and all_archives is not None:
                main_chars_list = list(all_archives.keys())
                char_archive_full = all_archives.get(char_name, [])
                tier_val, tier_demos = _validate_status_tier_evidence(
                    new_val, char_name, char_archive_full, main_chars_list,
                )
                if tier_demos:
                    for old_r, new_r, partner_nm, why in tier_demos:
                        rejections.append(
                            f"{field}: status-tier demote "
                            f"'{old_r} is {partner_nm}' -> '{new_r} is "
                            f"{partner_nm}' ({why})"
                        )
                    new_val = tier_val
                    if str(new_val).strip() == str(old_val).strip():
                        rejections.append(
                            f"{field}: no-op after tier demote, skipping")
                        continue

            # 12. Inter-main couple continuity:
            # If a merge silently dropped a current-couple role (boyfriend /
            # girlfriend / partner / fiancé / wife / ...) between
            # ``char_name`` and another main, restore it unless the archive
            # has explicit breakup evidence.  Mirrors step 8b but for the
            # transient-couple tier — guards against "regression to None"
            # bugs (e.g. Monica ↔ Chandler S05E11; Ross ↔ Rachel S02E14–E15).
            if (char_name and all_archives is not None
                    and old_val and field == "relationships"):
                main_chars_list = list(all_archives.keys())
                char_archive_full = all_archives.get(char_name, [])
                preserved2, restored2 = _restore_dropped_inter_main_couple(
                    old_val, new_val, char_name,
                    char_archive_full, main_chars_list,
                )
                if restored2:
                    pretty = ", ".join(
                        f"{r} is {n}" for r, n in restored2
                    )
                    rejections.append(
                        f"{field}: replacement dropped current-couple "
                        f"role(s) [{pretty}] — restored from previous value"
                    )
                    new_val = preserved2
                    if str(new_val).strip() == str(old_val).strip():
                        rejections.append(
                            f"{field}: no-op after couple restore, skipping")
                        continue

            # 13. Inter-main status alignment:
            # If ``char_name`` claims a tier-N couple role (e.g. ``husband``)
            # with a main partner whose CURRENT persona is at a strictly
            # lower tier (e.g. partner still says ``girlfriend``), demote our
            # claim to match the partner's tier.  Only kicks in when
            # ``other_personas`` is supplied — i.e. during the live evolve
            # loop where each character's freshest persona is available.
            if (char_name and other_personas and field == "relationships"
                    and all_archives is not None):
                main_chars_list = list(all_archives.keys())
                aligned_val, alignments = _align_inter_main_status(
                    new_val, char_name, other_personas, main_chars_list,
                )
                if alignments:
                    for old_r, new_r, partner_nm, why in alignments:
                        rejections.append(
                            f"{field}: status-align "
                            f"'{old_r} is {partner_nm}' -> '{new_r} is "
                            f"{partner_nm}' ({why})"
                        )
                    new_val = aligned_val
                    if str(new_val).strip() == str(old_val).strip():
                        rejections.append(
                            f"{field}: no-op after status-align, skipping")
                        continue

        new_changes[field] = {
            "new_value": new_val,
            "merge_type": merge_type,
            "consumed_session_ids": ids,
        }

    decision["changes"] = new_changes
    if not new_changes:
        decision["should_update"] = False
        decision.pop("backstory_addendum", None)
    return decision, rejections


# ---------------------------------------------------------------------------
# Async LLM evaluation
# ---------------------------------------------------------------------------

async def evaluate_persona_change(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    character: str,
    identity: dict,
    persona: dict,
    track_a_entries: list[dict],
    track_b_entries: list[dict],
    max_retries: int = 3,
) -> dict | None:
    persona_text = _format_persona_for_prompt(persona)
    track_a_text = _format_archive_for_prompt(track_a_entries)
    track_b_text = _format_archive_for_prompt(track_b_entries)
    ev = _evidence_summary(track_b_entries)

    drift_signal = _drift_signal_block(persona)

    evidence_block = (
        f"## Lifetime evidence summary (Track B)\n"
        f"- Unique episodes with high-sig events: {ev['unique_episodes']}\n"
        f"- Total high-sig events: {ev['high_events']}\n"
        f"- Moderate-tier threshold (≥ 3 eps): "
        f"{'MET' if ev['moderate_threshold_met'] else 'not yet met'}\n"
        f"- Core-tier threshold "
        f"(≥ {THRESHOLDS['core']['min_episodes']} eps AND "
        f"≥ {THRESHOLDS['core']['min_high']} high): "
        f"{'MET' if ev['core_threshold_met'] else 'not yet met'}\n"
        f"\n"
        f"If a tier's threshold is *not yet met*, you cannot update fields at "
        f"that tier — the system will reject them.  Even if MET, only update "
        f"when the lifetime arc actually shows directional drift.\n"
        f"{drift_signal}"
    )

    user_prompt = (
        f"## Character: {character}\n\n"
        f"## Current identity\n"
        f"Backstory: {identity.get('backstory', 'N/A')}\n\n"
        f"## Current Persona (version {persona.get('version', '?')})\n"
        f"{persona_text}\n\n"
        f"{evidence_block}\n"
        f"## Track A — RECENT EVIDENCE (active sessions since last update; "
        f"use for relationships / occupation / demographics / hobbies)\n"
        f"{len(track_a_entries)} entries:\n"
        f"{track_a_text}\n\n"
        f"## Track B — LIFETIME PATTERN ARCHIVE (all high-significance events "
        f"to date; use for behavioral_tendencies / personality / "
        f"speaking_style)\n"
        f"{len(track_b_entries)} entries:\n"
        f"{track_b_text}\n"
    )

    async with semaphore:
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=3000,
                    timeout=120.0,
                )
                return _parse_json(resp.choices[0].message.content)
            except json.JSONDecodeError as e:
                sys.stderr.write(
                    f"\n    [{character}] JSON error (attempt {attempt}): {e}\n")
            except Exception as e:
                sys.stderr.write(
                    f"\n    [{character}] Error (attempt {attempt}): {e}\n")
            if attempt < max_retries:
                await asyncio.sleep(2 * attempt)
    return None


# ---------------------------------------------------------------------------
# Core evolution logic
# ---------------------------------------------------------------------------

def _next_version(current: str) -> str:
    num = int(current.lstrip("p")) + 1
    return f"p{num:02d}"


def _initialise_tracking(tree: dict) -> dict:
    """Add ``last_updated_at: null`` and ``update_count: 0`` to every persona
    field if missing."""
    out = copy.deepcopy(tree)
    persona = out.get("persona", {})
    for field, val in persona.items():
        if field == "version":
            continue
        if isinstance(val, dict) and "value" in val:
            val.setdefault("last_updated_at", None)
            val.setdefault("update_count", 0)
    return out


async def evolve_all_characters(
    client: AsyncOpenAI,
    model: str,
    trees: dict,
    archives: dict[str, list[dict]],
    episode_order: list[tuple[int, int]],
    workers: int,
    test_episode: tuple[int, int] | None = None,
):
    semaphore = asyncio.Semaphore(workers)

    current_trees: dict[str, dict] = {
        char: _initialise_tracking(tree) for char, tree in trees.items()
    }

    snapshots: dict[str, dict[str, dict]] = {char: {} for char in MAIN_CHARACTERS}
    evolution_log: list[dict] = []

    # Tracks when each (char, role|name) tuple FIRST appeared as a non-ex
    # romantic entry, used by the decay Rule B2 grace period.
    first_seen_ord: dict[tuple[str, str], int] = {}

    episodes_to_process = episode_order
    if test_episode:
        idx = episode_order.index(test_episode)
        episodes_to_process = episode_order[:idx + 1]

    total_updates = 0

    for season, episode in episodes_to_process:
        ep_tag = f"S{season:02d}E{episode:02d}"

        # -- expire old medium entries --
        for char in MAIN_CHARACTERS:
            _expire_old_medium(archives.get(char, []), ep_tag)

        tasks = []
        chars_eval: list[tuple[str, list[dict]]] = []

        for char in MAIN_CHARACTERS:
            archive = archives.get(char, [])
            active = [
                e for e in archive
                if e.get("status") == "active"
                and (e["season"], e["episode"]) <= (season, episode)
            ]
            # Track A — active recent evidence (low-resistance fields)
            track_a = _truncate_archive(active) if active else []
            # Track B — lifetime high-sig pattern archive (moderate / core)
            track_b = _build_lifetime_archive(archive, season, episode)

            if not track_a and not track_b:
                chars_eval.append((char, []))
                continue
            chars_eval.append((char, active))
            tasks.append(
                evaluate_persona_change(
                    client, semaphore, model, char,
                    current_trees[char].get("identity", {}),
                    current_trees[char].get("persona", {}),
                    track_a,
                    track_b,
                )
            )

        results_iter = iter(await asyncio.gather(*tasks)) if tasks else iter([])

        ep_updates_msg: list[str] = []
        ep_rejections: list[tuple[str, list[str]]] = []
        for char, active in chars_eval:
            if not active:
                snapshots[char][ep_tag] = copy.deepcopy(current_trees[char])
                continue

            decision = next(results_iter)
            if not decision:
                snapshots[char][ep_tag] = copy.deepcopy(current_trees[char])
                continue

            other_personas = {
                c: t.get("persona", {})
                for c, t in current_trees.items() if c != char
            }
            lifetime_b = _build_lifetime_archive(
                archives.get(char, []), season, episode
            )
            decision, rejections = validate_decision(
                decision, active, current_trees[char].get("persona", {}), ep_tag,
                char_name=char,
                all_archives=archives,
                other_personas=other_personas,
                lifetime_high=lifetime_b,
            )
            if rejections:
                ep_rejections.append((char, rejections))

            if decision.get("should_update") and decision.get("changes"):
                old_version = current_trees[char]["persona"].get("version", "p00")
                new_version = _next_version(old_version)
                changes = decision["changes"]

                for field, change in changes.items():
                    persona_field = current_trees[char]["persona"][field]
                    persona_field["value"] = change["new_value"]
                    persona_field["last_updated_at"] = ep_tag
                    persona_field["update_count"] = (
                        persona_field.get("update_count", 0) + 1
                    )

                current_trees[char]["persona"]["version"] = new_version
                cid = current_trees[char].get(
                    "character_id",
                    char.lower().replace(" ", "_"))
                current_trees[char]["tree_id"] = f"{cid}::{new_version}"

                addendum = decision.get("backstory_addendum")
                if addendum:
                    old_bs = current_trees[char]["identity"].get("backstory", "")
                    current_trees[char]["identity"]["backstory"] = (
                        f"{old_bs} {addendum}".strip())

                # Mark consumed sessions (union of all field-level evidence)
                consumed_ids: set[str] = set()
                for ch in changes.values():
                    consumed_ids.update(ch["consumed_session_ids"])
                for e in archives[char]:
                    if e["scene_id"] in consumed_ids and e["status"] == "active":
                        e["status"] = "consumed"
                        e["consumed_by"] = new_version
                        e["consumed_at_episode"] = ep_tag

                evolution_log.append({
                    "character": char,
                    "episode": ep_tag,
                    "from_version": old_version,
                    "to_version": new_version,
                    "changes": {
                        f: {k: v for k, v in ch.items()}
                        for f, ch in changes.items()
                    },
                    "backstory_addendum": addendum,
                    "reasoning": decision.get("reasoning", ""),
                    "sessions_consumed": len(consumed_ids),
                })
                total_updates += 1
                fields_str = ", ".join(changes.keys())
                ep_updates_msg.append(
                    f"{char}: {old_version}→{new_version} [{fields_str}]")

            # ---- auto ex- downgrade guard ------------------------------
            # After the LLM-driven update (or no-op), proactively scan
            # current relationships for partners with explicit breakup
            # evidence in this character's archive and rewrite the entry
            # to ``ex-<role>``.  This catches LLM conservativeness for
            # short relationship arcs (e.g. Joey-Kate, Joey-Kathy).
            rel_field = current_trees[char]["persona"].get("relationships")
            if isinstance(rel_field, dict):
                rel_val_now = rel_field.get("value", "")
                new_rel_val, downgrades = _auto_ex_downgrade(
                    rel_val_now, archives.get(char, []), char, ep_tag,
                )
                if downgrades:
                    rel_field["value"] = new_rel_val
                    rel_field["last_updated_at"] = ep_tag
                    old_v = current_trees[char]["persona"].get("version", "p00")
                    new_v = _next_version(old_v)
                    current_trees[char]["persona"]["version"] = new_v
                    cid = current_trees[char].get(
                        "character_id",
                        char.lower().replace(" ", "_"),
                    )
                    current_trees[char]["tree_id"] = f"{cid}::{new_v}"
                    evolution_log.append({
                        "character": char,
                        "episode": ep_tag,
                        "from_version": old_v,
                        "to_version": new_v,
                        "changes": {
                            "relationships": {
                                "old_value": rel_val_now,
                                "new_value": new_rel_val,
                                "merge_strategy": "auto_ex_downgrade",
                                "consumed_session_ids": [],
                                "downgrades": [
                                    {"role": r, "partner": p}
                                    for r, p in downgrades
                                ],
                            }
                        },
                        "backstory_addendum": None,
                        "reasoning": (
                            "Automatic ex- downgrade: explicit breakup "
                            "evidence found in archive."
                        ),
                        "sessions_consumed": 0,
                    })
                    total_updates += 1
                    pretty = ", ".join(f"{r} {p}" for r, p in downgrades)
                    ep_updates_msg.append(
                        f"{char}: {old_v}→{new_v} [auto-ex: {pretty}]"
                    )

            snapshots[char][ep_tag] = copy.deepcopy(current_trees[char])

        # ---- per-episode decay pass --------------------------------------
        # Run AFTER all characters have been LLM-evaluated and applied for
        # this episode, so inter-main reciprocity (Rule B) sees each
        # character's just-updated state.  Updates current_trees in place
        # and overwrites the snapshot for any character whose relationships
        # were decayed.
        for char in MAIN_CHARACTERS:
            rel_field = current_trees[char]["persona"].get("relationships")
            if not isinstance(rel_field, dict):
                continue
            rel_val = rel_field.get("value", "")
            if not rel_val:
                continue

            # Update first_seen_ord with any new non-ex romantic entries
            # introduced in this episode (so Rule B2's grace period starts
            # counting from now, not from a future episode).
            cur_ord = _episode_to_int(ep_tag)
            for role, name, _raw in _parse_relationship_entries(rel_val):
                role_l = role.lower().strip()
                if role_l.startswith("ex-") or role_l.startswith("late-"):
                    continue
                key = (char, f"{role_l}|{name.lower()}")
                if key not in first_seen_ord:
                    first_seen_ord[key] = cur_ord

            new_val, decays = decay_relationships_value(
                rel_val, char, ep_tag,
                archives.get(char, []),
                current_trees, snapshots,
                first_seen_ord, MAIN_CHARACTERS,
            )
            if not decays:
                continue

            rel_field["value"] = new_val
            rel_field["last_updated_at"] = ep_tag
            old_v = current_trees[char]["persona"].get("version", "p00")
            new_v = _next_version(old_v)
            current_trees[char]["persona"]["version"] = new_v
            cid = current_trees[char].get(
                "character_id", char.lower().replace(" ", "_"),
            )
            current_trees[char]["tree_id"] = f"{cid}::{new_v}"

            evolution_log.append({
                "character": char,
                "episode": ep_tag,
                "from_version": old_v,
                "to_version": new_v,
                "changes": {
                    "relationships": {
                        "old_value": rel_val,
                        "new_value": new_val,
                        "merge_strategy": "auto_decay",
                        "consumed_session_ids": [],
                        "decay_records": decays,
                    }
                },
                "backstory_addendum": None,
                "reasoning": "Per-episode decay (Rule A + Rule B).",
                "sessions_consumed": 0,
            })
            total_updates += 1

            decay_pretty = "; ".join(
                f"{d['from_entry']} → {d['to_entry']} [{d['reason']}]"
                for d in decays[:3]
            )
            if len(decays) > 3:
                decay_pretty += f" (+{len(decays) - 3} more)"
            ep_updates_msg.append(
                f"{char}: {old_v}→{new_v} [decay: {decay_pretty}]"
            )

            # Re-write the snapshot with the decayed value.
            snapshots[char][ep_tag] = copy.deepcopy(current_trees[char])

        if ep_updates_msg:
            print(f"  {ep_tag}:")
            for m in ep_updates_msg:
                print(f"    UPDATE  {m}")
            for char, reasons in ep_rejections:
                for r in reasons:
                    print(f"    REJECT  {char}: {r}")
        else:
            sys.stdout.write(f"\r  {ep_tag}: no changes")
            sys.stdout.flush()
            if ep_rejections:
                print()
                for char, reasons in ep_rejections:
                    for r in reasons:
                        print(f"    REJECT  {char}: {r}")

    print(f"\n\nTotal persona updates: {total_updates}")
    return snapshots, evolution_log


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

async def async_main(args):
    model = os.getenv("LLM_MODEL", "gpt-4.1")
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    if not api_key:
        raise RuntimeError("LLM_API_KEY not set in .env")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    intermediate_dir = PROCESSED_DIR / args.dataset / "intermediate"
    evolution_dir = intermediate_dir / "evolution"

    with open(intermediate_dir / "attribute_trees.json", encoding="utf-8") as f:
        trees = json.load(f)
    print(f"Loaded {len(trees)} initial attribute trees")

    global MAIN_CHARACTERS, SYSTEM_PROMPT
    MAIN_CHARACTERS = list(trees.keys())
    SYSTEM_PROMPT = _build_system_prompt(args.dataset)
    print(f"Main characters: {MAIN_CHARACTERS}")
    print(f"Show: {SHOW_NAMES.get(args.dataset, args.dataset)}")

    archives: dict[str, list[dict]] = {}
    for char in MAIN_CHARACTERS:
        fpath = evolution_dir / f"{char.replace(' ', '_')}_session_archive.json"
        if fpath.exists():
            with open(fpath, encoding="utf-8") as f:
                archives[char] = json.load(f)
            n_active = sum(1 for e in archives[char] if e.get("status") == "active")
            print(f"  {char}: {len(archives[char])} entries ({n_active} active)")
        else:
            archives[char] = []
            print(f"  {char}: no archive found")

    with open(intermediate_dir / "all_dialogues.json", encoding="utf-8") as f:
        all_samples = json.load(f)

    seen = set()
    episode_order: list[tuple[int, int]] = []
    for s in all_samples:
        season = s.get("_season") or s.get("_book")
        episode = s.get("_episode") or _parse_chapter_from_position(s.get("_position"))
        if season is None or episode is None:
            continue
        key = (season, episode)
        if key not in seen:
            seen.add(key)
            episode_order.append(key)

    test_ep = None
    if args.test_episode:
        m = re.match(r"S(\d+)E(\d+)", args.test_episode.upper())
        mb = re.match(r"B(\d+)C(\d+)", args.test_episode.upper())
        if m:
            test_ep = (int(m.group(1)), int(m.group(2)))
            print(f"\nTest mode: evaluating up to S{test_ep[0]:02d}E{test_ep[1]:02d}")
        elif mb:
            test_ep = (int(mb.group(1)), int(mb.group(2)))
            print(f"\nTest mode: evaluating up to Book{test_ep[0]} Chapter{test_ep[1]}")
        else:
            raise ValueError(f"Invalid format: {args.test_episode} (expected S01E05 or B1C2)")

    print(f"\nEpisode count: {len(episode_order)}")
    print(f"Model: {model}, API base: {base_url}")
    print("-" * 60)

    snapshots, evo_log = await evolve_all_characters(
        client, model, trees, archives, episode_order,
        workers=args.workers,
        test_episode=test_ep,
    )

    snap_path = evolution_dir / "persona_snapshots.json"
    snap_out = {}
    for char, ep_snaps in snapshots.items():
        snap_out[char] = {}
        for ep_tag, tree in ep_snaps.items():
            snap_out[char][ep_tag] = {
                "version": tree["persona"]["version"],
                "tree": tree,
            }
    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(snap_out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved persona snapshots -> {snap_path}")

    for char in MAIN_CHARACTERS:
        fpath = evolution_dir / f"{char.replace(' ', '_')}_session_archive.json"
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(archives[char], f, ensure_ascii=False, indent=2)

    if evo_log:
        log_path = evolution_dir / "evolution_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(evo_log, f, ensure_ascii=False, indent=2)
        print(f"Saved evolution log -> {log_path}")

        print("\n--- Evolution Summary ---")
        for entry in evo_log:
            print(f"  {entry['episode']} | {entry['character']}: "
                  f"{entry['from_version']}→{entry['to_version']}")
            for field, ch in entry["changes"].items():
                merge = ch.get("merge_type", "?")
                preview = ch.get("new_value", "")[:80]
                print(f"    {field} ({merge}): {preview}"
                      f"{'...' if len(ch.get('new_value', '')) > 80 else ''}")
    else:
        print("\nNo persona updates triggered.")

    print("\nDone.")


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate and apply persona evolution from session archives")
    ap.add_argument("--dataset", type=str, required=True,
                    choices=["Friends", "StarTrek_TNG", "TheOffice", "HPD"],
                    help="Dataset to process")
    ap.add_argument("--test_episode", type=str, default=None,
                    help="Evaluate up to this episode only (e.g. S01E05)")
    ap.add_argument("--workers", type=int, default=6,
                    help="Max concurrent API requests (default: 6)")
    args = ap.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
