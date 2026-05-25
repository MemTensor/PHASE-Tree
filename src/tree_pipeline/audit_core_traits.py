"""Per-descriptor audit for core persona fields (personality / speaking_style).

The standard episode-by-episode evolve loop in ``evolve_persona.py`` is
structurally biased against updating ``core`` fields:

* the LLM is asked, at every episode, "should we change this field given the
  recent + lifetime evidence?";
* anchored on the (already plausible) current value, the LLM almost always
  answers "no", because no single episode contains enough evidence to overturn
  a well-established trait;
* over 218 episodes × 6 characters this yields ≈ 0 core proposals, even when
  the show clearly contains slow drift (Rachel becoming less naive, Chandler
  becoming less anxious, etc.).

This script attacks the problem from a different angle:

1. **Wider window** – we audit at coarse-grained checkpoints (default: every
   20 episodes plus the series finale) rather than every episode.
2. **Descriptor-level question** – we parse the field's value into individual
   adjectives / phrases and ask the LLM to judge **each one** independently:
   ``still_accurate`` / ``partially_outdated`` / ``clearly_outdated``.
3. **Burden of proof on contradiction** – a descriptor only changes when the
   LLM cites ≥ 2 high-significance session IDs from the lifetime archive that
   directly contradict it. Absence of mention is NOT evidence of disappearance.
4. **Hard caps** – at most 2 descriptors may change per audit, replacements
   must be compact phrases, and we never wholesale-replace a field.

Inputs:
    ``phase_tree_data/processed/<dataset>/intermediate/evolution/persona_snapshots.json``
    ``phase_tree_data/processed/<dataset>/intermediate/evolution/<Char>_session_archive.json``
    ``phase_tree_data/processed/<dataset>/intermediate/all_dialogues.json``  (episode order)

Outputs (in-place):
    ``persona_snapshots.json``  -> updated personality/speaking_style values
        at audit episodes (and propagated forward to later snapshots)
    ``core_audit_log.json``     -> detailed audit decisions

Usage::

    python -m tree_pipeline.audit_core_traits --dataset Friends --dry-run \\
        --character "Rachel Green" --audit-episodes S05E24,S10E18

    # Full run on all 6 mains, every 20 episodes:
    python -m tree_pipeline.audit_core_traits --dataset Friends --workers 4
"""

import argparse
import asyncio
import copy
import json
import os
import re
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED_DIR = PROJECT_ROOT / "phase_tree_data" / "processed"

MAIN_CHARACTERS = [
    "Monica Geller", "Ross Geller", "Rachel Green",
    "Chandler Bing", "Joey Tribbiani", "Phoebe Buffay",
]

CORE_FIELDS = ("personality", "speaking_style")

# At most this many descriptors may change in a single audit pass for a given
# (character, field).  Prevents the LLM from rewriting an entire personality.
MAX_CHANGES_PER_AUDIT = 1

# A descriptor must accumulate at least this many distinct contradicting
# high-sig session IDs before it can be removed/refined.
MIN_CONTRADICTION_EVIDENCE = 2

# Default audit cadence: every N episodes (+ the final episode).
# At ≈24 eps per season, this lands roughly on per-season finales.
DEFAULT_AUDIT_INTERVAL = 24

# Once a descriptor (identified by its current text) has been changed in an
# audit pass, it cannot be changed again for this many subsequent audits.
# This prevents ping-pong refinements ("vulnerable" -> "guarded" -> "open" ...)
# that drift far from the original meaning across many checkpoints.
DESCRIPTOR_COOLDOWN_AUDITS = 2

# Per-character "sitcom DNA": descriptors that must NEVER be removed because
# they are stable comedic signatures.  Comparison is case-insensitive on the
# core lemma — i.e. matching is a substring check after lowercasing both
# sides.  Keep entries SHORT lemmas, not full descriptors.
SITCOM_DNA = {
    "Monica Geller":   ["competitive", "organized"],
    "Ross Geller":     ["sensitive", "sarcastic", "introspective"],
    "Rachel Green":    ["friendly", "expressive", "sociable"],
    "Chandler Bing":   ["sarcastic", "witty", "self-deprecating"],
    "Joey Tribbiani":  ["easygoing", "playful", "flirtatious", "generous",
                        "loyal", "naive"],
    "Phoebe Buffay":   ["quirky", "eccentric", "sing", "rhyme",
                        "metaphor", "whimsical"],
}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a character psychologist auditing a single deep-trait field of a \
long-running TV character. Unlike a per-episode update, your job is to look \
at the CHARACTER'S WHOLE ARC SO FAR and decide, **descriptor by descriptor**, \
whether each piece of the current trait description is still accurate.

You will receive:
- The character's identity backstory (for grounding).
- The trait field name (one of: personality, speaking_style).
- The CURRENT value of that field, broken into individual descriptors.
- A LIFETIME PATTERN ARCHIVE: every high-significance event from the start \
of the show up to the audit episode, sorted chronologically.

For EACH descriptor, return one of three statuses:

1. **still_accurate** — evidence is consistent with this descriptor, OR there \
is simply no clear contradiction. This must be the DEFAULT — when in doubt, \
return still_accurate.

2. **partially_outdated** — the descriptor was accurate early on but the \
LATER half of the archive shows the trait has weakened, transformed, or no \
longer dominates. You must:
   - cite at least {min_evidence} contradicting session_id values from the \
archive (events showing the trait fading);
   - propose a *refined* replacement (a compact phrase, not a sentence) that \
better captures the character's current expression of this trait.

3. **clearly_outdated** — multiple high-significance events directly \
contradict the descriptor. You must:
   - cite at least {min_evidence} contradicting session_id values;
   - propose a replacement descriptor that the evidence supports.

## Hard rules

- **ABSENCE OF MENTION IS NOT EVIDENCE OF DISAPPEARANCE.** If a descriptor \
is simply not referenced in the recent archive, that is NOT grounds for \
removal. Many traits remain true even when not foregrounded.
- **STABLE SITCOM SIGNATURES ARE PROTECTED.** The following descriptors / \
lemmas describe character DNA and MUST be returned as still_accurate:
{dna_block}
  Even if recent events appear to soften or counter them, treat these as \
permanent comedic markers. Mark them clearly_outdated only if you have \
overwhelming, multi-season evidence of permanent inversion.
- **PREFER NUANCE OVER REVERSAL.** A descriptor may "deepen" but should not \
flip to its semantic opposite. Good: "naive" -> "wiser" (deepening). Bad: \
"easygoing" -> "anxious" (reversal). Bad: "frequently sings" -> "rarely \
sings" (reversal).
- **NO PING-PONG.** Do not propose a refinement if the descriptor's current \
text was itself the result of a previous audit refinement (you can usually \
tell because the text is more abstract or compound, e.g. "growing \
self-assurance"). When in doubt, leave it alone.
- Do not collapse two descriptors into one or split one into two. Operate \
descriptor-by-descriptor at the same granularity as the input.
- Keep replacements compact: a single adjective or short noun-phrase. No \
full sentences, no explanations inside the replacement.
- If you mark a descriptor partially_outdated/clearly_outdated, the \
replacement MUST be semantically distinct from any other descriptor in the \
list (no duplicates).
- You may mark AT MOST {max_changes} descriptors as anything other than \
still_accurate in this audit. Choose the highest-confidence one(s).
- session_id values you cite MUST appear verbatim in the archive shown to \
you. Made-up IDs will be rejected.

## Output format (strict JSON)

{{
  "audit": [
    {{"descriptor": "<verbatim from input>",
      "status": "still_accurate|partially_outdated|clearly_outdated",
      "evidence_session_ids": ["<id1>", "<id2>"],
      "replacement": "<new descriptor or empty string if still_accurate>",
      "reasoning": "<one short sentence>"}}
  ]
}}

Return ONLY this JSON object — no extra prose.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ep_tag(season: int, episode: int) -> str:
    return f"S{season:02d}E{episode:02d}"


def _ep_ord(season: int, episode: int) -> int:
    return season * 100 + episode


def _ep_ord_tag(tag: str) -> int:
    m = re.match(r"S(\d+)E(\d+)", tag.upper())
    if not m:
        raise ValueError(f"Bad episode tag: {tag}")
    return _ep_ord(int(m.group(1)), int(m.group(2)))


def _split_descriptors(value: str) -> list[str]:
    """Split a personality/speaking_style value into descriptor phrases.

    Most p00 values look like:
        "Sensitive, thoughtful, sarcastic, ..., values friendship, honesty, ...,
         often melancholic, sentimental, and sometimes anxious."

    We split on commas and trailing periods, strip leading/trailing
    whitespace, drop empties.  We do NOT try to canonicalise; we want to
    preserve the LLM's exact phrasing so we can find it back during merge.
    """
    text = value.strip().rstrip(".")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    out = []
    for p in parts:
        p2 = re.sub(r"\s+(and)\s+", " ", p, flags=re.IGNORECASE)
        p2 = p2.strip().rstrip(".")
        if p2:
            out.append(p2)
    return out


def _join_descriptors(descs: list[str], original_value: str) -> str:
    """Join a list of descriptors back into a comma-separated string,
    preserving the trailing period if the original had one."""
    text = ", ".join(d.strip().rstrip(".") for d in descs if d.strip())
    if original_value.strip().endswith("."):
        text = text + "."
    return text


def _format_archive(entries: list[dict]) -> str:
    """Format archive entries with their session_id (== scene_id) so the LLM
    can cite them verbatim."""
    lines = []
    for e in entries:
        tag = _ep_tag(e["season"], e["episode"])
        sid = e.get("scene_id", "?")
        sig = e.get("significance", "?")
        sm = e.get("summary", "")
        lines.append(f"- [{tag}, session_id={sid}, sig={sig}] {sm}")
    return "\n".join(lines)


def _load_archive(evolution_dir: Path, character: str) -> list[dict]:
    fname = f"{character.replace(' ', '_')}_session_archive.json"
    path = evolution_dir / fname
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _lifetime_high_up_to(
    archive: list[dict],
    audit_season: int,
    audit_episode: int,
    since_season: int = 0,
    since_episode: int = 0,
) -> list[dict]:
    out = [
        e for e in archive
        if e.get("significance") == "high"
        and (e["season"], e["episode"]) <= (audit_season, audit_episode)
        and (e["season"], e["episode"]) > (since_season, since_episode)
    ]
    return sorted(out, key=lambda e: (e["season"], e["episode"]))


def _resolve_episode_order(all_samples: list[dict]) -> list[tuple[int, int]]:
    """Build a temporally-ordered list of (season, episode) keys.

    Supports both the standard ``_season``/``_episode`` schema (Friends,
    TheOffice, StarTrek_TNG) and HPD's ``_book``/``_position`` (where
    ``_position`` looks like ``"Book1-chapter2"``).
    """
    seen = set()
    order = []
    for s in all_samples:
        season = s.get("_season")
        episode = s.get("_episode")
        if season is None:
            season = s.get("_book")
        if episode is None:
            chap = _parse_chapter_from_position(s.get("_position"))
            episode = chap
        if season is None or episode is None:
            continue
        key = (int(season), int(episode))
        if key not in seen:
            seen.add(key)
            order.append(key)
    return order


def _parse_chapter_from_position(position: str | None) -> int | None:
    """Extract chapter number from HPD ``_position`` like ``"Book1-chapter2"`` → 2."""
    if not position:
        return None
    m = re.search(r"chapter(\d+)", position, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _default_audit_episodes(
    episode_order: list[tuple[int, int]],
    interval: int = DEFAULT_AUDIT_INTERVAL,
) -> list[tuple[int, int]]:
    """Return audit checkpoints: every ``interval``-th episode plus the last."""
    if not episode_order:
        return []
    pts = []
    for i, ep in enumerate(episode_order, start=1):
        if i % interval == 0:
            pts.append(ep)
    last = episode_order[-1]
    if not pts or pts[-1] != last:
        pts.append(last)
    return pts


# ---------------------------------------------------------------------------
# LLM call & response parsing
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


async def audit_field(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    model: str,
    character: str,
    identity: dict,
    field: str,
    value: str,
    archive: list[dict],
    audit_tag: str,
    max_retries: int = 3,
) -> dict | None:
    descriptors = _split_descriptors(value)
    if not descriptors:
        return None

    archive_text = _format_archive(archive)

    user_prompt = (
        f"## Character: {character}\n"
        f"## Audit episode: {audit_tag}\n"
        f"## Trait field: {field}\n\n"
        f"## Identity backstory\n"
        f"{identity.get('backstory', 'N/A')}\n\n"
        f"## Current {field} descriptors (operate on each ONE)\n"
        + "\n".join(f"  {i+1}. {d}" for i, d in enumerate(descriptors))
        + "\n\n"
        f"## Lifetime pattern archive ({len(archive)} high-sig events)\n"
        f"{archive_text}\n"
    )

    dna_lemmas = SITCOM_DNA.get(character, [])
    if dna_lemmas:
        dna_block = "  - " + ", ".join(f"'{l}'" for l in dna_lemmas)
    else:
        dna_block = "  - (none specific to this character)"
    sys_prompt = SYSTEM_PROMPT.format(
        min_evidence=MIN_CONTRADICTION_EVIDENCE,
        max_changes=MAX_CHANGES_PER_AUDIT,
        dna_block=dna_block,
    )

    async with sem:
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=2000,
                    timeout=120.0,
                )
                return _parse_json(resp.choices[0].message.content)
            except json.JSONDecodeError as e:
                sys.stderr.write(
                    f"\n  [{character}/{field}@{audit_tag}] JSON err "
                    f"(attempt {attempt}): {e}\n")
            except Exception as e:
                sys.stderr.write(
                    f"\n  [{character}/{field}@{audit_tag}] Error "
                    f"(attempt {attempt}): {e}\n")
            if attempt < max_retries:
                await asyncio.sleep(2 * attempt)
    return None


# ---------------------------------------------------------------------------
# Validation & merge
# ---------------------------------------------------------------------------

def _matches_dna(descriptor: str, dna_lemmas: list[str]) -> bool:
    """True if the descriptor contains any sitcom-DNA lemma (lowercased
    substring match).  We strip leading 'and ' / leading commas the
    splitter may have left, to make the substring check more robust."""
    text = descriptor.lower()
    text = re.sub(r"^\s*and\s+", "", text).strip()
    for lemma in dna_lemmas:
        if lemma.lower() in text:
            return True
    return False


def validate_and_apply(
    descriptors: list[str],
    audit_response: dict,
    archive: list[dict],
    character: str,
    locked_descriptors: set[str],
) -> tuple[list[str], list[dict]]:
    """Return (new_descriptors, change_log).

    ``locked_descriptors`` is a set of descriptor texts (lowercased, stripped)
    that are in cooldown from a previous audit and must not be changed again.
    """
    valid_ids = {str(e.get("scene_id")) for e in archive}
    dna_lemmas = SITCOM_DNA.get(character, [])
    audit = audit_response.get("audit", [])

    by_desc = {a.get("descriptor", "").strip(): a for a in audit}

    accepted = []
    rejected = []
    seen_changes = 0
    new_descs = []

    for d in descriptors:
        a = by_desc.get(d.strip())
        if not a:
            new_descs.append(d)
            continue

        status = a.get("status", "still_accurate")
        if status == "still_accurate":
            new_descs.append(d)
            continue

        # Cooldown: descriptor changed in a recent audit cannot be touched
        if d.strip().lower() in locked_descriptors:
            rejected.append({**a, "rejected_reason": "descriptor in cooldown"})
            new_descs.append(d)
            continue

        # Sitcom-DNA guard: descriptor matches a protected lemma
        if _matches_dna(d, dna_lemmas):
            rejected.append({
                **a,
                "rejected_reason": f"protected sitcom DNA (matches {dna_lemmas})",
            })
            new_descs.append(d)
            continue

        if seen_changes >= MAX_CHANGES_PER_AUDIT:
            rejected.append({
                **a,
                "rejected_reason": f"exceeded {MAX_CHANGES_PER_AUDIT} change cap",
            })
            new_descs.append(d)
            continue

        ev_ids = [str(x) for x in (a.get("evidence_session_ids") or [])]
        ev_ids_in_archive = [x for x in ev_ids if x in valid_ids]
        if len(ev_ids_in_archive) < MIN_CONTRADICTION_EVIDENCE:
            rejected.append({
                **a,
                "rejected_reason": (
                    f"insufficient archive-grounded evidence "
                    f"({len(ev_ids_in_archive)} of {MIN_CONTRADICTION_EVIDENCE})"
                ),
            })
            new_descs.append(d)
            continue

        replacement = (a.get("replacement") or "").strip().rstrip(".")
        if not replacement:
            rejected.append({**a, "rejected_reason": "empty replacement"})
            new_descs.append(d)
            continue

        # No semantic-opposite replacement against the same DNA list
        if _matches_dna(replacement, dna_lemmas) and not _matches_dna(d, dna_lemmas):
            rejected.append({**a, "rejected_reason": "replacement collides with sitcom DNA"})
            new_descs.append(d)
            continue

        # Replacement must be semantically distinct from existing descriptors
        existing_lc = {x.strip().lower() for x in (new_descs + descriptors) if x != d}
        if replacement.lower() in existing_lc:
            rejected.append({**a, "rejected_reason": "duplicate of existing descriptor"})
            new_descs.append(d)
            continue

        if len(replacement) > 80 or len(replacement.split()) > 12:
            rejected.append({**a, "rejected_reason": "replacement too long"})
            new_descs.append(d)
            continue

        accepted.append({**a, "evidence_session_ids": ev_ids_in_archive})
        seen_changes += 1
        new_descs.append(replacement)

    return new_descs, accepted + rejected


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_audit(args):
    model = os.getenv("LLM_MODEL", "gpt-4.1")
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    if not api_key:
        raise RuntimeError("LLM_API_KEY not set in .env")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    intermediate_dir = PROCESSED_DIR / args.dataset / "intermediate"
    evolution_dir = intermediate_dir / "evolution"
    snap_path = evolution_dir / "persona_snapshots.json"

    with open(snap_path, encoding="utf-8") as f:
        snapshots = json.load(f)

    with open(intermediate_dir / "all_dialogues.json", encoding="utf-8") as f:
        all_samples = json.load(f)
    episode_order = _resolve_episode_order(all_samples)

    if args.audit_episodes:
        audit_eps = []
        for tag in args.audit_episodes.split(","):
            tag = tag.strip().upper()
            m = re.match(r"S(\d+)E(\d+)", tag)
            if not m:
                raise ValueError(f"Bad audit episode tag: {tag}")
            audit_eps.append((int(m.group(1)), int(m.group(2))))
    else:
        audit_eps = _default_audit_episodes(episode_order, args.interval)

    audit_tags = [_ep_tag(s, e) for s, e in audit_eps]
    print(f"Audit checkpoints ({len(audit_tags)}): {', '.join(audit_tags)}")

    if args.character:
        chars = [c.strip() for c in args.character.split(",") if c.strip()]
    else:
        # Load main characters dynamically from attribute_trees.json so this
        # script works for any dataset (Friends, TheOffice, StarTrek_TNG, HPD).
        trees_path = intermediate_dir / "attribute_trees.json"
        if trees_path.exists():
            chars = list(json.loads(trees_path.read_text(encoding="utf-8")).keys())
        else:
            chars = list(MAIN_CHARACTERS)
    print(f"Characters: {chars}")

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    for f in fields:
        if f not in CORE_FIELDS:
            raise ValueError(f"Field {f} not in {CORE_FIELDS}")
    print(f"Fields: {fields}")

    archives = {c: _load_archive(evolution_dir, c) for c in chars}
    for c in chars:
        if c not in snapshots:
            raise RuntimeError(f"No snapshots for {c}")

    sem = asyncio.Semaphore(args.workers)
    audit_log: list[dict] = []

    # Progress through audit checkpoints in order.  Per character/field we
    # track the previous audit's (season, episode) so the lifetime archive
    # passed to the next audit excludes events already audited.  This makes
    # later audits focus on *new* drift since the previous checkpoint while
    # still letting the LLM see overall trajectory through identity backstory.
    prev_cut: dict[tuple[str, str], tuple[int, int]] = {
        (c, f): (0, 0) for c in chars for f in fields
    }

    # Descriptor cooldown: per (char, field), keep a list of
    #   (audit_index_when_changed, descriptor_text_after_change)
    # Any descriptor in cooldown for the next DESCRIPTOR_COOLDOWN_AUDITS
    # audits will be skipped to prevent ping-pong refinements.
    cooldown: dict[tuple[str, str], list[tuple[int, str]]] = {
        (c, f): [] for c in chars for f in fields
    }

    for audit_idx, (season, episode) in enumerate(audit_eps):
        ep_tag = _ep_tag(season, episode)
        if ep_tag not in snapshots[chars[0]]:
            print(f"  [{ep_tag}] not in snapshots, skipping")
            continue
        print(f"\n=== Audit @ {ep_tag} (idx={audit_idx}) ===")

        tasks = []
        task_keys = []
        for char in chars:
            snap = snapshots[char].get(ep_tag)
            if not snap:
                print(f"  {char}: no snapshot at {ep_tag}, skip")
                continue
            tree = snap["tree"]
            identity = tree.get("identity", {})
            persona = tree.get("persona", {})

            for field in fields:
                fdat = persona.get(field)
                if not isinstance(fdat, dict) or "value" not in fdat:
                    continue
                value = fdat["value"]
                descs = _split_descriptors(value)
                if len(descs) < 2:
                    continue

                since_s, since_e = prev_cut[(char, field)]
                lifetime = _lifetime_high_up_to(
                    archives.get(char, []), season, episode,
                    since_season=since_s, since_episode=since_e,
                )
                if len(lifetime) < MIN_CONTRADICTION_EVIDENCE * 2:
                    print(f"  {char}/{field}: only {len(lifetime)} new "
                          f"high-sig since last audit, skip")
                    prev_cut[(char, field)] = (season, episode)
                    continue

                tasks.append(audit_field(
                    client, sem, model, char, identity, field, value,
                    lifetime, ep_tag,
                ))
                # Build the cooldown set for this (char, field): any
                # descriptor whose change index is within DESCRIPTOR_COOLDOWN_AUDITS
                # of the current audit_idx is locked.
                locked = {
                    text.lower() for (idx, text) in cooldown[(char, field)]
                    if audit_idx - idx <= DESCRIPTOR_COOLDOWN_AUDITS
                }
                task_keys.append((char, field, value, lifetime, locked))

        if not tasks:
            continue

        results = await asyncio.gather(*tasks)

        for (char, field, value, lifetime, locked), resp in zip(task_keys, results):
            descs = _split_descriptors(value)
            if not resp:
                print(f"  {char}/{field}: LLM call failed")
                audit_log.append({
                    "episode": ep_tag, "character": char, "field": field,
                    "status": "llm_failed",
                })
                prev_cut[(char, field)] = (season, episode)
                continue

            new_descs, change_log = validate_and_apply(
                descs, resp, lifetime, char, locked,
            )
            applied = [c for c in change_log if "rejected_reason" not in c]

            if applied:
                new_value = _join_descriptors(new_descs, value)
                if not args.dry_run:
                    # Write updated value into THIS snapshot and propagate
                    # forward so all future snapshots inherit the new
                    # descriptor set.
                    forward_eps = [
                        t for t in snapshots[char]
                        if _ep_ord_tag(t) >= _ep_ord(season, episode)
                    ]
                    for t in forward_eps:
                        ftree = snapshots[char][t]["tree"]
                        fpersona = ftree.get("persona", {})
                        fdat = fpersona.get(field)
                        if not isinstance(fdat, dict):
                            continue
                        # Only update snapshots whose value still equals the
                        # pre-audit value at this audit point — if a later
                        # snapshot already diverged due to a regular evolve
                        # update, we leave it alone (downstream evolve wins).
                        if fdat.get("value", "") == value:
                            fdat["value"] = new_value
                            fdat["last_updated_at"] = ep_tag
                            fdat["last_change_source"] = "core_audit"
                        else:
                            break
                print(f"  {char}/{field}: {len(applied)} change(s) applied")
                for c in applied:
                    print(f"    [{c['status']}] '{c['descriptor']}' "
                          f"-> '{c['replacement']}'  ev={c['evidence_session_ids']}")
                    cooldown[(char, field)].append(
                        (audit_idx, c["replacement"].strip().lower())
                    )
            else:
                rejs = [c for c in change_log if "rejected_reason" in c]
                if rejs:
                    print(f"  {char}/{field}: {len(rejs)} change(s) rejected, "
                          f"no update")
                    for c in rejs:
                        print(f"    REJECT '{c.get('descriptor')}' -> "
                              f"'{c.get('replacement')}'  ({c['rejected_reason']})")
                else:
                    print(f"  {char}/{field}: all descriptors still_accurate")

            audit_log.append({
                "episode": ep_tag,
                "character": char,
                "field": field,
                "old_value": value,
                "new_value": _join_descriptors(new_descs, value) if applied else value,
                "applied": applied,
                "rejected": [c for c in change_log if "rejected_reason" in c],
                "still_accurate_count": sum(
                    1 for d in descs if d in new_descs and d not in [
                        c.get("descriptor") for c in applied
                    ]
                ),
                "lifetime_event_count": len(lifetime),
            })
            prev_cut[(char, field)] = (season, episode)

    fallback_log: list[dict] = []
    if getattr(args, "final_fallback", False):
        print("\n=== Final lifetime fallback audit ===")
        applied_pairs: set[tuple[str, str]] = {
            (e["character"], e["field"])
            for e in audit_log if e.get("applied")
        }
        all_pairs = [(c, f) for c in chars for f in fields]
        zero_pairs = [p for p in all_pairs if p not in applied_pairs]
        print(f"(char, field) pairs with zero prior applied changes: "
              f"{len(zero_pairs)} / {len(all_pairs)}")
        if not zero_pairs:
            print("All pairs got at least one update — fallback skipped.")
        else:
            last_season, last_episode = audit_eps[-1]
            last_tag = _ep_tag(last_season, last_episode)
            fb_tasks = []
            fb_keys = []
            for char, field in zero_pairs:
                snap = snapshots[char].get(last_tag)
                if not snap:
                    print(f"  {char}/{field}: no snapshot at {last_tag}, skip")
                    continue
                tree = snap["tree"]
                identity = tree.get("identity", {})
                persona = tree.get("persona", {})
                fdat = persona.get(field)
                if not isinstance(fdat, dict) or "value" not in fdat:
                    continue
                value = fdat["value"]
                descs = _split_descriptors(value)
                if len(descs) < 2:
                    continue
                arc = archives.get(char, [])
                lifetime_fb = [
                    e for e in arc
                    if (e["season"], e["episode"]) <= (last_season, last_episode)
                    and (
                        e.get("significance") == "high"
                        or (
                            e.get("significance") == "medium"
                            and field in (e.get("affected_fields") or [])
                        )
                    )
                ]
                lifetime_fb.sort(key=lambda e: (e["season"], e["episode"]))
                if len(lifetime_fb) < args.fallback_min_events:
                    print(f"  {char}/{field}: only {len(lifetime_fb)} lifetime "
                          f"events (need {args.fallback_min_events}), skip")
                    continue
                print(f"  {char}/{field}: firing fallback @ {last_tag} with "
                      f"{len(lifetime_fb)} events")
                fb_tasks.append(audit_field(
                    client, sem, model, char, identity, field, value,
                    lifetime_fb, last_tag,
                ))
                fb_keys.append((char, field, value, lifetime_fb, last_tag))

            if fb_tasks:
                # Temporarily bump the per-call change cap (no global mutate
                # outside this block).
                global MAX_CHANGES_PER_AUDIT
                _saved_cap = MAX_CHANGES_PER_AUDIT
                MAX_CHANGES_PER_AUDIT = max(_saved_cap, args.fallback_max_changes)
                try:
                    fb_results = await asyncio.gather(*fb_tasks)
                finally:
                    MAX_CHANGES_PER_AUDIT = _saved_cap

                for (char, field, value, lifetime_fb, last_tag), resp in zip(
                    fb_keys, fb_results,
                ):
                    descs = _split_descriptors(value)
                    if not resp:
                        print(f"  {char}/{field}: fallback LLM call failed")
                        fallback_log.append({
                            "episode": last_tag, "character": char,
                            "field": field, "status": "llm_failed",
                            "mode": "final_fallback",
                        })
                        continue
                    new_descs, change_log = validate_and_apply(
                        descs, resp, lifetime_fb, char, set(),
                    )
                    applied = [c for c in change_log
                               if "rejected_reason" not in c]
                    if applied and not args.dry_run:
                        new_value = _join_descriptors(new_descs, value)
                        forward_eps = [
                            t for t in snapshots[char]
                            if _ep_ord_tag(t) >= _ep_ord(last_season,
                                                         last_episode)
                        ]
                        for t in forward_eps:
                            ftree = snapshots[char][t]["tree"]
                            fpersona = ftree.get("persona", {})
                            ffdat = fpersona.get(field)
                            if not isinstance(ffdat, dict):
                                continue
                            if ffdat.get("value", "") == value:
                                ffdat["value"] = new_value
                                ffdat["last_updated_at"] = last_tag
                                ffdat["last_change_source"] = "core_audit_fallback"
                            else:
                                break
                    if applied:
                        print(f"  {char}/{field}: {len(applied)} fallback "
                              f"change(s) applied")
                        for c in applied:
                            print(f"    [{c['status']}] '{c['descriptor']}' "
                                  f"-> '{c['replacement']}'  "
                                  f"ev={c['evidence_session_ids']}")
                    else:
                        rejs = [c for c in change_log
                                if "rejected_reason" in c]
                        if rejs:
                            print(f"  {char}/{field}: {len(rejs)} fallback "
                                  f"proposal(s) rejected")
                            for c in rejs:
                                print(f"    REJECT '{c.get('descriptor')}' "
                                      f"-> '{c.get('replacement')}'  "
                                      f"({c['rejected_reason']})")
                        else:
                            print(f"  {char}/{field}: all descriptors "
                                  f"still_accurate (fallback)")
                    fallback_log.append({
                        "episode": last_tag,
                        "character": char,
                        "field": field,
                        "old_value": value,
                        "new_value": (_join_descriptors(new_descs, value)
                                      if applied else value),
                        "applied": applied,
                        "rejected": [c for c in change_log
                                     if "rejected_reason" in c],
                        "lifetime_event_count": len(lifetime_fb),
                        "mode": "final_fallback",
                    })

    if args.dry_run:
        print("\n[DRY-RUN] no files written")
    else:
        backup = snap_path.with_suffix(".json.before_audit")
        if not backup.exists():
            shutil.copy(snap_path, backup)
            print(f"\nBackup: {backup}")
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump(snapshots, f, ensure_ascii=False, indent=2)
        print(f"Updated snapshots: {snap_path}")

    log_path = evolution_dir / "core_audit_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(audit_log, f, ensure_ascii=False, indent=2)
    print(f"Audit log: {log_path}")

    if fallback_log:
        fb_path = evolution_dir / "core_audit_log.final_fallback.json"
        with open(fb_path, "w", encoding="utf-8") as f:
            json.dump(fallback_log, f, ensure_ascii=False, indent=2)
        print(f"Fallback log: {fb_path}")

    n_applied = sum(len(e.get("applied", [])) for e in audit_log)
    n_rejected = sum(len(e.get("rejected", [])) for e in audit_log)
    n_fb_applied = sum(len(e.get("applied", [])) for e in fallback_log)
    n_fb_rejected = sum(len(e.get("rejected", [])) for e in fallback_log)
    print(f"\nSummary: {n_applied} descriptor change(s) applied, "
          f"{n_rejected} rejected.")
    if fallback_log:
        print(f"Final fallback: {n_fb_applied} change(s) applied, "
              f"{n_fb_rejected} rejected.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["Friends", "TheOffice", "StarTrek_TNG", "HPD"],
    )
    ap.add_argument("--character", type=str, default=None,
                    help="Comma-separated character names (default: all 6 mains)")
    ap.add_argument("--fields", type=str, default=",".join(CORE_FIELDS),
                    help="Comma-separated core field names")
    ap.add_argument("--audit-episodes", type=str, default=None,
                    help="Comma-separated S##E## tags (default: every "
                         f"{DEFAULT_AUDIT_INTERVAL} eps + finale)")
    ap.add_argument("--interval", type=int, default=DEFAULT_AUDIT_INTERVAL,
                    help=f"Default audit interval in episodes "
                         f"(default: {DEFAULT_AUDIT_INTERVAL})")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true",
                    help="Run audits but do not write back to snapshots")
    ap.add_argument("--final_fallback", action="store_true",
                    help=("After the main loop, for any (character, field) "
                          "that received zero applied changes, run one extra "
                          "audit at the FINAL episode using a wider archive "
                          "(all lifetime high-sig + field-relevant medium-sig "
                          "events).  Useful for datasets with sparse high-sig "
                          "events (e.g. supporting characters in HPD)."))
    ap.add_argument("--fallback_min_events", type=int, default=4,
                    help="Min total events needed to fire a fallback audit.")
    ap.add_argument("--fallback_max_changes", type=int, default=2,
                    help="Max descriptors changed per fallback pass.")
    args = ap.parse_args()
    asyncio.run(run_audit(args))


if __name__ == "__main__":
    main()
