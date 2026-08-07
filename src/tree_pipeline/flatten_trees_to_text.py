"""Flatten attribute trees into continuous natural-language profile text.

Four modes are supported:

* ``phase`` (default) — per-sample trees with session/moment filled
  (short-term flow, frozen p00 persona).
  Input: ``intermediate/{split}_phase_updated_trees.json``
  Output: ``intermediate/{split}_phase_profile_texts.json``

* ``static`` — frozen long-term trees (session/moment cleared).
  Input: ``intermediate/attribute_trees.json``
  Output: ``intermediate/static_tree_profile_texts.json``

* ``long_term_phase`` — per-sample trees with time-evolved persona@T-1
  plus per-dialogue session/moment.
  Input: ``intermediate/{split}_long_term_phase_updated_trees.json``
  Output: ``intermediate/{split}_long_term_phase_profile_texts.json``

* ``long_term_dynamic`` — per-sample trees with time-evolved persona@T-1
  only (session/moment cleared) — counterpart of ``static`` for the long
  track.
  Input: ``intermediate/{split}_long_term_dynamic_trees.json``
  Output: ``intermediate/{split}_long_term_dynamic_profile_texts.json``

Semantic grouping order:

    Layer 1 (basic facts): name + demographics + occupation + backstory
    Layer 2 (social relations): relationships
    Layer 3 (intrinsic traits): personality + hobbies + speaking_style
    Layer 4 (behavioural tendencies): behavioral_tendencies
    Layer 5 (current state): session + moment (phase mode only)

Usage::

    # Phase: process one split
    python flatten_trees_to_text.py --dataset CharacterEval --mode phase --split ood_test

    # Phase: process all splits
    python flatten_trees_to_text.py --dataset CharacterEval --mode phase --split all

    # Static: generate per-character profile texts (--split is ignored)
    python flatten_trees_to_text.py --dataset CharacterEval --mode static
"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "LongEvoRoleBench" / "processed"


def _strip_trailing_punct(s: str) -> str:
    """Remove trailing punctuation (。，、；) from a string."""
    return s.rstrip("。，、；.， ")


_SYNONYM_MAP = {
    "爱好": ("爱好", "喜欢", "喜爱", "热爱"),
    "性格": ("性格",),
    "说话风格": ("说话风格", "说话"),
}


def _prefixed(prefix: str, value: str) -> str:
    """Add *prefix* only if *value* does not already start with a synonym."""
    value = _strip_trailing_punct(value)
    base = prefix.rstrip("为")
    synonyms = _SYNONYM_MAP.get(base, (base, prefix))
    for kw in synonyms:
        if value.startswith(kw):
            return value
    return f"{prefix}{value}"


def _is_english_tree(tree: dict) -> bool:
    """Detect whether a tree is English-language based on the name field."""
    name = tree.get("identity", {}).get("name") or ""
    return sum(c < "\u0080" for c in name) > len(name) * 0.5


_VERB_STARTERS = frozenset({
    "Shifted", "Became", "Moved", "Changed", "Decided", "Committed",
    "Promised", "Urged", "Agreed", "Expressed", "Reinforced", "Embraced",
    "Considered", "Acknowledged", "Proposed", "Invited", "Encouraged",
    "Began", "Reaffirmed", "Affirmed", "Determined", "Renewed",
    "Strengthened", "Clarified", "Intends", "Resolved", "Focused",
    "Recognized", "Accepted", "Vowed", "Opened", "Momentarily",
})


def _first_name(full_name: str) -> str:
    """Extract the given name, skipping honorific titles."""
    _TITLES = {"Dr.", "Mr.", "Mrs.", "Ms.", "Sgt.", "Sergeant", "Prof.",
               "Commander", "Captain", "Lieutenant", "Major", "General"}
    parts = full_name.split()
    if not parts:
        return "The character"
    for p in parts:
        if p not in _TITLES and not p.startswith("'") and not p.startswith('"'):
            return p
    return parts[-1]


def _add_subject(sentence: str, subj: str) -> str:
    """Prepend *subj* as grammatical subject when the sentence lacks one."""
    if not sentence:
        return sentence
    first_word = sentence.split()[0]
    if first_word == "From":
        rest = sentence[len("From"):].lstrip()
        return f"{subj} moved from {rest}"
    if first_word in _VERB_STARTERS:
        return f"{subj} {sentence[0].lower()}{sentence[1:]}"
    return sentence


def flatten_tree(tree: dict) -> str:
    """Convert an attribute tree into continuous natural-language text."""
    if _is_english_tree(tree):
        return _flatten_tree_en(tree)
    return _flatten_tree_zh(tree)


def _flatten_tree_en(tree: dict) -> str:
    """Flatten an English attribute tree into a paragraph."""
    identity = tree.get("identity", {})
    persona = tree.get("persona", {})
    session = tree.get("session", {})
    moment = tree.get("moment", {})

    def pval(field: str) -> str | None:
        v = persona.get(field, {})
        if isinstance(v, dict):
            return v.get("value")
        return v

    layer1_parts = []
    name = identity.get("name", "")
    if name:
        layer1_parts.append(name)
    demo = pval("demographics")
    if demo:
        layer1_parts.append(_strip_trailing_punct(demo))
    occ = pval("occupation")
    if occ:
        layer1_parts.append(_strip_trailing_punct(occ))
    layer1 = ", ".join(layer1_parts) if layer1_parts else ""

    backstory = identity.get("backstory")
    if backstory:
        layer1 = layer1 + ". " + _strip_trailing_punct(backstory) if layer1 else _strip_trailing_punct(backstory)

    rels = pval("relationships")
    layer2 = _strip_trailing_punct(rels) if rels else ""

    layer3_parts = []
    pers = pval("personality")
    if pers:
        layer3_parts.append(_strip_trailing_punct(pers))
    hobbies = pval("hobbies")
    if hobbies:
        layer3_parts.append(_strip_trailing_punct(hobbies))
    style = pval("speaking_style")
    if style:
        layer3_parts.append(_strip_trailing_punct(style))
    layer3 = ". ".join(layer3_parts) if layer3_parts else ""

    behav = pval("behavioral_tendencies")
    layer4 = _strip_trailing_punct(behav) if behav else ""

    state_sentences: list[str] = []
    subj = _first_name(name)

    emo = moment.get("emotion", "neutral")
    intensity = moment.get("emotion_intensity", 3)
    if emo and emo != "neutral" and intensity >= 3:
        state_sentences.append(f"Currently feeling {emo} (intensity {intensity}/10)")

    scene = moment.get("scene_context")
    if scene:
        state_sentences.append(_strip_trailing_punct(scene))

    learned = session.get("learned_info", [])
    if learned:
        state_sentences.append(
            "; ".join(_strip_trailing_punct(x) for x in learned))

    att_shifts = session.get("attitude_shifts", {})
    if att_shifts:
        for target, shift in att_shifts.items():
            shift_str = str(shift).strip()
            if "→" in shift_str:
                natural = shift_str.replace("→", " to ")
                state_sentences.append(
                    f"{subj}'s attitude toward {target} shifted from "
                    f"{natural}")
            else:
                state_sentences.append(
                    f"{subj}'s attitude toward {target} is {shift_str}")

    commits = session.get("commitments", [])
    if commits:
        for c in commits:
            state_sentences.append(
                _add_subject(_strip_trailing_punct(c), subj))

    stances = session.get("stance_changes", [])
    if stances:
        for s in stances:
            state_sentences.append(
                _add_subject(_strip_trailing_punct(s), subj))

    layer5 = ". ".join(state_sentences) if state_sentences else ""

    persona_layers = [l for l in [layer1, layer2, layer3, layer4] if l]
    persona_text = ". ".join(persona_layers)

    if layer5:
        return persona_text + ". " + layer5 + "."
    elif persona_text:
        return persona_text + "."
    else:
        return name or ""


def _flatten_tree_zh(tree: dict) -> str:
    """Flatten a Chinese attribute tree into a paragraph (original logic)."""
    identity = tree.get("identity", {})
    persona = tree.get("persona", {})
    session = tree.get("session", {})
    moment = tree.get("moment", {})

    def pval(field: str) -> str | None:
        v = persona.get(field, {})
        if isinstance(v, dict):
            return v.get("value")
        return v

    layer1_parts = []
    name = identity.get("name", "")
    if name:
        layer1_parts.append(name)

    demo = pval("demographics")
    if demo:
        layer1_parts.append(_strip_trailing_punct(demo))

    occ = pval("occupation")
    if occ:
        layer1_parts.append(_strip_trailing_punct(occ))

    layer1 = "，".join(layer1_parts) if layer1_parts else ""

    backstory = identity.get("backstory")
    if backstory:
        layer1 = layer1 + "。" + _strip_trailing_punct(backstory) if layer1 else _strip_trailing_punct(backstory)

    rels = pval("relationships")
    layer2 = _strip_trailing_punct(rels) if rels else ""

    layer3_parts = []

    pers = pval("personality")
    if pers:
        layer3_parts.append(_prefixed("性格", pers))

    hobbies = pval("hobbies")
    if hobbies:
        layer3_parts.append(_prefixed("爱好", hobbies))

    style = pval("speaking_style")
    if style:
        layer3_parts.append(_prefixed("说话风格为", style))

    layer3 = "，".join(layer3_parts) if layer3_parts else ""

    behav = pval("behavioral_tendencies")
    layer4 = _strip_trailing_punct(behav) if behav else ""

    state_parts = []

    emo = moment.get("emotion", "neutral")
    intensity = moment.get("emotion_intensity", 3)
    if emo and emo != "neutral" and intensity >= 3:
        state_parts.append(f"情绪{emo}（强度{intensity}/10）")

    scene = moment.get("scene_context")
    if scene:
        state_parts.append(_strip_trailing_punct(scene))

    learned = session.get("learned_info", [])
    if learned:
        state_parts.append("；".join(_strip_trailing_punct(x) for x in learned))

    att_shifts = session.get("attitude_shifts", {})
    if att_shifts:
        shifts = [f"对{k}的态度{v}" for k, v in att_shifts.items()]
        state_parts.append("，".join(shifts))

    commits = session.get("commitments", [])
    if commits:
        state_parts.append("；".join(_strip_trailing_punct(x) for x in commits))

    stances = session.get("stance_changes", [])
    if stances:
        state_parts.append("；".join(_strip_trailing_punct(x) for x in stances))

    layer5 = "，".join(state_parts) if state_parts else ""

    persona_layers = [l for l in [layer1, layer2, layer3, layer4] if l]
    persona_text = "。".join(persona_layers)

    if layer5:
        return persona_text + "。当前状态为" + layer5 + "。"
    elif persona_text:
        return persona_text + "。"
    else:
        return name or ""


def process_split_phase(intermediate_dir: Path, split_name: str):
    """Load per-sample phase trees for a split, flatten, and save."""
    trees_path = intermediate_dir / f"{split_name}_phase_updated_trees.json"
    if not trees_path.exists():
        print(f"[{split_name}] File not found: {trees_path}, skipping")
        return

    with open(trees_path, "r", encoding="utf-8") as f:
        trees = json.load(f)

    profile_texts = {}
    for qid, tree in trees.items():
        profile_texts[qid] = flatten_tree(tree)

    out_path = intermediate_dir / f"{split_name}_phase_profile_texts.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile_texts, f, ensure_ascii=False, indent=2)

    examples = list(profile_texts.items())[:3]
    print(f"\n[{split_name}] {len(profile_texts)} profile texts -> {out_path}")
    print("-" * 60)
    for qid, text in examples:
        print(f"  {qid}:")
        print(f"    {text[:150]}{'...' if len(text) > 150 else ''}")
    print()


def process_static(intermediate_dir: Path):
    """Flatten frozen attribute trees (session/moment cleared) per character."""
    import copy as _copy

    trees_path = intermediate_dir / "attribute_trees.json"
    if not trees_path.exists():
        print(f"File not found: {trees_path}")
        return

    with open(trees_path, "r", encoding="utf-8") as f:
        all_trees = json.load(f)

    profile_texts: dict[str, str] = {}
    for char_name, tree in all_trees.items():
        static_tree = _copy.deepcopy(tree)
        static_tree["session"] = {}
        static_tree["moment"] = {}
        profile_texts[char_name] = flatten_tree(static_tree)

    out_path = intermediate_dir / "static_tree_profile_texts.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile_texts, f, ensure_ascii=False, indent=2)

    examples = list(profile_texts.items())[:3]
    print(f"\n[static] {len(profile_texts)} profile texts -> {out_path}")
    print("-" * 60)
    for name, text in examples:
        print(f"  {name}:")
        print(f"    {text[:150]}{'...' if len(text) > 150 else ''}")
    print()


def _process_per_sample_split(
    intermediate_dir: Path,
    split_name: str,
    in_suffix: str,
    out_suffix: str,
):
    """Generic per-sample flattener parameterised by file suffix."""
    trees_path = intermediate_dir / f"{split_name}{in_suffix}.json"
    if not trees_path.exists():
        print(f"[{split_name}] File not found: {trees_path}, skipping")
        return

    with open(trees_path, "r", encoding="utf-8") as f:
        trees = json.load(f)

    profile_texts = {qid: flatten_tree(tree) for qid, tree in trees.items()}

    out_path = intermediate_dir / f"{split_name}{out_suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile_texts, f, ensure_ascii=False, indent=2)

    examples = list(profile_texts.items())[:3]
    print(f"\n[{split_name}] {len(profile_texts)} profile texts -> {out_path}")
    print("-" * 60)
    for qid, text in examples:
        print(f"  {qid}:")
        print(f"    {text[:150]}{'...' if len(text) > 150 else ''}")
    print()


def process_split_long_term_phase(intermediate_dir: Path, split_name: str):
    """Flatten long-term phase trees (persona@T-1 + session + moment)."""
    _process_per_sample_split(
        intermediate_dir, split_name,
        in_suffix="_long_term_phase_updated_trees",
        out_suffix="_long_term_phase_profile_texts",
    )


def process_split_long_term_dynamic(intermediate_dir: Path, split_name: str):
    """Flatten long-term dynamic trees (persona@T-1 only, no session/moment)."""
    _process_per_sample_split(
        intermediate_dir, split_name,
        in_suffix="_long_term_dynamic_trees",
        out_suffix="_long_term_dynamic_profile_texts",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Flatten attribute trees into profile text"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["CharacterEval", "RAIDEN", "ChatHaruhi", "SimsConv", "Friends", "TheOffice", "StarTrek_TNG", "HPD"],
                        help="Dataset to process")
    parser.add_argument("--mode", type=str, default="phase",
                        choices=["phase", "static",
                                 "long_term_phase", "long_term_dynamic"],
                        help="phase=per-sample short-term trees; "
                             "static=frozen long-term trees; "
                             "long_term_phase=per-sample trees with "
                             "persona@T-1 + session + moment; "
                             "long_term_dynamic=per-sample trees with "
                             "persona@T-1 only (default: phase)")
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "random_test", "ood_test", "all"],
                        help="Which split to process (per-sample modes only; "
                             "'all' = train+random_test+ood_test, then merge "
                             "into all_dialogues)")
    args = parser.parse_args()

    intermediate_dir = PROCESSED_DIR / args.dataset / "intermediate"

    if args.mode == "static":
        process_static(intermediate_dir)
        return

    splits = (
        ["train", "random_test", "ood_test"]
        if args.split == "all" else [args.split]
    )

    if args.mode == "phase":
        for split_name in splits:
            process_split_phase(intermediate_dir, split_name)
        if args.split == "all":
            _merge_to_all_dialogues(intermediate_dir)
    elif args.mode == "long_term_phase":
        for split_name in splits:
            process_split_long_term_phase(intermediate_dir, split_name)
        if args.split == "all":
            _merge_split_files(
                intermediate_dir,
                "_long_term_phase_profile_texts.json",
                "all_dialogues_long_term_phase_profile_texts.json",
            )
    elif args.mode == "long_term_dynamic":
        for split_name in splits:
            process_split_long_term_dynamic(intermediate_dir, split_name)
        if args.split == "all":
            _merge_split_files(
                intermediate_dir,
                "_long_term_dynamic_profile_texts.json",
                "all_dialogues_long_term_dynamic_profile_texts.json",
            )


def _merge_to_all_dialogues(intermediate_dir: Path) -> None:
    """Merge train/random_test/ood_test phase_profile_texts into all_dialogues."""
    _merge_split_files(
        intermediate_dir,
        "_phase_profile_texts.json",
        "all_dialogues_phase_profile_texts.json",
    )


def _merge_split_files(
    intermediate_dir: Path, in_suffix: str, out_name: str,
) -> None:
    """Generic merger across train/random_test/ood_test splits."""
    merged: dict[str, str] = {}
    for split in ["train", "random_test", "ood_test"]:
        p = intermediate_dir / f"{split}{in_suffix}"
        if not p.exists():
            print(f"  [merge {in_suffix}] {split} not found, skipping")
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged.update(data)
        print(f"  [merge {in_suffix}] {split}: {len(data)} entries")

    out_path = intermediate_dir / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\n  Merged {len(merged)} entries -> {out_path}")


if __name__ == "__main__":
    main()
