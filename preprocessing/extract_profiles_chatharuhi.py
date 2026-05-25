"""Extract character profiles from ChatHaruhi dialogue data using LLM.

ChatHaruhi provides only dialogue data — no explicit character profiles.
This script:

1. Loads and cleans the JSONL dialogue data.
2. Normalizes agent role names (strips action tags, merges variants).
3. Filters to main characters with ``≥ min_dialogues`` entries.
4. Auto-detects each character's primary language (EN / ZH) from their
   dialogue outputs.
5. Samples representative dialogues for each character.
6. Uses LLM with a language-matched prompt to synthesise a structured
   JSON character profile per character.
7. Saves profiles using unified English keys, language-native values,
   and ``null`` for genuinely unknown fields.

Input
    ``phase_tree_data/raw_data/ChatHaruhi/Haruhi_54K_v1.jsonl``

Output
    ``phase_tree_data/processed/ChatHaruhi/intermediate/raw_profiles.json``

Dependencies
    * ``openai``, ``python-dotenv`` — LLM API access.

Usage::

    python extract_profiles_chatharuhi.py
    python extract_profiles_chatharuhi.py --min_dialogues 50 --max_workers 16
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "phase_tree_data" / "raw_data"
PROCESSED_DIR = PROJECT_ROOT / "phase_tree_data" / "processed"
load_dotenv(PROJECT_ROOT / ".env")

RANDOM_SEED = 42

PROFILE_FIELDS = [
    "name", "gender", "age", "occupation",
    "personality", "values_and_beliefs", "emotional_patterns",
    "speaking_style", "catchphrases",
    "behavioral_traits", "expertise_and_skills", "quirks",
    "background", "relationships", "hobbies",
    "goals_and_motivations",
]

# ---------------------------------------------------------------------------
# Name normalisation helpers  (shared logic with preprocess_dialogues)
# ---------------------------------------------------------------------------

NORMALIZE_MAP = {
    "哈利": "Harry", "罗恩": "Ron", "赫敏": "Hermione",
    "邓布利多": "Dumbledore", "斯内普": "Snape", "小天狼星": "Sirius",
    "麦格教授": "Professor McGonagall",
    "教授麦格": "Professor McGonagall",
    "教授麦格教": "Professor McGonagall",
    "教授麦格教嗔地说道": "Professor McGonagall",
    "教授麦格教嗔怒地说道": "Professor McGonagall",
    "教授麦格教授": "Professor McGonagall",
    "教授麦格教练": "Professor McGonagall",
    "McGonagall教授": "Professor McGonagall",
    "教授特里劳妮": "Professor Trelawney",
    "Professor Snape": "Snape",
    "萧峰": "乔峰", "萧峰见": "乔峰",
}

NOISE_NAMES = frozenset({
    "", "旁白", "Narrator", "观众", "学生", "Student", "百姓A",
    "保安", "经理", "警察", "顾客", "囚犯", "士兵", "访客",
    "老人", "老僧", "陌生人", "女子", "年轻书生", "大哥", "和尚",
    "那人", "蒙面人", "黑衣女郎", "青衣少女", "沙溢吕秀才", "大聪明",
})


def normalize_name(name: str) -> str:
    cleaned = re.sub(r"（[^）]*）", "", name).strip()
    cleaned = re.sub(r"\([^)]*\)", "", cleaned).strip()
    return NORMALIZE_MAP.get(cleaned, cleaned)


def is_noise(name: str) -> bool:
    if name in NOISE_NAMES or len(name) > 15:
        return True
    if re.match(r"^('''|Apologies|对不起|抱歉)", name):
        return True
    return False


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_ASCII_RE = re.compile(r"[a-zA-Z]")


def _entry_lang(entry: dict) -> str:
    """Detect the language of a single dialogue entry from its response."""
    resp = entry.get("agent_response", "")
    if not resp:
        return "zh"
    ratio = len(_ASCII_RE.findall(resp)) / len(resp)
    return "en" if ratio > 0.5 else "zh"


def detect_language(dialogues: list[dict]) -> str:
    """Return ``'en'`` if the character predominantly speaks English."""
    total_chars = 0
    ascii_chars = 0
    for d in dialogues:
        resp = d.get("agent_response", "")
        total_chars += len(resp)
        ascii_chars += len(_ASCII_RE.findall(resp))
    ratio = ascii_chars / total_chars if total_chars else 0
    return "en" if ratio > 0.5 else "zh"


def filter_by_language(dialogues: list[dict], lang: str) -> list[dict]:
    """Keep only entries whose response language matches *lang*."""
    return [d for d in dialogues if _entry_lang(d) == lang]


# ---------------------------------------------------------------------------
# LLM prompts — one per language
# ---------------------------------------------------------------------------

_SYSTEM_EN = """\
You are a character profiling specialist. You will be given dialogue \
excerpts featuring a fictional character. Your task is to analyse their \
speech patterns, behaviour, and relationships to produce a comprehensive \
character profile in JSON.

## Critical rules
1. Use ONLY evidence from the provided dialogues. Do NOT use your own \
knowledge about the character, their franchise, or any external source.
2. For any field where the dialogues provide NO evidence, set its value \
to null. Never guess or fabricate.
3. **Language: ALL value strings MUST be written in English.** This is \
non-negotiable — do not mix in any other language.
4. Each fact goes in exactly ONE field — no duplication across fields.
5. Keep values concise. Prefer comma-separated descriptors over long prose.

## Field definitions — read each carefully before filling

### Basic info
- **name**: The character's name as it appears in the dialogues.
- **gender**: Gender if clearly inferable, otherwise null.
- **age**: Age or age range if inferable (e.g. "teenager", "mid-30s"), \
otherwise null.
- **occupation**: Job title, role, or social position (e.g. "student", \
"physicist", "headmaster").

### Inner world
- **personality**: Innate character traits, temperament \
(e.g. "brave, loyal, impulsive, empathetic"). Do NOT include values/beliefs, \
emotional patterns, or hobbies here.
- **values_and_beliefs**: Core worldview, moral principles, things they \
stand for (e.g. "believes in justice and fairness, values friendship above \
all, pragmatic worldview"). Distinct from personality traits.
- **emotional_patterns**: Typical emotional reactions, triggers, emotional \
coping style (e.g. "quick to anger when friends are threatened, hides \
vulnerability behind humor, anxious under pressure").
- **goals_and_motivations**: What drives the character, their ambitions, \
desires (e.g. "wants to defeat Voldemort, seeks to protect loved ones").

### Speech & behaviour
- **speaking_style**: A DESCRIPTION of HOW the character talks — tone, \
register, verbal habits, sentence patterns. Do NOT put actual quotes here. \
(e.g. "formal and verbose, frequently uses scientific jargon, tends to \
lecture others, dry wit").
- **catchphrases**: A JSON **array** of actual iconic lines, verbal tics, \
or recurring expressions quoted verbatim from the dialogues. Include 2-5 \
representative examples if available. This complements speaking_style with \
raw evidence. MUST be an array of strings, e.g. ["line1", "line2"].
- **behavioral_traits**: Recurring action patterns, social conduct, \
habitual behaviours (e.g. "takes charge in dangerous situations, \
methodical and organized, avoids confrontation").
- **expertise_and_skills**: Specific abilities, knowledge domains, or \
talents demonstrated in the dialogues (e.g. "skilled at Quidditch, \
knowledgeable about Defence Against the Dark Arts").
- **quirks**: Distinctive mannerisms, unusual habits, idiosyncrasies \
(e.g. "knocks three times before entering, insists on a specific seat").

### Context
- **background**: 1-2 sentence summary of the most pivotal life events \
visible from the dialogues. Do NOT include occupation, relationships, or \
personality here.
- **relationships**: Key interpersonal connections. Format each as \
"role is Name" (e.g. "best friend is Ron, mentor is Dumbledore").
- **hobbies**: Interests, specific likes and dislikes \
(e.g. "enjoys Quidditch, likes treacle tart, dislikes Potions class").

## Output — valid JSON only, no markdown fences
{
  "name": "...",
  "gender": "..." or null,
  "age": "..." or null,
  "occupation": "..." or null,
  "personality": "..." or null,
  "values_and_beliefs": "..." or null,
  "emotional_patterns": "..." or null,
  "goals_and_motivations": "..." or null,
  "speaking_style": "..." or null,
  "catchphrases": ["...", "..."] or null,
  "behavioral_traits": "..." or null,
  "expertise_and_skills": "..." or null,
  "quirks": "..." or null,
  "background": "..." or null,
  "relationships": "..." or null,
  "hobbies": "..." or null
}"""

_SYSTEM_ZH = """\
你是一位角色画像分析专家。你将收到一个虚构角色的对话片段，需要从多个维度全面\
分析该角色，生成结构化的 JSON 角色画像。

## 关键规则
1. 仅使用提供的对话内容作为证据。禁止使用你对该角色、作品或任何外部来源的知识。
2. 对于对话中完全没有线索的字段，值设为 null。禁止猜测或编造。
3. **语言：所有值字符串必须使用中文书写。** 这是强制要求，不允许混入其他语言。
4. 每条信息只出现在一个字段中，禁止跨字段重复。
5. 值要简洁，优先用逗号分隔的短描述，不要长段落叙述。

## 字段定义——填写前仔细阅读每个字段的边界

### 基本信息
- **name**：角色在对话中出现的名字。
- **gender**：性别，能明确推断则填写，否则 null。
- **age**：年龄或年龄段（如"青少年"、"三十多岁"），无法推断则 null。
- **occupation**：职业、身份、社会角色（如"学生"、"公公"、"香主"）。

### 内心世界
- **personality**（性格特质）：天生的性格、气质（如"勇敢、善良、冲动、\
义气深重"）。不要在此放价值观、情绪模式或爱好。
- **values_and_beliefs**（价值观与信念）：核心世界观、道德准则、信仰 \
（如"重义轻利、忠诚至上、实用主义"）。和性格特质不同维度。
- **emotional_patterns**（情绪模式）：典型的情绪反应方式、触发条件、\
情绪调节方式（如"朋友受威胁时容易愤怒、用幽默掩饰脆弱、压力下焦虑"）。
- **goals_and_motivations**（目标与动机）：驱动角色行动的欲望和追求 \
（如"想成为海贼王、追求财富、为父报仇"）。

### 言行模式
- **speaking_style**（说话风格）：描述角色的说话方式——语气、语域、\
口头习惯、句式特征（如"油腔滑调、善用比喻、爱用反问句、带有江湖气"）。\
不要放原文台词，只描述说话方式。
- **catchphrases**（口头禅/经典台词）：一个 JSON **数组**，从对话中\
直接引用的标志性台词、口头禅、常用表达，原文摘录 2-5 条。\
必须是字符串数组，如 ["台词1", "台词2"]。这是 speaking_style 的原始证据补充。
- **behavioral_traits**（行为特征）：反复出现的行为模式、社交方式、\
习惯性做法（如"擅长随机应变、善于讨好上级、遇事先逃跑"）。
- **expertise_and_skills**（专长与技能）：在对话中展现出的具体能力、\
知识领域、才艺（如"精通武功、善于赌术、熟悉宫廷礼仪"）。
- **quirks**（怪癖/标志性小习惯）：独特的行为习惯、癖好、标志性动作 \
（如"说谎时摸鼻子、吃东西前先闻一闻、喜欢给人起外号"）。

### 背景与关系
- **background**（背景经历）：1-2 句话概括对话中可见的最关键人生经历。\
不要在此重复职业、关系或性格。
- **relationships**（人际关系）：关键人物关系，格式为"关系为人名" \
（如"妻子为双儿，结拜兄弟为康熙，师傅为陈近南"）。
- **hobbies**（兴趣爱好）：具体的兴趣、喜好、厌恶 \
（如"喜欢赌钱、爱吃糖果、讨厌读书"）。

## 输出——仅输出合法 JSON，不要 markdown 代码块
{
  "name": "...",
  "gender": "..." 或 null,
  "age": "..." 或 null,
  "occupation": "..." 或 null,
  "personality": "..." 或 null,
  "values_and_beliefs": "..." 或 null,
  "emotional_patterns": "..." 或 null,
  "goals_and_motivations": "..." 或 null,
  "speaking_style": "..." 或 null,
  "catchphrases": ["...", "..."] 或 null,
  "behavioral_traits": "..." 或 null,
  "expertise_and_skills": "..." 或 null,
  "quirks": "..." 或 null,
  "background": "..." 或 null,
  "relationships": "..." 或 null,
  "hobbies": "..." 或 null
}"""


def _build_user_prompt(char_name: str, dialogues_text: str,
                       n_dialogues: int, lang: str) -> str:
    if lang == "en":
        return (
            f"Character name: {char_name}\n\n"
            f"Below are {n_dialogues} dialogue excerpts featuring this "
            f"character:\n\n{dialogues_text}\n\n"
            f"Analyse the dialogues and produce the JSON profile."
        )
    return (
        f"角色名称：{char_name}\n\n"
        f"以下是该角色的 {n_dialogues} 段对话：\n\n{dialogues_text}\n\n"
        f"请分析上述对话，输出 JSON 角色画像。"
    )


# ---------------------------------------------------------------------------
# Profile generation
# ---------------------------------------------------------------------------

def format_dialogue(entry: dict, agent_role: str, lang: str = "en") -> str:
    fallback = "User" if lang == "en" else "用户"
    user_role = entry.get("user_role", "").strip() or fallback
    lines = [f"{user_role}: {entry['user_question']}",
             f"{agent_role}: {entry['agent_response']}"]
    for turn in entry.get("more_dialogues", []):
        lines.append(turn)
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


def _normalise_profile(raw: dict) -> dict:
    """Ensure all expected fields exist; convert empty / 'unknown' to null."""
    skip = {"未知", "无", "暂无", "不详", "unknown", "Unknown", "N/A", "n/a",
            "none", "None", ""}
    out: dict = {}
    for key in PROFILE_FIELDS:
        val = raw.get(key)
        if key == "catchphrases":
            if isinstance(val, list):
                val = [str(v).strip() for v in val if v]
            elif isinstance(val, str) and val.strip() and val.strip() not in skip:
                val = [val.strip()]
            else:
                val = None
            out[key] = val if val else None
        else:
            if isinstance(val, list):
                val = ", ".join(str(v).strip() for v in val if v)
            if val is None or (isinstance(val, str) and val.strip() in skip):
                out[key] = None
            else:
                out[key] = val.strip() if isinstance(val, str) else val
    return out


def generate_profile(
    client: OpenAI, model: str, char_name: str,
    dialogues: list[dict], lang: str,
    n_sample: int = 30, max_retries: int = 3,
) -> dict:
    rng = random.Random(RANDOM_SEED)
    story = [d for d in dialogues if d.get("question_source") == "story"]
    pool = story if len(story) >= n_sample else dialogues
    sampled = rng.sample(pool, min(n_sample, len(pool)))

    parts = []
    for i, d in enumerate(sampled, 1):
        label = f"[Dialogue {i}]" if lang == "en" else f"[对话{i}]"
        parts.append(f"{label}\n{format_dialogue(d, char_name, lang)}")
    dialogues_text = "\n\n".join(parts)

    system = _SYSTEM_EN if lang == "en" else _SYSTEM_ZH
    user = _build_user_prompt(char_name, dialogues_text, len(sampled), lang)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=2500,
            )
            raw = _extract_json(resp.choices[0].message.content)
            return _normalise_profile(raw)
        except Exception as e:
            print(f"    [{char_name} attempt {attempt}] {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)

    return {k: None for k in PROFILE_FIELDS}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract character profiles from ChatHaruhi dialogues",
    )
    ap.add_argument("--min_dialogues", type=int, default=100,
                    help="Minimum dialogues to include a character")
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--n_sample", type=int, default=30,
                    help="Dialogues sampled per character for LLM analysis")
    args = ap.parse_args()

    src = RAW_DATA_DIR / "ChatHaruhi" / "Haruhi_54K_v1.jsonl"
    data: list[dict] = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} entries from {src}")

    role_dialogues: dict[str, list[dict]] = defaultdict(list)
    for entry in data:
        norm = normalize_name(entry["agent_role"])
        if not is_noise(norm):
            role_dialogues[norm].append(entry)

    main_chars = {k: v for k, v in role_dialogues.items()
                  if len(v) >= args.min_dialogues}
    print(f"\nMain characters (>= {args.min_dialogues} dialogues): "
          f"{len(main_chars)}")

    char_langs: dict[str, str] = {}
    for name in sorted(main_chars, key=lambda n: -len(main_chars[n])):
        lang = detect_language(main_chars[name])
        char_langs[name] = lang
        before = len(main_chars[name])
        main_chars[name] = filter_by_language(main_chars[name], lang)
        after = len(main_chars[name])
        dropped = before - after
        suffix = f"  (dropped {dropped} mismatched)" if dropped else ""
        print(f"  {name}: {after:>5} dialogues  [{lang}]{suffix}")

    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )
    model = os.getenv("LLM_MODEL", "gpt-4.1")
    char_names = sorted(main_chars)

    print(f"\nGenerating profiles via LLM ({model}, "
          f"{args.max_workers} workers) ...")
    profiles: dict[str, dict] = {}
    total_chars = len(char_names)
    t0 = time.time()

    def _gen(name: str):
        prof = generate_profile(
            client, model, name, main_chars[name],
            char_langs[name], args.n_sample,
        )
        prof["_lang"] = char_langs[name]
        return name, prof

    def _progress(done_n, total_n, elapsed_s, width=30):
        pct = done_n / total_n if total_n else 1
        filled = int(width * pct)
        bar = "█" * filled + "░" * (width - filled)
        rate = done_n / elapsed_s if elapsed_s > 0 else 0
        eta = (total_n - done_n) / rate if rate > 0 else 0
        return (f"\r  {bar} {pct:5.1%} | {done_n}/{total_n} | "
                f"{rate:.1f} it/s | ETA {eta:.0f}s")

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {pool.submit(_gen, n): n for n in char_names}
        for i, fut in enumerate(as_completed(futs), 1):
            name, prof = fut.result()
            profiles[name] = prof
            sys.stdout.write(_progress(i, total_chars, time.time() - t0))
            sys.stdout.flush()

    elapsed = time.time() - t0
    out_dir = PROCESSED_DIR / "ChatHaruhi" / "intermediate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw_profiles.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(profiles)} profiles saved -> {out_path} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
