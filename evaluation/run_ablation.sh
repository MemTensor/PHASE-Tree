#!/usr/bin/env bash
# ===========================================================================
# PHASE-Tree Ablation: Raw-Profile Reference Judge Scoring (Parallel)
# ===========================================================================
# Re-scores predictions using the RAW PROFILE persona reference
# (m2_raw_profile) instead of the full PHASE-Tree profile used in main.
#
# This ablation answers: "if the judge sees the same minimal raw description
# for every method (no tree, no session/moment), does m6 still win?"
#
# Prerequisites:
#   Main evaluation must have been run first (predictions must exist).
#
# Shared with main (symlinked, NOT re-generated):
#   - predictions.jsonl       (model output is independent of judge reference)
#   - embedding_scores.jsonl  (cosine similarity is independent of judge ref)
#
# Re-generated:
#   - judge_scores.jsonl      (LLM judge sees raw_profile only)
#
# Usage:
#   bash evaluation/run_ablation.sh <DATASET> [<MODE>] [METHODS...]
#
# Modes:
#   short-term   : m1..m4 + m6 (RAIDEN, CharacterEval, ...)     — default if unset
#   long-term    : m1..m6 (Friends, ...)                       — auto-detected
#
# Examples:
#   bash evaluation/run_ablation.sh RAIDEN
#   bash evaluation/run_ablation.sh Friends long-term
#   bash evaluation/run_ablation.sh Friends long-term m6_phase_tree
#
# Env knobs:
#   METHODS          override default method list (space-separated)
#   EVAL_SPLITS      override split list           (default: random_test ood_test)
#   NUM_WORKERS      LLM judge concurrency per task
#                    (default 10 short-term / 128 long-term)
#   RATE_LIMIT_SLEEP per-call sleep before API hit
#                    (default 0.1 short-term / 0.0 long-term)
#   SEQUENTIAL       =1 to run tasks serially (debug only; default parallel)
# ===========================================================================

set -uo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-/dev/shm/phase/.venv/bin/python}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# ---- Arg parsing ----------------------------------------------------------
DATASET="${1:?Usage: bash evaluation/run_ablation.sh <DATASET> [<MODE>] [METHODS...]}"
shift

# Optional MODE arg (short-term / long-term)
MODE=""
if [ "$#" -gt 0 ]; then
    case "$1" in
        short-term|long-term)
            MODE="$1"; shift ;;
    esac
fi

# Auto-detect MODE from data if not specified
MAIN_DIR="results/${DATASET}/prompt/main"
if [ -z "$MODE" ]; then
    if [ -d "${MAIN_DIR}/m5_dynamic_tree" ]; then
        MODE="long-term"
    elif [ -d "${MAIN_DIR}/m6_phase_tree" ]; then
        MODE="short-term"
    else
        echo "ERROR: cannot auto-detect MODE for ${DATASET} (no m5_dynamic_tree or m6_phase_tree dir in ${MAIN_DIR})" >&2
        exit 1
    fi
fi

# Default method list by MODE
if [ "$MODE" = "long-term" ]; then
    DEFAULT_METHODS=(m1_context_only m2_raw_profile m3_naive_rewrite m4_static_tree m5_dynamic_tree m6_phase_tree)
    DEFAULT_NUM_WORKERS=128
    DEFAULT_RATE_LIMIT=0.0
else
    DEFAULT_METHODS=(m1_context_only m2_raw_profile m3_naive_rewrite m4_static_tree m6_phase_tree)
    DEFAULT_NUM_WORKERS=10
    DEFAULT_RATE_LIMIT=0.1
fi

# METHODS: prefer positional args > env var > defaults
if [ "$#" -gt 0 ]; then
    METHODS=("$@")
elif [ -n "${METHODS:-}" ]; then
    IFS=' ' read -ra METHODS <<< "$METHODS"
else
    METHODS=("${DEFAULT_METHODS[@]}")
fi

IFS=' ' read -ra EVAL_SPLITS <<< "${EVAL_SPLITS:-random_test ood_test}"
NUM_WORKERS="${NUM_WORKERS:-$DEFAULT_NUM_WORKERS}"
RATE_LIMIT_SLEEP="${RATE_LIMIT_SLEEP:-$DEFAULT_RATE_LIMIT}"
SEQUENTIAL="${SEQUENTIAL:-0}"

DATA_DIR="phase_tree_data/processed/${DATASET}"
ABLATION_DIR="results/${DATASET}/prompt/ablation"
LOG_DIR="${ABLATION_DIR}/_logs"
mkdir -p "$LOG_DIR"

# Raw profile persona reference for the judge (original description, no tree)
PERSONA_DATA="${DATA_DIR}/m2_raw_profile/all_dialogues.json"

_start_time=$(date +%s)
_step_time=$_start_time
_elapsed() {
    local now=$(date +%s); local diff=$((now - _step_time)); _step_time=$now
    printf "%dm%02ds" $((diff/60)) $((diff%60))
}
_total_elapsed() {
    local now=$(date +%s); local diff=$((now - _start_time))
    printf "%dm%02ds" $((diff/60)) $((diff%60))
}

# ---- Banner ---------------------------------------------------------------
TOTAL_TASKS=$(( ${#METHODS[@]} * ${#EVAL_SPLITS[@]} ))
PARALLEL_API=$((NUM_WORKERS * TOTAL_TASKS))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║    Ablation: Raw-Profile Reference Judge Scoring (Parallel)  ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Dataset    : ${DATASET}"
echo "║  Mode       : ${MODE}"
echo "║  Methods    : ${METHODS[*]}"
echo "║  Splits     : ${EVAL_SPLITS[*]}"
echo "║  Persona ref: m2_raw_profile (original description)"
echo "║  Workers    : ${NUM_WORKERS}/task × ${TOTAL_TASKS} tasks = up to ${PARALLEL_API} concurrent API calls"
echo "║  Rate-sleep : ${RATE_LIMIT_SLEEP}s"
echo "║  Mode       : $([ "$SEQUENTIAL" = "1" ] && echo 'SERIAL (debug)' || echo 'PARALLEL')"
echo "║  Main dir   : ${MAIN_DIR}"
echo "║  Output dir : ${ABLATION_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ---- Pre-flight: Validate main results and persona data ------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PRE-FLIGHT ▸ Validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
MISSING=0
for METHOD in "${METHODS[@]}"; do
    for SPLIT in "${EVAL_SPLITS[@]}"; do
        PRED_FILE="${MAIN_DIR}/${METHOD}/${SPLIT}/predictions.jsonl"
        if [ -f "$PRED_FILE" ]; then
            COUNT=$(wc -l < "$PRED_FILE")
            printf "  ✓ %-32s %-12s  %s predictions\n" "$METHOD" "$SPLIT" "$COUNT"
        else
            printf "  ✗ %-32s %-12s  predictions NOT FOUND — run main eval first\n" "$METHOD" "$SPLIT"
            MISSING=$((MISSING + 1))
        fi
    done
done
if [ ! -f "$PERSONA_DATA" ]; then
    echo "  ✗ Persona ref: ${PERSONA_DATA} NOT FOUND"
    MISSING=$((MISSING + 1))
else
    PERSONA_COUNT=$($PYTHON -c "import json; print(len(json.load(open('${PERSONA_DATA}'))))" 2>/dev/null || echo "?")
    echo "  ✓ Persona ref: ${PERSONA_DATA} (${PERSONA_COUNT} samples)"
fi
if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "  ✗ ${MISSING} required file(s) missing. Cannot proceed."
    exit 1
fi
echo ""

# ---- Step 1: Set up ablation directories and symlink shared artifacts ----
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1/3 ▸ Symlink shared artifacts from main"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
_step_time=$(date +%s)

for METHOD in "${METHODS[@]}"; do
    for SPLIT in "${EVAL_SPLITS[@]}"; do
        SRC_DIR="$(cd "$ROOT_DIR" && realpath "${MAIN_DIR}/${METHOD}/${SPLIT}")"
        DST_DIR="${ABLATION_DIR}/${METHOD}/${SPLIT}"
        mkdir -p "$DST_DIR"

        for SHARED_FILE in predictions.jsonl embedding_scores.jsonl; do
            SRC_FILE="${SRC_DIR}/${SHARED_FILE}"
            DST_FILE="${DST_DIR}/${SHARED_FILE}"
            if [ -f "$SRC_FILE" ] && [ ! -e "$DST_FILE" ]; then
                ln -s "$SRC_FILE" "$DST_FILE"
            fi
        done
    done
done
echo "  ✓ Symlinks ready [$(_elapsed)]"

# ---- Step 2: Re-run LLM judge (PARALLEL across tasks) --------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2/3 ▸ LLM Judge (raw-profile reference)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
_step_time=$(date +%s)

JUDGE_PIDS=()
JUDGE_LABELS=()
for METHOD in "${METHODS[@]}"; do
    for SPLIT in "${EVAL_SPLITS[@]}"; do
        DST_DIR="${ABLATION_DIR}/${METHOD}/${SPLIT}"
        if [ ! -e "${DST_DIR}/predictions.jsonl" ]; then
            echo "  ⚠ ${METHOD}/${SPLIT}: predictions.jsonl missing, skipping"
            continue
        fi
        LOG_FILE="${LOG_DIR}/judge_${METHOD}_${SPLIT}.log"
        echo "  ${METHOD}/${SPLIT}  →  ${LOG_FILE}"

        if [ "$SEQUENTIAL" = "1" ]; then
            $PYTHON "${SCRIPT_DIR}/judge.py" \
                --predictions_dir "$DST_DIR" \
                --persona_data "$PERSONA_DATA" \
                --num_workers "$NUM_WORKERS" \
                --rate_limit_sleep "$RATE_LIMIT_SLEEP" \
                > "$LOG_FILE" 2>&1 \
                || echo "  ! ${METHOD}/${SPLIT} failed (see log)"
        else
            $PYTHON "${SCRIPT_DIR}/judge.py" \
                --predictions_dir "$DST_DIR" \
                --persona_data "$PERSONA_DATA" \
                --num_workers "$NUM_WORKERS" \
                --rate_limit_sleep "$RATE_LIMIT_SLEEP" \
                > "$LOG_FILE" 2>&1 &
            JUDGE_PIDS+=($!)
            JUDGE_LABELS+=("${METHOD}/${SPLIT}")
        fi
    done
done

if [ "$SEQUENTIAL" != "1" ]; then
    echo ""
    echo "  Waiting for ${#JUDGE_PIDS[@]} judge tasks ..."
    JUDGE_FAIL=0
    for i in "${!JUDGE_PIDS[@]}"; do
        pid="${JUDGE_PIDS[$i]}"
        label="${JUDGE_LABELS[$i]}"
        if wait "$pid"; then
            echo "    ✓ ${label}"
        else
            echo "    ✗ ${label} (see ${LOG_DIR}/judge_${label//\//_}.log)"
            JUDGE_FAIL=$((JUDGE_FAIL + 1))
        fi
    done
    echo "  ✓ Judge complete [$(_elapsed)] (${JUDGE_FAIL} failures)"
else
    echo "  ✓ Judge complete [$(_elapsed)]"
fi

# ---- Step 3: Generate ablation report ------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 3/3 ▸ Ablation Report"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
_step_time=$(date +%s)

SCORED_METHODS=()
for METHOD in "${METHODS[@]}"; do
    HAS_SCORES=false
    for SPLIT in "${EVAL_SPLITS[@]}"; do
        if [ -f "${ABLATION_DIR}/${METHOD}/${SPLIT}/judge_scores.jsonl" ]; then
            HAS_SCORES=true
            break
        fi
    done
    if [ "$HAS_SCORES" = true ]; then
        SCORED_METHODS+=("$METHOD")
    fi
done

if [ ${#SCORED_METHODS[@]} -gt 0 ]; then
    $PYTHON "${SCRIPT_DIR}/report.py" \
        --results_dir "$ABLATION_DIR" \
        --experiments "${SCORED_METHODS[@]}" \
        --splits "${EVAL_SPLITS[@]}" \
        --baseline m2_raw_profile \
        --per_character
    echo ""
    echo "  ✓ Report generated [$(_elapsed)]"

    # Visualization is best-effort
    if [ -f "${ABLATION_DIR}/summary.json" ]; then
        $PYTHON "${SCRIPT_DIR}/visualize.py" --results_dir "$ABLATION_DIR" --format pdf \
            || echo "  ⚠ visualize.py failed (non-fatal)"
    fi
else
    echo "  ⚠ No scored experiments found, skipping report."
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✓ Ablation complete!"
echo "║  Total elapsed: $(_total_elapsed)"
echo "║  Persona ref  : m2_raw_profile (original desc)"
echo "║  Outputs:"
echo "║    ${ABLATION_DIR}/summary.json"
echo "║    ${ABLATION_DIR}/report.md"
echo "║    ${ABLATION_DIR}/tables.tex"
echo "║    ${ABLATION_DIR}/figures/"
echo "╚══════════════════════════════════════════════════════════════╝"
