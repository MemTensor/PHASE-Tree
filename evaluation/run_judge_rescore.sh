#!/usr/bin/env bash
# ===========================================================================
# Batch LLM-as-Judge rescoring with an alternate judge model.
# ===========================================================================
# Scores all predictions under results/<DATASET>/prompt/main while keeping
# existing GPT judge results in judge_scores.jsonl untouched.  Non-gpt-4.1
# models write to model-specific files, e.g.:
#   judge_scores_glm-5.2.jsonl
#
# Usage:
#   bash evaluation/run_judge_rescore.sh CharacterEval
#   DATASET=CharacterEval JUDGE_MODEL=glm-5.2 NUM_WORKERS=128 \
#       bash evaluation/run_judge_rescore.sh
#
# Environment knobs:
#   DATASET          (default: CharacterEval)
#   JUDGE_MODEL      (default: glm-5.2)
#   NUM_WORKERS      (default: 128)
#   RATE_LIMIT_SLEEP (default: 0.0)
#   TRACK            (prompt | comparison | hypernet_p2p | phase_tree; default: prompt)
#   RESULTS_TAG      (default: main; use qwen3_32b / qwen3_0_6b / gemma_4_e4b_it for backbones)
#   SAMPLE_MANIFEST_DIR (unset: gpt-4.1→25% subsample, else full; empty=full)
#   MAX_RETRIES      (default: 10)
#   METHODS          (space-separated; default: auto-discover from results dir)
#   EVAL_SPLITS      (default: random_test ood_test)
#   PERSONA_DATA     (override persona ground-truth path)
#   PARALLEL_TASKS   (default: 1 — run one method/split at a time; set higher
#                     to launch multiple judge subprocesses in parallel)
#   ALLOW_INCOMPLETE (=1 to skip stubborn splits and continue; logs leftovers)
#   INCOMPLETE_MIN_PCT (default: 99 — treat split done if score >= this %)
#   SPLIT_MAX_ROUNDS (per-split retry rounds; default: MAX_ROUNDS)
#   WAIT_PID         (optional: wait for this PID before starting)
# ===========================================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$ROOT_DIR"

JUDGE_VENV="${ROOT_DIR}/.venv-judge"
if [ -x "${JUDGE_VENV}/bin/python" ]; then
    PYTHON="${JUDGE_VENV}/bin/python"
elif [ -x "/dev/shm/phase/.venv/bin/python" ]; then
    PYTHON="/dev/shm/phase/.venv/bin/python"
elif [ -x "${HOME}/miniconda3/envs/phase/bin/python" ]; then
    PYTHON="${HOME}/miniconda3/envs/phase/bin/python"
elif [ -x "${HOME}/anaconda3/envs/phase/bin/python" ]; then
    PYTHON="${HOME}/anaconda3/envs/phase/bin/python"
else
    PYTHON="$(command -v python3 || command -v python)"
fi

if ! "$PYTHON" -c "import openai" 2>/dev/null; then
    echo "Judge venv missing openai — running setup_judge_venv.sh..."
    bash evaluation/setup_judge_venv.sh
    PYTHON="${JUDGE_VENV}/bin/python"
fi
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

DATASET="${1:-${DATASET:-CharacterEval}}"
TRACK="${TRACK:-prompt}"
RESULTS_TAG="${RESULTS_TAG:-main}"
JUDGE_MODEL="${JUDGE_MODEL:-glm-5.2}"
NUM_WORKERS="${NUM_WORKERS:-128}"
RATE_LIMIT_SLEEP="${RATE_LIMIT_SLEEP:-0.0}"
PARALLEL_TASKS="${PARALLEL_TASKS:-1}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
WAIT_PID="${WAIT_PID:-}"
MAX_RETRIES="${MAX_RETRIES:-10}"
ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE:-0}"
INCOMPLETE_MIN_PCT="${INCOMPLETE_MIN_PCT:-99}"
if [ -z "${SAMPLE_MANIFEST_DIR+x}" ]; then
    if [ "$JUDGE_MODEL" = "gpt-4.1" ]; then
        SAMPLE_MANIFEST_DIR="results/_judge_samples_25pct"
    else
        SAMPLE_MANIFEST_DIR=""
    fi
fi

if [ -n "$WAIT_PID" ]; then
    echo "Waiting for PID ${WAIT_PID} to finish..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 10
    done
    echo "PID ${WAIT_PID} finished."
fi

DATA_DIR="LongEvoRoleBench/processed/${DATASET}"
RESULTS_DIR="results/${DATASET}/${TRACK}/${RESULTS_TAG}"
LOG_DIR="${RESULTS_DIR}/_logs"
mkdir -p "$LOG_DIR"

sample_ids_file_for_split() {
    local split="$1"
    local f="${SAMPLE_MANIFEST_DIR}/${DATASET}/${split}.json"
    if [ -n "${SAMPLE_MANIFEST_DIR:-}" ] && [ -f "$f" ]; then
        echo "$f"
    fi
}

split_is_done() {
    local score_n="$1"
    local expected_n="$2"
    if [ "$score_n" -ge "$expected_n" ]; then
        return 0
    fi
    if [ "$ALLOW_INCOMPLETE" = "1" ] || [ "$ALLOW_INCOMPLETE" = "true" ]; then
        if [ "$expected_n" -gt 0 ] && [ $((score_n * 100)) -ge $((expected_n * INCOMPLETE_MIN_PCT)) ]; then
            return 0
        fi
    fi
    return 1
}

log_incomplete_split() {
    local method="$1"
    local split="$2"
    local score_n="$3"
    local expected_n="$4"
    local manifest="${LOG_DIR}/incomplete_${SAFE_MODEL}.txt"
    echo "${DATASET}/${TRACK}/${RESULTS_TAG}/${method}/${split} ${score_n}/${expected_n}" >> "$manifest"
}

expected_count_for_split() {
    local split="$1"
    local pred_n="$2"
    local sample_file
    sample_file="$(sample_ids_file_for_split "$split")"
    if [ -n "$sample_file" ]; then
        $PYTHON -c "import json; print(len(json.load(open('$sample_file'))))"
    else
        echo "$pred_n"
    fi
}

discover_methods_from_results() {
    local methods=()
    local entry m
    for entry in "${RESULTS_DIR}"/*; do
        [ -d "$entry" ] || continue
        m="$(basename "$entry")"
        [[ "$m" == _* ]] && continue
        if compgen -G "${entry}/*/predictions.jsonl" > /dev/null; then
            methods+=("$m")
        fi
    done
    echo "${methods[@]}"
}

default_prompt_methods() {
    if [ -d "${DATA_DIR}/m5_dynamic_tree" ]; then
        echo "m1_context_only m2_raw_profile m3_naive_rewrite m4_static_tree m5_dynamic_tree m6_phase_tree"
    else
        echo "m1_context_only m2_raw_profile m3_naive_rewrite m4_static_tree m6_phase_tree"
    fi
}

if [ -n "${METHODS:-}" ]; then
    IFS=' ' read -ra METHODS_ARR <<< "$METHODS"
else
    if [ "$TRACK" = "prompt" ]; then
        IFS=' ' read -ra METHODS_ARR <<< "$(default_prompt_methods)"
    elif [ "$TRACK" = "comparison" ] && [ "$RESULTS_TAG" != "main" ]; then
        IFS=' ' read -ra METHODS_ARR <<< "rag pag cfg"
    else
        IFS=' ' read -ra METHODS_ARR <<< "$(discover_methods_from_results)"
    fi
fi
METHODS=("${METHODS_ARR[@]}")
IFS=' ' read -ra SPLITS <<< "${EVAL_SPLITS:-random_test ood_test}"
PERSONA_DATA="${PERSONA_DATA:-${DATA_DIR}/m6_phase_tree/all_dialogues.json}"

SAFE_MODEL="${JUDGE_MODEL//[^a-zA-Z0-9._-]/_}"
if [ "$JUDGE_MODEL" = "gpt-4.1" ]; then
    JUDGE_FILE="judge_scores.jsonl"
else
    JUDGE_FILE="judge_scores_${SAFE_MODEL}.jsonl"
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Judge Rescore — ${DATASET} / ${TRACK} / ${RESULTS_TAG}"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Judge model : ${JUDGE_MODEL}"
echo "║  Output file : ${JUDGE_FILE}"
echo "║  Track       : ${TRACK}"
echo "║  Tag         : ${RESULTS_TAG}"
echo "║  Methods     : ${METHODS[*]}"
echo "║  Thinking    : $([ "${DISABLE_THINKING}" = "1" ] && echo OFF || echo ON)"
echo "║  Workers     : ${NUM_WORKERS} per task"
echo "║  Parallel    : ${PARALLEL_TASKS} tasks"
echo "║  Persona ref : ${PERSONA_DATA}"
if [ -n "${SAMPLE_MANIFEST_DIR:-}" ]; then
    echo "║  Subsample   : 25% fixed (${SAMPLE_MANIFEST_DIR})"
else
    echo "║  Subsample   : full"
fi
echo "╚══════════════════════════════════════════════════════════════╝"

if [ ! -f "$PERSONA_DATA" ]; then
    echo "ERROR: persona data not found: $PERSONA_DATA" >&2
    exit 2
fi

if [ ${#METHODS[@]} -eq 0 ]; then
    echo "ERROR: no methods found under ${RESULTS_DIR}" >&2
    exit 2
fi

TASKS=()
INCOMPLETE_COUNT=0
COMPLETE_COUNT=0
for METHOD in "${METHODS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        OUT_DIR="${RESULTS_DIR}/${METHOD}/${SPLIT}"
        if [ ! -f "${OUT_DIR}/predictions.jsonl" ]; then
            echo "  [SKIP] ${OUT_DIR}/predictions.jsonl not found"
            continue
        fi
        pred_n=$(wc -l < "${OUT_DIR}/predictions.jsonl")
        expected_n="$(expected_count_for_split "$SPLIT" "$pred_n")"
        score_n=0
        if [ -f "${OUT_DIR}/${JUDGE_FILE}" ]; then
            score_n=$(wc -l < "${OUT_DIR}/${JUDGE_FILE}")
        fi
        if split_is_done "$score_n" "$expected_n"; then
            if [ "$score_n" -lt "$expected_n" ]; then
                echo "  [SKIP] ${METHOD}/${SPLIT}: ${score_n}/${expected_n} (≥${INCOMPLETE_MIN_PCT}%, defer remainder)"
                log_incomplete_split "$METHOD" "$SPLIT" "$score_n" "$expected_n"
            else
                echo "  [DONE] ${METHOD}/${SPLIT}: ${score_n}/${expected_n} (pred pool ${pred_n})"
            fi
            COMPLETE_COUNT=$((COMPLETE_COUNT + 1))
            continue
        fi
        echo "  [TODO] ${METHOD}/${SPLIT}: ${score_n}/${expected_n} (pred pool ${pred_n})"
        INCOMPLETE_COUNT=$((INCOMPLETE_COUNT + 1))
        TASKS+=("${METHOD}|${SPLIT}|${OUT_DIR}|${expected_n}")
    done
done

if [ ${#TASKS[@]} -eq 0 ]; then
    echo "All splits already complete for ${DATASET} (${JUDGE_MODEL})."
    exit 0
fi

echo "  Incomplete splits: ${INCOMPLETE_COUNT}, already complete: ${COMPLETE_COUNT}"

run_one_task() {
    local method="$1"
    local split="$2"
    local out_dir="$3"
    local log_file="${LOG_DIR}/judge_${SAFE_MODEL}_${method}_${split}.log"
    local sample_file
    sample_file="$(sample_ids_file_for_split "$split")"
    local sample_args=()
    if [ -n "$sample_file" ]; then
        sample_args=(--sample_ids_file "$sample_file")
    fi
    echo "  ▶ ${method}/${split} → ${JUDGE_FILE}"
    local thinking_args=()
    if [ "${DISABLE_THINKING}" = "1" ] || [ "${DISABLE_THINKING}" = "true" ]; then
        thinking_args+=(--disable_thinking)
    else
        thinking_args+=(--enable_thinking)
    fi
    JUDGE_MODEL="$JUDGE_MODEL" "$PYTHON" evaluation/judge.py \
        --predictions_dir "$out_dir" \
        --persona_data "$PERSONA_DATA" \
        --judge_model "$JUDGE_MODEL" \
        --num_workers "$NUM_WORKERS" \
        --rate_limit_sleep "$RATE_LIMIT_SLEEP" \
        --skip_embedding \
        --sequential_passes \
        "${sample_args[@]}" \
        "${thinking_args[@]}" \
        --max_retries "$MAX_RETRIES" \
        > "$log_file" 2>&1
}

START_TIME=$(date +%s)
FAIL=0
MAX_ROUNDS="${MAX_ROUNDS:-5}"
SPLIT_MAX_ROUNDS="${SPLIT_MAX_ROUNDS:-$MAX_ROUNDS}"
ROUND=1
PENDING=("${TASKS[@]}")

while [ ${#PENDING[@]} -gt 0 ] && [ "$ROUND" -le "$SPLIT_MAX_ROUNDS" ]; do
    if [ "$ROUND" -gt 1 ]; then
        echo ""
        echo "  ⟳ Retry round ${ROUND}/${MAX_ROUNDS} (${#PENDING[@]} incomplete splits)"
    fi
    RUNNING=0
    IDX=0
    NEXT_PENDING=()

    for entry in "${PENDING[@]}"; do
        IFS='|' read -r METHOD SPLIT OUT_DIR expected_n <<< "$entry"
        run_one_task "$METHOD" "$SPLIT" "$OUT_DIR" &
        RUNNING=$((RUNNING + 1))
        IDX=$((IDX + 1))

        if [ "$RUNNING" -ge "$PARALLEL_TASKS" ] || [ "$IDX" -eq "${#PENDING[@]}" ]; then
            for pid in $(jobs -p); do
                wait "$pid" || FAIL=$((FAIL + 1))
            done
            RUNNING=0
        fi
    done

    for entry in "${PENDING[@]}"; do
        IFS='|' read -r METHOD SPLIT OUT_DIR expected_n <<< "$entry"
        out="${OUT_DIR}/${JUDGE_FILE}"
        score_n=0
        [ -f "$out" ] && score_n=$(wc -l < "$out")
        if ! split_is_done "$score_n" "$expected_n"; then
            NEXT_PENDING+=("${METHOD}|${SPLIT}|${OUT_DIR}|${expected_n}")
        elif [ "$score_n" -lt "$expected_n" ]; then
            echo "  [SKIP] ${METHOD}/${SPLIT}: ${score_n}/${expected_n} (≥${INCOMPLETE_MIN_PCT}%, defer remainder)"
            log_incomplete_split "$METHOD" "$SPLIT" "$score_n" "$expected_n"
        fi
    done

    PENDING=("${NEXT_PENDING[@]}")
    ROUND=$((ROUND + 1))
done

if [ ${#PENDING[@]} -gt 0 ]; then
    for entry in "${PENDING[@]}"; do
        IFS='|' read -r METHOD SPLIT OUT_DIR expected_n <<< "$entry"
        out="${OUT_DIR}/${JUDGE_FILE}"
        score_n=0
        [ -f "$out" ] && score_n=$(wc -l < "$out")
        echo "  [DEFER] ${METHOD}/${SPLIT}: ${score_n}/${expected_n}"
        log_incomplete_split "$METHOD" "$SPLIT" "$score_n" "$expected_n"
    done
    if [ "$ALLOW_INCOMPLETE" = "1" ] || [ "$ALLOW_INCOMPLETE" = "true" ]; then
        echo "  ALLOW_INCOMPLETE=1 — continuing despite ${#PENDING[@]} deferred split(s)"
    else
        FAIL=$((FAIL + 1))
    fi
fi

END_TIME=$(date +%s)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Rescore complete [$((END_TIME - START_TIME))s] (${FAIL} failures)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for entry in "${TASKS[@]}"; do
    IFS='|' read -r METHOD SPLIT OUT_DIR expected_n <<< "$entry"
    out="${OUT_DIR}/${JUDGE_FILE}"
    score_n=0
    [ -f "$out" ] && score_n=$(wc -l < "$out")
    if split_is_done "$score_n" "$expected_n"; then
        if [ "$score_n" -lt "$expected_n" ]; then
            echo "    ${METHOD}/${SPLIT}: ${score_n}/${expected_n} SKIP (deferred)"
        else
            echo "    ${METHOD}/${SPLIT}: ${score_n}/${expected_n} ✓"
        fi
    elif [ -f "$out" ]; then
        echo "    ${METHOD}/${SPLIT}: ${score_n}/${expected_n} INCOMPLETE"
        FAIL=$((FAIL + 1))
    else
        echo "    ${METHOD}/${SPLIT}: MISSING ${JUDGE_FILE}"
        FAIL=$((FAIL + 1))
    fi
done

if [ "$FAIL" -gt 0 ]; then
    if [ "$ALLOW_INCOMPLETE" = "1" ] || [ "$ALLOW_INCOMPLETE" = "true" ]; then
        echo "  ALLOW_INCOMPLETE=1 — exiting 0 with ${FAIL} deferred split(s)"
        exit 0
    fi
    exit 1
fi
