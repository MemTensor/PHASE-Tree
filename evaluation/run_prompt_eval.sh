#!/usr/bin/env bash
# ===========================================================================
# PHASE-Tree Prompt Evaluation Pipeline (dataset-agnostic, multi-GPU)
# ===========================================================================
# Architecture:
#   - vLLM loaded ONCE per GPU (predict stage runs ${NUM_GPU_WORKERS} parallel workers)
#   - Each judge/embedding task runs in its own subprocess with ${NUM_WORKERS}
#     concurrent API workers (total API parallelism ≈ NUM_TASKS × NUM_WORKERS)
#
# Two dataset modes:
#   * short-term  (default): m1..m4 + m6, persona ref = m6_phase_tree/all_dialogues.json
#                            datasets: RAIDEN, CharacterEval, SimsConv, ChatHaruhi
#   * long-term            : m1..m6, persona ref = m6_phase_tree/all_dialogues.json
#                            datasets: Friends, HPD, StarTrek_TNG, TheOffice
#
# Steps: predict → judge → report → visualize
#
# Usage:
#   bash evaluation/run_prompt_eval.sh                     # default: RAIDEN, short-term
#   bash evaluation/run_prompt_eval.sh RAIDEN              # short-term auto-detected
#   bash evaluation/run_prompt_eval.sh Friends             # long-term auto-detected
#   bash evaluation/run_prompt_eval.sh Friends long-term   # explicit mode
#   MODE=long-term DATASET=Friends bash evaluation/run_prompt_eval.sh
#
# Environment knobs (all optional):
#   DATASET             (default $1 or RAIDEN)
#   MODE                (short-term | long-term; auto-detect if unset)
#   METHODS             (space-separated; override the default set for MODE)
#   PERSONA_DATA        (override persona ground-truth path)
#   BASELINE_METHOD     (default m2_raw_profile)
#   NUM_GPU_WORKERS     (default = visible GPU count, capped at 8)
#   BATCH_SIZE          (default 16)
#   NUM_WORKERS         (judge/embedding workers per task; short=10, long=32)
#   RATE_LIMIT_SLEEP    (per-call sleep in judge.py; short=0.1, long=0.0)
#   BACKEND             (default vllm)
#   MODEL               (default /dev/shm/Qwen2.5-7B-Instruct or models/Qwen2.5-7B-Instruct)
#   MAX_TOKENS          (default 256)
# ===========================================================================
set -uo pipefail
export PYTHONUNBUFFERED=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$ROOT_DIR"

# --- Python interpreter ---------------------------------------------------
if [ -x "/dev/shm/phase/.venv/bin/python" ]; then
    PYTHON="/dev/shm/phase/.venv/bin/python"
else
    PYTHON="$(command -v python)"
fi
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# --- Positional + env-var resolution --------------------------------------
DATASET="${1:-${DATASET:-RAIDEN}}"
MODE_ARG="${2:-${MODE:-}}"
shift $(( $# < 2 ? $# : 2 ))

# Auto-detect MODE from dataset directory layout if not specified.
DATA_DIR="LongEvoRoleBench/processed/${DATASET}"
if [ -z "$MODE_ARG" ]; then
    if [ -d "${DATA_DIR}/m5_dynamic_tree" ]; then
        MODE_ARG="long-term"
    elif [ -d "${DATA_DIR}/m6_phase_tree" ]; then
        MODE_ARG="short-term"
    else
        echo "ERROR: cannot auto-detect MODE for ${DATASET}. No m5_dynamic_tree or m6_phase_tree dir found under ${DATA_DIR}/" >&2
        exit 2
    fi
fi

case "$MODE_ARG" in
    short|short-term|shortterm)
        MODE="short-term"
        DEFAULT_METHODS="m1_context_only m2_raw_profile m3_naive_rewrite m4_static_tree m6_phase_tree"
        DEFAULT_PERSONA="${DATA_DIR}/m6_phase_tree/all_dialogues.json"
        DEFAULT_NUM_WORKERS=10
        DEFAULT_EMBED_WORKERS=10
        DEFAULT_RATE_SLEEP="0.1"
        ;;
    long|long-term|longterm)
        MODE="long-term"
        DEFAULT_METHODS="m1_context_only m2_raw_profile m3_naive_rewrite m4_static_tree m5_dynamic_tree m6_phase_tree"
        DEFAULT_PERSONA="${DATA_DIR}/m6_phase_tree/all_dialogues.json"
        # Long-term datasets (Friends: ~16K samples per split) — push the
        # judge/embed parallelism much higher.  judge=128 keeps prompt-token
        # throughput within typical LLM endpoint limits; embed=32 is plenty
        # because each call is a 64-text batch.
        DEFAULT_NUM_WORKERS=128
        DEFAULT_EMBED_WORKERS=32
        DEFAULT_RATE_SLEEP="0.0"
        ;;
    *)
        echo "ERROR: unknown MODE '$MODE_ARG' (expected: short-term | long-term)" >&2
        exit 2
        ;;
esac

# --- Resolve env-var overrides --------------------------------------------
IFS=' ' read -ra METHODS <<< "${METHODS:-$DEFAULT_METHODS}"
SPLITS_STR="${EVAL_SPLITS:-random_test ood_test}"
IFS=' ' read -ra SPLITS <<< "$SPLITS_STR"
PERSONA_DATA="${PERSONA_DATA:-$DEFAULT_PERSONA}"
BASELINE_METHOD="${BASELINE_METHOD:-m2_raw_profile}"

RESULTS_DIR="results/${DATASET}/prompt/main"

if [ -d "/dev/shm/Qwen2.5-7B-Instruct" ]; then
    DEFAULT_MODEL="/dev/shm/Qwen2.5-7B-Instruct"
else
    DEFAULT_MODEL="models/Qwen2.5-7B-Instruct"
fi
MODEL="${MODEL:-$DEFAULT_MODEL}"

MAX_TOKENS="${MAX_TOKENS:-256}"
BACKEND="${BACKEND:-vllm}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-$DEFAULT_NUM_WORKERS}"
# Fall back to NUM_WORKERS if DEFAULT_EMBED_WORKERS is unset (e.g. older
# script revision that was already parsed by a running bash instance).
EMBED_WORKERS="${EMBED_WORKERS:-${DEFAULT_EMBED_WORKERS:-$NUM_WORKERS}}"
RATE_LIMIT_SLEEP="${RATE_LIMIT_SLEEP:-$DEFAULT_RATE_SLEEP}"
CONCURRENT_PASSES="${CONCURRENT_PASSES:-1}"

# Discover visible GPU count; cap GPU-worker count at 8 (script was written
# for ≤8-GPU hosts; larger hosts can override NUM_GPU_WORKERS).
DETECTED_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
if [ "$DETECTED_GPUS" -lt 1 ]; then DETECTED_GPUS=1; fi
DEFAULT_GPU_WORKERS=$(( DETECTED_GPUS > 8 ? 8 : DETECTED_GPUS ))
NUM_GPU_WORKERS="${NUM_GPU_WORKERS:-$DEFAULT_GPU_WORKERS}"

LOG_DIR="${RESULTS_DIR}/_logs"
TASKS_DIR="${RESULTS_DIR}/_tasks"
mkdir -p "$LOG_DIR" "$TASKS_DIR"

START_TIME=$(date +%s)
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE-Tree Prompt Evaluation — ${DATASET} (${MODE})"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Methods : ${METHODS[*]}"
echo "║  Splits  : ${SPLITS[*]}"
echo "║  Persona : ${PERSONA_DATA}"
echo "║  Baseline: ${BASELINE_METHOD}"
echo "║  Backend : ${BACKEND} (batch=${BATCH_SIZE})"
echo "║  Model   : ${MODEL}"
echo "║  GPUs    : ${NUM_GPU_WORKERS} workers (model loaded ONCE per GPU)"
echo "║  API par : judge=${NUM_WORKERS} workers/task, embed=${EMBED_WORKERS} workers/task"
echo "║          ${#METHODS[@]} methods × ${#SPLITS[@]} splits × (judge ∥ embed) = up to $(( ${NUM_WORKERS} * ${#METHODS[@]} * ${#SPLITS[@]} + ${EMBED_WORKERS} * ${#METHODS[@]} * ${#SPLITS[@]} )) concurrent API calls"
echo "║  Rate slp: ${RATE_LIMIT_SLEEP}s/call ; concurrent_passes=${CONCURRENT_PASSES}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# =====================================================================
# STEP 1: Predictions (multi-task per GPU)
# =====================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1/4 ▸ Predictions (${NUM_GPU_WORKERS} GPU workers)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP1_START=$(date +%s)

# Build flat task list: method:split:prompt_mode
ALL_TASKS=()
for METHOD in "${METHODS[@]}"; do
    if [ "$METHOD" = "m1_context_only" ]; then
        PROMPT_MODE="baseline"
    else
        PROMPT_MODE="profile"
    fi
    for SPLIT in "${SPLITS[@]}"; do
        DATA_FILE="${DATA_DIR}/${METHOD}/${SPLIT}.json"
        if [ -f "$DATA_FILE" ]; then
            ALL_TASKS+=("${METHOD}:${SPLIT}:${PROMPT_MODE}")
        else
            echo "  [SKIP] ${DATA_FILE} not found"
        fi
    done
done

N_TASKS=${#ALL_TASKS[@]}
echo "  Total tasks: ${N_TASKS}, distributing across ${NUM_GPU_WORKERS} GPUs"

# Distribute tasks round-robin across GPUs
declare -A GPU_TASKS
for i in "${!ALL_TASKS[@]}"; do
    gpu_id=$((i % NUM_GPU_WORKERS))
    GPU_TASKS[$gpu_id]+="${ALL_TASKS[$i]};"
done

# Launch one worker per GPU
PREDICT_PIDS=()
PREDICT_DESCS=()

for gpu_id in $(seq 0 $((NUM_GPU_WORKERS - 1))); do
    task_str="${GPU_TASKS[$gpu_id]:-}"
    if [ -z "$task_str" ]; then
        continue
    fi

    # Build task JSON file for this GPU
    TASK_FILE="${TASKS_DIR}/tasks_gpu${gpu_id}.json"
    ENTRIES="["
    FIRST=true

    IFS=';' read -ra TASK_ITEMS <<< "$task_str"
    for item in "${TASK_ITEMS[@]}"; do
        [ -z "$item" ] && continue
        IFS=':' read -r method split prompt_mode <<< "$item"
        data_file="${DATA_DIR}/${method}/${split}.json"
        out_dir="${RESULTS_DIR}/${method}/${split}"

        if [ "$FIRST" = true ]; then
            FIRST=false
        else
            ENTRIES+=","
        fi
        ENTRIES+="
    {\"data\": \"${data_file}\", \"output_dir\": \"${out_dir}\", \"prompt_mode\": \"${prompt_mode}\"}"
    done
    ENTRIES+="
]"
    echo "$ENTRIES" > "$TASK_FILE"

    # Count tasks for this GPU
    n_gpu_tasks=$(echo "$task_str" | tr -cd ';' | wc -c)
    LOG_FILE="${LOG_DIR}/predict_gpu${gpu_id}.log"
    echo "  [GPU $gpu_id] ${n_gpu_tasks} tasks -> ${TASK_FILE}"

    # Resolve physical GPU id (allow GPU_IDS override to skip busy GPUs)
    PHYSICAL_GPU="$gpu_id"
    if [ -n "${GPU_IDS:-}" ]; then
        # GPU_IDS is a space-separated list, e.g. "1 2 3 4 5 6 7"
        # gpu_id (0..NUM_GPU_WORKERS-1) is the logical index into that list.
        IFS=' ' read -ra _GPU_ID_ARR <<< "$GPU_IDS"
        if [ -n "${_GPU_ID_ARR[$gpu_id]:-}" ]; then
            PHYSICAL_GPU="${_GPU_ID_ARR[$gpu_id]}"
        fi
    fi

    CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" $PYTHON evaluation/predict_prompt.py \
        --multi \
        --tasks "$TASK_FILE" \
        --model "$MODEL" \
        --backend "$BACKEND" \
        --batch_size "$BATCH_SIZE" \
        --max_tokens "$MAX_TOKENS" \
        --tensor_parallel 1 \
        > "$LOG_FILE" 2>&1 &

    PREDICT_PIDS+=($!)
    PREDICT_DESCS+=("GPU $gpu_id (${n_gpu_tasks} tasks)")
done

echo ""
echo "  Waiting for ${#PREDICT_PIDS[@]} GPU workers ..."
PREDICT_FAIL=0
for i in "${!PREDICT_PIDS[@]}"; do
    wait "${PREDICT_PIDS[$i]}" 2>/dev/null
    ec=$?
    if [ $ec -eq 0 ]; then
        echo "  OK: ${PREDICT_DESCS[$i]}"
    else
        echo "  FAIL (exit $ec): ${PREDICT_DESCS[$i]}"
        PREDICT_FAIL=$((PREDICT_FAIL + 1))
    fi
done
STEP1_END=$(date +%s)
echo "  Predictions complete [$((STEP1_END - STEP1_START))s] (${PREDICT_FAIL} failures)"
echo ""

for METHOD in "${METHODS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        pred="${RESULTS_DIR}/${METHOD}/${SPLIT}/predictions.jsonl"
        if [ -f "$pred" ]; then
            n=$(wc -l < "$pred")
            echo "    ${METHOD}/${SPLIT}: ${n} predictions"
        fi
    done
done

# =====================================================================
# STEP 2: Scoring (judge tasks in parallel, API-based)
# =====================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2/4 ▸ Scoring (parallel, API-based, judge=${NUM_WORKERS}, embed=${EMBED_WORKERS} workers/task, passes ∥)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP2_START=$(date +%s)

JUDGE_PIDS=()
for METHOD in "${METHODS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        OUT_DIR="${RESULTS_DIR}/${METHOD}/${SPLIT}"
        if [ ! -f "${OUT_DIR}/predictions.jsonl" ]; then
            echo "  [SKIP] ${OUT_DIR}/predictions.jsonl not found"
            continue
        fi
        LOG_FILE="${LOG_DIR}/judge_${METHOD}_${SPLIT}.log"
        echo "  ${METHOD}/${SPLIT}"
        JUDGE_ARGS=(
            --predictions_dir "$OUT_DIR"
            --persona_data "$PERSONA_DATA"
            --num_workers "$NUM_WORKERS"
            --rate_limit_sleep "$RATE_LIMIT_SLEEP"
        )
        # Only pass --embed_workers when the variable resolved to a value
        # (defensive: protects against bash running an older cached
        # variable-resolution block that never set EMBED_WORKERS).
        if [ -n "${EMBED_WORKERS:-}" ]; then
            JUDGE_ARGS+=(--embed_workers "$EMBED_WORKERS")
        fi
        if [ "${CONCURRENT_PASSES:-1}" = "0" ] || [ "${CONCURRENT_PASSES:-1}" = "false" ]; then
            JUDGE_ARGS+=(--sequential_passes)
        fi
        $PYTHON evaluation/judge.py "${JUDGE_ARGS[@]}" > "$LOG_FILE" 2>&1 &
        JUDGE_PIDS+=($!)
    done
done

echo ""
echo "  Waiting for all judge tasks..."
JUDGE_FAIL=0
for pid in "${JUDGE_PIDS[@]}"; do
    wait "$pid" || JUDGE_FAIL=$((JUDGE_FAIL + 1))
done
STEP2_END=$(date +%s)
echo "  Scoring complete [$((STEP2_END - STEP2_START))s] (${JUDGE_FAIL} failures)"

# =====================================================================
# STEP 3: Report
# =====================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 3/4 ▸ Report Generation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

SCORED=()
for METHOD in "${METHODS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        if [ -f "${RESULTS_DIR}/${METHOD}/${SPLIT}/judge_scores.jsonl" ]; then
            SCORED+=("$METHOD")
            break
        fi
    done
done

if [ ${#SCORED[@]} -gt 0 ]; then
    $PYTHON evaluation/report.py \
        --results_dir "$RESULTS_DIR" \
        --experiments "${SCORED[@]}" \
        --splits "${SPLITS[@]}" \
        --baseline "$BASELINE_METHOD" \
        --per_character
    echo "  Report generated"
else
    echo "  No scored experiments found, skipping."
fi

# =====================================================================
# STEP 4: Visualization
# =====================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 4/4 ▸ Visualization"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "${RESULTS_DIR}/summary.json" ]; then
    $PYTHON evaluation/visualize.py \
        --results_dir "$RESULTS_DIR" \
        --format pdf
    echo "  Figures generated"
else
    echo "  summary.json not found, skipping."
fi

END_TIME=$(date +%s)
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Pipeline complete! [$((END_TIME - START_TIME))s]"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Dataset     : ${DATASET} (${MODE})"
echo "║  Predictions : ${RESULTS_DIR}/"
echo "║  Logs        : ${LOG_DIR}/"
echo "╚══════════════════════════════════════════════════════════════╝"
