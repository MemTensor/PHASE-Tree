#!/usr/bin/env bash
# ===========================================================================
# PHASE-Tree Prompt Predictions — Qwen3-0.6B backbone (multi-dataset)
# ===========================================================================
# Reuses evaluation/predict_prompt.py (--multi) to run the same explicit
# textual-provision methods as results/*/prompt/main, but with Qwen3-0.6B.
#
# Outputs go to a separate tree so the original Qwen2.5-7B results are kept:
#   results/<DATASET>/prompt/qwen3_0_6b/<METHOD>/<SPLIT>/predictions.jsonl
#
# Default: all 8 LongEvoRoleBench datasets, methods matching prompt/main,
# GPUs 0-7 (one vLLM worker per GPU, tensor_parallel=1).
#
# Usage:
#   bash evaluation/run_prompt_predict_qwen3_0_6b.sh
#   DATASETS="CharacterEval RAIDEN" bash evaluation/run_prompt_predict_qwen3_0_6b.sh
#   NUM_SAMPLES=8 bash evaluation/run_prompt_predict_qwen3_0_6b.sh   # smoke test
#
# Environment knobs (all optional):
#   DATASETS              space-separated (default: all 8)
#   MODEL                 default /mnt/lstore/model/Qwen3-0.6B
#   RESULTS_TAG           default qwen3_0_6b  → results/<DS>/prompt/<TAG>/
#   GPU_IDS               default "0 1 2 3 4 5 6 7"
#   NUM_GPU_WORKERS       default = #GPU_IDS
#   TENSOR_PARALLEL       default 1
#   MAX_TOKENS            default 256
#   MAX_MODEL_LEN         default 16384
#   GPU_MEMORY_UTILIZATION default 0.50
#   BATCH_SIZE            default 64 (0.6B model; vLLM batches internally)
#   ENABLE_THINKING       =1 to keep Qwen3 thinking (default: off)
#   EVAL_SPLITS           default "random_test ood_test"
#   METHODS               override per-dataset method list
#   NUM_SAMPLES           debug: only first N samples per task
# ===========================================================================
set -uo pipefail
export PYTHONUNBUFFERED=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$ROOT_DIR"

# --- Python interpreter (phase env) ---------------------------------------
if [ -x "/dev/shm/phase/.venv/bin/python" ]; then
    PYTHON="/dev/shm/phase/.venv/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ] \
        && [[ "${CONDA_PREFIX}" == *phase* ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
else
    echo "ERROR: phase env not found at /dev/shm/phase/.venv/bin/python" >&2
    echo "       Recreate it or set PYTHON=/path/to/phase/python" >&2
    exit 2
fi
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
# Ensure venv binaries (ninja, etc.) are visible to vLLM/flashinfer JIT builds.
PYTHON_BIN_DIR="$(cd "$(dirname "$PYTHON")" && pwd)"
export PATH="${PYTHON_BIN_DIR}:/usr/local/cuda/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# System /usr/bin/nvcc is too old for sm_90a JIT; prefer CUDA 13 toolkit.
# Also skip FlashInfer sampler JIT (avoids nvcc arch mismatch on this host).
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

MODEL="${MODEL:-/mnt/lstore/model/Qwen3-0.6B}"
RESULTS_TAG="${RESULTS_TAG:-qwen3_0_6b}"
BACKEND="${BACKEND:-vllm}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
SPLITS_STR="${EVAL_SPLITS:-random_test ood_test}"
IFS=' ' read -ra SPLITS <<< "$SPLITS_STR"

GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
IFS=' ' read -ra GPU_ID_ARR <<< "$GPU_IDS"
NUM_GPU_WORKERS="${NUM_GPU_WORKERS:-${#GPU_ID_ARR[@]}}"

DEFAULT_DATASETS="RAIDEN CharacterEval SimsConv ChatHaruhi Friends HPD StarTrek_TNG TheOffice"
IFS=' ' read -ra DATASET_LIST <<< "${DATASETS:-$DEFAULT_DATASETS}"

if [ ! -d "$MODEL" ]; then
    echo "ERROR: model directory not found: $MODEL" >&2
    exit 2
fi

GLOBAL_TASKS_DIR="results/_prompt_${RESULTS_TAG}_tasks"
GLOBAL_LOG_DIR="results/_prompt_${RESULTS_TAG}_logs"
mkdir -p "$GLOBAL_TASKS_DIR" "$GLOBAL_LOG_DIR"

START_TIME=$(date +%s)
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Prompt Predictions — ${RESULTS_TAG}"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model   : ${MODEL}"
echo "║  Python  : ${PYTHON}"
echo "║  Datasets: ${DATASET_LIST[*]}"
echo "║  Splits  : ${SPLITS[*]}"
echo "║  GPUs    : ${GPU_ID_ARR[*]} (${NUM_GPU_WORKERS} workers, tp=${TENSOR_PARALLEL})"
echo "║  Mem util: ${GPU_MEMORY_UTILIZATION}"
echo "║  Thinking: $([ "${ENABLE_THINKING:-0}" = "1" ] && echo ON || echo OFF)"
echo "║  Output  : results/<DS>/prompt/${RESULTS_TAG}/"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Build flat task list across all datasets: ds:method:split:prompt_mode
ALL_TASKS=()
for DATASET in "${DATASET_LIST[@]}"; do
    DATA_DIR="LongEvoRoleBench/processed/${DATASET}"
    if [ ! -d "$DATA_DIR" ]; then
        echo "  [SKIP] missing data dir: ${DATA_DIR}"
        continue
    fi

    if [ -n "${METHODS:-}" ]; then
        IFS=' ' read -ra DS_METHODS <<< "$METHODS"
    elif [ -d "${DATA_DIR}/m5_dynamic_tree" ]; then
        DS_METHODS=(m1_context_only m2_raw_profile m3_naive_rewrite m4_static_tree m5_dynamic_tree m6_phase_tree)
    else
        DS_METHODS=(m1_context_only m2_raw_profile m3_naive_rewrite m4_static_tree m6_phase_tree)
    fi

    RESULTS_DIR="results/${DATASET}/prompt/${RESULTS_TAG}"
    mkdir -p "${RESULTS_DIR}/_logs" "${RESULTS_DIR}/_tasks"

    for METHOD in "${DS_METHODS[@]}"; do
        if [ "$METHOD" = "m1_context_only" ]; then
            PROMPT_MODE="baseline"
        else
            PROMPT_MODE="profile"
        fi
        for SPLIT in "${SPLITS[@]}"; do
            DATA_FILE="${DATA_DIR}/${METHOD}/${SPLIT}.json"
            if [ -f "$DATA_FILE" ]; then
                ALL_TASKS+=("${DATASET}:${METHOD}:${SPLIT}:${PROMPT_MODE}")
            else
                echo "  [SKIP] ${DATA_FILE} not found"
            fi
        done
    done
done

N_TASKS=${#ALL_TASKS[@]}
if [ "$N_TASKS" -eq 0 ]; then
    echo "ERROR: no tasks found" >&2
    exit 2
fi
echo "  Total tasks: ${N_TASKS}, distributing across ${NUM_GPU_WORKERS} GPUs"
echo ""

# Round-robin across logical GPU workers
declare -A GPU_TASKS
for i in "${!ALL_TASKS[@]}"; do
    gpu_slot=$((i % NUM_GPU_WORKERS))
    GPU_TASKS[$gpu_slot]+="${ALL_TASKS[$i]};"
done

PREDICT_PIDS=()
PREDICT_DESCS=()
THINKING_FLAG=()
if [ "${ENABLE_THINKING:-0}" = "1" ]; then
    THINKING_FLAG=(--enable_thinking)
fi
NUM_SAMPLES_FLAG=()
if [ -n "${NUM_SAMPLES:-}" ]; then
    NUM_SAMPLES_FLAG=(--num_samples "$NUM_SAMPLES")
fi

for gpu_slot in $(seq 0 $((NUM_GPU_WORKERS - 1))); do
    task_str="${GPU_TASKS[$gpu_slot]:-}"
    if [ -z "$task_str" ]; then
        continue
    fi

    PHYSICAL_GPU="${GPU_ID_ARR[$gpu_slot]:-$gpu_slot}"
    TASK_FILE="${GLOBAL_TASKS_DIR}/tasks_gpu${PHYSICAL_GPU}.json"
    ENTRIES="["
    FIRST=true
    n_gpu_tasks=0

    IFS=';' read -ra TASK_ITEMS <<< "$task_str"
    for item in "${TASK_ITEMS[@]}"; do
        [ -z "$item" ] && continue
        IFS=':' read -r dataset method split prompt_mode <<< "$item"
        data_file="LongEvoRoleBench/processed/${dataset}/${method}/${split}.json"
        out_dir="results/${dataset}/prompt/${RESULTS_TAG}/${method}/${split}"

        if [ "$FIRST" = true ]; then
            FIRST=false
        else
            ENTRIES+=","
        fi
        ENTRIES+="
    {\"data\": \"${data_file}\", \"output_dir\": \"${out_dir}\", \"prompt_mode\": \"${prompt_mode}\"}"
        n_gpu_tasks=$((n_gpu_tasks + 1))
    done
    ENTRIES+="
]"
    echo "$ENTRIES" > "$TASK_FILE"

    LOG_FILE="${GLOBAL_LOG_DIR}/predict_gpu${PHYSICAL_GPU}.log"
    echo "  [GPU ${PHYSICAL_GPU}] ${n_gpu_tasks} tasks -> ${TASK_FILE}"
    echo "             log: ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" $PYTHON evaluation/predict_prompt.py \
        --multi \
        --tasks "$TASK_FILE" \
        --model "$MODEL" \
        --backend "$BACKEND" \
        --batch_size "$BATCH_SIZE" \
        --max_tokens "$MAX_TOKENS" \
        --max_model_len "$MAX_MODEL_LEN" \
        --tensor_parallel "$TENSOR_PARALLEL" \
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
        "${THINKING_FLAG[@]}" \
        "${NUM_SAMPLES_FLAG[@]}" \
        > "$LOG_FILE" 2>&1 &

    PREDICT_PIDS+=($!)
    PREDICT_DESCS+=("GPU ${PHYSICAL_GPU} (${n_gpu_tasks} tasks)")
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
        echo "  FAIL (exit $ec): ${PREDICT_DESCS[$i]}  (see ${GLOBAL_LOG_DIR}/)"
        PREDICT_FAIL=$((PREDICT_FAIL + 1))
    fi
done

echo ""
echo "  Prediction counts:"
shopt -s nullglob
for DATASET in "${DATASET_LIST[@]}"; do
    RESULTS_DIR="results/${DATASET}/prompt/${RESULTS_TAG}"
    [ -d "$RESULTS_DIR" ] || continue
    for METHOD_DIR in "${RESULTS_DIR}"/m*/; do
        METHOD="$(basename "$METHOD_DIR")"
        for SPLIT in "${SPLITS[@]}"; do
            pred="${RESULTS_DIR}/${METHOD}/${SPLIT}/predictions.jsonl"
            if [ -f "$pred" ]; then
                n=$(wc -l < "$pred")
                echo "    ${DATASET}/${METHOD}/${SPLIT}: ${n}"
            fi
        done
    done
done
shopt -u nullglob

END_TIME=$(date +%s)
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Predictions complete! [$((END_TIME - START_TIME))s]"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Failures : ${PREDICT_FAIL}"
echo "║  Results  : results/<DS>/prompt/${RESULTS_TAG}/"
echo "║  Logs     : ${GLOBAL_LOG_DIR}/"
echo "╚══════════════════════════════════════════════════════════════╝"

exit "$PREDICT_FAIL"
