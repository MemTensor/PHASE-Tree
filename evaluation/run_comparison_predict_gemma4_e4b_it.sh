#!/usr/bin/env bash
# ===========================================================================
# PHASE-Tree Comparison Baselines — Gemma-4-E4B-it (multi-dataset, 8-GPU)
# ===========================================================================
# Runs rag / pag / cfg predictions for all 8 LongEvoRoleBench datasets with
# Gemma-4-E4B-it, mirroring results/*/comparison/main/{rag,pag,cfg} layout but
# writing to a separate tag so Qwen2.5-7B main results are preserved:
#   results/<DATASET>/comparison/gemma_4_e4b_it/<METHOD>/<SPLIT>/predictions.jsonl
#
# Usage:
#   bash evaluation/run_comparison_predict_gemma4_e4b_it.sh
#   METHODS="rag pag" DATASETS="CharacterEval" bash evaluation/run_comparison_predict_gemma4_e4b_it.sh
#   NUM_SAMPLES=8 bash evaluation/run_comparison_predict_gemma4_e4b_it.sh   # smoke test
#
# Environment knobs (all optional):
#   DATASETS              default: all 8
#   METHODS               default: "rag pag cfg"
#   MODEL                 default /mnt/lstore/model/gemma-4-E4B-it
#   LOCAL_EMBED_MODEL     default /mnt/lstore/model/Qwen3-Embedding-8B
#   RESULTS_TAG           default gemma_4_e4b_it
#   GPU_IDS               default "0 1 2 3 4 5 6 7"
#   TOP_K                 default 3 (matches existing comparison/main)
#   CFG_BATCH_SIZE        default 8 (4B model; larger than 32B)
#   GPU_MEMORY_UTILIZATION default 0.85
#   SKIP_JUDGE            default 1 (predictions only)
# ===========================================================================
set -uo pipefail
export PYTHONUNBUFFERED=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$ROOT_DIR"

if [ -x "/dev/shm/phase/.venv/bin/python" ]; then
    PYTHON="/dev/shm/phase/.venv/bin/python"
else
    echo "ERROR: phase env not found at /dev/shm/phase/.venv/bin/python" >&2
    exit 2
fi
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
PYTHON_BIN_DIR="$(cd "$(dirname "$PYTHON")" && pwd)"
export PATH="${PYTHON_BIN_DIR}:/usr/local/cuda/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

MODEL="${MODEL:-/mnt/lstore/model/gemma-4-E4B-it}"
LOCAL_EMBED_MODEL="${LOCAL_EMBED_MODEL:-/mnt/lstore/model/Qwen3-Embedding-8B}"
RESULTS_TAG="${RESULTS_TAG:-gemma_4_e4b_it}"
TOP_K="${TOP_K:-3}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
TEMPERATURE="${TEMPERATURE:-0.3}"
SEED="${SEED:-42}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.5}"
CFG_BATCH_SIZE="${CFG_BATCH_SIZE:-8}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
SKIP_JUDGE="${SKIP_JUDGE:-1}"

GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
IFS=' ' read -ra GPU_ID_ARR <<< "$GPU_IDS"
NUM_GPU_WORKERS="${NUM_GPU_WORKERS:-${#GPU_ID_ARR[@]}}"

DEFAULT_DATASETS="RAIDEN CharacterEval SimsConv ChatHaruhi Friends HPD StarTrek_TNG TheOffice"
IFS=' ' read -ra DATASET_LIST <<< "${DATASETS:-$DEFAULT_DATASETS}"

DEFAULT_METHODS="rag pag cfg"
IFS=' ' read -ra METHOD_LIST <<< "${METHODS:-$DEFAULT_METHODS}"

SPLITS_STR="${EVAL_SPLITS:-random_test ood_test}"
IFS=' ' read -ra SPLITS <<< "$SPLITS_STR"

GLOBAL_LOG_DIR="results/_comparison_${RESULTS_TAG}_logs"
mkdir -p "$GLOBAL_LOG_DIR"

if [ ! -d "$MODEL" ]; then
    echo "ERROR: model not found: $MODEL" >&2
    exit 2
fi

START_TIME=$(date +%s)
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Comparison Predictions — ${RESULTS_TAG} (Gemma-4-E4B-it)"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model      : ${MODEL}"
echo "║  Embed model: ${LOCAL_EMBED_MODEL}"
echo "║  Datasets   : ${DATASET_LIST[*]}"
echo "║  Methods    : ${METHOD_LIST[*]}"
echo "║  Splits     : ${SPLITS[*]}"
echo "║  GPUs       : ${GPU_ID_ARR[*]} (${NUM_GPU_WORKERS} workers)"
echo "║  Top-K      : ${TOP_K}"
echo "║  CFG batch  : ${CFG_BATCH_SIZE}"
echo "║  Output     : results/<DS>/comparison/${RESULTS_TAG}/"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Build flat task list: dataset:method:split
ALL_TASKS=()
for DATASET in "${DATASET_LIST[@]}"; do
    DATA_DIR="LongEvoRoleBench/processed/${DATASET}"
    if [ ! -d "$DATA_DIR" ]; then
        echo "  [SKIP] missing data dir: ${DATA_DIR}"
        continue
    fi
    POOL_PATH="${DATA_DIR}/m1_context_only/train.json"
    PROFILE_DATA="${DATA_DIR}/m2_raw_profile/all_dialogues.json"

    for METHOD in "${METHOD_LIST[@]}"; do
        case "$METHOD" in
            rag|pag)
                DATA_VARIANT="m1_context_only"
                if [ ! -f "$POOL_PATH" ]; then
                    echo "  [SKIP] ${DATASET}/${METHOD}: pool missing ${POOL_PATH}"
                    continue
                fi
                if [ "$METHOD" = "pag" ] && [ ! -f "$PROFILE_DATA" ]; then
                    echo "  [SKIP] ${DATASET}/pag: profile missing ${PROFILE_DATA}"
                    continue
                fi
                ;;
            cfg)
                DATA_VARIANT="m2_raw_profile"
                ;;
            *)
                echo "  [SKIP] unknown method: ${METHOD}"
                continue
                ;;
        esac

        for SPLIT in "${SPLITS[@]}"; do
            DATA_FILE="${DATA_DIR}/${DATA_VARIANT}/${SPLIT}.json"
            if [ -f "$DATA_FILE" ]; then
                ALL_TASKS+=("${DATASET}:${METHOD}:${SPLIT}")
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
echo "  Total tasks: ${N_TASKS}, ${NUM_GPU_WORKERS} GPU workers (sequential per GPU)"
echo ""

# Distribute tasks round-robin across GPU workers
declare -A GPU_TASKS
for i in "${!ALL_TASKS[@]}"; do
    gpu_slot=$((i % NUM_GPU_WORKERS))
    GPU_TASKS[$gpu_slot]+="${ALL_TASKS[$i]};"
done

NUM_SAMPLES_FLAG=()
if [ -n "${NUM_SAMPLES:-}" ]; then
    NUM_SAMPLES_FLAG=(--num_samples "$NUM_SAMPLES")
fi

run_gpu_worker() {
    local PHYSICAL_GPU="$1"
    local task_str="$2"
    local worker_log="${GLOBAL_LOG_DIR}/worker_gpu${PHYSICAL_GPU}.log"
    local fail=0

    {
        echo "=== GPU ${PHYSICAL_GPU} worker start $(date -Iseconds) ==="
        IFS=';' read -ra TASK_ITEMS <<< "$task_str"
        for item in "${TASK_ITEMS[@]}"; do
            [ -z "$item" ] && continue
            IFS=':' read -r dataset method split <<< "$item"

            DATA_DIR="LongEvoRoleBench/processed/${dataset}"
            POOL_PATH="${DATA_DIR}/m1_context_only/train.json"
            PROFILE_DATA="${DATA_DIR}/m2_raw_profile/all_dialogues.json"
            OUT_DIR="results/${dataset}/comparison/${RESULTS_TAG}/${method}/${split}"
            mkdir -p "$OUT_DIR"
            TASK_LOG="${OUT_DIR}/_predict.log"

            echo ""
            echo "--- [GPU ${PHYSICAL_GPU}] ${dataset}/${method}/${split} $(date -Iseconds) ---"

            case "$method" in
                rag)
                    DATA_FILE="${DATA_DIR}/m1_context_only/${split}.json"
                    if ! CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" $PYTHON evaluation/predict_rag.py \
                        --data "$DATA_FILE" \
                        --pool "$POOL_PATH" \
                        --output_dir "$OUT_DIR" \
                        --mode rag \
                        --top_k "$TOP_K" \
                        --model "$MODEL" \
                        --backend vllm \
                        --temperature "$TEMPERATURE" \
                        --max_tokens "$MAX_TOKENS" \
                        --max_model_len "$MAX_MODEL_LEN" \
                        --tensor_parallel 1 \
                        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
                        --seed "$SEED" \
                        --embed_batch_size "$EMBED_BATCH_SIZE" \
                        --embed_model "$LOCAL_EMBED_MODEL" \
                        "${NUM_SAMPLES_FLAG[@]}" \
                        > "$TASK_LOG" 2>&1; then
                        echo "FAIL: ${dataset}/${method}/${split} (see ${TASK_LOG})"
                        fail=$((fail + 1))
                    else
                        echo "OK: ${dataset}/${method}/${split}"
                    fi
                    ;;
                pag)
                    DATA_FILE="${DATA_DIR}/m1_context_only/${split}.json"
                    if ! CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" $PYTHON evaluation/predict_rag.py \
                        --data "$DATA_FILE" \
                        --pool "$POOL_PATH" \
                        --output_dir "$OUT_DIR" \
                        --mode pag \
                        --top_k "$TOP_K" \
                        --profile_data "$PROFILE_DATA" \
                        --model "$MODEL" \
                        --backend vllm \
                        --temperature "$TEMPERATURE" \
                        --max_tokens "$MAX_TOKENS" \
                        --max_model_len "$MAX_MODEL_LEN" \
                        --tensor_parallel 1 \
                        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
                        --seed "$SEED" \
                        --embed_batch_size "$EMBED_BATCH_SIZE" \
                        --embed_model "$LOCAL_EMBED_MODEL" \
                        "${NUM_SAMPLES_FLAG[@]}" \
                        > "$TASK_LOG" 2>&1; then
                        echo "FAIL: ${dataset}/${method}/${split} (see ${TASK_LOG})"
                        fail=$((fail + 1))
                    else
                        echo "OK: ${dataset}/${method}/${split}"
                    fi
                    ;;
                cfg)
                    DATA_FILE="${DATA_DIR}/m2_raw_profile/${split}.json"
                    if ! CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" $PYTHON evaluation/predict_cfg.py \
                        --data "$DATA_FILE" \
                        --output_dir "$OUT_DIR" \
                        --model "$MODEL" \
                        --guidance_scale "$GUIDANCE_SCALE" \
                        --temperature "$TEMPERATURE" \
                        --max_tokens "$MAX_TOKENS" \
                        --max_model_len "$MAX_MODEL_LEN" \
                        --batch_size "$CFG_BATCH_SIZE" \
                        --seed "$SEED" \
                        "${NUM_SAMPLES_FLAG[@]}" \
                        > "$TASK_LOG" 2>&1; then
                        echo "FAIL: ${dataset}/${method}/${split} (see ${TASK_LOG})"
                        fail=$((fail + 1))
                    else
                        echo "OK: ${dataset}/${method}/${split}"
                    fi
                    ;;
            esac
        done
        echo ""
        echo "=== GPU ${PHYSICAL_GPU} worker done, failures=${fail} $(date -Iseconds) ==="
        exit "$fail"
    } > "$worker_log" 2>&1
}

WORKER_PIDS=()
WORKER_DESCS=()
for gpu_slot in $(seq 0 $((NUM_GPU_WORKERS - 1))); do
    task_str="${GPU_TASKS[$gpu_slot]:-}"
    if [ -z "$task_str" ]; then
        continue
    fi
    PHYSICAL_GPU="${GPU_ID_ARR[$gpu_slot]:-$gpu_slot}"
    n_tasks=$(echo "$task_str" | tr ';' '\n' | grep -c . || true)
    echo "  [GPU ${PHYSICAL_GPU}] ${n_tasks} tasks -> ${GLOBAL_LOG_DIR}/worker_gpu${PHYSICAL_GPU}.log"
    run_gpu_worker "$PHYSICAL_GPU" "$task_str" &
    WORKER_PIDS+=($!)
    WORKER_DESCS+=("GPU ${PHYSICAL_GPU} (${n_tasks} tasks)")
done

echo ""
echo "  Waiting for ${#WORKER_PIDS[@]} GPU workers ..."
TOTAL_FAIL=0
for i in "${!WORKER_PIDS[@]}"; do
    wait "${WORKER_PIDS[$i]}" 2>/dev/null
    ec=$?
    if [ $ec -eq 0 ]; then
        echo "  OK: ${WORKER_DESCS[$i]}"
    else
        echo "  FAIL (exit $ec): ${WORKER_DESCS[$i]}  (see ${GLOBAL_LOG_DIR}/)"
        TOTAL_FAIL=$((TOTAL_FAIL + ec))
    fi
done

echo ""
echo "  Prediction counts:"
shopt -s nullglob
for DATASET in "${DATASET_LIST[@]}"; do
    for METHOD in "${METHOD_LIST[@]}"; do
        for SPLIT in "${SPLITS[@]}"; do
            pred="results/${DATASET}/comparison/${RESULTS_TAG}/${METHOD}/${SPLIT}/predictions.jsonl"
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
echo "║  Comparison predictions complete! [$((END_TIME - START_TIME))s]"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Task failures: ${TOTAL_FAIL}"
echo "║  Results      : results/<DS>/comparison/${RESULTS_TAG}/"
echo "║  Logs         : ${GLOBAL_LOG_DIR}/"
echo "╚══════════════════════════════════════════════════════════════╝"

exit "$([ "$TOTAL_FAIL" -eq 0 ] && echo 0 || echo 1)"
