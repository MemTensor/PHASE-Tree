#!/usr/bin/env bash
# ===========================================================================
# PHASE-Tree SFT Hypernetwork Evaluation (dataset-agnostic, multi-GPU)
# ===========================================================================
# Uses the PHASE-Tree SFT-finetuned hypermod checkpoint to generate
# per-character LoRA adapters and run personalised vLLM inference.
#
# Architecture: one vLLM + hypermod loaded per GPU.  Multiple (method, split)
# tasks are packed onto a single GPU via predict_phase_tree.py's --multi mode
# (model loaded ONCE per GPU, LoRA cache reused across tasks).
#
# Two dataset modes (auto-detected if MODE is unset):
#   * short-term : m2_raw_profile + m3_naive_rewrite + m4_static_tree + m6_phase_tree
#                  Per-character LoRA cache: m2, m4
#                  Datasets: RAIDEN, CharacterEval, SimsConv, ChatHaruhi
#   * long-term  : m2_raw_profile + m3_naive_rewrite + m4_static_tree
#                  + m5_dynamic_tree + m6_phase_tree
#                  Per-character LoRA cache: m2, m4
#                  Per-sample (auto-chunked LoRA gen): m3, m5, m6
#                  Datasets: Friends, HPD, StarTrek_TNG, TheOffice
#
# Usage:
#   bash evaluation/run_phase_tree_eval.sh                          # default: ALL datasets, auto mode
#   bash evaluation/run_phase_tree_eval.sh RAIDEN                   # single dataset, auto short-term
#   bash evaluation/run_phase_tree_eval.sh Friends long-term        # explicit mode
#   DATASETS="RAIDEN ChatHaruhi" bash evaluation/run_phase_tree_eval.sh  # subset
#   SKIP_JUDGE=1 bash evaluation/run_phase_tree_eval.sh             # predictions only
#
# Environment knobs (all optional):
#   DATASETS               (space-separated; default: ALL 8 datasets)
#   MODE                   (short-term | long-term; auto-detected per dataset)
#   METHODS                (override the default method list)
#   GPUS                   (comma-separated visible GPU IDs; default: all)
#   TEMPERATURE            (default 0.3)
#   MAX_TOKENS             (default 256)
#   SEED                   (default 42)
#   EMB_BATCH_SIZE         (default 64)
#   CHUNK_SIZE             (default 0 = auto; only used for per-sample methods)
#   GPU_MEMORY_UTILIZATION (default 0.7; vLLM GPU memory fraction)
#   SKIP_JUDGE             (=1 to stop after predictions)
#   SKIP_REPORT            (=1 to skip report.py)
#   SKIP_VIZ               (=1 to skip visualize.py)
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
    PYTHON="${PYTHON:-$(command -v python)}"
fi
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}:${PYTHONPATH:-}"

# ── Dataset resolution ─────────────────────────────────────────────────
ALL_DATASETS="RAIDEN CharacterEval HPD SimsConv ChatHaruhi Friends StarTrek_TNG TheOffice"
if [ -n "${1:-}" ]; then
    DATASETS_STR="$1"
    MODE_ARG="${2:-${MODE:-}}"
elif [ -n "${DATASETS:-}" ]; then
    DATASETS_STR="$DATASETS"
    MODE_ARG="${MODE:-}"
else
    DATASETS_STR="$ALL_DATASETS"
    MODE_ARG="${MODE:-}"
fi
IFS=' ' read -ra DATASET_LIST <<< "$DATASETS_STR"

# ── PHASE-Tree SFT Checkpoint + models ────────────────────────────────
SFT_CKPT_DIR="phase_tree_models/sft/hyper_lora"
if [ -f "/dev/shm/phase_tree_sft/hypermod.pt" ]; then
    CHECKPOINT="/dev/shm/phase_tree_sft/hypermod.pt"
else
    CHECKPOINT="${SFT_CKPT_DIR}/hypermod.pt"
fi
if [ -d "/dev/shm/Qwen3-Embedding-4B" ]; then
    EMB_MODEL="/dev/shm/Qwen3-Embedding-4B"
elif [ -d "/dev/shm/phase/models/Qwen3-Embedding-4B" ]; then
    EMB_MODEL="/dev/shm/phase/models/Qwen3-Embedding-4B"
else
    EMB_MODEL="${EMB_MODEL:-${ROOT_DIR}/models/Qwen3-Embedding-4B}"
fi
if [ -d "/dev/shm/Qwen2.5-7B-Instruct" ]; then
    MODEL_OVERRIDE="/dev/shm/Qwen2.5-7B-Instruct"
elif [ -d "/dev/shm/phase/models/Qwen2.5-7B-Instruct" ]; then
    MODEL_OVERRIDE="/dev/shm/phase/models/Qwen2.5-7B-Instruct"
else
    MODEL_OVERRIDE=""
fi

TEMPERATURE="${TEMPERATURE:-0.3}"
MAX_TOKENS="${MAX_TOKENS:-256}"
SEED="${SEED:-42}"
EMB_BATCH_SIZE="${EMB_BATCH_SIZE:-64}"
CHUNK_SIZE="${CHUNK_SIZE:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"

# Judge / report / viz
BASELINE_METHOD="${BASELINE_METHOD:-m2_raw_profile}"
NUM_WORKERS="${NUM_WORKERS:-10}"
EMBED_WORKERS="${EMBED_WORKERS:-10}"
RATE_LIMIT_SLEEP="${RATE_LIMIT_SLEEP:-0.1}"
CONCURRENT_PASSES="${CONCURRENT_PASSES:-1}"
SKIP_JUDGE="${SKIP_JUDGE:-0}"
SKIP_REPORT="${SKIP_REPORT:-0}"
SKIP_VIZ="${SKIP_VIZ:-0}"

PER_CHAR_METHODS=("m2_raw_profile" "m4_static_tree")

# ── GPU selection ──────────────────────────────────────────────────────
if [ -n "${GPUS:-}" ]; then
    IFS=',' read -ra GPU_LIST <<< "$GPUS"
else
    DETECTED_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ',' | sed 's/,$//' || echo "0")
    IFS=',' read -ra GPU_LIST <<< "$DETECTED_GPUS"
fi
NUM_GPUS=${#GPU_LIST[@]}
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "ERROR: no usable GPUs" >&2
    exit 2
fi

START_TIME=$(date +%s)
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE-Tree SFT Hypernet Evaluation (high-parallelism)      "
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Datasets   : ${DATASET_LIST[*]}"
echo "║  Checkpoint  : ${CHECKPOINT}"
echo "║  Emb model   : ${EMB_MODEL}"
echo "║  Model (LLM) : ${MODEL_OVERRIDE:-<from checkpoint>}"
echo "║  GPUs        : ${GPU_LIST[*]} (${NUM_GPUS} workers)"
echo "║  Per-char cache: ${PER_CHAR_METHODS[*]}"
echo "║  Chunk size  : ${CHUNK_SIZE} (0 = auto for per-sample methods)"
echo "║  gpu_mem_util : ${GPU_MEMORY_UTILIZATION}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Build flat task list across ALL datasets ───────────────────────────
declare -a ALL_TASKS
declare -a ALL_TASKS_DESC
declare -a ALL_DATASETS_USED

for DATASET in "${DATASET_LIST[@]}"; do
    DATA_DIR="LongEvoRoleBench/processed/${DATASET}"
    if [ ! -d "$DATA_DIR" ]; then
        echo "  [SKIP] Dataset dir not found: ${DATA_DIR}"
        continue
    fi

    # Auto-detect mode per dataset
    DS_MODE="$MODE_ARG"
    if [ -z "$DS_MODE" ]; then
        if [ -d "${DATA_DIR}/m5_dynamic_tree" ]; then
            DS_MODE="long-term"
        elif [ -d "${DATA_DIR}/m6_phase_tree" ]; then
            DS_MODE="short-term"
        else
            echo "  [SKIP] Cannot auto-detect mode for ${DATASET}"
            continue
        fi
    fi

    case "$DS_MODE" in
        short|short-term|shortterm)
            DS_METHODS_DEFAULT="m2_raw_profile m3_naive_rewrite m4_static_tree m6_phase_tree"
            ;;
        long|long-term|longterm)
            DS_METHODS_DEFAULT="m2_raw_profile m3_naive_rewrite m4_static_tree m5_dynamic_tree m6_phase_tree"
            ;;
        *)
            echo "  [SKIP] Unknown MODE '$DS_MODE' for ${DATASET}"
            continue
            ;;
    esac

    IFS=' ' read -ra DS_METHODS <<< "${METHODS:-$DS_METHODS_DEFAULT}"
    SPLITS=("random_test" "ood_test")

    RESULTS_DIR="results/${DATASET}/phase_tree/main"
    LORA_DIR="results/${DATASET}/phase_tree/generated_loras"
    TASKS_DIR="results/${DATASET}/phase_tree/_tasks"
    LOG_DIR="results/${DATASET}/phase_tree/_logs"
    mkdir -p "$LOG_DIR" "$TASKS_DIR"

    # Clean residual temp dirs
    find "$RESULTS_DIR" -maxdepth 2 -type d -name "_shared_temp_loras" -exec rm -rf {} + 2>/dev/null || true
    find "$RESULTS_DIR" -maxdepth 2 -type d -name "_temp_loras" -exec rm -rf {} + 2>/dev/null || true

    for METHOD in "${DS_METHODS[@]}"; do
        SAVE_LORA_VALUE=""
        for pc in "${PER_CHAR_METHODS[@]}"; do
            if [ "$METHOD" = "$pc" ]; then
                SAVE_LORA_VALUE="${LORA_DIR}/${METHOD}"
                break
            fi
        done
        for SPLIT in "${SPLITS[@]}"; do
            DATA_FILE="${DATA_DIR}/${METHOD}/${SPLIT}.json"
            if [ ! -f "$DATA_FILE" ]; then
                echo "  [SKIP] ${DATA_FILE} not found"
                continue
            fi
            ALL_TASKS+=("${DATASET}|${METHOD}|${SPLIT}|${SAVE_LORA_VALUE}")
            ALL_TASKS_DESC+=("${DATASET}/${METHOD}/${SPLIT}")
        done
    done
    ALL_DATASETS_USED+=("$DATASET")
done

N_TASKS=${#ALL_TASKS[@]}
echo "  Total tasks: ${N_TASKS}, distributing across ${NUM_GPUS} GPUs (${GPU_LIST[*]})"
echo ""

if [ "$N_TASKS" -eq 0 ]; then
    echo "ERROR: No tasks to run" >&2
    exit 1
fi

# ── Round-robin task distribution across GPUs ──────────────────────────
declare -A GPU_TASK_MAP
for i in "${!ALL_TASKS[@]}"; do
    slot=$(( i % NUM_GPUS ))
    GPU_TASK_MAP[$slot]+="${ALL_TASKS[$i]}"$'\n'
done

# ── Launch one worker per GPU ──────────────────────────────────────────
rm -rf /dev/shm/lora_tmp_* 2>/dev/null || true

PIDS=()
TASK_DESCS=()

for slot in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID="${GPU_LIST[$slot]}"
    task_str="${GPU_TASK_MAP[$slot]:-}"
    [ -z "$task_str" ] && continue

    TASK_FILE="results/_phase_tree_tasks/tasks_gpu${GPU_ID}.json"
    mkdir -p "$(dirname "$TASK_FILE")"
    COMBINED_LOG="results/_phase_tree_tasks/predict_gpu${GPU_ID}.log"

    ENTRIES="["
    FIRST=true
    GPU_DESC=""
    while IFS= read -r entry; do
        [ -z "$entry" ] && continue
        IFS='|' read -r ds m s save_dir <<< "$entry"
        df="LongEvoRoleBench/processed/${ds}/${m}/${s}.json"
        od="results/${ds}/phase_tree/main/${m}/${s}"
        if [ "$FIRST" = true ]; then FIRST=false; else ENTRIES+=","; fi
        if [ -n "$save_dir" ]; then
            ENTRIES+="
    {\"data\": \"${df}\", \"output_dir\": \"${od}\", \"save_loras_dir\": \"${save_dir}\"}"
        else
            ENTRIES+="
    {\"data\": \"${df}\", \"output_dir\": \"${od}\"}"
        fi
        GPU_DESC+="${ds}/${m}/${s} "
    done <<< "$task_str"
    ENTRIES+="
]"
    echo "$ENTRIES" > "$TASK_FILE"

    echo "  [GPU $GPU_ID] ${GPU_DESC}"

    MODEL_ARG=""
    if [ -n "$MODEL_OVERRIDE" ]; then
        MODEL_ARG="--model_override $MODEL_OVERRIDE"
    fi

    CUDA_VISIBLE_DEVICES="$GPU_ID" $PYTHON evaluation/predict_phase_tree.py \
        --multi \
        --tasks "$TASK_FILE" \
        --checkpoint "$CHECKPOINT" \
        --emb_model_override "$EMB_MODEL" \
        --temperature "$TEMPERATURE" \
        --max_tokens "$MAX_TOKENS" \
        --seed "$SEED" \
        --tensor_parallel 1 \
        --emb_batch_size "$EMB_BATCH_SIZE" \
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
        $MODEL_ARG \
        > "$COMBINED_LOG" 2>&1 &

    PIDS+=($!)
    TASK_DESCS+=("GPU $GPU_ID (${GPU_DESC})")
done

echo ""
echo "  Launched ${#PIDS[@]} workers across ${NUM_GPUS} GPUs"
echo "  Logs: results/_phase_tree_tasks/"
echo ""

# ── Wait for all workers ──────────────────────────────────────────────
echo "  Waiting for all workers ..."
FAILURES=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" 2>/dev/null
    ec=$?
    if [ $ec -eq 0 ]; then
        echo "  OK: ${TASK_DESCS[$i]}"
    else
        echo "  FAIL (exit $ec): ${TASK_DESCS[$i]}"
        FAILURES=$((FAILURES + 1))
    fi
done

PREDICT_END=$(date +%s)
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
if [ $FAILURES -eq 0 ]; then
    echo "║  Predict phase complete. [$((PREDICT_END - START_TIME))s]"
else
    echo "║  ${FAILURES}/${#PIDS[@]} predict workers FAILED. [$((PREDICT_END - START_TIME))s]"
fi
echo "╠══════════════════════════════════════════════════════════════╣"
for DATASET in "${ALL_DATASETS_USED[@]}"; do
    DATA_DIR="LongEvoRoleBench/processed/${DATASET}"
    DS_MODE="$MODE_ARG"
    if [ -z "$DS_MODE" ]; then
        if [ -d "${DATA_DIR}/m5_dynamic_tree" ]; then DS_MODE="long-term"; else DS_MODE="short-term"; fi
    fi
    case "$DS_MODE" in
        short|short-term|shortterm) DS_METHODS_DEFAULT="m2_raw_profile m3_naive_rewrite m4_static_tree m6_phase_tree" ;;
        *) DS_METHODS_DEFAULT="m2_raw_profile m3_naive_rewrite m4_static_tree m5_dynamic_tree m6_phase_tree" ;;
    esac
    IFS=' ' read -ra DS_METHODS <<< "${METHODS:-$DS_METHODS_DEFAULT}"
    for METHOD in "${DS_METHODS[@]}"; do
        for SPLIT in random_test ood_test; do
            pred="results/${DATASET}/phase_tree/main/${METHOD}/${SPLIT}/predictions.jsonl"
            if [ -f "$pred" ]; then
                n=$(wc -l < "$pred")
                echo "║  ${DATASET}/${METHOD}/${SPLIT}: ${n} predictions"
            fi
        done
    done
done
echo "╚══════════════════════════════════════════════════════════════╝"

# ── STEP 2: Judge + Embedding ──────────────────────────────────────────
if [ "$SKIP_JUDGE" = "1" ] || [ "$SKIP_JUDGE" = "true" ]; then
    echo ""
    echo "  [SKIP_JUDGE=1] Stopping after predictions."
    exit $FAILURES
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2 ▸ Scoring (parallel per dataset)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP2_START=$(date +%s)

JUDGE_PIDS=()
for DATASET in "${ALL_DATASETS_USED[@]}"; do
    DATA_DIR="LongEvoRoleBench/processed/${DATASET}"
    RESULTS_DIR="results/${DATASET}/phase_tree/main"
    LOG_DIR="results/${DATASET}/phase_tree/_logs"
    mkdir -p "$LOG_DIR"
    PERSONA_DATA="${DATA_DIR}/m6_phase_tree/all_dialogues.json"
    if [ ! -f "$PERSONA_DATA" ]; then
        echo "  [SKIP] ${PERSONA_DATA} not found for ${DATASET}"
        continue
    fi

    DS_MODE="$MODE_ARG"
    if [ -z "$DS_MODE" ]; then
        if [ -d "${DATA_DIR}/m5_dynamic_tree" ]; then DS_MODE="long-term"; else DS_MODE="short-term"; fi
    fi
    case "$DS_MODE" in
        short|short-term|shortterm) DS_METHODS_DEFAULT="m2_raw_profile m3_naive_rewrite m4_static_tree m6_phase_tree" ;;
        *) DS_METHODS_DEFAULT="m2_raw_profile m3_naive_rewrite m4_static_tree m5_dynamic_tree m6_phase_tree" ;;
    esac
    IFS=' ' read -ra DS_METHODS <<< "${METHODS:-$DS_METHODS_DEFAULT}"

    for METHOD in "${DS_METHODS[@]}"; do
        for SPLIT in random_test ood_test; do
            OUT_DIR="${RESULTS_DIR}/${METHOD}/${SPLIT}"
            if [ ! -f "${OUT_DIR}/predictions.jsonl" ]; then continue; fi
            LOG_FILE="${LOG_DIR}/judge_${METHOD}_${SPLIT}.log"
            echo "  ${DATASET}/${METHOD}/${SPLIT}"
            JUDGE_ARGS=(
                --predictions_dir "$OUT_DIR"
                --persona_data "$PERSONA_DATA"
                --num_workers "$NUM_WORKERS"
                --rate_limit_sleep "$RATE_LIMIT_SLEEP"
            )
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
done

echo ""
echo "  Waiting for all judge tasks (${#JUDGE_PIDS[@]} processes) ..."
JUDGE_FAIL=0
for pid in "${JUDGE_PIDS[@]}"; do
    wait "$pid" || JUDGE_FAIL=$((JUDGE_FAIL + 1))
done
STEP2_END=$(date +%s)
echo "  Scoring complete [$((STEP2_END - STEP2_START))s] (${JUDGE_FAIL} failures)"

# ── STEP 3: Report per dataset ─────────────────────────────────────────
if [ "$SKIP_REPORT" = "1" ] || [ "$SKIP_REPORT" = "true" ]; then
    echo ""
    echo "  [SKIP_REPORT=1] Skipping report.py"
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 3 ▸ Report Generation"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for DATASET in "${ALL_DATASETS_USED[@]}"; do
        DATA_DIR="LongEvoRoleBench/processed/${DATASET}"
        RESULTS_DIR="results/${DATASET}/phase_tree/main"
        DS_MODE="$MODE_ARG"
        if [ -z "$DS_MODE" ]; then
            if [ -d "${DATA_DIR}/m5_dynamic_tree" ]; then DS_MODE="long-term"; else DS_MODE="short-term"; fi
        fi
        case "$DS_MODE" in
            short|short-term|shortterm) DS_METHODS_DEFAULT="m2_raw_profile m3_naive_rewrite m4_static_tree m6_phase_tree" ;;
            *) DS_METHODS_DEFAULT="m2_raw_profile m3_naive_rewrite m4_static_tree m5_dynamic_tree m6_phase_tree" ;;
        esac
        IFS=' ' read -ra DS_METHODS <<< "${METHODS:-$DS_METHODS_DEFAULT}"

        SCORED=()
        for METHOD in "${DS_METHODS[@]}"; do
            for SPLIT in random_test ood_test; do
                if [ -f "${RESULTS_DIR}/${METHOD}/${SPLIT}/judge_scores.jsonl" ]; then
                    SCORED+=("$METHOD")
                    break
                fi
            done
        done
        if [ ${#SCORED[@]} -gt 0 ]; then
            echo "  ${DATASET}: reporting ${SCORED[*]}"
            $PYTHON evaluation/report.py \
                --results_dir "$RESULTS_DIR" \
                --experiments "${SCORED[@]}" \
                --splits "random_test" "ood_test" \
                --baseline "$BASELINE_METHOD" \
                --per_character
        fi
    done
fi

# ── STEP 4: Visualization ──────────────────────────────────────────────
if [ "$SKIP_VIZ" = "1" ] || [ "$SKIP_VIZ" = "true" ]; then
    echo ""
    echo "  [SKIP_VIZ=1] Skipping visualize.py"
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 4 ▸ Visualization"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for DATASET in "${ALL_DATASETS_USED[@]}"; do
        RESULTS_DIR="results/${DATASET}/phase_tree/main"
        if [ -f "${RESULTS_DIR}/summary.json" ]; then
            $PYTHON evaluation/visualize.py \
                --results_dir "$RESULTS_DIR" \
                --format pdf
            echo "  ${DATASET}: figures generated"
        fi
    done
fi

END_TIME=$(date +%s)
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE-Tree SFT pipeline complete! [$((END_TIME - START_TIME))s total]"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Datasets  : ${ALL_DATASETS_USED[*]}"
echo "║  Checkpoint: ${CHECKPOINT}"
echo "║  Results   : results/<DATASET>/phase_tree/main/"
echo "║  Tasks     : ${N_TASKS} total across ${NUM_GPUS} GPUs"
echo "╚══════════════════════════════════════════════════════════════╝"

exit $FAILURES
