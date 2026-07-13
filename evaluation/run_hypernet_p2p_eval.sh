#!/usr/bin/env bash
# ===========================================================================
# PHASE-Tree P2P Hypernetwork Evaluation (dataset-agnostic, multi-GPU)
# ===========================================================================
# One vLLM + hypermod loaded per GPU.  Multiple (method, split) tasks may be
# packed onto a single GPU via predict_hypernet.py's --multi mode (model
# loaded ONCE per GPU, LoRA cache reused across tasks).
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
#   bash evaluation/run_hypernet_p2p_eval.sh                          # default: RAIDEN, short-term
#   bash evaluation/run_hypernet_p2p_eval.sh RAIDEN                   # auto short-term
#   bash evaluation/run_hypernet_p2p_eval.sh Friends                  # auto long-term
#   bash evaluation/run_hypernet_p2p_eval.sh Friends long-term        # explicit
#
# Environment knobs (all optional):
#   DATASET                (default $1 or RAIDEN)
#   MODE                   (short-term | long-term; auto-detected if unset)
#   METHODS                (override the default method list)
#   GPUS                   (comma-separated visible GPU IDs; default: all)
#   TEMPERATURE            (default 0.3)
#   MAX_TOKENS             (default 256)
#   SEED                   (default 42)
#   EMB_BATCH_SIZE         (default 64)
#   CHUNK_SIZE             (default 0 = auto; only used for per-sample methods)
#   PERSONA_DATA           (judge persona ref; auto from mode if unset)
#   BASELINE_METHOD        (default m2_raw_profile)
#   NUM_WORKERS            (judge LLM workers per task; short=10, long=128)
#   EMBED_WORKERS          (embed workers per task; short=10, long=32)
#   RATE_LIMIT_SLEEP       (judge call sleep; short=0.1, long=0.0)
#   CONCURRENT_PASSES      (judge ∥ embed within each task; default 1)
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

# ── Positional + env resolution ────────────────────────────────────────
DATASET="${1:-${DATASET:-RAIDEN}}"
MODE_ARG="${2:-${MODE:-}}"

DATA_DIR="phase_tree_data/processed/${DATASET}"
if [ -z "$MODE_ARG" ]; then
    if [ -d "${DATA_DIR}/m5_dynamic_tree" ]; then
        MODE_ARG="long-term"
    elif [ -d "${DATA_DIR}/m6_phase_tree" ]; then
        MODE_ARG="short-term"
    else
        echo "ERROR: cannot auto-detect MODE for ${DATASET}" >&2
        exit 2
    fi
fi

case "$MODE_ARG" in
    short|short-term|shortterm)
        MODE="short-term"
        DEFAULT_METHODS="m2_raw_profile m3_naive_rewrite m4_static_tree m6_phase_tree"
        DEFAULT_PERSONA="${DATA_DIR}/m6_phase_tree/all_dialogues.json"
        DEFAULT_NUM_WORKERS=10
        DEFAULT_EMBED_WORKERS=10
        DEFAULT_RATE_SLEEP="0.1"
        ;;
    long|long-term|longterm)
        MODE="long-term"
        DEFAULT_METHODS="m2_raw_profile m3_naive_rewrite m4_static_tree m5_dynamic_tree m6_phase_tree"
        DEFAULT_PERSONA="${DATA_DIR}/m6_phase_tree/all_dialogues.json"
        DEFAULT_NUM_WORKERS=128
        DEFAULT_EMBED_WORKERS=32
        DEFAULT_RATE_SLEEP="0.0"
        ;;
    *)
        echo "ERROR: unknown MODE '$MODE_ARG'" >&2
        exit 2
        ;;
esac

# Per-character LoRA cache list (shared by both modes — only m2 and m4 produce
# a single profile per character)
PER_CHAR_METHODS=("m2_raw_profile" "m4_static_tree")

IFS=' ' read -ra METHODS <<< "${METHODS:-$DEFAULT_METHODS}"
SPLITS_STR="${EVAL_SPLITS:-random_test ood_test}"
IFS=' ' read -ra SPLITS <<< "$SPLITS_STR"

# ── Checkpoint + models ────────────────────────────────────────────────
if [ -f "/dev/shm/p2p_pretrained/hypermod.pt" ]; then
    CHECKPOINT="/dev/shm/p2p_pretrained/hypermod.pt"
else
    CHECKPOINT="phase_tree_models/p2p_pretrained/hypermod.pt"
fi
if [ -d "/dev/shm/Qwen3-Embedding-4B" ]; then
    EMB_MODEL="/dev/shm/Qwen3-Embedding-4B"
else
    EMB_MODEL="${EMB_MODEL:-${ROOT_DIR}/models/Qwen3-Embedding-4B}"
fi
if [ -d "/dev/shm/Qwen2.5-7B-Instruct" ]; then
    MODEL_OVERRIDE="/dev/shm/Qwen2.5-7B-Instruct"
else
    MODEL_OVERRIDE=""
fi

RESULTS_DIR="results/${DATASET}/hypernet_p2p/main"
LORA_DIR="results/${DATASET}/hypernet_p2p/generated_loras"
LOG_DIR="results/${DATASET}/hypernet_p2p/_logs"
TASKS_DIR="results/${DATASET}/hypernet_p2p/_tasks"
mkdir -p "$LOG_DIR" "$TASKS_DIR"

# Clean residual temp dirs from previous interrupted runs (AFS + ramdisk)
find "$RESULTS_DIR" -maxdepth 2 -type d -name "_shared_temp_loras" -exec rm -rf {} + 2>/dev/null || true
find "$RESULTS_DIR" -maxdepth 2 -type d -name "_temp_loras" -exec rm -rf {} + 2>/dev/null || true
rm -rf /dev/shm/lora_tmp_* 2>/dev/null || true

TEMPERATURE="${TEMPERATURE:-0.3}"
MAX_TOKENS="${MAX_TOKENS:-256}"
SEED="${SEED:-42}"
EMB_BATCH_SIZE="${EMB_BATCH_SIZE:-64}"
CHUNK_SIZE="${CHUNK_SIZE:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"

# Judge / report / viz wiring
PERSONA_DATA="${PERSONA_DATA:-$DEFAULT_PERSONA}"
BASELINE_METHOD="${BASELINE_METHOD:-m2_raw_profile}"
NUM_WORKERS="${NUM_WORKERS:-$DEFAULT_NUM_WORKERS}"
EMBED_WORKERS="${EMBED_WORKERS:-${DEFAULT_EMBED_WORKERS:-$NUM_WORKERS}}"
RATE_LIMIT_SLEEP="${RATE_LIMIT_SLEEP:-$DEFAULT_RATE_SLEEP}"
CONCURRENT_PASSES="${CONCURRENT_PASSES:-1}"
SKIP_JUDGE="${SKIP_JUDGE:-0}"
SKIP_REPORT="${SKIP_REPORT:-0}"
SKIP_VIZ="${SKIP_VIZ:-0}"

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
echo "║  P2P Hypernet Evaluation — ${DATASET} (${MODE})"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Methods    : ${METHODS[*]}"
echo "║  Splits     : ${SPLITS[*]}"
echo "║  Checkpoint : ${CHECKPOINT}"
echo "║  Emb model  : ${EMB_MODEL}"
echo "║  Model (LLM): ${MODEL_OVERRIDE:-<from checkpoint>}"
echo "║  GPUs       : ${GPU_LIST[*]} (${NUM_GPUS} workers)"
echo "║  Per-char cache: ${PER_CHAR_METHODS[*]}"
echo "║  Chunk size : ${CHUNK_SIZE} (0 = auto for per-sample methods)"
echo "║  gpu_mem_util: ${GPU_MEMORY_UTILIZATION} (vLLM share of GPU memory)"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Build flat task list with per-method save_loras_dir ─────────────────
declare -a ALL_TASKS
declare -a ALL_TASKS_DESC
for METHOD in "${METHODS[@]}"; do
    # Decide save_loras_dir
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
        ALL_TASKS+=("${METHOD}|${SPLIT}|${SAVE_LORA_VALUE}")
        ALL_TASKS_DESC+=("${METHOD}/${SPLIT}")
    done
done

N_TASKS=${#ALL_TASKS[@]}
echo "  Total tasks: ${N_TASKS}, distributing across ${NUM_GPUS} GPUs (${GPU_LIST[*]})"

# Round-robin task distribution across GPUs
declare -A GPU_TASKS
for i in "${!ALL_TASKS[@]}"; do
    slot=$(( i % NUM_GPUS ))
    GPU_TASKS[$slot]+="${ALL_TASKS[$i]}"$'\n'
done

# ── Launch one worker per GPU ──────────────────────────────────────────
PIDS=()
TASK_DESCS=()

for slot in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID="${GPU_LIST[$slot]}"
    task_str="${GPU_TASKS[$slot]:-}"
    [ -z "$task_str" ] && continue

    TASK_FILE="${TASKS_DIR}/tasks_gpu${GPU_ID}.json"

    # Build task JSON
    ENTRIES="["
    FIRST=true
    GPU_DESC=""
    while IFS= read -r entry; do
        [ -z "$entry" ] && continue
        IFS='|' read -r m s save_dir <<< "$entry"
        df="${DATA_DIR}/${m}/${s}.json"
        od="${RESULTS_DIR}/${m}/${s}"
        if [ "$FIRST" = true ]; then FIRST=false; else ENTRIES+=","; fi
        if [ -n "$save_dir" ]; then
            ENTRIES+="
    {\"data\": \"${df}\", \"output_dir\": \"${od}\", \"save_loras_dir\": \"${save_dir}\"}"
        else
            ENTRIES+="
    {\"data\": \"${df}\", \"output_dir\": \"${od}\"}"
        fi
        GPU_DESC+="${m}/${s} "
    done <<< "$task_str"
    ENTRIES+="
]"
    echo "$ENTRIES" > "$TASK_FILE"

    LOG_FILE="${LOG_DIR}/predict_gpu${GPU_ID}.log"
    echo "  [GPU $GPU_ID] tasks: ${GPU_DESC}-> ${TASK_FILE}"

    MODEL_ARG=""
    if [ -n "$MODEL_OVERRIDE" ]; then
        MODEL_ARG="--model_override $MODEL_OVERRIDE"
    fi

    CUDA_VISIBLE_DEVICES="$GPU_ID" $PYTHON evaluation/predict_hypernet.py \
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
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    TASK_DESCS+=("GPU $GPU_ID (${GPU_DESC})")
done

echo ""
echo "  Launched ${#PIDS[@]} workers across ${NUM_GPUS} GPUs"
echo "  Logs: ${LOG_DIR}/"
echo ""

# ── Wait for all tasks ────────────────────────────────────────────────
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
for METHOD in "${METHODS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        pred="${RESULTS_DIR}/${METHOD}/${SPLIT}/predictions.jsonl"
        if [ -f "$pred" ]; then
            n=$(wc -l < "$pred")
            echo "║  ${METHOD}/${SPLIT}: ${n} predictions"
        else
            echo "║  ${METHOD}/${SPLIT}: (missing)"
        fi
    done
done
echo "╚══════════════════════════════════════════════════════════════╝"

# ── STEP 2: Judge + Embedding (parallel, API-based) ────────────────────
if [ "$SKIP_JUDGE" = "1" ] || [ "$SKIP_JUDGE" = "true" ]; then
    echo ""
    echo "  [SKIP_JUDGE=1] Stopping after predictions."
    exit $FAILURES
fi

if [ ! -f "$PERSONA_DATA" ]; then
    echo ""
    echo "ERROR: PERSONA_DATA not found: $PERSONA_DATA" >&2
    echo "  Cannot run judge step. Set PERSONA_DATA or use SKIP_JUDGE=1." >&2
    exit 3
fi

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
echo "  Waiting for all judge tasks ..."
JUDGE_FAIL=0
for pid in "${JUDGE_PIDS[@]}"; do
    wait "$pid" || JUDGE_FAIL=$((JUDGE_FAIL + 1))
done
STEP2_END=$(date +%s)
echo "  Scoring complete [$((STEP2_END - STEP2_START))s] (${JUDGE_FAIL} failures)"

# ── STEP 3: Report ─────────────────────────────────────────────────────
if [ "$SKIP_REPORT" = "1" ] || [ "$SKIP_REPORT" = "true" ]; then
    echo ""
    echo "  [SKIP_REPORT=1] Skipping report.py"
else
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
fi

# ── STEP 4: Visualization ──────────────────────────────────────────────
if [ "$SKIP_VIZ" = "1" ] || [ "$SKIP_VIZ" = "true" ]; then
    echo ""
    echo "  [SKIP_VIZ=1] Skipping visualize.py"
else
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
fi

END_TIME=$(date +%s)
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Hypernet P2P pipeline complete! [$((END_TIME - START_TIME))s total]"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Dataset     : ${DATASET} (${MODE})"
echo "║  Predictions : ${RESULTS_DIR}/"
echo "║  Saved LoRAs : ${LORA_DIR}/"
echo "║  Logs        : ${LOG_DIR}/"
echo "║  Summary     : ${RESULTS_DIR}/summary.json"
echo "║  Report      : ${RESULTS_DIR}/report.md"
echo "║  Figures     : ${RESULTS_DIR}/figures/"
echo "╚══════════════════════════════════════════════════════════════╝"

exit $FAILURES
