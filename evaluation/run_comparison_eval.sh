#!/usr/bin/env bash
# ===========================================================================
# PHASE-Tree Comparison Baselines Pipeline (dataset-agnostic, multi-GPU)
# ===========================================================================
# Runs an external comparison baseline end-to-end:
#   STEP 1  predict   (per-method script, dispatched via $METHOD)
#   STEP 2  judge     (LLM judge + embedding similarity, shared judge.py)
#   STEP 3  report    (report.py)
#   STEP 4  visualize (visualize.py)
#
# Output goes to:
#   results/<DATASET>/comparison/main/<METHOD>/<SPLIT>/
#
# Supported methods:
#   * rag       — Retrieval-Augmented Generation (top-K demos, no profile)
#   * pag       — Profile-Augmented Generation   (top-K demos + raw profile)
#   * cfg       — Classifier-Free-Guidance decoding (HF backend, slow)
#   * steering  — Activation steering with per-character persona vectors
#   * mt_lora   — Multi-task LoRA shared across all characters
#   * oppu      — One-PEFT-per-User (per-character LoRA)
#
# Pre-trained-artifact requirements
#   steering : needs   phase_tree_models/steering/<DATASET>/persona_vectors.pt
#              (auto-built from m2_raw_profile/train.json on first run)
#   mt_lora  : needs   phase_tree_models/mt_lora/<DATASET>/   (run train_mt_lora.py)
#   oppu     : needs   phase_tree_models/oppu/<DATASET>/      (run train_oppu.py)
#
# Usage:
#   bash evaluation/run_comparison_eval.sh <DATASET> <METHOD> [<MODE>]
#
# Examples:
#   bash evaluation/run_comparison_eval.sh RAIDEN  rag
#   bash evaluation/run_comparison_eval.sh Friends pag long-term
#   METHODS="rag pag cfg steering mt_lora oppu" \
#       DATASET=RAIDEN bash evaluation/run_comparison_eval.sh
#
# Environment knobs (all optional):
#   DATASET             (default $1 or RAIDEN)
#   METHOD              (default $2 or rag; ignored if METHODS is set)
#   METHODS             (space-separated list to run sequentially)
#   MODE                (short-term | long-term; auto-detect if unset)
#   EVAL_SPLITS         (default: random_test ood_test)
#   TOP_K               (rag/pag, default 5)
#   POOL_FILE_NAME      (rag/pag, default: train.json)
#   PROFILE_DATA        (pag, default: <dataset>/m2_raw_profile/all_dialogues.json)
#   GUIDANCE_SCALE      (cfg, default 1.5)
#   ALPHA               (steering, default 4.0)
#   STEERING_VECTORS    (steering, default phase_tree_models/steering/<DATASET>/persona_vectors.pt)
#   STEERING_LAYER      (steering, default 18)
#   STEERING_NUM_PER_ROLE (steering calibration, default 50)
#   MT_LORA_PATH        (mt_lora, default phase_tree_models/mt_lora/<DATASET>)
#   OPPU_ROOT           (oppu,    default phase_tree_models/oppu/<DATASET>)
#   MAX_LORA_RANK       (mt_lora/oppu, default 16)
#   NUM_GPU_WORKERS     (default = visible GPU count, capped at 8)
#   NUM_WORKERS         (judge workers per task; short=10, long=128)
#   EMBED_WORKERS       (judge embedding workers per task; short=10, long=32)
#   RATE_LIMIT_SLEEP    (judge per-call sleep; short=0.1, long=0.0)
#   BACKEND             (default vllm; cfg/steering force hf)
#   MODEL               (default /dev/shm/Qwen2.5-7B-Instruct or models/Qwen2.5-7B-Instruct)
#   MAX_TOKENS          (default 256)
#   MAX_MODEL_LEN       (default 16384; RAG/PAG prompts can be large)
#   EMBED_BATCH_SIZE    (default 64) — used by retrieval encoder
#   EMBED_API_WORKERS   (default 8)  — used by retrieval encoder
#   RETRIEVAL_EMBED_MODEL / RETRIEVAL_EMBED_API_KEY / RETRIEVAL_EMBED_BASE_URL
#                       (retrieval embedding; defaults from .env → Qwen3-Embedding-4B local)
#                       Judge embedding uses EMBED_* (separate, stays OpenAI)
#   SKIP_JUDGE          (=1 to stop after predictions)
#   SKIP_REPORT         (=1 to skip report.py)
#   SKIP_VIZ            (=1 to skip visualize.py)
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
METHOD_ARG="${2:-${METHOD:-rag}}"
MODE_ARG="${3:-${MODE:-}}"

if [ -n "${METHODS:-}" ]; then
    IFS=' ' read -ra METHOD_LIST <<< "$METHODS"
else
    METHOD_LIST=("$METHOD_ARG")
fi

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
        DEFAULT_NUM_WORKERS=10
        DEFAULT_EMBED_WORKERS=10
        DEFAULT_RATE_SLEEP="0.1"
        ;;
    long|long-term|longterm)
        MODE="long-term"
        DEFAULT_NUM_WORKERS=128
        DEFAULT_EMBED_WORKERS=32
        DEFAULT_RATE_SLEEP="0.0"
        ;;
    *)
        echo "ERROR: unknown MODE '$MODE_ARG' (expected: short-term | long-term)" >&2
        exit 2
        ;;
esac

# Persona reference for the judge — always use m6_phase_tree (the most
# comprehensive description) so all methods are scored against the same GT.
DEFAULT_PERSONA="${DATA_DIR}/m6_phase_tree/all_dialogues.json"
PERSONA_DATA="${PERSONA_DATA:-$DEFAULT_PERSONA}"
BASELINE_METHOD="${BASELINE_METHOD:-m2_raw_profile}"

# Pool path (train split of m1_context_only — no profile bias, no test leakage)
POOL_FILE_NAME="${POOL_FILE_NAME:-train.json}"
POOL_PATH="${DATA_DIR}/m1_context_only/${POOL_FILE_NAME}"

# Profile lookup for PAG (raw profile texts)
PROFILE_DATA="${PROFILE_DATA:-${DATA_DIR}/m2_raw_profile/all_dialogues.json}"

# CFG / Steering / MT-LoRA / OPPU artifacts
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.5}"
ALPHA="${ALPHA:-4.0}"
STEERING_VECTORS="${STEERING_VECTORS:-phase_tree_models/steering/${DATASET}/persona_vectors.pt}"
STEERING_LAYER="${STEERING_LAYER:-18}"
STEERING_NUM_PER_ROLE="${STEERING_NUM_PER_ROLE:-50}"
MT_LORA_PATH="${MT_LORA_PATH:-phase_tree_models/mt_lora/${DATASET}}"
OPPU_ROOT="${OPPU_ROOT:-phase_tree_models/oppu/${DATASET}}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"

# --- Splits ---------------------------------------------------------------
SPLITS_STR="${EVAL_SPLITS:-random_test ood_test}"
IFS=' ' read -ra SPLITS <<< "$SPLITS_STR"

# --- Inference knobs ------------------------------------------------------
TOP_K="${TOP_K:-5}"

if [ -d "/dev/shm/Qwen2.5-7B-Instruct" ]; then
    DEFAULT_MODEL="/dev/shm/Qwen2.5-7B-Instruct"
else
    DEFAULT_MODEL="models/Qwen2.5-7B-Instruct"
fi
MODEL="${MODEL:-$DEFAULT_MODEL}"

MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
BACKEND="${BACKEND:-vllm}"
TEMPERATURE="${TEMPERATURE:-0.3}"
SEED="${SEED:-42}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"
EMBED_API_WORKERS="${EMBED_API_WORKERS:-8}"

# Local embedding model for retrieval (RAG/PAG). Loaded in-process, then freed.
if [ -d "/dev/shm/Qwen3-Embedding-4B" ]; then
    DEFAULT_EMBED_MODEL="/dev/shm/Qwen3-Embedding-4B"
elif [ -d "models/Qwen3-Embedding-4B" ]; then
    DEFAULT_EMBED_MODEL="models/Qwen3-Embedding-4B"
else
    DEFAULT_EMBED_MODEL=""
fi
LOCAL_EMBED_MODEL="${LOCAL_EMBED_MODEL:-$DEFAULT_EMBED_MODEL}"

# HF-only batched generation knobs (predict_cfg.py / predict_steering.py).
# Tune down if VRAM is tight — CFG keeps a parallel uncond KV-cache (~2×).
CFG_BATCH_SIZE="${CFG_BATCH_SIZE:-4}"
STEERING_BATCH_SIZE="${STEERING_BATCH_SIZE:-8}"

NUM_WORKERS="${NUM_WORKERS:-$DEFAULT_NUM_WORKERS}"
EMBED_WORKERS="${EMBED_WORKERS:-${DEFAULT_EMBED_WORKERS:-$NUM_WORKERS}}"
RATE_LIMIT_SLEEP="${RATE_LIMIT_SLEEP:-$DEFAULT_RATE_SLEEP}"
CONCURRENT_PASSES="${CONCURRENT_PASSES:-1}"

SKIP_JUDGE="${SKIP_JUDGE:-0}"
SKIP_REPORT="${SKIP_REPORT:-0}"
SKIP_VIZ="${SKIP_VIZ:-0}"

# --- GPU discovery --------------------------------------------------------
DETECTED_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
if [ "$DETECTED_GPUS" -lt 1 ]; then DETECTED_GPUS=1; fi
DEFAULT_GPU_WORKERS=$(( DETECTED_GPUS > 8 ? 8 : DETECTED_GPUS ))
NUM_GPU_WORKERS="${NUM_GPU_WORKERS:-$DEFAULT_GPU_WORKERS}"

START_TIME=$(date +%s)

# =====================================================================
# Outer loop: one full pipeline per METHOD in METHOD_LIST
# =====================================================================
for METHOD in "${METHOD_LIST[@]}"; do

    # ----------------------------------------------------------------
    # Per-method routing: select prediction script, data variant,
    # backend, and pre-flight checks.
    # ----------------------------------------------------------------
    METHOD_BACKEND="$BACKEND"
    case "$METHOD" in
        rag|pag)
            PRED_SCRIPT="predict_rag.py"
            DATA_VARIANT="m1_context_only"
            if [ ! -f "$POOL_PATH" ]; then
                echo "ERROR: retrieval pool not found: $POOL_PATH" >&2
                echo "  Set POOL_FILE_NAME or run preprocessing for ${DATASET}." >&2
                exit 3
            fi
            if [ "$METHOD" = "pag" ] && [ ! -f "$PROFILE_DATA" ]; then
                echo "ERROR: PAG requires --profile_data, not found: $PROFILE_DATA" >&2
                exit 3
            fi
            ;;
        cfg)
            PRED_SCRIPT="predict_cfg.py"
            DATA_VARIANT="m2_raw_profile"  # fair comparison: use raw profile, not phase-tree
            METHOD_BACKEND="hf"            # vLLM cannot do paired uncond pass
            ;;
        steering)
            PRED_SCRIPT="predict_steering.py"
            DATA_VARIANT="m1_context_only"  # baseline prompt at inference
            METHOD_BACKEND="hf"             # forward hooks need HF
            # Auto-build persona vectors if missing
            if [ ! -f "$STEERING_VECTORS" ]; then
                CALIB_DATA="${DATA_DIR}/m2_raw_profile/train.json"
                if [ ! -f "$CALIB_DATA" ]; then
                    echo "ERROR: steering calibration needs $CALIB_DATA but not found." >&2
                    exit 3
                fi
                mkdir -p "$(dirname "$STEERING_VECTORS")"
                CALIB_LOG="$(dirname "$STEERING_VECTORS")/_calibrate.log"
                echo ""
                echo "  [STEERING] persona_vectors.pt not found; calibrating ..."
                echo "    train_data : $CALIB_DATA"
                echo "    output     : $STEERING_VECTORS"
                echo "    layer      : $STEERING_LAYER, num_per_role : $STEERING_NUM_PER_ROLE"
                echo "    log        : $CALIB_LOG"
                CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" $PYTHON \
                    evaluation/steering_calibrate.py \
                    --train_data "$CALIB_DATA" \
                    --output "$STEERING_VECTORS" \
                    --model "$MODEL" \
                    --layer "$STEERING_LAYER" \
                    --num_per_role "$STEERING_NUM_PER_ROLE" \
                    --max_model_len "$MAX_MODEL_LEN" \
                    --seed "$SEED" \
                    > "$CALIB_LOG" 2>&1
                if [ $? -ne 0 ]; then
                    echo "ERROR: calibration failed; see $CALIB_LOG" >&2
                    exit 4
                fi
                echo "  [STEERING] calibration done."
            fi
            ;;
        mt_lora)
            PRED_SCRIPT="predict_mt_lora.py"
            DATA_VARIANT="m1_context_only"
            if [ ! -f "${MT_LORA_PATH}/adapter_model.safetensors" ] && \
               [ ! -f "${MT_LORA_PATH}/adapter_model.bin" ]; then
                echo "ERROR: MT-LoRA adapter not found at $MT_LORA_PATH" >&2
                echo "  Train it first:" >&2
                echo "    python evaluation/train_mt_lora.py \\" >&2
                echo "      --train_data ${DATA_DIR}/m1_context_only/train.json \\" >&2
                echo "      --output_dir ${MT_LORA_PATH}" >&2
                exit 3
            fi
            ;;
        oppu)
            PRED_SCRIPT="predict_oppu.py"
            DATA_VARIANT="m1_context_only"
            HAS_ANY_OPPU=0
            if [ -d "$OPPU_ROOT" ]; then
                # Look for at least one trained role
                for d in "$OPPU_ROOT"/*/; do
                    if [ -f "${d}adapter_model.safetensors" ] || \
                       [ -f "${d}adapter_model.bin" ]; then
                        HAS_ANY_OPPU=1
                        break
                    fi
                done
            fi
            if [ "$HAS_ANY_OPPU" -eq 0 ]; then
                echo "ERROR: no OPPU per-role adapters found under $OPPU_ROOT" >&2
                echo "  Train them first:" >&2
                echo "    python evaluation/train_oppu.py \\" >&2
                echo "      --train_data ${DATA_DIR}/m1_context_only/train.json \\" >&2
                echo "      --output_dir ${OPPU_ROOT}" >&2
                exit 3
            fi
            ;;
        *)
            echo "ERROR: unsupported METHOD '$METHOD'" >&2
            echo "       expected one of: rag pag cfg steering mt_lora oppu" >&2
            exit 2
            ;;
    esac

    RESULTS_DIR="results/${DATASET}/comparison/main/${METHOD}"
    LOG_DIR="${RESULTS_DIR}/_logs"
    mkdir -p "$LOG_DIR"

    METHOD_START=$(date +%s)

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Comparison Baseline — ${DATASET} / ${METHOD} (${MODE})"
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║  Method     : ${METHOD}"
    echo "║  Pred script: ${PRED_SCRIPT}"
    echo "║  Data dir   : ${DATA_DIR}/${DATA_VARIANT}/"
    echo "║  Splits     : ${SPLITS[*]}"
    case "$METHOD" in
        rag|pag)
            echo "║  Top-K      : ${TOP_K}"
            echo "║  Pool       : ${POOL_PATH}"
            [ "$METHOD" = "pag" ] && echo "║  Profile    : ${PROFILE_DATA}"
            ;;
        cfg)
            echo "║  γ (guidance): ${GUIDANCE_SCALE}"
            ;;
        steering)
            echo "║  α          : ${ALPHA}, layer=${STEERING_LAYER}"
            echo "║  Vectors    : ${STEERING_VECTORS}"
            ;;
        mt_lora)
            echo "║  Adapter    : ${MT_LORA_PATH}"
            ;;
        oppu)
            echo "║  Adapters   : ${OPPU_ROOT}/<role>/"
            ;;
    esac
    echo "║  Persona GT : ${PERSONA_DATA}"
    echo "║  Model      : ${MODEL}"
    echo "║  Backend    : ${METHOD_BACKEND} (max_model_len=${MAX_MODEL_LEN})"
    echo "║  GPU workers: ${NUM_GPU_WORKERS} (model loaded ONCE per GPU per split)"
    echo "║  API par    : judge=${NUM_WORKERS} embed=${EMBED_WORKERS} workers/task"
    echo "║  Output     : ${RESULTS_DIR}/<split>/"
    echo "╚══════════════════════════════════════════════════════════════╝"

    # =====================================================================
    # STEP 1: Predictions (one process per split, distributed across GPUs)
    # =====================================================================
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 1/4 ▸ Predictions (${METHOD}, ${#SPLITS[@]} splits)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    STEP1_START=$(date +%s)

    PREDICT_PIDS=()
    PREDICT_DESCS=()

    SPLIT_IDX=0
    for SPLIT in "${SPLITS[@]}"; do
        DATA_FILE="${DATA_DIR}/${DATA_VARIANT}/${SPLIT}.json"
        if [ ! -f "$DATA_FILE" ]; then
            echo "  [SKIP] ${DATA_FILE} not found"
            SPLIT_IDX=$((SPLIT_IDX + 1))
            continue
        fi
        OUT_DIR="${RESULTS_DIR}/${SPLIT}"
        mkdir -p "$OUT_DIR"
        LOG_FILE="${LOG_DIR}/predict_${SPLIT}.log"

        GPU_ID=$(( SPLIT_IDX % NUM_GPU_WORKERS ))
        SPLIT_IDX=$((SPLIT_IDX + 1))

        echo "  [GPU ${GPU_ID}] ${SPLIT} -> ${LOG_FILE}"

        # ----- Method-specific argument construction --------------------
        case "$METHOD" in
            rag)
                CUDA_VISIBLE_DEVICES="$GPU_ID" $PYTHON evaluation/${PRED_SCRIPT} \
                    --data "$DATA_FILE" \
                    --pool "$POOL_PATH" \
                    --output_dir "$OUT_DIR" \
                    --mode rag \
                    --top_k "$TOP_K" \
                    --model "$MODEL" \
                    --backend "$METHOD_BACKEND" \
                    --temperature "$TEMPERATURE" \
                    --max_tokens "$MAX_TOKENS" \
                    --max_model_len "$MAX_MODEL_LEN" \
                    --tensor_parallel 1 \
                    --seed "$SEED" \
                    --embed_batch_size "$EMBED_BATCH_SIZE" \
                    ${LOCAL_EMBED_MODEL:+--embed_model "$LOCAL_EMBED_MODEL"} \
                    > "$LOG_FILE" 2>&1 &
                ;;
            pag)
                CUDA_VISIBLE_DEVICES="$GPU_ID" $PYTHON evaluation/${PRED_SCRIPT} \
                    --data "$DATA_FILE" \
                    --pool "$POOL_PATH" \
                    --output_dir "$OUT_DIR" \
                    --mode pag \
                    --top_k "$TOP_K" \
                    --profile_data "$PROFILE_DATA" \
                    --model "$MODEL" \
                    --backend "$METHOD_BACKEND" \
                    --temperature "$TEMPERATURE" \
                    --max_tokens "$MAX_TOKENS" \
                    --max_model_len "$MAX_MODEL_LEN" \
                    --tensor_parallel 1 \
                    --seed "$SEED" \
                    --embed_batch_size "$EMBED_BATCH_SIZE" \
                    ${LOCAL_EMBED_MODEL:+--embed_model "$LOCAL_EMBED_MODEL"} \
                    > "$LOG_FILE" 2>&1 &
                ;;
            cfg)
                CUDA_VISIBLE_DEVICES="$GPU_ID" $PYTHON evaluation/${PRED_SCRIPT} \
                    --data "$DATA_FILE" \
                    --output_dir "$OUT_DIR" \
                    --model "$MODEL" \
                    --guidance_scale "$GUIDANCE_SCALE" \
                    --temperature "$TEMPERATURE" \
                    --max_tokens "$MAX_TOKENS" \
                    --max_model_len "$MAX_MODEL_LEN" \
                    --batch_size "$CFG_BATCH_SIZE" \
                    --seed "$SEED" \
                    > "$LOG_FILE" 2>&1 &
                ;;
            steering)
                CUDA_VISIBLE_DEVICES="$GPU_ID" $PYTHON evaluation/${PRED_SCRIPT} \
                    --data "$DATA_FILE" \
                    --vectors "$STEERING_VECTORS" \
                    --output_dir "$OUT_DIR" \
                    --model "$MODEL" \
                    --alpha "$ALPHA" \
                    --temperature "$TEMPERATURE" \
                    --max_tokens "$MAX_TOKENS" \
                    --max_model_len "$MAX_MODEL_LEN" \
                    --batch_size "$STEERING_BATCH_SIZE" \
                    --seed "$SEED" \
                    > "$LOG_FILE" 2>&1 &
                ;;
            mt_lora)
                CUDA_VISIBLE_DEVICES="$GPU_ID" $PYTHON evaluation/${PRED_SCRIPT} \
                    --data "$DATA_FILE" \
                    --lora_path "$MT_LORA_PATH" \
                    --output_dir "$OUT_DIR" \
                    --model "$MODEL" \
                    --backend "$METHOD_BACKEND" \
                    --temperature "$TEMPERATURE" \
                    --max_tokens "$MAX_TOKENS" \
                    --max_model_len "$MAX_MODEL_LEN" \
                    --max_lora_rank "$MAX_LORA_RANK" \
                    --seed "$SEED" \
                    > "$LOG_FILE" 2>&1 &
                ;;
            oppu)
                CUDA_VISIBLE_DEVICES="$GPU_ID" $PYTHON evaluation/${PRED_SCRIPT} \
                    --data "$DATA_FILE" \
                    --oppu_root "$OPPU_ROOT" \
                    --output_dir "$OUT_DIR" \
                    --model "$MODEL" \
                    --temperature "$TEMPERATURE" \
                    --max_tokens "$MAX_TOKENS" \
                    --max_model_len "$MAX_MODEL_LEN" \
                    --max_lora_rank "$MAX_LORA_RANK" \
                    --seed "$SEED" \
                    > "$LOG_FILE" 2>&1 &
                ;;
        esac

        PREDICT_PIDS+=($!)
        PREDICT_DESCS+=("${SPLIT} (GPU ${GPU_ID})")
    done

    echo ""
    echo "  Waiting for ${#PREDICT_PIDS[@]} predict workers ..."
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
    for SPLIT in "${SPLITS[@]}"; do
        pred="${RESULTS_DIR}/${SPLIT}/predictions.jsonl"
        if [ -f "$pred" ]; then
            n=$(wc -l < "$pred")
            echo "    ${SPLIT}: ${n} predictions"
        else
            echo "    ${SPLIT}: (missing)"
        fi
    done

    # =====================================================================
    # STEP 2: Scoring (judge + embed in parallel, API-based)
    # =====================================================================
    if [ "$SKIP_JUDGE" = "1" ] || [ "$SKIP_JUDGE" = "true" ]; then
        echo ""
        echo "  [SKIP_JUDGE=1] Stopping after predictions for ${METHOD}."
        continue
    fi

    if [ ! -f "$PERSONA_DATA" ]; then
        echo "" >&2
        echo "ERROR: PERSONA_DATA not found: $PERSONA_DATA" >&2
        exit 4
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 2/4 ▸ Scoring (judge=${NUM_WORKERS}, embed=${EMBED_WORKERS} workers/task)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    STEP2_START=$(date +%s)

    JUDGE_PIDS=()
    JUDGE_DESCS=()
    for SPLIT in "${SPLITS[@]}"; do
        OUT_DIR="${RESULTS_DIR}/${SPLIT}"
        if [ ! -f "${OUT_DIR}/predictions.jsonl" ]; then
            echo "  [SKIP] ${OUT_DIR}/predictions.jsonl not found"
            continue
        fi
        LOG_FILE="${LOG_DIR}/judge_${SPLIT}.log"
        echo "  ${SPLIT} -> ${LOG_FILE}"

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
        JUDGE_DESCS+=("$SPLIT")
    done

    echo ""
    echo "  Waiting for ${#JUDGE_PIDS[@]} judge tasks ..."
    JUDGE_FAIL=0
    for i in "${!JUDGE_PIDS[@]}"; do
        if ! wait "${JUDGE_PIDS[$i]}"; then
            echo "    [FAIL] ${JUDGE_DESCS[$i]}"
            JUDGE_FAIL=$((JUDGE_FAIL + 1))
        fi
    done
    STEP2_END=$(date +%s)
    echo "  Scoring complete [$((STEP2_END - STEP2_START))s] (${JUDGE_FAIL} failures)"

    # =====================================================================
    # STEP 3: Report
    # =====================================================================
    if [ "$SKIP_REPORT" = "1" ] || [ "$SKIP_REPORT" = "true" ]; then
        echo ""
        echo "  [SKIP_REPORT=1] Skipping report.py for ${METHOD}"
    else
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  STEP 3/4 ▸ Report Generation"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        REPORT_PARENT="results/${DATASET}/comparison/main"
        if ls "${REPORT_PARENT}/${METHOD}"/*/judge_scores.jsonl >/dev/null 2>&1; then
            $PYTHON evaluation/report.py \
                --results_dir "$REPORT_PARENT" \
                --experiments "$METHOD" \
                --splits "${SPLITS[@]}" \
                --baseline "$METHOD" \
                --per_character
            echo "  Report generated"
        else
            echo "  No scored splits found, skipping report."
        fi
    fi

    # =====================================================================
    # STEP 4: Visualization
    # =====================================================================
    if [ "$SKIP_VIZ" = "1" ] || [ "$SKIP_VIZ" = "true" ]; then
        echo ""
        echo "  [SKIP_VIZ=1] Skipping visualize.py for ${METHOD}"
    else
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  STEP 4/4 ▸ Visualization"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        REPORT_PARENT="results/${DATASET}/comparison/main"
        if [ -f "${REPORT_PARENT}/summary.json" ]; then
            $PYTHON evaluation/visualize.py \
                --results_dir "$REPORT_PARENT" \
                --format pdf
            echo "  Figures generated"
        else
            echo "  summary.json not found, skipping."
        fi
    fi

    METHOD_END=$(date +%s)
    echo ""
    echo "  ✓ ${METHOD} pipeline complete [$((METHOD_END - METHOD_START))s]"
done

END_TIME=$(date +%s)
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  All comparison baselines complete! [$((END_TIME - START_TIME))s total]"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Dataset : ${DATASET} (${MODE})"
echo "║  Methods : ${METHOD_LIST[*]}"
echo "║  Outputs : results/${DATASET}/comparison/main/<method>/<split>/"
echo "╚══════════════════════════════════════════════════════════════╝"
