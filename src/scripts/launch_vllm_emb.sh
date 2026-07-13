#!/usr/bin/env bash
# ===========================================================================
# Launch a vLLM OpenAI-compatible embedding server on localhost:8000
# serving Qwen3-Embedding-4B for the PHASE-Tree training pipeline.
#
# This server is OPTIONAL.  The default training recipe in
# train_phase_tree_qwen_7b.sh uses the local embedding model directly
# (--use_api_embedding=false) and does NOT need this server.  Start it only
# when you launch the trainer (or another consumer) with
# --use_api_embedding=true pointing at http://localhost:8000.
#
# Usage:
#   bash PHASE-Tree/src/scripts/launch_vllm_emb.sh                       # GPUs 0,1, port 8000
#   GPUS=4,5 PORT=8003 bash PHASE-Tree/src/scripts/launch_vllm_emb.sh    # custom GPUs / port
#   TP_SIZE=1 GPUS=0 bash PHASE-Tree/src/scripts/launch_vllm_emb.sh      # single-GPU
#   MODEL=models/Qwen3-Embedding-4B bash ...                             # local-path model
#
# Env knobs:
#   GPUS               default "0,1" (comma list of GPU ids)
#   TP_SIZE            default 2     (tensor-parallel size; should match #GPUs)
#   PORT               default 8000
#   MODEL              default models/Qwen3-Embedding-4B (local symlink)
#   API_KEY            default EMPTY
#   MAX_MODEL_LEN      default 32768 (Qwen3-Embedding-4B native limit)
#   PYTHON             Python interpreter that has vLLM installed (default:
#                      "python" on PATH).  Make sure the PyTorch build inside
#                      this interpreter matches the local NVIDIA driver.
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${SRC_DIR}/.." && pwd)"

GPUS="${GPUS:-0,1}"
TP_SIZE="${TP_SIZE:-2}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-models/Qwen3-Embedding-4B}"
API_KEY="${API_KEY:-EMPTY}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
PYTHON="${PYTHON:-python}"

cd "${REPO_DIR}"

# Sanity: model directory must exist (even if it's a symlink to a real path).
if [ ! -d "${MODEL}" ] && [[ "${MODEL}" != Qwen/* && "${MODEL}" != */* ]]; then
    echo "ERROR: MODEL path '${MODEL}' is neither a local directory nor an HF id." >&2
    exit 1
fi

echo ">>> Launching vLLM embedding server"
echo "    cwd            : ${REPO_DIR}"
echo "    model          : ${MODEL}"
echo "    port           : ${PORT}"
echo "    GPUs           : ${GPUS}  (tensor-parallel-size=${TP_SIZE})"
echo "    max_model_len  : ${MAX_MODEL_LEN}"

# Pick the right --task / --runner flag for the installed vLLM version:
#   * vLLM <  0.10  uses `--task embed`            (e.g. project venv, 0.5.4)
#   * vLLM >= 0.20  uses `--runner pooling --convert embed`
VLLM_VER="$(${PYTHON} -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 0)"
VLLM_MAJOR_MINOR="${VLLM_VER%%.*}.${VLLM_VER#*.}"
VLLM_MAJOR_MINOR="${VLLM_MAJOR_MINOR%%.*}"
case "${VLLM_VER}" in
    0.[0-9].*|0.10.*|0.[0-9])  EMB_FLAGS=(--task embed) ;;
    *)                          EMB_FLAGS=(--runner pooling --convert embed) ;;
esac
echo "    vLLM           : ${VLLM_VER}  (flags: ${EMB_FLAGS[*]})"

CUDA_VISIBLE_DEVICES="${GPUS}" \
${PYTHON} -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    "${EMB_FLAGS[@]}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --port "${PORT}" \
    --api-key "${API_KEY}" \
    --max-model-len "${MAX_MODEL_LEN}"
