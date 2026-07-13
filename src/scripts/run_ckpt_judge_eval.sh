#!/usr/bin/env bash
# ===========================================================================
# Evaluate all hypermod checkpoints via full inference + LLM judge scoring.
#
# Samples 5% of each dataset's m6_phase_tree/random_test, generates predictions
# for each checkpoint step (5000..40000), then runs LLM-as-Judge scoring.
#
# Usage:
#   RUN_DIR=phase_tree_models/sft/hyper_lora/<your_run_id> \
#       bash src/scripts/run_ckpt_judge_eval.sh
#   RUN_DIR=... GPU=1 bash src/scripts/run_ckpt_judge_eval.sh
#   RUN_DIR=... bash src/scripts/run_ckpt_judge_eval.sh --skip_predict
#
# Environment:
#   RUN_DIR        Training run directory (REQUIRED).
#   GPU            GPU ID (default: 0).
#   SAMPLE_FRAC    Fraction of each test split to sample (default: 0.05).
#   PYTHON         Python interpreter to use (default: "python" on PATH).
#   ROOT_DIR       Repository root (default: directory above src/).
# ===========================================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SRC_DIR}/.." && pwd)}"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}:${PYTHONPATH:-}"

if [ -z "${RUN_DIR:-}" ]; then
    echo "ERROR: RUN_DIR is required, e.g.:" >&2
    echo "  RUN_DIR=phase_tree_models/sft/hyper_lora/<your_run_id> bash $0" >&2
    exit 1
fi
GPU="${GPU:-0}"
SAMPLE_FRAC="${SAMPLE_FRAC:-0.05}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Checkpoint Judge Evaluation                                 "
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Run dir     : ${RUN_DIR}"
echo "║  GPU         : ${GPU}"
echo "║  Sample frac : ${SAMPLE_FRAC}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

$PYTHON src/scripts/eval_ckpt_judge_scores.py \
    --run_dir "$RUN_DIR" \
    --gpu "$GPU" \
    --sample_frac "$SAMPLE_FRAC" \
    "$@"
