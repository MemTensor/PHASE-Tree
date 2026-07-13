#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SFT teacher-forced loss on all PHASE-Tree eval tasks (random_test + ood_test
# from phase_tree_hyper_lora.yaml when run args.yaml has empty eval_ds_info).
# Same math as training: sft_trainer.validate + get_loss_batch.
#
# After EACH 10k checkpoint, results go to:
#   <run_dir>/eval_ckpt_val_loss/it_<step>.json
#   <run_dir>/eval_ckpt_val_loss/metrics.csv   (rebuilt each full pass over steps)
#
# Teacher-forced CE loss only (no generation). For end-to-end generative
# evaluation, see run_hypernet_p2p_eval.sh.
#
# Usage:
#   bash evaluation/run_phase_tree_ckpt_val_loss.sh
#   RUNS="phase_tree_models/.../runA" GPU=1 bash evaluation/run_phase_tree_ckpt_val_loss.sh
#   OUT_CSV=/tmp/all_runs.csv bash evaluation/run_phase_tree_ckpt_val_loss.sh   # optional combined CSV
#   EVAL_SUFFIX=random_test bash ...   # only *_random_test tasks (optional)
# -----------------------------------------------------------------------------
set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

GPU="${GPU:-0}"
STEP_START="${STEP_START:-10000}"
STEP_STRIDE="${STEP_STRIDE:-10000}"
STEP_END="${STEP_END:-40000}"
BASE_CONFIG="${BASE_CONFIG:-${ROOT}/src/configs/phase_tree_hyper_lora.yaml}"

RUNS="${RUNS:-phase_tree_models/sft/hyper_lora/20260512-202647_ipM66823 phase_tree_models/sft/hyper_lora/20260513-110904_RontK04z}"

declare -a CMD=( "$PYTHON" src/scripts/eval_hypermod_ckpt_val_loss.py
  --step_start "$STEP_START" --step_stride "$STEP_STRIDE" --step_end "$STEP_END"
  --base_config "$BASE_CONFIG"
  --device "cuda:${GPU}"
)

if [[ -n "${OUT_CSV:-}" ]]; then
  CMD+=( --out_csv "$OUT_CSV" )
fi

if [[ -n "${EVAL_SUFFIX:-}" ]]; then
  CMD+=( --eval_suffix "$EVAL_SUFFIX" )
fi

for d in $RUNS; do
  CMD+=( --run_dir "$d" )
done

echo "Running: ${CMD[*]}"
CUDA_VISIBLE_DEVICES="$GPU" exec "${CMD[@]}"
