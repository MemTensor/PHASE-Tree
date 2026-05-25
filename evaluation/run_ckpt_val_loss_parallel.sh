#!/usr/bin/env bash
# ===========================================================================
# Parallel Checkpoint Validation-Loss Evaluation (multi-GPU)
# ===========================================================================
# Evaluates multiple hypernetwork SFT checkpoints (at specific iteration
# steps) in parallel across available GPUs.  Each (run, step) pair is
# dispatched to a separate GPU; results are merged into a per-run
# metrics.csv at the end.
#
# Usage:
#   bash evaluation/run_ckpt_val_loss_parallel.sh
#
# Environment overrides:
#   EVAL_STEPS   (space-separated step list; default: 10000 20000 30000 40000)
#   EVAL_GPUS    (space-separated GPU IDs; default: all visible GPUs)
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="/dev/shm/phase/.venv/bin/python"
EVAL_PY="src/scripts/eval_hypermod_ckpt_val_loss.py"
BASE_CONFIG="src/configs/phase_tree_hyper_lora.yaml"
LOG_DIR="phase_tree_models/sft/hyper_lora"

RUNS=(
  "phase_tree_models/sft/hyper_lora/20260512-202647_ipM66823"
  "phase_tree_models/sft/hyper_lora/20260513-110904_RontK04z"
)
STEPS=(${EVAL_STEPS:-10000 20000 30000 40000})
if [ -n "${EVAL_GPUS:-}" ]; then
  GPUS=(${EVAL_GPUS})
else
  GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' '))
  if [ ${#GPUS[@]} -eq 0 ]; then GPUS=(0); fi
fi

# Build flat job list: "run|step"
declare -a JOBS=()
for run in "${RUNS[@]}"; do
  for step in "${STEPS[@]}"; do
    JOBS+=("${run}|${step}")
  done
done

echo "=== Launching ${#JOBS[@]} jobs across GPUs: ${GPUS[*]} ==="

pids=()
gpu_for_pid=()
job_desc=()

for i in "${!JOBS[@]}"; do
  IFS='|' read -r run step <<< "${JOBS[$i]}"
  gpu_idx=$(( i % ${#GPUS[@]} ))
  gpu="${GPUS[$gpu_idx]}"
  run_name=$(basename "$run")
  log="${LOG_DIR}/eval_${run_name}_it${step}.log"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    exec "$PYTHON" "$EVAL_PY" \
      --run_dir "$run" \
      --step_start "$step" --step_end "$step" --step_stride 1 \
      --base_config "$BASE_CONFIG" \
      --no_metrics_csv \
      --device cuda:0
  ) >"$log" 2>&1 &

  pids+=($!)
  gpu_for_pid+=("$gpu")
  job_desc+=("${run_name}/it_${step}")
  echo "[$(date +%H:%M:%S)] pid=$!  gpu=$gpu  ${run_name}  step=$step"

  sleep 2
done

echo ""
echo "=== Waiting for ${#pids[@]} jobs... ==="

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[$(date +%H:%M:%S)] ✓  ${job_desc[$i]}  (gpu ${gpu_for_pid[$i]})"
  else
    echo "[$(date +%H:%M:%S)] ✗  ${job_desc[$i]}  FAILED (gpu ${gpu_for_pid[$i]})"
    fail=$((fail + 1))
  fi
done

# ---- Merge per-step JSONs into metrics.csv per run ----
echo ""
echo "=== Merging results ==="
for run in "${RUNS[@]}"; do
  out_dir="${run}/eval_ckpt_val_loss"
  csv="${out_dir}/metrics.csv"
  [ -d "$out_dir" ] || continue

  "$PYTHON" -c "
import json, csv, glob, os, sys

out_dir = '$out_dir'
csv_path = '$csv'
json_files = sorted(glob.glob(os.path.join(out_dir, 'it_*.json')))
if not json_files:
    print(f'  No JSON files in {out_dir}')
    sys.exit(0)

fieldnames = ['step', 'split', 'sft_loss', 'per_token_acc', 'entropy']
rows = []
for jf in json_files:
    data = json.load(open(jf))
    step = data['step']
    for split_name, metrics in data['splits'].items():
        rows.append({
            'step': step,
            'split': split_name,
            'sft_loss': metrics.get('sft_loss'),
            'per_token_acc': metrics.get('per_token_acc'),
            'entropy': metrics.get('entropy'),
        })

rows.sort(key=lambda r: (r['step'], r['split']))
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print(f'  Wrote {csv_path}  ({len(rows)} rows from {len(json_files)} files)')
"
done

echo ""
if [ "$fail" -eq 0 ]; then
  echo "=== ALL DONE (${#pids[@]} jobs succeeded) ==="
else
  echo "=== DONE with $fail failure(s) ==="
  exit 1
fi
