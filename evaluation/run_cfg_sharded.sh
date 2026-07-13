#!/usr/bin/env bash
# Shard a CFG prediction job across multiple GPUs, then merge into main output_dir.
set -uo pipefail
export PYTHONUNBUFFERED=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$ROOT_DIR"

PYTHON="/dev/shm/phase/.venv/bin/python"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PATH="/dev/shm/phase/.venv/bin:/usr/local/cuda/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

DATA="${1:?data json}"
OUT_DIR="${2:?output dir}"
GPU_LIST="${3:?gpu ids space-separated e.g. '0 3 4 5'}"
IFS=' ' read -ra GPUS <<< "$GPU_LIST"
NUM_SHARDS=${#GPUS[@]}

MODEL="${MODEL:-/mnt/lstore/model/Qwen3-32B}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.5}"
CFG_BATCH_SIZE="${CFG_BATCH_SIZE:-2}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
SEED="${SEED:-42}"

SHARD_DIR="${OUT_DIR}/_shards"
mkdir -p "$SHARD_DIR" "${OUT_DIR}/_logs"

echo "Sharding CFG job: $DATA -> $OUT_DIR across GPUs: ${GPUS[*]}"

# Build disjoint shard JSON files from remaining samples only
$PYTHON <<PY
import json, os
data = json.load(open("$DATA"))
pred = os.path.join("$OUT_DIR", "predictions.jsonl")
done = set()
if os.path.isfile(pred):
    for line in open(pred):
        if line.strip():
            done.add(json.loads(line)["question_id"])
remaining = [s for s in data if s["question_id"] not in done]
n = $NUM_SHARDS
print(f"  done={len(done)} remaining={len(remaining)} shards={n}")
for i in range(n):
    shard = remaining[i::n]  # round-robin split
    path = os.path.join("$SHARD_DIR", f"shard{i}.json")
    json.dump(shard, open(path, "w"), ensure_ascii=False)
    print(f"  shard{i}: {len(shard)} samples -> {path}")
PY

# Stop any existing single-GPU job on same output (avoid duplicate work)
pkill -f "predict_cfg.py --data ${DATA} --output_dir ${OUT_DIR}" 2>/dev/null || true
sleep 3

PIDS=()
for i in "${!GPUS[@]}"; do
    GPU="${GPUS[$i]}"
    SHARD_DATA="${SHARD_DIR}/shard${i}.json"
    SHARD_OUT="${SHARD_DIR}/gpu${GPU}"
    mkdir -p "$SHARD_OUT"
    LOG="${OUT_DIR}/_logs/shard_gpu${GPU}.log"
    echo "  [GPU ${GPU}] shard${i} -> ${SHARD_OUT}"
    CUDA_VISIBLE_DEVICES="$GPU" $PYTHON evaluation/predict_cfg.py \
        --data "$SHARD_DATA" \
        --output_dir "$SHARD_OUT" \
        --model "$MODEL" \
        --guidance_scale "$GUIDANCE_SCALE" \
        --temperature 0.3 \
        --max_tokens "$MAX_TOKENS" \
        --max_model_len "$MAX_MODEL_LEN" \
        --batch_size "$CFG_BATCH_SIZE" \
        --seed "$SEED" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo "Waiting for ${#PIDS[@]} shard workers ..."
FAIL=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || FAIL=$((FAIL+1))
done

# Merge shard predictions into main output (preserve existing + append new)
$PYTHON <<PY
import json, os, glob
main_pred = os.path.join("$OUT_DIR", "predictions.jsonl")
records = {}
if os.path.isfile(main_pred):
    for line in open(main_pred):
        if line.strip():
            r = json.loads(line)
            records[r["question_id"]] = r
for path in sorted(glob.glob(os.path.join("$SHARD_DIR", "gpu*/predictions.jsonl"))):
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            records[r["question_id"]] = r
with open(main_pred, "w", encoding="utf-8") as f:
    for r in records.values():
        f.write(json.dumps(r, ensure_ascii=False) + "\\n")
print(f"Merged {len(records)} predictions -> {main_pred}")
PY

echo "Shard merge done, failures=${FAIL}"
exit "$FAIL"
