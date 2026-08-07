#!/usr/bin/env bash
# Master launcher: Qwen3-235B-A22B explicit inference (prompt + comparison)
set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$ROOT_DIR"

MASTER_LOG="results/_qwen3_235b_a22b_master.log"
mkdir -p results

{
    echo "=== Qwen3-235B-A22B master launcher start $(date -Iseconds) ==="
    echo "SKIP_PROMPT=${SKIP_PROMPT:-0} SKIP_COMPARISON=${SKIP_COMPARISON:-0}"
    FAIL=0

    if [ "${SKIP_PROMPT:-0}" != "1" ]; then
        echo ">>> Phase 1: Prompt predictions ..."
        if ! bash evaluation/run_prompt_predict_qwen3_235b_a22b.sh; then
            FAIL=$((FAIL + 1))
        fi
    fi

    if [ "${SKIP_COMPARISON:-0}" != "1" ]; then
        echo ">>> Phase 2: Comparison predictions ..."
        if ! bash evaluation/run_comparison_predict_qwen3_235b_a22b.sh; then
            FAIL=$((FAIL + 1))
        fi
    fi

    echo "=== Qwen3-235B-A22B done $(date -Iseconds), failures=${FAIL} ==="
    exit "$FAIL"
} 2>&1 | tee -a "$MASTER_LOG"
