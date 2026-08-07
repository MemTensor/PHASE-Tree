#!/usr/bin/env bash
# ===========================================================================
# Master launcher: Gemma-4-E4B-it explicit inference (prompt + comparison)
# ===========================================================================
# Runs all prompt/main methods and comparison rag/pag/cfg across 8 datasets,
# 8 GPUs, phase venv. Outputs under results/*/prompt|comparison/gemma_4_e4b_it/
#
# Usage:
#   bash evaluation/run_gemma4_e4b_it_all.sh
#   SKIP_PROMPT=1 bash evaluation/run_gemma4_e4b_it_all.sh      # comparison only
#   SKIP_COMPARISON=1 bash evaluation/run_gemma4_e4b_it_all.sh  # prompt only
#   NUM_SAMPLES=8 bash evaluation/run_gemma4_e4b_it_all.sh    # smoke test
# ===========================================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$ROOT_DIR"

MASTER_LOG="results/_gemma4_e4b_it_master.log"
mkdir -p results

{
    echo "=== Gemma-4-E4B-it master launcher start $(date -Iseconds) ==="
    echo "ROOT: $ROOT_DIR"
    echo "SKIP_PROMPT=${SKIP_PROMPT:-0} SKIP_COMPARISON=${SKIP_COMPARISON:-0}"
    echo ""

    FAIL=0

    if [ "${SKIP_PROMPT:-0}" != "1" ]; then
        echo ">>> Phase 1: Prompt predictions (m1-m6) ..."
        if ! bash evaluation/run_prompt_predict_gemma4_e4b_it.sh; then
            echo ">>> Phase 1 FAILED (exit $?)"
            FAIL=$((FAIL + 1))
        else
            echo ">>> Phase 1 OK"
        fi
        echo ""
    else
        echo ">>> Phase 1: SKIPPED (SKIP_PROMPT=1)"
        echo ""
    fi

    if [ "${SKIP_COMPARISON:-0}" != "1" ]; then
        echo ">>> Phase 2: Comparison predictions (rag/pag/cfg) ..."
        if ! bash evaluation/run_comparison_predict_gemma4_e4b_it.sh; then
            echo ">>> Phase 2 FAILED (exit $?)"
            FAIL=$((FAIL + 1))
        else
            echo ">>> Phase 2 OK"
        fi
        echo ""
    else
        echo ">>> Phase 2: SKIPPED (SKIP_COMPARISON=1)"
        echo ""
    fi

    echo "=== Gemma-4-E4B-it master launcher done $(date -Iseconds), failures=${FAIL} ==="
    exit "$FAIL"
} 2>&1 | tee -a "$MASTER_LOG"
