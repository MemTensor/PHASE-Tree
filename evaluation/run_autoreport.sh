#!/usr/bin/env bash
# ===========================================================================
# Generate aggregated reports for comparison and hypernet_p2p (all datasets)
# ===========================================================================
# Supports incremental updates: new methods are added without destroying
# existing results. Safe to re-run after adding new baselines.
#
# Usage:
#   bash evaluation/run_autoreport.sh                      # all datasets, all tracks
#   bash evaluation/run_autoreport.sh TheOffice            # single dataset
#   TRACKS="comparison" bash evaluation/run_autoreport.sh  # only comparison track
#   FORCE=1 bash evaluation/run_autoreport.sh              # force full recompute
#
# Environment:
#   DATASETS     (space-separated list; default: all 8)
#   TRACKS       (space-separated; default: "prompt comparison hypernet_p2p phase_tree")
#   SPLITS       (default: "random_test ood_test")
#   FORCE        (=1 to force full recomputation)
#   JUDGE_MODELS (space-separated; default: "gpt-4.1 glm-5.2 deepseek-v4-flash")
#   COMPARISON_BASELINE  (default: rag)
#   PROMPT_BASELINE      (default: m2_raw_profile)
#   HYPERNET_BASELINE    (default: m2_raw_profile)
# ===========================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$ROOT_DIR"

if [ -x "/dev/shm/phase/.venv/bin/python" ]; then
    PYTHON="/dev/shm/phase/.venv/bin/python"
elif [ -x "${HOME}/miniconda3/envs/phase/bin/python" ]; then
    PYTHON="${HOME}/miniconda3/envs/phase/bin/python"
else
    PYTHON="$(command -v python3)"
fi
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

ALL_DATASETS="RAIDEN CharacterEval SimsConv ChatHaruhi Friends HPD StarTrek_TNG TheOffice"
DATASETS="${1:-${DATASETS:-$ALL_DATASETS}}"
IFS=' ' read -ra DATASET_LIST <<< "$DATASETS"

TRACKS="${TRACKS:-prompt comparison hypernet_p2p phase_tree}"
IFS=' ' read -ra TRACK_LIST <<< "$TRACKS"

SPLITS="${SPLITS:-random_test ood_test}"
FORCE="${FORCE:-0}"
JUDGE_MODELS="${JUDGE_MODELS:-gpt-4.1 glm-5.2 deepseek-v4-flash}"
COMPARISON_BASELINE="${COMPARISON_BASELINE:-rag}"
PROMPT_BASELINE="${PROMPT_BASELINE:-m2_raw_profile}"
HYPERNET_BASELINE="${HYPERNET_BASELINE:-m2_raw_profile}"

FORCE_FLAG=""
if [ "$FORCE" = "1" ] || [ "$FORCE" = "true" ]; then
    FORCE_FLAG="--force"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Report All — Incremental Aggregated Reports"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Datasets : ${DATASET_LIST[*]}"
echo "║  Tracks   : ${TRACK_LIST[*]}"
echo "║  Splits   : ${SPLITS}"
echo "║  Judges   : ${JUDGE_MODELS}"
echo "║  Force    : ${FORCE}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

TOTAL=0
UPDATED=0
SKIPPED=0

for DATASET in "${DATASET_LIST[@]}"; do
    for TRACK in "${TRACK_LIST[@]}"; do
        RESULTS_DIR="results/${DATASET}/${TRACK}/main"
        if [ ! -d "$RESULTS_DIR" ]; then
            continue
        fi

        TOTAL=$((TOTAL + 1))

        case "$TRACK" in
            comparison)
                BASELINE="$COMPARISON_BASELINE"
                ;;
            prompt)
                BASELINE="$PROMPT_BASELINE"
                ;;
            hypernet_p2p|phase_tree)
                BASELINE="$HYPERNET_BASELINE"
                ;;
            *)
                BASELINE=""
                ;;
        esac

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  ${DATASET} / ${TRACK}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        $PYTHON evaluation/autoreport.py \
            --results_dir "$RESULTS_DIR" \
            --splits $SPLITS \
            --judge_models $JUDGE_MODELS \
            ${BASELINE:+--baseline "$BASELINE"} \
            $FORCE_FLAG

        if [ $? -eq 0 ]; then
            UPDATED=$((UPDATED + 1))
        else
            echo "  [WARN] report_all.py returned non-zero for ${DATASET}/${TRACK}"
        fi
        echo ""
    done
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Done: ${UPDATED}/${TOTAL} reports processed"
echo "╚══════════════════════════════════════════════════════════════╝"
