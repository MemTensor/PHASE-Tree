#!/usr/bin/env bash
# ===========================================================================
# Run judge rescoring across ALL datasets for a given results track.
# ===========================================================================
# Tracks: prompt | comparison | hypernet_p2p | phase_tree
#
# Usage:
#   TRACK=comparison JUDGE_MODEL=glm-5.2 NUM_WORKERS=128 bash evaluation/run_judge_rescore_batch.sh
#   RESULTS_TAG=qwen3_32b JUDGE_MODEL=gpt-4.1 NUM_WORKERS=512 bash evaluation/run_judge_rescore_batch.sh
#   TRACK=hypernet_p2p PARALLEL_DATASETS=3 bash evaluation/run_judge_rescore_batch.sh
#   (hypernet_p2p defaults: DATASETS=CharacterEval METHODS=m2_raw_profile)
#   (phase_tree defaults:   DATASETS=CharacterEval METHODS=m6_phase_tree)
#
# Environment knobs:
#   TRACK              (default: prompt)
#   RESULTS_TAG        (default: main)
#   SAMPLE_MANIFEST_DIR (unset: gpt-4.1→25% subsample, else full; empty=full)
#   DATASETS           (override dataset list; default: auto-discover)
#   METHODS            (override method list for all datasets in this batch)
#   SKIP_DATASETS      (space-separated datasets to exclude)
#   PARALLEL_DATASETS  (default: 1; set 3 for 3 datasets in parallel)
#   NUM_WORKERS        (default: 128)
#   JUDGE_MODEL        (default: glm-5.2)
#   DISABLE_THINKING   (default: 1)
#   MAX_ROUNDS         (default: 10)
#   ALLOW_INCOMPLETE   (=1 to skip stubborn splits/datasets and continue)
#   INCOMPLETE_MIN_PCT (default: 99)
#   SPLIT_MAX_ROUNDS   (per-split retries; default: MAX_ROUNDS)
# ===========================================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$ROOT_DIR"

if [ -x "/dev/shm/phase/.venv/bin/python" ]; then
    PYTHON="/dev/shm/phase/.venv/bin/python"
else
    PYTHON="$(command -v python3 || command -v python)"
fi

TRACK="${TRACK:-prompt}"
RESULTS_TAG="${RESULTS_TAG:-main}"
JUDGE_MODEL="${JUDGE_MODEL:-glm-5.2}"
if [ -z "${SAMPLE_MANIFEST_DIR+x}" ]; then
    if [ "$JUDGE_MODEL" = "gpt-4.1" ]; then
        SAMPLE_MANIFEST_DIR="results/_judge_samples_25pct"
    else
        SAMPLE_MANIFEST_DIR=""
    fi
fi
NUM_WORKERS="${NUM_WORKERS:-128}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
PARALLEL_DATASETS="${PARALLEL_DATASETS:-1}"
SKIP_DATASETS="${SKIP_DATASETS:-}"
ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE:-0}"
INCOMPLETE_MIN_PCT="${INCOMPLETE_MIN_PCT:-99}"

# hypernet_p2p / phase_tree: scoped CharacterEval subsets unless overridden
case "$TRACK" in
    hypernet_p2p)
        DATASETS="${DATASETS:-CharacterEval}"
        METHODS="${METHODS:-m2_raw_profile}"
        ;;
    phase_tree)
        DATASETS="${DATASETS:-CharacterEval}"
        METHODS="${METHODS:-m6_phase_tree}"
        ;;
    comparison)
        if [ "$RESULTS_TAG" != "main" ] && [ -z "${METHODS:-}" ]; then
            METHODS="rag pag cfg"
        fi
        ;;
esac

SAFE_MODEL="${JUDGE_MODEL//[^a-zA-Z0-9._-]/_}"
if [ "$JUDGE_MODEL" = "gpt-4.1" ]; then
    JUDGE_FILE="judge_scores.jsonl"
else
    JUDGE_FILE="judge_scores_${SAFE_MODEL}.jsonl"
fi

track_root() {
    echo "results/${1}/${TRACK}/${RESULTS_TAG}"
}

expected_count_for_pred() {
    local pred_path="$1"
    local pred_n="$2"
    if [ -z "${SAMPLE_MANIFEST_DIR:-}" ]; then
        echo "$pred_n"
        return
    fi
    local ds split sample_file
    ds="$(echo "$pred_path" | sed -n 's#results/\([^/]*\)/.*#\1#p')"
    split="$(basename "$(dirname "$pred_path")")"
    sample_file="${SAMPLE_MANIFEST_DIR}/${ds}/${split}.json"
    if [ -f "$sample_file" ]; then
        $PYTHON -c "import json; print(len(json.load(open('$sample_file'))))"
    else
        echo "$pred_n"
    fi
}

discover_datasets() {
    local ds path
    for path in results/*/"${TRACK}/${RESULTS_TAG}"; do
        [ -d "$path" ] || continue
        ds="$(basename "$(dirname "$(dirname "$path")")")"
        if [ -n "$SKIP_DATASETS" ] && [[ " ${SKIP_DATASETS} " == *" ${ds} "* ]]; then
            continue
        fi
        if compgen -G "${path}/*/predictions.jsonl" > /dev/null \
           || compgen -G "${path}/*/*/predictions.jsonl" > /dev/null; then
            echo "$ds"
        fi
    done
}

dataset_complete() {
    local dataset="$1"
    local pred glm pred_n glm_n root method_filter=() expected_n
    root="$(track_root "$dataset")"
    if [ -n "${METHODS:-}" ]; then
        IFS=' ' read -ra method_filter <<< "$METHODS"
    fi
    while IFS= read -r pred; do
        if [ ${#method_filter[@]} -gt 0 ]; then
            local matched=0 m
            for m in "${method_filter[@]}"; do
                if [[ "$pred" == *"/${m}/"* ]]; then
                    matched=1
                    break
                fi
            done
            [ "$matched" -eq 1 ] || continue
        fi
        glm="${pred/predictions.jsonl/${JUDGE_FILE}}"
        pred_n=$(wc -l < "$pred")
        expected_n="$(expected_count_for_pred "$pred" "$pred_n")"
        glm_n=0
        [ -f "$glm" ] && glm_n=$(wc -l < "$glm")
        if [ "$glm_n" -ge "$expected_n" ]; then
            continue
        fi
        if [ "$ALLOW_INCOMPLETE" = "1" ] || [ "$ALLOW_INCOMPLETE" = "true" ]; then
            if [ "$expected_n" -gt 0 ] && [ $((glm_n * 100)) -ge $((expected_n * INCOMPLETE_MIN_PCT)) ]; then
                continue
            fi
        fi
        return 1
    done < <(find "$root" -name predictions.jsonl | sort)
    return 0
}

resolve_methods() {
    local dataset="$1"
    local root methods=() entry m
    root="$(track_root "$dataset")"
    for entry in "${root}"/*; do
        [ -d "$entry" ] || continue
        m="$(basename "$entry")"
        [[ "$m" == _* ]] && continue
        if compgen -G "${entry}/*/predictions.jsonl" > /dev/null; then
            methods+=("$m")
        fi
    done
    if [ ${#methods[@]} -gt 0 ]; then
        echo "${methods[*]}"
        return
    fi
    if [ "$TRACK" = "comparison" ] && [ "$RESULTS_TAG" != "main" ]; then
        echo "rag pag cfg"
        return
    fi
    local data_dir="LongEvoRoleBench/processed/${dataset}"
    if [ -d "${data_dir}/m5_dynamic_tree" ]; then
        echo "m1_context_only m2_raw_profile m3_naive_rewrite m4_static_tree m5_dynamic_tree m6_phase_tree"
    else
        echo "m1_context_only m2_raw_profile m3_naive_rewrite m4_static_tree m6_phase_tree"
    fi
}

if [ -n "${DATASETS:-}" ]; then
    IFS=' ' read -ra DATASET_LIST <<< "$DATASETS"
else
    mapfile -t DATASET_LIST < <(discover_datasets | sort -u)
fi

PENDING=()
for dataset in "${DATASET_LIST[@]}"; do
    if dataset_complete "$dataset"; then
        echo "  [DONE] ${dataset}/${TRACK}/${RESULTS_TAG} — all splits scored"
    else
        PENDING+=("$dataset")
    fi
done

if [ ${#PENDING[@]} -eq 0 ]; then
    echo "All datasets already scored for track=${TRACK} tag=${RESULTS_TAG} model=${JUDGE_MODEL}."
    exit 0
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Judge Rescore Pipeline — track=${TRACK} tag=${RESULTS_TAG}"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Pending       : ${PENDING[*]}"
if [ -n "${METHODS:-}" ]; then
    echo "║  Methods       : ${METHODS}"
fi
echo "║  Parallel ds   : ${PARALLEL_DATASETS}"
echo "║  Workers/ds    : ${NUM_WORKERS}"
echo "║  Judge model   : ${JUDGE_MODEL}"
echo "║  Output file   : ${JUDGE_FILE}"
if [ -n "${SAMPLE_MANIFEST_DIR:-}" ]; then
    echo "║  Subsample     : 25% fixed (${SAMPLE_MANIFEST_DIR})"
else
    echo "║  Subsample     : full"
fi
echo "║  Thinking      : $([ "${DISABLE_THINKING}" = "1" ] && echo OFF || echo ON)"
echo "╚══════════════════════════════════════════════════════════════╝"

run_dataset() {
    local dataset="$1"
    local methods
    if [ -n "${METHODS:-}" ]; then
        methods="$METHODS"
    else
        methods="$(resolve_methods "$dataset")"
    fi
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  START ${dataset}/${TRACK}/${RESULTS_TAG}  (methods: ${methods})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    TRACK="$TRACK" \
    RESULTS_TAG="$RESULTS_TAG" \
    SAMPLE_MANIFEST_DIR="$SAMPLE_MANIFEST_DIR" \
    METHODS="$methods" \
    NUM_WORKERS="$NUM_WORKERS" \
    PARALLEL_TASKS=1 \
    JUDGE_MODEL="$JUDGE_MODEL" \
    DISABLE_THINKING="$DISABLE_THINKING" \
    MAX_ROUNDS="${MAX_ROUNDS:-10}" \
    ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE}" \
    INCOMPLETE_MIN_PCT="${INCOMPLETE_MIN_PCT}" \
    SPLIT_MAX_ROUNDS="${SPLIT_MAX_ROUNDS:-1}" \
        bash evaluation/run_judge_rescore.sh "$dataset"
    local ec=$?
    if [ $ec -eq 0 ]; then
        echo "  ✓ ${dataset}/${TRACK}/${RESULTS_TAG} complete → next dataset"
    else
        echo "  ✗ ${dataset}/${TRACK}/${RESULTS_TAG} failed (exit $ec)" >&2
    fi
    return $ec
}

BATCH_START=$(date +%s)
TOTAL_FAIL=0
MAX_ROUNDS="${MAX_ROUNDS:-10}"
ROUND=1
WORK_LIST=("${PENDING[@]}")

while [ ${#WORK_LIST[@]} -gt 0 ] && [ "$ROUND" -le "$MAX_ROUNDS" ]; do
    if [ "$ROUND" -gt 1 ]; then
        echo ""
        echo "  ⟳ Batch retry round ${ROUND}/${MAX_ROUNDS} (${#WORK_LIST[@]} datasets remaining)"
    fi
    ROUND_FAIL=0
    IDX=0
    RUNNING=0
    NEXT_WORK=()

    for dataset in "${WORK_LIST[@]}"; do
        run_dataset "$dataset" &
        RUNNING=$((RUNNING + 1))
        IDX=$((IDX + 1))

        if [ "$RUNNING" -ge "$PARALLEL_DATASETS" ] || [ "$IDX" -eq "${#WORK_LIST[@]}" ]; then
            for pid in $(jobs -p); do
                wait "$pid" || ROUND_FAIL=$((ROUND_FAIL + 1))
            done
            RUNNING=0
        fi
    done

    for dataset in "${WORK_LIST[@]}"; do
        if dataset_complete "$dataset"; then
            echo "  ✓ ${dataset}/${TRACK}/${RESULTS_TAG} fully complete"
        else
            echo "  ⧗ ${dataset}/${TRACK}/${RESULTS_TAG} still incomplete — will retry"
            NEXT_WORK+=("$dataset")
        fi
    done

    WORK_LIST=("${NEXT_WORK[@]}")
    TOTAL_FAIL=$ROUND_FAIL
    ROUND=$((ROUND + 1))
done

if [ ${#WORK_LIST[@]} -gt 0 ]; then
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
fi

BATCH_END=$(date +%s)
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Track ${TRACK}/${RESULTS_TAG} complete [$((BATCH_END - BATCH_START))s] failures=${TOTAL_FAIL}"
echo "╚══════════════════════════════════════════════════════════════╝"

for dataset in "${PENDING[@]}"; do
    if dataset_complete "$dataset"; then
        echo "  ✓ ${dataset}"
    else
        echo "  ✗ ${dataset} (incomplete)"
    fi
done

if [ "$TOTAL_FAIL" -gt 0 ]; then
    if [ "$ALLOW_INCOMPLETE" = "1" ] || [ "$ALLOW_INCOMPLETE" = "true" ]; then
        echo "  ALLOW_INCOMPLETE=1 — batch exiting 0 with deferred datasets"
        exit 0
    fi
    exit 1
fi
