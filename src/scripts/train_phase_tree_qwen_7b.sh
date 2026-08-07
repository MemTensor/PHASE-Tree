#!/usr/bin/env bash
# ===========================================================================
# PHASE-Tree hypermod fine-tuning launcher (Qwen2.5-7B-Instruct)
# ===========================================================================
# Warm-starts hypermod weights from a pretrained checkpoint
# (default: it_20000) and continues hyper-LoRA SFT on the 8 PHASE-Tree
# role-play datasets, m6_phase_tree variant only.
#
# Hyperparameters are tuned for warm-start fine-tuning (NOT pretraining):
#   * lr           = 5e-6     (smaller than the original pretrain lr of 2e-5
#                              to preserve the warm-start prior)
#   * warmup_frac  = 0.05     (short ramp; pretrain used 0.20)
#   * epochs       = 40000    (total training steps)
#   * val_freq     = 5000     (checkpoint saved every 5000 steps)
#   * sampler      = HierarchicalBatchSampler + sqrt_size mixture
#   * skip_val     = true     (eval suite registered but dormant by default)
#
# Usage:
#   bash PHASE-Tree/src/scripts/train_phase_tree_qwen_7b.sh
#
#   # Use a different starting checkpoint:
#   INIT_CKPT=/path/to/your/hypermod.pt \
#       bash PHASE-Tree/src/scripts/train_phase_tree_qwen_7b.sh
#
#   # Train from scratch (no warm-start):
#   INIT_CKPT="" bash PHASE-Tree/src/scripts/train_phase_tree_qwen_7b.sh
#
# Env knobs (all optional):
#   INIT_CKPT             Pretrained hypermod.pt to warm-start from. Empty = scratch.
#   CONFIG                Override training YAML.
#   TRAIN_SCRIPT          Override the train entry-point .py
#   CUDA_VISIBLE_DEVICES  GPU id(s) to use (default: 0).
#   WANDB_MODE            Default: disabled
#   LR / WARMUP / EPOCHS  Override matching --lr / --warmup_frac / --epochs.
#   PYTHON                Python interpreter (default: "python" on PATH).
#                         Make sure its PyTorch build matches the local NVIDIA
#                         driver.
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${SRC_DIR}/.." && pwd)"

INIT_CKPT="${INIT_CKPT-${REPO_DIR}/phase_tree_models/phase_tree_pretrained/hypermod.pt}"
CONFIG="${CONFIG:-${SRC_DIR}/configs/phase_tree_hyper_lora.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SRC_DIR}/scripts/train_custom_sft.py}"

LR="${LR:-5e-6}"
WARMUP="${WARMUP:-0.05}"
EPOCHS="${EPOCHS:-40000}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# Code uses cwd-relative paths for ds_kwargs.path
# (e.g. "./LongEvoRoleBench/processed/<DATASET>/m6_phase_tree/train.json"), so we MUST run
# from the PHASE-Tree repo root.  (chat_templates and tasks are resolved via
# REPO_ROOT inside the library, so they're cwd-independent.)
cd "${REPO_DIR}"

if [ -n "${INIT_CKPT}" ] && [ ! -f "${INIT_CKPT}" ]; then
    echo "ERROR: --init_hypermod_from points to a missing file: ${INIT_CKPT}" >&2
    echo "       Set INIT_CKPT=\"\" to train from scratch instead." >&2
    exit 1
fi

INIT_FLAG=()
if [ -n "${INIT_CKPT}" ]; then
    INIT_FLAG=(--init_hypermod_from="${INIT_CKPT}")
    echo ">>> Warm-starting hypermod weights from: ${INIT_CKPT}"
else
    echo ">>> Training hypermod from scratch (no warm-start checkpoint)"
fi

echo ">>> cwd            : ${REPO_DIR}"
echo ">>> CONFIG         : ${CONFIG}"
echo ">>> TRAIN_SCRIPT   : ${TRAIN_SCRIPT}"
echo ">>> lr/warmup/epoch: ${LR} / ${WARMUP} / ${EPOCHS}"

PYTHON="${PYTHON:-python}"
echo ">>> python         : ${PYTHON}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
WANDB_MODE="${WANDB_MODE:-disabled}" \
${PYTHON} "${TRAIN_SCRIPT}" \
    "${CONFIG}" \
    --model_dir=models/Qwen2.5-7B-Instruct \
    --emb_model=models/Qwen3-Embedding-4B \
    --warmup_frac="${WARMUP}" --lr="${LR}" \
    --grad_accum_steps=2 \
    --epochs="${EPOCHS}" \
    --exp_setup=hyper_lora --encoder_type=linear \
    --l2_reg_generated_w=1e-3 --label_smoothing=0.1 \
    --neftune_noise_alpha=5 --weight_decay=1e-2 \
    --val_batch_size=16 \
    --use_api_embedding=false \
    --logging_freq=50 \
    --n_tasks_per_batch=6 \
    --n_points_per_task=2 \
    --dataset_sampling_strategy=sqrt_size \
    --skip_val=true \
    --val_freq=5000 \
    --top_k_checkpoints=999 \
    --gradient_checkpointing=true \
    "${INIT_FLAG[@]}"
