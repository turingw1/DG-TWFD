#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ma-user/workspace/Zhengwei/DG-TWFD"
PYTHON="${ROOT}/.conda_envs/dg_twfd_a100/bin/python"
RUN_NAME="${RUN_NAME:-cifar10_nogan_dsm_10k_mb4_gb16}"
OUT_DIR="${OUT_DIR:-/cache/Zhengwei/DG-TWFD-runtime/runs/ctm_nogan_20260429/${RUN_NAME}}"
DATA_DIR="${DATA_DIR:-/cache/Zhengwei/DG-TWFD-runtime/datasets/cifar10-cond/train}"
TEACHER="${TEACHER:-/cache/Zhengwei/DG-TWFD-runtime/.torch/dnnlib/downloads/8a0cade18e36ff627fc6c7a9dce1bd11_https___nvlabs-fi-cdn.nvidia.com_edm_pretrained_edm-cifar10-32x32-cond-vp.pkl}"
REF="${REF:-/cache/Zhengwei/DG-TWFD-runtime/checkpoints/baselines/ctm_cifar10/ctm-cifar10/cifar10-32x32.npz}"
TOTAL_STEPS="${TOTAL_STEPS:-10000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
MICROBATCH="${MICROBATCH:-4}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-5000}"
EVAL_BATCH="${EVAL_BATCH:-500}"
VRAM_LIMIT_MB="${VRAM_LIMIT_MB:-50000}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

EXTRA_ARGS=()
if [ -n "${RESUME_CHECKPOINT}" ]; then
  EXTRA_ARGS+=(--resume_checkpoint="${RESUME_CHECKPOINT}")
fi

mkdir -p "${OUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMPI_COMM_WORLD_SIZE=1
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export HOSTNAME="${HOSTNAME:-localhost}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONPATH="${ROOT}/scripts/baselines/compat:${PYTHONPATH:-}"

cd "${ROOT}/refs/ctm-cifar10"

"${PYTHON}" cm_train.py \
  --save_interval=1000 \
  --save_check_period=100000000 \
  --check_dm_performance=False \
  --eval_fid=False \
  --eval_similarity=False \
  --eval_interval=-1 \
  --eval_num_samples="${EVAL_NUM_SAMPLES}" \
  --eval_batch="${EVAL_BATCH}" \
  --sampling_batch=64 \
  --eval_large_nfe=False \
  --compute_ema_fids=False \
  --intermediate_samples=False \
  --total_training_steps="${TOTAL_STEPS}" \
  --log_interval=100 \
  --microbatch="${MICROBATCH}" \
  --global_batch_size="${GLOBAL_BATCH_SIZE}" \
  --ema_rate=0.999 \
  --class_cond=True \
  --num_classes=10 \
  --num_workers=4 \
  --teacher_model_path="${TEACHER}" \
  --data_dir="${DATA_DIR}" \
  --ref_path="${REF}" \
  --out_dir="${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"
