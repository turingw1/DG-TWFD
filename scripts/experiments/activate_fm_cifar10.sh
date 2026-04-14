#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced, not executed." >&2
  echo "Example: source scripts/experiments/activate_fm_cifar10.sh baseline v1" >&2
  exit 1
fi

variant="${1:-baseline}"
tag="${2:-v1}"

case "${variant}" in
  baseline)
    export FM_CONFIG="configs/experiment/fm_cifar10_baseline.yaml"
    exp_prefix="fm_cifar10_baseline"
    ;;
  map_branch)
    export FM_CONFIG="configs/experiment/fm_cifar10_map_branch.yaml"
    exp_prefix="fm_cifar10_map_branch"
    ;;
  map_branch_quick)
    export FM_CONFIG="configs/experiment/fm_cifar10_map_branch_quick.yaml"
    exp_prefix="fm_cifar10_map_branch_quick"
    ;;
  map_branch_timewarp_probe)
    export FM_CONFIG="configs/experiment/fm_cifar10_map_branch_timewarp_probe.yaml"
    exp_prefix="fm_cifar10_map_branch_timewarp_probe"
    ;;
  map_branch_timewarp_smoke)
    export FM_CONFIG="configs/experiment/fm_cifar10_map_branch_timewarp_smoke.yaml"
    exp_prefix="fm_cifar10_map_branch_timewarp_smoke"
    ;;
  stable)
    export FM_CONFIG="configs/experiment/fm_cifar10_stable.yaml"
    exp_prefix="fm_cifar10_stable"
    ;;
  *)
    candidate="configs/experiment/${variant}.yaml"
    if [[ -f "${candidate}" ]]; then
      export FM_CONFIG="${candidate}"
      exp_prefix="${variant}"
    else
      echo "Unknown FM variant: ${variant}" >&2
      echo "Expected one of: baseline map_branch map_branch_quick map_branch_timewarp_probe map_branch_timewarp_smoke stable or any config stem under configs/experiment/" >&2
      return 1
    fi
    ;;
esac

export PROJ="${PROJ:-/data2/yl7622/Zhengwei/DG-TWFD}"
export ENV_NAME="${ENV_NAME:-consistency}"
export DATA_ROOT="${DATA_ROOT:-${PROJ}/datasets}"
export RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJ}/eval}"
export REF_ROOT="${REF_ROOT:-${PROJ}/refs}"
export TRAJ_ROOT="${TRAJ_ROOT:-${PROJ}/teacher_traj/cifar10_ddpm128_p33}"
export IMAGENET_RAW_ROOT="${IMAGENET_RAW_ROOT:-${DATA_ROOT}/imagenet_raw}"
export IMAGENET64_PREPROCESSED="${IMAGENET64_PREPROCESSED:-${DATA_ROOT}/imagenet64}"
export IMAGENET64_REFERENCE_NPZ="${IMAGENET64_REFERENCE_NPZ:-${REF_ROOT}/VIRTUAL_imagenet64_labeled.npz}"
export OFFICIAL_REFERENCE_NPZ="${OFFICIAL_REFERENCE_NPZ:-}"
export IMAGENET64_TEACHER_CKPT="${IMAGENET64_TEACHER_CKPT:-checkpoints/teachers/edm_imagenet64_ema.pt}"
export HF_HOME="${HF_HOME:-${PROJ}/.hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
unset HF_ENDPOINT
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
unset DGFM_TORCH_FIDELITY_MIRROR_PREFIX
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export EXP_VARIANT="${variant}"
export EXP_TAG="${tag}"
export EXP_NAME="${exp_prefix}_${tag}"
export FM_EXP="${exp_prefix}_${tag}"
export EXP_SOURCE="${FM_CONFIG}"
export RUN_ROOT="${RUNS_ROOT}/${FM_EXP}"
export CKPT_DIR="${RUN_ROOT}/checkpoints"
export SAMPLE_ROOT="${RUN_ROOT}/samples"
export LOG_ROOT="${RUN_ROOT}/logs"
export METRIC_ROOT="${EVAL_ROOT}/${FM_EXP}"
unset ARCHIVE_ROOT
unset DGFM_ARCHIVE_ROOT
export TORCH_CACHE_ROOT="${TORCH_CACHE_ROOT:-${PROJ}/.torch}"
export TORCH_HOME="${TORCH_HOME:-${TORCH_CACHE_ROOT}}"
export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
export INFER_CUDA_VISIBLE_DEVICES="${INFER_CUDA_VISIBLE_DEVICES:-0}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
mkdir -p "${DATA_ROOT}" "${RUNS_ROOT}" "${EVAL_ROOT}" "${REF_ROOT}" 2>/dev/null || true
mkdir -p "${CKPT_DIR}" "${SAMPLE_ROOT}" "${LOG_ROOT}" "${METRIC_ROOT}" 2>/dev/null || true
mkdir -p "${TORCH_HOME}" "${HF_HUB_CACHE}" 2>/dev/null || true

echo "Activated dgfm experiment"
echo "  PROJ=${PROJ}"
echo "  variant=${variant}"
echo "  EXP_VARIANT=${EXP_VARIANT}"
echo "  EXP_TAG=${EXP_TAG}"
echo "  EXP_NAME=${EXP_NAME}"
echo "  EXP_SOURCE=${EXP_SOURCE}"
echo "  FM_CONFIG=${FM_CONFIG}"
echo "  FM_EXP=${FM_EXP}"
echo "  RUN_ROOT=${RUN_ROOT}"
echo "  METRIC_ROOT=${METRIC_ROOT}"
echo "  TRAJ_ROOT=${TRAJ_ROOT}"
echo "  REF_ROOT=${REF_ROOT}"
echo "  IMAGENET_RAW_ROOT=${IMAGENET_RAW_ROOT}"
echo "  IMAGENET64_PREPROCESSED=${IMAGENET64_PREPROCESSED}"
echo "  IMAGENET64_REFERENCE_NPZ=${IMAGENET64_REFERENCE_NPZ}"
echo "  OFFICIAL_REFERENCE_NPZ=${OFFICIAL_REFERENCE_NPZ}"
echo "  IMAGENET64_TEACHER_CKPT=${IMAGENET64_TEACHER_CKPT}"
echo "  HF_HOME=${HF_HOME}"
echo "  HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "  HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
echo "  TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
echo "  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "  TORCH_HOME=${TORCH_HOME}"
echo "  TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES}"
echo "  INFER_CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES}"
echo "  NNODES=${NNODES}"
echo "  NODE_RANK=${NODE_RANK}"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "  MASTER_ADDR=${MASTER_ADDR}"
echo "  MASTER_PORT=${MASTER_PORT}"
