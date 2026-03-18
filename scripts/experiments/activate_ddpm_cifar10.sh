#!/usr/bin/env bash

# Usage:
#   source scripts/experiments/activate_ddpm_cifar10.sh [variant] [tag]
#
# Examples:
#   source scripts/experiments/activate_ddpm_cifar10.sh
#   source scripts/experiments/activate_ddpm_cifar10.sh stable v3
#   source scripts/experiments/activate_ddpm_cifar10.sh ablate_match exp01
#
# Optional environment controls:
#   export DG_TWFD_SHARD_ROOT=/cache/Zhengwei/dg_twfd_shards/some_data_version
#   export DG_TWFD_DATA_TAG=ddpm_cifar10_teacher128_grid129

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced, not executed." >&2
  echo "Example: source scripts/experiments/activate_ddpm_cifar10.sh stable" >&2
  exit 1
fi

variant="${1:-stable}"
tag="${2:-}"

case "${variant}" in
  base)
    train_mode="train_a100_base"
    exp_prefix="ddpm_cifar10_a100_base"
    ;;
  stable)
    train_mode="train_a100_stable"
    exp_prefix="ddpm_cifar10_a100_stable"
    ;;
  stable_dit)
    train_mode="train_a100_stable_dit"
    exp_prefix="ddpm_cifar10_a100_stable_dit"
    ;;
  ablate_match)
    train_mode="train_a100_ablate_match"
    exp_prefix="ddpm_cifar10_ablate_match"
    ;;
  ablate_defect)
    train_mode="train_a100_ablate_defect"
    exp_prefix="ddpm_cifar10_ablate_defect"
    ;;
  ablate_warp)
    train_mode="train_a100_ablate_warp"
    exp_prefix="ddpm_cifar10_ablate_warp"
    ;;
  ablate_boundary)
    train_mode="train_a100_ablate_boundary"
    exp_prefix="ddpm_cifar10_ablate_boundary"
    ;;
  *)
    echo "Unknown experiment variant: ${variant}" >&2
    echo "Expected one of: base stable stable_dit ablate_match ablate_defect ablate_warp ablate_boundary" >&2
    return 1
    ;;
esac

if [[ -n "${tag}" ]]; then
  exp_name="${exp_prefix}_${tag}"
else
  exp_name="${exp_prefix}"
fi

export PROJ="${PROJ:-$HOME/workspace/Zhengwei/DG-TWFD}"
export ENV_NAME="${ENV_NAME:-consistency}"
export TEACHER_ID="${TEACHER_ID:-google/ddpm-cifar10-32}"
export TRAIN_MODE="${train_mode}"

if [[ "${variant}" == ablate_* ]]; then
  export EXP_NAME_ABL="${exp_name}"
  export EXP_NAME="${exp_name}"
  export RUN_ROOT_ABL="/cache/Zhengwei/dg_twfd_runs/${EXP_NAME_ABL}"
  export CKPT_DIR_ABL="${RUN_ROOT_ABL}/checkpoints"
  export ARTIFACT_ROOT_ABL="${RUN_ROOT_ABL}/samples"
  export TRAIN_LOG_ABL="${RUN_ROOT_ABL}/train.log"
else
  export EXP_NAME="${exp_name}"
fi

if [[ -n "${DG_TWFD_SHARD_ROOT:-}" ]]; then
  export SHARD_ROOT="${DG_TWFD_SHARD_ROOT}"
elif [[ -n "${DG_TWFD_DATA_TAG:-}" ]]; then
  export SHARD_ROOT="/cache/Zhengwei/dg_twfd_shards/${DG_TWFD_DATA_TAG}"
else
  export SHARD_ROOT="/cache/Zhengwei/dg_twfd_shards/${EXP_NAME}"
fi

export RUN_ROOT="/cache/Zhengwei/dg_twfd_runs/${EXP_NAME}"
export CKPT_DIR="${RUN_ROOT}/checkpoints"
export ARTIFACT_ROOT="${RUN_ROOT}/samples"
export TRAIN_LOG="${RUN_ROOT}/train.log"

mkdir -p "${CKPT_DIR}" "${ARTIFACT_ROOT}" 2>/dev/null || true
if [[ "${variant}" == ablate_* ]]; then
  mkdir -p "${CKPT_DIR_ABL}" "${ARTIFACT_ROOT_ABL}" 2>/dev/null || true
fi

echo "Activated DG-TWFD experiment"
echo "  variant=${variant}"
echo "  TRAIN_MODE=${TRAIN_MODE}"
echo "  EXP_NAME=${EXP_NAME}"
echo "  SHARD_ROOT=${SHARD_ROOT}"
echo "  RUN_ROOT=${RUN_ROOT}"
if [[ "${variant}" == ablate_* ]]; then
  echo "  EXP_NAME_ABL=${EXP_NAME_ABL}"
  echo "  CKPT_DIR_ABL=${CKPT_DIR_ABL}"
fi
