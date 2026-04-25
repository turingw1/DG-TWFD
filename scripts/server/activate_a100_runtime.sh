#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced, not executed." >&2
  echo "Example: source scripts/server/activate_a100_runtime.sh" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

_safe_symlink() {
  local link_path="$1"
  local target_path="$2"

  if [[ -L "$link_path" ]]; then
    local current
    current="$(readlink "$link_path")"
    if [[ "$current" == "$target_path" ]]; then
      return 0
    fi
    rm "$link_path"
  elif [[ -e "$link_path" ]]; then
    echo "Refusing to replace existing non-symlink path: $link_path" >&2
    return 1
  fi

  ln -s "$target_path" "$link_path"
}

export PROJ="${PROJ:-$ROOT_DIR}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-/cache/Zhengwei/DG-TWFD-runtime}"
export BACKUP_ROOT="${BACKUP_ROOT:-/temp/Zhengwei/DG-TWFD-backups}"
export CACHE_ROOT="${CACHE_ROOT:-$RUNTIME_ROOT/.cache}"
export DATA_ROOT="${DATA_ROOT:-$PROJ/datasets}"
export RUNS_ROOT="${RUNS_ROOT:-$PROJ/runs}"
export EVAL_ROOT="${EVAL_ROOT:-$PROJ/eval}"
export RESULTS_ROOT="${RESULTS_ROOT:-$PROJ/results}"
export TRAJ_ROOT="${TRAJ_ROOT:-$PROJ/teacher_traj/cifar10_ddpm128_p33}"
export REF_ROOT="${REF_ROOT:-$PROJ/refs}"
export HF_HOME="${HF_HOME:-$PROJ/.hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$PROJ/.torch}"
export DNNLIB_CACHE_DIR="${DNNLIB_CACHE_DIR:-$TORCH_HOME/dnnlib}"
export DG_TWFD_BACKUP_ROOT="${DG_TWFD_BACKUP_ROOT:-$PROJ/backup_runs}"
export TMPDIR="${TMPDIR:-/tmp/dg_twfd}"
export DG_TWFD_TMP_BACKUP_ROOT="${DG_TWFD_TMP_BACKUP_ROOT:-$BACKUP_ROOT/tmp_scratch}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$RUNTIME_ROOT/.conda_pkgs}"
export DG_TWFD_A100_ENV="${DG_TWFD_A100_ENV:-$PROJ/.conda_envs/dg_twfd_a100}"
export IMAGENET_RAW_ROOT="${IMAGENET_RAW_ROOT:-$DATA_ROOT/imagenet_raw}"
export IMAGENET64_PREPROCESSED="${IMAGENET64_PREPROCESSED:-$DATA_ROOT/imagenet64}"
export IMAGENET64_REFERENCE_NPZ="${IMAGENET64_REFERENCE_NPZ:-$REF_ROOT/VIRTUAL_imagenet64_labeled.npz}"
export OFFICIAL_REFERENCE_NPZ="${OFFICIAL_REFERENCE_NPZ:-}"
export IMAGENET64_TEACHER_CKPT="${IMAGENET64_TEACHER_CKPT:-$RUNTIME_ROOT/checkpoints/teachers/edm_imagenet64_ema.pt}"
export EDM_CIFAR10_NETWORK="${EDM_CIFAR10_NETWORK:-https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl}"
export EDM_CIFAR10_FID_REF="${EDM_CIFAR10_FID_REF:-https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz}"
export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0}"
export INFER_CUDA_VISIBLE_DEVICES="${INFER_CUDA_VISIBLE_DEVICES:-0}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_ROOT/matplotlib}"

source "$ROOT_DIR/scripts/server/network_profiles.sh"
dg_twfd_net_apply "${DG_TWFD_NETWORK_PROFILE:-proxy}"

mkdir -p \
  "$RUNTIME_ROOT" \
  "$BACKUP_ROOT" \
  "$CACHE_ROOT" \
  "$RUNTIME_ROOT/datasets" \
  "$RUNTIME_ROOT/runs" \
  "$RUNTIME_ROOT/eval" \
  "$RUNTIME_ROOT/results" \
  "$RUNTIME_ROOT/teacher_traj" \
  "$RUNTIME_ROOT/.hf_home" \
  "$RUNTIME_ROOT/.torch" \
  "$RUNTIME_ROOT/.modelscope" \
  "$RUNTIME_ROOT/.conda_envs" \
  "$RUNTIME_ROOT/.conda_pkgs" \
  "$RUNTIME_ROOT/checkpoints/teachers" \
  "$TMPDIR" \
  "$BACKUP_ROOT/tmp_scratch" \
  "$DG_TWFD_TMP_BACKUP_ROOT" \
  "$XDG_CACHE_HOME" \
  "$MPLCONFIGDIR"

_safe_symlink "$PROJ/datasets" "$RUNTIME_ROOT/datasets"
_safe_symlink "$PROJ/runs" "$RUNTIME_ROOT/runs"
_safe_symlink "$PROJ/eval" "$RUNTIME_ROOT/eval"
_safe_symlink "$PROJ/results" "$RUNTIME_ROOT/results"
_safe_symlink "$PROJ/teacher_traj" "$RUNTIME_ROOT/teacher_traj"
_safe_symlink "$PROJ/.hf_home" "$RUNTIME_ROOT/.hf_home"
_safe_symlink "$PROJ/.torch" "$RUNTIME_ROOT/.torch"
_safe_symlink "$PROJ/.conda_envs" "$RUNTIME_ROOT/.conda_envs"
_safe_symlink "$PROJ/backup_runs" "$BACKUP_ROOT"

export PYTHONPATH="$PROJ/src:$PROJ/refs/edm:${PYTHONPATH:-}"

echo "Activated A100 runtime layout"
echo "  PROJ=$PROJ"
echo "  RUNTIME_ROOT=$RUNTIME_ROOT"
echo "  BACKUP_ROOT=$BACKUP_ROOT"
echo "  DATA_ROOT=$DATA_ROOT"
echo "  RUNS_ROOT=$RUNS_ROOT"
echo "  EVAL_ROOT=$EVAL_ROOT"
echo "  RESULTS_ROOT=$RESULTS_ROOT"
echo "  TRAJ_ROOT=$TRAJ_ROOT"
echo "  HF_HOME=$HF_HOME"
echo "  TORCH_HOME=$TORCH_HOME"
echo "  DNNLIB_CACHE_DIR=$DNNLIB_CACHE_DIR"
echo "  MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-}"
echo "  DG_TWFD_BACKUP_ROOT=$DG_TWFD_BACKUP_ROOT"
echo "  TMPDIR=$TMPDIR"
echo "  DG_TWFD_TMP_BACKUP_ROOT=$DG_TWFD_TMP_BACKUP_ROOT"
echo "  DG_TWFD_A100_ENV=$DG_TWFD_A100_ENV"
echo "  DG_TWFD_NETWORK_PROFILE=${DG_TWFD_NETWORK_PROFILE:-}"
echo "  HF_ENDPOINT=${HF_ENDPOINT:-}"
