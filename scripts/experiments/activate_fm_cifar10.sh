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
  stable)
    export FM_CONFIG="configs/experiment/fm_cifar10_stable.yaml"
    exp_prefix="fm_cifar10_stable"
    ;;
  *)
    echo "Unknown FM variant: ${variant}" >&2
    echo "Expected one of: baseline stable" >&2
    return 1
    ;;
esac

export PROJ="${PROJ:-$HOME/workspace/Zhengwei/DG-TWFD}"
export ENV_NAME="${ENV_NAME:-consistency}"
export DATA_ROOT="${DATA_ROOT:-/cache/Zhengwei/datasets}"
export RUNS_ROOT="${RUNS_ROOT:-/cache/Zhengwei/dgfm_runs}"
export EVAL_ROOT="${EVAL_ROOT:-/cache/Zhengwei/dgfm_eval}"
export FM_EXP="${exp_prefix}_${tag}"
export RUN_ROOT="${RUNS_ROOT}/${FM_EXP}"
export CKPT_DIR="${RUN_ROOT}/checkpoints"
export SAMPLE_ROOT="${RUN_ROOT}/samples"
export LOG_ROOT="${RUN_ROOT}/logs"
export METRIC_ROOT="${EVAL_ROOT}/${FM_EXP}"
export TORCH_HOME="${TORCH_HOME:-${RUN_ROOT}/.torch}"
mkdir -p "${CKPT_DIR}" "${SAMPLE_ROOT}" "${LOG_ROOT}" "${METRIC_ROOT}" 2>/dev/null || true

echo "Activated dgfm experiment"
echo "  variant=${variant}"
echo "  FM_CONFIG=${FM_CONFIG}"
echo "  FM_EXP=${FM_EXP}"
echo "  RUN_ROOT=${RUN_ROOT}"
echo "  METRIC_ROOT=${METRIC_ROOT}"
