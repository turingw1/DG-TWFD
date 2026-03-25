#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced, not executed." >&2
  echo "Example: source scripts/experiments/activate_fm_timewarp_sampling.sh phase_a v1" >&2
  exit 1
fi

variant="${1:-phase_a}"
tag="${2:-v1}"

case "${variant}" in
  phase_a)
    export FM_TIMEWARP_CONFIG="configs/experiment/fm_timewarp_sampling_phase_a.yaml"
    exp_prefix="fm_timewarp_sampling_phase_a"
    ;;
  phase_b1)
    export FM_TIMEWARP_CONFIG="configs/experiment/fm_timewarp_sampling_phase_b1.yaml"
    exp_prefix="fm_timewarp_sampling_phase_b1"
    ;;
  *)
    echo "Unknown fm_timewarp_sampling variant: ${variant}" >&2
    echo "Expected: phase_a phase_b1" >&2
    return 1
    ;;
esac

export PROJ="${PROJ:-$HOME/workspace/Zhengwei/DG-TWFD}"
export ENV_NAME="${ENV_NAME:-consistency}"
export DATA_ROOT="${DATA_ROOT:-/cache/Zhengwei/datasets}"
export RUNS_ROOT="${RUNS_ROOT:-/cache/Zhengwei/dgfm_runs}"
export EVAL_ROOT="${EVAL_ROOT:-/cache/Zhengwei/dgfm_eval}"
export BASELINE_EXP="${BASELINE_EXP:-fm_cifar10_baseline_v2}"
export BASELINE_CKPT="${BASELINE_CKPT:-${RUNS_ROOT}/${BASELINE_EXP}/checkpoints/best.pt}"
export FM_TIMEWARP_EXP="${exp_prefix}_${tag}"
export FM_TIMEWARP_RUN_ROOT="${RUNS_ROOT}/${FM_TIMEWARP_EXP}"
export FM_TIMEWARP_EVAL_ROOT="${EVAL_ROOT}/${FM_TIMEWARP_EXP}"
export FM_TIMEWARP_SAMPLE_ROOT="${FM_TIMEWARP_RUN_ROOT}/samples"
export TORCH_HOME="${TORCH_HOME:-${FM_TIMEWARP_RUN_ROOT}/.torch}"
mkdir -p "${FM_TIMEWARP_RUN_ROOT}" "${FM_TIMEWARP_EVAL_ROOT}" "${FM_TIMEWARP_SAMPLE_ROOT}" 2>/dev/null || true

echo "Activated fm_timewarp_sampling experiment"
echo "  variant=${variant}"
echo "  FM_TIMEWARP_CONFIG=${FM_TIMEWARP_CONFIG}"
echo "  FM_TIMEWARP_EXP=${FM_TIMEWARP_EXP}"
echo "  BASELINE_CKPT=${BASELINE_CKPT}"
echo "  FM_TIMEWARP_EVAL_ROOT=${FM_TIMEWARP_EVAL_ROOT}"
