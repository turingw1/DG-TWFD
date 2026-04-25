#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash experiments/edm_first/scripts/launch_train.sh <config> <run_tag> [resume]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
config="$1"
tag="$2"
resume="${3:-}"

source "$ROOT_DIR/scripts/server/activate_a100_runtime.sh" >/dev/null
source "$ROOT_DIR/scripts/server/network_profiles.sh"
dg_twfd_net_apply "${DG_TWFD_EDM_TRAIN_NETWORK_PROFILE:-proxy}"

cd "$ROOT_DIR"
run_root="runs/${tag}"
mkdir -p "$run_root/logs"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:$ROOT_DIR/refs/edm:${PYTHONPATH:-}"

cmd=("$DG_TWFD_A100_ENV/bin/python" "experiments/edm_first/train_edm_map.py" "--config" "$config" "--run-root" "$run_root")
if [[ -n "$resume" ]]; then
  cmd+=("--resume" "$resume")
fi

CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0}" "${cmd[@]}" 2>&1 | tee "$run_root/logs/train.stdout_stderr.txt"
