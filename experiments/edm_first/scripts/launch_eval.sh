#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash experiments/edm_first/scripts/launch_eval.sh <config> <run_tag> <checkpoint>" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
config="$1"
tag="$2"
checkpoint="$3"

source "$ROOT_DIR/scripts/server/activate_a100_runtime.sh" >/dev/null
source "$ROOT_DIR/scripts/server/network_profiles.sh"
dg_twfd_net_apply "${DG_TWFD_EDM_EVAL_NETWORK_PROFILE:-proxy}"

cd "$ROOT_DIR"
eval_root="eval/${tag}"
mkdir -p "$eval_root"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:$ROOT_DIR/refs/edm:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="${INFER_CUDA_VISIBLE_DEVICES:-0}" "$DG_TWFD_A100_ENV/bin/python" \
  experiments/edm_first/eval_edm_map.py \
  --config "$config" \
  --checkpoint "$checkpoint" \
  --eval-root "$eval_root" \
  2>&1 | tee "$eval_root/eval.stdout_stderr.txt"
