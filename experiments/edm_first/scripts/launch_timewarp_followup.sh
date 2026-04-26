#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash experiments/edm_first/scripts/launch_timewarp_followup.sh <source_checkpoint> [run_tag] [tmux_session]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source_checkpoint="$1"
config="${DG_TWFD_TIMEWARP_FOLLOWUP_CONFIG:-experiments/edm_first/configs/cifar10_edm_map_onestep_prior_msdefect_timewarp_8h.yaml}"

source "$ROOT_DIR/scripts/server/activate_a100_runtime.sh" >/dev/null
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:$ROOT_DIR/experiments/edm_first:$ROOT_DIR/refs/edm:${PYTHONPATH:-}"

source_step="$("$DG_TWFD_A100_ENV/bin/python" - "$source_checkpoint" <<'PY'
import sys
import torch

ckpt = torch.load(sys.argv[1], map_location="cpu")
print(int(ckpt.get("step", 0)))
PY
)"

run_tag="${2:-edm_first_cifar10_onestep_msdefect_timewarp_e505a_from_step${source_step}}"
tmux_session="${3:-e505a_timewarp}"
source_session="${DG_TWFD_SOURCE_SESSION:-e504a_msdefect}"
stop_source="${DG_TWFD_STOP_SOURCE_ON_SUCCESS:-1}"

if tmux has-session -t "$tmux_session" 2>/dev/null; then
  echo "timewarp follow-up already running: ${tmux_session}"
  exit 0
fi

if [[ "$stop_source" == "1" ]] && tmux has-session -t "$source_session" 2>/dev/null; then
  echo "requesting graceful stop for source session: ${source_session}"
  tmux send-keys -t "$source_session" C-c
fi

if [[ "$stop_source" == "1" ]] && tmux has-session -t "$source_session" 2>/dev/null; then
  wait_seconds="${DG_TWFD_SOURCE_STOP_WAIT_SECONDS:-900}"
  deadline=$((SECONDS + wait_seconds))
  while tmux has-session -t "$source_session" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 15
  done
fi

if tmux has-session -t "$source_session" 2>/dev/null; then
  echo "source session still running after wait; not launching a second training job on the same GPU"
  exit 1
fi

echo "launching timewarp follow-up run_tag=${run_tag} source_step=${source_step}"
tmux new-session -d -s "$tmux_session" \
  "cd '$ROOT_DIR' && bash experiments/edm_first/scripts/launch_train.sh '$config' '$run_tag' '$source_checkpoint'"

tmux new-session -d -s "${tmux_session}_eval_watch" \
  "cd '$ROOT_DIR' && export DG_TWFD_EVAL_BASELINE_SUMMARY=eval/edm_first_cifar10_onestep_msdefect_e504a_step250_steps16/reports/summary.json && export DG_TWFD_EVAL_TARGET_RATIO=0.5 && export DG_TWFD_EVAL_PRIMARY_STEP=1 && bash experiments/edm_first/scripts/watch_eval_checkpoints.sh '$config' '$run_tag' '$run_tag' 250 250 2048 1 2 4 8 16"

echo "timewarp follow-up started: ${tmux_session}, ${tmux_session}_eval_watch"
