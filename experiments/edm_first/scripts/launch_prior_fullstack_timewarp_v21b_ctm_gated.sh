#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash experiments/edm_first/scripts/launch_prior_fullstack_timewarp_v21b_ctm_gated.sh <source_checkpoint> [run_tag] [tmux_session]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source_checkpoint="$1"
config="${DG_TWFD_FULLSTACK_CONFIG:-experiments/edm_first/configs/cifar10_edm_map_prior_fullstack_timewarp_v21b_ctm_gated.yaml}"
baseline_summary="${DG_TWFD_EVAL_BASELINE_SUMMARY:-eval/edm_first_cifar10_prior_fullstack_timewarp_v20_endpoint_balanced_from_v19_step7106_step3000_budget/reports/summary.json}"
source_session="${DG_TWFD_WAIT_FOR_SOURCE_SESSION:-v21_ctm_aligned}"

source "$ROOT_DIR/scripts/server/activate_a100_runtime.sh" >/dev/null
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:$ROOT_DIR/experiments/edm_first:$ROOT_DIR/refs/edm:${PYTHONPATH:-}"

if [[ ! -f "$source_checkpoint" ]]; then
  echo "missing source checkpoint: ${source_checkpoint}" >&2
  exit 1
fi

source_step="$("$DG_TWFD_A100_ENV/bin/python" - "$source_checkpoint" <<'PY'
import sys
import torch

ckpt = torch.load(sys.argv[1], map_location="cpu")
print(int(ckpt.get("step", 0)))
PY
)"

run_tag="${2:-edm_first_cifar10_prior_fullstack_timewarp_v21b_ctm_gated_from_v21_step${source_step}}"
tmux_session="${3:-v21b_ctm_gated}"

if [[ -n "$source_session" ]] && tmux has-session -t "$source_session" 2>/dev/null; then
  wait_session="${tmux_session}_wait"
  if tmux has-session -t "$tmux_session" 2>/dev/null; then
    echo "v21b CTM-gated train already running: ${tmux_session}"
  elif tmux has-session -t "$wait_session" 2>/dev/null; then
    echo "v21b CTM-gated wait already running: ${wait_session}"
  else
    tmux new-session -d -s "$wait_session" \
      "cd '$ROOT_DIR' && while tmux has-session -t '$source_session' 2>/dev/null; do sleep 180; done; DG_TWFD_WAIT_FOR_SOURCE_SESSION= bash experiments/edm_first/scripts/launch_prior_fullstack_timewarp_v21b_ctm_gated.sh '$source_checkpoint' '$run_tag' '$tmux_session'"
    echo "v21b CTM-gated wait started: ${wait_session}; waiting for ${source_session}"
  fi
  echo "run_tag=${run_tag}"
  exit 0
fi

if tmux has-session -t "$tmux_session" 2>/dev/null; then
  echo "v21b CTM-gated train already running: ${tmux_session}"
else
  tmux new-session -d -s "$tmux_session" \
    "cd '$ROOT_DIR' && bash experiments/edm_first/scripts/launch_train.sh '$config' '$run_tag' '$source_checkpoint'"
  echo "v21b CTM-gated train started: ${tmux_session}"
fi

if tmux has-session -t "${tmux_session}_eval_watch" 2>/dev/null; then
  echo "v21b CTM-gated eval watcher already running: ${tmux_session}_eval_watch"
else
  tmux new-session -d -s "${tmux_session}_eval_watch" \
    "cd '$ROOT_DIR' && export DG_TWFD_EVAL_COMPARE_IDENTITY=1 && export DG_TWFD_EVAL_COMPARE_BUDGET=1 && export DG_TWFD_EVAL_TMUX_SESSION='$tmux_session' && export DG_TWFD_EVAL_FINAL_WHEN_TRAIN_EXIT=1 && export DG_TWFD_EVAL_BASELINE_SUMMARY='$baseline_summary' && export DG_TWFD_EVAL_TARGET_RATIO=1.0 && export DG_TWFD_EVAL_PRIMARY_STEP=1 && bash experiments/edm_first/scripts/watch_eval_checkpoints.sh '$config' '$run_tag' '$run_tag' 250 250 '${DG_TWFD_EVAL_FID_SAMPLES:-2048}' 1 2 4 8 16"
  echo "v21b CTM-gated eval watcher started: ${tmux_session}_eval_watch"
fi

if tmux has-session -t "${tmux_session}_backup" 2>/dev/null; then
  echo "v21b CTM-gated project backup watcher already running: ${tmux_session}_backup"
else
  tmux new-session -d -s "${tmux_session}_backup" \
    "cd '$ROOT_DIR' && bash experiments/edm_first/scripts/watch_project_backup_v11.sh '$run_tag' 180 '$(basename "$ROOT_DIR")'"
  echo "v21b CTM-gated project backup watcher started: ${tmux_session}_backup"
fi

if [[ "${DG_TWFD_START_HOURLY_SUPERVISOR:-1}" == "1" ]]; then
  if tmux has-session -t "${tmux_session}_2h" 2>/dev/null; then
    echo "v21b CTM-gated 2h supervisor already running: ${tmux_session}_2h"
  else
    tmux new-session -d -s "${tmux_session}_2h" \
      "cd '$ROOT_DIR' && export DG_TWFD_SUPERVISE_HOURS='${DG_TWFD_SUPERVISE_HOURS:-8}' && export DG_TWFD_SUPERVISE_INTERVAL_SECONDS='${DG_TWFD_SUPERVISE_INTERVAL_SECONDS:-7200}' && export DG_TWFD_SUPERVISE_TMUX_SESSION='$tmux_session' && bash experiments/edm_first/scripts/supervise_run_hourly_v11.sh '$run_tag' '$(basename "$ROOT_DIR")'"
    echo "v21b CTM-gated 2h supervisor started: ${tmux_session}_2h"
  fi
fi

echo "run_tag=${run_tag}"
