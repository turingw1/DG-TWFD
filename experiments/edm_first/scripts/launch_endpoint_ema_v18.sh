#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash experiments/edm_first/scripts/launch_endpoint_ema_v18.sh <source_checkpoint> [run_tag] [tmux_session]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source_checkpoint="$1"
config="${DG_TWFD_ENDPOINT_EMA_CONFIG:-experiments/edm_first/configs/cifar10_edm_map_endpoint_ema_v18.yaml}"
run_tag="${2:-edm_first_cifar10_endpoint_ema_v18_from_v17_last}"
tmux_session="${3:-v18_endpoint_ema}"
source_session="${DG_TWFD_SOURCE_SESSION:-v17_rqs_fastwarp}"

source "$ROOT_DIR/scripts/server/activate_a100_runtime.sh" >/dev/null
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:$ROOT_DIR/experiments/edm_first:$ROOT_DIR/refs/edm:${PYTHONPATH:-}"

if tmux has-session -t "$source_session" 2>/dev/null; then
  wait_session="${tmux_session}_wait"
  if tmux has-session -t "$wait_session" 2>/dev/null; then
    echo "endpoint EMA v18 wait session already running: ${wait_session}"
  else
    tmux new-session -d -s "$wait_session" \
      "cd '$ROOT_DIR' && while tmux has-session -t '$source_session' 2>/dev/null; do sleep 300; done; bash experiments/edm_first/scripts/launch_endpoint_ema_v18.sh '$source_checkpoint' '$run_tag' '$tmux_session'"
    echo "endpoint EMA v18 queued after ${source_session}: ${wait_session}"
  fi
  exit 0
fi

if [[ ! -f "$source_checkpoint" ]]; then
  echo "missing source checkpoint: ${source_checkpoint}" >&2
  exit 1
fi

if tmux has-session -t "$tmux_session" 2>/dev/null; then
  echo "endpoint EMA v18 train already running: ${tmux_session}"
else
  tmux new-session -d -s "$tmux_session" \
    "cd '$ROOT_DIR' && bash experiments/edm_first/scripts/launch_train.sh '$config' '$run_tag' '$source_checkpoint'"
  echo "endpoint EMA v18 train started: ${tmux_session}"
fi

if tmux has-session -t "${tmux_session}_eval_watch" 2>/dev/null; then
  echo "endpoint EMA v18 eval watcher already running: ${tmux_session}_eval_watch"
else
  tmux new-session -d -s "${tmux_session}_eval_watch" \
    "cd '$ROOT_DIR' && export DG_TWFD_EVAL_TMUX_SESSION='$tmux_session' && export DG_TWFD_EVAL_FINAL_WHEN_TRAIN_EXIT=1 && export DG_TWFD_EVAL_BASELINE_SUMMARY='eval/edm_first_cifar10_prior_fullstack_timewarp_v17_rqs_fastwarp_from_step11855_step9750/reports/summary.json' && export DG_TWFD_EVAL_TARGET_RATIO=1.0 && export DG_TWFD_EVAL_PRIMARY_STEP=1 && bash experiments/edm_first/scripts/watch_eval_checkpoints.sh '$config' '$run_tag' '$run_tag' 250 250 '${DG_TWFD_EVAL_FID_SAMPLES:-2048}' 1 2 4 8 16"
  echo "endpoint EMA v18 eval watcher started: ${tmux_session}_eval_watch"
fi

if tmux has-session -t "${tmux_session}_backup" 2>/dev/null; then
  echo "endpoint EMA v18 project backup watcher already running: ${tmux_session}_backup"
else
  tmux new-session -d -s "${tmux_session}_backup" \
    "cd '$ROOT_DIR' && bash experiments/edm_first/scripts/watch_project_backup_v11.sh '$run_tag' 180 '$(basename "$ROOT_DIR")'"
  echo "endpoint EMA v18 project backup watcher started: ${tmux_session}_backup"
fi

if [[ "${DG_TWFD_START_HOURLY_SUPERVISOR:-1}" == "1" ]]; then
  if tmux has-session -t "${tmux_session}_2h" 2>/dev/null; then
    echo "endpoint EMA v18 2h supervisor already running: ${tmux_session}_2h"
  else
    tmux new-session -d -s "${tmux_session}_2h" \
      "cd '$ROOT_DIR' && export DG_TWFD_SUPERVISE_HOURS='${DG_TWFD_SUPERVISE_HOURS:-8}' && export DG_TWFD_SUPERVISE_INTERVAL_SECONDS='${DG_TWFD_SUPERVISE_INTERVAL_SECONDS:-7200}' && export DG_TWFD_SUPERVISE_TMUX_SESSION='$tmux_session' && bash experiments/edm_first/scripts/supervise_run_hourly_v11.sh '$run_tag' '$(basename "$ROOT_DIR")'"
    echo "endpoint EMA v18 2h supervisor started: ${tmux_session}_2h"
  fi
fi

echo "run_tag=${run_tag}"
