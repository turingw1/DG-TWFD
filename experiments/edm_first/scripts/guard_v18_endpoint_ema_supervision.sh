#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_TAG="${DG_TWFD_V18_RUN_TAG:-edm_first_cifar10_endpoint_ema_v18_from_v17_last}"
PROJECT_NAME="${DG_TWFD_PROJECT_NAME:-$(basename "$ROOT_DIR")}"
TRAIN_SESSION="${DG_TWFD_V18_TRAIN_SESSION:-v18_endpoint_ema}"
SUP_SESSION="${DG_TWFD_V18_SUP_SESSION:-v18_endpoint_ema_2h}"
INTERVAL="${DG_TWFD_SUPERVISE_INTERVAL_SECONDS:-7200}"
CHECK_INTERVAL="${DG_TWFD_V18_GUARD_CHECK_INTERVAL:-300}"
HOURS="${DG_TWFD_SUPERVISE_HOURS:-999}"
LOG_DIR="${ROOT_DIR}/runs/${RUN_TAG}/logs"
LOG_FILE="${LOG_DIR}/v18_supervision_guard.log"

mkdir -p "$LOG_DIR"
echo "v18 supervision guard started train_session=${TRAIN_SESSION} sup_session=${SUP_SESSION} interval=${INTERVAL}" | tee -a "$LOG_FILE"

while true; do
  if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
    if ! tmux has-session -t "$SUP_SESSION" 2>/dev/null; then
      tmux new-session -d -s "$SUP_SESSION" \
        "cd '$ROOT_DIR' && export DG_TWFD_SUPERVISE_HOURS='$HOURS' && export DG_TWFD_SUPERVISE_INTERVAL_SECONDS='$INTERVAL' && export DG_TWFD_SUPERVISE_TMUX_SESSION='$TRAIN_SESSION' && bash experiments/edm_first/scripts/supervise_run_hourly_v11.sh '$RUN_TAG' '$PROJECT_NAME'"
      echo "$(date -Is) started missing supervisor ${SUP_SESSION}" | tee -a "$LOG_FILE"
    else
      echo "$(date -Is) supervisor alive ${SUP_SESSION}" | tee -a "$LOG_FILE"
    fi
    sleep "$INTERVAL"
  else
    echo "$(date -Is) waiting for train session ${TRAIN_SESSION}" | tee -a "$LOG_FILE"
    sleep "$CHECK_INTERVAL"
  fi
done
