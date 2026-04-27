#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${BASELINE_LOW_VRAM_RUN_TAG:-baseline_low_vram_guarded_20260427}"
BACKUP_ROOT="${BASELINE_LIVE_BACKUP_ROOT:-/temp/Zhengwei/DG-TWFD-backups/experiment_evidence/baselines_low_vram_live_20260427}"
LOG_ROOT="runs/${RUN_TAG}/logs"
SUP_LOG="${BASELINE_SUPERVISOR_LOG:-${BACKUP_ROOT}/supervisor.log}"
PID_FILE="${BASELINE_GUARD_PID_FILE:-${BACKUP_ROOT}/guard.pid}"
CHECK_SECONDS="${BASELINE_SUPERVISOR_CHECK_SECONDS:-60}"
RESTART_COOLDOWN_SECONDS="${BASELINE_SUPERVISOR_RESTART_COOLDOWN_SECONDS:-30}"

mkdir -p "$BACKUP_ROOT" "$LOG_ROOT"

log() {
  echo "[$(date -Is)] $*" | tee -a "$SUP_LOG"
}

guard_finished() {
  grep -q "low-vram baseline guard finished" "${LOG_ROOT}/guard.log" 2>/dev/null
}

pid_alive() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" >/dev/null 2>&1
}

start_guard() {
  log "starting guard"
  (
    set +e
    export DG_TWFD_LOCAL_PROXY="${DG_TWFD_LOCAL_PROXY:-http://127.0.0.1:18080}"
    export DG_TWFD_BASELINE_NETWORK_PROFILE="${DG_TWFD_BASELINE_NETWORK_PROFILE:-proxy}"
    export BASELINE_LOW_VRAM_RUN_TAG="$RUN_TAG"
    export BASELINE_START_MAX_MB="${BASELINE_START_MAX_MB:-62000}"
    export BASELINE_KILL_MAX_MB="${BASELINE_KILL_MAX_MB:-64000}"
    export BASELINE_GUARD_CHECK_SECONDS="${BASELINE_GUARD_CHECK_SECONDS:-15}"
    export BASELINE_COOLDOWN_SECONDS="${BASELINE_COOLDOWN_SECONDS:-180}"
    export BASELINE_PAUSE_FOR_MAIN_TRAIN="${BASELINE_PAUSE_FOR_MAIN_TRAIN:-1}"
    export BASELINE_PAUSE_FOR_MAIN_EVAL="${BASELINE_PAUSE_FOR_MAIN_EVAL:-1}"
    export BASELINE_MAIN_EVAL_MARGIN="${BASELINE_MAIN_EVAL_MARGIN:-0}"
    export BASELINE_EDM_BATCH="${BASELINE_EDM_BATCH:-1}"
    export BASELINE_CD_BATCH="${BASELINE_CD_BATCH:-1}"
    export BASELINE_FID_BATCH="${BASELINE_FID_BATCH:-1}"
    export BASELINE_EDM_RESUME_CHUNK_SIZE="${BASELINE_EDM_RESUME_CHUNK_SIZE:-5000}"
    export BASELINE_NUM_SAMPLES="${BASELINE_NUM_SAMPLES:-50000}"
    export BASELINE_STEPS="${BASELINE_STEPS:-1 2 4 8}"
    bash scripts/baselines/run_low_vram_guarded_baselines.sh
    rc="$?"
    echo "[$(date -Is)] guard child exited rc=${rc}" >> "$SUP_LOG"
    exit "$rc"
  ) >> "${LOG_ROOT}/supervisor_child.stdout_stderr.txt" 2>&1 &
  echo "$!" > "$PID_FILE"
  log "guard pid=$(cat "$PID_FILE")"
}

log "supervisor started run_tag=${RUN_TAG}"

while true; do
  if guard_finished; then
    log "guard finished; supervisor exiting"
    exit 0
  fi

  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if pid_alive "$pid"; then
    sleep "$CHECK_SECONDS"
    continue
  fi

  if [[ -n "${pid:-}" ]]; then
    log "guard pid not alive: ${pid}; restarting after ${RESTART_COOLDOWN_SECONDS}s"
    sleep "$RESTART_COOLDOWN_SECONDS"
  fi
  start_guard
  sleep "$CHECK_SECONDS"
done
