#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source scripts/server/activate_a100_runtime.sh >/dev/null
source scripts/server/network_profiles.sh
dg_twfd_net_apply "${DG_TWFD_BASELINE_NETWORK_PROFILE:-proxy}"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:$ROOT_DIR/refs/edm:$ROOT_DIR/refs/consistency_models:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

RUN_TAG="${BASELINE_LOW_VRAM_RUN_TAG:-baseline_low_vram_guarded_20260427}"
LOG_ROOT="runs/${RUN_TAG}/logs"
LOG_FILE="${LOG_ROOT}/guard.log"
BACKUP_ROOT="${BASELINE_LIVE_BACKUP_ROOT:-/temp/Zhengwei/DG-TWFD-backups/experiment_evidence/baselines_low_vram_live_20260427}"

CHECK_SECONDS="${BASELINE_GUARD_CHECK_SECONDS:-15}"
START_MAX_MB="${BASELINE_START_MAX_MB:-70000}"
KILL_MAX_MB="${BASELINE_KILL_MAX_MB:-76000}"
COOLDOWN_SECONDS="${BASELINE_COOLDOWN_SECONDS:-300}"
EDM_BATCH="${BASELINE_EDM_BATCH:-1}"
CD_BATCH="${BASELINE_CD_BATCH:-1}"
FID_BATCH="${BASELINE_FID_BATCH:-1}"
NUM_SAMPLES="${BASELINE_NUM_SAMPLES:-50000}"
STEPS=(${BASELINE_STEPS:-1 2 4 8})
EDM_RESUME_CHUNK_SIZE="${BASELINE_EDM_RESUME_CHUNK_SIZE:-1000}"
PAUSE_FOR_MAIN_TRAIN="${BASELINE_PAUSE_FOR_MAIN_TRAIN:-0}"
PAUSE_FOR_MAIN_EVAL="${BASELINE_PAUSE_FOR_MAIN_EVAL:-0}"
MAIN_RUN_TAG="${BASELINE_MAIN_RUN_TAG:-edm_first_cifar10_onestep_msdefect_e504a_resume_from1250}"
MAIN_EVAL_PREFIX="${BASELINE_MAIN_EVAL_PREFIX:-${MAIN_RUN_TAG}}"
MAIN_EVAL_INTERVAL="${BASELINE_MAIN_EVAL_INTERVAL:-250}"
MAIN_EVAL_MARGIN="${BASELINE_MAIN_EVAL_MARGIN:-50}"

mkdir -p "$LOG_ROOT" "$BACKUP_ROOT"

log() {
  echo "[$(date -Is)] $*" | tee -a "$LOG_FILE"
}

gpu_mem_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

main_eval_active() {
  pgrep -f "experiments/edm_first/eval_edm_map.py.*${MAIN_EVAL_PREFIX}" >/dev/null 2>&1
}

main_train_active() {
  pgrep -f "experiments/edm_first/train_edm_map.py.*${MAIN_RUN_TAG}" >/dev/null 2>&1
}

pause_for_main_train() {
  [[ "$PAUSE_FOR_MAIN_TRAIN" =~ ^(1|true|yes|on)$ ]]
}

pause_for_main_eval() {
  [[ "$PAUSE_FOR_MAIN_EVAL" =~ ^(1|true|yes|on)$ ]]
}

main_latest_step() {
  local train_log="runs/${MAIN_RUN_TAG}/logs/train.jsonl"
  [[ -f "$train_log" ]] || {
    echo 0
    return 1
  }
  "$DG_TWFD_A100_ENV/bin/python" - "$train_log" <<'PY'
import json
import sys
from collections import deque

path = sys.argv[1]
latest = 0
try:
    with open(path, "r", encoding="utf-8") as handle:
        for line in deque(handle, maxlen=80):
            try:
                item = json.loads(line)
            except Exception:
                continue
            latest = max(latest, int(item.get("step", 0) or 0))
except FileNotFoundError:
    pass
print(latest)
raise SystemExit(0 if latest > 0 else 1)
PY
}

main_eval_pending() {
  local latest
  latest="$(main_latest_step 2>/dev/null || true)"
  [[ "$latest" =~ ^[0-9]+$ ]] || return 1
  (( latest > 0 )) || return 1

  local prev_eval=$(( (latest / MAIN_EVAL_INTERVAL) * MAIN_EVAL_INTERVAL ))
  local next_eval=$(( ((latest / MAIN_EVAL_INTERVAL) + 1) * MAIN_EVAL_INTERVAL ))

  if (( prev_eval >= MAIN_EVAL_INTERVAL )); then
    local summary="eval/${MAIN_EVAL_PREFIX}_step${prev_eval}/reports/summary.json"
    if [[ ! -s "$summary" ]]; then
      return 0
    fi
  fi

  if (( next_eval - latest <= MAIN_EVAL_MARGIN )); then
    return 0
  fi

  return 1
}

sync_backup() {
  mkdir -p \
    "$BACKUP_ROOT/results" \
    "$BACKUP_ROOT/logs/${RUN_TAG}" \
    "$BACKUP_ROOT/eval_reports/edm_imagenet64_public_eval_full" \
    "$BACKUP_ROOT/eval_reports/cd_imagenet64_lpips_full"

  cp -a results/baselines/. "$BACKUP_ROOT/results/" 2>/dev/null || true
  cp -a "$LOG_ROOT"/. "$BACKUP_ROOT/logs/${RUN_TAG}/" 2>/dev/null || true
  cp -a eval/edm_imagenet64_public_eval_full/reports/. "$BACKUP_ROOT/eval_reports/edm_imagenet64_public_eval_full/" 2>/dev/null || true
  cp -a eval/cd_imagenet64_lpips_full/reports/. "$BACKUP_ROOT/eval_reports/cd_imagenet64_lpips_full/" 2>/dev/null || true

  {
    echo "updated=$(date -Is)"
    echo "run_tag=${RUN_TAG}"
    echo "gpu_mem_used_mb=$(gpu_mem_used_mb || echo unknown)"
    echo "main_train_active=$(main_train_active && echo yes || echo no)"
    echo "main_eval_active=$(main_eval_active && echo yes || echo no)"
    echo "main_eval_pending=$(main_eval_pending && echo yes || echo no)"
    echo "main_latest_step=$(main_latest_step 2>/dev/null || echo unknown)"
    echo "main_eval_interval=${MAIN_EVAL_INTERVAL}"
    echo "main_eval_margin=${MAIN_EVAL_MARGIN}"
    echo "pause_for_main_train=${PAUSE_FOR_MAIN_TRAIN}"
    echo "pause_for_main_eval=${PAUSE_FOR_MAIN_EVAL}"
    echo "start_max_mb=${START_MAX_MB}"
    echo "kill_max_mb=${KILL_MAX_MB}"
    echo "batch_edm=${EDM_BATCH}"
    echo "batch_cd=${CD_BATCH}"
    echo "batch_fid=${FID_BATCH}"
    echo "num_samples=${NUM_SAMPLES}"
    echo "steps=${STEPS[*]}"
    echo "edm_resume_chunk_size=${EDM_RESUME_CHUNK_SIZE}"
  } > "$BACKUP_ROOT/STATUS.txt"
}

wait_until_safe() {
  while true; do
    local mem
    mem="$(gpu_mem_used_mb || echo 999999)"
    if pause_for_main_train && main_train_active; then
      log "main e504a train active; waiting ${COOLDOWN_SECONDS}s"
      sync_backup
      sleep "$COOLDOWN_SECONDS"
      continue
    fi
    if pause_for_main_eval && main_eval_active; then
      log "main e504a eval active; waiting ${COOLDOWN_SECONDS}s"
      sync_backup
      sleep "$COOLDOWN_SECONDS"
      continue
    fi
    if pause_for_main_eval && main_eval_pending; then
      log "main e504a eval pending/near checkpoint; waiting ${COOLDOWN_SECONDS}s"
      sync_backup
      sleep "$COOLDOWN_SECONDS"
      continue
    fi
    if [[ "$mem" =~ ^[0-9]+$ ]] && (( mem <= START_MAX_MB )); then
      return 0
    fi
    log "gpu memory ${mem}MB > start limit ${START_MAX_MB}MB; waiting ${COOLDOWN_SECONDS}s"
    sync_backup
    sleep "$COOLDOWN_SECONDS"
  done
}

terminate_group() {
  local pgid="$1"
  kill -TERM "-${pgid}" >/dev/null 2>&1 || true
  sleep 10
  kill -KILL "-${pgid}" >/dev/null 2>&1 || true
}

run_guarded() {
  local name="$1"
  shift
  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    wait_until_safe
    sync_backup
    log "starting ${name}, attempt=${attempt}: $*"

    setsid nice -n 19 ionice -c3 "$@" >>"$LOG_ROOT/${name}.stdout_stderr.txt" 2>&1 &
    local child="$!"
    local guard_reason=""

    while kill -0 "$child" >/dev/null 2>&1; do
      sleep "$CHECK_SECONDS"
      sync_backup
      local mem
      mem="$(gpu_mem_used_mb || echo 999999)"
      if pause_for_main_train && main_train_active; then
        guard_reason="main_train_active"
      elif pause_for_main_eval && main_eval_active; then
        guard_reason="main_eval_active"
      elif pause_for_main_eval && main_eval_pending; then
        guard_reason="main_eval_pending"
      elif [[ "$mem" =~ ^[0-9]+$ ]] && (( mem > KILL_MAX_MB )); then
        guard_reason="gpu_mem_${mem}_gt_${KILL_MAX_MB}"
      fi

      if [[ -n "$guard_reason" ]]; then
        log "stopping ${name} child=${child}: ${guard_reason}"
        terminate_group "$child"
        break
      fi
    done

    local rc=0
    wait "$child" || rc="$?"
    sync_backup

    if [[ -n "$guard_reason" ]]; then
      log "${name} interrupted by guard (${guard_reason}); cooldown ${COOLDOWN_SECONDS}s before retry"
      sleep "$COOLDOWN_SECONDS"
      continue
    fi
    if [[ "$rc" -eq 0 ]]; then
      log "${name} completed"
      return 0
    fi

    log "${name} failed with rc=${rc}; not retrying unexpected failure"
    return "$rc"
  done
}

main() {
  log "low-vram baseline guard started"
  log "steps=${STEPS[*]} samples=${NUM_SAMPLES} edm_batch=${EDM_BATCH} cd_batch=${CD_BATCH} fid_batch=${FID_BATCH} edm_resume_chunk=${EDM_RESUME_CHUNK_SIZE}"

  run_guarded edm_imagenet64 \
    "$DG_TWFD_A100_ENV/bin/python" scripts/run_edm_cifar10_eval.py \
      --config configs/experiment/edm_imagenet64_public_eval.yaml \
      --sample-root runs/edm_imagenet64_public_eval_full/samples \
      --eval-root eval/edm_imagenet64_public_eval_full \
      --steps "${STEPS[@]}" \
      --num-samples "$NUM_SAMPLES" \
      --batch "$EDM_BATCH" \
      --resume-chunk-size "$EDM_RESUME_CHUNK_SIZE"

  "$DG_TWFD_A100_ENV/bin/python" scripts/baselines/export_unified_baseline_csv.py --write-empty | tee -a "$LOG_FILE"
  sync_backup

  run_guarded cd_imagenet64_lpips \
    "$DG_TWFD_A100_ENV/bin/python" scripts/baselines/run_cd_imagenet64_eval.py \
      --checkpoint /cache/Zhengwei/DG-TWFD-runtime/checkpoints/baselines/consistency_models/cd_imagenet64_lpips.pt \
      --sample-root runs/cd_imagenet64_lpips_full/samples \
      --eval-root eval/cd_imagenet64_lpips_full \
      --csv-out results/baselines/baseline_cd_imagenet64.csv \
      --steps "${STEPS[@]}" \
      --num-samples "$NUM_SAMPLES" \
      --batch "$CD_BATCH" \
      --fid-batch "$FID_BATCH"

  "$DG_TWFD_A100_ENV/bin/python" scripts/baselines/export_unified_baseline_csv.py --write-empty | tee -a "$LOG_FILE"
  sync_backup
  log "low-vram baseline guard finished"
}

main "$@"
