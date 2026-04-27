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
ACTIVE_CHILD_PGID=""

CHECK_SECONDS="${BASELINE_GUARD_CHECK_SECONDS:-15}"
START_MAX_MB="${BASELINE_START_MAX_MB:-70000}"
KILL_MAX_MB="${BASELINE_KILL_MAX_MB:-79000}"
COOLDOWN_SECONDS="${BASELINE_COOLDOWN_SECONDS:-300}"
EDM_BATCH="${BASELINE_EDM_BATCH:-8}"
CD_BATCH="${BASELINE_CD_BATCH:-8}"
FID_BATCH="${BASELINE_FID_BATCH:-32}"
NUM_SAMPLES="${BASELINE_NUM_SAMPLES:-50000}"
STEPS=(${BASELINE_STEPS:-1 2 4 8})
EDM_RESUME_CHUNK_SIZE="${BASELINE_EDM_RESUME_CHUNK_SIZE:-1000}"
EDM_SAMPLE_ROOT="${BASELINE_EDM_SAMPLE_ROOT:-runs/edm_imagenet64_public_eval_full/samples}"
EDM_EVAL_ROOT="${BASELINE_EDM_EVAL_ROOT:-eval/edm_imagenet64_public_eval_full}"
CD_SAMPLE_ROOT="${BASELINE_CD_SAMPLE_ROOT:-runs/cd_imagenet64_lpips_full/samples}"
CD_EVAL_ROOT="${BASELINE_CD_EVAL_ROOT:-eval/cd_imagenet64_lpips_full}"
CD_CSV_OUT="${BASELINE_CD_CSV_OUT:-results/baselines/baseline_cd_imagenet64.csv}"
CD_CHECKPOINT="${BASELINE_CD_CHECKPOINT:-/cache/Zhengwei/DG-TWFD-runtime/checkpoints/baselines/consistency_models/cd_imagenet64_lpips.pt}"
CD_METHOD="${BASELINE_CD_METHOD:-CD-LPIPS-official}"
PAUSE_FOR_MAIN_TRAIN="${BASELINE_PAUSE_FOR_MAIN_TRAIN:-0}"
PAUSE_FOR_MAIN_EVAL="${BASELINE_PAUSE_FOR_MAIN_EVAL:-0}"
MAIN_RUN_TAG="${BASELINE_MAIN_RUN_TAG:-edm_first_cifar10_onestep_msdefect_e504a_resume_from1250}"
MAIN_EVAL_PREFIX="${BASELINE_MAIN_EVAL_PREFIX:-${MAIN_RUN_TAG}}"
MAIN_EVAL_INTERVAL="${BASELINE_MAIN_EVAL_INTERVAL:-250}"
MAIN_EVAL_MARGIN="${BASELINE_MAIN_EVAL_MARGIN:-50}"
SYNC_SAMPLES="${BASELINE_SYNC_SAMPLES:-1}"
SAMPLE_SYNC_SECONDS="${BASELINE_SAMPLE_SYNC_SECONDS:-300}"
LAST_SAMPLE_SYNC_FILE="${BACKUP_ROOT}/.last_sample_sync_epoch"

mkdir -p "$LOG_ROOT" "$BACKUP_ROOT"

log() {
  echo "[$(date -Is)] $*" | tee -a "$LOG_FILE"
}

on_exit() {
  local rc="$?"
  if [[ -n "${ACTIVE_CHILD_PGID:-}" ]]; then
    terminate_group "$ACTIVE_CHILD_PGID"
  fi
  log "low-vram baseline guard exiting rc=${rc}"
  sync_samples_backup 1 >/dev/null 2>&1 || true
  sync_backup >/dev/null 2>&1 || true
}

trap on_exit EXIT
trap 'log "low-vram baseline guard received TERM"; exit 143' TERM
trap 'log "low-vram baseline guard received INT"; exit 130' INT

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
  local edm_eval_name cd_eval_name
  edm_eval_name="$(basename "$EDM_EVAL_ROOT")"
  cd_eval_name="$(basename "$CD_EVAL_ROOT")"

  mkdir -p \
    "$BACKUP_ROOT/results" \
    "$BACKUP_ROOT/logs/${RUN_TAG}" \
    "$BACKUP_ROOT/eval_reports/${edm_eval_name}" \
    "$BACKUP_ROOT/eval_reports/${cd_eval_name}" \
    "$BACKUP_ROOT/sample_archives"

  cp -a results/baselines/. "$BACKUP_ROOT/results/" 2>/dev/null || true
  cp -a "$LOG_ROOT"/. "$BACKUP_ROOT/logs/${RUN_TAG}/" 2>/dev/null || true
  cp -a "$EDM_EVAL_ROOT/reports/." "$BACKUP_ROOT/eval_reports/${edm_eval_name}/" 2>/dev/null || true
  cp -a "$CD_EVAL_ROOT/reports/." "$BACKUP_ROOT/eval_reports/${cd_eval_name}/" 2>/dev/null || true
  sync_samples_backup 0

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
    echo "edm_sample_root=${EDM_SAMPLE_ROOT}"
    echo "edm_eval_root=${EDM_EVAL_ROOT}"
    echo "cd_sample_root=${CD_SAMPLE_ROOT}"
    echo "cd_eval_root=${CD_EVAL_ROOT}"
    echo "cd_csv_out=${CD_CSV_OUT}"
    echo "cd_checkpoint=${CD_CHECKPOINT}"
    echo "cd_method=${CD_METHOD}"
    echo "sync_samples=${SYNC_SAMPLES}"
    echo "sample_sync_seconds=${SAMPLE_SYNC_SECONDS}"
  } > "$BACKUP_ROOT/STATUS.txt"
}

sync_samples_backup() {
  local force="${1:-0}"
  [[ "$SYNC_SAMPLES" =~ ^(1|true|yes|on)$ ]] || return 0

  local now last
  now="$(date +%s)"
  last=0
  [[ -s "$LAST_SAMPLE_SYNC_FILE" ]] && last="$(cat "$LAST_SAMPLE_SYNC_FILE" 2>/dev/null || echo 0)"
  [[ "$last" =~ ^[0-9]+$ ]] || last=0

  if [[ "$force" != "1" ]] && (( now - last < SAMPLE_SYNC_SECONDS )); then
    return 0
  fi

  mkdir -p "$BACKUP_ROOT/sample_archives"

  local step src count archive count_file archived_count
  for step in "${STEPS[@]}"; do
    src="${EDM_SAMPLE_ROOT}/steps${step}"
    [[ -d "$src" ]] || continue
    count="$(find "$src/images" -type f -name '*.png' 2>/dev/null | wc -l | tr -d ' ')"
    [[ "$count" =~ ^[0-9]+$ ]] || count=0
    (( count > 0 )) || continue

    archive="$BACKUP_ROOT/sample_archives/edm_step${step}_images.tar"
    count_file="$BACKUP_ROOT/sample_archives/edm_step${step}_images.count"
    archived_count=0
    [[ -s "$count_file" ]] && archived_count="$(cat "$count_file" 2>/dev/null || echo 0)"
    [[ "$archived_count" =~ ^[0-9]+$ ]] || archived_count=0
    if (( count >= NUM_SAMPLES && archived_count >= NUM_SAMPLES && count <= archived_count )); then
      continue
    fi

    tar --no-xattrs --exclude='.DS_Store' --exclude='._*' --exclude='__MACOSX' -cf "$archive" "$src" 2>/dev/null || true
    echo "$count" > "$count_file"
  done

  echo "$now" > "$LAST_SAMPLE_SYNC_FILE"
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
    ACTIVE_CHILD_PGID="$child"
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
    ACTIVE_CHILD_PGID=""
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
      --sample-root "$EDM_SAMPLE_ROOT" \
      --eval-root "$EDM_EVAL_ROOT" \
      --steps "${STEPS[@]}" \
      --num-samples "$NUM_SAMPLES" \
      --batch "$EDM_BATCH" \
      --resume-chunk-size "$EDM_RESUME_CHUNK_SIZE"

  "$DG_TWFD_A100_ENV/bin/python" scripts/baselines/export_unified_baseline_csv.py --write-empty | tee -a "$LOG_FILE"
  sync_backup

  run_guarded cd_imagenet64_lpips \
    "$DG_TWFD_A100_ENV/bin/python" scripts/baselines/run_cd_imagenet64_eval.py \
      --checkpoint "$CD_CHECKPOINT" \
      --method "$CD_METHOD" \
      --sample-root "$CD_SAMPLE_ROOT" \
      --eval-root "$CD_EVAL_ROOT" \
      --csv-out "$CD_CSV_OUT" \
      --steps "${STEPS[@]}" \
      --num-samples "$NUM_SAMPLES" \
      --batch "$CD_BATCH" \
      --fid-batch "$FID_BATCH"

  "$DG_TWFD_A100_ENV/bin/python" scripts/baselines/export_unified_baseline_csv.py --write-empty | tee -a "$LOG_FILE"
  sync_backup
  log "low-vram baseline guard finished"
}

main "$@"
