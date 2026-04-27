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

mkdir -p "$LOG_ROOT" "$BACKUP_ROOT"

log() {
  echo "[$(date -Is)] $*" | tee -a "$LOG_FILE"
}

gpu_mem_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

main_eval_active() {
  pgrep -f "experiments/edm_first/eval_edm_map.py.*edm_first_cifar10_onestep_msdefect_e504a_resume_from1250" >/dev/null 2>&1
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
    echo "main_eval_active=$(main_eval_active && echo yes || echo no)"
    echo "start_max_mb=${START_MAX_MB}"
    echo "kill_max_mb=${KILL_MAX_MB}"
    echo "batch_edm=${EDM_BATCH}"
    echo "batch_cd=${CD_BATCH}"
    echo "batch_fid=${FID_BATCH}"
    echo "num_samples=${NUM_SAMPLES}"
    echo "steps=${STEPS[*]}"
  } > "$BACKUP_ROOT/STATUS.txt"
}

wait_until_safe() {
  while true; do
    local mem
    mem="$(gpu_mem_used_mb || echo 999999)"
    if main_eval_active; then
      log "main e504a eval active; waiting ${COOLDOWN_SECONDS}s"
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
      if main_eval_active; then
        guard_reason="main_eval_active"
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
  log "steps=${STEPS[*]} samples=${NUM_SAMPLES} edm_batch=${EDM_BATCH} cd_batch=${CD_BATCH} fid_batch=${FID_BATCH}"

  run_guarded edm_imagenet64 \
    "$DG_TWFD_A100_ENV/bin/python" scripts/run_edm_cifar10_eval.py \
      --config configs/experiment/edm_imagenet64_public_eval.yaml \
      --sample-root runs/edm_imagenet64_public_eval_full/samples \
      --eval-root eval/edm_imagenet64_public_eval_full \
      --steps "${STEPS[@]}" \
      --num-samples "$NUM_SAMPLES" \
      --batch "$EDM_BATCH"

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
