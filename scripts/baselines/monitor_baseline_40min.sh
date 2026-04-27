#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${BASELINE_LOW_VRAM_RUN_TAG:-baseline_low_vram_guarded_20260427}"
BASELINE_SESSION="${BASELINE_TMUX_SESSION:-baseline_low_vram_20260427}"
INTERVAL_SECONDS="${BASELINE_MONITOR_INTERVAL_SECONDS:-2400}"
BACKUP_ROOT="${BASELINE_LIVE_BACKUP_ROOT:-/temp/Zhengwei/DG-TWFD-backups/experiment_evidence/baselines_low_vram_live_20260427}"
LOG_FILE="${BASELINE_MONITOR_LOG:-${BACKUP_ROOT}/monitor_40min.log}"
REQUIRE_TMUX="${BASELINE_MONITOR_REQUIRE_TMUX:-0}"

mkdir -p "$(dirname "$LOG_FILE")"

count_pngs() {
  local path="$1"
  if [[ -d "$path" ]]; then
    find "$path" -type f -name '*.png' | wc -l
  else
    echo 0
  fi
}

snapshot() {
  {
    echo "===== monitor $(date -Is) ====="
    echo "[gpu]"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || true
    echo

    echo "[live-status]"
    cat "$BACKUP_ROOT/STATUS.txt" 2>/dev/null || true
    echo

    echo "[main-train-tail]"
    tail -n 5 runs/edm_first_cifar10_onestep_msdefect_e504a_resume_from1250/logs/train.jsonl 2>/dev/null || true
    echo

    echo "[main-eval-watch-tail]"
    tail -n 40 runs/edm_first_cifar10_onestep_msdefect_e504a_resume_from1250/logs/watch_eval.log 2>/dev/null || true
    echo

    echo "[baseline-guard-tail]"
    tail -n 80 "runs/${RUN_TAG}/logs/guard.log" 2>/dev/null || true
    echo

    echo "[baseline-edm-tail]"
    tail -n 60 "runs/${RUN_TAG}/logs/edm_imagenet64.stdout_stderr.txt" 2>/dev/null || true
    echo

    echo "[baseline-cd-tail]"
    tail -n 60 "runs/${RUN_TAG}/logs/cd_imagenet64_lpips.stdout_stderr.txt" 2>/dev/null || true
    echo

    echo "[sample-counts]"
    for step in 1 2 4 8; do
      echo "edm_imagenet64_step${step}=$(count_pngs "runs/edm_imagenet64_public_eval_full/samples/steps${step}/images")"
      echo "cd_imagenet64_lpips_step${step}=$(count_pngs "runs/cd_imagenet64_lpips_full/samples/steps${step}/images")"
    done
    echo

    echo "[processes]"
    ps -eo pid,ppid,pgid,stat,etime,pcpu,args \
      | grep -E 'run_low_vram_guarded_baselines|run_edm_cifar10_eval.py|run_cd_imagenet64_eval.py|refs/edm/generate.py|train_edm_map.py|eval_edm_map.py' \
      | grep -v grep || true
    echo
  } >> "$LOG_FILE"
}

while true; do
  snapshot
  if grep -q "low-vram baseline guard finished" "runs/${RUN_TAG}/logs/guard.log" 2>/dev/null; then
    echo "===== monitor $(date -Is): baseline finished =====" >> "$LOG_FILE"
    break
  fi
  if [[ "$REQUIRE_TMUX" =~ ^(1|true|yes|on)$ ]] && ! tmux has-session -t "$BASELINE_SESSION" 2>/dev/null; then
    echo "===== monitor $(date -Is): baseline session missing =====" >> "$LOG_FILE"
    break
  fi
  sleep "$INTERVAL_SECONDS"
done
