#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
config="${1:-experiments/edm_first/configs/cifar10_edm_map_onestep_prior_msdefect_8h.yaml}"
run_tag="${2:-edm_first_cifar10_onestep_msdefect_e504a}"
eval_prefix="${3:-edm_first_cifar10_onestep_msdefect_e504a}"
baseline_summary="${4:-eval/edm_first_cifar10_onestep_msdefect_e504a_step250_steps16/reports/summary.json}"

interval_seconds="${DG_TWFD_SUPERVISION_INTERVAL_SECONDS:-3600}"
first_delay_seconds="${DG_TWFD_SUPERVISION_FIRST_DELAY_SECONDS:-0}"
max_hours="${DG_TWFD_SUPERVISION_MAX_HOURS:-7}"
fid_samples="${DG_TWFD_SUPERVISION_FID_SAMPLES:-2048}"
steps=(${DG_TWFD_SUPERVISION_STEPS:-1 2 4 8 16})
run_success_action="${DG_TWFD_SUPERVISION_RUN_SUCCESS_ACTION:-1}"

source "$ROOT_DIR/scripts/server/activate_a100_runtime.sh" >/dev/null
source "$ROOT_DIR/scripts/server/network_profiles.sh"
dg_twfd_net_apply "${DG_TWFD_EDM_EVAL_NETWORK_PROFILE:-proxy}"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:$ROOT_DIR/experiments/edm_first:$ROOT_DIR/refs/edm:${PYTHONPATH:-}"

run_root="runs/${run_tag}"
ckpt_dir="${run_root}/checkpoints"
log_dir="${run_root}/logs"
mkdir -p "$log_dir"
supervisor_log="${log_dir}/hourly_supervisor.log"

exec > >(tee -a "$supervisor_log") 2>&1

echo "hourly_supervisor started at $(date '+%F %T %z')"
echo "config=${config}"
echo "run_tag=${run_tag}"
echo "eval_prefix=${eval_prefix}"
echo "baseline_summary=${baseline_summary}"
echo "interval_seconds=${interval_seconds} max_hours=${max_hours} fid_samples=${fid_samples} steps=${steps[*]}"

if (( first_delay_seconds > 0 )); then
  echo "initial delay: ${first_delay_seconds}s"
  sleep "$first_delay_seconds"
fi

start_epoch="$(date +%s)"
max_seconds="$("$DG_TWFD_A100_ENV/bin/python" - "$max_hours" <<'PY'
import sys
print(int(float(sys.argv[1]) * 3600))
PY
)"

iteration=0
while true; do
  iteration=$((iteration + 1))
  now_epoch="$(date +%s)"
  elapsed=$((now_epoch - start_epoch))
  stamp="$(date '+%Y%m%d_%H%M%S')"
  echo
  echo "========== hourly iteration ${iteration} at $(date '+%F %T %z') elapsed=${elapsed}s =========="

  echo "--- tmux sessions ---"
  tmux ls || true
  echo "--- gpu ---"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  echo "--- latest train log ---"
  tail -n 12 "${log_dir}/train.jsonl" || true
  echo "--- latest watcher log ---"
  tail -n 20 "${log_dir}/watch_eval.log" || true
  echo "--- checkpoints ---"
  find "$ckpt_dir" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort || true

  latest_ckpt="${ckpt_dir}/last.pt"
  if [[ ! -f "$latest_ckpt" ]]; then
    latest_ckpt="$(find "$ckpt_dir" -maxdepth 1 -type f -name 'step*.pt' | sort -V | tail -1 || true)"
  fi
  if [[ -z "${latest_ckpt}" || ! -f "$latest_ckpt" ]]; then
    echo "no checkpoint available; sleeping ${interval_seconds}s"
    sleep "$interval_seconds"
    continue
  fi

  step="$("$DG_TWFD_A100_ENV/bin/python" - "$latest_ckpt" <<'PY'
import sys
import torch
try:
    ckpt = torch.load(sys.argv[1], map_location="cpu")
except Exception:
    print("")
    raise SystemExit(0)
print(int(ckpt.get("step", 0)))
PY
)"
  if [[ -z "$step" || "$step" == "0" ]]; then
    echo "checkpoint not readable: ${latest_ckpt}; sleeping ${interval_seconds}s"
    sleep "$interval_seconds"
    continue
  fi

  frozen="${ckpt_dir}/hourly_step${step}.pt"
  if [[ ! -f "$frozen" ]]; then
    tmp="${frozen}.tmp"
    cp -a "$latest_ckpt" "$tmp"
    mv "$tmp" "$frozen"
  fi

  eval_tag="${eval_prefix}_hourly_step${step}_${stamp}"
  eval_root="eval/${eval_tag}"
  mkdir -p "$eval_root"
  echo "--- evaluating checkpoint step=${step} eval_tag=${eval_tag} ---"
  CUDA_VISIBLE_DEVICES="${INFER_CUDA_VISIBLE_DEVICES:-0}" "$DG_TWFD_A100_ENV/bin/python" \
    experiments/edm_first/eval_edm_map.py \
    --config "$config" \
    --checkpoint "$frozen" \
    --eval-root "$eval_root" \
    --warp-mode auto \
    --steps "${steps[@]}" \
    --fid-samples "$fid_samples" \
    2>&1 | tee "$eval_root/eval.stdout_stderr.txt"

  "$DG_TWFD_A100_ENV/bin/python" \
    experiments/edm_first/scripts/check_fid_thresholds.py \
    --baseline-summary "$baseline_summary" \
    --summary "$eval_root/reports/summary.json" \
    --out-dir "$eval_root/reports" \
    --target-ratio "${DG_TWFD_EVAL_TARGET_RATIO:-0.5}" \
    --primary-step "${DG_TWFD_EVAL_PRIMARY_STEP:-1}"

  echo "--- summary ---"
  cat "$eval_root/reports/summary.csv"
  echo "--- threshold verdict ---"
  cat "$eval_root/reports/threshold_verdict.json"

  backup_root="/temp/Zhengwei/DG-TWFD-backups/experiment_evidence/${eval_tag}"
  if [[ -d "/temp/Zhengwei" ]]; then
    mkdir -p "$backup_root/eval"
    cp -a "$eval_root/reports" "$backup_root/eval/" 2>/dev/null || true
    for step_dir in "$eval_root"/steps*; do
      [[ -d "$step_dir" ]] || continue
      name="$(basename "$step_dir")"
      mkdir -p "$backup_root/eval/$name"
      cp -a "$step_dir/metrics.json" "$step_dir/fixed_seed_grid.png" "$backup_root/eval/$name/" 2>/dev/null || true
    done
    find "$backup_root" -maxdepth 4 -type f | sort > "$backup_root/MANIFEST.txt"
  fi

  primary_target_met="$("$DG_TWFD_A100_ENV/bin/python" - "$eval_root/reports/threshold_verdict.json" <<'PY'
import json
import sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
print("1" if payload.get("primary_target_met") else "0")
PY
)"
  if [[ "$primary_target_met" == "1" ]]; then
    echo "primary FID target met; preparing timewarp follow-up"
    if [[ "$run_success_action" == "1" ]]; then
      bash experiments/edm_first/scripts/launch_timewarp_followup.sh "$frozen"
    fi
    exit 0
  fi

  now_epoch="$(date +%s)"
  elapsed=$((now_epoch - start_epoch))
  if (( elapsed >= max_seconds )); then
    report_path="${run_root}/reports/hourly_supervision_blockers_${stamp}.md"
    "$DG_TWFD_A100_ENV/bin/python" \
      experiments/edm_first/scripts/analyze_hourly_supervision.py \
      --run-tag "$run_tag" \
      --eval-prefix "$eval_prefix" \
      --baseline-summary "$baseline_summary" \
      --repo-root "$ROOT_DIR" \
      --out "$report_path"
    if [[ -d "/temp/Zhengwei" ]]; then
      backup_report="/temp/Zhengwei/DG-TWFD-backups/experiment_evidence/${run_tag}_hourly_supervision_blockers_${stamp}.md"
      cp -a "$report_path" "$backup_report" 2>/dev/null || true
    fi
    echo "7h supervision window elapsed without hitting target; blocker report: ${report_path}"
    exit 0
  fi

  echo "sleeping ${interval_seconds}s"
  sleep "$interval_seconds"
done
