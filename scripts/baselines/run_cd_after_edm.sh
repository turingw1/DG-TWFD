#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source scripts/server/activate_a100_runtime.sh >/dev/null
source scripts/server/network_profiles.sh
dg_twfd_net_apply "${DG_TWFD_BASELINE_NETWORK_PROFILE:-proxy}"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:$ROOT_DIR/refs/edm:$ROOT_DIR/refs/consistency_models:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMPI_MCA_btl="${OMPI_MCA_btl:-^openib}"
export OMPI_MCA_btl_openib_warn_no_device_params_found="${OMPI_MCA_btl_openib_warn_no_device_params_found:-0}"

wait_session="${WAIT_FOR_TMUX_SESSION:-baseline_external_full}"
run_tag="${BASELINE_RUN_TAG:-cd_imagenet64_lpips_full_20260426}"
log_root="runs/${run_tag}/logs"
mkdir -p "$log_root"
log_file="$log_root/cd_after_edm.log"

log() {
  echo "[$(date -Is)] $*" | tee -a "$log_file"
}

log "waiting for tmux session ${wait_session} to finish"
while tmux has-session -t "$wait_session" 2>/dev/null; do
  sleep "${WAIT_POLL_SECONDS:-300}"
done
log "wait session finished; starting CD ImageNet64 baseline"

ckpt="${CD_IMAGENET64_CKPT:-/cache/Zhengwei/DG-TWFD-runtime/checkpoints/baselines/consistency_models/cd_imagenet64_lpips.pt}"
if [[ ! -s "$ckpt" ]]; then
  log "missing checkpoint: $ckpt"
  exit 1
fi

steps=(${CD_BASELINE_STEPS:-1 2 4 8})
num_samples="${CD_NUM_SAMPLES:-50000}"

status=0
for batch in "${CD_BATCH:-8}" 4 2 1; do
  log "running CD ImageNet64 batch=${batch} steps=${steps[*]} num_samples=${num_samples}"
  if "$DG_TWFD_A100_ENV/bin/python" scripts/baselines/run_cd_imagenet64_eval.py \
      --checkpoint "$ckpt" \
      --sample-root runs/cd_imagenet64_lpips_full/samples \
      --eval-root eval/cd_imagenet64_lpips_full \
      --csv-out results/baselines/baseline_cd_imagenet64.csv \
      --steps "${steps[@]}" \
      --num-samples "$num_samples" \
      --batch "$batch" \
      2>&1 | tee "$log_root/cd_imagenet64_batch${batch}.stdout_stderr.txt"; then
    status=0
    break
  fi
  status=1
  log "CD ImageNet64 failed with batch=${batch}; retrying smaller batch"
done

"$DG_TWFD_A100_ENV/bin/python" scripts/baselines/export_unified_baseline_csv.py --write-empty | tee -a "$log_file"
if [[ -d /temp/Zhengwei ]]; then
  backup_root="/temp/Zhengwei/DG-TWFD-backups/experiment_evidence/baselines_$(date +%Y%m%d)"
  mkdir -p "$backup_root"
  cp -a results/baselines/. "$backup_root/"
  find "$backup_root" -maxdepth 1 -type f | sort > "$backup_root/MANIFEST.txt"
fi

log "CD ImageNet64 baseline finished status=${status}"
exit "$status"
