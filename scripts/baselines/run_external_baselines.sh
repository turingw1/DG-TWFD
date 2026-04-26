#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source scripts/server/activate_a100_runtime.sh >/dev/null
source scripts/server/network_profiles.sh
dg_twfd_net_apply "${DG_TWFD_BASELINE_NETWORK_PROFILE:-proxy}"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:$ROOT_DIR/refs/edm:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_tag="${BASELINE_RUN_TAG:-external_baselines_$(date +%Y%m%d_%H%M%S)}"
log_root="runs/${run_tag}/logs"
mkdir -p "$log_root" results/baselines

steps=(${BASELINE_STEPS:-1 2 4 8})
num_samples="${BASELINE_NUM_SAMPLES:-50000}"
cifar_batch="${EDM_CIFAR10_BATCH:-16}"
imagenet64_batch="${EDM_IMAGENET64_BATCH:-8}"

log_file="$log_root/baseline_runner.log"

log() {
  echo "[$(date -Is)] $*" | tee -a "$log_file"
}

refresh_exports() {
  "$DG_TWFD_A100_ENV/bin/python" scripts/baselines/export_unified_baseline_csv.py --write-empty | tee -a "$log_file"
  if [[ -d /temp/Zhengwei ]]; then
    backup_root="/temp/Zhengwei/DG-TWFD-backups/experiment_evidence/baselines_$(date +%Y%m%d)"
    mkdir -p "$backup_root"
    cp -a results/baselines/. "$backup_root/"
    find "$backup_root" -maxdepth 1 -type f | sort > "$backup_root/MANIFEST.txt"
  fi
}

probe_assets() {
  "$DG_TWFD_A100_ENV/bin/python" scripts/baselines/probe_baseline_assets.py \
    --output results/baselines/asset_probe.json | tee -a "$log_file"
}

run_edm_eval() {
  local dataset="$1"
  local config="$2"
  local sample_root="$3"
  local eval_root="$4"
  local batch="$5"

  log "starting EDM ${dataset}: steps=${steps[*]} num_samples=${num_samples} batch=${batch}"
  "$DG_TWFD_A100_ENV/bin/python" scripts/run_edm_cifar10_eval.py \
    --config "$config" \
    --sample-root "$sample_root" \
    --eval-root "$eval_root" \
    --steps "${steps[@]}" \
    --num-samples "$num_samples" \
    --batch "$batch" \
    2>&1 | tee "$log_root/edm_${dataset}.stdout_stderr.txt"
  log "finished EDM ${dataset}"
  refresh_exports
}

run_edm_with_retry() {
  local dataset="$1"
  local config="$2"
  local sample_root="$3"
  local eval_root="$4"
  shift 4
  local batches=("$@")

  local batch
  for batch in "${batches[@]}"; do
    if run_edm_eval "$dataset" "$config" "$sample_root" "$eval_root" "$batch"; then
      return 0
    fi
    log "EDM ${dataset} failed with batch=${batch}; retrying smaller batch if available"
  done
  log "EDM ${dataset} failed for all requested batches"
  return 1
}

main() {
  log "baseline runner started run_tag=${run_tag}"
  log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits | tee -a "$log_file" || true

  probe_assets
  refresh_exports

  status=0
  if [[ "${RUN_EDM_CIFAR10:-1}" == "1" ]]; then
    run_edm_with_retry \
      cifar10 \
      configs/experiment/edm_cifar10_public_eval.yaml \
      runs/edm_cifar10_public_eval_full/samples \
      eval/edm_cifar10_public_eval_full \
      "$cifar_batch" 8 4 || status=1
  fi

  if [[ "${RUN_EDM_IMAGENET64:-1}" == "1" ]]; then
    run_edm_with_retry \
      imagenet64 \
      configs/experiment/edm_imagenet64_public_eval.yaml \
      runs/edm_imagenet64_public_eval_full/samples \
      eval/edm_imagenet64_public_eval_full \
      "$imagenet64_batch" 4 2 || status=1
  fi

  probe_assets
  refresh_exports
  log "baseline runner finished status=${status}"
  exit "$status"
}

main "$@"
