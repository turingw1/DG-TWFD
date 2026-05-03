#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: bash experiments/edm_first/scripts/watch_eval_budget_checkpoints.sh <config> <run_tag> <eval_prefix> <start_step> [interval] [fid_samples] [steps...]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
config="$1"
run_tag="$2"
eval_prefix="$3"
next_step="$4"
interval="${5:-250}"
fid_samples="${6:-2048}"
shift 6 || true
steps=("$@")
if [[ ${#steps[@]} -eq 0 ]]; then
  steps=(1 2 4 8 16)
fi

source "$ROOT_DIR/scripts/server/activate_a100_runtime.sh" >/dev/null
source "$ROOT_DIR/scripts/server/network_profiles.sh"
dg_twfd_net_apply "${DG_TWFD_EDM_EVAL_NETWORK_PROFILE:-proxy}"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:$ROOT_DIR/experiments/edm_first:$ROOT_DIR/refs/edm:${PYTHONPATH:-}"

run_root="runs/${run_tag}"
ckpt_dir="${run_root}/checkpoints"
last_ckpt="${ckpt_dir}/last.pt"
watch_log="${run_root}/logs/watch_budget_eval.log"
mkdir -p "${run_root}/logs"

echo "watch_budget_eval started run_tag=${run_tag} next_step=${next_step} interval=${interval} fid_samples=${fid_samples} steps=${steps[*]}" | tee -a "$watch_log"

while true; do
  if [[ ! -f "$last_ckpt" ]]; then
    echo "last checkpoint missing: ${last_ckpt}" | tee -a "$watch_log"
    sleep 180
    continue
  fi

  step="$("$DG_TWFD_A100_ENV/bin/python" - "$last_ckpt" <<'PY'
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
    echo "checkpoint not readable yet: ${last_ckpt}" | tee -a "$watch_log"
    sleep 180
    continue
  fi

  if (( step < next_step )); then
    sleep 180
    continue
  fi

  eval_tag="${eval_prefix}_step${step}_budget"
  eval_root="eval/${eval_tag}"
  if [[ -d "${eval_root}/reports" ]]; then
    echo "budget checkpoint step=${step} already has evaluation; advancing" | tee -a "$watch_log"
    next_step=$(( ((step / interval) + 1) * interval ))
    sleep 180
    continue
  fi

  frozen="${ckpt_dir}/step${step}.pt"
  if [[ ! -f "$frozen" ]]; then
    tmp="${frozen}.tmp"
    cp -a "$last_ckpt" "$tmp"
    mv "$tmp" "$frozen"
  fi

  mkdir -p "$eval_root"
  echo "evaluating budget checkpoint step=${step} eval_tag=${eval_tag}" | tee -a "$watch_log"
  CUDA_VISIBLE_DEVICES="${INFER_CUDA_VISIBLE_DEVICES:-0}" "$DG_TWFD_A100_ENV/bin/python" \
    experiments/edm_first/eval_edm_map.py \
    --config "$config" \
    --checkpoint "$frozen" \
    --eval-root "$eval_root" \
    --warp-mode budget \
    --steps "${steps[@]}" \
    --fid-samples "$fid_samples" \
    2>&1 | tee "$eval_root/eval.stdout_stderr.txt"

  next_step=$(( ((step / interval) + 1) * interval ))
  sleep 180
done
