#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: bash experiments/edm_first/scripts/watch_eval_checkpoints.sh <config> <run_tag> <eval_prefix> <start_step> [interval] [fid_samples] [steps...]" >&2
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

compare_identity="$("$DG_TWFD_A100_ENV/bin/python" - "$config" <<'PY'
import os
import sys
from src.edm_map_lib import load_config

mode = os.environ.get("DG_TWFD_EVAL_COMPARE_IDENTITY", "auto").strip().lower()
if mode in {"0", "false", "no", "never", "off"}:
    print("0")
    raise SystemExit(0)
if mode in {"1", "true", "yes", "always", "on"}:
    print("1")
    raise SystemExit(0)
cfg = load_config(sys.argv[1])
print("1" if bool(cfg.get("timewarp", {}).get("enabled", True)) else "0")
PY
)"

run_root="runs/${run_tag}"
ckpt_dir="${run_root}/checkpoints"
last_ckpt="${ckpt_dir}/last.pt"
watch_log="${run_root}/logs/watch_eval.log"
mkdir -p "${run_root}/logs"

echo "watch_eval started run_tag=${run_tag} next_step=${next_step} interval=${interval} fid_samples=${fid_samples} steps=${steps[*]} compare_identity=${compare_identity}" | tee -a "$watch_log"

while true; do
  if [[ ! -f "$last_ckpt" ]]; then
    echo "last checkpoint missing: ${last_ckpt}" | tee -a "$watch_log"
    sleep 180
    continue
  fi

  step="$("$DG_TWFD_A100_ENV/bin/python" - "$last_ckpt" <<'PY'
import sys
import torch
path = sys.argv[1]
try:
    ckpt = torch.load(path, map_location="cpu")
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

  frozen="${ckpt_dir}/step${step}.pt"
  if [[ ! -f "$frozen" ]]; then
    tmp="${frozen}.tmp"
    cp -a "$last_ckpt" "$tmp"
    mv "$tmp" "$frozen"
  fi

  eval_tag="${eval_prefix}_step${step}"
  eval_root="eval/${eval_tag}"
  mkdir -p "$eval_root"
  echo "evaluating checkpoint step=${step} eval_tag=${eval_tag}" | tee -a "$watch_log"
  CUDA_VISIBLE_DEVICES="${INFER_CUDA_VISIBLE_DEVICES:-0}" "$DG_TWFD_A100_ENV/bin/python" \
    experiments/edm_first/eval_edm_map.py \
    --config "$config" \
    --checkpoint "$frozen" \
    --eval-root "$eval_root" \
    --warp-mode auto \
    --steps "${steps[@]}" \
    --fid-samples "$fid_samples" \
    2>&1 | tee "$eval_root/eval.stdout_stderr.txt"

  if [[ -n "${DG_TWFD_EVAL_BASELINE_SUMMARY:-}" && -f "${DG_TWFD_EVAL_BASELINE_SUMMARY}" ]]; then
    "$DG_TWFD_A100_ENV/bin/python" \
      experiments/edm_first/scripts/check_fid_thresholds.py \
      --baseline-summary "${DG_TWFD_EVAL_BASELINE_SUMMARY}" \
      --summary "$eval_root/reports/summary.json" \
      --out-dir "$eval_root/reports" \
      --target-ratio "${DG_TWFD_EVAL_TARGET_RATIO:-0.5}" \
      --primary-step "${DG_TWFD_EVAL_PRIMARY_STEP:-1}" \
      2>&1 | tee -a "$watch_log"
  fi

  identity_eval_root=""
  if [[ "$compare_identity" == "1" ]]; then
    identity_eval_root="eval/${eval_tag}_identity"
    mkdir -p "$identity_eval_root"
    echo "evaluating identity-clock comparison step=${step} eval_tag=${eval_tag}_identity" | tee -a "$watch_log"
    CUDA_VISIBLE_DEVICES="${INFER_CUDA_VISIBLE_DEVICES:-0}" "$DG_TWFD_A100_ENV/bin/python" \
      experiments/edm_first/eval_edm_map.py \
      --config "$config" \
      --checkpoint "$frozen" \
      --eval-root "$identity_eval_root" \
      --warp-mode identity \
      --steps "${steps[@]}" \
      --fid-samples "$fid_samples" \
      2>&1 | tee "$identity_eval_root/eval.stdout_stderr.txt"

    "$DG_TWFD_A100_ENV/bin/python" - "$eval_root/reports/summary.json" "$identity_eval_root/reports/summary.json" "$eval_root/reports/timewarp_comparison" <<'PY'
import csv
import json
import sys
from pathlib import Path

auto_path = Path(sys.argv[1])
identity_path = Path(sys.argv[2])
out_prefix = Path(sys.argv[3])
auto = json.loads(auto_path.read_text(encoding="utf-8"))
identity = json.loads(identity_path.read_text(encoding="utf-8"))
identity_by_step = {int(item["step_count"]): item for item in identity["records"]}
rows = []
for auto_row in auto["records"]:
    step = int(auto_row["step_count"])
    if step not in identity_by_step:
        continue
    identity_row = identity_by_step[step]
    auto_fid = float(auto_row["fid"])
    identity_fid = float(identity_row["fid"])
    delta = auto_fid - identity_fid
    rows.append(
        {
            "step_count": step,
            "auto_fid": auto_fid,
            "identity_fid": identity_fid,
            "fid_delta_auto_minus_identity": delta,
            "fid_ratio_auto_over_identity": auto_fid / identity_fid if identity_fid else None,
            "verdict": "warp_better" if delta < 0.0 else "identity_better_or_equal",
            "auto_u_grid": auto_row.get("u_grid"),
            "identity_u_grid": identity_row.get("u_grid"),
        }
    )
out_prefix.parent.mkdir(parents=True, exist_ok=True)
(out_prefix.with_suffix(".json")).write_text(json.dumps({"records": rows}, indent=2), encoding="utf-8")
with (out_prefix.with_suffix(".csv")).open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["step_count"])
    writer.writeheader()
    writer.writerows(rows)
PY
  fi

  project_name="$(basename "$ROOT_DIR")"
  project_temp="${DG_TWFD_PROJECT_TEMP:-/temp/Zhengwei/projects/${project_name}}"
  backup_root="${project_temp}/critical/eval/${eval_tag}"
  if [[ -d "/temp/Zhengwei" ]]; then
    mkdir -p "$backup_root/runs/checkpoints" "$backup_root/eval"
    cp -a "$frozen" "$backup_root/runs/checkpoints/"
    cp -a "$eval_root/reports" "$backup_root/eval/" 2>/dev/null || true
    for step_dir in "$eval_root"/steps*; do
      [[ -d "$step_dir" ]] || continue
      name="$(basename "$step_dir")"
      mkdir -p "$backup_root/eval/$name"
      cp -a "$step_dir/metrics.json" "$step_dir/fixed_seed_grid.png" "$backup_root/eval/$name/" 2>/dev/null || true
    done
    if [[ -n "$identity_eval_root" && -d "$identity_eval_root" ]]; then
      cp -a "$identity_eval_root/reports" "$backup_root/eval_identity/" 2>/dev/null || true
      for step_dir in "$identity_eval_root"/steps*; do
        [[ -d "$step_dir" ]] || continue
        name="$(basename "$step_dir")"
        mkdir -p "$backup_root/eval_identity/$name"
        cp -a "$step_dir/metrics.json" "$step_dir/fixed_seed_grid.png" "$backup_root/eval_identity/$name/" 2>/dev/null || true
      done
    fi
    cp -a "$eval_root/reports/timewarp_comparison."* "$backup_root/eval/reports/" 2>/dev/null || true
    find "$backup_root" -maxdepth 4 -type f | sort > "$backup_root/MANIFEST.txt"
  fi

  next_step=$(( ((step / interval) + 1) * interval ))
  echo "evaluation complete step=${step}; next_step=${next_step}" | tee -a "$watch_log"
  sleep 180
done
