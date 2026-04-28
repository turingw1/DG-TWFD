#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
run_tag="${1:-edm_first_cifar10_prior_fullstack_timewarp_v11a_from_step1750}"
project_name="${2:-$(basename "$ROOT_DIR")}"
hours="${DG_TWFD_SUPERVISE_HOURS:-5}"
interval="${DG_TWFD_SUPERVISE_INTERVAL_SECONDS:-3600}"
project_temp="${DG_TWFD_PROJECT_TEMP:-/temp/Zhengwei/projects/${project_name}}"
stamp="$(date +%Y%m%d_%H%M%S)"
run_root="${ROOT_DIR}/runs/${run_tag}"
out_dir="${run_root}/reports/hourly_supervision_v11_${stamp}"
jsonl="${out_dir}/summary.jsonl"
md="${out_dir}/final_report.md"
temp_log="${project_temp}/logs/${run_tag}_hourly_supervision_${stamp}.log"
temp_report="${project_temp}/critical/analysis/${run_tag}_hourly_supervision_${stamp}.md"

mkdir -p "$out_dir" "$(dirname "$temp_log")" "$(dirname "$temp_report")"

collect_once() {
  local iteration="$1"
  source "$ROOT_DIR/scripts/server/activate_a100_runtime.sh" >/dev/null 2>&1 || true
  python_bin="${DG_TWFD_A100_ENV:-python}"
  if [[ -x "${DG_TWFD_A100_ENV:-}/bin/python" ]]; then
    python_bin="${DG_TWFD_A100_ENV}/bin/python"
  fi
  "$python_bin" - "$ROOT_DIR" "$run_tag" "$project_temp" "$iteration" "$jsonl" <<'PY'
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
run_tag = sys.argv[2]
project_temp = Path(sys.argv[3])
iteration = int(sys.argv[4])
jsonl = Path(sys.argv[5])
run_root = root / "runs" / run_tag

def read_jsonl_tail(path: Path, limit: int = 24) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]

def eval_step_from_path(path: Path) -> int:
    matches = re.findall(r"_step(\d+)", str(path))
    return int(matches[-1]) if matches else -1

def latest_eval_summary(suffix: str = "") -> tuple[int | None, list[dict]]:
    pattern = f"{run_tag}_step*{suffix}/reports/summary.csv"
    paths = list((root / "eval").glob(pattern))
    if suffix:
        paths = [path for path in paths if path.parent.parent.name.endswith(suffix)]
    else:
        paths = [
            path
            for path in paths
            if not path.parent.parent.name.endswith("_identity")
            and not path.parent.parent.name.endswith("_budget")
        ]
    paths = sorted(paths, key=eval_step_from_path)
    if not paths:
        return None, []
    path = paths[-1]
    step = eval_step_from_path(path)
    step = step if step >= 0 else None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return step, rows

def latest_comparison() -> tuple[int | None, list[dict]]:
    paths = sorted((root / "eval").glob(f"{run_tag}_step*/reports/timewarp_comparison.csv"), key=eval_step_from_path)
    if not paths:
        return None, []
    path = paths[-1]
    step = eval_step_from_path(path)
    step = step if step >= 0 else None
    with path.open("r", encoding="utf-8", newline="") as handle:
        return step, list(csv.DictReader(handle))

def compact_fids(rows: list[dict]) -> dict[str, float]:
    out = {}
    for row in rows:
        try:
            out[str(int(float(row["step_count"])))] = round(float(row["fid"]), 4)
        except Exception:
            pass
    return out

def run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    except Exception as exc:
        return f"unavailable: {exc}"

train_rows = read_jsonl_tail(run_root / "logs" / "train.jsonl")
latest_train = train_rows[-1] if train_rows else {}
window = train_rows[-8:]

def avg(key: str) -> float | None:
    vals = [float(row[key]) for row in window if key in row and row[key] is not None]
    return round(sum(vals) / len(vals), 6) if vals else None

auto_step, auto_rows = latest_eval_summary("")
identity_step, identity_rows = latest_eval_summary("_identity")
budget_step, budget_rows = latest_eval_summary("_budget")
comparison_step, comparison_rows = latest_comparison()

ckpt_dir = run_root / "checkpoints"
ckpts = sorted(ckpt_dir.glob("step*.pt"), key=lambda p: int(re.search(r"step(\d+)", p.name).group(1)) if re.search(r"step(\d+)", p.name) else -1)
last_ckpt = ckpts[-1].name if ckpts else None
last_sync_path = project_temp / "critical" / "runs" / run_tag / "LAST_SYNC.txt"
last_backup_sync = last_sync_path.read_text(encoding="utf-8").strip() if last_sync_path.exists() else None

summary = {
    "iteration": iteration,
    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "train_step": int(latest_train.get("step", 0) or 0),
    "loss": latest_train.get("loss"),
    "anchor_loss": latest_train.get("anchor_loss"),
    "bridge_loss": latest_train.get("bridge_loss"),
    "defect_loss": latest_train.get("defect_loss"),
    "qmax": latest_train.get("max_qphi_over_qbase"),
    "entropy": latest_train.get("entropy_q_phi"),
    "window_avg_loss": avg("loss"),
    "window_avg_anchor": avg("anchor_loss"),
    "window_avg_bridge": avg("bridge_loss"),
    "window_avg_defect": avg("defect_loss"),
    "latest_checkpoint": last_ckpt,
    "latest_auto_eval_step": auto_step,
    "auto_fid": compact_fids(auto_rows),
    "latest_identity_eval_step": identity_step,
    "identity_fid": compact_fids(identity_rows),
    "latest_budget_eval_step": budget_step,
    "budget_fid": compact_fids(budget_rows),
    "latest_comparison_step": comparison_step,
    "timewarp_delta": {
        str(int(float(row["step_count"]))): round(float(row["fid_delta_auto_minus_identity"]), 4)
        for row in comparison_rows
        if row.get("step_count") and row.get("fid_delta_auto_minus_identity")
    },
    "backup_last_sync": last_backup_sync,
    "gpu": run_cmd(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"]),
    "tmux": run_cmd(["tmux", "ls"]),
}

warnings = []
if not summary["train_step"]:
    warnings.append("missing_train_log")
supervise_tmux_session = os.environ.get("DG_TWFD_SUPERVISE_TMUX_SESSION", "").strip()
if supervise_tmux_session and supervise_tmux_session not in summary["tmux"]:
    warnings.append("train_tmux_missing")
if summary["train_step"] >= 1000 and summary.get("qmax") and float(summary["qmax"]) < 1.02:
    warnings.append("timewarp_density_still_near_identity")
if summary["latest_auto_eval_step"] and summary["latest_identity_eval_step"] == summary["latest_auto_eval_step"]:
    deltas = [float(v) for v in summary["timewarp_delta"].values()]
    if deltas and min(deltas) >= 0.0:
        warnings.append("auto_warp_not_better_than_identity_yet")
summary["warnings"] = warnings

jsonl.parent.mkdir(parents=True, exist_ok=True)
with jsonl.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

auto = ",".join(f"{k}:{v}" for k, v in summary["auto_fid"].items()) or "-"
budget = ",".join(f"{k}:{v}" for k, v in summary["budget_fid"].items()) or "-"
delta = ",".join(f"{k}:{v}" for k, v in summary["timewarp_delta"].items()) or "-"
warn = ",".join(warnings) or "ok"

def fmt(value, digits: int = 6) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)

print(
    f"[hour {iteration}] step={summary['train_step']} "
    f"loss={fmt(summary['loss'])} anchor={fmt(summary['anchor_loss'])} "
    f"bridge={fmt(summary['bridge_loss'])} defect={fmt(summary['defect_loss'])} "
    f"qmax={fmt(summary['qmax'], 4)} eval_step={summary['latest_auto_eval_step']} "
    f"auto_fid={auto} budget_fid={budget} warp_delta={delta} status={warn}"
)
PY
}

write_report() {
  python3 - "$jsonl" "$md" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

jsonl = Path(sys.argv[1])
md = Path(sys.argv[2])
rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]

lines = [f"# {len(rows)}-Point Hourly Supervision Report", ""]
lines.append("| hour | time | train step | loss | anchor | bridge | defect | qmax | eval step | auto FID@1/2/4/8/16 | budget FID@1/2/4/8/16 | warnings |")
lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
for row in rows:
    fids = row.get("auto_fid", {})
    fid_text = "/".join(str(fids.get(str(step), "-")) for step in [1, 2, 4, 8, 16])
    budget_fids = row.get("budget_fid", {})
    budget_text = "/".join(str(budget_fids.get(str(step), "-")) for step in [1, 2, 4, 8, 16])

    def fmt(value, digits: int = 6) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value):.{digits}f}"
        except Exception:
            return str(value)

    lines.append(
        f"| {row['iteration']} | {row['time']} | {row['train_step']} | "
        f"{fmt(row['loss'])} | {fmt(row['anchor_loss'])} | "
        f"{fmt(row['bridge_loss'])} | {fmt(row['defect_loss'])} | "
        f"{fmt(row['qmax'], 4)} | {row.get('latest_auto_eval_step') or '-'} | "
        f"{fid_text} | {budget_text} | {', '.join(row.get('warnings') or ['ok'])} |"
    )

lines.extend(["", "## Final Snapshot", ""])
if rows:
    last = rows[-1]
    lines.append(f"- Latest train step: {last['train_step']}")
    lines.append(f"- Latest auto eval step: {last.get('latest_auto_eval_step')}")
    lines.append(f"- Latest identity eval step: {last.get('latest_identity_eval_step')}")
    lines.append(f"- Latest budget eval step: {last.get('latest_budget_eval_step')}")
    lines.append(f"- Timewarp deltas auto-minus-identity: {last.get('timewarp_delta')}")
    lines.append(f"- Backup last sync: {last.get('backup_last_sync')}")
    lines.append(f"- GPU: `{last.get('gpu')}`")

md.parent.mkdir(parents=True, exist_ok=True)
md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(md)
PY
}

echo "hourly v11 supervision started run_tag=${run_tag} hours=${hours} interval=${interval}" | tee -a "$temp_log"
collect_once 0 | tee -a "$temp_log"
for iteration in $(seq 1 "$hours"); do
  sleep "$interval"
  collect_once "$iteration" | tee -a "$temp_log"
done
write_report | tee -a "$temp_log"
cp -af "$md" "$temp_report" 2>/dev/null || true
echo "hourly v11 supervision finished report=${md} temp_report=${temp_report}" | tee -a "$temp_log"
