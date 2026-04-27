#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash experiments/edm_first/scripts/launch_prior_fullstack_timewarp.sh <source_checkpoint> [run_tag] [tmux_session]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source_checkpoint="$1"
config="${DG_TWFD_FULLSTACK_CONFIG:-experiments/edm_first/configs/cifar10_edm_map_prior_fullstack_timewarp_12h.yaml}"

source "$ROOT_DIR/scripts/server/activate_a100_runtime.sh" >/dev/null
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:$ROOT_DIR/experiments/edm_first:$ROOT_DIR/refs/edm:${PYTHONPATH:-}"

source_step="$("$DG_TWFD_A100_ENV/bin/python" - "$source_checkpoint" <<'PY'
import sys
import torch

ckpt = torch.load(sys.argv[1], map_location="cpu")
print(int(ckpt.get("step", 0)))
PY
)"

run_tag="${2:-edm_first_cifar10_prior_fullstack_timewarp_v11a_from_step${source_step}}"
tmux_session="${3:-v11a_fullstack_tw}"

if tmux has-session -t "$tmux_session" 2>/dev/null; then
  echo "fullstack timewarp train already running: ${tmux_session}"
else
  tmux new-session -d -s "$tmux_session" \
    "cd '$ROOT_DIR' && bash experiments/edm_first/scripts/launch_train.sh '$config' '$run_tag' '$source_checkpoint'"
  echo "fullstack timewarp train started: ${tmux_session}"
fi

if tmux has-session -t "${tmux_session}_eval_watch" 2>/dev/null; then
  echo "fullstack timewarp eval watcher already running: ${tmux_session}_eval_watch"
else
  tmux new-session -d -s "${tmux_session}_eval_watch" \
    "cd '$ROOT_DIR' && export DG_TWFD_EVAL_COMPARE_IDENTITY=1 && export DG_TWFD_EVAL_BASELINE_SUMMARY=eval/edm_first_cifar10_onestep_msdefect_e504a_resume_from1250_step1750/reports/summary.json && export DG_TWFD_EVAL_TARGET_RATIO=1.0 && export DG_TWFD_EVAL_PRIMARY_STEP=1 && bash experiments/edm_first/scripts/watch_eval_checkpoints.sh '$config' '$run_tag' '$run_tag' 250 250 2048 1 2 4 8 16"
  echo "fullstack timewarp eval watcher started: ${tmux_session}_eval_watch"
fi

if tmux has-session -t "${tmux_session}_backup" 2>/dev/null; then
  echo "fullstack timewarp project backup watcher already running: ${tmux_session}_backup"
else
  tmux new-session -d -s "${tmux_session}_backup" \
    "cd '$ROOT_DIR' && bash experiments/edm_first/scripts/watch_project_backup_v11.sh '$run_tag' 180 '$(basename "$ROOT_DIR")'"
  echo "fullstack timewarp project backup watcher started: ${tmux_session}_backup"
fi

echo "run_tag=${run_tag}"
