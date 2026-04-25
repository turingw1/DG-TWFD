#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash experiments/edm_first/scripts/launch_all.sh <config> <run_tag>" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
config="$1"
tag="$2"

cd "$ROOT_DIR"
bash experiments/edm_first/scripts/launch_train.sh "$config" "$tag"
bash experiments/edm_first/scripts/launch_eval.sh "$config" "$tag" "runs/${tag}/checkpoints/best.pt"
