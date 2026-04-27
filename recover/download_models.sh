#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-DG-TWFD}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/home/ma-user/workspace/Zhengwei}"
PROJECT_ROOT="${PROJECT_ROOT:-${WORKSPACE_ROOT}/${PROJECT_NAME}}"

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "Project workspace missing: $PROJECT_ROOT" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

if [[ -f scripts/clone_reference_repos.sh ]]; then
  bash scripts/clone_reference_repos.sh || true
fi

if [[ -f scripts/baselines/download_baseline_assets.sh ]]; then
  bash scripts/baselines/download_baseline_assets.sh || true
fi

echo "Model/reference recovery commands completed."
