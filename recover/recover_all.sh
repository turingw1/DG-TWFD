#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-DG-TWFD}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/home/ma-user/workspace/Zhengwei}"
PROJECT_ROOT="${PROJECT_ROOT:-${WORKSPACE_ROOT}/${PROJECT_NAME}}"
PROJECT_TEMP="${PROJECT_TEMP:-/temp/Zhengwei/projects/${PROJECT_NAME}}"
CACHE_ROOT="${CACHE_ROOT:-/cache/Zhengwei/${PROJECT_NAME}}"
REMOTE_URL="${REMOTE_URL:-git@github.com:turingw1/DG-TWFD.git}"
BRANCH="${BRANCH:-DG_TWFD_v3}"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$WORKSPACE_ROOT" "$PROJECT_TEMP/logs" /cache/Zhengwei

if [[ -e "$PROJECT_ROOT" ]]; then
  backup="${WORKSPACE_ROOT}/${PROJECT_NAME}.pre_recover.${TS}"
  mv "$PROJECT_ROOT" "$backup"
  echo "Backed up existing workspace to $backup"
fi

git clone "$REMOTE_URL" "$PROJECT_ROOT"
cd "$PROJECT_ROOT"
git checkout "$BRANCH"

if [[ -f "$PROJECT_TEMP/critical/code/latest/repo_head.bundle" ]]; then
  git fetch "$PROJECT_TEMP/critical/code/latest/repo_head.bundle" HEAD:refs/tmp/dg_twfd_recovery_head || true
fi
if [[ -s "$PROJECT_TEMP/critical/code/latest/dirty.patch" ]]; then
  git apply "$PROJECT_TEMP/critical/code/latest/dirty.patch" || true
fi
if [[ -f "$PROJECT_TEMP/critical/code/latest/untracked_files.tar.gz" ]]; then
  tar -xzf "$PROJECT_TEMP/critical/code/latest/untracked_files.tar.gz" -C "$PROJECT_ROOT"
fi

if [[ ! -e "$CACHE_ROOT" && -d /cache/Zhengwei/DG-TWFD-runtime ]]; then
  ln -s /cache/Zhengwei/DG-TWFD-runtime "$CACHE_ROOT"
fi
mkdir -p "$CACHE_ROOT"

bash "$PROJECT_TEMP/recover/download_models.sh" || true
bash "$PROJECT_TEMP/recover/download_dataset.sh" || true

cat <<REPORT
Recovery finished.
Project: $PROJECT_NAME
Workspace: $PROJECT_ROOT
Cache: $CACHE_ROOT
Project temp: $PROJECT_TEMP

Manual checks:
  git -C "$PROJECT_ROOT" status --short --branch
  source "$PROJECT_ROOT/scripts/server/activate_a100_runtime.sh"
REPORT
