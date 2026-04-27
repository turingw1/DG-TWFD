#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-DG-TWFD}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/home/ma-user/workspace/Zhengwei}"
PROJECT_ROOT="${PROJECT_ROOT:-${WORKSPACE_ROOT}/${PROJECT_NAME}}"
CACHE_ROOT="${CACHE_ROOT:-/cache/Zhengwei/${PROJECT_NAME}}"
LEGACY_CACHE_ROOT="${LEGACY_CACHE_ROOT:-/cache/Zhengwei/DG-TWFD-runtime}"

mkdir -p "$WORKSPACE_ROOT" /cache/Zhengwei

if [[ ! -e "$CACHE_ROOT" && -d "$LEGACY_CACHE_ROOT" ]]; then
  ln -s "$LEGACY_CACHE_ROOT" "$CACHE_ROOT"
fi
if [[ ! -e "$CACHE_ROOT" ]]; then
  mkdir -p "$CACHE_ROOT"
fi

if [[ -d "$PROJECT_ROOT" ]]; then
  cd "$PROJECT_ROOT"
  if [[ -f scripts/server/activate_a100_runtime.sh ]]; then
    # shellcheck disable=SC1091
    source scripts/server/activate_a100_runtime.sh
  fi
fi

echo "PROJECT_NAME=$PROJECT_NAME"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "CACHE_ROOT=$CACHE_ROOT"
