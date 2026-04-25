#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/server/activate_a100_runtime.sh"

label="${1:-manual_backup}"
timestamp="$(date +%Y%m%d_%H%M%S)"
dest="${DG_TWFD_TMP_BACKUP_ROOT}/${timestamp}_${label}"

mkdir -p "$dest"

if [[ ! -d "$TMPDIR" ]]; then
  echo "TMPDIR does not exist: $TMPDIR" >&2
  exit 1
fi

if command -v rsync >/dev/null 2>&1; then
  rsync -a "$TMPDIR"/ "$dest"/
else
  cp -a "$TMPDIR"/. "$dest"/
fi

echo "Backed up TMPDIR"
echo "  source=$TMPDIR"
echo "  dest=$dest"
