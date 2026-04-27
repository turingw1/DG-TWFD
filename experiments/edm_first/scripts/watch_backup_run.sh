#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash experiments/edm_first/scripts/watch_backup_run.sh <run_tag> [interval_sec] [backup_root]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
run_tag="$1"
interval="${2:-180}"
backup_root="${3:-/temp/Zhengwei/DG-TWFD-backups/experiment_evidence/${run_tag}_live}"

cd "$ROOT_DIR"
run_root="runs/${run_tag}"
log_file="${run_root}/logs/watch_backup.log"
mkdir -p "${run_root}/logs" "$backup_root/runs/logs" "$backup_root/runs/checkpoints"

copy_stable_file() {
  local src="$1"
  local dst="$2"
  [[ -f "$src" ]] || return 0

  local size1 mtime1 size2 mtime2
  size1="$(stat -c '%s' "$src" 2>/dev/null || true)"
  mtime1="$(stat -c '%Y' "$src" 2>/dev/null || true)"
  [[ -n "$size1" && -n "$mtime1" ]] || return 0
  sleep 2
  size2="$(stat -c '%s' "$src" 2>/dev/null || true)"
  mtime2="$(stat -c '%Y' "$src" 2>/dev/null || true)"
  [[ "$size1" == "$size2" && "$mtime1" == "$mtime2" ]] || return 0

  mkdir -p "$(dirname "$dst")"
  cp -af "$src" "$dst"
}

echo "watch_backup started run_tag=${run_tag} interval=${interval} backup_root=${backup_root}" | tee -a "$log_file"

while true; do
  if [[ ! -d "$run_root" ]]; then
    echo "run root missing: ${run_root}" | tee -a "$log_file"
    sleep "$interval"
    continue
  fi

  mkdir -p "$backup_root/runs/logs" "$backup_root/runs/checkpoints"
  if [[ -d "${run_root}/logs" ]]; then
    while IFS= read -r -d '' log_path; do
      cp -af "$log_path" "$backup_root/runs/logs/$(basename "$log_path")"
    done < <(find "${run_root}/logs" -maxdepth 1 -type f -print0 2>/dev/null)
  fi

  if [[ -d "${run_root}/checkpoints" ]]; then
    while IFS= read -r -d '' ckpt; do
      copy_stable_file "$ckpt" "$backup_root/runs/checkpoints/$(basename "$ckpt")"
    done < <(find "${run_root}/checkpoints" -maxdepth 1 -type f \( -name 'step*.pt' -o -name 'last.pt' -o -name 'best.pt' \) -print0 2>/dev/null)
  fi

  find "$backup_root" -maxdepth 4 -type f | sort > "$backup_root/MANIFEST.txt"
  date -Is > "$backup_root/LAST_SYNC.txt"
  echo "backup sync complete $(date -Is)" | tee -a "$log_file"
  sleep "$interval"
done
