#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
project_name="${1:-$(basename "$ROOT_DIR")}"
project_temp="${DG_TWFD_PROJECT_TEMP:-/temp/Zhengwei/projects/${project_name}}"
codex_home="${CODEX_HOME:-$HOME/.codex}"
out_dir="${project_temp}/codex"
out_path="${out_dir}/codex_latest.tar.gz"
timestamp="$(date +%Y%m%d_%H%M%S)"
version_path="${out_dir}/codex_${timestamp}_$$.tar.gz"
tmp_path="$version_path"
list_file="${TMPDIR:-/tmp}/${project_name}_codex_backup_files.$$"
backup_complete=0

cleanup() {
  if [[ "$backup_complete" != "1" ]]; then
    rm -f "$tmp_path"
  fi
  rm -f "$list_file"
}
trap cleanup EXIT

mkdir -p "$out_dir"
: > "$list_file"

if [[ -d "$codex_home/sessions" ]]; then
  printf '%s\n' "sessions" >> "$list_file"
fi
for file in session_index.jsonl auth.json config.toml; do
  if [[ -f "$codex_home/$file" ]]; then
    printf '%s\n' "$file" >> "$list_file"
  fi
done
while IFS= read -r -d '' sqlite_file; do
  sqlite_rel="${sqlite_file#"$codex_home"/}"
  printf '%s\n' "$sqlite_rel" >> "$list_file"
done < <(find "$codex_home" -maxdepth 1 -type f -name 'state_*.sqlite' -print0 2>/dev/null)

if [[ ! -s "$list_file" ]]; then
  echo "[$project_name] no codex files found under $codex_home" >&2
  exit 1
fi

COPYFILE_DISABLE=1 tar \
  --warning=no-file-changed \
  --ignore-failed-read \
  --exclude='.DS_Store' \
  --exclude='._*' \
  -C "$codex_home" \
  -czf "$tmp_path" \
  --files-from "$list_file"

tar -tzf "$tmp_path" >/dev/null

# Some project /temp mounts reject atomic rename over an existing file. Keep the
# verified versioned archive as a fallback, then refresh the canonical path.
cp -pf "$tmp_path" "$out_path"
tar -tzf "$out_path" >/dev/null
backup_complete=1
printf '[%s] codex 会话已备份\n' "$project_name"
