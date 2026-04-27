#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash experiments/edm_first/scripts/watch_project_backup_v11.sh <run_tag> [interval_sec] [project_name]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
run_tag="$1"
interval="${2:-180}"
project_name="${3:-$(basename "$ROOT_DIR")}"
project_temp="${DG_TWFD_PROJECT_TEMP:-/temp/Zhengwei/projects/${project_name}}"
critical_root="${project_temp}/critical"
logs_root="${project_temp}/logs"
run_root="${ROOT_DIR}/runs/${run_tag}"
backup_run_root="${critical_root}/runs/${run_tag}"
backup_eval_root="${critical_root}/eval"
log_file="${run_root}/logs/watch_project_backup_v11.log"
temp_log="${logs_root}/${run_tag}_backup_v11.log"
last_codex_sync=0
codex_interval="${DG_TWFD_CODEX_BACKUP_INTERVAL_SEC:-3600}"

mkdir -p "${run_root}/logs" "$logs_root" "${backup_run_root}/logs" "${backup_run_root}/checkpoints" "$backup_eval_root"

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

sync_eval_dir() {
  local eval_dir="$1"
  [[ -d "$eval_dir" ]] || return 0
  local eval_name
  eval_name="$(basename "$eval_dir")"
  mkdir -p "${backup_eval_root}/${eval_name}"

  for file in config.yaml eval.stdout_stderr.txt; do
    [[ -f "${eval_dir}/${file}" ]] && cp -af "${eval_dir}/${file}" "${backup_eval_root}/${eval_name}/${file}"
  done
  if [[ -d "${eval_dir}/reports" ]]; then
    mkdir -p "${backup_eval_root}/${eval_name}/reports"
    cp -af "${eval_dir}/reports"/. "${backup_eval_root}/${eval_name}/reports/" 2>/dev/null || true
  fi
  while IFS= read -r -d '' step_dir; do
    local step_name
    step_name="$(basename "$step_dir")"
    mkdir -p "${backup_eval_root}/${eval_name}/${step_name}"
    for file in metrics.json fixed_seed_grid.png sigma_grid.pt u_grid.pt; do
      [[ -f "${step_dir}/${file}" ]] && cp -af "${step_dir}/${file}" "${backup_eval_root}/${eval_name}/${step_name}/${file}"
    done
  done < <(find "$eval_dir" -maxdepth 1 -type d -name 'steps*' -print0 2>/dev/null)
}

log_line() {
  local line="$1"
  printf '%s\n' "$line" | tee -a "$log_file" "$temp_log"
}

log_line "watch_project_backup_v11 started run_tag=${run_tag} interval=${interval} project_temp=${project_temp}"

while true; do
  if [[ ! -d "$run_root" ]]; then
    log_line "run root missing: ${run_root}"
    sleep "$interval"
    continue
  fi

  mkdir -p "${backup_run_root}/logs" "${backup_run_root}/checkpoints" "${backup_run_root}/analysis"
  if [[ -d "${run_root}/logs" ]]; then
    while IFS= read -r -d '' log_path; do
      cp -af "$log_path" "${backup_run_root}/logs/$(basename "$log_path")"
    done < <(find "${run_root}/logs" -maxdepth 1 -type f -print0 2>/dev/null)
  fi

  for file in train_summary.json; do
    [[ -f "${run_root}/${file}" ]] && cp -af "${run_root}/${file}" "${backup_run_root}/${file}"
  done

  if [[ -d "${run_root}/checkpoints" ]]; then
    while IFS= read -r -d '' ckpt; do
      copy_stable_file "$ckpt" "${backup_run_root}/checkpoints/$(basename "$ckpt")"
    done < <(find "${run_root}/checkpoints" -maxdepth 1 -type f \( -name 'step*.pt' -o -name 'last.pt' -o -name 'best.pt' \) -print0 2>/dev/null)
  fi

  if [[ -d "${ROOT_DIR}/eval" ]]; then
    while IFS= read -r -d '' eval_dir; do
      sync_eval_dir "$eval_dir"
    done < <(find "${ROOT_DIR}/eval" -maxdepth 1 -type d \( -name "${run_tag}_step*" -o -name "${run_tag}_hourly*" \) -print0 2>/dev/null)
  fi

  now="$(date +%s)"
  if (( now - last_codex_sync >= codex_interval )); then
    bash "${ROOT_DIR}/scripts/server/backup_codex_project_v11.sh" "$project_name" | tee -a "$log_file" "$temp_log"
    last_codex_sync="$now"
  fi

  find "$backup_run_root" "$backup_eval_root" -maxdepth 5 -type f | sort > "${backup_run_root}/MANIFEST.txt"
  date -Is > "${backup_run_root}/LAST_SYNC.txt"
  log_line "project backup sync complete $(date -Is)"
  sleep "$interval"
done
