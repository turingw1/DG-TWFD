#!/usr/bin/env bash
set -euo pipefail

PID="${1:?usage: monitor_gpu_process.sh <pid> <limit_mb> <log_path>}"
LIMIT_MB="${2:?usage: monitor_gpu_process.sh <pid> <limit_mb> <log_path>}"
LOG_PATH="${3:?usage: monitor_gpu_process.sh <pid> <limit_mb> <log_path>}"
INTERVAL="${INTERVAL:-60}"

mkdir -p "$(dirname "${LOG_PATH}")"

while kill -0 "${PID}" 2>/dev/null; do
  USED_MB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR == 1 {print $1}')"
  NOW="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "${NOW},pid=${PID},used_mb=${USED_MB},limit_mb=${LIMIT_MB}" >> "${LOG_PATH}"
  if [ "${USED_MB}" -gt "${LIMIT_MB}" ]; then
    echo "${NOW},killing pid=${PID}: used_mb=${USED_MB} > limit_mb=${LIMIT_MB}" >> "${LOG_PATH}"
    kill "${PID}" 2>/dev/null || true
    sleep 10
    kill -9 "${PID}" 2>/dev/null || true
    exit 2
  fi
  sleep "${INTERVAL}"
done

echo "$(date '+%Y-%m-%d %H:%M:%S'),pid=${PID},exited" >> "${LOG_PATH}"
