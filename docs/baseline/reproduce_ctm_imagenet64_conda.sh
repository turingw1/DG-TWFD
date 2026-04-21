#!/usr/bin/env bash
set -euo pipefail

# Conda-first ImageNet64 CTM baseline reproduction. This script assumes the
# checkpoint and reference statistics are already present on the server.

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
CTM_IM64_DIR="${CTM_IM64_DIR:-$REPO_ROOT/refs/ctm/code}"

STEPS="${STEPS:-1 2 4 8}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-50000}"
NUM_GPUS="${NUM_GPUS:-2}"
IM64_BATCH="${IM64_BATCH:-250}"
IM64_OUT="${IM64_OUT:-$REPO_ROOT/runs/baseline/ctm_imagenet64}"
RUN_SAMPLE="${RUN_SAMPLE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

need_file() {
  local name="$1"
  local path="$2"
  if [[ -z "$path" || ! -f "$path" ]]; then
    echo "Missing $name: $path" >&2
    exit 2
  fi
}

need_dir() {
  local name="$1"
  local path="$2"
  if [[ -z "$path" || ! -d "$path" ]]; then
    echo "Missing $name: $path" >&2
    exit 2
  fi
}

need_python_module() {
  local module="$1"
  python - "$module" <<'PY'
import importlib
import sys

module = sys.argv[1]
try:
    importlib.import_module(module)
except Exception as exc:
    print(f"Missing Python module {module}: {exc}", file=sys.stderr)
    sys.exit(2)
PY
}

sample_count_per_gpu() {
  python - "$TOTAL_SAMPLES" "$NUM_GPUS" <<'PY'
import math
import sys

print(math.ceil(int(sys.argv[1]) / int(sys.argv[2])))
PY
}

checkpoint_step() {
  python - "$1" <<'PY'
import os
import sys

print(os.path.splitext(os.path.basename(sys.argv[1]))[0][-6:])
PY
}

checkpoint_ema() {
  python - "$1" <<'PY'
import os
import sys

parts = os.path.splitext(os.path.basename(sys.argv[1]))[0].split("_")
ema = "model"
if len(parts) >= 2:
    try:
        value = float(parts[-2])
        if value in (0.999, 0.9999, 0.9999432189950708):
            ema = parts[-2]
    except ValueError:
        pass
print(ema)
PY
}

sample_dir() {
  local steps="$1"
  local ckpt_step
  local ema
  ckpt_step="$(checkpoint_step "$IM64_CKPT")"
  ema="$(checkpoint_ema "$IM64_CKPT")"
  printf "%s/ctm_exact_sampler_%s_steps_%s_itrs_%s_ema_" \
    "$IM64_OUT" "$steps" "$ckpt_step" "$ema"
}

run_sampling_step() {
  local steps="$1"
  local per_gpu
  per_gpu="$(sample_count_per_gpu)"

  mkdir -p "$IM64_OUT"
  echo "Sampling ImageNet64 step=$steps into $(sample_dir "$steps")"

  local pids=()
  for gpu in $(seq 0 "$((NUM_GPUS - 1))"); do
    (
      cd "$CTM_IM64_DIR"
      mpiexec -n 1 python image_sample.py \
        --data_name=imagenet64 \
        --attention_type=legacy \
        --class_cond=True \
        --num_classes=1000 \
        --eval_batch=250 \
        --eval_fid=True \
        --eval_similarity=False \
        --check_dm_performance=False \
        --out_dir "$IM64_OUT" \
        --model_path "$IM64_CKPT" \
        --training_mode=ctm \
        --eval_num_samples="$per_gpu" \
        --batch_size="$IM64_BATCH" \
        --device_id="$gpu" \
        --sampler=exact \
        --sampling_steps="$steps" \
        --save_format=npz \
        --stochastic_seed=True \
        --use_MPI=True
    ) &
    pids+=("$!")
  done

  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" != "0" ]]; then
    echo "Sampling failed for ImageNet64 step=$steps" >&2
    exit "$status"
  fi
}

run_eval_step() {
  local steps="$1"
  local dir
  dir="$(sample_dir "$steps")"
  echo "Evaluating ImageNet64 step=$steps from $dir"
  (
    cd "$CTM_IM64_DIR"
    python evaluations/evaluator.py "$IM64_REF" "$dir"
  )
}

main() {
  need_dir "CTM ImageNet64 code dir" "$CTM_IM64_DIR"
  need_file "IM64_CKPT" "${IM64_CKPT:-}"
  need_file "IM64_REF" "${IM64_REF:-}"
  command -v mpiexec >/dev/null

  need_python_module blobfile
  need_python_module einops
  need_python_module mpi4py
  need_python_module numpy
  need_python_module scipy
  need_python_module torch
  need_python_module torchvision
  if [[ "$RUN_EVAL" == "1" ]]; then
    need_python_module tensorflow
  fi

  for steps in $STEPS; do
    if [[ "$RUN_SAMPLE" == "1" ]]; then
      run_sampling_step "$steps"
    fi
    if [[ "$RUN_EVAL" == "1" ]]; then
      run_eval_step "$steps"
    fi
  done
}

main "$@"
