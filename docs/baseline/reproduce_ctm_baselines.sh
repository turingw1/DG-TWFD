#!/usr/bin/env bash
set -euo pipefail

# Server-side helper for reproducing CTM CIFAR10 and ImageNet64 baselines on
# two GPUs. Configure paths with environment variables before running.

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
CTM_CIFAR_DIR="${CTM_CIFAR_DIR:-$REPO_ROOT/refs/ctm-cifar10}"
CTM_IM64_DIR="${CTM_IM64_DIR:-$REPO_ROOT/refs/ctm/code}"

STEPS="${STEPS:-1 2 4 8}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-50000}"
NUM_GPUS="${NUM_GPUS:-2}"
RUN_CIFAR="${RUN_CIFAR:-1}"
RUN_IM64="${RUN_IM64:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

CIFAR_BATCH="${CIFAR_BATCH:-1000}"
IM64_BATCH="${IM64_BATCH:-250}"

CIFAR_OUT="${CIFAR_OUT:-$REPO_ROOT/runs/baseline/ctm_cifar10}"
IM64_OUT="${IM64_OUT:-$REPO_ROOT/runs/baseline/ctm_imagenet64}"

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

sample_count_per_gpu() {
  python - "$TOTAL_SAMPLES" "$NUM_GPUS" <<'PY'
import math
import sys
total = int(sys.argv[1])
num_gpus = int(sys.argv[2])
print(math.ceil(total / num_gpus))
PY
}

checkpoint_step() {
  python - "$1" <<'PY'
import os
import sys
stem = os.path.splitext(os.path.basename(sys.argv[1]))[0]
print(stem[-6:])
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
  local out_root="$1"
  local ckpt="$2"
  local steps="$3"
  local ckpt_step
  local ema
  ckpt_step="$(checkpoint_step "$ckpt")"
  ema="$(checkpoint_ema "$ckpt")"
  printf "%s/ctm_exact_sampler_%s_steps_%s_itrs_%s_ema_" \
    "$out_root" "$steps" "$ckpt_step" "$ema"
}

run_two_gpu_sampling() {
  local workdir="$1"
  local ckpt="$2"
  local out_root="$3"
  local batch="$4"
  local steps="$5"
  shift 5
  local extra_flags=("$@")
  local per_gpu
  per_gpu="$(sample_count_per_gpu)"

  mkdir -p "$out_root"
  echo "Sampling step=$steps into $(sample_dir "$out_root" "$ckpt" "$steps")"

  for gpu in $(seq 0 "$((NUM_GPUS - 1))"); do
    (
      cd "$workdir"
      mpiexec -n 1 python image_sample.py "${extra_flags[@]}" \
        --out_dir "$out_root" \
        --model_path "$ckpt" \
        --training_mode=ctm \
        --eval_num_samples="$per_gpu" \
        --batch_size="$batch" \
        --device_id="$gpu" \
        --sampler=exact \
        --sampling_steps="$steps" \
        --save_format=npz \
        --stochastic_seed=True \
        --use_MPI=True
    ) &
  done
  wait
}

eval_cifar() {
  local steps="$1"
  local dir
  dir="$(sample_dir "$CIFAR_OUT" "$CIFAR_CKPT" "$steps")"
  echo "Evaluating CIFAR10 step=$steps from $dir"
  (
    cd "$CTM_CIFAR_DIR"
    python fid_npzs.py \
      --ref "$CIFAR_REF" \
      --images "$dir" \
      --num_samples "$TOTAL_SAMPLES" \
      --batch_size 500 \
      --device cuda:0
  )
}

eval_im64() {
  local steps="$1"
  local dir
  dir="$(sample_dir "$IM64_OUT" "$IM64_CKPT" "$steps")"
  echo "Evaluating ImageNet64 step=$steps from $dir"
  (
    cd "$CTM_IM64_DIR"
    python evaluations/evaluator.py "$IM64_REF" "$dir"
  )
}

main() {
  need_dir "CTM CIFAR10 repo" "$CTM_CIFAR_DIR"
  need_dir "CTM ImageNet64 code dir" "$CTM_IM64_DIR"
  command -v mpiexec >/dev/null

  if [[ "$RUN_CIFAR" == "1" ]]; then
    need_file "CIFAR_CKPT" "${CIFAR_CKPT:-}"
    need_file "CIFAR_REF" "${CIFAR_REF:-}"
    for steps in $STEPS; do
      run_two_gpu_sampling \
        "$CTM_CIFAR_DIR" "$CIFAR_CKPT" "$CIFAR_OUT" "$CIFAR_BATCH" "$steps" \
        --class_cond=False
      if [[ "$RUN_EVAL" == "1" ]]; then
        eval_cifar "$steps"
      fi
    done
  fi

  if [[ "$RUN_IM64" == "1" ]]; then
    need_file "IM64_CKPT" "${IM64_CKPT:-}"
    need_file "IM64_REF" "${IM64_REF:-}"
    for steps in $STEPS; do
      run_two_gpu_sampling \
        "$CTM_IM64_DIR" "$IM64_CKPT" "$IM64_OUT" "$IM64_BATCH" "$steps" \
        --data_name=imagenet64 \
        --class_cond=True \
        --num_classes=1000 \
        --eval_batch=250 \
        --eval_fid=True \
        --eval_similarity=False \
        --check_dm_performance=False
      if [[ "$RUN_EVAL" == "1" ]]; then
        eval_im64 "$steps"
      fi
    done
  fi
}

main "$@"
