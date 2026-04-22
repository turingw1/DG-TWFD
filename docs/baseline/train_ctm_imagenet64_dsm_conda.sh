#!/usr/bin/env bash
set -euo pipefail

# Train an ImageNet64 CTM+DSM checkpoint without GAN using the conda runtime.
# Defaults match the current server layout and can be overridden with env vars.

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
CTM_IM64_DIR="${CTM_IM64_DIR:-$REPO_ROOT/refs/ctm/code}"
SERVER_HOME_ROOT="${SERVER_HOME_ROOT:-/homes/yl7622/Zhengwei}"
SERVER_PROJECT_ROOT="${SERVER_PROJECT_ROOT:-/data2/yl7622/Zhengwei}"

NUM_GPUS="${NUM_GPUS:-2}"
IM64_TEACHER="${IM64_TEACHER:-$REPO_ROOT/author_ckpt/edm_imagenet64_ema.pt}"
IM64_REF="${IM64_REF:-$REPO_ROOT/author_ckpt/VIRTUAL_imagenet64_labeled.npz}"
IM64_DATA_DIR="${IM64_DATA_DIR:-$SERVER_HOME_ROOT/datasets/imagenet64/train}"
IM64_OUT="${IM64_OUT:-$SERVER_PROJECT_ROOT/output/CTM_DSM}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

TRAIN_STEPS="${TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
SAVE_CHECK_PERIOD="${SAVE_CHECK_PERIOD:--1}"
EVAL_INTERVAL="${EVAL_INTERVAL:--1}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MICROBATCH="${MICROBATCH:-8}"
LR="${LR:-0.00004}"
NUM_HEUN_STEP="${NUM_HEUN_STEP:-39}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EMA_RATE="${EMA_RATE:-0.999,0.9999,0.9999432189950708}"

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

main() {
  need_dir "CTM ImageNet64 code dir" "$CTM_IM64_DIR"
  need_file "IM64_TEACHER" "${IM64_TEACHER:-}"
  need_dir "IM64_DATA_DIR" "${IM64_DATA_DIR:-}"
  if [[ -n "$IM64_REF" ]]; then
    need_file "IM64_REF" "$IM64_REF"
  fi
  if [[ -n "$RESUME_CHECKPOINT" ]]; then
    need_file "RESUME_CHECKPOINT" "$RESUME_CHECKPOINT"
  fi
  command -v mpiexec >/dev/null

  need_python_module blobfile
  need_python_module einops
  need_python_module mpi4py
  need_python_module numpy
  need_python_module scipy
  need_python_module torch
  need_python_module torchvision

  mkdir -p "$IM64_OUT"
  echo "Training ImageNet64 CTM+DSM without GAN"
  echo "Output: $IM64_OUT"
  echo "Teacher: $IM64_TEACHER"
  echo "Reference stats: $IM64_REF"
  echo "Data: $IM64_DATA_DIR"
  echo "Steps: $TRAIN_STEPS, save_interval: $SAVE_INTERVAL"
  echo "Global batch: $GLOBAL_BATCH_SIZE, microbatch: $MICROBATCH, lr: $LR"

  local resume_flags=()
  if [[ -n "$RESUME_CHECKPOINT" ]]; then
    resume_flags+=(--resume_checkpoint "$RESUME_CHECKPOINT")
  fi

  (
    cd "$CTM_IM64_DIR"
    mpiexec -n "$NUM_GPUS" python cm_train.py \
      --data_name=imagenet64 \
      --attention_type=legacy \
      --class_cond=True \
      --num_classes=1000 \
      --teacher_model_path "$IM64_TEACHER" \
      --data_dir "$IM64_DATA_DIR" \
      --ref_path "$IM64_REF" \
      --out_dir "$IM64_OUT" \
      --training_mode=ctm \
      --gan_training=False \
      --diffusion_training=True \
      --eval_fid=False \
      --eval_similarity=False \
      --check_dm_performance=False \
      --compute_ema_fids=False \
      --eval_interval "$EVAL_INTERVAL" \
      --save_check_period "$SAVE_CHECK_PERIOD" \
      --save_interval "$SAVE_INTERVAL" \
      --log_interval "$LOG_INTERVAL" \
      --total_training_steps "$TRAIN_STEPS" \
      --global_batch_size "$GLOBAL_BATCH_SIZE" \
      --microbatch "$MICROBATCH" \
      --lr "$LR" \
      --num_heun_step "$NUM_HEUN_STEP" \
      --ema_rate "$EMA_RATE" \
      --use_MPI=True \
      --num_workers "$NUM_WORKERS" \
      "${resume_flags[@]}"
  )

  echo "Done. Checkpoints are under $IM64_OUT"
  echo "Expected names include modelXXXXXX.pt and ema_<rate>_XXXXXX.pt"
}

main "$@"
