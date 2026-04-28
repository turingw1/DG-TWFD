#!/usr/bin/env bash
set -euo pipefail

ROOT="${DG_TWFD_ROOT:-/home/ma-user/workspace/Zhengwei/DG-TWFD}"
PYTHON_BIN="${DG_TWFD_PYTHON:-${ROOT}/.conda_envs/dg_twfd_a100/bin/python}"
RUN_ID="${BASELINE_REVALIDATION_ID:-baselines_revalidated_20260428}"
STABLE_ROOT="${BASELINE_REVALIDATION_STABLE_ROOT:-/temp/Zhengwei/projects/DG-TWFD/critical/analysis/${RUN_ID}}"
LOG_ROOT="${BASELINE_REVALIDATION_LOG_ROOT:-/temp/Zhengwei/projects/DG-TWFD/logs/${RUN_ID}}"
RUNTIME_RUN_ROOT="runs/${RUN_ID}"
RUNTIME_EVAL_ROOT="eval/${RUN_ID}"
STEPS=(${BASELINE_REVALIDATION_STEPS:-1 2 4 8})
NUM_SAMPLES="${BASELINE_REVALIDATION_NUM_SAMPLES:-50000}"
FID_BATCH="${BASELINE_REVALIDATION_FID_BATCH:-512}"
CTM_CIFAR_BATCH="${CTM_CIFAR_BATCH:-500}"
CTM_IMAGENET_BATCH="${CTM_IMAGENET_BATCH:-250}"

mkdir -p "${STABLE_ROOT}/reports" "${LOG_ROOT}"

cat >"${STABLE_ROOT}/MANIFEST.txt" <<EOF
run_id=${RUN_ID}
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
num_fid_samples=${NUM_SAMPLES}
steps=${STEPS[*]}
standard=CTM 50k revalidation using official CTM sampling code, official checkpoints available locally, exact sampler, EDM fid.py references.
runtime_run_root=${ROOT}/${RUNTIME_RUN_ROOT}
runtime_eval_root=${ROOT}/${RUNTIME_EVAL_ROOT}
EOF

cd "${ROOT}"

echo "[ctm-50k] running CIFAR-10 revalidation"
"${PYTHON_BIN}" scripts/baselines/run_ctm_cifar10_eval.py \
  --sample-root "${RUNTIME_RUN_ROOT}/ctm_cifar10_50k/samples" \
  --eval-root "${RUNTIME_EVAL_ROOT}/ctm_cifar10_50k" \
  --csv-out "${STABLE_ROOT}/baseline_ctm_cifar10_50k.csv" \
  --steps "${STEPS[@]}" \
  --num-samples "${NUM_SAMPLES}" \
  --batch "${CTM_CIFAR_BATCH}" \
  --fid-batch "${FID_BATCH}" \
  2>&1 | tee "${LOG_ROOT}/ctm_cifar10_50k.log"

cp -f "${RUNTIME_EVAL_ROOT}/ctm_cifar10_50k/reports/summary.json" "${STABLE_ROOT}/reports/baseline_ctm_cifar10_50k_summary.json"
cp -f "${RUNTIME_EVAL_ROOT}/ctm_cifar10_50k/reports/summary.csv" "${STABLE_ROOT}/reports/baseline_ctm_cifar10_50k_summary.csv"

echo "[ctm-50k] running ImageNet64 revalidation"
"${PYTHON_BIN}" scripts/baselines/run_ctm_imagenet64_eval.py \
  --sample-root "${RUNTIME_RUN_ROOT}/ctm_imagenet64_50k/samples" \
  --eval-root "${RUNTIME_EVAL_ROOT}/ctm_imagenet64_50k" \
  --csv-out "${STABLE_ROOT}/baseline_ctm_imagenet64_50k.csv" \
  --steps "${STEPS[@]}" \
  --num-samples "${NUM_SAMPLES}" \
  --batch "${CTM_IMAGENET_BATCH}" \
  --fid-batch "${FID_BATCH}" \
  2>&1 | tee "${LOG_ROOT}/ctm_imagenet64_50k.log"

cp -f "${RUNTIME_EVAL_ROOT}/ctm_imagenet64_50k/reports/summary.json" "${STABLE_ROOT}/reports/baseline_ctm_imagenet64_50k_summary.json"
cp -f "${RUNTIME_EVAL_ROOT}/ctm_imagenet64_50k/reports/summary.csv" "${STABLE_ROOT}/reports/baseline_ctm_imagenet64_50k_summary.csv"

echo "[ctm-50k] completed; stable reports: ${STABLE_ROOT}"
