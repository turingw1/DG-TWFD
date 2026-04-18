# DGTD v3 Round-4 Online Mainline Server Smoke Instructions

This document is the server-side validation checklist for round 4 online-mainline
acceptance.

Primary code/doc context:

- [`docs/dgtd_v3_round4_online_mainline_patch_notes.md`](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/dgtd_v3_round4_online_mainline_patch_notes.md)
- [`docs/dgtd_v3_round4_online_mainline_verification.md`](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/dgtd_v3_round4_online_mainline_verification.md)

## Workflow Rule

From this round onward, every eval cycle should follow this process:

1. I update this server smoke document with the exact commands and required evidence.
2. You run the commands on the server.
3. You paste back the requested outputs.
4. I update this same document again with the actual returned results and revise the
   verification conclusion accordingly.

So this file is not only an instruction note. It is also the round-4 server truth
record.

## Goal

The goal is to verify five runtime claims on the server:

1. online teacher data path is active
2. online continuation really appears in DGTD residual logs
3. warmup no longer behaves like an obvious direct-only corner case
4. new alpha/source diagnostics are actually emitted
5. sample/eval still load the saved checkpoint and warp state correctly

## Expected Key Signals

For this round, the most important runtime fields are:

- `online_teacher_data`
- `continuation_sources`
- `online_anchor_used_rate`
- `online_continuation_rate`
- `cached_fallback_rate`
- `exact_mask_hit_rate`
- `alpha_online_mean`
- `alpha_online_min`
- `alpha_online_max`
- `train_bridge_state_teacher_error`
- `train_bridge_u_teacher_error`
- `train_teacher_rel_error_mean`
- `eta`
- `stage`

For a healthy round-4 smoke, the strongest signs are:

- `online_teacher_data=true`
- `continuation_sources.online > 0`
- `online_continuation_rate > 0`
- `cached_fallback_rate` not dominating
- `alpha_online_*` present and finite

## 0. Server Bootstrap

Run this from a fresh shell:

```bash
export PROJ=/data2/yl7622/Zhengwei/DG-TWFD
cd "$PROJ"

source /data2/yl7622/anaconda/etc/profile.d/conda.sh

if [ -d "$PROJ/.conda_envs/dgfm_map" ]; then
  conda activate "$PROJ/.conda_envs/dgfm_map"
else
  echo "Missing conda env at $PROJ/.conda_envs/dgfm_map"
  exit 1
fi
```

Then print:

```bash
pwd
which python
python -V
conda info --envs | sed -n '1,20p'
```

## 1. Repo And Commit Check

Run:

```bash
cd "$PROJ"
git branch --show-current
git rev-parse HEAD
git show --stat --oneline --no-patch HEAD
git show --stat --oneline --no-patch 054351e
```

Return these outputs exactly.

## 2. Activation And Variable Setup

Follow the map-branch server workflow:

```bash
cd "$PROJ"
source scripts/experiments/activate_fm_cifar10.sh dgtd_v3_smoke round4_online_mainline
```

Then print:

```bash
echo "PROJ=$PROJ"
echo "FM_CONFIG=$FM_CONFIG"
echo "EXP_VARIANT=$EXP_VARIANT"
echo "EXP_TAG=$EXP_TAG"
echo "FM_EXP=$FM_EXP"
echo "RUN_ROOT=$RUN_ROOT"
echo "CKPT_DIR=$CKPT_DIR"
echo "LOG_ROOT=$LOG_ROOT"
echo "METRIC_ROOT=$METRIC_ROOT"
echo "SAMPLE_ROOT=$SAMPLE_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "REF_ROOT=$REF_ROOT"
echo "TORCH_HOME=$TORCH_HOME"
echo "HF_HOME=$HF_HOME"
echo "TRAIN_CUDA_VISIBLE_DEVICES=$TRAIN_CUDA_VISIBLE_DEVICES"
echo "INFER_CUDA_VISIBLE_DEVICES=$INFER_CUDA_VISIBLE_DEVICES"
```

## 3. Online Teacher Preflight

Run:

```bash
cd "$PROJ"
python - <<'PY'
from dgfm.config import load_experiment_config
from dgtd.teacher import build_teacher_adapter

cfg = load_experiment_config(
    "configs/experiment/dgtd_cifar10_v3_smoke.yaml",
    overrides=[
        "dgtd.disable_online_teacher=false",
        "dgtd.use_online_teacher_data=true",
        "teacher.local_files_only=false",
    ],
)
adapter = build_teacher_adapter(cfg)
print("online_teacher_built", adapter.online_teacher is not None)
print("teacher_type", type(adapter.online_teacher).__name__ if adapter.online_teacher is not None else None)
adapter.prepare("cpu")
print("online_teacher_prepare", "ok")
PY
```

Pass condition:

- `online_teacher_built True`
- `online_teacher_prepare ok`

## 4. Dataset Root Validation

Run:

```bash
echo "DATA_ROOT=$DATA_ROOT"
find "$DATA_ROOT/cifar10" -maxdepth 2 | head -n 30
```

Pass condition:

- `cifar-10-batches-py` exists under `$DATA_ROOT/cifar10`

## 5. Train Smoke

Use a dedicated isolated root:

```bash
export DGTD_V3_R4_ROOT=/tmp/dgtd_v3_round4_online_mainline_smoke
rm -rf "$DGTD_V3_R4_ROOT"
mkdir -p "$DGTD_V3_R4_ROOT"
```

Run train:

```bash
cd "$PROJ"
python scripts/run_train.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --run-root "$DGTD_V3_R4_ROOT" \
  --set dgtd.disable_online_teacher=false \
  --set dgtd.use_online_teacher_data=true \
  --set teacher.local_files_only=false \
  2>&1 | tee "$DGTD_V3_R4_ROOT/train.stdout_stderr.txt"
```

Then return:

```bash
echo "==== tail train.jsonl ===="
tail -n 5 "$DGTD_V3_R4_ROOT/logs/train.jsonl"

echo "==== checkpoint files ===="
find "$DGTD_V3_R4_ROOT/checkpoints" -maxdepth 2 -type f | sort
```

## 6. Sample Smoke

Run:

```bash
cd "$PROJ"
python scripts/run_sample_dgtd.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --checkpoint "$DGTD_V3_R4_ROOT/checkpoints/last.pt" \
  --output-dir "$DGTD_V3_R4_ROOT/sample" \
  --steps 4 \
  2>&1 | tee "$DGTD_V3_R4_ROOT/sample.stdout_stderr.txt"
```

Then return:

```bash
echo "==== sample files ===="
find "$DGTD_V3_R4_ROOT/sample" -maxdepth 3 -type f | sort
```

## 7. Eval Smoke

Run:

```bash
cd "$PROJ"
python scripts/run_eval.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --checkpoint "$DGTD_V3_R4_ROOT/checkpoints/last.pt" \
  --eval-root "$DGTD_V3_R4_ROOT/eval" \
  --steps 1 2 4 \
  2>&1 | tee "$DGTD_V3_R4_ROOT/eval.stdout_stderr.txt"
```

Then return:

```bash
echo "==== eval files ===="
find "$DGTD_V3_R4_ROOT/eval" -maxdepth 4 -type f | sort
```

## 8. Diagnostics Plot Export

Run:

```bash
cd "$PROJ"
python scripts/plot_dgtd_diagnostics.py \
  --log "$DGTD_V3_R4_ROOT/logs/train.jsonl" \
  --output-dir "$DGTD_V3_R4_ROOT/diag" \
  2>&1 | tee "$DGTD_V3_R4_ROOT/diag.stdout_stderr.txt"
```

Then return:

```bash
echo "==== diag files ===="
find "$DGTD_V3_R4_ROOT/diag" -maxdepth 3 -type f | sort
```

## 9. What You Need To Paste Back

Please paste back all of the following:

1. section 1 outputs
2. section 2 outputs
3. section 3 outputs
4. section 4 outputs
5. full stdout/stderr from train smoke
6. `tail -n 5 "$DGTD_V3_R4_ROOT/logs/train.jsonl"`
7. sample file list
8. full stdout/stderr from eval smoke
9. eval file list
10. diag file list

If any command fails, also paste:

- the full traceback
- `ls -R "$DGTD_V3_R4_ROOT" | sed -n '1,200p'`

## 10. Result Record Template

After you return the server results, I will update the section below rather than
keeping the evidence only in chat.

### 10.1 Run Status

- `status`: pending
- `server_date`: pending
- `commit_checked`: pending

### 10.2 Online Teacher Preflight

- `online_teacher_built`: pending
- `teacher_type`: pending
- `online_teacher_prepare`: pending

### 10.3 Train Smoke

- `train_exit_status`: pending
- `online_teacher_data`: pending
- `continuation_sources.online`: pending
- `online_continuation_rate`: pending
- `cached_fallback_rate`: pending
- `exact_mask_hit_rate`: pending
- `alpha_online_mean`: pending
- `alpha_online_min`: pending
- `alpha_online_max`: pending
- `train_bridge_state_teacher_error`: pending
- `train_bridge_u_teacher_error`: pending
- `train_teacher_rel_error_mean`: pending
- `eta`: pending
- `stage`: pending

### 10.4 Sample / Eval / Diag

- `sample_exit_status`: pending
- `eval_exit_status`: pending
- `diag_exit_status`: pending
- `sample_files_present`: pending
- `eval_files_present`: pending
- `diag_files_present`: pending

### 10.5 Acceptance Update

- `updated_by_agent_after_server_return`: pending
- `revised_verdict`: pending
- `notes`: pending

