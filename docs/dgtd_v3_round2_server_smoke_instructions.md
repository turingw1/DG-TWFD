# DGTD v3 Round-2 Server Smoke Instructions

This document is the server-side acceptance checklist for commit:

- `054351e Patch DGTD v3 residual teacher and sigma core`

Use this document on the server only.

It is intentionally more detailed than the compact smoke block in
[docs/experiments/DG_TWFD_v3/DEVELOPMENT_VALIDATION.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/DG_TWFD_v3/DEVELOPMENT_VALIDATION.md),
because this round may start from a shell that does **not** already have:

- `PROJ`
- `TRAJ_ROOT`
- `RUN_ROOT`
- `METRIC_ROOT`
- the correct conda environment

This workflow follows the preserved server rules in:

- [docs/experiments/map_branch/HANDOFF_2026-04-16.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/HANDOFF_2026-04-16.md)
- [docs/experiments/map_branch/ENVIRONMENT.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/ENVIRONMENT.md)
- [docs/experiments/map_branch/A100_PIPELINE.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/A100_PIPELINE.md)
- [docs/experiments/map_branch/EXP_LOG.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/EXP_LOG.md)

Runtime evidence for this branch must come from the server, not from the local machine.

## Goal

Return enough server evidence to verify:

1. smoke train runs end-to-end
2. checkpoint saves and loads
3. DGTD sample uses warp `r_to_t`
4. eval runs on the saved checkpoint
5. `train.jsonl` contains the new DGTD diagnostics
6. diagnostics plot export succeeds

## Important Constraint About Online Teacher

This round can and should **prefer online teacher as the intended direction**,
but the current DGTD v3 implementation is **not yet a pure online-teacher
trainer**.

Current code facts:

- `DGTDTrainer.run()` still builds data through `build_cache_dataloaders(...)`
  in [src/dgtd/train_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:587)
- `build_cache_dataloaders(...)` always instantiates
  `TrajectoryCacheDataset(...)` in [src/dgtd/cache.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/cache.py:218)
- `TeacherAdapter.local_flow()` only uses the online path if the teacher object
  implements `local_flow(...)`, checked at
  [src/dgtd/teacher.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:71)
- the current `DiffusersDDPMTeacher` does **not** implement `local_flow(...)`;
  it implements trajectory sampling APIs instead, see
  [src/dgfm/teachers/diffusers_ddpm.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/teachers/diffusers_ddpm.py:113)

So the practical rule for this instruction file is:

- we will explicitly enable online teacher during smoke to check that the
  server can load it
- but we still need a valid `target.shard_root` for the current DGTD v3 trainer

If the project later lands a true online-teacher DGTD trainer path, this
instruction file should be simplified accordingly.

## Naming For This Round

To mirror the style in `EXP_LOG.md`, this smoke uses:

- `EXP_VARIANT=dgtd_v3_smoke`
- `EXP_TAG=round2_verify`

Activation command:

```bash
source scripts/experiments/activate_fm_cifar10.sh dgtd_v3_smoke round2_verify
```

This is a smoke / verification run, not a formal experiment campaign row.
Using `--set target.shard_root=...` is acceptable here because the goal is
acceptance debugging, not final experiment logging.

## 0. Server Bootstrap

Run this block first, exactly in order.

```bash
export PROJ=/data2/yl7622/Zhengwei/DG-TWFD
cd "$PROJ"

source /data2/yl7622/anaconda/etc/profile.d/conda.sh

if [ -d "$PROJ/.conda_envs/dgfm_map" ]; then
  conda activate "$PROJ/.conda_envs/dgfm_map"
else
  echo "Missing conda env at $PROJ/.conda_envs/dgfm_map"
  echo "If needed, create it with:"
  echo "  cd $PROJ"
  echo "  bash scripts/experiments/create_map_branch_env.sh dgfm_map"
  exit 1
fi
```

Then paste back:

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
git show --stat --oneline --no-patch 054351e
git show --stat --oneline --no-patch 4f99df6
```

Expected:

- branch should be the DGTD v3 working branch you are using for this round
- `054351e` should exist
- `4f99df6` should exist because it adds the verification docs

## 2. Activation And Variable Setup

This is the key step that creates the server-local variable set described in
`ENVIRONMENT.md` and used consistently in `EXP_LOG.md`.

Run:

```bash
cd "$PROJ"
source scripts/experiments/activate_fm_cifar10.sh dgtd_v3_smoke round2_verify
```

Then immediately print the key variables:

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
echo "TRAJ_ROOT=$TRAJ_ROOT"
echo "TORCH_HOME=$TORCH_HOME"
echo "HF_HOME=$HF_HOME"
echo "TRAIN_CUDA_VISIBLE_DEVICES=$TRAIN_CUDA_VISIBLE_DEVICES"
echo "INFER_CUDA_VISIBLE_DEVICES=$INFER_CUDA_VISIBLE_DEVICES"
echo "NNODES=$NNODES"
echo "NODE_RANK=$NODE_RANK"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
```

## 3. Online Teacher Preflight

Before the actual smoke run, confirm that the server can build and prepare the
online teacher stack.

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

If this step fails because weights are not cached and the server cannot
download, return the traceback before continuing.

## 4. Trajectory Root Validation

Even though the long-term direction is online teacher, the current DGTD v3
trainer still requires shard-backed dataloaders. So for this round, we only do
a **minimal** shard-root check, not a long cache-discovery workflow.

If the activated `TRAJ_ROOT` already exists, validate it with:

```bash
if [ -d "$TRAJ_ROOT" ]; then
  echo "Resolved TRAJ_ROOT exists"
  find "$TRAJ_ROOT" -maxdepth 2 | head -n 30
else
  echo "Resolved TRAJ_ROOT does not exist: $TRAJ_ROOT"
fi
```

If the activated `TRAJ_ROOT` does **not** exist, do this one discovery step and
stop there if nothing valid is found:

```bash
echo "==== teacher_traj candidates under $PROJ ===="
find "$PROJ" -maxdepth 4 -type d \( -name 'teacher_traj' -o -name 'cifar10_ddpm128_p33' \) | sort
```

If you find the correct cache root manually, export it in the current shell:

```bash
export TRAJ_ROOT=/actual/path/to/the/cifar10_teacher_cache
echo "TRAJ_ROOT=$TRAJ_ROOT"
find "$TRAJ_ROOT" -maxdepth 2 | head -n 30
```

Pass condition:

- `TRAJ_ROOT` exists
- it contains either:
  - `train/` and `val/`, or
  - flat shard files

If you cannot resolve a valid `TRAJ_ROOT`, stop and return the outputs from this section.

## 5. Entry-Point Import Check

Before starting smoke, confirm the Python entrypoints import successfully in the
actual server environment.

Run:

```bash
cd "$PROJ"
python scripts/run_train.py --help | head -n 20
python scripts/run_sample_dgtd.py --help | head -n 20
python scripts/run_eval.py --help | head -n 20
python scripts/plot_dgtd_diagnostics.py --help | head -n 20
```

Pass condition:

- all four commands print help
- no import traceback appears

## 6. Dedicated Smoke Root

Even though activation created `$RUN_ROOT` and `$METRIC_ROOT`, for this
acceptance round we use a separate temp root so the smoke result is isolated and
easy to inspect.

Run:

```bash
export DGTD_V3_R2_ROOT=/tmp/dgtd_v3_round2_smoke
rm -rf "$DGTD_V3_R2_ROOT"
mkdir -p "$DGTD_V3_R2_ROOT"

echo "DGTD_V3_R2_ROOT=$DGTD_V3_R2_ROOT"
```

## 7. Smoke Train

Run the train smoke and capture stdout/stderr.

```bash
cd "$PROJ"
python scripts/run_train.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --run-root "$DGTD_V3_R2_ROOT" \
  --set dgtd.disable_online_teacher=false \
  --set teacher.local_files_only=false \
  --set target.shard_root="$TRAJ_ROOT" \
  2>&1 | tee "$DGTD_V3_R2_ROOT/train.stdout_stderr.txt"
```

Immediately after train finishes, run:

```bash
echo "==== smoke root tree ===="
find "$DGTD_V3_R2_ROOT" -maxdepth 4 | sort

echo "==== checkpoints ===="
find "$DGTD_V3_R2_ROOT/checkpoints" -maxdepth 2 -type f | sort

echo "==== logs ===="
find "$DGTD_V3_R2_ROOT/logs" -maxdepth 2 -type f | sort

echo "==== tail train.jsonl ===="
tail -n 5 "$DGTD_V3_R2_ROOT/logs/train.jsonl"
```

Train pass condition:

- `$DGTD_V3_R2_ROOT/checkpoints/last.pt` exists
- `$DGTD_V3_R2_ROOT/logs/train.jsonl` exists

## 8. Smoke Sample

Run sample from the saved `last.pt`.

```bash
cd "$PROJ"
python scripts/run_sample_dgtd.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --checkpoint "$DGTD_V3_R2_ROOT/checkpoints/last.pt" \
  --output-dir "$DGTD_V3_R2_ROOT/sample" \
  --steps 4 \
  --set target.shard_root="$TRAJ_ROOT" \
  2>&1 | tee "$DGTD_V3_R2_ROOT/sample.stdout_stderr.txt"
```

Then list outputs:

```bash
echo "==== sample files ===="
find "$DGTD_V3_R2_ROOT/sample" -maxdepth 2 -type f | sort
```

Sample pass condition:

- `samples.pt`
- `time_grid.pt`
- `grid.png`

all exist under `$DGTD_V3_R2_ROOT/sample`.

## 9. Smoke Eval

Run eval from the same `last.pt`.

```bash
cd "$PROJ"
python scripts/run_eval.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --checkpoint "$DGTD_V3_R2_ROOT/checkpoints/last.pt" \
  --eval-root "$DGTD_V3_R2_ROOT/eval" \
  --steps 1 2 4 \
  --set target.shard_root="$TRAJ_ROOT" \
  2>&1 | tee "$DGTD_V3_R2_ROOT/eval.stdout_stderr.txt"
```

Then list outputs:

```bash
echo "==== eval files ===="
find "$DGTD_V3_R2_ROOT/eval" -maxdepth 4 -type f | sort
```

Eval pass condition:

- some report or sample artifact is written under `$DGTD_V3_R2_ROOT/eval`
- no Python traceback in `eval.stdout_stderr.txt`

## 10. Diagnostics Export

Export the DGTD diagnostics from the produced history file.

```bash
cd "$PROJ"
python scripts/plot_dgtd_diagnostics.py \
  --history "$DGTD_V3_R2_ROOT/logs/train.jsonl" \
  --output-dir "$DGTD_V3_R2_ROOT/diag" \
  2>&1 | tee "$DGTD_V3_R2_ROOT/diag.stdout_stderr.txt"
```

Then list outputs:

```bash
echo "==== diag files ===="
find "$DGTD_V3_R2_ROOT/diag" -maxdepth 2 -type f | sort
```

Expected outputs include:

- `summary.json`
- `latest_bins.json`
- usually several `.png` plots if matplotlib is available

## 11. Structured Log Check

Run this small JSON check and paste the output back.

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("/tmp/dgtd_v3_round2_smoke/logs/train.jsonl")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
last = rows[-1]
keys = [
    "train_direct_teacher_error",
    "train_bridge_teacher_error",
    "train_direct_bridge_gap",
    "continuation_sources",
    "entropy_q_phi",
    "kl_qD_qphi",
    "max_qphi_over_qbase",
    "argmax_q_phi",
    "q_phi",
    "q_D",
    "D_bar",
    "K_bar",
    "HF_bar",
    "time_grid",
    "eta",
    "beta",
    "stage",
    "train_noisy_defect",
    "train_mid_defect",
    "train_clean_defect",
]
print("history_rows", len(rows))
for key in keys:
    print(key, key in last)
print("continuation_sources", last.get("continuation_sources"))
print("eta", last.get("eta"))
print("beta", last.get("beta"))
print("stage", last.get("stage"))
print("time_grid", last.get("time_grid"))
PY
```

## 12. What To Return

Please return the following, in this order:

1. output from Section 0
2. output from Section 1
3. output from Section 2
4. output from Section 3
5. output from Section 4
6. output from Section 5
7. full stdout/stderr from:
   - `train.stdout_stderr.txt`
   - `sample.stdout_stderr.txt`
   - `eval.stdout_stderr.txt`
   - `diag.stdout_stderr.txt`
8. `tail -n 5 /tmp/dgtd_v3_round2_smoke/logs/train.jsonl`
9. file lists from:
   - `sample`
   - `eval`
   - `diag`
10. output from the structured log check in Section 11

## 13. If Any Step Fails

If any command fails, stop there and also return:

```bash
echo "==== pwd ===="
pwd

echo "==== python ===="
which python
python -V

echo "==== env roots ===="
echo "PROJ=$PROJ"
echo "FM_CONFIG=$FM_CONFIG"
echo "RUN_ROOT=$RUN_ROOT"
echo "CKPT_DIR=$CKPT_DIR"
echo "LOG_ROOT=$LOG_ROOT"
echo "METRIC_ROOT=$METRIC_ROOT"
echo "SAMPLE_ROOT=$SAMPLE_ROOT"
echo "TRAJ_ROOT=$TRAJ_ROOT"
echo "TORCH_HOME=$TORCH_HOME"
echo "HF_HOME=$HF_HOME"

echo "==== nvidia-smi ===="
nvidia-smi

echo "==== smoke tree ===="
find "$DGTD_V3_R2_ROOT" -maxdepth 4 | sort
```

If the failure is before Section 7, also return:

```bash
echo "==== activation output recheck ===="
source scripts/experiments/activate_fm_cifar10.sh dgtd_v3_smoke round2_verify
```

If the failure is specifically in Section 9 `eval`, also return:

```bash
echo "==== TORCH_HOME files ===="
find "$TORCH_HOME" -maxdepth 4 -type f | sort | head -n 100

echo "==== HF_HOME files ===="
find "$HF_HOME" -maxdepth 4 -type f | sort | head -n 100
```

That will give enough information to separate:

- code bug
- cache/path problem
- missing trajectory root
- checkpoint schema issue
- missing FID weights
- CUDA / driver / NCCL issue
