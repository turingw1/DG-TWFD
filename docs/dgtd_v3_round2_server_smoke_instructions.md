# DGTD v3 Round-2 Server Smoke Instructions

This document is the server-side acceptance checklist for commit:

- `054351e Patch DGTD v3 residual teacher and sigma core`

Use this document on the server only.

Per the preserved workflow in:

- [docs/experiments/map_branch/HANDOFF_2026-04-16.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/HANDOFF_2026-04-16.md)
- [docs/experiments/map_branch/ENVIRONMENT.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/ENVIRONMENT.md)
- [docs/experiments/DG_TWFD_v3/DEVELOPMENT_VALIDATION.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/DG_TWFD_v3/DEVELOPMENT_VALIDATION.md)

local runtime results do **not** count as acceptance evidence for this branch.

## Goal

Return enough server evidence to verify:

1. smoke train runs end-to-end
2. checkpoint saves and loads
3. DGTD sample uses warp `r_to_t`
4. eval runs on the saved checkpoint
5. `train.jsonl` contains the new DGTD diagnostics
6. diagnostics plot export succeeds

## 0. Server Session Setup

Run these first and paste the outputs back.

```bash
cd /data2/yl7622/Zhengwei/DG-TWFD
source /data2/yl7622/anaconda/etc/profile.d/conda.sh
conda activate /data2/yl7622/Zhengwei/DG-TWFD/.conda_envs/dgfm_map

git branch --show-current
git rev-parse HEAD
git show --stat --oneline --no-patch 054351e

source scripts/experiments/activate_fm_cifar10.sh dgtd_v3_smoke round2_verify

echo "TRAJ_ROOT=$TRAJ_ROOT"
find "$TRAJ_ROOT" -maxdepth 2 | head -n 20

python scripts/run_train.py --help | head -n 20
python scripts/run_sample_dgtd.py --help | head -n 20
python scripts/run_eval.py --help | head -n 20
```

Pass condition:

- conda env activates
- target cache is visible
- the three entrypoints import and print help

## 1. Clean Smoke Root

Use a dedicated temporary root for this verification round.

```bash
export DGTD_V3_R2_ROOT=/tmp/dgtd_v3_round2_smoke
rm -rf "$DGTD_V3_R2_ROOT"
mkdir -p "$DGTD_V3_R2_ROOT"
```

If you do not want to remove an existing root, replace the path with another empty temp directory.

## 2. Smoke Train

Run the train smoke exactly once and capture stdout/stderr.

```bash
python scripts/run_train.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --run-root "$DGTD_V3_R2_ROOT" \
  --set target.shard_root="$TRAJ_ROOT" \
  2>&1 | tee "$DGTD_V3_R2_ROOT/train.stdout_stderr.txt"
```

Immediately after train finishes, run:

```bash
echo "==== checkpoints ===="
find "$DGTD_V3_R2_ROOT/checkpoints" -maxdepth 2 -type f | sort

echo "==== logs ===="
find "$DGTD_V3_R2_ROOT/logs" -maxdepth 2 -type f | sort

echo "==== tail train.jsonl ===="
tail -n 5 "$DGTD_V3_R2_ROOT/logs/train.jsonl"
```

## 3. Smoke Sample

Run sample from the saved `last.pt`.

```bash
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

## 4. Smoke Eval

Run eval from the same `last.pt`.

```bash
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

## 5. Diagnostics Export

Export the DGTD diagnostic plots from the produced history file.

```bash
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

## 6. Structured Log Check

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

## 7. What To Return

Please return the following, in this order:

1. output of the setup block in Section 0
2. full stdout/stderr from:
   - `train.stdout_stderr.txt`
   - `sample.stdout_stderr.txt`
   - `eval.stdout_stderr.txt`
   - `diag.stdout_stderr.txt`
3. `tail -n 5 /tmp/dgtd_v3_round2_smoke/logs/train.jsonl`
4. file lists from:
   - `sample`
   - `eval`
   - `diag`
5. output of the structured log check in Section 6

## 8. If Any Step Fails

If any command fails, stop there and also return:

```bash
echo "==== pwd ===="
pwd

echo "==== python ===="
which python
python -V

echo "==== env roots ===="
echo "PROJ=$PROJ"
echo "TRAJ_ROOT=$TRAJ_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
echo "TORCH_HOME=$TORCH_HOME"
echo "HF_HOME=$HF_HOME"

echo "==== nvidia-smi ===="
nvidia-smi

echo "==== smoke tree ===="
find "$DGTD_V3_R2_ROOT" -maxdepth 4 | sort
```

If the failure is during `eval`, also return:

```bash
find "$TORCH_HOME" -maxdepth 4 -type f | sort | head -n 100
find "$HF_HOME" -maxdepth 4 -type f | sort | head -n 100
```

That will give enough information to separate:

- code bug
- cache/path problem
- checkpoint schema issue
- missing FID weights
- CUDA / driver / NCCL issue
