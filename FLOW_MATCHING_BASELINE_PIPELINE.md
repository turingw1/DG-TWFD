# Flow Matching Baseline Pipeline

This document is the Phase-1 replacement for the old A100 DDPM pipeline. It keeps the same workflow philosophy:
- one activation path,
- one training path,
- one evaluation path,
- one result root.

## 0. Scope
Phase 1 targets:
- baseline continuous flow matching reproduction,
- CIFAR-10 first,
- few-step evaluation under one interface,
- future-ready teacher and time-warp hooks.

Phase 1 does not include semigroup defect in the core training path.

## 1. Environment

```bash
cd ~/workspace/Zhengwei/DG-TWFD
conda create -n consistency python=3.10 -y
conda activate consistency
python -m pip install -U pip
python -m pip install -e .
python -m pip install -e ./flow_matching
python -m pip install torchmetrics torchvision torchdiffeq clean-fid pandas
```

Recommended runtime roots:

```bash
export PROJ=~/workspace/Zhengwei/DG-TWFD
export DATA_ROOT=/cache/Zhengwei/datasets
export RUNS_ROOT=/cache/Zhengwei/dgfm_runs
export EVAL_ROOT=/cache/Zhengwei/dgfm_eval
export FM_EXP=fm_cifar10_baseline_v1
export RUN_ROOT=$RUNS_ROOT/$FM_EXP
export CKPT_DIR=$RUN_ROOT/checkpoints
export SAMPLE_ROOT=$RUN_ROOT/samples
export METRIC_ROOT=$EVAL_ROOT/$FM_EXP
mkdir -p "$CKPT_DIR" "$SAMPLE_ROOT" "$METRIC_ROOT"
```

## 2. Expected Directory Layout

```text
/cache/Zhengwei/
├── datasets/
│   ├── cifar10/
│   ├── imagenet32/
│   └── imagenet64/
├── dgfm_runs/
│   └── fm_cifar10_baseline_v1/
│       ├── checkpoints/
│       ├── logs/
│       └── samples/
└── dgfm_eval/
    └── fm_cifar10_baseline_v1/
        ├── ckpt_best/
        ├── ckpt_last/
        └── reports/
```

## 3. Dataset Preparation

### CIFAR-10
No manual preprocessing is required if torchvision download is enabled.

```bash
cd $PROJ
conda activate consistency
python scripts/build_dataset.py --dataset cifar10 --data-root $DATA_ROOT/cifar10
```

Expected:
- train and test data are available under `$DATA_ROOT/cifar10`
- the config uses `train` for training and `test` or a held-out validation policy for evaluation

### Future ImageNet Preparation
Not required for Phase 1, but reserve:

```bash
python scripts/build_dataset.py --dataset imagenet32 --data-root $DATA_ROOT/imagenet32
python scripts/build_dataset.py --dataset imagenet64 --data-root $DATA_ROOT/imagenet64
```

## 4. Baseline Training Command for CIFAR-10

Primary launch command:

```bash
cd $PROJ
conda activate consistency
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py \
  --config configs/experiment/fm_cifar10_baseline.yaml \
  --run-root $RUN_ROOT
```

Required outputs:
- `$RUN_ROOT/checkpoints/best.pt`
- `$RUN_ROOT/checkpoints/last.pt`
- `$RUN_ROOT/logs/train.jsonl`
- `$RUN_ROOT/logs/config_resolved.yaml`

## 5. Resume Training

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py \
  --config configs/experiment/fm_cifar10_baseline.yaml \
  --run-root $RUN_ROOT \
  --resume $CKPT_DIR/last.pt
```

## 6. Baseline Evaluation Command

Evaluate the best checkpoint over multiple NFEs in one call:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval.py \
  --config configs/experiment/fm_cifar10_baseline.yaml \
  --checkpoint $CKPT_DIR/best.pt \
  --eval-root $METRIC_ROOT \
  --steps 1 2 4 8 16
```

Expected outputs:
- per-step generated samples
- FID metrics
- fixed-seed preview grids
- one summary table under `$METRIC_ROOT/reports/summary.csv`

## 7. Sampling and Visualization Command

Single checkpoint qualitative sampling:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_sample.py \
  --config configs/experiment/fm_cifar10_baseline.yaml \
  --checkpoint $CKPT_DIR/best.pt \
  --output-dir $SAMPLE_ROOT \
  --steps 16 \
  --num-samples 64 \
  --fixed-seed 42
```

Expected outputs:
- `$SAMPLE_ROOT/steps16/samples.pt`
- `$SAMPLE_ROOT/steps16/grid.png`
- `$SAMPLE_ROOT/steps16/step_stats.json`

## 8. Evaluate a Specific Checkpoint

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval.py \
  --config configs/experiment/fm_cifar10_baseline.yaml \
  --checkpoint $CKPT_DIR/epoch_0100.pt \
  --eval-root $METRIC_ROOT/epoch_0100 \
  --steps 4 8 16
```

## 9. Few-Step Evaluation Protocol
Mandatory step counts in Phase 1:
- `1`
- `2`
- `4`
- `8`
- `16`

Each evaluation run must export:
- `metrics.json`
- `metrics.csv`
- `grid.png`
- `sample_manifest.json`
- optional cached generated samples for metric reuse

## 10. Logs, Checkpoints, Images, Metrics

- Training logs: `$RUN_ROOT/logs/`
- Checkpoints: `$RUN_ROOT/checkpoints/`
- Qualitative samples: `$RUN_ROOT/samples/`
- Evaluation metrics: `$METRIC_ROOT/`
- Final tables: `$METRIC_ROOT/reports/`

## 11. Reference Targets from Flow Matching Paper
Use the original Flow Matching paper as the baseline reference for practical target ranges on supported image datasets.

Paper-level reference targets for FM with OT paths:
- CIFAR-10: `FID ~= 6.35`, `NFE ~= 142`
- ImageNet 32x32: `FID ~= 5.02`, `NFE ~= 122`
- ImageNet 64x64: `FID ~= 14.45`, `NFE ~= 138`

These values are from the original FM paper benchmark table as surfaced in public summaries of the paper's reported results. The vendored `flow_matching` example README reports stronger later-guide-code results for CIFAR-10 (`FID 2.07` with a different training/eval recipe). Phase 1 should therefore track two references:
- paper reference target: baseline scientific sanity check,
- repo-example reference target: practical implementation target for the vendored code family.

Use the following interpretation:
- first reproduce stable CIFAR-10 training with correct qualitative samples,
- then compare your FID curve against the paper target range,
- only after that compare against the stronger example-repo target.

## 12. Future ImageNet Preparation
Phase 1 does not require full ImageNet runs, but the framework must already support:
- dataset configs for `imagenet32` and `imagenet64`,
- image-folder loading,
- class-conditional labels,
- distributed evaluation.

Reserved launch style:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_train.py \
  --config configs/experiment/fm_imagenet32_baseline.yaml \
  --run-root /cache/Zhengwei/dgfm_runs/fm_imagenet32_baseline_v1
```

## 13. Teacher Interface Usage
Baseline training must default to:
- `teacher.type = none`

Future teacher-based experiments must preserve the same CLI shape:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py \
  --config configs/experiment/fm_cifar10_teacher.yaml \
  --run-root /cache/Zhengwei/dgfm_runs/fm_cifar10_teacher_v1
```

Only the config changes. The trainer entrypoint stays the same.

## 14. Future Time-Warp Hooks
Time-warp is not enabled in baseline Phase 1 training, but the framework must reserve:
- scheduler wrapper insertion,
- warped time sampling for evaluation,
- identity-warp default for baseline runs.

The user-facing command must remain unchanged; time-warp should be enabled by config only.
