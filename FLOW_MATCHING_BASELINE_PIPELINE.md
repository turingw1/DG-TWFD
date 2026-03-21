# Flow Matching Baseline Pipeline

This is the Phase-1 baseline training path for the refactored framework.

## 0. Activate Experiment

```bash
cd ~/workspace/Zhengwei/DG-TWFD
source scripts/experiments/activate_fm_cifar10.sh baseline v1
conda activate consistency
```

Current variants:
- `baseline`
- `stable`

## 1. Environment

```bash
cd $PROJ
python -m pip install -U pip
python -m pip install -e .
python -m pip install -e ./flow_matching
python -m pip install torchvision torchmetrics torchdiffeq clean-fid pandas
```

## 2. Dataset Preparation

### CIFAR-10

```bash
cd $PROJ
python scripts/build_dataset.py --dataset cifar10 --data-root $DATA_ROOT/cifar10
```

### Future ImageNet Preparation

```bash
python scripts/build_dataset.py --dataset imagenet32 --data-root $DATA_ROOT/imagenet32
python scripts/build_dataset.py --dataset imagenet64 --data-root $DATA_ROOT/imagenet64
```

## 3. Expected Result Layout

```text
$RUN_ROOT/
├── checkpoints/
├── logs/
└── samples/

$METRIC_ROOT/
├── steps1/
├── steps2/
├── steps4/
├── steps8/
├── steps16/
└── reports/
```

## 4. Baseline Training Command

```bash
cd $PROJ
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT
```

Produced files:
- `$CKPT_DIR/best.pt`
- `$CKPT_DIR/last.pt`
- `$LOG_ROOT/config_resolved.yaml`
- `$LOG_ROOT/train.jsonl`

## 5. Smoke Training Command

Use this before a long run:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT/smoke \
  --set train.epochs=1 \
  --set train.batch_size=8 \
  --set train.num_workers=0
```

## 6. Resume Training

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT \
  --resume $CKPT_DIR/last.pt
```

## 7. Evaluate a Checkpoint

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --eval-root $METRIC_ROOT \
  --steps 1 2 4 8 16
```

Current Phase-1 evaluator exports:
- cached `samples.pt`
- `grid.png`
- lightweight `metrics.json`
- `reports/summary.json`

Full FID/KID integration is the next implementation slice.

## 8. Qualitative Sampling

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_sample.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --output-dir $SAMPLE_ROOT/steps16 \
  --steps 16 \
  --num-samples 64 \
  --fixed-seed 42
```

## 9. Few-Step Evaluation Protocol
Mandatory step counts:
- `1`
- `2`
- `4`
- `8`
- `16`

## 10. Benchmark Targets
Use the Flow Matching paper as the scientific baseline reference, and the vendored `flow_matching` repo example results as the engineering reference.

Reference targets to track:
- CIFAR-10 paper baseline: `FID ~= 6.35`
- ImageNet 32x32 paper baseline: `FID ~= 5.02`
- ImageNet 64x64 paper baseline: `FID ~= 14.45`
- vendored image example README for CIFAR-10: `FID 2.07`

Interpretation:
- first get stable training and sensible qualitative samples,
- then add full FID evaluation,
- only then compare against the paper and repo-example targets.

## 11. Teacher Interface Usage
Phase 1 baseline uses:
- `teacher.type = none`

Future teacher-based runs must keep the same CLI shape and change only config.

## 12. Time-Warp Hook
Time-warp is not active in Phase 1 baseline training. The hook is reserved in scheduler config and will be enabled through config, not new CLI entrypoints.
