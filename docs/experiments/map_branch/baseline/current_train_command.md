# Current `dgfm` Baseline Train Command

This file records the current map-branch training command on the `dgfm` line.

Primary operational source:

- `docs/experiments/map_branch/A100_PIPELINE.md`

## Current clean baseline

The current clean baseline recipe is:

- variant: `fm_cifar10_map_branch_s1_e6_budget_full`
- tag in the handoff: `e602a`

Activation:

```bash
source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e6_budget_full e602a
```

Train:

```bash
CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES} torchrun \
  --standalone \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT \
  --verbose
```

Resume:

```bash
CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES} torchrun \
  --standalone \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT \
  --resume $CKPT_DIR/last.pt \
  --verbose
```

## Current fullstack reference

If endpoint and learnable timewarp need to stay on, the current fullstack
variant is:

- variant: `fm_cifar10_map_branch_s1_e6_budget_fullstack`
- tag in the handoff: `e603a`

Activation:

```bash
source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e6_budget_fullstack e603a
```

The train command itself stays the same after activation.

## Current policy

The current line follows these rules:

1. activate through `scripts/experiments/activate_fm_cifar10.sh`
2. use committed config files only
3. do not use `--set` in formal experiments
4. treat `docs/experiments/map_branch/EXPERIMENT_LOG.md` as the experiment
   entry ledger

## Current eval command family

Few-step eval:

```bash
CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} python scripts/run_eval.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --eval-root $METRIC_ROOT \
  --steps 1 2 4 8 16 32 64 128 256
```

Held-out defect:

```bash
CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} python scripts/run_evaluate_defect.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --out $METRIC_ROOT/defect/heldout.json
```

Official-style metrics:

```bash
CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} python scripts/run_export_samples_npz.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --out $METRIC_ROOT/official/step16_samples.npz \
  --steps 16

CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} python scripts/run_evaluate_metrics.py \
  --config $FM_CONFIG \
  --samples $METRIC_ROOT/official/step16_samples.npz \
  --reference ${OFFICIAL_REFERENCE_NPZ:-$IMAGENET64_REFERENCE_NPZ} \
  --out $METRIC_ROOT/official/step16_metrics.json
```
