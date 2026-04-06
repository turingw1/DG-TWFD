# MAP Branch Experiment Pipeline

## 1. Environment

If the server does not already have a usable conda environment, create one
first:

```bash
cd ~/workspace/Zhengwei/DG-TWFD
bash scripts/experiments/create_map_branch_env.sh dgfm_map
```

Then use:

```bash
cd ~/workspace/Zhengwei/DG-TWFD
git checkout map_branch_ctm_explicit_map
git pull --ff-only
conda activate dgfm_map
```

## 2. Activate experiment

```bash
source scripts/experiments/activate_fm_cifar10.sh map_branch v1
```

This sets:
- `FM_CONFIG=configs/experiment/fm_cifar10_map_branch.yaml`
- `RUN_ROOT=/cache/Zhengwei/dgfm_runs/fm_cifar10_map_branch_v1`
- `CKPT_DIR=$RUN_ROOT/checkpoints`
- `SAMPLE_ROOT=$RUN_ROOT/samples`
- `LOG_ROOT=$RUN_ROOT/logs`
- `METRIC_ROOT=/cache/Zhengwei/dgfm_eval/fm_cifar10_map_branch_v1`

## 3. Dataset preparation

Manual-first check:

```bash
python scripts/build_dataset.py --dataset cifar10 --data-root $DATA_ROOT/cifar10
```

If missing and you explicitly want download:

```bash
python scripts/build_dataset.py --dataset cifar10 --data-root $DATA_ROOT/cifar10 --download
```

## 4. Map-branch training command

```bash
source scripts/experiments/activate_fm_cifar10.sh map_branch v1
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py --config $FM_CONFIG --run-root $RUN_ROOT
```

Current first-stage target mode:
- `target.builder=analytic_path`

Current semantics:
- preserve `dgfm` time semantics
- sample `0 <= t < s <= 1`
- train `M_theta(x_t, t, s) -> x_s`

## 5. Resume command

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT \
  --resume $CKPT_DIR/last.pt
```

## 6. Map-branch evaluation command

If FID weights download is slow on A100, set a mirror before evaluation:

```bash
export DGFM_TORCH_FIDELITY_MIRROR_PREFIX=https://githubfast.com/
```

If you already have the Inception weights locally, use:

```bash
export DGFM_TORCH_FIDELITY_WEIGHTS_PATH=/path/to/weights-inception-2015-12-05-6726825d.pth
```

Smoke evaluation:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --eval-root $METRIC_ROOT/smoke \
  --steps 1 2 4 8 16 \
  --fid-samples 5000 \
  --fid-batch-size 128
```

Formal evaluation:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --eval-root $METRIC_ROOT \
  --steps 1 2 4 8 16
```

## 7. Qualitative sampling command

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_sample.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --output-dir $SAMPLE_ROOT/steps16 \
  --steps 16 \
  --num-samples 64 \
  --fixed-seed 42
```

## 8. Multistep panel command

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_multistep_panel.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --output-dir $SAMPLE_ROOT/multistep_panel \
  --steps 1 2 4 8 16 \
  --num-examples 8 \
  --fixed-seed 42
```

## 9. Output layout

Training:
- `$CKPT_DIR/best.pt`
- `$CKPT_DIR/last.pt`
- `$LOG_ROOT/train.jsonl`
- `$LOG_ROOT/config_resolved.yaml`

Evaluation:
- `$METRIC_ROOT/reports/summary.csv`
- `$METRIC_ROOT/reports/best.json`
- `$METRIC_ROOT/steps16/fixed_seed_grid.png`

Sampling:
- `$SAMPLE_ROOT/steps16/grid.png`
- `$SAMPLE_ROOT/multistep_panel/multistep_panel.png`

## 10. Trajectory-shard teacher targets

Not implemented in the first map-branch patch.

Planned insertion point:
- `src/dgfm/targets/builder.py`
- future mode:
  - `target.builder=trajectory_shard`

Expected future role:
- consume cached `(x_t, t, s, x_s_target)` supervision
- keep `MapTrainer` unchanged

## 11. Future teacher sampler mode

Not implemented in the first map-branch patch.

Planned insertion point:
- `src/dgfm/targets/builder.py`
- future mode:
  - `target.builder=teacher_sampler`

Expected future role:
- use a high-NFE teacher rollout to build `x_s_target`
- compare explicit map learning against sampler-distilled targets

## 12. Future time-warp integration

The current map branch is designed so time-warp attaches in two places:

1. sampler time grid
- replace uniform `t_0 < ... < t_K`
- preserve iterative map rollout

2. training tuple sampling
- replace current `(t, s)` sampling in `AnalyticPathTargetBuilder`
- keep trainer and model interface stable
