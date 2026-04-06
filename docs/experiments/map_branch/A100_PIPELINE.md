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
conda activate /cache/$USER/conda_envs/dgfm_map
```

## 2. Activate experiment

```bash
source scripts/experiments/activate_fm_cifar10.sh map_branch v2
```

This sets:
- `FM_CONFIG=configs/experiment/fm_cifar10_map_branch.yaml`
- `RUN_ROOT=/cache/Zhengwei/dgfm_runs/fm_cifar10_map_branch_v2`
- `CKPT_DIR=$RUN_ROOT/checkpoints`
- `SAMPLE_ROOT=$RUN_ROOT/samples`
- `LOG_ROOT=$RUN_ROOT/logs`
- `METRIC_ROOT=/cache/Zhengwei/dgfm_eval/fm_cifar10_map_branch_v2`
- `TRAJ_ROOT=/cache/Zhengwei/dgfm_teacher_traj/cifar10_ddpm128_p33`
- `DGFM_ARCHIVE_ROOT=/temp/Zhengwei/dgfm_runs/fm_cifar10_map_branch_v2`

Training will mirror:
- `logs/config_resolved.yaml`
- `logs/train.jsonl`
- `checkpoints/last.pt`
- `checkpoints/best.pt`

into the archive root during training.

## 3. CIFAR-10 preparation

Manual-first check:

```bash
python scripts/build_dataset.py --dataset cifar10 --data-root $DATA_ROOT/cifar10
```

If missing and you explicitly want download:

```bash
python scripts/build_dataset.py --dataset cifar10 --data-root $DATA_ROOT/cifar10 --download
```

## 4. Teacher trajectory preparation

Current map-branch supervision comes from offline teacher trajectories, not
from the old analytic path target.

Teacher rollout policy:
- backend:
  - `diffusers_ddpm`
- model:
  - `google/ddpm-cifar10-32`
- internal solver:
  - `ddim`
- internal teacher steps:
  - `128`
- retained trajectory anchors:
  - `33`
- retained time semantics:
  - ascending `u-grid`
  - `u=0.0` means noisiest state
  - `u=1.0` means cleanest state

Rationale:
- keep teacher rollout strong with `128` internal steps
- retain only `33` anchors to keep shard size manageable
- still cover local / mid / endpoint transitions for `1/2/4/8/16` map rollout evaluation

Prepare the trajectory cache:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/prepare_teacher_trajectories.py \
  --config $FM_CONFIG \
  --output-root $TRAJ_ROOT \
  --batch-size 64
```

Expected files:
- `$TRAJ_ROOT/train/manifest.yaml`
- `$TRAJ_ROOT/val/manifest.yaml`
- `$TRAJ_ROOT/train/train_00000.pt`
- `$TRAJ_ROOT/val/val_00000.pt`

## 5. Map-branch training command

```bash
source scripts/experiments/activate_fm_cifar10.sh map_branch v2
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py --config $FM_CONFIG --run-root $RUN_ROOT
```

Current target mode:
- `target.builder=trajectory_shard`

Current semantics:
- preserve `dgfm` time semantics
- sample `0 <= t < s <= 1`
- train `M_theta(x_t, t, s) -> x_s_teacher`

Pair sampling policy on retained teacher grid:
- `pair_short_max = 4`
- `pair_mid_max = 12`
- `pair_long_max = 32`
- `pair_endpoint_weight = 0.35`
- `high_noise_t_weight = 0.75`
- `high_noise_t_fraction = 0.35`

This mirrors the CTM intuition:
- sample a state on a strong teacher trajectory
- choose a later teacher state
- match the explicit map to that teacher transition

Current A100-oriented throughput settings:
- `train.batch_size = 256`
- `train.num_workers = 8`
- `train.persistent_workers = true`
- `train.prefetch_factor = 4`
- `runtime.cudnn_benchmark = true`

These changes are intended to improve throughput while keeping the training
objective and model architecture unchanged.

## 6. Resume command

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT \
  --resume $CKPT_DIR/last.pt
```

## 7. Map-branch evaluation command

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

## 8. Qualitative sampling command

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_sample.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --output-dir $SAMPLE_ROOT/steps16 \
  --steps 16 \
  --num-samples 64 \
  --fixed-seed 42
```

## 9. Multistep panel command

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_multistep_panel.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --output-dir $SAMPLE_ROOT/multistep_panel \
  --steps 1 2 4 8 16 \
  --num-examples 8 \
  --fixed-seed 42
```

## 10. Output layout

Training:
- `$CKPT_DIR/best.pt`
- `$CKPT_DIR/last.pt`
- `$LOG_ROOT/train.jsonl`
- `$LOG_ROOT/config_resolved.yaml`
- `$TRAJ_ROOT/train/manifest.yaml`
- `$TRAJ_ROOT/val/manifest.yaml`
- `/temp/Zhengwei/dgfm_runs/<FM_EXP>/checkpoints/best.pt`
- `/temp/Zhengwei/dgfm_runs/<FM_EXP>/checkpoints/last.pt`
- `/temp/Zhengwei/dgfm_runs/<FM_EXP>/logs/train.jsonl`
- `/temp/Zhengwei/dgfm_runs/<FM_EXP>/logs/config_resolved.yaml`

Evaluation:
- `$METRIC_ROOT/reports/summary.csv`
- `$METRIC_ROOT/reports/best.json`
- `$METRIC_ROOT/steps16/fixed_seed_grid.png`

Sampling:
- `$SAMPLE_ROOT/steps16/grid.png`
- `$SAMPLE_ROOT/multistep_panel/multistep_panel.png`

## 11. Current teacher trajectory implementation

Implemented in:
- `src/dgfm/teachers/diffusers_ddpm.py`
- `src/dgfm/datasets/trajectory.py`
- `src/dgfm/targets/builder.py`
- `scripts/prepare_teacher_trajectories.py`

The current branch trains from cached teacher trajectories. Online
`teacher_sampler` remains a future extension.

## 12. Future online teacher sampler mode

Not implemented in the first map-branch patch.

Planned insertion point:
- `src/dgfm/targets/builder.py`
- future mode:
  - `target.builder=teacher_sampler`

Expected future role:
- use a high-NFE teacher rollout to build `x_s_target`
- compare explicit map learning against sampler-distilled targets

## 13. Future time-warp integration

The current map branch is designed so time-warp attaches in two places:

1. sampler time grid
- replace uniform `t_0 < ... < t_K`
- preserve iterative map rollout

2. training tuple sampling
- replace current `(t, s)` sampling in `AnalyticPathTargetBuilder`
- keep trainer and model interface stable
