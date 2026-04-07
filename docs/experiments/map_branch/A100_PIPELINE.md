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

Before entering the pipeline, first update
[EXPERIMENT_LOG.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/EXPERIMENT_LOG.md).

Then activate the selected experiment once:

```bash
source scripts/experiments/activate_fm_cifar10.sh <EXP_VARIANT> <EXP_TAG>
```

This sets stable environment variables for all later commands in this document:
- `EXP_VARIANT=<selected variant>`
- `EXP_TAG=<selected tag>`
- `EXP_NAME=<resolved experiment name>`
- `EXP_SOURCE=<resolved config path>`
- `FM_CONFIG=<resolved config path>`
- `RUN_ROOT=/cache/Zhengwei/dgfm_runs/$FM_EXP`
- `CKPT_DIR=$RUN_ROOT/checkpoints`
- `SAMPLE_ROOT=$RUN_ROOT/samples`
- `LOG_ROOT=$RUN_ROOT/logs`
- `METRIC_ROOT=/cache/Zhengwei/dgfm_eval/$FM_EXP`
- `TRAJ_ROOT=/cache/Zhengwei/dgfm_teacher_traj/cifar10_ddpm128_p33`
- `HF_HOME=/cache/huggingface`
- `HF_HUB_CACHE=/cache/huggingface/hub`
- `HF_ENDPOINT=https://hf-mirror.com`
- `DGFM_ARCHIVE_ROOT=/temp/Zhengwei/dgfm_runs/$FM_EXP`

Recommended policy:
- keep this pipeline document stable
- switch runs by updating `EXPERIMENT_LOG.md`
- re-run the activation script with the selected `EXP_VARIANT` and `EXP_TAG`
- after activation, use the fixed commands below without editing this file

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

## 4. Teacher target mode

Current `map_branch` uses **online `teacher_sampler` targets** by default.
Offline teacher trajectories remain available as a fallback path, but they are
no longer the primary training mode.

Quick diagnostic variant:
- `map_branch_quick`
- teacher internal steps:
  - `32`
- target start scales:
  - `18`
- lighter endpoint supervision:
  - lower weight
  - lower frequency
  - smaller endpoint batch
- intended use:
  - quickly verify whether CTM-aligned changes produce a useful FID trend
  - not intended as the final reported run

Teacher rollout policy:
- backend:
  - `diffusers_ddpm`
- model:
  - `google/ddpm-cifar10-32`
- internal solver:
  - `ddim`
- internal teacher steps:
  - `128`
- target discrete scales:
  - `33`
- retained time semantics:
  - ascending `u-grid`
  - `u=0.0` means noisiest state
  - `u=1.0` means cleanest state

Primary online target path:
- sample teacher noise state `x_0`
- rollout teacher with `128` internal DDIM steps
- retain `33` CTM-like discrete scales on ascending `u-grid`
- sample a discrete start index `t_idx`
- sample `num_heun_step`
- sample `s_idx` from the valid suffix with `sample_s_strategy=uniform`
- supervise the explicit map on `M_theta(x_t, t, s) -> x_s_teacher`

Optional offline cache path:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/prepare_teacher_trajectories.py \
  --config $FM_CONFIG \
  --output-root $TRAJ_ROOT \
  --batch-size 64
```

Use the offline cache only when online teacher sampling is too slow for the
current server session.

If teacher loading fails with:

```text
LocalEntryNotFoundError: Cannot find an appropriate cached snapshot folder ...
```

use one of these two fixes.

Fix A: point to the shared HuggingFace cache before running the command:

```bash
export HF_HOME=/cache/huggingface
export HF_HUB_CACHE=/cache/huggingface/hub
export HF_ENDPOINT=https://hf-mirror.com
```

Fix B: allow online lookup once if the server has outbound access:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/prepare_teacher_trajectories.py \
  --config $FM_CONFIG \
  --output-root $TRAJ_ROOT \
  --batch-size 64 \
  --set teacher.local_files_only=false
```

When `teacher.local_files_only=false`, the current branch will use:

```bash
HF_ENDPOINT=https://hf-mirror.com
```

by default. Override it manually if your environment requires a different mirror
or the official HuggingFace endpoint.

If your server already has a local snapshot path, you can also bypass the repo
lookup entirely:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/prepare_teacher_trajectories.py \
  --config $FM_CONFIG \
  --output-root $TRAJ_ROOT \
  --batch-size 64 \
  --set teacher.name_or_path='/cache/huggingface/hub/models--google--ddpm-cifar10-32/snapshots/<snapshot_id>'
```

## 5. Map-branch training command

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py --config $FM_CONFIG --run-root $RUN_ROOT
```

If the online teacher is not already cached under `/cache/huggingface`, allow
one mirrored online fetch:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT \
  --set teacher.local_files_only=false
```

Quick diagnostic run:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py --config $FM_CONFIG --run-root $RUN_ROOT
```

Recommended use:
- first activate the quick experiment from `EXPERIMENT_LOG.md`
- run the same training command shown above
- check whether `train_pixel_loss / train_perceptual_loss / train_endpoint_loss` move in the expected direction
- then activate the full experiment and rerun the same command

Current target mode:
- `target.builder=teacher_sampler`

Current semantics:
- preserve `dgfm` time semantics
- sample `0 <= t < s <= 1`
- train `M_theta(x_t, t, s) -> x_s_teacher`

Current CTM-like discrete target policy:
- `target.sampling_mode = ctm_discrete`
- `target.start_scales = 33`
- `target.num_heun_step = 17`
- `target.num_heun_step_random = true`
- `target.heun_step_strategy = weighted`
- `target.sample_s_strategy = uniform`

CTM-aligned additions in the current branch:
- online teacher target construction
- CTM-like discrete time-pair sampling
- CTM-style preconditioning on the explicit map UNet
- perceptual loss on direct map matching
- endpoint few-step teacher loss on rollout from the same teacher `x_0`

Endpoint loss status:
- kept as an auxiliary interface
- not treated as the primary CTM-faithfulness mechanism
- safe to ablate independently in later experiments

Preconditioning is treated here as a training/sampling stabilization tip:
- normalize noisy inputs before they enter the map UNet
- keep a skip/output scaling around the current state
- make the few-step map updates numerically easier to learn
- do not treat it as a separate architecture line in the experiment narrative

Current A100-oriented throughput settings:
- `train.batch_size = 128`
- `train.num_workers = 8`
- `train.persistent_workers = true`
- `train.prefetch_factor = 4`
- `runtime.cudnn_benchmark = true`

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
  --fid-batch-size 128 \
  --sample-batch-size 32
```

Formal evaluation:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --eval-root $METRIC_ROOT \
  --steps 1 2 4 8 16
```

Current evaluation memory guards:
- `eval.sample_batch_size = 64`
- `eval.fixed_grid_batch_size = 16`
- use `--sample-batch-size` to shrink per-forward rollout further if `16`-step eval still pressures memory

## 8. Qualitative sampling command

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_sample.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --output-dir $SAMPLE_ROOT/steps16 \
  --steps 16 \
  --num-samples 64 \
  --sample-batch-size 16 \
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

## 11. Current target implementation

Primary implementation:
- `src/dgfm/targets/builder.py`
- `src/dgfm/teachers/diffusers_ddpm.py`

Optional offline cache path:
- `src/dgfm/datasets/trajectory.py`
- `scripts/prepare_teacher_trajectories.py`

## 12. Future time-warp integration

The current map branch is designed so time-warp attaches in two places:

1. sampler time grid
- replace uniform `t_0 < ... < t_K`
- preserve iterative map rollout

2. training tuple sampling
- replace current `(t, s)` sampling in `AnalyticPathTargetBuilder`
- keep trainer and model interface stable
