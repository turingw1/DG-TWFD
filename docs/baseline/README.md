# CTM Baseline Reproduction

This folder contains a server-side reproduction helper for CTM baselines on
CIFAR10 and ImageNet64.

The local reference repositories are expected at:

- `refs/ctm-cifar10`, with origin `https://github.com/turingw1/ctm-cifar10.git`
- `refs/ctm`, with origin `https://github.com/turingw1/ctm.git`

Use `reproduce_ctm_baselines.sh` on a two-GPU server, for example two A6000
cards. The script generates and evaluates 1, 2, 4, and 8 step CTM samples.

## Required Inputs

Set these paths before running:

```bash
export REPO_ROOT=/data2/yl7622/Zhengwei/DG-TWFD

export CIFAR_CKPT=/data2/yl7622/Zhengwei/DG-TWFD/author_ckpt/cifar10_ctm_checkpoint.pt
export CIFAR_REF=/data2/yl7622/Zhengwei/cifar10-32x32.npz
export CIFAR_OUT=/data2/yl7622/Zhengwei/output/cifar10

export IM64_CKPT=/data2/yl7622/Zhengwei/DG-TWFD/author_ckpt/ema_0.999_049000.pt
export IM64_REF=/data2/yl7622/Zhengwei/DG-TWFD/author_ckpt/VIRTUAL_imagenet64_labeled.npz
export IM64_OUT=/data2/yl7622/Zhengwei/output/imagenet64
```

Optional controls:

```bash
export STEPS="1 2 4 8"
export TOTAL_SAMPLES=50000
export CIFAR_BATCH=1000
export IM64_BATCH=250
export RUN_CIFAR=0
export RUN_IM64=1
export RUN_EVAL=1
```

## Run

For the current ImageNet64-only conda path, use:

```bash
conda activate ctm
export REPO_ROOT=/path/to/DG-TWFD
export IM64_CKPT=/path/to/imagenet64_ctm_checkpoint.pt
export IM64_REF=/path/to/VIRTUAL_imagenet64_labeled.npz
export IM64_OUT=/path/to/output/imagenet64
bash docs/baseline/reproduce_ctm_imagenet64_conda.sh
```

`IM64_CKPT` must be an ImageNet64 CTM checkpoint, typically named like
`ema_0.999_XXXXXX.pt`. Do not point this runner at `edm_imagenet64_ema.pt` or
`cd_imagenet64_lpips.pt`; those checkpoints do not contain the CTM-specific
`time_embed_s` and `emb_layers_s` weights required by `--training_mode=ctm`.

This path forces `--attention_type=legacy` and the local `refs/ctm/code` import
now falls back when `flash_attn` or `xformers` is absent. It is slower than
flash attention but avoids the CUDA toolkit and wheel compatibility issues that
make conda-only reproduction fragile.

The older combined CIFAR10/ImageNet64 helper is still available:

```bash
conda activate consistency
bash docs/baseline/reproduce_ctm_baselines.sh
```

## Train CTM+DSM Without GAN

Store ImageNet64 data outside `/data2` and `/data` if those filesystems are
short on space. On the current server, use `/homes/yl7622/Zhengwei` as the
dataset root:

```bash
export SERVER_ROOT=/homes/yl7622/Zhengwei
export IM64_DATA_ROOT=$SERVER_ROOT/datasets/imagenet64
export KAGGLE_CONFIG_DIR=$SERVER_ROOT/.kaggle

mkdir -p "$IM64_DATA_ROOT" "$KAGGLE_CONFIG_DIR"
chmod 700 "$KAGGLE_CONFIG_DIR"
```

If a partial download already exists under `/data/yl7662/Zhengwei`, remove only
the old ImageNet64 dataset/cache paths before re-downloading:

```bash
rm -rf /data/yl7662/Zhengwei/datasets/imagenet64
rm -rf /data/yl7662/Zhengwei/.kaggle
```

Place `kaggle.json` under `$KAGGLE_CONFIG_DIR/kaggle.json`, then download and
unzip the dataset:

```bash
python -m pip install kaggle
kaggle datasets download -d wangzilin20078/imagenet64 --unzip -p "$IM64_DATA_ROOT"
```

After unzip, inspect the layout and set `IM64_DATA_DIR` to the directory that
contains the training images or class subdirectories:

```bash
find "$IM64_DATA_ROOT" -maxdepth 3 -type d | head -40
find "$IM64_DATA_ROOT" -maxdepth 3 -type f | head -20

# Common choices, depending on the unzip layout:
export IM64_DATA_DIR=$IM64_DATA_ROOT/train
# or:
# export IM64_DATA_DIR=$IM64_DATA_ROOT
```

To quickly train an ImageNet64 CTM checkpoint without the GAN stage, use:

```bash
conda activate ctm
export REPO_ROOT=/data2/yl7622/Zhengwei/DG-TWFD
export IM64_TEACHER=/data2/yl7622/Zhengwei/DG-TWFD/author_ckpt/edm_imagenet64_ema.pt
export IM64_DATA_DIR=/homes/yl7622/Zhengwei/datasets/imagenet64/train
export IM64_OUT=/data2/yl7622/Zhengwei/output/CTM_DSM

export TRAIN_STEPS=10000
export SAVE_INTERVAL=1000
export GLOBAL_BATCH_SIZE=128
export MICROBATCH=8
export LR=0.00004

bash docs/baseline/train_ctm_imagenet64_dsm_conda.sh
```

This explicitly sets `--gan_training=False` and `--diffusion_training=True`.
It also sets `--eval_interval=-1` and `--save_check_period=-1` by default, so
training does not spend time sampling/evaluating during the run. Checkpoints are
saved by `--save_interval` and at the end of training:

```text
model010000.pt
ema_0.999_010000.pt
ema_0.9999_010000.pt
ema_0.9999432189950708_010000.pt
opt010000.pt
target_model010000.pt
```

If the environment is missing runtime dependencies, install them before
sampling. The minimum modules used by the baseline script are:

```bash
python -m pip install blobfile mpi4py numpy scipy torch torchvision
python -m pip install flash-attn --no-build-isolation
python -m pip install "tensorflow[and-cuda]"
```

If your server already provides CUDA-enabled PyTorch through conda or a cluster
module, keep that PyTorch install and only add missing packages such as
`blobfile`, `mpi4py`, `flash-attn`, and `tensorflow`. ImageNet64 imports
`flash_attn` at module load time, so it is required even when only sampling.
ImageNet64 training may also require `xformers`.

The script launches one independent `mpiexec -n 1` process per GPU. This is
intentional: the upstream sampling scripts only save samples from rank 0, so a
single `mpiexec -n 2` job would waste the second GPU instead of writing twice as
many samples.
https://drive.usercontent.google.com/download?id=1GQW-iHMpDWQyC2L3LD2m7S2AyHscS4Sp&export=download&authuser=0&confirm=t&uuid=854a8ee3-1385-4c0f-ab0b-d45b2acd0e01&at=ALBwUgloNvAAwbRKIhSpdwLC9sXm:1776790262136
## Outputs

For each step, generated samples are written under the CTM script's automatic
directory name, for example:

```text
ctm_exact_sampler_1_steps_XXXXXX_itrs_0.999_ema_
ctm_exact_sampler_2_steps_XXXXXX_itrs_0.999_ema_
ctm_exact_sampler_4_steps_XXXXXX_itrs_0.999_ema_
ctm_exact_sampler_8_steps_XXXXXX_itrs_0.999_ema_
```

CIFAR10 evaluation uses `fid_npzs.py`. ImageNet64 evaluation uses
`evaluations/evaluator.py`, matching the CTM README and reporting FID, sFID,
Inception Score, precision, and recall.
