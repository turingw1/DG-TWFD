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

export CIFAR_CKPT=/data2/yl7622/Zhengwei/cifar10_ctm_checkpoint.pt
export CIFAR_REF=/data2/yl7622/Zhengwei/cifar10-32x32.npz
export CIFAR_OUT=/data2/yl7622/Zhengwei/output/cifar10

export IM64_CKPT=/data2/yl7622/Zhengwei/DG-TWFD/ct_imagenet64.pt
export IM64_REF=/data2/yl7622/Zhengwei/DG-TWFD/VIRTUAL_imagenet64_labeled.npz
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

```bash
conda activate consistency
bash docs/baseline/reproduce_ctm_baselines.sh
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
