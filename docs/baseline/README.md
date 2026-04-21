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
export REPO_ROOT=/path/to/DG-TWFD

export CIFAR_CKPT=/path/to/cifar10_ctm_checkpoint.pt
export CIFAR_REF=/path/to/cifar10-32x32.npz
export CIFAR_OUT=/path/to/output/cifar10

export IM64_CKPT=/path/to/imagenet64_ctm_checkpoint.pt
export IM64_REF=/path/to/VIRTUAL_imagenet64_labeled.npz
export IM64_OUT=/path/to/output/imagenet64
```

Optional controls:

```bash
export STEPS="1 2 4 8"
export TOTAL_SAMPLES=50000
export CIFAR_BATCH=1000
export IM64_BATCH=250
export RUN_CIFAR=1
export RUN_IM64=1
export RUN_EVAL=1
```

## Run

```bash
conda activate consistency
bash docs/baseline/reproduce_ctm_baselines.sh
```

The script launches one independent `mpiexec -n 1` process per GPU. This is
intentional: the upstream sampling scripts only save samples from rank 0, so a
single `mpiexec -n 2` job would waste the second GPU instead of writing twice as
many samples.

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
