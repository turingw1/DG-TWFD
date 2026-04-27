# Baseline Status

Last updated: 2026-04-28

Active baseline run:

```text
launcher: scripts/baselines/run_ctm_imagenet64_eval.py
scope: CTM official ImageNet64 5k FID at steps 1/2/4/8
checkpoint: /cache/Zhengwei/DG-TWFD-runtime/checkpoints/baselines/ctm/ctm_imagenet64_ema_0.999.pt
sample root: runs/ctm_imagenet64_5k/samples
eval root: eval/ctm_imagenet64_5k
```

Current baseline budget:

```text
num_fid_samples: 5000
steps: 1 2 4 8
policy: run remaining official baselines to completion, keep CSV reports in /temp via results/baselines symlink
```

## Scope

This page tracks external and schedule baselines requested for the DG-TWFD v3
paper tables. The canonical merged-output schema is:

```text
dataset,method,step,fid,is,recall,checkpoint,eval_script,notes
```

Unified CSV exports live under:

```text
results/baselines/
```

`results/` is a symlink to `/cache/Zhengwei/DG-TWFD-runtime/results`, and
`results/baselines` is redirected to `/temp/Zhengwei/DG-TWFD-backups/experiment_evidence/baselines_20260426`.
The unified CSV reports therefore survive `/cache` loss as long as `/temp`
remains intact.

Current small CSV exports are also backed up at:

```text
/temp/Zhengwei/DG-TWFD-backups/experiment_evidence/baselines_20260426/
```

Regenerate from existing reports:

```bash
python3 scripts/baselines/export_unified_baseline_csv.py --write-empty
```

Long-running EDM official baseline launcher:

```bash
bash scripts/baselines/run_external_baselines.sh
```

The comparison role of each baseline is documented in:

```text
docs/experiments/DG_TWFD_v3/BASELINE_COMPARISON_GUIDE.md
```

## Current GPU Constraint

As of 2026-04-28 01:40 Asia/Shanghai, no main training process is active and
GPU memory was idle before starting CTM ImageNet64. Baseline jobs may use GPU 0
up to the agreed 75 GiB ceiling, while avoiding code changes to the main
experiment.

## Baseline Roots

Reference repositories currently present:

```text
refs/edm
refs/ctm
refs/ctm-cifar10
refs/consistency_models
refs/optimalsteps
refs/entropic_time_schedulers
refs/tcm
```

These reference directories are intentionally ignored by git under `refs/*`.

AYS currently has no local repo. The official project page exposes a paper,
Colab, and quickstart schedules rather than a checked-out project directory:

```text
https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/
https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html
```

Archived CTM reproduction notes remain at:

```text
docs/archive/low_signal_2026-04-25/baseline/
```

## Existing Results

### EDM Official CIFAR-10

Output:

```text
results/baselines/baseline_edm_cifar10.csv
```

Source report:

```text
eval/edm_cifar10_public_eval_e501ref/reports/summary.json
```

Checkpoint:

```text
https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
```

FID reference:

```text
https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
```

Current numbers are 1024-sample smoke/reference measurements, not final 50k
or 5k FID:

| step | fid |
|---:|---:|
| 1 | 679.611000 |
| 2 | 473.607000 |
| 4 | 115.246000 |
| 8 | 33.067500 |

### EDM Official ImageNet64

Output:

```text
results/baselines/baseline_edm_imagenet64.csv
```

Current 5k FID:

| step | fid |
|---:|---:|
| 1 | 623.860000 |
| 2 | 438.527000 |
| 4 | 93.126600 |
| 8 | 11.650400 |

### OpenAI Consistency Models ImageNet64

Outputs:

```text
results/baselines/baseline_cd_imagenet64.csv
results/baselines/baseline_cd_imagenet64_5k.csv
results/baselines/baseline_cd_imagenet64_l2_5k.csv
results/baselines/baseline_ct_imagenet64_5k.csv
```

CD-LPIPS official 5k FID:

| step | fid |
|---:|---:|
| 1 | 13.025300 |
| 2 | 11.611200 |
| 4 | 11.559600 |
| 8 | 10.879900 |

CD-L2 official 5k FID:

| step | fid |
|---:|---:|
| 1 | 20.309900 |
| 2 | 14.257700 |
| 4 | 13.835400 |
| 8 | 12.411900 |

CT official 5k FID:

| step | fid |
|---:|---:|
| 1 | 19.141800 |
| 2 | 17.466500 |
| 4 | 18.837000 |
| 8 | 19.034100 |

### OptimalSteps-like CIFAR-10

Output:

```text
results/baselines/schedule_optimalsteps_cifar10.csv
```

Source report:

```text
results/time_coordinate_ablation/e405b_optimalsteps_like_vs_timewarp_20260426/comparison.csv
```

This is a preliminary OSS-like schedule adapted from `refs/optimalsteps` on the
old DDPM/DGTD checkpoint `e405b`, which failed the sample-quality gate. It is
useful for infrastructure validation only and must be replaced before paper
tables.

| step | fid |
|---:|---:|
| 1 | 377.867701 |
| 2 | 366.215745 |
| 4 | 369.488902 |
| 8 | 369.984571 |

## Pending Outputs

The following files still exist as header-only placeholders under
`results/baselines` and have no valid baseline rows yet:

```text
baseline_ctm_imagenet64.csv
baseline_ctm_cifar10.csv
schedule_ays_cifar10.csv
schedule_ays_imagenet64.csv
schedule_optimalsteps_imagenet64.csv
schedule_entropic_cifar10.csv
schedule_entropic_imagenet64.csv
baseline_tcm_cifar10.csv
baseline_tcm_imagenet64.csv
```

Known blockers:

```text
CTM ImageNet64: official checkpoint is cached; 5k run in progress.
CTM CIFAR-10: repo exists; official checkpoint path still needs validation.
AYS: schedule integration pending; use official schedule values, no training.
OptimalSteps ImageNet64: search/eval pending on a usable checkpoint.
Entropic: repo cloned; schedule integration pending.
TCM: repo cloned; optional checkpoint/eval setup pending.
```

## Next Safe Execution Order

When GPU 0 is idle and the main watcher is not running, continue CTM ImageNet64
with:

```bash
.conda_envs/dg_twfd_a100/bin/python scripts/baselines/run_ctm_imagenet64_eval.py \
  --checkpoint /cache/Zhengwei/DG-TWFD-runtime/checkpoints/baselines/ctm/ctm_imagenet64_ema_0.999.pt \
  --method CTM-official \
  --sample-root runs/ctm_imagenet64_5k/samples \
  --eval-root eval/ctm_imagenet64_5k \
  --csv-out results/baselines/baseline_ctm_imagenet64.csv \
  --steps 1 2 4 8 \
  --num-samples 5000 \
  --batch 250 \
  --fid-batch 512
```

Then regenerate:

```bash
python3 scripts/baselines/export_unified_baseline_csv.py --write-empty
```

OpenAI consistency-model baselines require the local compatibility patch
recorded at:

```text
patches/consistency_models_unet_baseline_compat.patch
```

## Asset Probe

The current machine asset probe is generated at:

```text
results/baselines/asset_probe.json
```

Generate it with:

```bash
python3 scripts/baselines/probe_baseline_assets.py
```
