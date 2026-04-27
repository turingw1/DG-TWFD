# Baseline Status

Last updated: 2026-04-28 Asia/Shanghai

Active baseline run:

```text
none
```

Current baseline budget:

```text
num_fid_samples: 5000
steps: 1 2 4 8
policy: run valid official baselines to completion, keep CSV reports in the stable per-project /temp tree
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
`results/baselines` is redirected to the stable project-isolated path:

```text
/temp/Zhengwei/projects/DG-TWFD/critical/analysis/baselines
```

Runner summary reports are mirrored under:

```text
/temp/Zhengwei/projects/DG-TWFD/critical/analysis/baselines/reports
```

TCM checkpoint backups are kept under:

```text
/temp/Zhengwei/projects/DG-TWFD/critical/ckpt/baselines/tcm
```

Regenerate empty placeholders and merged exports from known reports:

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

As of 2026-04-28, the main DG-TWFD training process is active. Baseline jobs may
use GPU 0 up to the agreed 75 GiB ceiling, but no baseline job is currently
running. Baseline work must not modify main experiment code or checkpoints.

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

AYS currently has no local runnable integration for the DG-TWFD CIFAR-10 or
ImageNet64 EDM/EDM2 setup. Do not fabricate AYS rows without a verified schedule
source and runner.

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

Current numbers are 1024-sample smoke/reference measurements, not final 5k or
50k FID:

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

OpenAI consistency-model baselines require the local compatibility patch
recorded at:

```text
patches/consistency_models_unet_baseline_compat.patch
```

### CTM Official ImageNet64

Output:

```text
results/baselines/baseline_ctm_imagenet64.csv
```

Source report:

```text
eval/ctm_imagenet64_5k/reports/summary.json
```

Current 5k FID:

| step | fid |
|---:|---:|
| 1 | 8.824660 |
| 2 | 8.677950 |
| 4 | 9.262070 |
| 8 | 10.080900 |

### CTM Official CIFAR-10

Output:

```text
results/baselines/baseline_ctm_cifar10.csv
```

Source report:

```text
eval/ctm_cifar10_5k/reports/summary.json
```

Current conditional 5k FID:

| step | fid |
|---:|---:|
| 1 | 6.444080 |
| 2 | 6.249970 |
| 4 | 6.450460 |
| 8 | 6.848800 |

### TCM Official CIFAR-10

Output:

```text
results/baselines/baseline_tcm_cifar10.csv
```

Runner:

```text
scripts/baselines/run_tcm_eval.py
```

Current 5k FID:

| step | fid |
|---:|---:|
| 1 | 7.168140 |
| 2 | 6.764130 |
| 4 | 6.904860 |
| 8 | 7.387160 |

Notes:

```text
step 1 uses the official 1-step path.
step 2 uses the official README mid_t=0.821 setting.
steps 4/8 use a geometric extension from mid_t=0.821 to sigma_min=0.002.
```

### TCM Official ImageNet64

Output:

```text
results/baselines/baseline_tcm_imagenet64.csv
```

Runner:

```text
scripts/baselines/run_tcm_eval.py
```

Current 5k FID:

| step | fid |
|---:|---:|
| 1 | 10.960100 |
| 2 | 9.951880 |
| 4 | 10.508700 |
| 8 | 20.957300 |

Notes:

```text
step 1 uses the official 1-step path.
step 2 uses the official README mid_t=0.821 setting.
steps 4/8 use a geometric extension from mid_t=0.821 to sigma_min=0.002.
```

### Entropic Schedule CIFAR-10

Output:

```text
results/baselines/schedule_entropic_cifar10.csv
```

Runner:

```text
scripts/baselines/run_entropic_schedule_eval.py
```

Current official schedule 5k FID with SDDIM:

| step | fid |
|---:|---:|
| 1 | 387.689000 |
| 2 | 384.919000 |
| 4 | 114.771000 |
| 8 | 56.378900 |

### Entropic Schedule ImageNet64

Output:

```text
results/baselines/schedule_entropic_imagenet64.csv
```

Runner:

```text
scripts/baselines/run_entropic_schedule_eval.py
```

Current official schedule 5k FID with SDDIM:

| step | fid |
|---:|---:|
| 1 | 288.205000 |
| 2 | 282.971000 |
| 4 | 80.052900 |
| 8 | 39.884100 |

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
useful for infrastructure validation only and must not be used as a paper table
baseline without replacement.

| step | fid |
|---:|---:|
| 1 | 377.867701 |
| 2 | 366.215745 |
| 4 | 369.488902 |
| 8 | 369.984571 |

## Pending or Blocked Outputs

The following files still exist as header-only placeholders under
`results/baselines`:

```text
schedule_ays_cifar10.csv
schedule_ays_imagenet64.csv
schedule_optimalsteps_imagenet64.csv
```

Known blockers:

```text
AYS CIFAR-10/ImageNet64: no verified local runner or official schedule mapping for this EDM/EDM2 setup.
OptimalSteps ImageNet64: refs/optimalsteps is present, but no validated runner maps it to the official EDM2 ImageNet64 checkpoint/evaluation flow.
```

These rows should stay empty until the method-specific schedule source and
sampling path are auditable. The current completed set covers all baseline
families that are locally runnable without fabricating method behavior.

## Asset Probe

The current machine asset probe is generated at:

```text
results/baselines/asset_probe.json
```

Generate it with:

```bash
python3 scripts/baselines/probe_baseline_assets.py
```
