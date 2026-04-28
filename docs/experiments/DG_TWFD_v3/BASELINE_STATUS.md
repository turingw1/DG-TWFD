# Baseline Status

Last updated: 2026-04-28 Asia/Shanghai

Active baseline run:

```text
CTM 50k revalidation is running under baselines_revalidated_20260428; keep existing 5k CSVs unchanged.
current phase: CTM CIFAR-10 50k, exact sampler, steps 1/2/4/8
stable root: /temp/Zhengwei/projects/DG-TWFD/critical/analysis/baselines_revalidated_20260428
log root: /temp/Zhengwei/projects/DG-TWFD/logs/baselines_revalidated_20260428
```

Current baseline budget:

```text
fast comparison num_fid_samples: 5000 unless explicitly marked otherwise
official-audit num_fid_samples: 50000 when a baseline's own reproduction notes require it
steps: 1 2 4 8
policy: keep the old fast table immutable; write stricter follow-up runs into a new /temp result folder
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

## Fair-Comparison Protocol

The primary DG-TWFD comparison is an internal fair-comparison protocol, not a
claim to reproduce every baseline paper table exactly. Fair comparison here
means:

```text
same dataset label and image resolution
same reported step counts: 1, 2, 4, 8
same FID implementation and reference statistics when possible: refs/edm/fid.py with EDM fid-refs
same fast FID budget when possible: 5000 generated samples
official public checkpoint and public sampling code when locally available
no method-specific hyperparameter search unless documented as an official sampling setting
old results are not overwritten; stricter follow-up runs use a new output root
```

This protocol is useful for comparing methods under the same practical
constraints as DG-TWFD evaluation. It should not be described as an official
paper reproduction when a baseline's own repo requires a different sample
count, reference statistic, checkpoint variant, or sampling option.

When a baseline's official notes require a stricter evaluation that is still
compatible with the fair-comparison setup, run it separately as an audit. The
current audit root is:

```text
/temp/Zhengwei/projects/DG-TWFD/critical/analysis/baselines_revalidated_20260428
```

The corresponding cache-only sample and eval roots are:

```text
runs/baselines_revalidated_20260428/
eval/baselines_revalidated_20260428/
```

These cache roots are intentionally not treated as durable evidence; the stable
CSV, summary reports, logs, and manifest must be copied to `/temp`.

## Baseline Audit Matrix

### EDM Official

Fair-comparison role:

```text
teacher/sampler anchor using official EDM sampler and EDM FID references.
```

Current status:

```text
CIFAR-10 row is only a 1024-sample smoke/reference run and must not be used as a final table row.
ImageNet64 row is a 5k fast-comparison run.
```

Follow-up requirement:

```text
If EDM is included in final quantitative tables, rerun the relevant dataset at the same sample budget used for DG-TWFD and label that budget explicitly.
```

### OpenAI Consistency Models CD/CT

Fair-comparison role:

```text
official checkpoint and official ImageNet64 sampling code, evaluated at the shared 1/2/4/8 step grid.
```

Current status:

```text
CD-LPIPS, CD-L2, and CT ImageNet64 are 5k fast-comparison rows.
The multistep timestamp grids are recorded in each CSV notes field.
These are fair internal comparisons, but not a full official 50k reproduction.
```

Follow-up requirement:

```text
No immediate rerun is required for the fast table. Run 50k only if the paper table needs official-reproduction-grade CD/CT numbers.
```

### CTM Official

Fair-comparison role:

```text
official CTM sampling code with exact sampler on the shared 1/2/4/8 step grid.
```

Current status:

```text
Existing CTM rows are 5k fast-comparison runs and are retained unchanged.
CIFAR-10 uses the downloaded official-folder checkpoint model043000.pt.
The downloaded CIFAR-10 folder does not contain the EMA checkpoint files referenced by the training log.
The CIFAR-10 log records sub-2 FID-50k for EMA variants, so the current 5k model043000.pt result must not be interpreted as the CTM paper-best reproduction.
ImageNet64 uses ctm_imagenet64_ema_0.999.pt, but the current internal FID uses EDM imagenet-64x64 reference stats, not the CTM repo's author-provided ImageNet64 stats.
```

Official-audit requirement:

```text
The CTM CIFAR-10 README requires >=50k samples for correct FID evaluation.
Run CTM 50k revalidation into baselines_revalidated_20260428 without overwriting the 5k rows.
Keep the result labeled as "current local checkpoint, 50k EDM-FID audit" unless the exact EMA paper checkpoint and author reference stats are available.
```

Current revalidation launcher:

```bash
bash scripts/baselines/run_ctm_50k_revalidation.sh
```

### TCM Official

Fair-comparison role:

```text
official TCM checkpoint and sampling code, evaluated with the shared FID implementation and step grid.
```

Current status:

```text
1-step uses the official 1-step path.
2-step uses the official README midpoint mid_t=0.821.
4-step and 8-step are clearly marked as a geometric extension from the official midpoint to sigma_min=0.002.
The 4/8-step rows are fair engineering comparisons at the shared step grid, but they are not official TCM schedule claims.
```

Follow-up requirement:

```text
No immediate rerun is required for the fast table. If TCM is used as a primary paper baseline, run a separate audit for the official step counts and keep 4/8 extensions labeled.
```

### Entropic Time Scheduler

Fair-comparison role:

```text
official precomputed Entropic schedules inserted into the local SDDIM evaluation path.
```

Current status:

```text
CIFAR-10 and ImageNet64 are 5k fast-comparison rows.
The results are valid for the implemented SDDIM schedule baseline, but the large FID values should be described as this implementation/solver configuration, not as a universal limit of the Entropic method.
```

Follow-up requirement:

```text
No immediate rerun is required unless a different official solver configuration is identified and mapped cleanly to the same datasets.
```

### OptimalSteps and AYS

Fair-comparison role:

```text
schedule baselines only after a verified schedule source and runnable local mapping exist.
```

Current status:

```text
OptimalSteps-like CIFAR-10 is infrastructure validation on an old failed e405b checkpoint, not a paper-ready baseline.
AYS CIFAR-10, AYS ImageNet64, and OptimalSteps ImageNet64 remain header-only because no verified runner/schedule mapping is available.
```

Follow-up requirement:

```text
Do not fabricate rows. Add rows only after the schedule source, checkpoint mapping, and evaluation path are auditable.
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
