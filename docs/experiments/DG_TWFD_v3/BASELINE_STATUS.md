# Baseline Status

Last updated: 2026-04-28 Asia/Shanghai

Active baseline run:

```text
CTM 50k revalidation is complete; keep existing 5k CSVs unchanged.
final CTM 50k audit root: /temp/Zhengwei/projects/DG-TWFD/critical/analysis/baselines_revalidated_20260428_final
final CTM 50k CSV: /temp/Zhengwei/projects/DG-TWFD/critical/analysis/baselines_revalidated_20260428_final/baseline_ctm_50k_final.csv
completed records: CIFAR-10 and ImageNet64, steps 1/2/4/8, 50000 samples per record
invalidated partial: original CTM ImageNet64 50k step 4 stopped at 192/200 npz shards and is excluded from final records
```

EDM checkpoint schedule/time-warp follow-up is complete:

```text
run root: eval/edm_schedule_warp_5k_20260428
summary CSV: eval/edm_schedule_warp_5k_20260428/reports/summary.csv
stable-style CSV export: results/baselines/edm_schedule_warp_5k_20260428/edm_schedule_warp_cifar10_5k_summary.csv
stable full archive: /temp/Zhengwei/projects/DG-TWFD/critical/analysis/edm_schedule_warp_5k_20260428.tar.gz
samples: eval/edm_schedule_warp_5k_20260428/samples/{strategy}/steps{1,2,4,8}/images
previews: eval/edm_schedule_warp_5k_20260428/previews/{strategy}_steps{1,2,4,8}.png
schedules: eval/edm_schedule_warp_5k_20260428/schedules/{strategy}/steps{1,2,4,8}.json
runner: scripts/baselines/run_edm_schedule_warp_eval.py
protocol: CIFAR-10, official EDM cond-VP checkpoint, EDM fid.py, 5000 samples, deterministic custom EDM/Heun sigma grid
```

EDM schedule/time-warp FID-5k:

| Method | Step 1 | Step 2 | Step 4 | Step 8 | Note |
|---|---:|---:|---:|---:|---|
| EDM + OptimalSteps-adapted schedule | 315.41200 | 235.05800 | 34.77990 | 9.90991 | DP over 64-step dense EDM teacher trajectory |
| EDM + Entropic schedule | 315.41200 | 467.56400 | 135.38000 | 26.78320 | Entropic uncond-VP time function transferred to cond-VP checkpoint |
| EDM + piecewise-linear time warp | 315.41200 | 280.95100 | 229.90400 | 182.19500 | Fixed proxy-density inverse-CDF warp; negative-control quality |
| EDM + spline time warp | 315.41200 | 281.01300 | 229.96700 | 182.28300 | Same proxy with monotone PCHIP inverse-CDF |

These rows are schedule-only baselines on the EDM checkpoint. They do not train
or use a DG-TWFD student map. `steps=1` is explicitly defined in the runner as a
single `sigma_max -> 0` EDM transition to avoid the official EDM `num_steps=1`
grid degeneracy.

CTM checkpoint schedule/time-warp follow-up is complete:

```text
runner: scripts/baselines/run_ctm_schedule_warp_eval.py
run root: runs/ctm_schedule_warp_5k_20260428/samples/{dataset}/{strategy}/steps{1,2,4,8}/images
eval root: eval/ctm_schedule_warp_5k_20260428/{dataset}
summary CSVs:
  eval/ctm_schedule_warp_5k_20260428/cifar10/reports/summary.csv
  eval/ctm_schedule_warp_5k_20260428/imagenet64/reports/summary.csv
stable-style CSV exports:
  results/baselines/ctm_schedule_warp_5k_20260428/ctm_schedule_warp_cifar10_5k_summary.csv
  results/baselines/ctm_schedule_warp_5k_20260428/ctm_schedule_warp_imagenet64_5k_summary.csv
previews: eval/ctm_schedule_warp_5k_20260428/{dataset}/previews/{strategy}_steps{1,2,4,8}.png
schedules: eval/ctm_schedule_warp_5k_20260428/{dataset}/schedules/{strategy}/steps{1,2,4,8}.json
sample count check: all 32 image directories contain 5000 PNG files
```

Protocol: the runner loads the CTM checkpoints directly and applies custom
sigma nodes by calling CTM exact transitions `G_theta(x_t, t, s)` for every
interval. These are CTM-specific schedule diagnostics, not EDM sampler runs.
The four strategies are:

```text
optimalsteps_ctm: dynamic programming on a dense CTM exact trajectory.
entropic_ctm: transfer the precomputed entropic time function to CTM sigma nodes.
piecewise_linear_ctm: inverse-CDF warp from a CTM self-consistency residual proxy.
spline_warp_ctm: same proxy with monotone PCHIP inverse-CDF interpolation.
```

CTM schedule/time-warp CIFAR-10 FID-5k:

| Method | Step 1 | Step 2 | Step 4 | Step 8 | Note |
|---|---:|---:|---:|---:|---|
| CTM + OptimalSteps-adapted schedule | 6.39022 | 6.33884 | 6.41577 | 6.62873 | DP against dense CTM exact chain |
| CTM + Entropic schedule | 6.39022 | 6.31943 | 6.66228 | 6.98590 | Entropic schedule transfer to CTM |
| CTM + piecewise-linear time warp | 6.39022 | 6.37918 | 6.37642 | 6.37383 | CTM residual-proxy inverse-CDF warp |
| CTM + spline time warp | 6.39022 | 6.37878 | 6.37888 | 6.37933 | Same proxy with monotone PCHIP inverse-CDF |

CTM schedule/time-warp ImageNet64 FID-5k:

| Method | Step 1 | Step 2 | Step 4 | Step 8 | Note |
|---|---:|---:|---:|---:|---|
| CTM + OptimalSteps-adapted schedule | 8.85793 | 8.93934 | 9.24441 | 9.82932 | DP against dense CTM exact chain |
| CTM + Entropic schedule | 8.85793 | 9.42372 | 9.80729 | 11.01760 | Entropic schedule transfer to CTM |
| CTM + piecewise-linear time warp | 8.85793 | 9.51683 | 8.93845 | 8.84899 | CTM residual-proxy inverse-CDF warp |
| CTM + spline time warp | 8.85793 | 9.46341 | 8.92669 | 8.84654 | Same proxy with monotone PCHIP inverse-CDF |

These CTM rows use a deterministic per-seed local generation protocol and EDM
FID references. They are useful for isolating the effect of CTM time-node
selection under the shared DG-TWFD 5k protocol, but they do not replace the
official CTM 50k audit rows above.

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

The Chinese paper-writing report is documented in:

```text
docs/experiments/DG_TWFD_v3/BASELINE_REPORT_CN.md
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

The current ImageNet64 recovery root is:

```text
/temp/Zhengwei/projects/DG-TWFD/critical/analysis/baselines_revalidated_20260428_recovery
```

The final merged CTM 50k audit root is:

```text
/temp/Zhengwei/projects/DG-TWFD/critical/analysis/baselines_revalidated_20260428_final
```

The corresponding cache-only sample and eval roots are:

```text
runs/baselines_revalidated_20260428/
eval/baselines_revalidated_20260428/
runs/baselines_revalidated_20260428_recovery/
eval/baselines_revalidated_20260428_recovery/
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

Final 50k audit records:

```text
CIFAR-10,  step 1: FID 1.743220
CIFAR-10,  step 2: FID 1.616910
CIFAR-10,  step 4: FID 1.830040
CIFAR-10,  step 8: FID 2.101430
ImageNet64, step 1: FID 2.379590
ImageNet64, step 2: FID 2.212310
ImageNet64, step 4: FID 2.893610
ImageNet64, step 8: FID 3.867740
```

Final audit artifact paths:

```text
/temp/Zhengwei/projects/DG-TWFD/critical/analysis/baselines_revalidated_20260428_final/baseline_ctm_50k_final.csv
/temp/Zhengwei/projects/DG-TWFD/critical/analysis/baselines_revalidated_20260428_final/reports/baseline_ctm_50k_final_summary.json
/temp/Zhengwei/projects/DG-TWFD/critical/analysis/baselines_revalidated_20260428_final/reports/baseline_ctm_50k_final_summary.csv
```

The final merged report uses the original 50k root for CIFAR-10 steps 1/2/4/8
and ImageNet64 steps 1/2. ImageNet64 steps 4/8 come from the recovery root,
where batch size was lowered to 100 only to fit alongside the current main
experiment. The sampling rule, checkpoint, step count, FID implementation, and
FID reference remain the CTM 50k audit standard.

### CTM Schedule/Time-Warp Follow-up

Fair-comparison role:

```text
diagnose whether CTM quality changes when only the CTM exact-transition sigma nodes are changed by external schedule/time-warp rules.
```

Current status:

```text
CIFAR-10 and ImageNet64 are complete at 5000 samples for steps 1/2/4/8.
The dedicated runner uses CTM's exact G_theta(x_t,t,s) transition for each custom interval.
All 32 sample directories contain 5000 PNG files.
These rows are FID-5k diagnostics and do not replace the CTM 50k official-code audit.
```

Artifact paths:

```text
scripts/baselines/run_ctm_schedule_warp_eval.py
eval/ctm_schedule_warp_5k_20260428/cifar10/reports/summary.csv
eval/ctm_schedule_warp_5k_20260428/imagenet64/reports/summary.csv
results/baselines/ctm_schedule_warp_5k_20260428/ctm_schedule_warp_cifar10_5k_summary.csv
results/baselines/ctm_schedule_warp_5k_20260428/ctm_schedule_warp_imagenet64_5k_summary.csv
```

Interpretation:

```text
optimalsteps_ctm and entropic_ctm test whether externally designed schedules transfer to CTM exact transitions.
piecewise_linear_ctm and spline_warp_ctm test fixed-form time-warp parameterizations using a CTM self-consistency residual proxy.
Unlike DG-TWFD, these baselines do not train a student map or optimize the model; they only change the evaluation-time sigma grid.
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
