# Baseline Status

Last updated: 2026-04-26

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

`results/` is a symlink to `/cache/Zhengwei/DG-TWFD-runtime/results`, so these
CSV files must be regenerated from tracked code if `/cache` is cleared.

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

The main experiment `e504a_msdefect` is running on GPU 0, and its watcher
`watch_eval_checkpoints.sh` may launch intermittent evaluations. No external
baseline generation/evaluation should be started until that training and watcher
are stopped or explicitly moved off the GPU.

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
FID:

| step | fid |
|---:|---:|
| 1 | 679.611000 |
| 2 | 473.607000 |
| 4 | 115.246000 |
| 8 | 33.067500 |

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

The following files exist as header-only placeholders under `results/baselines`
and have no valid baseline rows yet:

```text
baseline_edm_imagenet64.csv
baseline_ctm_imagenet64.csv
baseline_ctm_cifar10.csv
baseline_cd_imagenet64.csv
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
EDM ImageNet64: official checkpoint/ref not cached; config now exists.
CTM CIFAR-10/ImageNet64: repos exist, official checkpoints not found locally.
CD ImageNet64: repo exists, checkpoint/eval output not found locally.
AYS: schedule integration pending; use official schedule values, no training.
OptimalSteps ImageNet64: search/eval pending on a usable checkpoint.
Entropic: repo cloned; schedule integration pending.
TCM: repo cloned; optional checkpoint/eval setup pending.
```

## Next Safe Execution Order

When GPU 0 is idle and the main watcher is not running, continue with:

```bash
source scripts/server/activate_a100_runtime.sh
source scripts/server/network_profiles.sh
dg_twfd_net_apply proxy
export PYTHONPATH=$PWD:$PWD/src:$PWD/refs/edm:${PYTHONPATH:-}

$DG_TWFD_A100_ENV/bin/python scripts/run_edm_cifar10_eval.py \
  --config configs/experiment/edm_imagenet64_public_eval.yaml \
  --sample-root runs/edm_imagenet64_public_eval_e501ref/samples \
  --eval-root eval/edm_imagenet64_public_eval_e501ref \
  --steps 1 2 4 8
```

Then regenerate:

```bash
python3 scripts/baselines/export_unified_baseline_csv.py --write-empty
```

Do not start CTM/CD/TCM runs until the required official checkpoint paths are
registered in this page or in a committed config.

## Asset Probe

The current machine asset probe is generated at:

```text
results/baselines/asset_probe.json
```

Generate it with:

```bash
python3 scripts/baselines/probe_baseline_assets.py
```
