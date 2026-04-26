# EDM-First Supervision Log

Last updated: 2026-04-26

## Active Run

```text
run tag: edm_first_cifar10_onestep_msdefect_e504a
tmux: e504a_msdefect
watcher: e504a_eval_watch
checkpoint interval: 250 steps
watch eval steps: 1, 2, 4, 8, 16
FID sample count: 2048 for watcher diagnostics
```

## Current FID Baseline

The first e504a diagnostic checkpoint is:

```text
runs/edm_first_cifar10_onestep_msdefect_e504a/checkpoints/step250.pt
```

Current 2048-sample FID baseline from:

```text
eval/edm_first_cifar10_onestep_msdefect_e504a_step250_steps16/reports/summary.csv
```

| steps | FID |
|---:|---:|
| 1 | 177.890188 |
| 2 | 46.285510 |
| 4 | 49.293783 |
| 8 | 70.910603 |
| 16 | 86.566645 |

The active one-step decision threshold is a 50% FID reduction from the current
1-step baseline:

```text
1-step FID <= 88.945094
```

Secondary multi-step thresholds for a full-train decision:

| steps | 50% FID target |
|---:|---:|
| 2 | 23.142755 |
| 4 | 24.646892 |
| 8 | 35.455302 |
| 16 | 43.283323 |

Interpretation rule: do not judge full-train value from train loss alone. Wait
until a diagnostic checkpoint reaches the primary one-step threshold, or until a
clear multi-step regression/improvement pattern appears across at least two
consecutive watcher evaluations.

The active watcher is configured to compare future evaluations against this
baseline and write:

```text
eval/<tag>/reports/threshold_verdict.json
eval/<tag>/reports/threshold_verdict.csv
```

The full-train analysis trigger is:

```text
primary_target_met: true
```

## Timewarp Tracking Rule

e504a is intentionally an identity-clock run:

```text
timewarp.enabled: false
```

Therefore its timewarp effect is `N/A` and it serves as the no-warp baseline for
the current one-step endpoint objective.

From this point onward, watcher evaluations use steps `1 2 4 8 16`. When a
config enables `timewarp.enabled: true`, the watcher also runs an identity-clock
comparison and writes:

```text
eval/<tag>/reports/timewarp_comparison.csv
eval/<tag>/reports/timewarp_comparison.json
```

The key column is:

```text
fid_delta_auto_minus_identity
```

Negative values mean the learned/default timewarp is better than identity for
that step budget. Positive values mean identity is better or equal and the warp
objective should not be trusted for that budget without modification.
