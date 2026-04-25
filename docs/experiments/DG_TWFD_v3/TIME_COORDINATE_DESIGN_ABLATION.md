# Time-Coordinate Design Ablation

This note defines the horizontal baseline comparison between the learned
DGTD timewarp and an `OptimalSteps-like` schedule baseline adapted from
`refs/optimalsteps`.

## Goal

Hold the checkpoint fixed and compare only the sampling time-coordinate design.
The result should fit one table with at least:

- FID at 1, 2, 4, and 8 steps
- fixed-seed sample grids at 1, 2, 4, and 8 steps

## Protocol

- Use the same checkpoint for both rows.
- Keep the same `num_fid_samples`, `fid_batch_size`, `sample_batch_size`, and
  `fixed_seed`.
- `Timewarp` row:
  - use the checkpoint's default learned time grid
- `OptimalSteps-like` row:
  - generate per-step schedule JSON under an independent output root
  - search is adapted from `refs/optimalsteps`
  - in this repo, the cost is rollout-state MSE under the explicit-map student
    interface, not a direct reproduction of the OSS paper numbers
- Step `1` uses the identity grid `[0.0, 1.0]` for the OSS-like row because
  a single-step schedule has no free coordinate design

## Isolation Rule

- Do not edit main training code for this ablation.
- Do not write into the active main-experiment run root.
- Write schedules, eval outputs, tables, and sample grids only into a separate
  ablation output root.

## Script

Use:

```bash
python scripts/run_time_coordinate_ablation.py \
  --config <config> \
  --checkpoint <best.pt> \
  --output-root <independent-output-root> \
  --steps 1 2 4 8
```

Main outputs:

- `comparison.md`
- `comparison.csv`
- `comparison.json`
- `acceptance.json`
- `metadata.json`
- `oss_schedules/oss_schedule_steps{K}.json`
- `eval_timewarp/steps{K}/fixed_seed_grid.png`
- `eval_optimalsteps_like/steps{K}/fixed_seed_grid.png`

## Acceptance

The ablation is acceptable when:

- both schedule policies finish for steps `1 2 4 8`
- both policies write FID summaries for all requested steps
- both policies write fixed-seed grids for all requested steps
- all OSS-like schedule JSON files exist and are monotone
- step `1` OSS-like schedule is exactly `[0.0, 1.0]`

## Interpretation Rule

This ablation isolates schedule design only. If the checkpoint is still
noise-like, the table is still valid as a baseline comparison, but it is not
evidence that either schedule policy is good in an absolute generative sense.

## Current Validation Run

The first isolated validation run uses the current best available completed
timewarp checkpoint:

- config:
  - `configs/experiment/dgtd_cifar10_v3_probe_fast_teacher.yaml`
- checkpoint:
  - `runs/dgtd_cifar10_v3_probe_fast_teacher_e405b/checkpoints/best.pt`
- output root:
  - `results/time_coordinate_ablation/e405b_optimalsteps_like_vs_timewarp_20260426`
- FID sample count:
  - `1024`
- status:
  - `acceptance.json -> pass`

Observed comparison table:

| Steps | Timewarp FID | OptimalSteps-like FID | Delta (OSS - Timewarp) |
| --- | ---: | ---: | ---: |
| 1 | 378.745 | 377.868 | -0.878 |
| 2 | 368.126 | 366.216 | -1.910 |
| 4 | 367.314 | 369.489 | +2.175 |
| 8 | 369.314 | 369.985 | +0.670 |

Interpretation:

- the harness works and both rows are directly comparable in one table
- the checkpoint remains noise-like, so the schedule comparison is only a
  relative baseline result
- on this checkpoint, `OptimalSteps-like` helps slightly at `1` and `2` steps,
  but the learned default timewarp is better at `4` and `8` steps
