# EDM-First Continuous Map Distillation

This is an isolated experiment track for testing whether DG-TWFD should move
away from DDPM discrete trajectories and use an EDM-style continuous teacher.
It intentionally does not modify the active `src/dgtd` DDPM/DGTD mainline.

## Current Hypothesis

DDPM trajectory distillation is a poor fit for the current explicit-map line
because the teacher path is discrete and expensive:

- retained DDPM trajectory points require interpolation for arbitrary map
  times, so the student can learn interpolation artifacts instead of the
  teacher flow;
- online DDPM continuation is slow because every batch needs many denoiser
  calls;
- completed DDPM probes reached stable losses but samples stayed noise-like.

The first EDM-first test replaces the discrete DDPM path with a continuous
sigma-space teacher transition:

```text
M_theta(sigma_t, sigma_s, x_t)
```

The teacher is the official EDM CIFAR-10 checkpoint. The student is initialized
from the same checkpoint and wrapped as a continuous Euler map, so the first
large run tests the schedule/time-warp and transition-learning path rather than
capacity-from-scratch.

## CTM Details To Borrow

The implementation mirrors the CTM choices that are useful here:

- train in EDM sigma coordinates, not DDPM timestep indices;
- use EDM preconditioning through the official EDM network;
- use an outer Euler map `x_s = (sigma_s / sigma_t) x_t + (1 - sigma_s / sigma_t) D_theta(x_t, sigma_t)`;
- keep a consistency/defect loss in addition to teacher matching;
- treat time design as a first-class variable.

This track keeps time warp in the architecture by sampling triplets in a
normalized clock `u in [0, 1]`, mapping `u` to EDM sigma values, and updating a
learned density warp from observed normalized defect.

## Commands

Asset check:

```bash
bash experiments/edm_first/scripts/check_edm_assets.sh
```

Smoke:

```bash
bash experiments/edm_first/scripts/launch_all.sh \
  experiments/edm_first/configs/cifar10_edm_map_warp_smoke.yaml \
  edm_first_cifar10_smoke
```

Long first result:

```bash
bash experiments/edm_first/scripts/launch_all.sh \
  experiments/edm_first/configs/cifar10_edm_map_warp_8h.yaml \
  edm_first_cifar10_warp_e501a
```

Extended continuation:

```bash
bash experiments/edm_first/scripts/launch_train.sh \
  experiments/edm_first/configs/cifar10_edm_map_warp_8h_b128.yaml \
  edm_first_cifar10_warp_e502a \
  runs/edm_first_cifar10_warp_e501a/checkpoints/best.pt
```

Outputs:

- training run: `runs/<tag>`
- evaluation: `eval/<tag>`
- config copy and train log: `runs/<tag>/logs`
- checkpoint: `runs/<tag>/checkpoints/best.pt`
- FID/sample summary: `eval/<tag>/reports/summary.csv`

## 2026-04-26 First Results

`edm_first_cifar10_warp_e501a` completed 2000 training steps in 1054.8 seconds
on one A100. The run saw 128k samples and produced a usable CIFAR-like rollout.

Learned-warp eval, 1024 samples:

| Steps | Approx FID | Clock grid |
| --- | ---: | --- |
| 1 | 339.01 | `[0.0, 1.0]` |
| 2 | 106.23 | `[0.0, 0.6852, 1.0]` |
| 4 | 53.83 | `[0.0, 0.5330, 0.6852, 0.8058, 1.0]` |
| 8 | 39.46 | `[0.0, 0.4214, 0.5330, 0.6153, 0.6852, 0.7468, 0.8058, 0.8691, 1.0]` |

Identity-clock eval on the same checkpoint:

| Steps | Approx FID |
| --- | ---: |
| 1 | 339.01 |
| 2 | 124.48 |
| 4 | 61.10 |
| 8 | 38.84 |

Current verdict: learned time warp helps low-step schedules, but the 8-step
schedule needs a step-budget-aware warp objective.

`edm_first_cifar10_warp_e502a` resumed from the `e501a` best checkpoint with
batch size 128. GPU memory use was about 64.7GB on A100 80GB. It was stopped
after the step-5000 checkpoint to evaluate before spending the full 8-hour
budget.

Learned-warp eval, 2048 samples:

| Steps | Approx FID |
| --- | ---: |
| 1 | 337.45 |
| 2 | 151.78 |
| 4 | 56.76 |
| 8 | 35.67 |

Identity-clock eval on the same checkpoint, 2048 samples:

| Steps | Approx FID |
| --- | ---: |
| 1 | 337.45 |
| 2 | 140.76 |
| 4 | 54.24 |
| 8 | 26.89 |

Updated verdict: the continuous EDM map is viable, but the current learned warp
is not robust. Keep time warp in the architecture, but replace the passive
defect-density schedule with a step-budget-aware schedule objective.
