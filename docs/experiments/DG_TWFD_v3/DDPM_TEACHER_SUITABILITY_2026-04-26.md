# DDPM Teacher Suitability Decision

Date: 2026-04-26

## Decision

The DDPM/discrete-teacher path is paused as the primary DG-TWFD route. It is
not mathematically impossible to distill from a discrete DDPM teacher, but it is
the wrong first target for the current explicit-map + time-warp architecture.

The active route is now EDM-first: continuous sigma-space teacher transitions,
online Heun teacher rollout, explicit student map, and time warp retained as a
learned schedule / sampling density component.

## Evidence Chain

DDPM probes produced stable optimization signals but did not produce usable
CIFAR-10 samples.

| Run | Setting | Best available metric | Result |
| --- | --- | --- | --- |
| `e405b` | fast DDPM teacher, learned warp | approx FID@512 `373.26` at 2 steps | failed `sample_not_noise_like` |
| `e406a` | fast DDPM teacher, no warp | approx FID@512 `386.56` at 2 steps | failed `sample_not_noise_like` |
| `e407a` | stronger endpoint anchor, longer DDPM probe | approx FID@1024 `427.45` at 8 steps | failed `sample_not_noise_like` |

The user's earlier 100-epoch DGFM/DDPM run is consistent with this: the model
only showed blurry contours after 16+ steps and stayed around FID 300. This is
not enough evidence to keep DDPM as the main distillation object.

Main technical reasons:

- DDPM teacher states are tied to a discrete timestep grid, while the explicit
  map needs arbitrary `(t, s)` transitions. Interpolation becomes a core
  supervision error source, not a minor implementation detail.
- Online DDPM continuation is expensive because a useful endpoint often needs
  many denoiser calls per batch. This makes diagnostic iteration and long
  training slow.
- Loss/defect reduction did not correlate with image quality in the DDPM
  probes, so the current objective can overfit trajectory artifacts while
  rollout samples remain noise-like.
- Time warp is hard to evaluate under a discrete teacher because a learned warp
  can select intervals where teacher interpolation is worst, masking whether
  the warp is helping the generative path.

## EDM-First Evidence

An isolated EDM-first track was added under `experiments/edm_first/`. It does
not modify the active DDPM/DGTD code path under `src/dgtd`.

The first EDM-first student uses the official EDM CIFAR-10 checkpoint as a
continuous teacher. The student is initialized from the same checkpoint and
wrapped as a continuous Euler map:

```text
x_s = x_t + (sigma_s - sigma_t) * (x_t - D_theta(x_t, sigma_t)) / sigma_t
```

Time warp is retained by sampling triplets in normalized clock coordinates
`u in [0, 1]`, mapping `u` to EDM sigma values, and updating a learned monotone
density from normalized defect.

| Run | Setting | Eval samples | Result |
| --- | --- | --- | --- |
| `e500c` | 20-step smoke, learned warp | 128 | clear non-noise CIFAR-like samples; approx FID improves with steps |
| `e501a` | 2000-step EDM-first learned-warp train | 1024 | approx FID `339.01/106.23/53.83/39.46` at `1/2/4/8` steps |
| `e501a-identity` | same checkpoint, identity clock | 1024 | approx FID `339.01/124.48/61.10/38.84` at `1/2/4/8` steps |
| `e501ref` | official EDM checkpoint, official sampler/protocol | 1024 | FID `679.611/473.607/115.246/33.0675` at `1/2/4/8` sampler steps |

The official EDM reference uses a different sampler and official FID reference,
so its numeric values are a reference point rather than a strict apples-to-apples
comparison with the torch-fidelity student metrics.

## Time Warp Verdict

Time warp should stay in the final architecture.

Current learned warp is already useful at low step counts:

- 2 steps: learned warp `106.23` vs identity `124.48`
- 4 steps: learned warp `53.83` vs identity `61.10`

At 8 steps it is slightly worse than identity:

- 8 steps: learned warp `39.46` vs identity `38.84`

This means the current warp is not yet globally optimal, but it is not a failed
component. The next warp work should make the warp step-budget-aware and
schedule-aware instead of only learning a passive defect density.

## Plan Change

The old DDPM result-first plan is closed at the diagnostic stage. Do not launch
DDPM `e402a` full training or DDPM `oss001` until there is a specific reason to
revisit DDPM.

The new primary route is:

1. Keep EDM-first code isolated until the objective is stable.
2. Add endpoint / residual / velocity target ablations.
3. Add a real no-warp training control, not only identity-clock eval.
4. Improve time warp with step-budget-aware schedule learning.
5. Only after CIFAR-10 is stable, migrate the validated EDM-first components
   back into the main `src/dgtd` architecture.

## Preserved Results

Primary evidence paths:

- training: `runs/edm_first_cifar10_warp_e501a`
- learned-warp eval: `eval/edm_first_cifar10_warp_e501a`
- identity-clock eval: `eval/edm_first_cifar10_identity_e501a`
- official EDM reference: `eval/edm_cifar10_public_eval_e501ref`

Critical checkpoint and summaries are backed up under:

```text
/temp/Zhengwei/DG-TWFD-backups/experiment_evidence/20260426_edm_first_e501a
```
