# MAP Branch Technical Report

## Migration summary

What was migrated from CTM conceptually:
- explicit map learning objective
- model interface centered on `(x_t, t, s) -> x_s`
- iterative map rollout for few-step generation

What was intentionally not migrated:
- sigma-space training semantics
- target-model stack
- DSM branch
- GAN branch
- legacy distributed and evaluation system

## Resulting branch design

The new branch turns `dgfm` into a dual-mode framework:

- velocity branch:
  - learns `v_theta(x_t, t)`
  - samples by ODE integration

- explicit map branch:
  - learns `M_theta(x_t, t, s) -> x_s`
  - samples by iterative map rollout

Both branches share:
- config system
- dataloaders
- run-root layout
- checkpoint format
- evaluation and visualization entrypoints

## Current first-stage map algorithm

- keep current `dgfm` time semantics
- use online teacher rollouts by default
- retain a fixed CTM-like discrete scale grid
- sample `0 <= t < s <= 1` through discrete index sampling
- train on `(x_t, t, s, x_s_teacher)` pairs
- train direct map supervision with pixel MSE / Huber
- keep endpoint rollout loss as an auxiliary interface
- evaluate at `1/2/4/8/16` steps via iterative rollout

## Why this is the correct bridge

This design is the correct bridge between:

- pure FM velocity training,
- future teacher-switchable supervision,
- time-warped sampling,
- semigroup-defect regularization.

Reason:
- explicit map learning is the right substrate for map composition and teacher map targets,
- teacher trajectory targets move the branch substantially closer to CTM-style rollout supervision,
- but preserving `dgfm` time semantics avoids a premature rewrite into CTM’s full sigma-space stack.

## Next attachment points

- teacher switching:
  - `src/dgfm/targets/*`
- time-warp:
  - `src/dgfm/schedulers/*`
  - future `(t,s)` sampling policy
- semigroup defect:
  - future map-level regularizer comparing direct vs composed maps
