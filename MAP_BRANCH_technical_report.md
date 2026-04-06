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
- sample `0 <= t < s <= 1`
- build `x_t` and `x_s_target` from the current path analytically
- train direct map supervision with pixel MSE / Huber
- evaluate at `1/2/4/8/16` steps via iterative rollout

## Why this is the correct bridge

This design is the correct bridge between:

- pure FM velocity training,
- future teacher-switchable supervision,
- time-warped sampling,
- semigroup-defect regularization.

Reason:
- explicit map learning is the right substrate for map composition and teacher map targets,
- but preserving `dgfm` time semantics avoids a premature rewrite into CTM’s full sigma-space stack.

## Next attachment points

- teacher switching:
  - `src/dgfm/targets/*`
- time-warp:
  - `src/dgfm/schedulers/*`
  - future `(t,s)` sampling policy
- semigroup defect:
  - future map-level regularizer comparing direct vs composed maps
