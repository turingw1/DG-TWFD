# MAP Branch Technical Report

## Migration summary

What was migrated from CTM conceptually:
- explicit map learning objective
- model interface centered on `(x_t, t, s) -> x_s`
- iterative map rollout for few-step generation

What was intentionally not migrated:
- sigma-space training semantics
- full CTM target-model stack
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

## Current map algorithm shape

- keep current `dgfm` time semantics
- use online teacher rollouts by default
- retain a fixed CTM-like discrete scale grid
- sample CTM-style triplets:
  - `0 <= t < t_dt <= s <= 1`
- build teacher bridge states:
  - `x_t`
  - `x_t_dt`
  - `x_s_teacher`
- train with explicit estimate / target semantics:
  - `estimate = M_theta(x_t, t, s)`
  - `target = stop_grad(M_target(x_t_dt, t_dt, s))` by default
- use EMA shadow as the default `target_source`
- use EMA rollout as the default `bridge_source` when available
- keep teacher `x_s_teacher` available as a fallback anchor source
- keep endpoint rollout loss as an auxiliary interface
- evaluate at `1/2/4/8/16` steps via iterative rollout

This should now be described as:

- CTM-style target-construction interface

not yet as:

- CTM-faithful target generation

because the intermediate bridge state is now produced by a map-branch rollout
bridge rather than a CTM-faithful Heun solver.

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
  - current learned monotone warp with shared train/sample/eval usage
- semigroup defect:
  - current first-stage auxiliary objective comparing direct vs composed maps
    across warped intervals
- future CTM-style defect signal should be aligned with target construction,
  not left as an isolated regularizer
- target construction:
  - current implementation now separates:
    - estimate source
    - target source
    - stop-grad policy
    - bridge source
  - future work should replace the current map rollout bridge with a more
    faithful solver-derived bridge state

## Current time-warp status

- the branch now supports a learnable monotone time-warp module
- target building and few-step rollout can consume the same warped grid
- the trainer can update warp parameters from a defect-driven auxiliary loss
- checkpoints persist the learned warp state
- evaluation exports the actual warped `time_grid` used at each step count

This should be treated as:

- a validated infrastructure layer for adaptive time allocation

not as:

- a completed CTM-style target-construction migration
