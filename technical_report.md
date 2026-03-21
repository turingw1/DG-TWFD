# Technical Report

## Final Recommended Architecture
Use `flow_matching` as the core abstraction source for:
- probability paths,
- schedulers,
- solver interfaces,
- model wrappers.

Build the new framework around a separate research-ready package with explicit modules for datasets, trainers, evaluators, teachers, and experiment configs.

## Why `flow_matching` Is the Core Abstraction Source
It already exposes the abstractions that matter for Phase 1:
- `ProbPath.sample(...)` for constructing `x_t` and supervision targets,
- scheduler objects for time parameterization,
- solver objects for inference over custom time grids,
- schedule transformation logic that naturally supports future time-warp work.

These abstractions are cleaner and more future-proof than the current DG-TWFD monolithic trainer stack.

## Why Example Training Scripts Must Not Remain the Long-Term Base
The vendored `flow_matching` examples are useful references, but they are not a clean framework boundary.

Problems:
- training logic is embedded in modality-specific example folders,
- evaluation is not a first-class subsystem,
- image and text examples use different config and launcher philosophies,
- checkpointing/logging/reporting are not unified.

Therefore:
- reuse the core library,
- rewrite the training/evaluation surface.

## How the New Framework Supports Few-Step Evaluation and Future Time-Warp
Few-step evaluation becomes a first-class evaluator protocol driven by solver time grids over `1/2/4/8/16` steps. Time-warp is reserved as a scheduler-level hook so that baseline runs use identity scheduling, while future experiments can enable learned monotonic reparameterization without changing trainer or evaluator APIs.

## Why Semigroup Defect Is Deferred
Semigroup defect requires a clean explicit `M(t,s,x)` map abstraction. The current `flow_matching` core learns a velocity field `u_t(x)` and integrates it with an ODE solver, which is not yet the right boundary for a clean defect module. Forcing defect into Phase 1 would couple the new architecture to an unresolved map-vs-velocity design. The correct move is:
- Phase 1: baseline FM + evaluation + time-warp-ready scheduler hooks,
- Phase 2: introduce explicit flow-map abstractions and then add defect cleanly.

## Decision
Use `flow_matching` partially and refactor around it. Do not keep the existing example scripts or the current DG-TWFD trainer as the long-term base. The new architecture should be baseline-first, evaluator-first, and teacher-pluggable from day one.
