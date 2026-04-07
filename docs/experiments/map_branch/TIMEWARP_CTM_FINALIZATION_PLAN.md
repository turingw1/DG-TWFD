# MAP Branch TimeWarp + CTM Finalization Plan

This document defines the design-level plan for the next major update on
`map_branch_ctm_explicit_map`.

The intent is to make the next code update structurally close to the final
branch shape, instead of continuing with short-lived heuristics.

## Design position

- do not chase strict CTM sigma-space reproduction
- do absorb the parts of `ctm-cifar10` that matter for target semantics
- treat time-warp as a first-class trainable module, not a one-off sampler trick
- keep `dgfm` as the host framework and preserve shared entrypoints
- separate three concepts clearly:
  - target construction
  - time reparameterization
  - preconditioning / stabilization

## Current gap

The branch has already moved from heuristic pair buckets to CTM-like discrete
pair sampling, but the remaining structure is still incomplete:

- training target grid is still built from a fixed linear grid
- sampling still defaults to uniform time allocation unless a time grid is
  passed manually
- target construction is still teacher trajectory supervision, not CTM-style
  estimate/target generation
- there is no learnable objective yet that pushes the time grid to reduce
  defect
- current preconditioning is useful but only CTM-inspired, not CTM-faithful

## Target end-state

The desired branch shape is:

1. a monotone `time_warp` module maps linear time into a warped time measure
2. both training and sampling use the same warped grid semantics
3. defect-derived statistics can optimize the warp parameters
4. target construction moves from plain path supervision toward CTM-style
   estimate / target semantics
5. preconditioning remains available as a stabilization device, but is
   documented separately from target semantics

## Task decomposition

## Task 1. Introduce a custom TimeWarp module

### Goal

- stop depending on a fixed linear or fixed CTM sigma grid
- introduce a monotone learnable time-warp module:
  - input: `t_linear in [0, 1]`
  - output: `t_warped in [0, 1]`

### Requirements

- monotone non-decreasing
- endpoint-preserving:
  - `g(0)=0`
  - `g(1)=1`
- differentiable w.r.t. warp parameters
- usable in:
  - training target building
  - few-step map rollout
  - diagnostics / exported time-grid inspection

### Proposed implementation

- reuse the piecewise-linear monotone CDF structure already present in
  [src/dg_twfd/models/timewarp.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dg_twfd/models/timewarp.py)
- expose a `dgfm`-side wrapper so map-branch code does not depend on
  `dg_twfd` internals
- support two modes:
  - identity / disabled
  - learnable monotone warp

### Expected file touchpoints

- create or extend:
  - `src/dgfm/schedulers/timewarp.py`
  - `src/dgfm/schedulers/__init__.py`
- config:
  - `configs/scheduler/timewarp_hook.yaml`

## Task 2. Replace linear discrete time with TimeWarp pipeline

### Goal

- replace direct use of `torch.linspace(0, 1, ...)` in target building
- make all discrete time grids flow through:
  - `t_linear -> time_warp -> t_warped`

### Requirements

- preserve current training compatibility
- preserve current `0 <= t < s <= 1` semantics at the API boundary
- allow time-warp to be disabled without changing outputs relative to the
  current implementation

### Proposed training path

1. build `t_linear_grid`
2. apply warp:
   - `t_warped_grid = g_phi(t_linear_grid)`
3. use `t_warped_grid` in target pair lookup
4. keep current tensor contract:
   - `TargetBatch(x_t, x_s_target, t, s, x_0, x_1)`

### Proposed sampling path

1. build a linear step schedule for the requested `step_count`
2. pass it through the same warp module
3. use the warped schedule as the rollout grid

### Expected file touchpoints

- `src/dgfm/targets/builder.py`
- `src/dgfm/samplers/map_sampler.py`
- `src/dgfm/evaluators/common.py`
- `scripts/run_sample.py`
- `scripts/run_multistep_panel.py`
- `src/dgfm/evaluators/runner.py`

## Task 3. Define a learnable defect objective for time-warp

### Goal

- use defect statistics to optimize `time_warp`
- make the time grid respond to actual composition difficulty instead of fixed
  hand-tuning

### Requirements

- defect signal must backpropagate to warp parameters
- objective may be per-step or aggregated
- objective must remain numerically stable under limited few-step rollout

### Proposed defect sources

- map composition defect:
  - compare direct map against composed map along warped times
- rollout endpoint defect:
  - compare warped-grid student endpoint against teacher or target endpoint
- optional per-interval defect statistics:
  - mean defect by interval
  - weighted defect by interval occupancy

### Proposed optimization shape

- stage A:
  - compute defect diagnostics only
- stage B:
  - optimize warp parameters with frozen map
- stage C:
  - alternate map updates and warp updates

### Important note

This task is the first place where time-warp becomes learnable in the training
loop. Tasks 1 and 2 only establish the pipeline.

### Current implementation status

- phase A is complete:
  - runtime warped grid construction is shared by target building, endpoint
    rollout, evaluation, and qualitative sampling
- phase B is now minimally implemented:
  - `MapTrainer` can instantiate a learnable monotone warp module
  - the warp module has its own optimizer
  - a defect-driven auxiliary update can optimize the warp with frozen map
    parameters
  - checkpoints now save and restore the learned warp state
- current defect objective is still a first-stage surrogate:
  - direct-vs-composed map defect over warped adjacent intervals
  - plus interval-balance regularization to avoid pathological collapse
- this is enough to validate:
  - whether `time_grid` changes during training
  - whether defect decreases under warp updates
- this is not yet the final CTM-style target-construction objective

## Task 4. Use the warped time grid during sampling

### Goal

- use `t_warped` for rollout step placement
- improve quality-per-step allocation

### Requirements

- the same warp semantics must be visible in both train and sample
- evaluation reports must record the effective time grid
- quality/speed comparison must be done against the uniform-grid baseline

### Current implementation status

- sampling, fixed-grid evaluation, and multistep qualitative panels now load
  the checkpointed warp state when available
- evaluation metrics now export the effective `time_grid` used for each
  `step_count`
- this makes it possible to compare:
  - uniform baseline
  - learned warp from a trained checkpoint

### Evaluation plan

- compare:
  - `uniform`
  - `warped`
- step counts:
  - `1 / 2 / 4 / 8 / 16`
- record:
  - FID
  - samples/sec
  - NFE
  - fixed-seed qualitative grids
  - exported `time_grid`

## Task 5. Upgrade target construction toward CTM-style semantics

### Goal

- move beyond simple teacher trajectory regression
- align more closely with `ctm-cifar10` target semantics

### CTM reference points

Use the following parts of
[ctm-cifar10/cm/karras_diffusion.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/ctm-cifar10/cm/karras_diffusion.py)
as semantic references:

- `t_dt` intermediate-step meaning
- `heun_solver(...)`
- `get_ctm_target(...)`
- `get_ctm_estimate(...)`

### Required semantic changes

- target generation must allow combinations of:
  - teacher
  - target model
  - current model
- stop-grad boundaries must be explicit
- target building must no longer be equivalent to plain trajectory regression

### Proposed interface direction

Keep `MapTrainer` stable and push complexity into target construction:

- current:
  - `build_from_batch(...) -> TargetBatch`
- next:
  - `build_from_batch(...) -> TargetBatch + optional diagnostics`

Potential internal concepts:

- `t`
- `t_dt`
- `s`
- `estimate_source`
- `target_source`
- `stop_grad_policy`

### Staging rule

- first abstract the interface and semantics
- then refine internals toward a more CTM-like implementation
- do not attempt a full CTM stack migration in one patch

## Task 6. Keep preconditioning explicitly CTM-inspired

### Goal

- prevent conceptual collapse between target semantics and preconditioning

### Required documentation rule

Current implementation in
[src/dgfm/models/map.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/models/map.py)
must continue to be described as:

- `CTM-inspired stabilization`

and not as:

- strict CTM sigma-space preconditioning reproduction

### Engineering rule

- keep the current implementation usable
- do not block task 5 on full preconditioning faithfulness
- keep preconditioning and target-construction evolution as separate axes

## Planned implementation order

## Phase 1. Pipe-cleaning

- task 1
- task 2
- acceptance:
  - training and sampling both accept a warped grid
  - disabled warp reproduces current behavior

## Phase 2. Diagnostics

- task 3 in diagnostics-only form
- task 4 with reporting
- acceptance:
  - defect and time-grid diagnostics appear in logs or eval artifacts

## Phase 3. CTM-style targets

- task 5
- acceptance:
  - target builder no longer acts as pure teacher trajectory regression

## Phase 4. Stabilization cleanup

- task 6
- acceptance:
  - docs and configs clearly separate target semantics from preconditioning

## Code mapping

### Training-time grid and target construction

- [src/dgfm/targets/builder.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/targets/builder.py)
- [src/dgfm/targets/pair_sampling.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/targets/pair_sampling.py)
- [src/dgfm/trainers/map.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/trainers/map.py)

### Sampling-time grid

- [src/dgfm/samplers/map_sampler.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/samplers/map_sampler.py)
- [src/dgfm/evaluators/common.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/evaluators/common.py)
- [src/dgfm/evaluators/runner.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/evaluators/runner.py)

### Time-warp module

- [src/dgfm/schedulers/timewarp.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/schedulers/timewarp.py)
- [configs/scheduler/timewarp_hook.yaml](/home/gzwlinux/vscode/gitProject/DG-TWFD/configs/scheduler/timewarp_hook.yaml)

### CTM semantic references

- [ctm-cifar10/cm/resample.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/ctm-cifar10/cm/resample.py)
- [ctm-cifar10/cm/script_util.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/ctm-cifar10/cm/script_util.py)
- [ctm-cifar10/cm/karras_diffusion.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/ctm-cifar10/cm/karras_diffusion.py)

## Acceptance criteria for the final update

- the branch has one shared time-warp semantics across train and sample
- the branch can log or export warped grids for every evaluated step count
- the branch can optimize warp parameters from defect-related signals
- target construction semantics are clearly no longer just path regression
- docs explicitly preserve the `CTM-inspired stabilization` wording for
  preconditioning
