# MAP Branch Master Plan

## Executive design decision

- Migrate **CTM’s explicit flow-map task form**, not the full CTM repository.
- Keep `dgfm` as the host framework.
- Upgrade `dgfm` from a single-mode velocity-field framework into a **dual-mode framework**:
  - `velocity_fm`
  - `explicit_map`
- Preserve:
  - config loading
  - run-root / checkpoint / sample / eval layout
  - FID / NFE / visualization tooling
- Reject full CTM migration because:
  - CTM code is tightly coupled to sigma-space, target-model orchestration, DSM, GAN, and legacy distributed infrastructure.
  - those parts are not required for the first map-branch bridge.

## New architecture

### Shared infrastructure
- `scripts/run_train.py`
- `scripts/run_eval.py`
- `scripts/run_sample.py`
- `scripts/run_multistep_panel.py`
- `src/dgfm/config/*`
- `src/dgfm/datasets/*`
- `src/dgfm/evaluators/*`
- `src/dgfm/models/ema.py`

### Velocity branch
- model: `build_velocity_model()`
- trainer: `BaselineTrainer`
- sampler: ODE integration
- evaluator: shared `EvaluationRunner`

### Explicit map branch
- model: `build_map_model()`
- trainer: `MapTrainer`
- target builder: `build_target_builder()`
- sampler: `rollout_with_map()`
- evaluator: `MapEvaluationRunner`

### Future hooks
- teacher switching: `src/dgfm/targets/*`
- time-warp: `src/dgfm/schedulers/*`
- semigroup defect: future map-level regularizer module

## Required code changes

### Create
- `src/dgfm/models/map.py`
- `src/dgfm/trainers/map.py`
- `src/dgfm/targets/__init__.py`
- `src/dgfm/targets/builder.py`
- `src/dgfm/samplers/__init__.py`
- `src/dgfm/samplers/map_sampler.py`
- `src/dgfm/evaluators/map_eval.py`
- `configs/model/map_unet.yaml`
- `configs/target/teacher_trajectory.yaml`
- `configs/eval/map_branch.yaml`
- `configs/experiment/fm_cifar10_map_branch.yaml`
- `src/dgfm/teachers/diffusers_ddpm.py`
- `src/dgfm/teachers/factory.py`
- `src/dgfm/datasets/trajectory.py`
- `scripts/prepare_teacher_trajectories.py`

### Modify
- `scripts/run_train.py`
- `scripts/run_eval.py`
- `scripts/run_sample.py`
- `scripts/run_multistep_panel.py`
- `scripts/experiments/activate_fm_cifar10.sh`
- `src/dgfm/trainers/__init__.py`
- `src/dgfm/models/__init__.py`
- `src/dgfm/evaluators/__init__.py`
- `src/dgfm/evaluators/common.py`
- `src/dgfm/evaluators/runner.py`
- `src/dgfm/evaluators/qualitative.py`

### Keep stable
- `src/dgfm/trainers/baseline.py`
- `src/dgfm/models/official_unet.py`
- `src/dgfm/paths/*`
- `configs/experiment/fm_cifar10_baseline.yaml`

## Data flow of the new map branch

1. Run an offline teacher rollout from Gaussian noise.
2. Retain a compact teacher trajectory on `0 <= u <= 1`.
3. Save shards under `target.shard_root`.
4. Sample training tuples `(x_t, t, s, x_s_teacher)` from those shards.
5. Forward explicit map model:
   - `x_s_hat = M_theta(x_t, t, s)`.
6. Compute direct supervised map loss:
   - current implementation: pixel MSE / Huber.
7. Update model + EMA.
8. Evaluate by iterative rollout over a fixed time grid:
   - `x_{k+1} = M_theta(x_k, t_k, t_{k+1})`
9. Report FID / NFE / fixed-seed grids / multistep panel.

## Future extensibility

### Time-warp
- attach at sampler time-grid generation
- later attach at map training time-pair sampling

### Teacher switching
- current implemented mode:
  - `trajectory_shard`
- next mode:
  - `teacher_sampler`
- keep `MapTrainer` unchanged

### Semigroup defect
- attach as an additional map-level loss:
  - compare `M(t,u,x_t)` vs `M(s,u,M(t,s,x_t))`

### Richer CTM-style parameterization
- add alternative map parameterizations in `src/dgfm/models/map_*.py`
- keep `forward(x_t, t, s, **cond) -> x_s_hat` stable
