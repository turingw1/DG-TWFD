# Technical Report: Assessing `flow_matching` as a Base Architecture

## 1. Executive Verdict
`flow_matching` is a good conceptual and engineering base for continuous-time velocity-based generative modeling. It is structurally strong in path abstraction, scheduler abstraction, solver abstraction, and post-training schedule transformation. It is weak as a unified research repo because most training, logging, checkpointing, and evaluation code lives in modality-specific `examples/` trees rather than the core library. Migration is recommended, but not as-is. Reuse the core library abstractions directly; rewrite or heavily wrap the example training pipelines.

## 2. Repository Architecture
The repository splits cleanly into a reusable core and non-reusable examples.

- Core library: `flow_matching/flow_matching/`
  - `path/` defines probability paths and schedulers.
  - `solver/` defines continuous and discrete samplers.
  - `loss/` provides discrete generalized KL.
  - `utils/model_wrapper.py` defines the wrapper contract expected by solvers and scheduler transforms.
- Image example: `examples/image/`
  - `train.py` is the main training entrypoint.
  - `training/train_loop.py` samples times and path states, computes loss, and updates the model.
  - `training/eval_loop.py` handles sample generation and FID.
  - `models/` holds UNet backbones and EMA wrapper.
- Text example: `examples/text/`
  - `run_train.py` launches Hydra-configured distributed training.
  - `logic/flow.py` builds path, source distribution, and loss.
  - `logic/training.py` performs the actual FM step.

Component interaction is simple: the example code samples `(x_0, x_1, t)`, calls a `ProbPath` to construct `x_t` and a target, trains a model on that target, then uses a solver to integrate the learned field at inference time.

## 3. Core Mathematical Object Learned by the Repo
The core continuous object is a conditional velocity field, not a flow map.

- Inputs: `x_t`, scalar `t`, optional conditioning.
- Output: `u_t(x_t)` in continuous image FM.
- Supervision target: `dx_t` returned by `PathSample` from `ProbPath.sample(...)`.
- Time parameterization: normalized `t in [0,1]`, then interpreted by the chosen scheduler.
- Sampling: numerically integrate the learned velocity with `ODESolver.sample(...)`.

For discrete FM, the learned object is a posterior-like predictor over `x_1` conditioned on `x_t`, later converted to a probability velocity or used directly by `MixtureDiscreteEulerSolver`.

This repo does not natively train `M(t,s,x)` or any explicit time-to-time map.

## 4. Training and Sampling Mechanics
Continuous image FM:

- Source is Gaussian noise, target is data.
- Path is usually `CondOTProbPath`, i.e. affine path `x_t = sigma_t x_0 + alpha_t x_1`.
- Training samples `t ~ U[0,1]` or an EDM-style skewed distribution.
- `path.sample(...)` returns `x_t` and `dx_t`.
- Objective is plain MSE: `||model(x_t, t) - dx_t||^2`.
- Sampling solves the ODE from `t=0` to `t=1` using `ODESolver`.

Discrete FM:

- Path is `MixtureDiscreteProbPath`.
- Training predicts logits for `x_1 | x_t`.
- Objective is either cross-entropy or `MixturePathGeneralizedKL`.
- Sampling uses `MixtureDiscreteEulerSolver`.

This is closest to standard flow matching with explicit probability-path sampling. It is not a flow-map method, not rectified-flow-specific, and not built around arbitrary `t -> s` supervision.

## 5. Compatibility with My Planned Method
| Module | Compatibility | Why |
|---|---|---|
| Semigroup defect | Partial | Needs an explicit `M(t,s,x)` abstraction |
| Time-warp | Good | Scheduler layer is already explicit |
| Boundary correction | Good | Best inserted as model wrapper near high-noise input |
| Few-step robust inference | Good | Solvers already accept custom time grids |

### 5.1 Semigroup Defect Module
Not naturally supported. The missing abstraction is an explicit flow map `M(t,s,x)`. The current core exposes only:

- path sampling `ProbPath.sample(x_0, x_1, t)`
- velocity model `u_t(x)`
- solver integration over a time grid

You can define `M(t,s,x)` by integrating the ODE from `t` to `s`, but that is expensive and not a first-class reusable function. To add defect cleanly, introduce a reusable map wrapper on top of `ODESolver` or train a new model with `(x,t,s) -> x_s`.

### 5.2 Time-Warp Module
This repo is unusually compatible with time-warp. Best insertion points:

- scheduler definitions in `path/scheduler/scheduler.py`
- post-training wrapper `ScheduleTransformedModel`
- training-time `t` sampling in the example loops
- solver time grids in `ODESolver.sample(...)`

For a learnable monotonic `g_phi(t)`, the cleanest route is to implement a new scheduler or a scheduler wrapper. This is much easier here than in CTM.

### 5.3 Boundary Correction
Yes. Best insertion point is a `ModelWrapper` around the velocity model. That wrapper can:

- detect high-noise times,
- apply a boundary correction module,
- then forward to the base model or alter its output.

This keeps training and solver code unchanged.

### 5.4 Step-Robust Few-Step Inference
Yes. `ODESolver.sample(...)` already accepts arbitrary `time_grid`. A unified `1/2/4/8/16-step` evaluator only requires:

- centralizing time-grid construction,
- centralizing solver choice,
- centralizing metrics/snapshot logging now spread across `examples/image/training/eval_loop.py`.

## 6. Migration Plan into My Existing Code Framework
Priority order:

Reuse directly:
- `ProbPath`, `PathSample`, `AffineProbPath`, `CondOTProbPath`
- schedulers in `path/scheduler/`
- `ODESolver`
- `ScheduleTransformedModel`

Wrap:
- model interface via `ModelWrapper`
- time-grid builders for few-step evaluation
- solver calls into a clean `sample_n_steps(n)` API

Rewrite:
- all training loops from `examples/image/` and `examples/text/`
- config handling for image examples
- evaluation orchestration and logging
- checkpoint interfaces

Keep decoupled:
- modality-specific backbones
- dataset loading
- metrics

## 7. Minimal Refactor Plan
Smallest useful changes:

1. Add a first-class `FlowMap` interface:
   - `forward(x, t, s, **cond) -> x_s`
   - initially implement it through solver integration over the learned velocity.
2. Separate example training code from FM math:
   - extract `sample_training_tuple(...)`
   - extract `compute_velocity_loss(...)`
   - extract `evaluate_nfe(...)`
3. Introduce centralized experiment config for image training.
4. Add a reusable time-grid builder for `1/2/4/8/16` step evaluation.
5. Add wrapper hooks:
   - `BoundaryCorrectedModelWrapper`
   - `TimeWarpedScheduler` or learnable scheduler wrapper

Without these, semigroup-defect experiments will become invasive patches inside example scripts.

## 8. Risk Assessment
Main risks:

- Core library and training pipelines are split; examples are not clean research infrastructure.
- Continuous FM is hard-wired to velocity supervision, not time-to-time mapping.
- Image example has weak config modularity compared with text example.
- Few-step evaluation exists only as ad hoc evaluation code, not a reusable benchmark interface.
- Semigroup defect built via solver calls may be computationally expensive.
- Training/inference mismatch can appear if you add time-warp only at inference without retraining.

## 9. Recommended Development Order
1. Reproduce a baseline with the existing velocity FM setup.
2. Migrate only the reusable core (`path`, `scheduler`, `solver`, `ModelWrapper`) into your framework boundary.
3. Add unified few-step evaluation over custom time grids.
4. Add time-warp as a scheduler-level module.
5. Add boundary correction as a model wrapper.
6. Add semigroup defect only after defining a clean reusable `M(t,s,x)`.

This order is safest because time-warp and few-step evaluation fit the existing abstractions, while semigroup defect requires a deeper shift from velocity-field training toward map-based training.

## 10. Final Recommendation
Use `flow_matching` partially and refactor around it.

- Use it as the engineering base for path, scheduler, and solver abstractions.
- Do not use the example training code as your long-term research framework.
- Borrow its scheduler and solver design directly.
- Add your own explicit `M(t,s,x)` layer before implementing semigroup defect.
