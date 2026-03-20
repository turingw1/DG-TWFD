# Technical Report: Assessing CTM as a Base Architecture

## 1. Executive Verdict
CTM is not a good direct base architecture for a clean research codebase. It is better used as a partial technical reference, mainly for its learned `t -> s` traversal formulation and few-step sampling logic. Its main strengths are explicit arbitrary-time traversal, integrated CTM/DSM/GAN training, and native few-step evaluation hooks. Its liabilities are heavy coupling between training logic, scheduler logic, model parameterization, evaluation, and dataset-specific defaults.

## 2. Project-Level Technical Claim
CTM is trying to learn a model-conditioned traversal map between two noise levels, not just a score and not just a one-step consistency projection. In code, the learned object appears as `G_theta(x_t, t, s)`, constructed in `cm/karras_diffusion.py:get_denoised_and_G()`. The model is called with both `t` and `s`; an inner object `g_theta` is formed first, then an outer `t,s`-dependent projection gives `G_theta`.

Compared with standard Consistency Models, CTM does not only learn projection to `sigma_min`; it can map from noisy state `x_t` to arbitrary target noise `s`. Compared with score-based diffusion, the teacher score model is used mainly to generate target traversals via Heun integration (`heun_solver()`), while the trained model directly learns traversal outputs rather than a pure score field.

## 3. Repository Architecture
The wiring is centralized around `cm_train.py`, `cm/train_util.py`, `cm/script_util.py`, and `cm/karras_diffusion.py`.

- `cm_train.py` parses one large argparse surface by merging multiple default dictionaries from `script_util.py`.
- `script_util.py` creates the model, diffusion helper, schedule samplers, and data defaults. For CIFAR10, model creation is routed to `EDMPrecond_CTM` in `cm/networks.py`.
- `train_util.py` owns the real training loop. It maintains `model`, `target_model`, optional `teacher_model`, optional discriminator, EMA copies, evaluation, sampling, checkpointing, and optimizer state.
- `karras_diffusion.py` contains the CTM math implementation: scaling rules, teacher Heun traversal, target construction, CTM loss, DSM loss, GAN loss, and model output interpretation.
- `sample_util.py` implements all samplers (`exact`, `onestep`, `cm_multistep`, `gamma`, `gamma_multistep`, `heun`, etc.).
- `image_sample.py` and `application_sample.py` are sampling entrypoints. They mostly wire argparse + checkpoint loading + `karras_sample()`.

The repository is operationally driven by shell commands under `commands/`, not by reusable high-level experiment objects.

## 4. Learned Object and Parameterization
The code is not learning a pure score network in CTM mode. It learns a dual-purpose parameterization:

- model input: `x_t`, rescaled `t`, and optionally rescaled `s`
- model output: a network output that is converted into `g_theta`
- outer projection: `G_theta(x_t, t, s)` using `get_outer_scalings(t, s, ...)`

In `get_denoised_and_G()`:
- `t` and `s` are represented as Karras/EDM-style sigmas, not normalized `[0,1]`.
- `rescaling_t()` maps sigma to a log-like scalar fed to the network.
- in CTM mode, `g_theta` is formed with inner scaling, then `G_theta` is formed with outer scaling.

So the “trajectory model” abstraction is present mathematically, but not as a clean explicit class. It is implicit inside `KarrasDenoiser`.

## 5. Training Mechanics
Training is implemented in `KarrasDenoiser.ctm_losses()` and called from `CMTrainLoop.forward_backward()`.

Flow:
- sample `t` via `schedule_sampler.sample_t()`
- compute `t_dt` via fixed/random Heun step count
- sample `s` via `schedule_sampler.sample_s()`
- build `x_t = x_0 + noise * t`
- compute CTM estimate via `get_ctm_estimate()`
- compute teacher/target traversal via `heun_solver()` + `get_ctm_target()`
- compute `consistency_loss`
- optionally add `denoising_loss` (DSM)
- optionally add `d_loss` (GAN)

CTM+DSM is the essential research path here. CTM+DSM+GAN is optional and entangled through the same loop, but GAN-specific branches are conditional and can be disabled.

## 6. Sampling Mechanics
Sampling is implemented by `sample_util.karras_sample()`.

- deterministic traversal: `exact`, `cm_multistep`, `onestep`
- stochastic diffusion-style traversal: `heun`, `dpm`, `ancestral`
- gamma variants: `gamma`, `gamma_multistep`

Long jumps are natively supported through explicit `ts` lists in `exact`, `cm_multistep`, and `gamma_multistep`. Deterministic vs stochastic behavior is handled by sampler choice and whether fresh noise is re-injected.

Unified few-step evaluation exists, but is not cleanly abstracted. In `train_util.py:evaluation()` the system hard-codes 1/2/4 steps and 18-step CIFAR evaluation.

## 7. Compatibility with My Planned Method

### 7.1 Semigroup Defect
Yes, partially. The natural insertion point is `KarrasDenoiser.ctm_losses()`, because it already samples `(t, t_dt, s)` and constructs both direct estimates and target traversals. However, what is missing is a first-class reusable `M(t,s,x)` interface. Today that map is implicit in `get_ctm_estimate()` / `get_denoised_and_G()`, so semigroup defect can be added, but only after extracting that traversal into an explicit callable abstraction.

### 7.2 Time-Warp
Yes, but not cleanly. Best insertion points:
- `schedule_sampler.sample_t()` / `sample_s()`
- `get_t()` in `KarrasDenoiser`
- `sample_util.karras_sample()` schedule construction

The current code assumes sigma schedules are the canonical time variable. A learnable monotonic `g_phi(t)` can be inserted, but you would need to refactor schedule creation away from raw sigma-index arithmetic.

### 7.3 Boundary Correction
Yes. Best insertion point is around high-noise-end calls to `get_denoised_and_G()` in either:
- `karras_sample()` denoiser wrapper, or
- `ctm_losses()` before target/estimate comparison.

This can be added as an extra correction module without changing the core CTM target construction.

### 7.4 Step-Robust Evaluation
Partially. The code already evaluates multiple NFEs, but the interface is fragmented and hard-coded. It can support 1/2/4/8/16-step evaluation cleanly only after refactoring `evaluation()` and `karras_sample()` into a unified evaluator API.

## 8. Architectural Friction and Refactor Burden
Refactor burden is high.

Main friction points:
- hidden coupling between training loop and diffusion helper
- many dataset-specific defaults hard-coded in `script_util.py`
- shell-command-driven usage rather than reusable experiment objects
- teacher path assumptions embedded in model creation and training
- evaluation and checkpointing mixed directly into training loop
- sampler behavior encoded as string switches in one large function

The repo is technically rich, but not modular.

## 9. Migration Plan into My Framework
Reuse directly:
- CTM traversal math in `get_denoised_and_G()`
- target construction pattern in `ctm_losses()`
- few-step sampler logic from `sample_util.py`

Wrap:
- Heun teacher traversal
- time/sigma schedule utilities
- sampler families

Rewrite:
- training loop
- config/arg handling
- evaluation orchestration
- checkpoint interface

Discard:
- command-script dependence
- mixed GAN/discriminator plumbing unless needed
- dataset-path defaults in `script_util.py`

## 10. Minimal Refactor Plan
Smallest useful conversion:

1. Extract a `TraversalModel` interface:
   - `forward(x_t, t, s, **kwargs) -> x_s`
2. Isolate teacher traversal:
   - `teacher_step(x_t, t, s)`
3. Split `ctm_losses()` into:
   - time-pair sampling
   - estimate construction
   - target construction
   - loss assembly
4. Move sampler strings into separate sampler classes/functions
5. Replace argparse-default sprawl with structured config
6. Standardize a unified `evaluate_nfe([1,2,4,8,16])`

Without these steps, semigroup defect and time-warp additions will remain invasive patches.

## 11. Comparative Judgment: CTM vs flow_matching
| Question | Better Choice |
|---|---|
| Better engineering base | `flow_matching` |
| Better conceptual reference for arbitrary `t -> s` traversal | `CTM` |
| Easier for adding semigroup defect | `flow_matching` |
| Easier for adding time-warp | `flow_matching` |
| Better reference for few-step diffusion-like traversal targets | `CTM` |
| Safer for long-term research iteration | `flow_matching` |

Reason: `flow_matching` exposes clean path/solver/model-wrapper abstractions, while CTM buries its best ideas inside a large coupled training system.

## 12. Final Recommendation
Use CTM partially and refactor heavily.

More specifically:
- do not use CTM directly as the base repository,
- do use it as a technical reference for arbitrary-time traversal parameterization, teacher target construction, and few-step evaluation patterns,
- keep `flow_matching` or your own cleaner framework as the engineering base,
- port only the CTM-specific traversal logic you actually need.
