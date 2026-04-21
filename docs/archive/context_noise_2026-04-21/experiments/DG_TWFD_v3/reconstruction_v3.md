You are a senior diffusion-model engineer. Your task is to refactor the current heuristic four-loss distillation baseline into a unified Defect-Guided Trajectory Distillation (DGTD) implementation.

Do not implement this as a naive sum of four losses. The core objective must be one Warp-Weighted Consistency Residual.

## 0. First inspect local resources

Assume this workspace:

DGTD_workspace/
  refs/
    consistency_models/
    ctm/
    flow_matching/
    edm/
    rectified_diffusion/
    min_snr/
    optimalsteps/
  /docs/experiments/map_branch/teacher/
    teacher_entry.md
    teacher_config.yaml
    cache_schema.md
  /docs/experiments/map_branch/baseline/
    current_losses.md
    current_train_command.md
    current_metrics.md

First read:
- /docs/experiments/map_branch/baseline/current_losses.md
- /docs/experiments/map_branch/teacher/teacher_entry.md
- /docs/experiments/map_branch/teacher/cache_schema.md
- refs/edm training and sampling code
- refs/flow_matching examples and model interfaces
- refs/ctm training objective and trajectory sampling logic
- refs/consistency_models training objective

Do not invent APIs. Inspect the existing files and adapt to the actual project structure.

## 1. Mathematical objective

Replace the old loss stack:

L_map + lambda_comp L_comp + lambda_boundary L_boundary + lambda_warp L_warp

with one unified objective:

R_eta(t,s,u;x_t)
=
M_theta(x_t,t,u)
-
T_eta(s,u, M_theta(x_t,t,s))

where:

T_eta(s,u,z)
=
eta * Phi_T(s,u,z)
+
(1-eta) * stopgrad(M_theta(z,s,u))

If online teacher local solver Phi_T is unavailable, implement the fallback in this order:

1. cached teacher interpolation if z is near a cached teacher trajectory state
2. detached student bootstrap
3. optional teacher one-step local solver hook, left as interface

Final loss:

L_DGTD
=
E_{t>s>u ~ q_phi}
[
  omega(t,s,u) * ||R_eta(t,s,u;x_t)||^2_{G(u)}
]

where G(u) is a detail-aware metric.

This one residual must cover:
- flow-map matching
- semigroup/composition consistency
- low-noise detail/boundary behavior
- defect-guided time allocation

Do not add separate boundary/perceptual/straightness losses in the first implementation.

## 2. Defect definition

Implement sample defect:

D_sample
=
||R_eta(t,s,u;x_t)||^2_{G(u)}
/
(eps + ||x_u_teacher - x_t||_2^2)

Maintain EMA bin statistics over B time bins:

D_bar[b]   # normalized teacher-anchored semigroup defect
K_bar[b]   # finite-difference teacher curvature proxy
HF_bar[b]  # high-frequency residual near low sigma
count[b]

EMA update:

D_bar[b] = mu * D_bar[b] + (1-mu) * D_sample

Use mu:
- warmup: 0.95
- adaptive/main: 0.99

Curvature proxy from cached teacher trajectory:

v_i = (x_{i+1} - x_i) / (t_{i+1} - t_i)

K_i = ||v_{i+1}-v_i||^2 / (eps + ||v_i||^2)

HF residual:

HF = ||Laplacian(R_eta)||_2^2

## 3. Defect-guided warp

Implement a monotone time-density warp.

Use B bins. Parameterize q_phi with cumulative softplus:

density_raw: learnable Tensor[B]
q_phi = softplus(density_raw) + eps
q_phi = q_phi / q_phi.sum()
cdf_phi = cumsum(q_phi)

This represents:

g_phi'(t) = q_phi(t)
g_phi(t) = integral_0^t q_phi(tau) d tau

Implement:
- t_to_r(t)
- r_to_t(r)
- sample_triplets(batch_size)
- kl_to_target_density(q_D)

Build target defect density:

A_b =
normalize(D_bar[b])
+ 0.25 * normalize(K_bar[b])
+ 0.5 * normalize(HF_bar[b])

q_D[b] ∝ (eps + A_b)^beta * q_base[b]

Smooth q_D with small 1D kernel or EMA.

Train warp by:

L_phi = KL(stopgrad(q_D) || q_phi)

Important:
This is not a fourth student loss. It is an outer-loop sampler/time-coordinate update.

## 4. Training schedule

Use three stages.

### Stage 1: warm-up, first 10%-15% steps

- q_phi remains close to q_base
- beta = 0
- rho = 0
- eta = 1.0
- train only unified residual L_DGTD
- collect D_bar, K_bar, HF_bar, but do not let defect control sampling yet

### Stage 2: adaptive, middle 60%-70% steps

- beta increases to beta_final, e.g. 0.5-1.0
- update q_D every update_density_every steps
- update q_phi by KL every update_warp_every steps
- eta decays from 1.0 to eta_min, e.g. 0.3-0.5
- sample t,s,u uniformly in warped time r-space and map back through r_to_t

### Stage 3: flatten, final 15%-20% steps

- reduce beta or increase temperature
- make q_D less sharp
- keep defect statistics but avoid overfitting only high-defect bins
- preserve global coverage

## 5. Sampling t,s,u

Do not sample t,s,u independently.

Sample in warped time:

r_t ~ Uniform(0,1)
Delta_1, Delta_2 from mixture:

short:  [1/64, 1/16], probability 0.5
medium: [1/16, 1/4], probability 0.3
long:   [1/4, 1], probability 0.2

Require Delta_2 > Delta_1.

r_s = clamp(r_t - Delta_1, 0, 1)
r_u = clamp(r_t - Delta_2, 0, 1)

Then:

t = r_to_t(r_t)
s = r_to_t(r_s)
u = r_to_t(r_u)

Ensure t > s > u according to the project’s time convention. If the project uses sigma decreasing/increasing in the opposite direction, handle it explicitly and document it.

## 6. Detail-aware metric G(u)

Implement first version without LPIPS.

metric_norm(R,u):

lambda_l2(u) * ||R||_2^2
+
lambda_hf(u) * ||Laplacian(R)||_2^2

lambda_hf(u) should increase near low sigma / clean endpoint.

Use project’s sigma(t) if available. Otherwise approximate with t.

Example:

lambda_hf = lambda_hf_max * exp(-sigma(u)^2 / sigma_detail^2)

Default:
lambda_hf_max = 0.1 initially
sigma_detail chosen near low-noise regime
make configurable

## 7. Importance weighting

Implement mild correction:

omega(t,s,u)
=
w_edm(u) * w_snr(u) * p_corr(t)^(-kappa)

Default:
kappa = 0.5

If EDM/Min-SNR weighting exists in the project, reuse it. Otherwise implement clean modular placeholders.

Do not overcomplicate the first version.

## 8. Required new modules

Create these files unless the project already has equivalent locations:

dgtd/
  warp.py
  defect.py
  metrics.py
  cache.py
  train_dgtd.py
  sample_dgtd.py

### dgtd/warp.py

Implement class MonotoneDensityWarp:

- __init__(num_bins, eps=1e-6, init="uniform")
- density()
- cdf()
- t_to_r(t)
- r_to_t(r)
- sample_triplets(batch_size, device)
- kl_to_target_density(q_D)
- state_dict/load_state_dict compatible

Use torch.searchsorted for inverse CDF with linear interpolation.

### dgtd/defect.py

Implement:

- compute_dgtd_residual(student, teacher_adapter, x_t, t, s, u, eta)
- compute_sample_defect(R, x_t, x_u_teacher, metric_value)
- update_ema_bins(stats, bin_ids, values, mu)
- build_target_density(D_bar, K_bar, HF_bar, q_base, beta, eps)
- smooth_density(q)

### dgtd/metrics.py

Implement:

- laplacian_filter(x)
- high_frequency_norm(x)
- metric_norm(R, u, sigma_fn=None, lambda_hf_max=0.1)
- min_snr_weight(t_or_sigma)
- edm_weight(t_or_sigma)

### dgtd/cache.py

Implement TrajectoryCacheDataset:

Expected cache item:
{
  "times": Tensor[m],
  "states": Tensor[m,C,H,W],
  "curvature": optional Tensor[m-2],
  "cond": optional
}

Functions:
- interpolate_state(traj, t)
- interpolate_curvature(traj, t)
- get_teacher_pair(traj, t, u)

Use nearest-neighbor first if interpolation is hard; add linear interpolation after correctness is verified.

### dgtd/train_dgtd.py

Implement high-level loop with:

1. load teacher trajectory cache
2. initialize student M_theta
3. initialize MonotoneDensityWarp
4. initialize defect statistics
5. warm-up/adaptive/flatten stage scheduler
6. compute unified DGTD residual
7. update student every step
8. update q_D every update_density_every
9. update warp every update_warp_every
10. log:
   - L_DGTD
   - mean defect
   - D_bar histogram
   - q_phi density
   - q_D target density
   - low-sigma HF residual
   - eta, beta

### dgtd/sample_dgtd.py

Implement two sampling modes.

Mode A: uniform warped-time sampling

For K steps:
r_i = i/K
t_i = r_to_t(r_i)
x_{t_{i+1}} = M_theta(x_{t_i}, t_i, t_{i+1})

Mode B: optional DP schedule

Leave as clean TODO interface:
- estimate interval cost C(i,j)
- DP select K nodes
- save schedule as JSON

Do not block Mode A on Mode B.

## 9. Teacher adapter

Create a TeacherAdapter abstraction.

It should expose:

- local_flow(s,u,z)
- cached_state(traj,t)
- sigma(t)
- precondition(x,t) if needed
- get_condition(batch) if conditional

If online teacher local_flow is not available, return None and allow fallback to detached student bootstrap.

Do not hard-code SDXL unless current project uses SDXL.

## 10. Training loop pseudo-code

for step in range(num_steps):

    stage = get_stage(step)

    traj = cache.sample(batch_size)

    r_t, r_s, r_u = warp.sample_triplets(batch_size)
    t = warp.r_to_t(r_t)
    s = warp.r_to_t(r_s)
    u = warp.r_to_t(r_u)

    x_t = cache.interpolate_state(traj, t)
    x_u_teacher = cache.interpolate_state(traj, u)

    x_s_pred = student(x_t, t, s)
    x_u_direct = student(x_t, t, u)

    teacher_cont = teacher.local_flow(s, u, x_s_pred)

    if teacher_cont is None:
        teacher_cont = stopgrad(student(x_s_pred, s, u))

    self_cont = stopgrad(student(x_s_pred, s, u))

    target = eta * teacher_cont + (1 - eta) * self_cont

    R = x_u_direct - target

    loss_metric = metric_norm(R, u)
    omega = compute_weight(t, s, u)

    loss = mean(omega * loss_metric)

    update student theta

    D_sample = loss_metric / (eps + norm(x_u_teacher - x_t)^2)
    update EMA stats

    if step >= warmup_end and step % update_density_every == 0:
        q_D = build_target_density(...)
        q_D = smooth(q_D)

    if step >= warmup_end and step % update_warp_every == 0:
        L_phi = warp.kl_to_target_density(q_D)
        update phi only

    log everything

## 11. Validation and diagnostics

Implement a script or hook to plot/save:

- defect heatmap over time bins
- learned q_phi
- target q_D
- curvature K_bar
- HF_bar
- sampling schedules for K = 1,2,4,8,16
- loss curve by bin

The main sanity checks:

1. During warm-up, q_phi should stay close to base.
2. During adaptive stage, q_phi should allocate more density to high D_bar and low-sigma HF regions.
3. In flatten stage, q_phi should become less sharp.
4. DGTD loss should not explode when warp updates start.
5. Multi-step inference should show less degradation from 1 to 8/16 steps than baseline.

## 12. Ablations to prepare

Implement config flags:

--disable_warp
--disable_hf_metric
--disable_teacher_anchor
--uniform_time
--logit_normal_time
--defect_beta
--ema_mu
--eta_min
--update_density_every
--update_warp_every

Required ablations:

A. baseline current losses
B. unified DGTD residual without learned warp
C. DGTD + defect-guided warp
D. DGTD + defect-guided warp + low-sigma HF metric
E. DGTD + uniform warped inference

## 13. Important constraints

- Keep the implementation minimal and correct first.
- Do not invent unsupported teacher APIs.
- Do not silently ignore tensor shape/time convention issues.
- Add assertions for t>s>u or the project’s equivalent convention.
- All new logic must be configurable.
- Prefer offline cached teacher trajectories to expensive online teacher calls.
- Avoid JVPs in version 1.
- Avoid LPIPS in version 1.
- The final objective must remain the single DGTD residual.

## 14. Deliverables

After implementation, output:

1. A concise summary of files changed.
2. The exact training command.
3. The exact sampling command.
4. A list of assumptions about teacher cache format.
5. Any TODOs left for online teacher local_flow or DP schedule.
6. A minimal smoke test command that runs one training step and one sampling call.