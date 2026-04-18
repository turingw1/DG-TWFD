# DGTD v3 Round-3 Online Teacher Query Report

This report audits the current `DG_TWFD_v3` implementation from the standpoint
of "making online teacher the real mainline". It does not propose a code patch.
It records what the current code can do, what it cannot do, and which online
continuation design is the best next step.

Evidence sources used in this report:

- code under `src/dgtd/*`
- teacher implementation under `src/dgfm/teachers/*`
- training/runtime entrypoints under `scripts/*`
- current configs under `configs/*`
- tests in [tests/test_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/tests/test_dgtd.py:1)
- prior server-backed runtime summary in
  [docs/dgtd_v3_round2_verification.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/dgtd_v3_round2_verification.md:1)

The most important framing detail is this:

- the current code does support **online trajectory generation from x0**
- the current code does **not** support **online continuation from arbitrary
  `z_s`**
- therefore the current "online teacher" path is only online at the
  data/trajectory layer, not at the DGTD residual continuation layer

## 1. Current Online-Teacher Capability Map

### A. Can the online teacher generate a trajectory from real data / x0?

Yes.

Code path:

- [DGTDTrainer.run()](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:627)
  builds the teacher adapter and then chooses dataloaders via
  [select_dgtd_dataloaders()](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:117)
- when `dgtd.disable_online_teacher=false` and
  `dgtd.use_online_teacher_data=true`, the loader switches to
  `build_image_dataloaders(...)` instead of cache shards at
  [src/dgtd/train_dgtd.py:126](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:126)
- during `_run_epoch`, the image batch is converted into a teacher trajectory by
  [teacher_adapter.online_trajectory_from_x0(...)](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:319)
- `TeacherAdapter.online_trajectory_from_x0()` is implemented at
  [src/dgtd/teacher.py:61](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:61)
- this calls
  [DiffusersDDPMTeacher.sample_trajectory_from_x0(...)](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/teachers/diffusers_ddpm.py:126)
- the result is converted into the same `times/states/curvature/cond` payload as
  offline shards by
  [build_trajectory_payload(...)](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/cache.py:43)

Interpretation:

- online teacher trajectory generation is real and currently working
- the training loop is using online-generated `x_t`, `x_s_teacher`,
  `x_u_teacher`
- the current online path still reuses the same interpolation/cache-shaped
  payload interface used by offline trajectory shards

### B. Can the online teacher continue from arbitrary `z_s` to `u`?

No, not in the current code.

The only entrypoint that can produce `source=online` is
[TeacherAdapter.local_flow()](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:82).
That function only takes the online branch when:

- `self.online_teacher is not None`
- and `hasattr(self.online_teacher, "local_flow")`

See
[src/dgtd/teacher.py:98](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:98).

`DiffusersDDPMTeacher` implements:

- `prepare(...)`
- `sample_x0(...)`
- `sample_trajectory(...)`
- `sample_trajectory_from_x0(...)`

See
[src/dgfm/teachers/diffusers_ddpm.py:49](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/teachers/diffusers_ddpm.py:49),
[src/dgfm/teachers/diffusers_ddpm.py:76](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/teachers/diffusers_ddpm.py:76),
[src/dgfm/teachers/diffusers_ddpm.py:114](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/teachers/diffusers_ddpm.py:114),
[src/dgfm/teachers/diffusers_ddpm.py:126](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/teachers/diffusers_ddpm.py:126).

It does **not** implement `local_flow(...)`.

So the answer is not "unsupported in principle", but "unsupported by the current
teacher interface".

### C. Does the online teacher have an explicit `local_flow(s,u,z)` API?

No.

The class
[DiffusersDDPMTeacher](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/teachers/diffusers_ddpm.py:16)
has no `local_flow` method at all. This is the direct reason the adapter never
returns `CONTINUATION_SOURCE_ONLINE` from the current diffusers teacher.

### D. If not, can existing sampler/scheduler primitives cheaply simulate it?

Yes, but only at different cost/faithfulness tradeoffs.

The raw building blocks already exist:

- timestep mapping from teacher time `tau` to inference index via
  [_noise_time_to_inference_index()](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/teachers/diffusers_ddpm.py:86)
- one-teacher-step denoising through
  [_forward_eps()](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/teachers/diffusers_ddpm.py:93)
- multi-step rollout between scheduler indices through
  [_rollout_between_indices()](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/teachers/diffusers_ddpm.py:101)

What is missing is only the public interface that applies those operators to an
arbitrary `z_s`.

The important caution is that "possible" does not mean "cheap":

- one or a few solver steps from `z_s` is low/medium extra cost
- a true full online rollout from every `z_s` to `u` is high cost
- any such rollout also introduces a scheduler-consistency question because the
  current global sigma logic in DGTD is not derived from the teacher scheduler

### E. In current DGTD training, where does online teacher participate, and
where does it not participate?

Current online teacher **does participate** in:

- loading and preparing the diffusers teacher:
  [src/dgtd/train_dgtd.py:630](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:630)
- choosing image dataloaders for training:
  [src/dgtd/train_dgtd.py:632](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:632)
- materializing a teacher trajectory batch from real images:
  [src/dgtd/train_dgtd.py:319](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:319)
- providing `x_t`, `x_s_teacher`, `x_u_teacher`, and `curvature` through
  interpolation over that trajectory:
  [src/dgtd/train_dgtd.py:341](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:341)

Current online teacher **does not participate** in:

- `teacher_cont` inside the DGTD residual unless `online_teacher.local_flow`
  exists:
  [src/dgtd/teacher.py:98](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:98)
- sample/eval entrypoints, which only load model/checkpoint and do not build the
  teacher at all:
  [scripts/run_sample_dgtd.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/scripts/run_sample_dgtd.py:1),
  [scripts/run_eval.py](/home/gzwlinux/vscode/gitProject/DG-TWFD/scripts/run_eval.py:1)

Practical conclusion:

- "online teacher" is currently a **trajectory/data feature**
- it is **not yet a continuation feature**

## 2. Why `continuation_sources.online = 0.0`

The current explanation is fully determined by the code; there is no ambiguity.

### 2.1 What the server evidence says

The server-backed verification doc records:

- `online_teacher_built True`
- `online_teacher_prepare ok`
- `online_teacher_data: true`
- `continuation_sources: {"online": 0.0, "cached_affine": 0.25, "cached_exact": 0.75, "bootstrap": 0.0}`

See
[docs/dgtd_v3_round2_verification.md:35](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/dgtd_v3_round2_verification.md:35)
and
[docs/dgtd_v3_round2_verification.md:154](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/dgtd_v3_round2_verification.md:154).

### 2.2 Why this happens in code

The controlling logic is:

1. Trainer chooses online image dataloaders when online teacher exists:
   [src/dgtd/train_dgtd.py:117](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:117)
2. Trainer converts image batch into an online teacher trajectory:
   [src/dgtd/train_dgtd.py:319](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:319)
3. DGTD residual calls `teacher_adapter.local_flow(...)` with that trajectory and
   `x_s_pred`:
   [src/dgtd/defect.py:28](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/defect.py:28)
4. `TeacherAdapter.local_flow(...)` only emits `source=online` if
   `hasattr(self.online_teacher, "local_flow")`:
   [src/dgtd/teacher.py:98](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:98)
5. Since `DiffusersDDPMTeacher` does not implement `local_flow`, control falls
   through to the trajectory-based branch:
   [src/dgtd/teacher.py:109](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:109)
6. That branch computes:
   - `cached_exact` when `rel_error <= proximity_rtol`:
     [src/dgtd/teacher.py:115](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:115)
   - otherwise `cached_affine`:
     [src/dgtd/teacher.py:127](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:127)

This means the current runtime state is:

- online teacher **is built**
- online teacher **is called**
- online teacher **does provide the trajectory**
- online teacher **does not provide continuation**

So `continuation_sources.online = 0.0` happens because the code path that sets
that source ID is never entered.

### 2.3 Which config flags matter, and which do not

Flags that affect the data path:

- `dgtd.disable_online_teacher`
- `dgtd.use_online_teacher_data`

See
[configs/experiment/dgtd_cifar10_v3.yaml:52](/home/gzwlinux/vscode/gitProject/DG-TWFD/configs/experiment/dgtd_cifar10_v3.yaml:52)
and
[src/dgtd/train_dgtd.py:117](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:117).

Flags that affect the continuation fallback path:

- `dgtd.teacher_continuation_mode`
- `dgtd.teacher_keep_cached_exact`
- `dgtd.teacher_proximity_rtol`

See
[configs/experiment/dgtd_cifar10_v3.yaml:53](/home/gzwlinux/vscode/gitProject/DG-TWFD/configs/experiment/dgtd_cifar10_v3.yaml:53)
and
[src/dgtd/teacher.py:170](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:170).

Important negative fact:

- there is currently **no config flag** that can make `source=online` appear if
  the teacher class itself lacks `local_flow`

### 2.4 One more operational nuance

The default main experiment config still has:

- `dgtd.disable_online_teacher: true`

at
[configs/experiment/dgtd_cifar10_v3.yaml:85](/home/gzwlinux/vscode/gitProject/DG-TWFD/configs/experiment/dgtd_cifar10_v3.yaml:85).

So the online path is not even the default branch state yet; it is currently
activated by CLI override during smoke.

## 3. Candidate Designs for True Online Continuation

The following three designs are ordered from least invasive to most faithful.

### 3.1 Plan A: Online single-step surrogate

Core idea:

- from `z_s`, run the teacher for one or a few actual scheduler steps toward `u`
- use that result as `teacher_cont`
- if `u` is farther away than the allowed teacher-step budget, stop early and
  treat the result as a surrogate continuation

Minimal interface:

- add `local_flow(s, u, z, *, max_teacher_steps=K)` to
  [DiffusersDDPMTeacher](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgfm/teachers/diffusers_ddpm.py:16)

Likely files:

- `src/dgfm/teachers/diffusers_ddpm.py`
- `src/dgtd/teacher.py`
- `tests/test_dgtd.py`
- maybe `docs/dgtd_v3_round2_patch_notes.md` or a new patch note

Mathematical form:

- let `i_s = index(s)` and `i_u = index(u)`
- define `K_eff = min(K, i_u - i_s)` under the current descending DDIM schedule
- compute:
  `teacher_cont_online(z_s; s,u) = RolloutTeacher(z_s, i_s -> i_s + K_eff)`

Training cost:

- extra teacher UNet calls per batch and per DGTD sample
- cost scales with chosen `K`
- materially higher than the current affine fallback

Pros:

- gives a real online continuation source
- reuses existing scheduler stepping machinery
- low interface complexity

Risks:

- if `K` is too small, the continuation may be temporally misaligned with `u`
- if `K` is too large, throughput may collapse
- detached online exact continuation still has the warmup direct-only corner case

### 3.2 Plan B: Teacher-conditioned affine / Jacobian-lite continuation

Core idea:

- keep the current "teacher anchor + student-sensitive correction" structure
- but make the anchor explicitly online by deriving it from the current online
  teacher trajectory and, optionally, one lightweight teacher local gain
  estimate

Minimal interface options:

1. cheapest version:
   - no new heavy teacher API
   - adapter computes:
     `teacher_cont_online(z_s; s,u) = x_u_teacher_online + alpha_online(s,u) * (z_s - x_s_teacher_online)`
2. slightly richer version:
   - teacher exposes a cheap scalar/diagonal gain estimator, for example
     `estimate_local_gain(s, u, teacher_s, teacher_u)`

Likely files:

- `src/dgtd/teacher.py`
- optionally `src/dgfm/teachers/diffusers_ddpm.py`
- `src/dgtd/sigma.py` only if the online gain is tied to scheduler-derived sigma
- tests and docs

Mathematical form:

- simplest:
  `teacher_cont_online(z_s; s,u) = x_u_teacher + alpha_online(s,u) * (z_s - x_s_teacher)`
- current branch already has the same shape for offline cached affine at
  [src/dgtd/teacher.py:127](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/teacher.py:127)
- the only change is that `x_s_teacher` and `x_u_teacher` are explicitly treated
  as **online teacher anchors**, not as cache fallbacks

Training cost:

- near current cost if `alpha_online` is cheap
- much cheaper than a true per-sample solver rollout

Pros:

- most compatible with the current DGTD residual
- naturally preserves bridge-side gradient even when `eta=1.0`, because
  `teacher_cont_online` still depends on `z_s`
- easy to demote `cached_exact/cached_affine` to fallback only
- aligns with the current logging and `source_ids` machinery

Risks:

- not a "true solver continuation"
- quality depends on how good the chosen `alpha_online` is
- if `alpha_online` remains just `sigma(u)/sigma(s)`, then "online" is mainly in
  the anchors, not in the dynamics

### 3.3 Plan C: True online local_flow

Core idea:

- the teacher explicitly supports continuation from arbitrary `z_s`
- DGTD continuation becomes:
  `teacher_cont_online(z_s; s,u) = TeacherLocalFlow(z_s, s -> u)`

Minimal interface:

- implement `local_flow(s, u, z, *, extra=None)` on the teacher class
- optionally add a batched scheduler-index helper

Likely files:

- `src/dgfm/teachers/diffusers_ddpm.py`
- `src/dgtd/teacher.py`
- tests, smoke docs, profiling docs

Mathematical form:

- let `i_s = index(s)` and `i_u = index(u)`
- compute:
  `teacher_cont_online(z_s; s,u) = RolloutTeacher(z_s, i_s -> i_u)`

Training cost:

- highest of the three plans
- effectively duplicates a teacher rollout from `z_s` for every DGTD triplet

Pros:

- conceptually the cleanest
- genuinely makes `source=online` the dominant continuation source
- most faithful to the intended teacher dynamics

Risks:

- worst throughput hit
- highest implementation complexity
- most sensitive to mismatch between DGTD continuous times and teacher discrete
  scheduler indices
- strongest teacher-drift/teacher-domain mismatch risk if `z_s` lies off the
  teacher trajectory manifold

## 4. Recommended Mainline Design

Recommended mainline: **Plan B, teacher-conditioned online affine continuation**.

This is the single recommended mainline because it best matches the current
branch constraints:

1. it keeps the existing DGTD residual geometry almost unchanged
2. it preserves bridge-side gradient even at high teacher weight
3. it upgrades the online teacher from "trajectory generator only" to "actual
   continuation source" without paying full per-sample solver cost
4. it uses machinery the branch already has:
   - online trajectory generation
   - sigma helper
   - affine continuation shape
   - continuation source logging

Why not Plan A as the mainline:

- it is a useful stepping stone, but a capped one/few-step surrogate is too
  dependent on an arbitrary step budget
- for large `s -> u` gaps, it becomes hard to interpret whether the continuation
  is teacher-consistent or just a short denoise heuristic
- it still tends to produce detached exact-like teacher outputs, which leaves the
  warmup direct-only corner case mostly intact

Why not Plan C as the mainline:

- it is the cleanest conceptually, but it is too expensive for the current
  branch stage
- the current few-step DGTD workflow already spends teacher compute to build the
  online trajectory batch; a second full local-flow rollout from `z_s` is the
  most expensive possible option
- it should be a later branch or ablation, not the next mainline step

### 4.1 What this recommendation means concretely

Under the recommended mainline:

- `teacher_cont` should still be treated as a **teacher-owned target** and kept
  detached with respect to teacher parameters
- but the continuation formula itself should remain a function of `z_s`, so the
  bridge side still receives gradient

In other words:

- teacher branch: detached
- continuation map wrt `z_s`: allowed

That is exactly why the affine/Jacobian-lite family fits the current branch.

### 4.2 What should happen to cached sources

Under the recommended mainline:

- `cached_exact` and current `cached_affine` should become fallback paths
- they should not remain the dominant continuation sources once online teacher is
  enabled

The intended priority becomes:

1. online-conditioned affine/Jacobian-lite continuation
2. cached exact or cached affine only when online path is unavailable
3. bootstrap last

### 4.3 Interaction with warp learning

This recommendation is also the most compatible with the current warp logic:

- current warp learning uses defect and teacher-aligned metrics computed from
  the same batch trajectory
- online-conditioned affine continuation reuses that same trajectory anchor and
  therefore introduces less new variance than a full teacher rollout from `z_s`

There is still a possible new bias:

- if the online continuation keeps too strong a clean-end sensitivity, warp may
  over-concentrate where the online affine correction is sharp

But that risk is already observable through current diagnostics:

- `entropy_q_phi`
- `kl_qD_qphi`
- `max_qphi_over_qbase`

So the current branch already has enough logging to watch that interaction.

## 5. Warmup and Gradient Interaction Under Online Teacher

### 5.1 Why current `eta=1.0` still creates a direct-only corner case

Current stage scheduling sets:

- warmup `eta = 1.0`

at
[src/dgtd/train_dgtd.py:165](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/train_dgtd.py:165).

Current DGTD continuation mixing is:

- `x_u_cont = eta * teacher_cont + (1 - eta) * bridge_cont`

at
[src/dgtd/defect.py:42](/home/gzwlinux/vscode/gitProject/DG-TWFD/src/dgtd/defect.py:42).

So if:

- `eta = 1.0`
- and `teacher_cont` is detached exact teacher output

then:

- `x_u_cont` has no student dependency
- `bridge_residual = x_u_cont - stopgrad(x_u_direct)` has no bridge gradient
- only the direct branch is trained

This is exactly the corner case already noted in the earlier verification.

### 5.2 Warmup strategy candidate 1: start `eta` below 1.0

Idea:

- do not start warmup at `eta=1.0`
- instead start with something like `eta_start in [0.5, 0.8]`

Effect:

- even exact detached teacher continuation leaves some `(1 - eta)` mass on
  `bridge_cont`
- bridge branch gets gradient from step 0

Pros:

- minimal change
- works for all continuation types

Risks:

- reduces the teacher anchor exactly when the branch is least trained
- may weaken stability if the student bridge is initially very poor

### 5.3 Warmup strategy candidate 2: keep strong teacher weight, but use
teacher-aligned affine residual injection

Idea:

- let the continuation remain strongly teacher-anchored
- but ensure the teacher continuation itself depends on `z_s`
- that means even `eta=1.0` does not kill bridge gradient

Effect:

- preserves a strong teacher target
- avoids the direct-only corner case without forcing lower `eta`

Pros:

- best matched to the recommended Plan B
- keeps the meaning of warmup as "teacher-dominant"
- avoids relying on weak student bridge early in training

Risks:

- requires the online continuation formula to be carefully stabilized
- if the local gain is too large, gradients may become noisy

### 5.4 Recommended warmup strategy

Recommended strategy: **candidate 2 as the main rule, plus a small nonzero
student bridge fraction as safety**.

Concretely, for the next design round the most stable policy is:

1. make online continuation itself student-sensitive, not pure detached exact
2. optionally avoid `eta=1.0` exactly; use a cap slightly below 1.0, such as
   `eta_max < 1`

Reason:

- the current branch already benefits from affine continuation because it keeps
  bridge gradients alive
- reproducing that property in the online mainline is the least disruptive
  choice
- lowering `eta` alone is a weaker fix, because it solves the gradient issue by
  weakening the teacher rather than by fixing the continuation geometry

## 6. What Exact Runtime Evidence Is Still Missing

The following runtime evidence is still missing before a final online-teacher
patch should be designed:

1. one-batch timing breakdown for:
   - online trajectory generation
   - current forward/backward
   - any prototype online continuation call
2. actual per-step or per-epoch `continuation_sources` over more than one smoke
   batch
3. a short run showing:
   - `train_loss`
   - `val_loss`
   - `direct_teacher_error`
   - `bridge_teacher_error`
   - `direct_bridge_gap`
   - `entropy_q_phi`
   - `kl_qD_qphi`
4. `q_phi` and `q_D` evolution over at least several dozen updates after warmup
5. one-batch distribution of:
   - sampled `t`
   - sampled `s`
   - sampled `u`
   - `teacher_rel_error`
   - exact-mask hit rate
6. real throughput numbers for online-teacher smoke:
   - images/sec
   - seconds/epoch
   - max GPU memory
7. actual teacher discrete timestep mapping for a sample of `s` and `u`, to see
   how coarse `index(s)` and `index(u)` are under the current 128-step DDIM
   teacher
8. evidence for whether the teacher scheduler's notion of noise is well aligned
   with current DGTD `sigma(t) = 1 - t`

None of these are conceptual requests. All of them can be gathered by a short
server run and a few printed tensors/statistics.

## 7. Exact Questions Back to the Project Owner

These are the exact follow-up questions I would send back before drafting the
next implementation patch.

1. Please provide the latest server `train.jsonl` tail for a run with:
   - `dgtd.disable_online_teacher=false`
   - at least 10 to 20 train updates, not just a single smoke step
2. Please provide one verbose training line or JSON payload showing:
   - `continuation_sources`
   - `eta`
   - `teacher_rel_error`
   - `direct_teacher_error`
   - `bridge_teacher_error`
   - `direct_bridge_gap`
3. Please provide one-batch profiling for the current online-teacher train step:
   - trajectory materialization time
   - forward/backward time
   - peak GPU memory
4. Please confirm whether the desired mainline definition of "online teacher"
   means:
   - online trajectory only
   - or true online continuation from arbitrary `z_s`
5. Please provide the exact server command and resolved config used for the run
   that produced:
   - `online_teacher_data=true`
   - `continuation_sources.online=0.0`
6. Please print, for one real batch, a small sample of:
   - `s`
   - `u`
   - `teacher_rel_error`
   - whether each sample was tagged `cached_exact` or `cached_affine`
7. Please provide the teacher scheduler mapping evidence for one batch:
   - `scheduler.timesteps[:16]`
   - `index(s)` and `index(u)` for a few sampled pairs
   - any teacher-side sigma or alpha quantities if available
8. Please confirm whether the project owner wants the next mainline to optimize
   for:
   - maximum fidelity to teacher dynamics
   - or minimum runtime cost with stable bridge gradients

## Bottom Line

The current branch is not blocked on the online-teacher **data** path anymore.
It is blocked on the online-teacher **continuation** path.

The best next-step mainline is not a full expensive teacher `local_flow`
rollout. It is an online-conditioned affine/Jacobian-lite continuation that:

- makes `source=online` real
- keeps teacher targets detached
- preserves bridge gradients even under strong teacher weighting
- and treats current cached paths as fallback rather than primary supervision
