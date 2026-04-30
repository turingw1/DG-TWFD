# DG-TWFD v3 Experiment Evaluation

This is the high-signal evaluation memo for DG-TWFD v3. It should be updated
only when a result changes the experimental decision, reveals a blocker, or
changes the next full-stack direction. It is not a step-by-step run ledger.

## Update Policy

- Update after milestone FID evaluations, corrective runtime actions, or
  algorithmic conclusions.
- Keep raw artifacts in `eval/`, `runs/`, `results/`, and the project-isolated
  temp tree `/temp/Zhengwei/projects/DG-TWFD/critical`.
- Record paths and decision-relevant metrics here, not large files.

## Current Active Track

- Track: EDM-first CIFAR-10 full-stack prior map with learned timewarp.
- Run tag: `edm_first_cifar10_prior_fullstack_timewarp_v15_multimid_from_step10750`.
- Config:
  `experiments/edm_first/configs/cifar10_edm_map_prior_fullstack_timewarp_v15_multimid.yaml`.
- Initialization checkpoint:
  `runs/edm_first_cifar10_prior_fullstack_timewarp_v14_guarded_from_step7750/checkpoints/step10750.pt`.
- Important detail: v15 branches from v14's best overall budget-policy
  checkpoint. It adds preservation at `u=0.25/0.5/0.75` to target the remaining
  low/mid-budget composition gap while reducing LR.
- Live backup:
  `/temp/Zhengwei/projects/DG-TWFD/critical/runs/edm_first_cifar10_prior_fullstack_timewarp_v15_multimid_from_step10750`.
- Milestone backups:
  `/temp/Zhengwei/projects/DG-TWFD/critical/eval/edm_first_cifar10_prior_fullstack_timewarp_v15_multimid_from_step10750_step*`.

## Latest Decision Metrics

FID uses 2048 generated samples for the active watcher.

| checkpoint | FID@1 | FID@2 | FID@4 | FID@8 | FID@16 | decision signal |
|---|---:|---:|---:|---:|---:|---|
| e504a step250 baseline | 177.890 | 46.286 | 49.294 | 70.911 | 86.567 | threshold reference |
| e504a step1250 restored | 135.003 | 44.315 | 52.436 | 81.607 | 96.386 | 1-step improved, multi-step regressed |
| resume-from1250 step250 | 126.059 | 44.376 | 53.666 | 83.676 | 98.459 | 1-step still improving |
| resume-from1250 step500 | 117.980 | 44.656 | 54.701 | 85.333 | 100.047 | 1-step improving, composition worsening |
| resume-from1250 step1750 | 91.325 | 46.693 | 59.096 | 90.190 | 103.199 | endpoint nearly at target, multi-step still regressed |

Primary endpoint target remains `FID@1 <= 88.945`, which is a 50% reduction
from the e504a step250 baseline. The last endpoint-only checkpoint,
resume-from1250 step1750, reached `91.325`, close to the endpoint target but
with worsening multi-step FID. This is why the active track is now full-stack
composition training rather than more endpoint-only continuation.

## Full-Stack Timewarp Launch

The v11a run started on 2026-04-28 from the endpoint checkpoint at step1750.
It changes the objective from direct endpoint matching to a full-stack prior
objective:

- `sigma_max -> sigma_s` is supervised by a one-step teacher transition.
- `sigma_max -> 0` keeps direct endpoint supervision.
- `sigma_max -> sigma_s -> 0` is trained against the same high-quality teacher
  rollout endpoint, so composed inference is no longer only a detached
  diagnostic.
- direct-vs-bridge raw defect is used as the timewarp density signal.

Initial training signal is healthy enough to continue the long run:

| follow-up step | loss | match | anchor | bridge | defect | timewarp qmax |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.164855 | 0.019213 | 0.026507 | 0.086160 | 0.073551 | 1.0000 |
| 25 | 0.166869 | 0.068927 | 0.028375 | 0.067120 | 0.059379 | 1.0004 |
| 50 | 0.157794 | 0.040373 | 0.025856 | 0.072934 | 0.066003 | 1.0008 |

The main interpretation is not that step50 proves image quality yet. It shows
the full-stack objective is numerically stable, preserves the direct endpoint
scale, gives bridge/defect a non-trivial signal several orders of magnitude
larger than the old normalized defect, and successfully starts updating the
learned timewarp. The first decision-quality image metric should come from the
step250 learned-warp vs identity-clock evaluation.

The step250 evaluation completed and gives the first decision-quality signal:

| checkpoint / clock | FID@1 | FID@2 | FID@4 | FID@8 | FID@16 |
|---|---:|---:|---:|---:|---:|
| prior endpoint step1750 | 91.325 | 46.693 | 59.096 | 90.190 | 103.199 |
| full-stack v11a step250 auto warp | 91.244 | 38.037 | 47.949 | 77.666 | 92.325 |
| full-stack v11a step250 identity | 91.244 | 37.970 | 47.877 | 77.568 | 92.277 |

The important result is that full-stack composition training improves the
multi-step curve while preserving endpoint quality. The timewarp result is not
yet a learned-warp win: `max_qphi_over_qbase` is only `1.004` at step250, so
the learned grid is still effectively identity. This is the correct next
optimization target. The run should continue so defect statistics can create a
meaningfully non-uniform density; the evaluation watcher is already set to
compare auto warp against identity again at step500.

The 5-hour supervision window confirms that this was not a transient early
effect:

| eval step | FID@1 | FID@2 | FID@4 | FID@8 | FID@16 | qmax near window |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 94.337 | 34.858 | 36.965 | 58.766 | 72.993 | 1.020 |
| 1750 | 90.969 | 36.391 | 32.514 | 46.440 | 56.961 | 1.035 |
| 2500 | 87.346 | 38.694 | 32.064 | 39.203 | 46.004 | 1.052 |
| 3250 | 85.789 | 38.741 | 31.897 | 34.647 | 39.106 | 1.070 |
| 4000 | 84.491 | 37.870 | 31.609 | 31.562 | 34.770 | 1.086 |

The run later continued to eval step6750 with `76.251 / 36.578 / 30.516 /
26.536 / 28.459`. The current bottleneck has shifted: full-stack composition
training is working, but learned timewarp is not uniformly better than identity.
By step6750, auto warp improves FID@4/8/16 over identity by about
`1.204/0.414/0.414`, but it worsens FID@2 by about `1.797`. The density is
moving (`qmax` about `1.13` in the train log), yet it is still too weak and not
step-budget aware enough. The next useful optimization should target the
timewarp objective or evaluation schedule, not replace the full-stack student
training.

The 12-hour v11a run ended normally at train step8878 due `max_seconds`.
The final evaluated checkpoint is step8750:

| checkpoint | FID@1 | FID@2 | FID@4 | FID@8 | FID@16 |
|---|---:|---:|---:|---:|---:|
| v11a step8750 auto warp | 68.851 | 43.673 | 32.686 | 28.092 | 30.602 |
| v11a best per column | 68.851 @8750 | 35.051 @1000 | 30.516 @6750 | 26.536 @6750 | 28.459 @6750 |

The final endpoint is much stronger, but late training regresses the 2-step
and the best multi-step checkpoints. The best all-around composition checkpoint
is currently step6750, while step8750 is the endpoint-specialized checkpoint.
The final auto-vs-identity deltas at step8750 are `0.000 / +3.210 / -2.417 /
-0.752 / -0.927`: learned warp is now clearly beneficial for 4/8/16, but
harmful for 2. This is the strongest evidence so far that a single global
timewarp density is the wrong abstraction for all step budgets.

Follow-up check at 2026-04-28 14:54 +08:00 confirms that the v11a run and its
step8750 auto/identity evaluations have finished, and the GPU is idle. This
turns the active decision from supervision to checkpoint selection and next
experiment design. The correct interpretation is:

- v11a is a successful full-stack student experiment.
- v11a is not a sufficient timewarp solution, because its single global density
  creates a real 2-step regression while helping 4/8/16.
- step6750 should be preserved as the current best composition checkpoint.
- step8750 should be preserved as the endpoint-specialized checkpoint.
- the next main experiment should not be a blind continuation from step8750;
  it should branch from step6750 when optimizing few-step robustness.
- the next timewarp experiment should become step-budget aware, either through
  separate `q_phi_N` schedules or an explicit policy that uses identity/fixed
  schedule for 2-step and learned warp for 4/8/16.

## v12a Improvement Plan

The next run is `edm_first_cifar10_prior_fullstack_timewarp_v12a_from_step6750`.
It branches from v11a step6750, because that checkpoint is the best current
composition checkpoint. The endpoint-specialized step8750 checkpoint is kept as
an endpoint reference, but should not be the default initialization for
few-step robustness.

The v12a changes are intentionally narrow:

- Evaluation gains a `budget` warp policy: use identity below
  `budget_warp_min_steps=4`, and learned warp at 4/8/16. This directly tests
  the observed fact that learned warp helps 4/8/16 but hurts 2-step.
- Full-stack training can use `defect_grad_mode: bridge_to_direct`, so the
  direct endpoint branch is not pulled around by a weaker composed branch.
  Defect becomes a bridge consistency regularizer rather than a two-way tug.
- The warp density signal can use `composition_gap`: direct-vs-bridge gap plus
  teacher bridge error, instead of only raw direct/bridge disagreement. This
  makes the learned density respond to where composition is actually worse than
  the direct endpoint.
- The new timewarp target is flattened with `flatten_mix=0.20` and lower
  `beta=0.60`, preventing a single global density from over-specializing while
  budget-aware eval handles the known 2-step conflict.
- The launch config uses `batch_size=64` so the run can coexist with the
  independent CTM baseline revalidation without killing or pausing that track.

The success criterion for v12a is not only lower FID@1. The first useful
outcome is a better policy curve than v11a step6750, especially preserving
2-step identity behavior while improving or at least holding 4/8/16.

The first v12a milestone at step250 is positive enough to keep running. It was
evaluated with auto, identity, and budget policy at 2048 samples while the
independent CTM baseline revalidation was also using the GPU.

| v12a step250 mode | FID@1 | FID@2 | FID@4 | FID@8 | FID@16 |
|---|---:|---:|---:|---:|---:|
| auto learned warp | 76.821 | 36.969 | 30.308 | 26.215 | 27.338 |
| identity | 76.821 | 34.608 | 31.896 | 26.684 | 27.778 |
| budget policy | 76.821 | 34.608 | 30.308 | 26.215 | 27.338 |

This confirms the diagnosis from v11a: the learned warp is still harmful for
2-step (`+2.360` FID vs identity), but useful at 4/8/16 (`-1.588/-0.469/-0.440`
vs identity). The budget policy gives the intended curve by using identity for
2-step and learned warp from 4-step onward. Compared with the v11a step6750
auto checkpoint, v12a step250 is already slightly better at 4/8 and clearly
better at 16, while endpoint is essentially preserved within small-sample FID
noise. Training should continue; the next decision point is step500 and then
whether the 4/8/16 gains keep improving without reintroducing 2-step damage.

The step1000 milestone gives a stronger decision signal. It confirms that the
budget policy is not just a step250 artifact; it is the right inference policy
for the current single learned density.

| v12a step1000 mode | FID@1 | FID@2 | FID@4 | FID@8 | FID@16 |
|---|---:|---:|---:|---:|---:|
| auto learned warp | 74.370 | 39.810 | 30.392 | 26.037 | 27.831 |
| identity | 74.370 | 35.778 | 32.850 | 26.685 | 28.546 |
| budget policy | 74.370 | 35.778 | 30.392 | 26.037 | 27.831 |

The learned warp advantage over identity has widened to
`-2.458/-0.648/-0.715` FID at `4/8/16`, while the auto-warp penalty at 2-step
has also widened to `+4.032`. This is now a clean separation: timewarp is useful
for mid/high step budgets, but the low-budget path needs identity or a
separate learned policy. The endpoint has improved from v12a step250 to
step1000 (`76.821 -> 74.370`), but 2-step identity also drifted worse
(`34.608 -> 35.778`). The next iteration should therefore add explicit
low-step preservation pressure instead of increasing global warp strength.

Supervision through step8250 shows that v12a has split into two useful but
different checkpoints: late checkpoints are endpoint-specialized, while the
best policy curve for few-step sampling happened much earlier.

| v12a checkpoint | budget FID@1 | budget FID@2 | budget FID@4 | budget FID@8 | budget FID@16 | interpretation |
|---:|---:|---:|---:|---:|---:|---|
| 250 | 76.821 | 34.608 | 30.308 | 26.215 | 27.338 | best 2-step and 16-step |
| 500 | 75.961 | 34.937 | 30.259 | 26.134 | 27.511 | best 4-step |
| 1000 | 74.370 | 35.778 | 30.392 | 26.037 | 27.831 | early balanced policy |
| 2500 | 71.157 | 38.293 | 31.816 | 26.645 | 29.126 | endpoint improves, multi-step drifts |
| 5000 | 65.810 | 38.450 | 32.753 | 27.039 | 29.217 | drift plateau begins |
| 8000 | 61.872 | 36.668 | 31.511 | 26.029 | 27.645 | best complete endpoint and 8-step |
| 8250 | 61.686 | 36.659 | 31.483 | 25.961 | 27.542 | latest complete eval; 8-step improves |

The learned warp itself is still becoming more distinct: by step8250 it helps
identity by `-2.771/-0.562/-0.869` FID at `4/8/16`, but hurts 2-step by
`+3.928`. The direction is stable: learned timewarp is genuinely useful for
mid/high step budgets, while one global density remains wrong for the lowest
step budget. For a SOTA sprint, the current v12a late checkpoint should be
treated as an endpoint teacher/reference and a strong 8-step candidate, not as
the best 2/4-step sampler. The next branch should either start from the early
multi-step checkpoint family, or distill the late endpoint quality back into
the early policy curve with explicit 2-step and 4-step preservation.

The external baseline gap is still large. CTM 50k audit on CIFAR-10 reports
`1.743/1.617/1.830/2.101` at `1/2/4/8`, so v12a is not yet a competitive final
sampler. Its value for the SOTA sprint is diagnostic: it proves the project can
learn a non-trivial timewarp advantage over identity at selected budgets, and
it exposes exactly where the next architecture/objective must become
budget-conditioned.

## Training Signal Interpretation

As of 2026-04-27 20:50 +08:00, the resumed run is live and has reached step725
in the train log. The latest logged loss is `0.058607`, with match loss
`0.030704`, perceptual loss `0.111610`, and normalized defect
`1.234e-05`. The local minimum is still around step650 (`loss=0.054875`).
This shows the direct endpoint objective is still optimizing, but there is no
new milestone FID after step500 yet.

The FID trend is more important than loss here. The loss improvement correlates
with continued 1-step FID improvement, but it does not improve multi-step
rollouts. FID@4, FID@8, and FID@16 continue to worsen as the endpoint map gets
better. That is the key experimental fact.

## Algorithm Diagnosis

The current `prior_endpoint` objective is strongly biased toward direct
`sigma_max -> 0` endpoint matching:

- The teacher target is `x_u_ref`, generated by an 18-step EDM rollout from
  pure prior noise to clean output.
- The student direct output `x_u_direct` is trained against that endpoint with
  match loss and perceptual loss.
- In the prior-endpoint path, `x_s_ref` is currently set to `x_u_ref`; there is
  no separate teacher target for the intermediate sigma point.
- The bridge path computes `x_s` and `x_u_bridge` under `no_grad`, then uses a
  small defect penalty between `x_u_direct` and `x_u_bridge.detach()`.
- The observed normalized defect is around `1e-5`; with `defect_weight: 0.05`,
  its effective contribution is tiny compared with match and perceptual loss.

This explains the result pattern: the model becomes a better one-step endpoint
projector, but repeated application of that projector is not trained to compose.
The multi-step degradation is therefore not surprising and should not be
treated as a tuning accident.

## Module-Level Readout For Full Stack

- Objective: endpoint matching is doing what it was asked to do, but the
  current loss does not define a stable semigroup/composition constraint. This
  is the most important algorithmic blocker before treating the model as a
  few-step sampler.
- Defect signal: `norm_defect` stays around `1e-5`, so the weighted defect term
  is too small to carry schedule learning or bridge correction. It is useful as
  an instrument, but not yet as a primary training force.
- Timewarp: the config keeps `timewarp.enabled: false`; current results are an
  identity-clock endpoint baseline. Learned timewarp should be launched only
  after the defect source becomes meaningful or as a controlled diagnostic
  against identity, not as the next main bet by itself.
- Evaluation: FID@1 is improving, while FID@4/8/16 is worsening. The evaluator
  is therefore correctly exposing the composition failure; future full-stack
  experiments must keep `1/2/4/8/16` reporting and identity comparison.
- Runtime: effective speed is about `17.5s/step` overall, with the latest
  250-step window around `18.5s/step` during mixed server load. This makes
  250-step milestone evals the right decision cadence.

## Timewarp Implications

Turning on timewarp alone is unlikely to solve the full problem unless the
defect signal becomes more meaningful. Current timewarp support can warp the
sampled intermediate `u_mid` in prior-endpoint mode and update `q_phi` from
defect, but the defect is weak and indirect. If timewarp learns from this signal
as-is, it may mostly chase noise or endpoint-projection artifacts rather than a
true composition difficulty profile.

For the full-stack timewarp experiment, the best direction is:

1. Start from the best no-warp endpoint checkpoint only as initialization.
2. Add true intermediate supervision, not just endpoint supervision:
   train `sigma_max -> sigma_s` and `sigma_s -> 0` against teacher-derived
   intermediate/endpoint targets.
3. Make the defect term comparable and interpretable after intermediate targets
   exist; then timewarp can use defect as a meaningful density signal.
4. Always evaluate learned warp against identity clock at `1/2/4/8/16`; learned
   warp is useful only if it improves composition, not just 1-step quality.
5. Keep a step-budget-aware schedule/search baseline in the loop, because the
   identity Karras grid is currently exposing the composition failure.

User guidance on 2026-04-28 sets the next core goal: the full-stack experiment
must preserve the endpoint improvement while using match and defect to train
multi-step robustness. The desired behavior is that increasing the inference
step budget improves quality instead of degrading it. Timewarp is not an
optional add-on for this phase; one of the central goals is to demonstrate a
real learned-timewarp advantage over the identity clock and over fixed
schedules used during earlier evals.

The implementation implication is:

1. Keep direct endpoint supervision so FID@1 does not regress unnecessarily.
2. Add real intermediate teacher targets so the student learns both
   `sigma_max -> sigma_s` and `sigma_s -> 0` transitions.
3. Make the bridge defect a trainable composition signal, not only a weak
   detached diagnostic. Defect should expose where direct and composed paths
   disagree and should drive both student robustness and timewarp density.
4. Evaluate every milestone with identity and learned timewarp at
   `1/2/4/8/16`; the success criterion is not just better FID@1, but a curve
   where additional steps are neutral-to-beneficial.
5. Treat the learned timewarp schedule as a first-class artifact: save its
   density, entropy, max-density ratio, and per-bin defect statistics with each
   checkpoint so its advantage can be explained rather than only observed.

## Runtime And Supervision Notes

The current v12a training is compute-bound rather than memory-bound. During the
step1000 window the GPU reported about `39GB / 80GB` memory usage but `100%`
GPU utilization and roughly `385W` power. At the 2026-04-29 00:41 +08:00
supervision point it still reported about `28.8GB / 80GB` and `100%`
utilization. The lower memory footprint comes from `batch_size=64`, CIFAR-10
resolution, and coexistence with independent baseline/eval work; it is not
evidence that the run is idle. The run is configured for `max_seconds=43200`,
and every 250-step checkpoint is evaluated with auto, identity, and budget
policies.

User guidance on 2026-04-27 supersedes the earlier operational assumption:
the baseline experiment and the main DG-TWFD experiment are independent tracks.
The baseline is not to be killed, paused, or treated as a main-experiment
confounder merely because main training/evaluation is active.

The correct baseline policy is:

1. Monitor baseline progress and resource status, but do not terminate baseline
   processes during normal operation.
2. Do not use main train/eval activity as a default guard condition for
   baseline shutdown. `BASELINE_PAUSE_FOR_MAIN_TRAIN` and
   `BASELINE_PAUSE_FOR_MAIN_EVAL` should default to `0`.
3. Only intervene in baseline processes if the user explicitly asks, or if
   there is a concrete hard failure such as OOM, disk exhaustion, or corrupt
   output that threatens recoverability.
4. Keep baseline outputs and supervision evidence backed up independently under
   `/temp/Zhengwei/DG-TWFD-backups/experiment_evidence`.

The baseline guard that was already running before this policy correction may
still report an older pause flag in `STATUS.txt`. Do not kill that running
baseline just to refresh local shell variables; let it continue, and rely on the
corrected defaults for future launches.

## Current Recommendations

1. Treat v12a step10500 as the current best endpoint and budget-policy
   checkpoint for 1/4/8/16-step evaluation.
2. Preserve early v12a checkpoints as diagnostics for 2-step identity behavior,
   but do not treat them as better global checkpoints after the final v12a
   readout.
3. Make the next branch explicitly low-step aware: add preservation pressure
   for 2-step identity, keep 4-step from drifting, and consider a
   budget-conditioned or per-step warp head instead of one global learned
   density.
4. Use v12a step10500 as the v13 initialization, but add a targeted midpoint
   composition constraint rather than blindly extending the same objective.
5. Preserve both v11a key checkpoints: step6750 for composition and step8750
   for endpoint. v12a correctly branched from step6750.
6. Keep v1.1 project backups active under
   `/temp/Zhengwei/projects/DG-TWFD/critical`, and keep Codex session backups
   under `/temp/Zhengwei/projects/DG-TWFD/codex`.

## 2026-04-29 V12a Final Readout And V13 Plan

The v12a run completed naturally at train step10650 after the configured
12-hour wall-clock limit. The last fully evaluated checkpoint is step10500.
This changes the earlier step8250 interpretation: late training did not merely
improve the one-step endpoint. It also produced the best observed learned-warp
quality at 4, 8, and 16 steps under the budget policy.

Selected v12a FID-2048 checkpoints:

| train step | budget FID@1 | budget FID@2 | budget FID@4 | budget FID@8 | budget FID@16 | auto-minus-identity @2/@4/@8/@16 |
|---:|---:|---:|---:|---:|---:|---|
| 250 | 76.821 | 34.608 | 30.308 | 26.215 | 27.338 | +2.360 / -1.588 / -0.469 / -0.440 |
| 2500 | 71.157 | 38.293 | 31.816 | 26.645 | 29.126 | +5.600 / -3.308 / -0.724 / -0.956 |
| 8250 | 61.686 | 36.659 | 31.483 | 25.961 | 27.541 | +3.927 / -2.771 / -0.562 / -0.869 |
| 10500 | 59.246 | 34.881 | 29.997 | 24.914 | 26.055 | +3.391 / -2.351 / -0.486 / -0.779 |

The strongest v12a checkpoint is therefore step10500 for endpoint quality and
for learned-warp 4/8/16-step quality. The remaining failure mode is sharp and
diagnostic: the learned global warp is consistently harmful at 2 steps, while
the identity clock is still best for that budget. The current solution is the
budget policy, which uses identity below 4 steps and learned warp at 4+ steps.
That policy is not just an evaluation trick; it reflects a real incompatibility
between a single global density and different inference budgets.

NeurIPS-level bottleneck assessment:

1. The objective now learns useful composition for moderate budgets, but it
   under-constrains the exact midpoint composition used by 2-step identity
   inference. This is the most actionable local bottleneck.
2. The defect signal is informative for schedule density, not yet a sufficient
   guarantee of all-budget robustness. A global `q_phi` can improve 4/8/16
   while degrading 2-step because the loss samples a continuum of midpoints
   and does not explicitly protect the coarse identity midpoint.
3. The run is still far from CTM-level SOTA on absolute CIFAR-10 FID, so v12a
   should be treated as a validated diagnostic/training scaffold rather than a
   final sampler. The next gain must come from a better training constraint,
   not only from longer continuation.
4. Runtime is compute-bound. Extra objective terms must be justified by a clear
   bottleneck. The next experiment adds only one fixed midpoint preservation
   branch, targeted at the observed 2-step failure.

V13 is launched from v12a step10500 with a fixed midpoint preservation loss at
`u=0.5`. It keeps the v12 full-stack losses and learned timewarp, but adds
explicit pressure that the composed path `sigma_max -> sigma(0.5) -> 0` remains
close to the teacher endpoint, close to the teacher midpoint, and close to the
direct student endpoint. The intended acceptance criteria are:

1. Do not regress v12a step10500 budget FID@4/8/16 materially.
2. Improve or at least stabilize identity/budget FID@2 relative to v12a
   step10500.
3. Keep the learned timewarp advantage at 4+ steps negative versus identity.
4. If the extra fixed-midpoint branch slows progress without improving 2-step,
   the next direction should be budget-conditioned timewarp or a per-budget
   schedule head rather than more scalar reweighting.

## 2026-04-29 V13 Seven-Hour Supervision

The v13 midpoint-preservation run is a successful next iteration. It started
from v12a step10500 and was supervised for seven hourly intervals. At the
seventh-hour readout, the latest fully evaluated checkpoint is step4250.

Budget-policy FID-2048 progression:

| checkpoint | FID@1 | FID@2 | FID@4 | FID@8 | FID@16 |
|---:|---:|---:|---:|---:|---:|
| v12a step10500 | 59.246 | 34.881 | 29.997 | 24.914 | 26.055 |
| v13 step750 | 60.328 | 34.884 | 28.808 | 25.844 | 27.343 |
| v13 step1750 | 59.706 | 34.381 | 28.592 | 25.525 | 27.006 |
| v13 step3000 | 58.936 | 33.977 | 28.270 | 25.174 | 26.481 |
| v13 step3500 | 58.631 | 33.702 | 27.831 | 24.820 | 26.032 |
| v13 step4250 | 58.396 | 33.407 | 27.450 | 24.521 | 25.739 |

This is the first checkpoint family in the current sequence that improves over
the v12a final checkpoint at every reported budget. The key result is not only
the lower endpoint FID; it is that explicit midpoint preservation fixed the
previous 2-step identity/budget weakness while preserving and strengthening the
learned-timewarp advantage at 4/8/16 steps.

At step4250, learned timewarp is still harmful at 2 steps and beneficial at
4/8/16: auto-minus-identity is approximately `+2.76 / -1.84 / -0.48 / -0.69`
for `2/4/8/16`. This validates the budget policy as a necessary inference-time
adapter: identity remains the right low-step clock, while learned timewarp is
the right 4+ step clock. The remaining SOTA-facing bottleneck is absolute FID,
not the local schedule/composition failure that blocked v12a.

Next iteration guidance:

1. Let v13 continue to its time limit while the loss stays stable and the
   budget curve keeps improving.
2. Preserve v13 step3500 and step4250 as the first all-budget-improving
   checkpoints relative to v12a.
3. For the next code iteration, avoid increasing scalar loss complexity first.
   The more promising move is budget-conditioned timewarp, because the data
   repeatedly show that a single global warp cannot serve 2-step and 4+ step
   inference equally well.
4. Keep the `u=0.5` preservation branch in the default full-stack recipe until
   a more explicit per-budget schedule module replaces it.

## 2026-04-29 V13 Plateau Check At Step6000

The post-seven-hour continuation has not entered a confirmed plateau. The
latest evaluated checkpoint is v13 step6000, with budget-policy FID-2048:
`57.283 / 32.574 / 26.468 / 23.748 / 24.660` for `1/2/4/8/16`. This is a
large improvement over v13 step4250 and over v12a step10500.

Recent budget mean over 4/8/16:

| checkpoint | mean FID@4/8/16 |
|---:|---:|
| step4750 | 25.559 |
| step5000 | 25.469 |
| step5250 | 25.470 |
| step5500 | 25.244 |
| step5750 | 25.082 |
| step6000 | 24.959 |

The step5250 pause was a local fluctuation, not a dead zone: step5500,
step5750, and step6000 resumed clear improvement. The correct action is to
continue v13 to the configured time limit while the new plateau guard watches
for insufficient 4-eval progress. If the guard reports less than `0.05` mean
FID drop across the last four budget evaluations, the next move should be to
preserve the best v13 checkpoint and branch into budget-conditioned timewarp
rather than spend more time on scalar reweighting.

## 2026-04-29 V13 Final Readout And V14 Guarded Continuation

The v13 run completed normally at the configured wall-clock limit. The last
training log entry is step7828, but the last fully saved and evaluated
checkpoint is step7750. GPU was idle after completion, so the remaining v13
eval/hourly watcher sessions were waiting on a step8000 checkpoint that would
not appear.

Final v13 budget-policy FID-2048:

| checkpoint | FID@1 | FID@2 | FID@4 | FID@8 | FID@16 | mean FID@4/8/16 |
|---:|---:|---:|---:|---:|---:|---:|
| v12a step10500 | 59.246 | 34.881 | 29.997 | 24.914 | 26.055 | 26.989 |
| v13 step6000 | 57.283 | 32.574 | 26.468 | 23.748 | 24.660 | 24.959 |
| v13 step7000 | 56.600 | 32.076 | 26.139 | 23.359 | 24.192 | 24.563 |
| v13 step7500 | 56.598 | 32.080 | 25.978 | 23.255 | 24.013 | 24.415 |
| v13 step7750 | 56.569 | 32.050 | 26.009 | 23.229 | 23.920 | 24.386 |

The plateau concern is valid but not confirmed. The final one-step interval is
shallow, and FID@4 locally prefers step7500 by about `0.031`, but the aggregate
`4/8/16` curve still improves from step7000 through step7750. The run should
therefore be treated as a lower-slope continuation point, not as a dead end.

Decision:

1. Preserve v13 step7750 as the current best all-around checkpoint; preserve
   step7500 as the best isolated FID@4 checkpoint.
2. Do not keep sleeping on the completed v13 watcher stack; that only monitors
   an idle GPU.
3. Start v14 from v13 step7750 with a lower LR (`4e-7`), slightly stronger
   bridge/defect/preserve pressure, the same budget inference policy, and a
   final-checkpoint evaluation guard. This is less risky than a full
   budget-conditioned timewarp rewrite while the curve is still descending.
4. If v14 fails to beat the v13 step7750 `4/8/16` mean after four evaluated
   checkpoints, the next algorithmic move should be budget-conditioned
   timewarp/per-budget schedule heads, not further scalar loss tuning.

The code now saves a final checkpoint when the wall-clock limit stops training
between save intervals, and the eval watcher can evaluate that final checkpoint
when the train tmux session exits. This directly addresses the v13 issue where
the final logged step had no evaluable checkpoint.

## 2026-04-29 V14 Two-Hour Supervision

The v14 guarded continuation is not wasting compute. At the first two-hour
supervision point, training is live around step2400 with GPU utilization at
100%. The completed budget-policy evaluations through step2000 show that the
lower-LR guarded continuation improves endpoint, 4-step, 8-step, and 16-step
quality over v13 step7750, while 2-step remains the main regression to monitor.

Budget-policy FID-2048:

| checkpoint | FID@1 | FID@2 | FID@4 | FID@8 | FID@16 | mean FID@4/8/16 |
|---:|---:|---:|---:|---:|---:|---:|
| v13 step7750 | 56.569 | 32.050 | 26.009 | 23.229 | 23.920 | 24.386 |
| v14 step250 | 54.536 | 32.688 | 26.370 | 22.262 | 23.803 | 24.145 |
| v14 step1000 | 54.632 | 32.760 | 26.222 | 22.177 | 23.650 | 24.017 |
| v14 step1500 | 54.110 | 32.437 | 25.909 | 22.048 | 23.424 | 23.793 |
| v14 step2000 | 54.443 | 32.592 | 25.773 | 21.996 | 23.361 | 23.710 |

Interpretation: v14 successfully extends the v13 gains and pulls the
`4/8/16` mean down by about `0.676` FID from the previous best. The persistent
2-step weakness confirms the earlier diagnosis that low-budget inference needs
a budget-conditioned schedule or extra low-step preservation; however, because
the high-budget curve is still descending and the 2-step regression is not
exploding, the correct action is to continue the current run and reassess on a
two-hour cadence.

## 2026-04-29 V14 Four-Hour Supervision

The second two-hour checkpoint strengthens the v14 decision. Training is live
around step4200, GPU utilization remains near 100%, and completed budget
evaluations through step4000 show continued improvement without a plateau.

Recent v14 budget-policy FID-2048:

| checkpoint | FID@1 | FID@2 | FID@4 | FID@8 | FID@16 | mean FID@4/8/16 |
|---:|---:|---:|---:|---:|---:|---:|
| v14 step2250 | 54.162 | 32.470 | 25.659 | 21.930 | 23.275 | 23.621 |
| v14 step2750 | 53.886 | 32.336 | 25.532 | 21.823 | 23.176 | 23.510 |
| v14 step3500 | 53.871 | 32.238 | 25.512 | 21.735 | 23.063 | 23.436 |
| v14 step4000 | 53.618 | 32.115 | 25.446 | 21.656 | 22.975 | 23.359 |

Relative to v13 step7750, v14 step4000 improves FID@1 by about `2.95`,
FID@4 by `0.56`, FID@8 by `1.57`, and FID@16 by `0.95`. The 2-step budget
metric is now only about `0.065` worse than v13, so the early low-step
regression is mostly recovering. The current action remains continue; do not
change the objective mid-run while the curve is still descending.

## 2026-04-30 V14 Final Readout And V15 Plan

The v14 guarded continuation completed normally at the 12-hour wall-clock
limit. The final-checkpoint guard worked: training stopped after step10801,
that non-interval checkpoint was saved, and the eval watcher evaluated it
before exiting. GPU was idle after completion.

Best v14 budget-policy FID-2048:

| metric | best checkpoint | value |
|---|---:|---:|
| FID@1 | step10000 | 52.296 |
| FID@2 | step10000 | 31.179 |
| FID@4 | step10500 | 24.158 |
| FID@8 | step10750 | 20.665 |
| FID@16 | step10750 | 21.678 |
| mean FID@4/8/16 | step10750 | 22.167 |
| mean FID@2/4/8/16 | step10750 | 24.436 |

The selected handoff checkpoint is v14 step10750, because it is the best
overall multi-step checkpoint and nearly ties the final checkpoint. Compared
with v13 step7750, it improves budget FID by about `4.14 / 0.81 / 1.85 /
2.56 / 2.24` at `1/2/4/8/16`. Learned timewarp remains useful at 4+ steps:
at step10750 it improves identity by about `0.98 / 0.27 / 0.25` FID at
`4/8/16`, while budget policy correctly keeps identity for 2-step.

The remaining bottleneck is no longer a collapse or dead zone. It is a slower
low/mid-budget composition limit: late v14 still improves, but the final
four-eval mean drop is below the plateau guard threshold. The next run should
not simply continue the same scalar recipe. V15 branches from v14 step10750,
lowers LR, and uses multi-midpoint preservation at `u=0.25/0.5/0.75`, aligning
training pressure with the 4-step identity grid and the learned 4+ step warp
grid. The acceptance criterion is to beat v14 step10750's `4/8/16` mean while
not regressing FID@2 beyond the small-sample noise band.

## 2026-04-30 Documentation Audit And V15 Supervision Repair

The core documentation now covers every decision-relevant stage of the main
experiment:

1. e504a / resume-from1250 established the endpoint target and showed that a
   pure one-step objective was insufficient for robust multi-step inference.
2. v11 full-stack timewarp introduced the first combined endpoint, defect,
   match, bridge, and timewarp training path.
3. v12a added budget-aware inference selection and clarified that learned
   timewarp is beneficial at 4+ steps while identity time remains safer at
   2-step.
4. v13 added midpoint preservation and turned the previous fragile curve into
   a stable but lower-slope continuation.
5. v14 guarded continuation confirmed that the method was still learning:
   best budget FID-2048 reached `52.424 / 31.240 / 24.159 / 20.665 / 21.678`
   at `1/2/4/8/16` from step10750.
6. v15 branches from v14 step10750 and tests whether multi-midpoint
   preservation at `u=0.25/0.5/0.75` better matches the low/mid-budget
   composition bottleneck.

Baseline status remains isolated in `BASELINE_STATUS.md` and
`BASELINE_REPORT_CN.md`; it is not part of the main training process and should
not be killed during DG-TWFD supervision.

Operational correction: v15 training continued healthily, but the first
eval/backup/2h watcher stack exited after the step750 budget evaluation. This
was corrected by rerunning the v15 launcher, which reused the live training
session and restarted `v15_fullstack_tw_eval_watch`, `v15_fullstack_tw_backup`,
and `v15_fullstack_tw_2h`.

Early v15 signal is positive enough to continue. At step750, budget FID-2048 is
`51.139 / 30.423 / 22.619 / 20.851 / 21.306` at `1/2/4/8/16`, already beating
the v14 step10750 handoff at `1/2/4/16` and nearly tying 8-step. The learned
warp remains beneficial at 4+ steps, while budget policy keeps 2-step on the
identity clock. The current risk is not wasted training compute; it is making
sure the restarted eval watcher catches up from the live checkpoint stream.
The project backup watcher was also changed to keep checkpoint backups bounded
and to avoid syncing regenerable eval tensors by default; metrics, reports, and
fixed-seed preview images remain backed up.

## 2026-04-30 V15 Step10000 Positive Signal

The restarted eval watcher has caught up to a decision-relevant checkpoint.
V15 step10000 budget-policy FID-2048 is:

| checkpoint | FID@1 | FID@2 | FID@4 | FID@8 | FID@16 | mean FID@4/8/16 |
|---:|---:|---:|---:|---:|---:|---:|
| v14 step10750 | 52.424 | 31.240 | 24.159 | 20.665 | 21.678 | 22.167 |
| v15 step750 | 51.139 | 30.423 | 22.619 | 20.851 | 21.306 | 21.592 |
| v15 step10000 | 49.361 | 29.519 | 21.479 | 20.008 | 20.355 | 20.614 |

This is the strongest evidence so far that multi-midpoint preservation is
solving the low/mid-budget composition bottleneck rather than merely extending
v14. Relative to the v14 handoff, v15 step10000 improves FID by about
`3.06 / 1.72 / 2.68 / 0.66 / 1.32` at `1/2/4/8/16`, and improves the
`4/8/16` mean by about `1.55`.

The timewarp-specific readout still supports the budget policy: learned warp
improves identity by about `0.83 / 0.17 / 0.12` at `4/8/16`, while auto warp
is still worse than identity at 2-step by about `1.56`. The correct inference
path remains identity for 1/2-step and learned timewarp for 4+ steps.
