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
- Run tag: `edm_first_cifar10_prior_fullstack_timewarp_v11a_from_step1750`.
- Config: `experiments/edm_first/configs/cifar10_edm_map_prior_fullstack_timewarp_12h.yaml`.
- Initialization checkpoint:
  `runs/edm_first_cifar10_onestep_msdefect_e504a_resume_from1250/checkpoints/step1750.pt`.
- Important detail: this run loads the step1750 student weights but does not
  resume optimizer state or global step count because the config sets
  `resume_optimizer: false` and `resume_step: false`.
- Live backup:
  `/temp/Zhengwei/projects/DG-TWFD/critical/runs/edm_first_cifar10_prior_fullstack_timewarp_v11a_from_step1750`.
- Milestone backups:
  `/temp/Zhengwei/projects/DG-TWFD/critical/eval/edm_first_cifar10_prior_fullstack_timewarp_v11a_from_step1750_step*`.

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

The success criterion for v12a is not only lower FID@1. The first useful
outcome is a better policy curve than v11a step6750, especially preserving
2-step identity behavior while improving or at least holding 4/8/16.

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

The current training is expensive: latest logs imply roughly `17.8s/step`, so
one 250-step checkpoint is about 74 minutes before evaluation overhead. With
`max_seconds: 28800`, the active run is more like an 8-hour continuation window
than a 50000-step run.

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

1. Do not continue the same v11a objective blindly from step8750. Endpoint
   improves, but multi-step quality peaked earlier.
2. Treat the timewarp objective as the next bottleneck. The learned clock is now
   useful for `4/8/16` but harmful for `2`; add step-budget-aware pressure or
   per-step schedule selection before claiming a general timewarp advantage.
3. Preserve both key checkpoints: step6750 for composition and step8750 for
   endpoint. The next run should branch from step6750 if the goal is few-step
   quality, or use step8750 only for endpoint-focused comparison.
4. Keep v1.1 project backups active under
   `/temp/Zhengwei/projects/DG-TWFD/critical`, and keep Codex session backups
   under `/temp/Zhengwei/projects/DG-TWFD/codex`.
