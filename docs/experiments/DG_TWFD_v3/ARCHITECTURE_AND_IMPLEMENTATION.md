# DG_TWFD_v3 Implementation Map

This is the concise active implementation map for current development. The
legacy long-form architecture note is archived at
`docs/archive/context_noise_2026-04-21/experiments/DG_TWFD_v3/ARCHITECTURE_AND_IMPLEMENTATION_legacy.md`.

## Active Thesis

`DG_TWFD_v3` trains an explicit map model with a unified DGTD residual and a
defect-guided monotone time warp. The intended evidence chain is:

`defect-guided time warping -> lower semigroup defect -> more stable reuse of
one checkpoint across step counts -> smoother quality-speed curve`.

## Time Convention

- `0.0` is noisy.
- `1.0` is clean.
- Triplets follow `0 <= t < s < u <= 1`.
- Clean dataloader images must be forward-noised before they are used as the
  noisy endpoint of an online teacher trajectory.

## Active Mainline

- Objective: `train.objective=dgtd_map`
- Trainer: `src/dgtd/train_dgtd.py`
- Teacher route: online teacher data + online trajectory anchors
- Continuation: affine / Jacobian-lite online continuation as the main residual
  source
- Fallback: cached continuation only when the online route cannot serve the
  local flow query
- Residual: symmetric half-stopgrad DGTD residual
- Warp: defect-guided density, implemented by `src/dgtd/warp.py`
- Sampling: `scripts/run_sample_dgtd.py` and `src/dgtd/sample_dgtd.py`
- Evaluation: `scripts/run_eval.py` through `src/dgfm/evaluators/common.py`

## Primary Configs

- Full run: `configs/experiment/dgtd_cifar10_v3.yaml`
- Smoke run: `configs/experiment/dgtd_cifar10_v3_smoke.yaml`
- EDM baseline eval: `configs/experiment/edm_cifar10_public_eval.yaml`

## Key Files

- `src/dgtd/train_dgtd.py`: training loop, online teacher data path, logging
- `src/dgtd/teacher.py`: online trajectory and cached fallback routing
- `src/dgtd/defect.py`: DGTD residual and defect/density target construction
- `src/dgtd/metrics.py`: sigma-aware weighting and detail metric helpers
- `src/dgtd/sigma.py`: centralized sigma/time conversion helpers
- `src/dgtd/warp.py`: monotone learned time-density warp
- `src/dgtd/sample_dgtd.py`: DGTD map rollout sampler
- `src/dgfm/evaluators/common.py`: metric dispatch and sample export path
- `scripts/analyze_dgtd_run.py`: returned server-run diagnosis

## Server Workflow

Use `docs/experiments/DG_TWFD_v3/EXPERIMENT_LOG.md` to choose an experiment and
activate it with:

```bash
source scripts/experiments/activate_fm_cifar10.sh <variant> <tag>
```

Then use `docs/experiments/DG_TWFD_v3/PIPELINE.md` for train, sample, eval, and
resume commands. Formal experiments run on the server; local work is limited to
focused edits and lightweight checks.

## Default Debug Order

1. Check the current config under `configs/experiment/`.
2. Inspect the relevant code path in `src/dgtd/` or `src/dgfm/`.
3. Use returned server logs and `scripts/analyze_dgtd_run.py` outputs.
4. Update the smallest affected code/docs surface.
5. Run local `py_compile` or targeted tests when feasible.
6. Commit the completed patch.

Do not read `docs/archive/` or untracked `refs/` unless the user asks for a
specific archived/reference detail.
