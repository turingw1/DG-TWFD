# Evaluation System Plan

## 1. Goal
Make evaluation a first-class subsystem. It must not remain split across `sample.py`, `scripts/profile_infer.py`, and `scripts/preview_samples.py`.

## 2. Evaluator Module Structure

```text
src/dgfm/evaluators/
├── registry.py
├── runner.py
├── checkpoint_sweep.py
├── metrics/
│   ├── fid.py
│   ├── kid.py
│   ├── inception_score.py
│   └── summary.py
├── sampling/
│   ├── generate.py
│   ├── grids.py
│   ├── fixed_seed.py
│   └── cache.py
├── reporting/
│   ├── tables.py
│   ├── json_export.py
│   └── markdown_export.py
└── protocols/
    ├── baseline.py
    └── few_step.py
```

## 3. Metric Computation Pipeline
1. Resolve checkpoint(s).
2. Resolve evaluation protocol: step counts, sample count, fixed seeds, dataset stats.
3. Generate or load cached samples per `(checkpoint, step_count, seed_policy)`.
4. Compute metrics on cached sample set.
5. Save per-run metrics and update aggregate summary table.

## 4. Sampling Pipeline
Inputs:
- checkpoint
- model config
- solver config
- step counts
- sample count
- seed policy

Outputs per `(checkpoint, nfe)`:
- `samples/` image files or tensor archive
- `grid.png`
- `step_stats.json`
- `sampler_profile.json`

## 5. Fixed-Seed Visualization Workflow
Mandatory qualitative protocol:
- one global fixed seed list for all experiments
- identical batch size and ordering across checkpoints
- per-step-count preview grid
- side-by-side export for checkpoint comparison

Mandatory files:
- `fixed_seed_manifest.json`
- `grid_steps1.png`
- `grid_steps2.png`
- `grid_steps4.png`
- `grid_steps8.png`
- `grid_steps16.png`

## 6. Metric Output Format
Per checkpoint per step count:
- `metrics.json`
- `metrics.csv`

Aggregate experiment report:
- `summary.csv`
- `summary.md`
- `best_by_fid.json`

Suggested `metrics.json` schema:

```json
{
  "experiment": "fm_cifar10_baseline_v1",
  "checkpoint": "best.pt",
  "step_count": 16,
  "num_samples": 50000,
  "fid": 0.0,
  "kid": null,
  "inception_score": null,
  "sampler": {
    "type": "ode",
    "method": "midpoint"
  },
  "cache_hit": true
}
```

## 7. Per-Checkpoint Evaluation Protocol
Mandatory:
- `best.pt`
- `last.pt`

Optional sweep:
- every `N` epochs
- explicit user-specified checkpoint list

CLI shape:

```bash
python scripts/run_eval.py \
  --config <experiment-config> \
  --checkpoint <ckpt> [--checkpoint <ckpt> ...] \
  --steps 1 2 4 8 16
```

## 8. Multi-Step Evaluation Protocol
Phase 1 mandatory NFE set:
- `1`
- `2`
- `4`
- `8`
- `16`

All metrics and grids must be indexed by `step_count`.

## 9. Mandatory vs Optional Metrics

### Mandatory
- FID
- sample grid visualization
- saved generated samples or cached tensor archive
- per-checkpoint evaluation
- multi-step comparison table

### Optional
- KID
- Inception Score
- precision/recall for generative models
- sampler latency and NFE profiling

## 10. Mandatory Visualizations
- fixed-seed sample grid per step count
- best-vs-last checkpoint side-by-side grid
- FID-vs-step-count curve
- optional FID-vs-checkpoint curve

## 11. Caching Strategy
Cache key must include:
- checkpoint hash or path
- resolved config hash
- sampler method
- step count
- sample count
- seed policy
- dataset stats version

Cache location:
- `/cache/Zhengwei/dgfm_eval/<exp>/cache/<cache_key>/`

Do not recompute if `metrics.json` and sample manifest already exist for the same cache key.

## 12. Reusable Code Sources

| Source | Borrow | Rewrite |
|---|---|---|
| Current repo (`sample.py`, `profile_infer.py`, `preview_samples.py`) | sample dumping, step stats, preview generation | centralized evaluator orchestration |
| `flow_matching` | solver abstraction, time-grid-driven sampling, evaluation direction | FID/reporting/checkpoint sweep infra |
| `ctm` | multi-NFE evaluation mindset, reporting discipline | training-loop-bound evaluation code |

### Best reuse choices
- Borrow `flow_matching` solver interface for continuous baseline evaluation.
- Borrow current repo's preview/grid logic only as utility functions.
- Borrow CTM's idea of evaluating multiple step counts from one centralized pass, not its coupled implementation.

## 13. Final Evaluation Entry Points
Required scripts:
- `scripts/run_eval.py`: checkpoint evaluation
- `scripts/run_sample.py`: qualitative sampling only
- `scripts/run_report.py`: aggregate report generation

The user should not call metric internals directly.
