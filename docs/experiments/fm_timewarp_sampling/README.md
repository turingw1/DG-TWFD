# fm_timewarp_sampling

This directory contains the experiment documentation for the `fm_timewarp_sampling` line.
Keep all future time-warp experiment notes here instead of the repository root.

## Version Plan

- `phase_a_v1`
  - sampling-only time-warp study
  - fixed handcrafted warp strategies
  - no training changes
  - compare `uniform`, `source_dense_power2`, `data_dense_power2`, `random_dirichlet`

- `phase_b_v1`
  - learnable warp module
  - training-time warp parameterization
  - same evaluation protocol as Phase A

- `phase_c_v1`
  - optional teacher-coupled warp optimization
  - reserved for later

## Output Contract

Each experiment should produce:

- per-strategy, per-step `metrics.json`
- full `summary.csv/json`
- compact comparison table
- `fid_vs_nfe.png`
- fixed-seed strategy comparison panel

## Naming

Use:

- config: `configs/experiment/fm_timewarp_sampling_<phase>.yaml`
- activation: `scripts/experiments/activate_fm_timewarp_sampling.sh`
- doc: `docs/experiments/fm_timewarp_sampling/PHASE_<X>_PIPELINE.md`
