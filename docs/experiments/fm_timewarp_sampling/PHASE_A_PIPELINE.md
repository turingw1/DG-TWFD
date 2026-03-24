# fm_timewarp_sampling Phase A Pipeline

Phase A evaluates whether sampling-time warp alone improves fixed-step generation quality.
It does not modify training. It reuses an existing baseline checkpoint.

## 0. Activate

```bash
cd ~/workspace/Zhengwei/DG-TWFD
source scripts/experiments/activate_fm_timewarp_sampling.sh phase_a v1
conda activate consistency
```

Default baseline checkpoint:

```bash
echo $BASELINE_CKPT
```

Override it if needed:

```bash
export BASELINE_CKPT=/cache/Zhengwei/dgfm_runs/fm_cifar10_baseline_v2/checkpoints/best.pt
```

## 1. Warp Strategies

Phase A compares:

- `uniform`
- `source_dense_power2`
- `data_dense_power2`
- `random_dirichlet`

Interpretation:

- `source_dense_power2`: denser near the source/noise side
- `data_dense_power2`: denser near the data side
- `random_dirichlet`: random monotone grid with fixed seed

## 2. Fast Smoke

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_timewarp_sampling.py \
  --config $FM_TIMEWARP_CONFIG \
  --checkpoint $BASELINE_CKPT \
  --eval-root $FM_TIMEWARP_EVAL_ROOT/smoke \
  --steps 4 8 16 \
  --fid-samples 5000 \
  --fid-batch-size 128
```

## 3. Full Phase A Run

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_timewarp_sampling.py \
  --config $FM_TIMEWARP_CONFIG \
  --checkpoint $BASELINE_CKPT \
  --eval-root $FM_TIMEWARP_EVAL_ROOT \
  --steps 4 8 16 25 32
```

## 4. Outputs

```text
$FM_TIMEWARP_EVAL_ROOT/
├── uniform/
├── source_dense_power2/
├── data_dense_power2/
├── random_dirichlet/
└── reports/
    ├── summary.json
    ├── summary.csv
    ├── compact_summary.json
    ├── compact_summary.csv
    ├── best_by_strategy.json
    ├── best_overall.json
    ├── fid_vs_nfe.png
    ├── strategy_panel_steps16.png
    └── strategy_panel_steps16.pt
```

Per strategy and step:

- `metrics.json`
- `generated_stats.npz`
- `fixed_seed_grid.png`
- `fixed_seed_images/`
- `fixed_seed_samples.pt`
- `time_grid.pt`

## 5. Minimal Readout

```bash
cat $FM_TIMEWARP_EVAL_ROOT/reports/best_overall.json
cat $FM_TIMEWARP_EVAL_ROOT/reports/best_by_strategy.json
```

```bash
python - <<'PY'
import csv, os, pathlib
path = pathlib.Path(os.environ['FM_TIMEWARP_EVAL_ROOT']) / 'reports' / 'compact_summary.csv'
with path.open() as f:
    for row in csv.DictReader(f):
        print(row['strategy'], row['step_count'], row['nfe'], row['fid'], row['delta_fid_vs_uniform'])
PY
```

## 6. Decision Rule

Phase A is considered positive if at least one non-uniform strategy:

- beats `uniform` at the same step count
- and improves the fixed-seed qualitative panel consistently

If no strategy clearly beats `uniform`, do not proceed directly to learnable warp training.
