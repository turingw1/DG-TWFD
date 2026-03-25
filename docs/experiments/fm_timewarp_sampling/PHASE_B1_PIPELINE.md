# fm_timewarp_sampling Phase B1 Pipeline

Phase B1 does not retrain the flow model.
It optimizes a scalar `gamma` inside the `data_dense_power@gamma` sampling family and selects the best warp by a weighted few-step FID objective.

Prerequisite:
- a trained baseline checkpoint must already exist
- if it does not, complete [FLOW_MATCHING_BASELINE_PIPELINE.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/FLOW_MATCHING_BASELINE_PIPELINE.md) first

## 0. Activate

```bash
cd ~/workspace/Zhengwei/DG-TWFD
source scripts/experiments/activate_fm_timewarp_sampling.sh phase_b1 v1
conda activate consistency
```

Default baseline checkpoint:

```bash
echo $BASELINE_CKPT
```

## 1. Search Objective

Phase B1 searches over:

```text
t = 1 - (1 - u)^gamma
```

Default optimization target:

- objective steps: `8, 16`
- objective weights: `0.5, 0.5`

This means the selected warp minimizes:

```text
0.5 * FID@8 + 0.5 * FID@16
```

## 2. Fast Smoke

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_timewarp_search.py \
  --config $FM_TIMEWARP_CONFIG \
  --checkpoint $BASELINE_CKPT \
  --eval-root $FM_TIMEWARP_EVAL_ROOT/smoke \
  --steps 4 8 16 \
  --fid-samples 5000 \
  --fid-batch-size 128
```

## 3. Full Phase B1 Run

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_timewarp_search.py \
  --config $FM_TIMEWARP_CONFIG \
  --checkpoint $BASELINE_CKPT \
  --eval-root $FM_TIMEWARP_EVAL_ROOT \
  --steps 4 8 16 25 32
```

## 4. Outputs

```text
$FM_TIMEWARP_EVAL_ROOT/
├── uniform/
├── data_dense_power2/
├── data_dense_power_at_1p1000/
├── data_dense_power_at_1p2000/
├── ...
└── reports/
    ├── search_trace.jsonl
    ├── gamma_summary.json
    ├── gamma_summary.csv
    ├── compact_search_results.json
    ├── compact_search_results.csv
    ├── best_gamma.json
    ├── objective_vs_gamma.png
    ├── fid_vs_gamma.png
    ├── best_gamma_panel_steps8.png
    ├── best_gamma_panel_steps16.png
    └── ...
```

## 5. Logging

Detailed log:

- `reports/search_trace.jsonl`
- one line per evaluated `(gamma, step_count)`
- includes the raw metric row and the final search objective

Compact outputs:

- `reports/gamma_summary.csv`
- `reports/compact_search_results.csv`
- `reports/best_gamma.json`

## 6. Minimal Readout

```bash
cat $FM_TIMEWARP_EVAL_ROOT/reports/best_gamma.json
cat $FM_TIMEWARP_EVAL_ROOT/reports/compact_search_results.csv
```

```bash
python - <<'PY'
import json, os, pathlib
root = pathlib.Path(os.environ['FM_TIMEWARP_EVAL_ROOT']) / 'reports'
best = json.loads((root / 'best_gamma.json').read_text())
print('best_gamma', best['gamma'])
print('best_strategy', best['search_strategy'])
print('best_objective', best['objective'])
PY
```

## 7. Decision Rule

Phase B1 is positive if the selected `best_gamma`:

- beats `uniform` on the objective steps
- stays competitive or improves at higher step counts
- and produces better qualitative panels than `uniform`

If it wins only on the search objective but degrades strongly elsewhere, do not move directly to a training-time warp module.
