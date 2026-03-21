# Flow Matching Baseline Pipeline

This is the active Phase-1 baseline workflow for the refactored `dgfm` stack.
It is the only document you should follow for the new baseline path.
The current baseline is no longer the old lightweight velocity CNN. It now targets the
official `flow_matching/examples/image` CIFAR-10 recipe more closely:
- official-style UNet backbone
- EMA checkpointing
- skewed timestep sampling
- Heun2-style evaluation solver

## 0. Activate Experiment

```bash
cd ~/workspace/Zhengwei/DG-TWFD
source scripts/experiments/activate_fm_cifar10.sh baseline v1
conda activate consistency
```

Current variants:
- `baseline`
- `stable`

Important:
- runs produced before the official-style UNet/EMA update are not comparable to the current baseline
- start a fresh experiment tag, for example `baseline v2`, when collecting the next official-aligned result

Local smoke convention:
- local verification runs should stay under `./outputs/debug/`
- A100 formal runs should use the activated `/cache/...` paths

## 1. Environment

```bash
cd $PROJ
python -m pip install -U pip
python -m pip install -e .
python -m pip install -e ./flow_matching
python -m pip install torchvision scipy pandas torch_fidelity
```

Notes:
- `run_eval.py` uses `torch_fidelity` InceptionV3-2048 features for FID.
- The first FID run may download Inception weights if they are not already cached.
- Use `TORCH_HOME` under the run root so pretrained weights and caches do not spill into the home directory.
- Dataset preparation is manual-first. Training will not auto-download CIFAR-10.
- If GitHub download is slow, set a mirror before evaluation:

```bash
export DGFM_TORCH_FIDELITY_MIRROR_PREFIX=https://githubfast.com/
```

This rewrites the default `torch_fidelity` weight URL to:
- `https://githubfast.com/https://github.com/...`

If you already downloaded the weight file manually, you can pin it directly:

```bash
export DGFM_TORCH_FIDELITY_WEIGHTS_PATH=/path/to/weights-inception-2015-12-05-6726825d.pth
```

## 2. Dataset Preparation

### CIFAR-10

Manual-first check:

```bash
cd $PROJ
python scripts/build_dataset.py --dataset cifar10 --data-root $DATA_ROOT/cifar10
```

If the dataset is missing and you explicitly want the script to download it once:

```bash
python scripts/build_dataset.py --dataset cifar10 --data-root $DATA_ROOT/cifar10 --download
```

Fastest practical option:
- manually place `cifar-10-batches-py/` under `$DATA_ROOT/cifar10`
- then run the check command above

Expected layout:

```text
$DATA_ROOT/cifar10/
├── cifar-10-batches-py/
└── .dgfm_cache/               # created later by run_eval.py for cached FID stats
```

### Future ImageNet Preparation

```bash
python scripts/build_dataset.py --dataset imagenet32 --data-root $DATA_ROOT/imagenet32
python scripts/build_dataset.py --dataset imagenet64 --data-root $DATA_ROOT/imagenet64
```

## 3. Result Layout

```text
$RUN_ROOT/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── logs/
│   ├── config_resolved.yaml
│   └── train.jsonl
└── samples/

$METRIC_ROOT/
├── steps1/
│   ├── metrics.json
│   ├── generated_stats.npz
│   ├── fixed_seed_samples.pt
│   ├── fixed_seed_grid.png
│   └── fixed_seed_images/
├── steps2/
├── steps4/
├── steps8/
├── steps16/
└── reports/
    ├── summary.json
    ├── summary.csv
    └── best.json
```

## 4. Baseline Training

Training expects the dataset to already exist. It will fail fast instead of downloading in the background.

```bash
cd $PROJ
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT
```

Produced files:
- `$CKPT_DIR/best.pt`
- `$CKPT_DIR/last.pt`
- `$LOG_ROOT/config_resolved.yaml`
- `$LOG_ROOT/train.jsonl`

Checkpoint note:
- `best.pt` and `last.pt` now store both raw model weights and `ema_model`
- evaluation and sampling use EMA weights by default

## 5. Smoke Training

Use this before a long run:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT/smoke \
  --set train.epochs=1 \
  --set train.batch_size=8 \
  --set train.num_workers=0 \
  --set train.max_train_batches=8 \
  --set train.max_val_batches=4
```

## 6. Resume Training

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT \
  --resume $CKPT_DIR/last.pt
```

## 7. Full Baseline Evaluation

This computes cached reference statistics on the CIFAR-10 test split, then evaluates the checkpoint at `1/2/4/8/16` ODE steps.

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --eval-root $METRIC_ROOT \
  --steps 1 2 4 8 16
```

Default evaluation protocol:
- metric: `FID`
- reference split: `test`
- reference size: full test set
- generated sample count per step: `50000`
- FID batch size: `256`
- solver method: `heun2`
- qualitative grid seed: `42`
- qualitative grid size: `64`

Outputs per step:
- `metrics.json`: FID, `integration_steps`, `nfe`, runtime metadata, and cache references
- `generated_stats.npz`: Gaussian stats of generated features
- `fixed_seed_grid.png`: fixed-seed qualitative grid
- `fixed_seed_images/*.png`: dumped qualitative images
- `fixed_seed_samples.pt`: saved tensor grid payload

Summary outputs:
- `reports/summary.json`
- `reports/summary.csv`
- `reports/best.json`

Evaluation notes:
- `nfe` is the hardware-independent compute budget and should be used for method comparison.
- `elapsed_sec` and `samples_per_sec` are hardware-dependent and should only be compared on the same machine/setup.
- `num_fid_samples=50000` is the full evaluation setting. Smaller values are only approximate FID and should not be used as headline numbers.

## 8. Fast Evaluation Smoke

Use this before a 50k-sample full evaluation:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --eval-root $METRIC_ROOT/smoke \
  --steps 1 4 16 \
  --fid-samples 512 \
  --fid-batch-size 64 \
  --set eval.fixed_grid_size=16 \
  --set eval.dump_image_count=16
```

Recommended approximate FID settings:
- `--fid-samples 5000`: quick model ranking
- `--fid-samples 10000`: stronger smoke comparison
- `50000` only for final reporting

## 9. Qualitative Sampling Only

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_sample.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --output-dir $SAMPLE_ROOT/steps16 \
  --steps 16 \
  --num-samples 64 \
  --fixed-seed 42
```

Outputs:
- `$SAMPLE_ROOT/steps16/samples.pt`
- `$SAMPLE_ROOT/steps16/grid.png`
- `$SAMPLE_ROOT/steps16/images/*.png`

## 10. How To Judge The Baseline Run

A baseline run is considered complete only if all of the following exist:
- `checkpoints/best.pt`
- `logs/train.jsonl`
- `metrics/steps1/metrics.json`
- `metrics/steps16/metrics.json`
- `metrics/reports/summary.csv`
- `metrics/reports/best.json`
- `metrics/steps16/fixed_seed_grid.png`

Minimal inspection commands:

```bash
cat $METRIC_ROOT/reports/best.json
python - <<'PY'
import json, pathlib, os
path = pathlib.Path(os.environ['METRIC_ROOT']) / 'reports' / 'summary.json'
rows = json.loads(path.read_text())
for row in rows:
    print(row['step_count'], row['fid'])
PY
```

## 11. Benchmark Reference

Use two references:
- scientific reference: Flow Matching paper
- engineering reference: `flow_matching/examples/image/README.md`

For this repo, the most actionable near-term target is the vendored CIFAR-10 image example:
- `flow_matching/examples/image/README.md` reports `FID 2.07` for CIFAR-10 with a stronger training recipe and long schedule

Interpretation:
- your Phase-1 baseline should first establish a stable FID-vs-step curve and sensible qualitative grids
- after that, compare your best `step_count` and full-sample FID against the vendored example and the paper tables
- when comparing to official results, use the `nfe` field together with FID rather than wall-clock time
- do not compare smoke-eval FID numbers to the paper

## 12. Teacher Interface Usage

Phase 1 baseline uses:
- `teacher.type = none`

Future teacher-based experiments must keep the same CLI shape and switch only config.

## 13. Time-Warp Hook

Time-warp is not active in Phase 1 baseline training.
The future hook remains configuration-driven; do not create a new user-facing CLI path for it.
