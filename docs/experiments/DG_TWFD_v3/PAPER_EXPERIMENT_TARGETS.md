# DGTD v3 Paper Experiment Targets

Purpose: organize all DGTD v3 comparisons into paper-ready tables. Values in
parentheses are planning estimates, not reported results. Replace parenthesized
numbers only after server experiments finish.

## Result Logic

| Claim chain | Evidence table / figure | Primary metric | Expected paper signal |
| --- | --- | --- | --- |
| defect-guided time warping | Table 3, Fig. B-C | `q_phi`, `D_bar`, defect | high-defect intervals get more time mass |
| lower semigroup defect | Table 2, Fig. B | defect, direct/bridge gap | Full has lowest or fastest-falling defect |
| stable cross-step reuse | Table 1, Fig. A/F | FID@1/2/4/8, Recall | one checkpoint improves smoothly with steps |
| quality-speed-coverage balance | Table 5 | FID, Recall, NFE, latency | Full is not only a one-step specialist |

## Table 1: Main Cross-Step Reusability

| Row | Model | Config / source | How to run | What it tests | FID@1 | FID@2 | FID@4 | FID@8 | Recall@8 | Latency@8 | Expected conclusion |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| M0 | Base map distill | `fm_cifar10_map_branch` | train/eval with `scripts/run_train.py` and `scripts/run_eval.py` | ordinary explicit map without DGTD feedback | (85) | (76) | (73) | (74) | (0.43) | (1.0x) | learns jumps but reuses poorly |
| M1 | DGTD NoWarp | `dgtd_cifar10_v3_ablation_no_warp` | same pipeline, uniform/base time | defect signal without learned warp | (72) | (58) | (49) | (45) | (0.48) | (1.0x) | defect helps composition |
| M2 | Full DGTD | `dgtd_cifar10_v3` | online-mainline full run | defect + learned warp + detail metric | (70) | (52) | (39) | (32) | (0.54) | (1.05x) | best smooth cross-step reuse |

Brief: this is the main result table. Full does not need to win at 1-step; the
paper point is smoother and stronger improvement from 1 to 8 steps under one
checkpoint.

## Table 2: Defect Usage Ablation

| Row | Variant | Required config state | Defect in loss | Defect updates warp/schedule | How to experiment | FID@1 | FID@4 | FID@8 | Final defect | Direct-bridge gap | Conclusion role |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A0 | Full DGTD | exists: `dgtd_cifar10_v3` | yes | yes | formal full/short run | (70) | (39) | (32) | (0.18) | (0.010) | reference |
| A1 | w/o defect | needs committed config | no | no | disable DGTD residual feedback; use map/consistency baseline | (86) | (75) | (76) | (0.45) | (0.035) | defect is necessary |
| A2 | monitor-only defect | needs committed config | no or detached | no | log `D_bar` but do not train/update from it | (82) | (68) | (66) | (0.39) | (0.030) | measuring defect is not enough |
| A3 | defect no learned warp | exists: `dgtd_cifar10_v3_ablation_no_warp` | yes | no | uniform/base time with DGTD residual | (72) | (49) | (45) | (0.25) | (0.018) | closed-loop warp releases value |

Brief: this table answers whether defect is the causal training signal. A1/A2
are not currently formal configs and should be added only after M0/M1/M2 are
stable.

## Table 3: Warp Variant Comparison

| Row | Warp | Config / source | Definition | How to experiment | FID@2 | FID@4 | FID@8 | Entropy `q_phi` | Max `q_phi/q_base` | Expected conclusion |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| W0 | No warp | `dgtd_cifar10_v3_ablation_no_warp` | original/base time | run DGTD with `disable_warp=true` | (58) | (49) | (45) | (4.16) | (1.0) | no reparameterization is limited |
| W1 | source-dense | `fm_cifar10_map_branch_s1_e5_warp_source_dense` | hand-dense source region | map-branch control run | (78) | (70) | (69) | n/a | n/a | heuristic density helps weakly |
| W2 | data-dense | `fm_cifar10_map_branch_s1_e5_warp_data_dense` | hand-dense data/target region | map-branch control run | (76) | (67) | (65) | n/a | n/a | data heuristic is still not adaptive |
| W3 | spline, no DGTD feedback | `fm_cifar10_map_branch_s1_e5_warp_spline` | learnable monotone spline | map-branch spline control | (74) | (63) | (60) | (3.8) | (2.5) | learnable warp alone is insufficient |
| W4 | defect-guided learned warp | `dgtd_cifar10_v3` | density learned from defect bins | full DGTD pipeline | (52) | (39) | (32) | (3.4) | (3.2) | feedback, not spline form, is key |

Brief: W3 vs W4 is the important comparison. It prevents the claim from
collapsing into "any learnable warp works."

## Table 4: Boundary / Detail Refinement

| Row | Variant | Config state | How to experiment | FID@1 | FID@8 | HF residual | Visual effect | Conclusion role |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| B0 | Full without extra boundary head | current `dgtd_cifar10_v3` | default mainline | (70) | (32) | (0.021) | stable composition | clean main method |
| B1 | + boundary corrector | needs config/code if re-enabled | enable boundary module only | (66) | (31) | (0.020) | better first jump | boundary is a refinement |
| B2 | + low-noise detail metric | current metric path / no-HF ablation contrast | compare to `dgtd_cifar10_v3_ablation_warp_no_hf` | (71) | (30) | (0.016) | sharper low-noise detail | detail metric helps late steps |
| B3 | + both | needs combined config | enable both if B1/B2 help | (64) | (29) | (0.014) | best texture if stable | optional final polish |

Brief: this table is secondary. It should not carry the main paper claim; it
only separates composition stability from low-noise detail repair.

## Table 5: System-Level Few-Step Comparison

| Method | Data source | Single checkpoint reused? | FID@1 | FID@2 | FID@4 | FID@8 | Recall / coverage | NFE | Main interpretation |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PD | published / re-run if available | weak/no | published | published | published | published | published | 1-8 | progressive distillation reference |
| CD | published / re-run if available | weak/no | published | published | published | published | published | 1-8 | consistency baseline |
| CTM | published or `refs/ctm-cifar10` | yes | published | published | published | published | published | 1-8 | strong few-step baseline |
| Full DGTD | this repo | yes | (70) | (52) | (39) | (32) | (0.54) | 1-8 | balanced reusable checkpoint |

Brief: this table should not overclaim beating CTM in every column. The target
claim is stable step scaling and balanced coverage under one checkpoint.

## Table 6: Mechanism Figures

| Figure | Source files | How to generate | Expected value / pattern | Paper message |
| --- | --- | --- | --- | --- |
| Fig. A: step-FID curve | `$METRIC_ROOT/reports/summary.csv` | plot FID vs steps for M0/M1/M2 | Full slope remains negative through 8 steps | same checkpoint keeps improving |
| Fig. B: defect heatmap | `logs/train.jsonl` | use `D_bar` over epochs/bins | high bins fall from (0.45) to (0.18) | semigroup defect is reduced |
| Fig. C: learned density | `q_phi`, `q_D` in `train.jsonl` | plot final `q_phi` vs `q_D` | density expands high-defect intervals | warp follows defect feedback |
| Fig. D: time histogram | sampled `time_grid` / `q_phi` | compare base vs learned grid | clean/mid bins get nonuniform mass | training distribution really changed |
| Fig. E: HF residual | `HF_bar`, `train_low_sigma_hf` | plot low-sigma residual | no-HF (0.025), Full (0.016) | detail metric repairs low-noise region |
| Fig. F: visual grids | `$SAMPLE_ROOT/steps*/grid.png` | fixed seed grids at 1/2/4/8 | progressively less noisy, not blurrier | qualitative cross-step reuse |

## Recommended Paper Order

| Section | Table / figure | Purpose |
| --- | --- | --- |
| Main Result | Table 1, Fig. A/F | establish cross-step reuse |
| Mechanism | Table 6, Fig. B/C/D | show defect reduction and time redistribution |
| Core Ablation | Table 2 | prove defect feedback is necessary |
| Warp Comparison | Table 3 | prove adaptive warp is not a heuristic artifact |
| Detail Refinement | Table 4 | explain low-noise/boundary polish |
| Published Baselines | Table 5 | position against few-step literature |

## Immediate Server Instructions For Baseline Data

These commands produce the baseline columns now. Run on the Linux server only.
Do not adapt these to Mac local paths.

### 1. Common setup

```bash
export PROJ=/data2/yl7622/Zhengwei/DG-TWFD
cd "$PROJ"

source /data2/yl7622/anaconda/etc/profile.d/conda.sh
conda activate "$PROJ/.conda_envs/dgfm_map"

git checkout DG_TWFD_v3
git pull --ff-only
```

### 2. Main-Base: ordinary map baseline for Table 1 row M0

```bash
source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch paper_m0_base

CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES} torchrun \
  --standalone \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT \
  --verbose \
  2>&1 | tee $RUN_ROOT/train.stdout_stderr.txt

CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} python scripts/run_eval.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --eval-root $METRIC_ROOT \
  --steps 1 2 4 8 \
  2>&1 | tee $METRIC_ROOT/eval.stdout_stderr.txt
```

Return:

```bash
tail -n 10 "$LOG_ROOT/train.jsonl"
find "$CKPT_DIR" -maxdepth 2 -type f | sort
find "$METRIC_ROOT" -maxdepth 4 -type f | sort
cat "$METRIC_ROOT/reports/summary.csv"
cat "$METRIC_ROOT/reports/summary.json"
```

### 3. Main-NoWarp: DGTD defect without learned warp for Table 1 row M1

```bash
source scripts/experiments/activate_fm_cifar10.sh dgtd_cifar10_v3_ablation_no_warp paper_m1_nowarp

CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES} torchrun \
  --standalone \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT \
  --verbose \
  2>&1 | tee $RUN_ROOT/train.stdout_stderr.txt

CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} python scripts/run_eval.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --eval-root $METRIC_ROOT \
  --steps 1 2 4 8 \
  2>&1 | tee $METRIC_ROOT/eval.stdout_stderr.txt
```

Return the same files as M0.

### 4. Warp heuristic baselines for Table 3 rows W1-W3

```bash
for variant in \
  fm_cifar10_map_branch_s1_e5_warp_source_dense \
  fm_cifar10_map_branch_s1_e5_warp_data_dense \
  fm_cifar10_map_branch_s1_e5_warp_spline
do
  source scripts/experiments/activate_fm_cifar10.sh "$variant" "paper_${variant}"

  CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES} torchrun \
    --standalone \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/run_train.py \
    --config $FM_CONFIG \
    --run-root $RUN_ROOT \
    --verbose \
    2>&1 | tee $RUN_ROOT/train.stdout_stderr.txt

  CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} python scripts/run_eval.py \
    --config $FM_CONFIG \
    --checkpoint $CKPT_DIR/best.pt \
    --eval-root $METRIC_ROOT \
    --steps 1 2 4 8 \
    2>&1 | tee $METRIC_ROOT/eval.stdout_stderr.txt

  cat "$METRIC_ROOT/reports/summary.csv"
done
```

### 5. Compact analysis artifact for every run

```bash
python scripts/analyze_dgtd_run.py \
  --run-root "$RUN_ROOT" \
  --eval-root "$METRIC_ROOT" \
  --sample-root "$SAMPLE_ROOT" \
  --config "$FM_CONFIG" \
  --output "$RUN_ROOT/dgtd_analysis.md" \
  --json-output "$RUN_ROOT/dgtd_analysis.json"
```

For non-DGTD map baselines, if `analyze_dgtd_run.py` fails because DGTD fields
are absent, return only `train.jsonl`, `summary.csv`, `summary.json`, checkpoint
list, and fixed-seed sample grids.
