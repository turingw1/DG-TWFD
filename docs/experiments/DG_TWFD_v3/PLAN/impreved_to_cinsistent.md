2026-04-26 执行状态：本计划已从“建议路线”升级为当前主线。DDPM/DGTD 离散 teacher 路线在 `e405b/e406a/e407a` 后暂停；第一版 EDM-first 连续 teacher 代码已放在 `experiments/edm_first/`，并完成 `e501a/e502a` 训练与评测。当前证据见 `docs/experiments/DG_TWFD_v3/DDPM_TEACHER_SUITABILITY_2026-04-26.md`。

先给你结论：这份服务器计划应该**优先把“能产出真实结果的代码任务”提前**，暂时把论文文字、复杂图美化、CTM 重训、附录 toy 等后移。理由很明确：EDM 官方仓库已经提供了 `train.py`、`generate.py`、预训练 checkpoint，以及 CIFAR-10 / ImageNet64 的推荐采样设置，能最快把“连续型 teacher + online rollout + 自动评测”闭环打通；CTM 也有官方 PyTorch 代码，但重训成本更高，适合作为第二阶段公平基线。AYS 是 ICML 2024 的正式论文，Optimal Stepsize 也有官方代码仓库；这些更适合先作为**schedule baseline 接口预留**，而不是立刻成为第一优先级跑完的对象。评测上，`torch-fidelity` 官方文档强调 FID/IS/KID 的实现细节会显著影响数值可比性，所以评测脚本必须统一、自动化。([GitHub][1])

下面这份就是我建议你直接交给 Codex，在服务器上推进实现的 **plan 形式文档**。它把需要代码完成的部分提前，并按“先出结果、再扩 baseline、最后补论文图表”的顺序组织。

````markdown id="9xjqeo"
# DG-TWFD Server Execution Plan (EDM-first)
Status: primary implementation plan for the next 10 days  
Goal: use **EDM-style continuous teacher + online rollout** to obtain real, reproducible experiment results on server as early as possible, while preparing clean automation for later baseline expansion.

---

## 0. High-level strategy

This phase is **not** paper-polish first.
This phase is:

1. migrate the current method toward an **EDM-like continuous teacher setting**
2. make the **training/eval loop run end-to-end on server**
3. optimize algorithm supervision and get **real evidence**
4. make baseline acquisition and evaluation **automated**
5. only after that, expand figures/tables

Core priority order:

- P0: server-closed-loop training and evaluation
- P1: EDM-based mainline experiments
- P2: automated baseline acquisition / evaluation
- P3: schedule baselines and fair CTM rerun hooks
- P4: figure/table exporters for paper

---

## 1. Directory and ownership

All new implementation should be centered around:

`DG-TWFD/refs/edm/experiments/dg_twfd_teacher_proxy/`

Use the following layout:

```text
DG-TWFD/refs/edm/experiments/dg_twfd_teacher_proxy/
  README.md
  PLAN.md
  configs/
    cifar10/
    imagenet64/
  scripts/
    launch_train.sh
    launch_eval.sh
    launch_all.sh
    launch_baselines.sh
    summarize_runs.py
    export_paper_tables.py
  src/
    teacher/
    scheduling/
    warp/
    defect/
    evaluators/
    runners/
  outputs/
  results/
  logs/
````

Rules:

* `configs/` decides experiment design
* `src/` contains reusable logic
* `scripts/` contains server entrypoints only
* `results/` contains paper-facing summaries
* all experiments must be reproducible from one shell command

---

## 2. Immediate code-first milestones

## Milestone A: EDM continuous-teacher training mainline

Target: get one **real**, server-runnable mainline using EDM-like continuous teacher online rollout.

### A1. Teacher adapter

Implement a unified EDM teacher adapter that supports:

* loading official EDM pretrained checkpoints
* online local rollout from `(x_t, t)` to `(x_s, s)`
* optional triplet rollout `(t, s, r)` for defect computation
* CIFAR-10 first, ImageNet64 second

Required output API:

```python
teacher = build_teacher(cfg)

x_s = teacher.transition(x_t, t, s, cond=None)
x_r = teacher.transition(x_t, t, r, cond=None)

triplet = teacher.sample_triplet(batch, sampler_cfg)
# returns x_t, x_s, x_r, t, s, r, meta
```

Notes:

* no offline trajectory bank in this phase
* everything should work in online mode
* keep the API independent from student architecture

Acceptance:

* one smoke script on CIFAR-10 can call `teacher.transition`
* one batch triplet generation succeeds on server GPU
* latency is logged

### A2. Continuous-time student training entry

Implement one clean train entry for the current DG-TWFD mainline under EDM teacher:

Suggested file:
`src/runners/train_dgtwfd_edm.py`

Must support:

* direct endpoint prediction as default
* optional residual / velocity prediction modes
* `L_match + lambda_def * L_def`
* same backbone interface across targets
* logging:

  * train/loss_match
  * train/loss_def
  * train/loss_total
  * val/match_mse
  * val/raw_defect
  * val/norm_defect

Acceptance:

* CIFAR-10 short run finishes
* checkpoint, jsonl, sample outputs are generated
* eval script can read produced checkpoint

### A3. Evaluation pipeline

Implement one evaluation entry that supports:

* 1 / 2 / 4 / 8 step generation
* FID
* IS for CIFAR-10
* optional Recall/Precision for ImageNet64
* match MSE
* raw defect
* normalized defect

Suggested file:
`src/runners/eval_dgtwfd_edm.py`

Output files:

* `summary.json`
* `summary.csv`
* `samples/grid_step1.png`
* `samples/grid_step2.png`
* `samples/grid_step4.png`
* `samples/grid_step8.png`

Acceptance:

* one command can evaluate a checkpoint at 1/2/4/8
* metrics are aggregated into one csv row
* no manual postprocessing required

---

## 3. Main experiments to run first

## Experiment Group E1: Mainline viability

Goal: verify that the algorithm can close the loop under EDM teacher.

### E1.1 CIFAR-10 full DG-TWFD mainline

Config name suggestion:
`configs/cifar10/dgtwfd_edm_main.yaml`

Must produce:

* FID@1/2/4/8
* IS@1/2/4/8
* match MSE
* raw defect
* normalized defect

This is the first experiment that must be real.

### E1.2 CIFAR-10 no-warp baseline

Config name:
`configs/cifar10/dgtwfd_edm_no_warp.yaml`

Purpose:

* verify whether warp helps once EDM teacher is used online

### E1.3 CIFAR-10 identity-clock baseline

Config name:
`configs/cifar10/dgtwfd_edm_identity.yaml`

Purpose:

* verify whether redefining time is necessary

### E1.4 ImageNet64 full DG-TWFD mainline

Config name:
`configs/imagenet64/dgtwfd_edm_main.yaml`

Only start this after CIFAR-10 closes the loop.

---

## 4. Prediction-target ablation (must be code-complete early)

This is one of the fastest high-value ablations.

## Experiment Group E2: target parameterization

Goal: under the same EDM teacher and same training budget, compare:

* endpoint
* residual
* velocity

Config suggestions:

* `dgtwfd_edm_target_endpoint.yaml`
* `dgtwfd_edm_target_residual.yaml`
* `dgtwfd_edm_target_velocity.yaml`

Fixed variables:

* same backbone
* same teacher
* same optimizer
* same updates
* same time sampler
* same eval seeds

Required outputs:

* `FID@4`
* `match_mse`
* `normalized_defect`

Result export format:

```csv
dataset,target,fid4,match_mse,raw_defect,norm_defect,checkpoint,seed
```

Acceptance:

* CIFAR-10 three-target table is real
* ImageNet64 target table is desirable but secondary
* if endpoint wins or is the most stable default, freeze this as the paper default

---

## 5. Time schedule / time-coordinate comparisons

Do not over-expand at first. Make the interface ready now; fill methods incrementally.

## Experiment Group E3: time design

Primary rows to support now:

* identity clock
* full DG-TWFD

Secondary hooks to implement now, even if not all are run immediately:

* EDM schedule reference
* AYS-style schedule
* OptimalSteps-style schedule

Implementation requirement:
create a unified schedule API in:

`src/scheduling/registry.py`

with interface:

```python
schedule = build_schedule(cfg)

times = schedule.make_steps(num_steps=4, teacher=teacher, model=model, meta=...)
```

Minimum methods:

* `identity`
* `warped_uniform`
* `edm_reference`
* placeholder hooks for `ays_like` and `optimalsteps_like`

Important:

* “EDM reference” should mean using EDM-native low-step or teacher-native spacing logic
* “AYS-like” and “optimalsteps-like” should be clearly documented as **our reimplemented schedule policies under our protocol**, not direct claims of reproducing their paper numbers

Acceptance:

* schedule API exists and is reusable
* identity vs warped_uniform can be run immediately
* later methods can be plugged in without refactoring trainer/evaluator

---

## 6. Baseline acquisition and automation

Need to make baseline code and execution paths explicit.

## Baseline B1: EDM official

Purpose:

* teacher source
* public low-step reference
* schedule reference

Acquisition:

* use the official NVlabs/edm repo already in `refs/edm`
* document the checkpoint paths in `README.md`
* add one script to verify checkpoint download / existence

Script:
`scripts/check_edm_assets.sh`

Must verify:

* CIFAR-10 checkpoint path
* ImageNet64 checkpoint path
* FID ref stats path
* output write permission

## Baseline B2: CTM

Purpose:

* later fair comparison / rerun

For now:

* do **not** prioritize full rerun before mainline is stable
* but prepare acquisition and one launcher

Required now:

* `docs/baselines/ctm_setup.md`
* `scripts/launch_ctm_baseline.sh`

Must state:

* repo path
* env
* expected config
* expected output locations
* fairness caveats

## Baseline B3: schedule baselines

Need explicit “how to obtain / emulate” notes for:

* EDM
* AYS
* OptimalSteps

Deliverable:
`docs/baselines/schedule_methods.md`

For each baseline include:

* paper
* code repo if available
* whether directly reusable
* whether we reimplement its idea or reproduce its original code
* what exact row name it corresponds to in our paper

Acceptance:

* no ambiguous schedule row name remains
* every baseline row has a documented provenance

---

## 7. Automation and server workflow

The server should be able to run the full experiment loop with one command family.

## Required launch scripts

### `scripts/launch_train.sh`

Inputs:

* config path
* run tag
* GPUs
* optional overrides

Outputs:

* run directory
* train log
* checkpoint path

### `scripts/launch_eval.sh`

Inputs:

* config path
* checkpoint path
* eval steps list
* output dir

Outputs:

* summary json/csv
* sample grids
* metrics logs

### `scripts/launch_all.sh`

One command to do:

* train
* eval 1/2/4/8
* summarize

### `scripts/launch_baselines.sh`

Batch-launch:

* identity
* no-warp
* full DG-TWFD
* target ablations

### `scripts/summarize_runs.py`

Aggregate all runs into:

* `results/leaderboard.csv`
* `results/prediction_target_table.csv`
* `results/time_design_table.csv`

Acceptance:

* no manual copying of metrics into paper tables
* every paper table can be regenerated from csv exports

---

## 8. Paper-facing outputs that code must generate automatically

These are still code tasks and should be prepared early.

## Required outputs

Code must be able to export:

### Tables

* mainline multistep comparison csv
* prediction-target ablation csv
* time-design ablation csv

### Figures

* step-vs-FID curve
* defect-over-training curve
* time-warp visualization
* qualitative 1/2/4/8-step sample grid

Preferred formats:

* CSV for tables
* PDF for figures

Suggested scripts:

* `scripts/export_paper_tables.py`
* `scripts/export_paper_figures.py`

Acceptance:

* figure output is PDF
* no hand-made plotting in notebook required for core figures

---

## 9. Concrete execution order on server

## Phase 1: close the loop

1. `check_edm_assets.sh`
2. CIFAR-10 teacher transition smoke test
3. CIFAR-10 train short run
4. CIFAR-10 eval 1/2/4/8
5. verify metric outputs

## Phase 2: first real results

1. CIFAR-10 full DG-TWFD
2. CIFAR-10 identity
3. CIFAR-10 no-warp
4. CIFAR-10 endpoint/residual/velocity

## Phase 3: extend scope

1. ImageNet64 full DG-TWFD
2. ImageNet64 identity
3. optional ImageNet64 target ablation

## Phase 4: baseline expansion

1. EDM reference row generation
2. AYS-like / OptimalSteps-like schedule hooks
3. CTM setup and optional rerun preparation

---

## 10. Immediate deliverables Codex must create first

These should be implemented before anything else:

* [ ] `PLAN.md` in `dg_twfd_teacher_proxy`
* [ ] `README.md` cleaned and expanded
* [ ] `src/teacher/edm_teacher.py`
* [ ] `src/runners/train_dgtwfd_edm.py`
* [ ] `src/runners/eval_dgtwfd_edm.py`
* [ ] `src/scheduling/registry.py`
* [ ] `scripts/check_edm_assets.sh`
* [ ] `scripts/launch_train.sh`
* [ ] `scripts/launch_eval.sh`
* [ ] `scripts/launch_all.sh`
* [ ] `scripts/summarize_runs.py`
* [ ] one CIFAR-10 main config
* [ ] one CIFAR-10 identity config
* [ ] one CIFAR-10 target-endpoint config
* [ ] one CIFAR-10 target-residual config
* [ ] one CIFAR-10 target-velocity config

---

## 11. Definition constraints (must be respected in implementation)

To keep the experiments fair and paper-usable:

### Match MSE

Always compute in map space:

```python
match_mse = mse(M_theta(t, s, x_t), x_s_teacher)
```

### Raw defect

```python
raw_defect = mse(
    M_theta(t, r, x_t),
    M_theta(s, r, M_theta(t, s, x_t))
)
```

### Normalized defect

```python
norm_defect = raw_defect / (eps + mse(x_r_teacher, x_t))
```

### Target parameterizations

* endpoint:
  `M = F(x_t, t, s)`
* residual:
  `M = x_t + F(x_t, t, s)`
* velocity:
  `M = x_t - (t - s) * F(x_t, t, s)`

These definitions must appear identically in:

* code
* README
* result export headers
* final paper table text

---

## 12. What not to do now

* do not overbuild the toy experiment yet
* do not hand-fill paper numbers
* do not start CTM fair rerun before EDM mainline is stable
* do not add too many schedule baselines before identity/full are real
* do not split logic across too many unrelated folders

---

## 13. Success criteria for this phase

This phase is successful if, within the next execution cycle, the repo can produce:

1. one real CIFAR-10 mainline result under EDM teacher
2. one real CIFAR-10 prediction-target ablation table
3. one real CIFAR-10 time-design ablation row pair (identity vs full)
4. all outputs automatically summarized into csv/md/pdf
5. a clean path to add ImageNet64 and CTM next

```

这份计划的依据是：EDM 官方仓库已经把低步数采样、CIFAR-10/ImageNet64 推荐设置、预训练 checkpoint 与 sampler ablation 暴露得很清楚，非常适合先做连续 teacher 的 online rollout 主线；CTM 官方代码可作为后续公平基线；AYS 和 Optimal Stepsize 更适合先纳入统一 schedule API 和 baseline provenance 文档，而不是一开始就强行并行复现。:contentReference[oaicite:1]{index=1}

如果你同意，我下一步就直接帮你把上面这份 `PLAN.md` 再压缩成一版**更像 Codex 可执行任务单**的 prompt。
::contentReference[oaicite:2]{index=2}
```

[1]: https://github.com/nvlabs/edm?utm_source=chatgpt.com "GitHub - NVlabs/edm: Elucidating the Design Space of Diffusion-Based Generative Models (EDM) · GitHub"
