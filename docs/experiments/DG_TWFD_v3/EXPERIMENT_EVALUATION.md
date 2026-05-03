# DG-TWFD v3 实验评估报告

本文档只记录会影响实验决策的结果、瓶颈和下一步设计。原始日志、图片、
checkpoint 与完整 eval artifacts 保留在 `runs/`、`eval/`、`results/` 和
`/temp/Zhengwei/projects/DG-TWFD/critical/`。

## 当前目标

最终 CIFAR-10 目标曲线：

| NFE | 1 | 2 | 4 | 8 |
|---:|---:|---:|---:|---:|
| target FID | 3.20 | 2.84 | 2.62 | 2.50 |

当前在线监督使用 FID-2048，主要用于方向判断；最终结论必须用更大样本数复验。

## 当前最新实验状态

- Run tag:
  `edm_first_cifar10_prior_fullstack_timewarp_v21b_ctm_gated_from_v21_step250`
- Config:
  `experiments/edm_first/configs/cifar10_edm_map_prior_fullstack_timewarp_v21b_ctm_gated.yaml`
- Source checkpoint:
  `runs/edm_first_cifar10_prior_fullstack_timewarp_v21_ctm_aligned_from_v20_step3292/checkpoints/step250.pt`
- 核心改动：
  1. 在 EDM student 外加入 `TimeConditionedTransitionAdapter`，让 `student_transition`
     显式依赖 `sigma_s`，从“denoiser + Euler”推进为可学习的 `t -> s` 转移。
  2. 保留 pretrained EDM base 的低学习率，同时给 transition adapter 更高学习率。
  3. 将 real-data DSM 从弱 anchor 提升为有效训练信号。
  4. 增加 real-data transition loss，直接训练真实 CIFAR-10 噪声态到中间态/endpoint。
  5. 保留 RQS timewarp 与 2-step `u=0.60` budget policy，用于组合采样对照。

运行状态：v21b 因 `train.max_seconds=28800` 到达 8 小时时间预算后自然完成，
最后 checkpoint 为 `step7630.pt`。训练没有数值爆炸，但 FID 曲线给出负面验收：
单步只有极弱改善，多步 composition 从 step250 后持续退化。

## 关键结果表

FID 为在线 FID-2048，使用 budget policy 时 2-step 固定 `u=0.60`，4+ step 使用
learned RQS warp。

| 阶段 | 关键 checkpoint | FID@1 | FID@2 | FID@4 | FID@8 | FID@16 | 决策含义 |
|---|---|---:|---:|---:|---:|---:|---|
| e504a baseline | step250 | 177.890 | 46.286 | 49.294 | 70.911 | 86.567 | endpoint 很弱，composition 也差 |
| EDM-first endpoint | resume step1750 | 91.325 | 46.693 | 59.096 | 90.190 | 103.199 | endpoint 快速改善，但多步严重退化 |
| v11 full-stack | step8750 / best comp | 68.851 | 35.051 | 30.516 | 26.536 | 28.459 | full-stack 有效，但 timewarp 不稳 |
| v12a budget policy | step10500 | 59.246 | 34.881 | 29.997 | 24.914 | 26.055 | budget policy 明确：2-step 不宜用 learned warp |
| v13 midpoint preserve | step7750 | 56.569 | 32.050 | 26.009 | 23.229 | 23.920 | midpoint preservation 修复低/中步组合 |
| v14 guarded cont. | step10750 | 52.424 | 31.240 | 24.159 | 20.665 | 21.678 | 高步数大幅改善，2-step 仍慢 |
| v15 multi-mid | step11855 | 49.281 | 29.528 | 21.398 | 19.902 | 20.279 | multi-midpoint 是真实算法增益 |
| v17 RQS warp | step9750 | 48.841 | 24.310 | 21.112 | 19.899 | 19.657 | RQS 对 4+ step 有效；2-step 需 budget clock |
| v18 endpoint EMA | step4684 | 44.325 | 25.580 | 21.097 | 20.325 | 21.273 | EMA/real-data anchor 强化 endpoint，但损伤 8/16 |
| v19 recovery | step7106 | 46.795 | 23.945 | 20.821 | 20.041 | 20.578 | 回收 8/16 与最佳 2-step，但 endpoint 回退 |
| v20 endpoint-balanced | step3000 | 45.614 | 23.977 | 20.958 | 19.845 | 20.334 | 稳定微调，斜率太小 |
| v21 CTM-aligned | step250 | 44.830 | 24.020 | 20.799 | 20.241 | 20.602 | endpoint/4-step 立刻改善，8/16 初期受扰动 |
| v21 CTM-aligned | step500 | 44.971 | 24.044 | 20.811 | 20.206 | 20.589 | 1/4 优于 v20，8/16 开始轻微回收 |
| v21 CTM-aligned | step750 | 45.100 | 24.061 | 20.835 | 20.183 | 20.568 | 8/16 继续回收，但 endpoint 连续回弹 |
| v21 CTM-aligned | step8111 | 44.987 | 26.967 | 25.259 | 24.296 | 26.736 | endpoint 小幅保留，composition 明确失败 |
| v21b CTM-gated | step250 | 44.843 | 24.034 | 20.831 | 20.225 | 20.596 | 保守 CTM 复现早期增益，等待 step500/750 验证是否抗退化 |
| v21b CTM-gated | step1000 | 44.988 | 24.130 | 20.956 | 20.179 | 20.499 | 8/16 短暂最健康，1/2/4 已开始回弹 |
| v21b CTM-gated | step7500 | 44.600 | 27.012 | 26.152 | 23.099 | 23.767 | 单步微降但 composition 长跑失败 |

## 实验演化脉络

1. **e504a / resume-from1250**  
   目标是先证明一阶 endpoint 可以被训练。结果显示 FID@1 从 177.890 降到
   91.325，但多步组合快速恶化，说明单纯 endpoint matching 会破坏 semigroup
   consistency。

2. **v11-v12a full-stack/timewarp**  
   引入 endpoint、match、defect、bridge 与 learned timewarp。主要收获不是
   SOTA，而是确认了 budget-dependent clock：2-step 更适合 identity/fixed
   midpoint，4+ step 才从 learned warp 获益。

3. **v13-v15 midpoint preservation**  
   中间点 preservation 是第一类稳定有效的算法改进。它把模型从 endpoint-only
   拉回可组合的 trajectory family，v15 将 4/8/16 推到约 20-21 区间。

4. **v16-v17 RQS timewarp**  
   piecewise density warp 表达力不足，RQS 增加了连续单调曲率。结论是 RQS
   对 4+ step 有稳定正增益，但无法解决 FID@1，也无法自动解决 2-step。

5. **v18-v20 endpoint 与 recovery**  
   EMA 和 real-data denoise anchor 能明显改善 FID@1，但 endpoint-only 会损伤
   8/16。v19/v20 证明可以回收 composition，但保守微调的学习斜率太小。

6. **v21 CTM-aligned**  
   当前核心判断：DG-TWFD 与 CTM 的 loss 形态接近，都是 match +
   semigroup/trajectory consistency；但此前模型参数化仍是 EDM denoiser +
   Euler。v21 开始补上 CTM 的关键条件：显式 `s` 条件、真实数据 DSM、真实数据
   transition。

## 当前瓶颈诊断

### 1. FID@1 的主瓶颈是参数化，不是 timewarp

旧 `student_transition(x_t, sigma_t, sigma_s)` 本质是：

1. student 只看 `x_t, sigma_t`，输出 denoised image；
2. 代码用 Euler 公式把它外推到 `sigma_s`。

当 `sigma_s=0` 时，单步生成几乎就是高噪声 EDM denoiser 的一次输出。EDM
teacher 原本服务于多步 ODE/SDE，不是直接学习 one-step transport map。因此
只改 warp、bin 数、RQS 曲率，都不能根本解决 FID@1。

### 2. CTM no-GAN 快速到 FID<10 的原因

CTM 的关键优势不是 GAN，而是训练范式：

- 模型显式接收 `t` 和 `s`，学习 `G_theta(x_t, t, s)`；
- 使用 stop-grad target model 形成稳定 consistency target；
- 用 Heun teacher target 对齐 trajectory；
- real-data DSM 是主训练信号，并用 adaptive balance 与 consistency loss 协调；
- CIFAR 训练学习率和 batch 都明显更强。

v20 之前 DG-TWFD 的 real-data anchor 权重只有 `0.035`，主要仍在做 prior rollout
pixel/perceptual matching，所以表现为保守 fine-tune，而不是能力跃迁。

### 3. v21 早期正负信号

正信号：

- step250 FID@1 从 v20 step3000 的 45.614 降到 44.830；
- FID@4 从 20.958 降到 20.799；
- 训练 loss 无爆炸，GPU 占用约 45-48GB，eval 可并行。

风险：

- 8/16 从 v20 的 19.845/20.334 退到 20.241/20.602；
- step500 FID@1 比 step250 回弹到 44.971，说明 adapter/DSM 的早期更新
  可能在 endpoint 与 composition 间拉扯；但 8/16 已从 step250 的
  20.241/20.602 轻微回收到 20.206/20.589；
- step750 进一步确认 tradeoff：8/16 回收到 20.183/20.568，但 FID@1 回弹到
  45.100。若下一轮 FID@1 继续回弹，说明当前 data transition/adapter 更新正在
  牺牲 endpoint，应分支 v21b 调低 data transition 或加强 endpoint/EMA 保护；
- step8111 给出明确负结论：FID@1 虽从中期最差约 46.2 回到 44.987，但
  4/8/16 已退化到 25.259/24.296/26.736，远差于 v20 的
  20.958/19.845/20.334。问题不是训练崩溃，而是 `data_transition_weight=0.85`
  与 `transition_adapter_lr=3e-5` 在长跑中持续改写 transition family，超过
  preservation 的约束能力；
- v21b 从 v21 step250 接续后，首轮 step250 budget FID 为
  44.843/24.034/20.831/20.225/20.596，基本复现 v21 早期健康区间，并没有
  立刻造成新的 8/16 退化。真正验收点是 step500/750 是否不再出现 v21 那种
  4/8/16 单调恶化；
- auto 2-step 继续错误，必须看 budget policy，不应报告 auto-2。

当前判断：v21 证明 “显式 `s` 条件 + real-data transition” 有早期价值，但
当前权重过强，不能直接长跑。v21b 从 v21 step250 继续，保留早期 endpoint/4-step
收益，同时降低 adapter/data-transition 强度并提高 preservation。

### 4. v21b 完整验收

v21b 是对 v21 的保守修正：`transition_adapter_lr=8e-6`，
`data_transition_weight=0.25`，`data_transition_endpoint_prob=0.35`，并提高
preserve bridge/defect/perceptual。训练按 8 小时时间预算自然结束于 step7630。

最佳 budget FID-2048 分步并不来自同一 checkpoint：

| metric | best checkpoint | best FID |
|---|---:|---:|
| FID@1 | step7500 | 44.600 |
| FID@2 | step250 | 24.034 |
| FID@4 | step250 | 20.831 |
| FID@8 | step1000 | 20.179 |
| FID@16 | step2000 | 20.427 |

这组结果说明 v21b 不是 “慢热”。如果方法有效，至少应在 4/8/16 上维持或下降；
实际是从 step250 开始持续退化：

- budget mean FID@4/8/16: step250 `20.551` -> step7500 `24.339`；
- FID@2: `24.034` -> `27.012`；
- FID@1: `44.843` -> `44.600`，8 小时只降低 `0.242`。

训练 loss 也支持这个判断：前 40 条与后 40 条记录的平均 `loss`、`anchor_loss`、
`bridge_loss`、`defect_loss` 基本不降，`data_denoise_loss` 也没有系统性改善。
这说明当前 residual adapter + 加权 loss 主要在已学到的 transition family 上做
小幅扰动，不是在学习 CTM 那种强 one-step transport 能力。

### 5. 达到目标曲线的可行性估计

目标 FID@1/2/4/8 为 `3.20/2.84/2.62/2.50`。按 v21b 的实际斜率估计，当前
方法直接长跑不可行：

- FID@1 从 step250 到 step7500 只改善 `0.242`，仍差目标约 `41.4`；
- FID@2/4/8 与目标的差距约 `21.2/18.2/17.7`，且方向是变差；
- 即便线性外推 FID@1 的微弱改善，也需要不现实的训练步数，而且多步会继续被破坏。

因此当前结论是：**v21b 不能作为继续长跑冲 SOTA 的主线**。它证明保守 gating 可以
减轻 v21 的灾难性 16-step 崩坏，但不能产生能力跃迁。下一版必须改训练范式，而不是
再调小权重或继续堆 timewarp。

## 机制图 v1 观察

- 生成脚本：
  `scripts/figures/build_timewarp_mechanism_figure.py`
- 输出目录：
  `docs/experiments/DG_TWFD_v3/figures/mechanism/timewarp_mechanism_20260502/`
- 数据：v21 `step2500.pt` 的 EMA student、learned RQS warp、24 条 held-out
  CIFAR-10 条件轨迹、32 段 dense teacher path、8-step composed student path。

结论需要谨慎表述。A/B/C 图能支持 “teacher path 不变，learned clock 改变
waypoint 分配，并在 finite semigroup-defect segments 上降低组合缺陷”：

- identity mean semigroup defect: `7.82e-05`
- DG-TWFD mean semigroup defect: `6.60e-05`
- held-out trajectories improved: `87.5%`

但同一 original-time dense grid 上用线性插值衡量的 path-state MSE 仍然更差
（identity `1.06` vs DG `2.98`）。这说明当前 warp 的机制证据更适合写成
composition-defect reduction，而不是直接宣称低步数学生轨迹全程更贴近 teacher
state path。若主文要使用 “closer trajectory” 叙述，需要继续改进 student
transition 本身，或改用真实可访问 waypoint/segment-level metric。

## 下一步监督规则

1. 停止把 v21/v21b 作为长跑主线；保留关键 checkpoint 和完整 eval reports。
2. 下一版不再只调权重，应接近 CTM 的训练骨架：
   - 引入 stop-grad EMA target model，consistency target 不从同一个在线 student
     反向传播；
   - 用 teacher Heun/EDM dense rollout 形成稳定 `x_t -> x_s` target，而不是只靠
     residual adapter 修正 Euler；
   - 将 real-data DSM 与 consistency 做 adaptive balance，避免 endpoint 与
     composition 互相拉扯；
   - 强化 `s` 条件注入位置，让主网络真正学习 `G(x_t,t,s)`，而不是外层小 adapter。
3. timewarp 仍作为低步数 traversal 的优势模块保留，但只能在 student transition
   本身有足够生成能力后发挥作用；现阶段不应把时间投入到单独微调 warp。

## Artifact 索引

- v21 config:
  `experiments/edm_first/configs/cifar10_edm_map_prior_fullstack_timewarp_v21_ctm_aligned.yaml`
- v21 launcher:
  `experiments/edm_first/scripts/launch_prior_fullstack_timewarp_v21_ctm_aligned.sh`
- v21 run:
  `runs/edm_first_cifar10_prior_fullstack_timewarp_v21_ctm_aligned_from_v20_step3292`
- v21 eval:
  `eval/edm_first_cifar10_prior_fullstack_timewarp_v21_ctm_aligned_from_v20_step3292_step*`
- temp project backup:
  `/temp/Zhengwei/projects/DG-TWFD/`

## 操作原则

- baseline 和 DG-TWFD 主实验互不冲突，不杀 baseline。
- 大 checkpoint 只保留关键节点；日志、eval reports、分析文档保留。
- 每次关键代码/文档变更必须 git commit/push，并执行
  `bash scripts/server/backup_codex_project_v11.sh DG-TWFD`。
