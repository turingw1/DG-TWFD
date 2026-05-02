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

## 当前活跃实验

- Run tag:
  `edm_first_cifar10_prior_fullstack_timewarp_v21_ctm_aligned_from_v20_step3292`
- Config:
  `experiments/edm_first/configs/cifar10_edm_map_prior_fullstack_timewarp_v21_ctm_aligned.yaml`
- Source checkpoint:
  `runs/edm_first_cifar10_prior_fullstack_timewarp_v20_endpoint_balanced_from_v19_step7106/checkpoints/step3292.pt`
- 核心改动：
  1. 在 EDM student 外加入 `TimeConditionedTransitionAdapter`，让 `student_transition`
     显式依赖 `sigma_s`，从“denoiser + Euler”推进为可学习的 `t -> s` 转移。
  2. 保留 pretrained EDM base 的低学习率，同时给 transition adapter 更高学习率。
  3. 将 real-data DSM 从弱 anchor 提升为有效训练信号。
  4. 增加 real-data transition loss，直接训练真实 CIFAR-10 噪声态到中间态/endpoint。
  5. 保留 RQS timewarp 与 2-step `u=0.60` budget policy，用于组合采样对照。

验收标准：v21 必须显著改善 v20 的 FID@1/FID@4，且后续训练要回收初期 8/16
步退化；如果 8/16 长时间不回收，说明 CTM-style 数据信号过强而 composition
保护不足，需要调低 data transition 或加强 preservation。

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
- auto 2-step 继续错误，必须看 budget policy，不应报告 auto-2。

当前判断：v21 的方向是对的，因为它第一次在很早步数就显著拉动 endpoint；
但 loss 配比还没有完成，后续监督重点是 8/16 是否回收。

## 下一步监督规则

1. 继续运行 v21，并每两小时检查 budget FID@1/2/4/8/16。
2. 如果 FID@1 持续下降且 8/16 回收，保持 v21。
3. 如果 FID@1 下降但 8/16 不回收，开启 v21b：
   - 降低 `data_transition_weight` 或 endpoint probability；
   - 提高 preserve bridge / preserve perceptual；
   - 或让 adapter 只在 low-NFE 训练阶段生效，high-NFE composition 主要更新 base。
4. 如果 FID@1 停滞，下一步不是再加 timewarp，而是更接近 CTM：
   - stop-grad EMA target model；
   - adaptive DSM/consistency balance；
   - 更强的 `s` 条件注入，而不是小 residual adapter。

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
