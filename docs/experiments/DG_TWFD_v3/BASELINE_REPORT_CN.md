# DG-TWFD v3 Baseline 实验整理报告

Last updated: 2026-04-29 Asia/Shanghai

本文档整理 DG-TWFD v3 当前可用 baseline 的实验标准、结果位置、数值结果与论文写作口径。目标不是宣称完整复现每个 baseline 论文表格，而是在 DG-TWFD 的实验限制下给出可审计、可追溯、尽量公平的横向比较。

## 1. 对比目标

DG-TWFD 需要回答的主要问题是：在相同数据集、相同分辨率、相同少步采样预算下，我们的方法相对已有 teacher sampler、consistency/distillation 模型和 schedule baseline 的生成质量如何。

当前 baseline 被分成三类：

| 类别 | 用途 | 论文使用建议 |
|---|---|---|
| 严格 50k 审计 | 对官方明确要求大样本 FID 的方法进行更严谨复评 | 可作为主要 CTM 对比结果，需注明是本地 checkpoint + EDM FID reference 的 50k audit |
| 5k fast comparison | 在统一少步网格上快速比较多类 baseline | 可用于内部公平对比或附表，表头必须标注 FID-5k |
| 占位/基础设施验证 | runner、schedule 或 checkpoint 映射尚未完全验证 | 不进入论文主表，不应写成有效 baseline |

## 2. 统一实验标准

当前公平对比协议如下：

| 项目 | 标准 |
|---|---|
| 数据集 | CIFAR-10 32x32、ImageNet64 64x64 |
| step 网格 | 1、2、4、8 |
| 指标 | FID；IS/Recall 暂未作为当前 baseline 主指标 |
| FID 实现 | `refs/edm/fid.py` |
| FID reference | EDM fid-refs：`cifar10-32x32.npz`、`imagenet-64x64.npz` |
| fast budget | 5000 samples，除非明确标注为 smoke |
| audit budget | 50000 samples，用于 baseline 官方说明要求大样本评估的情况 |
| checkpoint/code | 优先使用官方公开 checkpoint 与官方 sampling code |
| 调参原则 | 不做 method-specific 搜索；仅使用官方采样设置或明确记录的工程扩展 |
| 结果管理 | 旧 5k 结果不覆盖；严格复评写入新的 `/temp` 稳定目录 |

因此，论文中应避免把这些结果描述为“完全复现官方论文表格”。更准确的表述是：under a shared DG-TWFD evaluation protocol / under our fair-comparison protocol。

## 3. 结果与目录

稳定结果根目录：

```text
/temp/Zhengwei/projects/DG-TWFD/critical/analysis/
```

已有 5k baseline 汇总：

```text
results/baselines/
```

最终 CTM 50k 审计结果：

```text
/temp/Zhengwei/projects/DG-TWFD/critical/analysis/baselines_revalidated_20260428_final/baseline_ctm_50k_final.csv
/temp/Zhengwei/projects/DG-TWFD/critical/analysis/baselines_revalidated_20260428_final/reports/baseline_ctm_50k_final_summary.json
/temp/Zhengwei/projects/DG-TWFD/critical/analysis/baselines_revalidated_20260428_final/reports/baseline_ctm_50k_final_summary.csv
```

CTM no-GAN CIFAR-10 诊断性重训结果：

```text
/temp/Zhengwei/projects/DG-TWFD/critical/analysis/ctm_nogan_20260429/cifar10_ema010000_5k/eval/reports/summary.csv
/temp/Zhengwei/projects/DG-TWFD/critical/analysis/ctm_nogan_20260429/cifar10_ema010000_5k/results/ctm_nogan_20260429/baseline_ctm_nogan_cifar10_ema010000_5k.csv
/temp/Zhengwei/projects/DG-TWFD/critical/analysis/ctm_nogan_20260429/cifar10_ema010000_5k/samples_steps1_2_4_8.tar.gz
```

CTM 50k 样本目录：

```text
runs/baselines_revalidated_20260428/ctm_cifar10_50k/samples/steps{1,2,4,8}/images
runs/baselines_revalidated_20260428/ctm_imagenet64_50k/samples/steps{1,2}/images
runs/baselines_revalidated_20260428_recovery/ctm_imagenet64_50k_batch100/samples/steps{4,8}/images
```

以上 8 个 CTM 50k 样本目录均已核对为 50000 张 PNG。

## 4. CTM 50k 审计结果

CTM 是当前唯一已经完成严格 50k audit 的 baseline。重跑原因是 CTM CIFAR-10 官方说明要求至少 50k samples 才能进行正确 FID 评估；早期 5k fast result 只能保留为快速对比，不能解释为 CTM 官方级结果。

实验设置：

| 项目 | CIFAR-10 | ImageNet64 |
|---|---|---|
| sampling code | `refs/ctm-cifar10/image_sample.py` | `refs/ctm/code/image_sample.py` |
| checkpoint | `model043000.pt` | `ctm_imagenet64_ema_0.999.pt` |
| sampler | exact | exact |
| sample count | 50000 | 50000 |
| FID reference | EDM CIFAR-10 ref | EDM ImageNet64 ref |
| recovery note | 无 | step 4/8 使用 batch=100 recovery；算法、checkpoint、FID 标准未改变 |

最终 FID：

| Dataset | Method | Step | FID-50k |
|---|---|---:|---:|
| CIFAR-10 | CTM-official-cond | 1 | 1.743220 |
| CIFAR-10 | CTM-official-cond | 2 | 1.616910 |
| CIFAR-10 | CTM-official-cond | 4 | 1.830040 |
| CIFAR-10 | CTM-official-cond | 8 | 2.101430 |
| ImageNet64 | CTM-official | 1 | 2.379590 |
| ImageNet64 | CTM-official | 2 | 2.212310 |
| ImageNet64 | CTM-official | 4 | 2.893610 |
| ImageNet64 | CTM-official | 8 | 3.867740 |

注意事项：

| 项目 | 说明 |
|---|---|
| CIFAR-10 checkpoint | 本地下载目录只有 `model043000.pt` 和 optimizer 文件；未发现训练日志中提到的 EMA checkpoint 文件 |
| ImageNet64 reference | 当前使用 EDM `imagenet-64x64.npz`，不是 CTM 作者仓库可能使用的独立 author reference |
| 中断处理 | 原 ImageNet64 step 4 在旧运行根停于 192/200 npz shards，已明确排除；final 报告使用 recovery root 的完整 50k 结果 |
| 论文表述 | 建议写为 “CTM official checkpoint/code, exact sampler, 50k EDM-FID audit under our protocol” |

## 5. 5k Fast Baseline 结果

以下结果用于统一少步网格下的 fast comparison。除特别说明外，样本数为 5000。若放入论文，表头应写作 FID-5k，并在 caption 中说明这不是官方 full reproduction。

### CIFAR-10

| Method | Step 1 | Step 2 | Step 4 | Step 8 | 状态 |
|---|---:|---:|---:|---:|---|
| EDM official | 679.611000 | 473.607000 | 115.246000 | 33.067500 | 1024-sample smoke，不进最终表 |
| CTM-official-cond | 6.444080 | 6.249970 | 6.450460 | 6.848800 | 5k fast；已被 50k audit 替代为 CTM 主引用 |
| CTM-noGAN-DSM-10k-EMA0.999 | 11.862200 | 9.486860 | 9.441320 | 8.918910 | 本地 no-GAN CTM+DSM 10k 诊断性重训；不等同官方充分收敛模型 |
| TCM official | 7.168140 | 6.764130 | 6.904860 | 7.387160 | 1/2 step 接近官方用法；4/8 是几何扩展 |
| Entropic schedule + SDDIM | 387.689000 | 384.919000 | 114.771000 | 56.378900 | 本地 SDDIM 配置下的 schedule baseline |
| OptimalSteps-like | 377.867701 | 366.215745 | 369.488902 | 369.984571 | 旧 e405b failed checkpoint 上的基础设施验证，不进论文表 |

### ImageNet64

| Method | Step 1 | Step 2 | Step 4 | Step 8 | 状态 |
|---|---:|---:|---:|---:|---|
| EDM official | 623.860000 | 438.527000 | 93.126600 | 11.650400 | 5k fast teacher/sampler anchor |
| CD-LPIPS official | 13.025300 | 11.611200 | 11.559600 | 10.879900 | 5k fast；官方 checkpoint/code |
| CD-L2 official | 20.309900 | 14.257700 | 13.835400 | 12.411900 | 5k fast；官方 checkpoint/code |
| CT official | 19.141800 | 17.466500 | 18.837000 | 19.034100 | 5k fast；官方 checkpoint/code |
| CTM official | 8.824660 | 8.677950 | 9.262070 | 10.080900 | 5k fast；已被 50k audit 替代为 CTM 主引用 |
| TCM official | 10.960100 | 9.951880 | 10.508700 | 20.957300 | 1/2 step 接近官方用法；4/8 是几何扩展 |
| Entropic schedule + SDDIM | 288.205000 | 282.971000 | 80.052900 | 39.884100 | 本地 SDDIM 配置下的 schedule baseline |

## 6. EDM checkpoint 上的 schedule / time-warp follow-up

本轮新增 4 组 CIFAR-10 follow-up，目标是隔离“只改变 EDM teacher sampler 的时间网格”能带来的收益。它们不训练 DG-TWFD student，也不使用 DG-TWFD 的 learned map；所有方法共享同一个官方 EDM CIFAR-10 cond-VP checkpoint、同一批 seeds 和同一个 EDM FID reference。

产物位置：

```text
eval/edm_schedule_warp_5k_20260428/reports/summary.csv
results/baselines/edm_schedule_warp_5k_20260428/edm_schedule_warp_cifar10_5k_summary.csv
/temp/Zhengwei/projects/DG-TWFD/critical/analysis/edm_schedule_warp_5k_20260428.tar.gz
eval/edm_schedule_warp_5k_20260428/schedules/{strategy}/steps{1,2,4,8}.json
eval/edm_schedule_warp_5k_20260428/samples/{strategy}/steps{1,2,4,8}/images
eval/edm_schedule_warp_5k_20260428/previews/{strategy}_steps{1,2,4,8}.png
```

实验标准：

| 项目 | 设置 |
|---|---|
| checkpoint | `https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl` |
| FID reference | `https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz` |
| FID implementation | `refs/edm/fid.py` |
| sample count | 5000，seeds 0-4999 |
| step grid | 1、2、4、8 |
| NFE | `2 * steps - 1` |
| solver | deterministic EDM/Heun with custom sigma grid |
| batch | generation 512，FID 512 |
| runner | `scripts/baselines/run_edm_schedule_warp_eval.py` |

重要口径：

| 方法 | 实验含义 |
|---|---|
| `optimalsteps_edm` | 将 OptimalSteps 的 DP schedule search 适配到 EDM：64-step dense EDM/Heun teacher trajectory，search batch=8，用一次自定义 EDM interval 的 image-space MSE 作为 DP cost |
| `entropic_edm` | 只取 Entropic Time Scheduler 的预计算 CIFAR-10 time function，转换为 sigma grid 后仍用同一个 EDM cond-VP checkpoint 和同一个 EDM/Heun solver；该 schedule 文件标注为 uncond-VP，因此这是 schedule transfer，不是 Entropic 官方完整复现 |
| `piecewise_linear` | 用同一 EDM checkpoint 的 dense-grid 局部 proxy density 生成 piecewise-linear inverse-CDF time warp；proxy 是 Euler-Heun gap 加少量 step magnitude |
| `spline_warp` | 与 piecewise-linear 使用相同 proxy density，但用单调 PCHIP inverse-CDF spline 生成更平滑的 time warp |

`steps=1` 在该 runner 中明确定义为 `sigma_max -> 0` 的单次 EDM transition；这是为了避免官方 EDM `num_steps=1` 网格退化。该结果应作为本 follow-up 的自洽单步定义，不应直接与旧 EDM CIFAR-10 1024-sample smoke 行混用。

结果：

| Method | Step 1 | Step 2 | Step 4 | Step 8 | Best |
|---|---:|---:|---:|---:|---:|
| EDM + OptimalSteps-adapted schedule | 315.41200 | 235.05800 | 34.77990 | 9.90991 | 9.90991 |
| EDM + Entropic schedule | 315.41200 | 467.56400 | 135.38000 | 26.78320 | 26.78320 |
| EDM + piecewise-linear time warp | 315.41200 | 280.95100 | 229.90400 | 182.19500 | 182.19500 |
| EDM + spline time warp | 315.41200 | 281.01300 | 229.96700 | 182.28300 | 182.28300 |

解释：

1. `optimalsteps_edm` 是四组中最强的 schedule-only baseline，说明在相同 EDM checkpoint 下，仅通过 DP 选取 sigma grid 就能明显改善少步 EDM 采样，尤其是 step 4/8。
2. `entropic_edm` 在 step 8 有改善，但 step 2/4 不稳定。由于使用的是 uncond-VP 预计算 schedule transfer 到 cond-VP checkpoint，这一行只能说明该 schedule 在本协议下的表现。
3. `piecewise_linear` 与 `spline_warp` 是负向控制：当前 proxy density 将大量节点放在高噪声区，预览图也明显偏模糊，因此 FID 很差。它们不代表 DG-TWFD 的 learned warp 效果。
4. DG-TWFD 与这几组 baseline 的根本区别是：DG-TWFD 学习 student transition/map，并用 defect/consistency 信号优化；本 follow-up 只改变 teacher sampler 的时间坐标，不改变模型函数。

## 7. CTM checkpoint 上的 schedule / time-warp follow-up

本轮进一步把相同的 schedule/time-warp 问题放到 CTM 模型上测试。由于 CTM 的采样接口不是 EDM/Heun solver，不能简单复用 EDM runner；因此新增了 CTM 专用 runner，直接调用 CTM 的 exact transition：

```text
scripts/baselines/run_ctm_schedule_warp_eval.py
```

核心实验口径是：固定 CTM checkpoint、固定 FID 实现和 seeds，只改变 CTM 每一次 `G_theta(x_t,t,s)` transition 的 `t -> s` sigma 节点。也就是说，这组实验是真实作用在 CTM 模型的时间节点上，而不是把 EDM sampler 套到 CTM 上。

产物位置：

```text
eval/ctm_schedule_warp_5k_20260428/cifar10/reports/summary.csv
eval/ctm_schedule_warp_5k_20260428/imagenet64/reports/summary.csv
results/baselines/ctm_schedule_warp_5k_20260428/ctm_schedule_warp_cifar10_5k_summary.csv
results/baselines/ctm_schedule_warp_5k_20260428/ctm_schedule_warp_imagenet64_5k_summary.csv
runs/ctm_schedule_warp_5k_20260428/samples/{dataset}/{strategy}/steps{1,2,4,8}/images
eval/ctm_schedule_warp_5k_20260428/{dataset}/previews/{strategy}_steps{1,2,4,8}.png
eval/ctm_schedule_warp_5k_20260428/{dataset}/schedules/{strategy}/steps{1,2,4,8}.json
```

完整性检查：

| 项目 | 结果 |
|---|---|
| 数据集 | CIFAR-10、ImageNet64 |
| 方法数 | 4 |
| step 数 | 1、2、4、8 |
| 样本目录 | 32 个 |
| 每个目录图片数 | 5000 PNG |
| FID 样本数 | 5000 |

四个方法的定义：

| 方法 | 在 CTM 上的具体含义 |
|---|---|
| `optimalsteps_ctm` | 在 dense CTM exact trajectory 上做动态规划，最小化 sparse direct transition 与 dense CTM chain 的 image-space MSE |
| `entropic_ctm` | 读取 Entropic Time Scheduler 的预计算 time function，将其转换为 CTM sigma 节点，再用 CTM exact transition 执行 |
| `piecewise_linear_ctm` | 用 CTM 自身的 two-step vs one-step residual 作为 proxy density，通过 piecewise-linear inverse-CDF 生成时间 warp |
| `spline_warp_ctm` | 使用同一个 CTM residual proxy，但用单调 PCHIP spline inverse-CDF 生成更平滑的 warp |

实验标准：

| 项目 | CIFAR-10 | ImageNet64 |
|---|---|---|
| checkpoint | `model043000.pt` | `ctm_imagenet64_ema_0.999.pt` |
| transition | CTM exact `G_theta(x_t,t,s)` | CTM exact `G_theta(x_t,t,s)` |
| FID reference | EDM `cifar10-32x32.npz` | EDM `imagenet-64x64.npz` |
| sample count | 5000 | 5000 |
| generation batch | 500 | 250 |
| FID batch | 512 | 512 |

CIFAR-10 FID-5k：

| Method | Step 1 | Step 2 | Step 4 | Step 8 | Best |
|---|---:|---:|---:|---:|---:|
| CTM + OptimalSteps-adapted schedule | 6.39022 | 6.33884 | 6.41577 | 6.62873 | 6.33884 |
| CTM + Entropic schedule | 6.39022 | 6.31943 | 6.66228 | 6.98590 | 6.31943 |
| CTM + piecewise-linear time warp | 6.39022 | 6.37918 | 6.37642 | 6.37383 | 6.37383 |
| CTM + spline time warp | 6.39022 | 6.37878 | 6.37888 | 6.37933 | 6.37878 |

ImageNet64 FID-5k：

| Method | Step 1 | Step 2 | Step 4 | Step 8 | Best |
|---|---:|---:|---:|---:|---:|
| CTM + OptimalSteps-adapted schedule | 8.85793 | 8.93934 | 9.24441 | 9.82932 | 8.85793 |
| CTM + Entropic schedule | 8.85793 | 9.42372 | 9.80729 | 11.01760 | 8.85793 |
| CTM + piecewise-linear time warp | 8.85793 | 9.51683 | 8.93845 | 8.84899 | 8.84899 |
| CTM + spline time warp | 8.85793 | 9.46341 | 8.92669 | 8.84654 | 8.84654 |

结果解释：

1. CIFAR-10 上四类 CTM schedule 的差异很小，说明在当前 checkpoint 和 FID-5k 协议下，CTM exact transition 对这些外部节点变化并不敏感；`entropic_ctm` 的 step 2 略低，但 step 4/8 变差。
2. ImageNet64 上 `piecewise_linear_ctm` 和 `spline_warp_ctm` 在 step 8 略优于单步和其他 schedule，但增益很小；`optimalsteps_ctm` 和 `entropic_ctm` 随步数增加反而变差。
3. 这组实验不能替代 CTM 50k audit。它的用途是回答“只改变 CTM 的时间节点是否足够带来显著收益”。当前证据显示，单纯 schedule/warp 的收益有限且不稳定。
4. 与 DG-TWFD 的区别在于：DG-TWFD 学习或优化 student transition/map，并使用 defect/consistency 相关目标；本组 CTM follow-up 不训练模型，只在评估时改变 `t -> s` 节点。因此它是 schedule-only 对照，而不是 DG-TWFD 的消融替代。

论文使用建议：

```text
We additionally evaluate schedule-only variants on top of CTM by directly
changing the sigma nodes of CTM exact transitions. These diagnostics isolate
the effect of evaluation-time node selection from model training. Under the
5k DG-TWFD protocol, changing CTM time nodes gives only marginal and unstable
improvements, suggesting that the gains of DG-TWFD should not be attributed to
schedule changes alone.
```

中文可以写作：

```text
为了验证收益是否仅来自采样时间节点的重新分配，我们在固定 CTM checkpoint 的条件下，
直接改变 CTM exact transition 的 sigma 节点，并比较 OptimalSteps、Entropic、
piecewise-linear warp 与 spline warp。结果显示，在 FID-5k 协议下，这些
schedule-only 改动只带来很小且不稳定的变化，因此 DG-TWFD 的优势不能简单解释为
外部 schedule/warp 的效果。
```

## 8. CTM no-GAN 诊断性重训

本轮新增 CTM no-GAN 诊断性 baseline，目标是检查“不使用 GAN 辅助训练时，CTM+DSM 路径在当前工程预算下的表现”。该实验不是 CTM 论文官方 fully trained checkpoint 的替代，也不应写成官方 no-GAN 上限；它的作用是给出当前本地 no-GAN 训练路径的可审计结果。

训练设置：

| 项目 | 设置 |
|---|---|
| 数据集 | CIFAR-10 32x32 class-conditional |
| 训练代码 | `refs/ctm-cifar10/cm_train.py` |
| 配置来源 | `refs/ctm-cifar10/commands/cond_CTM+DSM_command.sh` 的 CTM+DSM no-GAN 路径 |
| 本地 launcher | `scripts/baselines/launch_ctm_cifar10_nogan_train.sh` |
| teacher | EDM CIFAR-10 cond-VP checkpoint |
| GAN 设置 | 未启用 `gan_training` / discriminator loss |
| batch | `global_batch_size=16`，`microbatch=4` |
| total steps | 10000 |
| evaluated checkpoint | `ema_0.999_010000.pt` |
| 显存约束 | baseline 进程自身约 6GB，低于 50GB 限制 |

checkpoint 与日志稳定位置：

```text
/temp/Zhengwei/projects/DG-TWFD/critical/ckpt/ctm_nogan_20260429/cifar10_nogan_dsm_10k_mb4_gb16_resume_from8000/ema_0.999_010000.pt
/temp/Zhengwei/projects/DG-TWFD/critical/logs/ctm_nogan_20260429/cifar10_nogan_dsm_10k_mb4_gb16_resume_from8000/progress.csv
/temp/Zhengwei/projects/DG-TWFD/critical/logs/ctm_nogan_20260429/cifar10_nogan_dsm_10k_mb4_gb16_resume_from8000/log.txt
```

评估设置：

| 项目 | 设置 |
|---|---|
| runner | `scripts/baselines/run_ctm_cifar10_eval.py` |
| sampling code | `refs/ctm-cifar10/image_sample.py` |
| sampler | official CTM `exact` |
| step grid | 1、2、4、8 |
| sample count | 5000 |
| batch | 500 |
| FID implementation | `refs/edm/fid.py` |
| FID reference | EDM `cifar10-32x32.npz` |

FID-5k 结果：

| Dataset | Method | Step 1 | Step 2 | Step 4 | Step 8 | Best |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-10 | CTM-noGAN-DSM-10k-EMA0.999 | 11.862200 | 9.486860 | 9.441320 | 8.918910 | 8.918910 |

产物位置：

```text
eval/ctm_nogan_20260429/cifar10_ema010000_5k/reports/summary.csv
results/baselines/ctm_nogan_20260429/baseline_ctm_nogan_cifar10_ema010000_5k.csv
runs/ctm_nogan_20260429/cifar10_ema010000_5k/samples/steps{1,2,4,8}/images
/temp/Zhengwei/projects/DG-TWFD/critical/analysis/ctm_nogan_20260429/cifar10_ema010000_5k/
```

完整性检查：

| 项目 | 结果 |
|---|---|
| step 数 | 1、2、4、8 |
| 每个 step PNG 数 | 5000 |
| summary | `summary.csv`、`summary.json` 均存在 |
| stable 样本归档 | `samples_steps1_2_4_8.tar.gz` |

ImageNet64 no-GAN 当前未启动。原因是本地只找到 ImageNet64 评估/样本残留和官方 CTM checkpoint，未找到可用于重新训练的真实 ImageNet64 train set，也未找到 CTM 训练命令要求的 EDM ImageNet64 teacher checkpoint。因此当前环境下直接启动 ImageNet64 no-GAN 训练会变成不可审计实验，不符合 baseline 记录标准。

解释：

1. 该 no-GAN 结果明显弱于 CTM official 50k audit，符合“短预算本地重训未充分收敛”的预期。
2. step 数增加时 FID 有改善，说明 checkpoint 仍能从多步 exact transition 受益。
3. 论文中只能把该结果写为 diagnostic retraining baseline，用于说明当前 no-GAN CTM+DSM 本地训练路径；不能用它替代官方 CTM 主 baseline。

## 9. 各 baseline 的论文解释口径

### EDM official

EDM 在这里主要是 teacher/sampler anchor。ImageNet64 有 5k fast result；CIFAR-10 当前只有 1024-sample smoke/reference result，因此不能作为最终 CIFAR-10 baseline 表格行。若论文主表需要 EDM CIFAR-10 或更严格 ImageNet64，应按与 DG-TWFD 相同预算重新跑并明确标注样本数。

### OpenAI Consistency Models CD/CT

CD-LPIPS、CD-L2、CT 使用官方 ImageNet64 checkpoint/code，并在共享 step grid 上进行 5k fast FID。它们适合说明 DG-TWFD 与 consistency/distillation 系列方法在统一少步预算下的差异，但当前不应写成官方 50k reproduction。

### CTM

CTM 是当前最严格的外部 baseline：已经完成 50k audit，且结果与 “CTM 在大样本 FID 下可达到低 FID” 的趋势一致。论文中可以优先引用 final 50k audit，而不是早期 5k CTM fast result。

新增 no-GAN CTM+DSM 10k 重训只作为诊断性结果，不能与官方 fully trained CTM checkpoint 混为一谈。若论文需要正式 no-GAN CTM 对比，需要补齐官方规模训练预算、完整数据和 teacher 资产后重新训练。

### TCM

TCM 使用官方 checkpoint/code。step 1 使用官方 1-step 路径，step 2 使用 README 中的 `mid_t=0.821` 设置；step 4 和 step 8 是为了统一少步网格而做的几何扩展。因此 TCM 4/8 只能写成 engineering extension under our grid，不能写成官方 TCM 4/8 设置。

### Entropic schedule

Entropic 结果是将官方预计算 schedule 放入本地 SDDIM evaluation path 后得到的 5k fast baseline。当前大 FID 应解释为该 schedule 在当前本地 solver/config 下的表现，不能泛化为 Entropic 方法本身的理论上限。

### AYS / OptimalSteps

AYS CIFAR-10、AYS ImageNet64、OptimalSteps ImageNet64 目前没有 verified local runner 或 schedule mapping。OptimalSteps-like CIFAR-10 只是在旧 e405b failed checkpoint 上做的基础设施验证。相关结果不应进入论文主表。

## 10. 建议的论文表格组织

推荐至少分两张表或在 caption 中清楚区分：

| 表格 | 内容 | 标注 |
|---|---|---|
| 主表或强 baseline 表 | DG-TWFD vs CTM 50k audit；必要时加入 DG-TWFD 同预算结果 | FID-50k；shared DG-TWFD evaluation protocol |
| fast comparison 表/附表 | CD/CT/TCM/Entropic/EDM ImageNet64 等 5k fast baseline | FID-5k；not official full reproduction |
| diagnostic 表/附表 | CTM no-GAN 10k、schedule-only 诊断实验 | FID-5k；diagnostic retraining or schedule-only |
| ablation/support 表 | OptimalSteps-like、smoke runs、blocked schedule placeholders | 仅作基础设施或失败路径说明，不作为 SOTA 对比 |

推荐表述：

```text
We evaluate external baselines under a shared DG-TWFD protocol: same dataset
resolution, same 1/2/4/8 step grid, and EDM fid.py reference statistics.
Unless otherwise stated, fast-comparison baselines use 5k generated samples.
For CTM, because the official notes require large-sample FID evaluation, we
additionally report a 50k-sample audit using the official CTM sampling code and
locally available official checkpoints.
```

中文写作可表述为：

```text
为了避免将不同论文中的指标设置直接混用，我们在统一的 DG-TWFD 评估协议下重新评估外部 baseline。
该协议固定数据集分辨率、采样步数网格、FID 实现和 reference statistics。对于 CTM 这类官方说明要求
大样本 FID 的方法，我们额外进行了 50k 样本审计；对于其他可运行 baseline，当前结果主要作为 5k
快速公平对比，不等同于官方论文表格复现。
```

## 11. 当前结论

1. CTM 已完成严格 50k audit，是当前最可靠的外部 baseline 结果。
2. CD/CT、TCM、Entropic、EDM ImageNet64 可作为 5k fast comparison，但应明确样本预算和协议差异。
3. CTM schedule/time-warp follow-up 已完成，可作为 schedule-only 诊断表或补充材料，用来说明固定 CTM checkpoint 时单纯改时间节点的收益有限。
4. CTM no-GAN CIFAR-10 10k 重训已完成，可作为诊断性 no-GAN 训练路径结果；ImageNet64 no-GAN 因训练数据和 teacher 资产缺失暂不启动。
5. EDM CIFAR-10 smoke、AYS、OptimalSteps-like 不能作为最终论文主表 baseline。
6. 如果最终论文要求所有 baseline 与 DG-TWFD 使用完全相同样本预算，则需要对尚未 50k 的 5k fast baselines 继续复评；否则必须在表格标题或 caption 中明确区分 FID-5k 与 FID-50k。
