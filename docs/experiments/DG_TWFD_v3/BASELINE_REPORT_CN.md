# DG-TWFD v3 Baseline 实验整理报告

Last updated: 2026-04-28 Asia/Shanghai

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

## 6. 各 baseline 的论文解释口径

### EDM official

EDM 在这里主要是 teacher/sampler anchor。ImageNet64 有 5k fast result；CIFAR-10 当前只有 1024-sample smoke/reference result，因此不能作为最终 CIFAR-10 baseline 表格行。若论文主表需要 EDM CIFAR-10 或更严格 ImageNet64，应按与 DG-TWFD 相同预算重新跑并明确标注样本数。

### OpenAI Consistency Models CD/CT

CD-LPIPS、CD-L2、CT 使用官方 ImageNet64 checkpoint/code，并在共享 step grid 上进行 5k fast FID。它们适合说明 DG-TWFD 与 consistency/distillation 系列方法在统一少步预算下的差异，但当前不应写成官方 50k reproduction。

### CTM

CTM 是当前最严格的外部 baseline：已经完成 50k audit，且结果与 “CTM 在大样本 FID 下可达到低 FID” 的趋势一致。论文中可以优先引用 final 50k audit，而不是早期 5k CTM fast result。

### TCM

TCM 使用官方 checkpoint/code。step 1 使用官方 1-step 路径，step 2 使用 README 中的 `mid_t=0.821` 设置；step 4 和 step 8 是为了统一少步网格而做的几何扩展。因此 TCM 4/8 只能写成 engineering extension under our grid，不能写成官方 TCM 4/8 设置。

### Entropic schedule

Entropic 结果是将官方预计算 schedule 放入本地 SDDIM evaluation path 后得到的 5k fast baseline。当前大 FID 应解释为该 schedule 在当前本地 solver/config 下的表现，不能泛化为 Entropic 方法本身的理论上限。

### AYS / OptimalSteps

AYS CIFAR-10、AYS ImageNet64、OptimalSteps ImageNet64 目前没有 verified local runner 或 schedule mapping。OptimalSteps-like CIFAR-10 只是在旧 e405b failed checkpoint 上做的基础设施验证。相关结果不应进入论文主表。

## 7. 建议的论文表格组织

推荐至少分两张表或在 caption 中清楚区分：

| 表格 | 内容 | 标注 |
|---|---|---|
| 主表或强 baseline 表 | DG-TWFD vs CTM 50k audit；必要时加入 DG-TWFD 同预算结果 | FID-50k；shared DG-TWFD evaluation protocol |
| fast comparison 表/附表 | CD/CT/TCM/Entropic/EDM ImageNet64 等 5k fast baseline | FID-5k；not official full reproduction |
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

## 8. 当前结论

1. CTM 已完成严格 50k audit，是当前最可靠的外部 baseline 结果。
2. CD/CT、TCM、Entropic、EDM ImageNet64 可作为 5k fast comparison，但应明确样本预算和协议差异。
3. EDM CIFAR-10 smoke、AYS、OptimalSteps-like 不能作为最终论文主表 baseline。
4. 如果最终论文要求所有 baseline 与 DG-TWFD 使用完全相同样本预算，则需要对尚未 50k 的 5k fast baselines 继续复评；否则必须在表格标题或 caption 中明确区分 FID-5k 与 FID-50k。
