# DGTD v3 Round-4 Online Mainline Verification

## Verdict

**结论：有条件通过**

这轮 patch 在代码语义上已经把 online teacher 从“仅在线轨迹/data provider”推进到了
DGTD continuation 主路径，且 bridge branch 在 online source 下可训练。  
但本次验收只做了静态代码核查和本地最小单测/梯度检查，没有做服务器 smoke/short-run
实跑，因此不建议直接跳 full run。

## Scope

本轮检查了：

- `docs/dgtd_v3_round4_online_mainline_patch_notes.md`
- `src/dgtd/cache.py`
- `src/dgtd/teacher.py`
- `src/dgtd/defect.py`
- `src/dgtd/train_dgtd.py`
- `src/dgtd/sigma.py`
- `src/dgtd/metrics.py`
- `src/dgfm/teachers/diffusers_ddpm.py`
- `configs/experiment/dgtd_cifar10_v3.yaml`
- `configs/experiment/dgtd_cifar10_v3_smoke.yaml`
- `tests/test_dgtd.py`

本地最小验证：

- `pytest tests/test_dgtd.py -q` -> `14 passed`
- 额外运行了一个小型 autograd 脚本，验证 online source 的 continuation 对 `z_s` 有梯度，
  且 `eta=0.95` 时 bridge 侧梯度非零

## 1. Online continuation 是否真的成为主源

### 1.1 Online teacher data path 仍然正常

`DGTDTrainer.run()` 先构建并 `prepare()` online teacher，然后通过
`select_dgtd_dataloaders(...)` 决定使用 image loader 还是 cache loader。

关键调用链：

- image loader 选择：[`src/dgtd/train_dgtd.py:117`](../src/dgtd/train_dgtd.py#L117)
- online batch 提取：[`src/dgtd/train_dgtd.py:80`](../src/dgtd/train_dgtd.py#L80)
- online trajectory materialization：[`src/dgtd/train_dgtd.py:330`](../src/dgtd/train_dgtd.py#L330)

当 `dgtd.use_online_teacher_data=true` 且 teacher 可构建时，训练不再读取 trajectory
cache，而是从图像 batch 构建在线 trajectory。

### 1.2 Online continuation 已经进入 residual 主路径

`TeacherAdapter.online_trajectory_from_x0(...)` 会把在线 rollout 包装为带
`teacher_anchor_online=true` 的 trajectory payload：

- online anchor 标记：[`src/dgtd/cache.py:43`](../src/dgtd/cache.py#L43)
- online trajectory 包装：[`src/dgtd/teacher.py:64`](../src/dgtd/teacher.py#L64)

`TeacherAdapter.local_flow(...)` 在进入 cached fallback 之前，优先检查
`teacher_anchor_online`：

- online anchor mask：[`src/dgtd/teacher.py:97`](../src/dgtd/teacher.py#L97)
- online continuation branch：[`src/dgtd/teacher.py:121`](../src/dgtd/teacher.py#L121)

满足 `used_online_anchor.any()` 且
`online_continuation_mode == "affine_mainline"` 时，它直接返回
`source_ids = CONTINUATION_SOURCE_ONLINE`，不会再把 online trajectory 名义上记成
`cached_affine/cached_exact`。

因此，代码逻辑上 `continuation_sources.online` 现在确实可以大于 `0`，并且在
online teacher data 主线下应成为默认主源。

### 1.3 Cached source 已退为 fallback

source 优先级现在是：

1. online anchor affine mainline
2. `online_teacher.local_flow(...)`，如果 teacher 真有这个接口
3. cached exact / cached affine
4. bootstrap

对应实现位置：

- online mainline：[`src/dgtd/teacher.py:122`](../src/dgtd/teacher.py#L122)
- optional teacher-native local flow：[`src/dgtd/teacher.py:140`](../src/dgtd/teacher.py#L140)
- cached fallback：[`src/dgtd/teacher.py:153`](../src/dgtd/teacher.py#L153)

`DiffusersDDPMTeacher` 当前并没有 `local_flow(...)`，只有 trajectory rollout：

- [`src/dgfm/teachers/diffusers_ddpm.py:97`](../src/dgfm/teachers/diffusers_ddpm.py#L97)

所以当前“online continuation”的真实含义是：

- online rollout 提供 teacher anchors
- DGTD continuation 用这些在线 anchors 构建 affine / Jacobian-lite mainline

这符合 patch notes 的设计目标，但它不是 teacher 内部的原生 online local solver。

## 2. Online continuation 的数学与梯度性质

### 2.1 公式已按 online affine / Jacobian-lite 实现

实现是：

```text
x_u_teacher_online.detach()
+ alpha(s,u) * ( z_s - x_s_teacher_online.detach() )
```

对应代码：

- [`src/dgtd/teacher.py:128`](../src/dgtd/teacher.py#L128)
- [`src/dgtd/teacher.py:129`](../src/dgtd/teacher.py#L129)

这与 patch note 里的目标形式一致：

\[
x_u^{T,online} + \alpha(s,u)(z_s - x_s^{T,online})
\]

### 2.2 Teacher anchor 仍然 detach

detach 点很清楚：

- `x_u_online.detach()`
- `x_s_online.detach()`
- 返回的 `teacher_s/teacher_u/rel_error/alpha` 也都 detach

对应代码：

- [`src/dgtd/teacher.py:129`](../src/dgtd/teacher.py#L129)
- [`src/dgtd/teacher.py:133`](../src/dgtd/teacher.py#L133)

所以 teacher 端仍是固定 anchor，不会反向污染 teacher 分支。

### 2.3 continuation 对 `z_s` 有梯度依赖，bridge 在 online source 下可训练

`compute_dgtd_residual(...)` 的关键链路是：

```text
x_s_pred = student(x_t, t, s)
teacher_cont = online_affine(x_s_pred)
bridge_cont = student(x_s_pred, s, u)
x_u_cont = eta * teacher_cont + (1 - eta) * bridge_cont
residual = x_u_direct - x_u_cont
```

对应代码：

- `x_s_pred`：[`src/dgtd/defect.py:26`](../src/dgtd/defect.py#L26)
- teacher_cont 输入 `x_s_pred`：[`src/dgtd/defect.py:28`](../src/dgtd/defect.py#L28)
- `bridge_cont`：[`src/dgtd/defect.py:35`](../src/dgtd/defect.py#L35)
- `x_u_cont` 混合：[`src/dgtd/defect.py:42`](../src/dgtd/defect.py#L42)

因此 online source 下的梯度路径是：

```text
loss
 -> residual
 -> x_u_cont
 -> teacher_cont( z_s = x_s_pred )    [eta 路径]
 -> bridge_cont( x_s_pred )           [1-eta 路径]
 -> x_s_pred
 -> student parameters
```

本地 autograd 验证结果：

- `source_online_all True`
- `used_online_anchor_all True`
- `z_grad_abs_sum 0.40000003576278687`
- `bridge_grad_with_online_eta095 0.7899999618530273`

这说明：

- online continuation 本身对 `z_s` 有非零梯度
- 即使 `eta=0.95`，bridge 侧参数梯度仍非零

## 3. Warmup 是否修正到位

### 3.1 `eta` 不再从 1.0 开始

默认 config 已改为：

- `eta_start: 0.95`

位置：

- full config：[`configs/experiment/dgtd_cifar10_v3.yaml:50`](../configs/experiment/dgtd_cifar10_v3.yaml#L50)
- smoke config：[`configs/experiment/dgtd_cifar10_v3_smoke.yaml:23`](../configs/experiment/dgtd_cifar10_v3_smoke.yaml#L23)

warmup 调度实现：

- [`src/dgtd/train_dgtd.py:157`](../src/dgtd/train_dgtd.py#L157)
- [`src/dgtd/train_dgtd.py:166`](../src/dgtd/train_dgtd.py#L166)

### 3.2 这次不是单靠“把 eta 降低一点”硬修

真正的修正有两层：

1. warmup 不再精确等于 `eta=1.0`
2. 更重要的是，online continuation 本身依赖 `z_s`

所以即便 teacher 权重仍很高，teacher-cont 路径也会对 `x_s_pred` 反传梯度。
这比单纯降低 `eta` 更稳，因为它保留了 teacher 主导，同时避免 warmup 退化成
明显的 direct-only。

### 3.3 是否还残留 direct-only corner case

对 online mainline 来说，明显的 direct-only corner case 已基本被消掉。  
但残余风险仍在 fallback 路径：

- 如果运行中退回 `cached_exact`
- 且 `eta` 很高

那么 exact 部分仍然是 detached anchor，bridge 梯度主要来自 `(1-eta) * bridge_cont`
这一路。由于现在 `eta_start=0.95` 而不是 `1.0`，这个 corner case 被弱化了，但没有从
所有 source 上彻底消失。

我的判断：

- 对 online mainline：warmup 设计已经足够稳，可以进入 short run
- 对 cached fallback：仍应监控 `cached_fallback_rate` 和 `exact_mask_hit_rate`

## 4. Alpha 设计

### 4.1 `alpha_online(s,u)` 来自统一 sigma 体系

`TeacherAdapter.local_gain(...)` 直接调用统一 sigma schedule：

- [`src/dgtd/teacher.py:86`](../src/dgtd/teacher.py#L86)

而 `SigmaSchedule.alpha(...)` 同时服务：

- teacher online continuation
- cached affine continuation
- `lambda_hf_weight`
- `edm_weight`
- `min_snr_weight`

其中后三者通过 `sigma_fn=teacher_adapter.sigma` 进入统一 sigma：

- `metric_norm` / `lambda_hf_weight`：[`src/dgtd/train_dgtd.py:378`](../src/dgtd/train_dgtd.py#L378)
- `edm_weight`：[`src/dgtd/train_dgtd.py:405`](../src/dgtd/train_dgtd.py#L405)
- `min_snr_weight`：[`src/dgtd/train_dgtd.py:410`](../src/dgtd/train_dgtd.py#L410)

### 4.2 默认 alpha 设计合理且有 safe range

默认：

- `sigma_mode: linear_1mt`
- `alpha_mode: clamped_ratio_sigma`
- `alpha_min: 0.05`
- `alpha_max: 1.0`

实现：

- sigma：[`src/dgtd/sigma.py:21`](../src/dgtd/sigma.py#L21)
- alpha：[`src/dgtd/sigma.py:36`](../src/dgtd/sigma.py#L36)

在当前时间语义下，`t < s < u` 且 `sigma(t)=1-t` 单调下降，所以通常
`sigma(u) / sigma(s) <= 1`。`clamped_ratio_sigma` 进一步避免 clean-end 过小或异常比值。

本地最小脚本里，`s=0.5, u=1.0` 时得到：

- `alpha = 0.05`

这和 `alpha_min` clamp 一致，说明 clean end 不会因为 ratio 过小而直接失去 bridge
耦合。

### 4.3 一个需要继续盯的风险

`metrics._resolve_sigma(...)` 在未显式传 `sigma_fn` 时仍回退到 `1 - t`：

- [`src/dgtd/metrics.py:24`](../src/dgtd/metrics.py#L24)

当前 DGTD trainer 已显式传入 `teacher_adapter.sigma`，所以本主线没问题；但如果未来有
别的调用点漏传 `sigma_fn`，仍可能重新引入隐式 `1-t` 假设。

## 5. Source 语义和日志

### 5.1 训练内部统计已足够支撑研究判断

`_run_epoch(...)` 里已实际累加并导出：

- `online_anchor_used_rate`
- `online_continuation_rate`
- `cached_fallback_rate`
- `exact_mask_hit_rate`
- `teacher_rel_error_mean`
- `alpha_online_mean`
- `alpha_online_min`
- `alpha_online_max`
- `continuation_sources`

统计位置：

- source / alpha 累加：[`src/dgtd/train_dgtd.py:507`](../src/dgtd/train_dgtd.py#L507)
- rate 生成：[`src/dgtd/train_dgtd.py:645`](../src/dgtd/train_dgtd.py#L645)
- continuation_sources 写入：[`src/dgtd/train_dgtd.py:689`](../src/dgtd/train_dgtd.py#L689)

### 5.2 epoch JSON payload 也真的写出了这些字段

写入 `train.jsonl` 的 payload 包含：

- `online_anchor_used_rate`
- `online_continuation_rate`
- `cached_fallback_rate`
- `exact_mask_hit_rate`
- `alpha_online_mean/min/max`
- `train_bridge_state_teacher_error`
- `train_bridge_u_teacher_error`
- `train_teacher_rel_error_mean`
- `continuation_sources`

位置：

- [`src/dgtd/train_dgtd.py:794`](../src/dgtd/train_dgtd.py#L794)

### 5.3 命名比上一轮清楚，但还有一处可再改进

这次把旧的 bridge 命名拆成：

- `bridge_state_teacher_error = MSE(x_s_pred, x_s_teacher)`
- `bridge_u_teacher_error = MSE(bridge_cont, x_u_teacher)`

实现：

- [`src/dgtd/train_dgtd.py:433`](../src/dgtd/train_dgtd.py#L433)
- [`src/dgtd/train_dgtd.py:434`](../src/dgtd/train_dgtd.py#L434)

这比 round-2 的 `bridge_teacher_error` 清楚很多。  
目前我认为 source 语义和日志已经足够支撑下一轮 short run。

## 6. Smoke / short-run 可行性

### 6.1 本轮没有服务器 smoke 结果

这次验收没有服务器环境结果，因此我不能用运行结果证明：

- `continuation_sources.online` 在真实 smoke 中一定非零
- online continuation 在实际 DDPM rollout 上一定占主导

### 6.2 本地最小一致性结果

本地可以确认的最小事实是：

- `pytest tests/test_dgtd.py -q` -> `14 passed`
- `test_online_trajectory_uses_online_continuation_source(...)` 明确验证了
  online trajectory 会得到 `source=online`
- `test_eta_below_one_keeps_bridge_gradient_with_exact_cached_teacher(...)`
  说明即便 fallback 到 cached exact，只要 `eta < 1`，bridge 梯度仍不为零

### 6.3 进入下一步的建议

建议：

- **可以进入 short run**
- **不建议直接 full run**

short run 的主要目标不是“看 loss 漂不漂亮”，而是确认：

- `online_teacher_data=true`
- `online_continuation_rate > 0`
- `cached_fallback_rate` 不高
- `alpha_online_mean/min/max` 分布合理
- `bridge_state_teacher_error` / `bridge_u_teacher_error` 没有明显恶化

## 7. Final Assessment

### 通过点

1. online trajectory anchor 已真实进入 DGTD continuation 主路径，不再只是 data provider。
2. online continuation 使用 detached teacher anchors + 对 `z_s` 的 affine 依赖，bridge 在 online source 下可训练。
3. warmup 不再是明显 direct-only；日志字段也足以区分 online 主源和 cached fallback。

### 剩余风险

1. 还没有服务器 smoke / short-run 结果，无法确认 `online_continuation_rate` 在真实 DDPM teacher 下确实稳定大于 0。
2. 当前“online continuation”依然是基于 online-generated trajectory anchors 的 affine mainline，而不是 teacher-native `local_flow`；命名上虽然已清晰，但能力边界要认清。
3. cached fallback，尤其 `cached_exact`，仍然保留高 teacher 权重下的弱 direct-only 风险；需要用 `cached_fallback_rate` 和 `exact_mask_hit_rate` 盯实跑日志。

## Recommendation

- **是否建议进入 short run：建议**
- **是否建议直接进入 full run：不建议**

