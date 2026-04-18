# Prompt — 设计并执行 DGTD online-mainline 的完整实验计划（以 full run 为目标）

你现在不要继续做功能开发，当前主任务是：**围绕已经打通的 online teacher continuation 主线，完成一套严谨的 full-run 实验设计与执行方案**，并输出可以支撑研究决策的日志、图表和结论。

## 0. 实验目标
当前分支已经在 smoke 级别证明：
- online continuation 能进入 DGTD residual 主路径
- bridge branch 在 online source 下可训练
- warmup 不再明显 direct-only
- eval 闭环能跑通

但这些都只是 smoke 证据，不足以证明：
1. 更长训练下是否稳定
2. online mainline 是否真的优于 cached-fallback 主线
3. warp 是否在中长期训练中学偏
4. few-step 质量是否实质提升

所以本轮实验的目标不是“再验证能不能跑”，而是回答下面四个研究问题：

### Q1. Stability
online continuation 主线在 short run / full run 下是否稳定，不会导致 loss、defect、warp collapse 或 clean-end 过耦合？

### Q2. Mainline value
相比旧的 cached-style continuation 主线，online continuation 是否带来更好的 bridge consistency、few-step FID 和更合理的 warp density？

### Q3. Warp validity
defect-guided warp 在 online mainline 下是否真的学到有意义的时间重分配，而不是简单向 clean end 坍缩？

### Q4. Cost-quality tradeoff
online continuation 带来的训练吞吐损失是否值得，其质量收益是否足够支撑主线继续沿此方向推进？

---

## 1. 核心原则
你不要一次性直接无脑跑最长 full run。  
你必须执行**分阶段 gated 实验**，每一阶段通过后才能进入下一阶段。

要求：
- 保持当前 online-mainline 代码不再大改
- 重点做训练/评估/对比实验
- 若发现严重异常，可以只做最小 instrumentation 补充，但不要再做方法改写
- 不要在这轮引入新的 loss 或新 teacher 机制

---

## 2. 实验分阶段设计

## Stage A — Online Mainline Short-Run Sanity
目的：
- 验证 online mainline 在超出 smoke 的训练长度下仍稳定
- 观察 warmup -> adaptive 的切换是否健康
- 检查 online continuation 是否持续是主源而非短暂现象

### Stage A 必做内容
使用 online mainline 配置，进行一个明显长于 smoke、但明显短于 full run 的 short run。

### Stage A 必须关注的指标
按 epoch 或固定 interval 记录：

#### 训练核心量
- train_loss
- val_loss
- train_defect
- val_defect
- train_direct_teacher_error
- val_direct_teacher_error
- train_bridge_state_teacher_error
- val_bridge_state_teacher_error
- train_bridge_u_teacher_error
- val_bridge_u_teacher_error
- train_direct_bridge_gap
- val_direct_bridge_gap

#### online continuation 相关
- continuation_sources
- online_continuation_rate
- cached_fallback_rate
- teacher_rel_error_mean
- exact_mask_hit_rate（若仍保留 exact gate）
- alpha_online_mean
- alpha_online_min
- alpha_online_max
- alpha saturation ratio（如果实现方便，请统计 hit alpha_min / alpha_max 的比例）

#### warp 相关
- q_phi
- q_D
- D_bar
- K_bar
- HF_bar
- entropy_q_phi
- KL(q_D || q_phi)
- max(q_phi / q_base)
- argmax_q_phi
- stage
- eta
- beta

#### 分区间统计
必须分 noisy / mid / clean 三段，分别记录：
- defect
- direct teacher error
- bridge state teacher error
- bridge u teacher error
- HF residual

#### 训练代价
- step time
- epoch time
- images/sec 或 samples/sec
- max GPU memory
- online trajectory generation time（如果可拆）
- continuation extra time（如果可拆）

### Stage A 升级到 Stage B 的判定标准
你必须根据结果给出“是否进入更长实验”的判断，至少要回答：

1. online continuation 是否持续占主导（不只是首个 epoch）
2. bridge gap 是否在下降，而不是长期停滞
3. q_phi 是否没有明显坍缩
4. alpha_online 是否没有在 clean-end 大面积饱和
5. 没有明显的 NaN、loss 发散、极端 throughput 崩坏

---

## Stage B — Mainline vs Baseline Controlled Comparison
目的：
回答 online continuation 主线到底有没有研究价值。

### Stage B 必做对比
至少做下面两组：

#### B1. 当前 online mainline
就是本轮已验证通过的 online continuation 主线。

#### B2. 对照组：cached-style continuation 主线
保持其他设置尽量一致，只把 continuation 主路径退回 cached-style / old fallback 主线。

如果预算允许，再加：

#### B3. online mainline but warp frozen
用于区分“online continuation 的收益”和“warp 学习的收益”。

### Stage B 必须比较的结果
对每组都输出：

#### few-step 生成质量
- FID @ 1 step
- FID @ 2 steps
- FID @ 4 steps
- FID @ 8 steps
- 若预算允许，再加 16 steps

#### 训练诊断
- final continuation source 分布
- final direct_bridge_gap
- final entropy_q_phi
- final KL(q_D || q_phi)
- final teacher_rel_error_mean
- alpha_online 统计

#### 计算代价
- total training wall-clock
- avg step time
- peak memory
- eval latency（按 step count）

#### 可视化
- fixed seed sample grids @ 1/2/4/8 steps
- q_phi vs q_D 曲线
- D_bar/K_bar/HF_bar 曲线
- continuation source 随训练变化曲线
- alpha_online 随训练变化曲线

### Stage B 研究判断
你必须明确写出：
- online mainline 是否比 cached baseline 更值得保留
- 收益主要来自哪里：
  - 更好的 bridge consistency
  - 更好的 low-sigma detail
  - 更合理的 warp
  - 或只是训练噪声变化
- 如果收益不显著，也要明确指出问题是：
  - quality 没提升
  - 训练太慢
  - warp 学偏
  - 还是 online continuation 其实没有实质改变 teacher-student mismatch

---

## Stage C — Full Run 主实验
只有当 Stage A 和 Stage B 给出正面信号时，才进入 full run。

### Stage C 目标
- 跑当前 online mainline 的主体完整训练
- 记录完整训练曲线
- 给出最终 few-step performance 与训练代价结论

### Stage C 必须输出
#### 最终主表
- FID @ 1/2/4/8/16
- best checkpoint 和 last checkpoint 各自结果
- train/eval latency
- peak VRAM
- total wall-clock

#### 最终诊断表
- continuation source 最终占比
- bridge gap 最终值
- online teacher rel error 最终值
- q_phi entropy 最终值
- q_phi / q_base 最大比值
- alpha_online 最终统计
- noisy / mid / clean 三段最终误差

#### 最终图
- 训练全过程 loss/defect 曲线
- continuation source 随 epoch 变化
- q_phi 与 q_D 随 epoch 变化
- alpha_online 随 epoch 变化
- 1/2/4/8/16 fixed-seed 样本网格

### Stage C 的最终结论必须回答
1. online mainline 能否作为正式主线继续保留？
2. 它最明显的收益体现在哪个 step 区间？
3. 它最明显的代价是什么？
4. 下一轮应该继续优化：
   - teacher continuation 几何
   - warp density
   - clean-end detail
   - 还是 inference schedule

---

## 3. 实验分析重点（必须写进报告）
你的实验报告不能只列数值，必须围绕以下分析展开。

### A. Online continuation 的真实性
即便 `continuation_sources.online=1.0`，也要解释：
- 它是不是始终提供有用 teacher signal
- 还是只是把 current trajectory anchor 重新包装了一遍

重点看：
- teacher_rel_error_mean
- bridge_u_teacher_error
- direct_bridge_gap
- 与 baseline 的差异

### B. Warp 是否真的“学到了”
必须分析：
- q_phi 是否只是向 clean end 偏
- 还是与 D_bar / K_bar 的高值区一致
- entropy 是否健康
- q_phi/q_base 是否过尖

### C. Warmup 是否真的修复了 old pathology
重点分析：
- warmup 期的 continuation source
- eta 变化
- bridge gap 变化
- bridge-side teacher error 是否从早期就可学习

### D. 质量与成本是否匹配
必须把：
- FID 改善
- 训练变慢多少
- 显存多多少
- 是否值得继续投入
放在一起分析

---

## 4. 你必须返回给我的材料
本轮实验结束后，你必须给我一份结构化报告，而不是只说“跑完了”。

### 必须输出的文档
请生成一份完整实验报告，内容至少包括：
1. 实验设置表
2. Stage A/B/C 各阶段结果
3. 主表（FID / latency / memory / wall-clock）
4. 关键曲线图
5. fixed-seed 样本网格
6. 风险与结论
7. 推荐下一步

### 必须返回的核心原始材料
至少包括：
- 最终 train history（或其整理版）
- 最终 eval summary
- 各 step count 的 FID 结果
- q_phi / q_D / D_bar / K_bar / HF_bar 数组
- continuation source 统计
- alpha_online 统计
- fixed-seed sample grids
- best checkpoint 与 last checkpoint 的对比结果

---

## 5. 你现在需要特别盯住的异常
如果出现下面任何一种现象，必须在报告里单独列为 warning：

1. `online_continuation_rate` 虽高，但 bridge 指标没有改善
2. `q_phi` 快速坍缩到 clean end
3. `alpha_online_max` 长期贴近上限，说明 clean-end 可能过强耦合
4. 1 step 提升但 4/8 step 不升反降
5. throughput 明显恶化但质量收益不明显
6. best checkpoint 和 last checkpoint 差异过大，说明训练不稳

---

## 6. 重要限制
- 这轮只做实验设计与执行，不做新方法开发
- 不要加入新的 loss
- 不要切换到另一个 teacher family
- 不要改掉当前 online-mainline 核心逻辑
- 如果你认为必须补充一个统计项，允许做最小日志补充，但要在报告里说明

---

## 7. 最终回答格式
在 chat 里不要只说“已完成”。  
请给出：

1. 你实际执行了哪些阶段
2. 每个阶段的结论
3. 是否建议继续 full run
4. 你最担心的两个风险是什么
5. 你推荐的下一步研究方向是什么