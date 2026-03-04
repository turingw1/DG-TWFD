# Prompt Phase 3: 损失函数与动态调度系统

你将基于 Phase 1-2 的工程，实现 DG-TWFD 的损失与动态调度系统。你必须在本阶段产出“完整可运行代码”，并在完成后立刻停止生成，等待我 Code Review 指令再进入下一阶段。

## 0. 总体要求
1. 实现统一损失：
   - \(L_{match}\): teacher map 回归
   - \(L_{def}\): semigroup defect
   - \(L_{warp}\): time-warp 学习项（用 teacher 轨迹有限差分，不用 JVP）
   - \(L_{boundary}\): 门控边界校正器损失
2. Defect-Adaptive Scheduler：
   - 实现 \(p_\eta(t)\propto \exp(\eta \widehat{d}(t))\)
   - \(\widehat{d}(t)\) 用在线统计估计（局部 defect 或 match 残差）
   - 必须提供采样接口，给 dataset 或 trainer 用
3. 4060 debug：必须提供 `tests/test_loss.py`，能完成一次 Forward+Backward，并验证无 JVP 瓶颈。

## 1. 新增文件结构（必须创建）
- `src/dg_twfd/losses/defect.py`
- `src/dg_twfd/losses/warp.py`
- `src/dg_twfd/losses/boundary.py`
- `src/dg_twfd/schedule/defect_adaptive.py`
- `src/dg_twfd/engine/metrics.py`（记录 defect 曲线与统计）
- `tests/test_loss.py`

## 2. Loss 详细定义与实现要求

### 2.1 Match Loss
在 batch 给出 `(x_t, x_s, t, s)`，其中 x_s 是 teacher target：
\[
L_{match} = \|M_\theta(t,s,x_t) - x_s\|_2^2
\]
实现时支持 l2 与 huber（config 控制）。

### 2.2 Semigroup Defect Loss
对同一 x_t，随机采样一个中间时间 u，满足 t > s > u。
实现：
1) `x_u_direct = M_theta(t,u,x_t)`
2) `x_s_mid = M_theta(t,s,x_t)`
3) `x_u_comp = M_theta(s,u,x_s_mid)`
4) \(L_{def}=\|x_u_direct - x_u_comp\|_2^2\)

要求：
- u 的采样必须可由 Defect-Adaptive Scheduler 提供
- 支持 per-pixel 平均
- 记录 defect 的 batch 均值到 metrics

### 2.3 Warp Loss（无 JVP）
实现一个 `WarpLoss`，用于更新 time-warp 参数 φ：
从同一条 teacher 轨迹中选三点 \(t_3>t_2>t_1\)，定义
\[
L_{warp} = \|\Phi_T(t_3\to t_2,x_{t_3}) - \Phi_T(t_2\to t_1,x_{t_2})\|_2^2
\]
实现方案：
- dataset 需要提供同一轨迹的多点（如 cached 模式下的轨迹 dict）
- 若 dataset 当前只输出 pair，则你要扩展 dataset 或写一个 “warp mini-batch sampler” 能从 teacher 轨迹缓存中抽 triplet
- 必须在注释中写明：warp loss 作用于时间测度学习，目的为降低 defect 下界与提高跨步数稳定

### 2.4 Boundary Loss
当 gate 启用时：
\[
L_{boundary} = \|B_\psi(x_{t_{max}}) - x_{t_{max}-\delta}^T\|_2^2
\]
实现要求：
- 支持 gate 权重 w（退火）
- 支持只在训练前若干 step 启用（由 config 控制）

## 3. Defect-Adaptive Scheduler 实现
实现 `DefectAdaptiveScheduler`：
- 内部维护一个离散时间桶（例如 64 或 128 bins）
- 每个 bin 维护 EMA 的 defect 统计 \(\widehat{d}(t)\)
- 采样分布：
  \[
  p_\eta(i) \propto \exp(\eta \cdot \widehat{d}_i)
  \]
- 提供方法：
  - `update(t, defect_value)`：将 batch 的 defect 按 t 落桶更新
  - `sample(batch_size)`：返回 t 样本
  - `set_eta(eta)`：支持随训练衰减 eta
- 工程要求：
  - 支持 torch 与 numpy
  - 支持可复现随机种子

## 4. tests/test_loss.py（必须可跑）
要求：
1. 加载 debug_4060 配置
2. 构造 DummyTeacherTrajectory + dataloader，取一个 batch
3. 构造 TimeWarpMonotone、FlowStudent、BoundaryCorrector
4. 计算一次总损失：
   \[
   L = L_{match} + \alpha L_{def} + \beta L_{warp} + \gamma L_{boundary}
   \]
   其中 warp/boundary 若数据不足可在测试里用简化路径，但必须保持接口一致并在注释说明
5. 做 backward：
   - 检查学生参数与 time-warp 参数存在非 None 梯度（至少一部分）
   - 断言无 NaN
6. 若 CUDA 可用：打印 peak memory
7. 明确说明：本实现不使用 JVP，仅有限差分与模型前向复合

## 5. 输出要求
- 逐文件给完整代码
- 每个 loss 与 scheduler 必须有公式注释与 config 入口
- 完成本阶段后必须停止生成并等待下一阶段指令
