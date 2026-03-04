# Prompt Phase 2: 核心架构组件与数学映射 

你将基于 Phase 1 的工程继续实现 DG-TWFD 的核心模型组件。你必须在本阶段产出“完整可运行代码”，并在完成后立刻停止生成，等待我 Code Review 指令再进入下一阶段。

## 0. 总体要求
1. 严格对齐 DG-TWFD 数学：
   - Time-warp：单调映射 \(u=g_\phi(t)\)，并实现可计算的近似逆 \(t\approx g_\phi^{-1}(u)\)
   - Flow student：\(M_\theta(t,s,x_t)\to x_s\)
   - 门控边界校正器：\(B_\psi(x_{t_{\max}})\to \hat{x}_{t_{\max}-\delta}\)，且带 gate，只在需要时启用
2. 4060 debug：必须提供 `tests/test_models.py`，用 fake tensor 输入，打印 shape 和 CUDA 峰值显存。
3. 代码风格：模块清晰、注释含公式、可扩展到 A100。

## 1. 新增文件结构（必须创建）
- `src/dg_twfd/models/timewarp.py`
- `src/dg_twfd/models/boundary.py`
- `src/dg_twfd/models/student.py`
- `src/dg_twfd/models/embeddings.py`（时间 embedding 与 (t,s) 条件编码）
- `tests/test_models.py`

并在 `src/dg_twfd/models/__init__.py` 导出主要类。

## 2. Time-Warp 模块 g_phi（强制实现单调与近似逆）
实现 `TimeWarpMonotone`：
- 输入：t in [0,1]，输出 u in [0,1]
- 单调性保障方案（任选一种并写清楚原因）：
  1) 累积 softplus：在固定网格上学习正值密度并累积归一化得到 CDF
  2) 单调样条（如 rational-quadratic spline 的单调版本），但请避免引入复杂外部依赖
建议优先用方案 1，便于工程落地。
- 必须实现方法：
  - `forward(t)->u`
  - `inverse(u)->t`：用二分搜索或插值近似，要求可在 torch 下运行（可不要求可微）
  - `grid_cache()`：可选，缓存离散映射表提高 inverse 速度
- 注释中写明：此 time-warp 的目标是“改变时间测度以降低复合误差与 semigroup defect”，并非强制状态空间几何直线。

## 3. 门控边界校正器 B_psi
实现 `BoundaryCorrector`：
- 输入：`x` (B,C,H,W) 和可选 `enabled: bool` 或 `gate_weight`  
- 输出：校正后的 `x_corr`
- 设计：
  - 网络结构尽量轻量，例如 2-4 层 Conv + GroupNorm + SiLU
  - 支持只在噪声端启用：当 `enabled=False` 时直接返回输入
  - gate 机制：支持一个标量 `w`，输出 `x + w * f(x)`，便于后续退火
- 注释需解释：用于解决第一跳方差爆炸，训练时门控启用，推理时可选启用，避免容量浪费。

## 4. Flow Student 网络骨干 M_theta
实现 `FlowStudent`：
- 接口：`forward(x_t, t, s) -> x_s_pred`
- 条件编码：
  - 使用 `TimeEmbedding`（sin/cos 或 MLP）将 t,s 编码
  - 输入网络的条件建议用 (t,s,delta=t-s) 三者
- 网络结构要求：
  - 工程上可用 U-Net mini 或 ResNet 卷积块
  - 必须支持 variable image_size 与 channels
  - 必须支持 AMP
- 输出：预测的 x_s，回归目标将由 Phase 3 的 loss 定义
- 可选：支持 predict residual（预测 x_s - x_t），但必须在注释中写清楚选择与数学对应

## 5. tests/test_models.py（必须可跑）
测试要求：
1. 加载 debug_4060 配置
2. 构造 TimeWarpMonotone，随机 t 生成 u，再 inverse 回 t_hat，打印 max error
3. 构造 BoundaryCorrector，输入 fake x，分别测试 enabled True/False，打印 shape
4. 构造 FlowStudent，输入 fake x_t, t, s，输出 x_s_pred，断言 shape 相同
5. 若 CUDA 可用：
   - 使用 torch.cuda.reset_peak_memory_stats
   - 前向一次后打印 peak memory
6. 完成后退出

## 6. 输出要求
- 逐文件给出完整代码
- 关键实现必须有 docstring 与公式注释
- 完成本阶段后必须停止生成并等待下一阶段指令