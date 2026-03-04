# Prompt Phase 1: 项目初始化与数据 Pipeline

你是资深 PyTorch 工程师。请为算法提案《Defect-Guided Time-Warp Flow Distillation (DG-TWFD)》创建一个可落地的工业级代码工程。你必须在本阶段产出“完整可运行代码”，并在完成后立刻停止生成，等待我 Code Review 指令再进入下一阶段。

## 0. 总体要求
1. 严格数学与架构保真：所有实现必须在注释里明确对应到 DG-TWFD 的数学对象与符号，尤其是后续阶段会用到的 \(
M_\theta(t,s,x_t),\ \Phi_T(t\to s,x_t),\ u=g_\phi(t)
\) 以及 semigroup defect 定义。
2. 硬件自适应：必须提供配置中心，内置 profile：`debug_4060` 与 `train_a100`，一键切换 batch size、num_workers、amp、gradient_accumulation、prefetch 等。
3. 结构化与文档化：本阶段必须先生成详尽 README.md，包含：
   - 项目概览与 DG-TWFD pipeline 描述
   - 目录树 ASCII
   - Mermaid 或 ASCII 的 pipeline 图（只用其中一种即可）
   - 每个模块、类、函数的职责说明（到文件级与关键类级）
4. 易于调试：本阶段先集成 logging（标准 logging 模块即可），后续阶段会接 wandb。日志要带时间戳、level、文件名行号。
5. 单元测试：必须输出 `tests/test_data.py`，使用 `debug_4060` profile 可在 8GB 显存环境运行。测试要做到：
   - 构造 Dummy teacher（不用下载模型）也能吐出 `(x_t, x_s, t, s)`
   - 显存安全：启用 AMP（如果可用），batch 极小，打印 torch.cuda.max_memory_allocated

## 1. 工程结构
用 `src/` 风格组织，并遵循可扩展训练代码工程习惯。必须创建以下文件与目录（可补充但不可缺）：
- `README.md`
- `pyproject.toml` 或 `requirements.txt`（二选一）
- `config/default.yaml`
- `config/profiles/debug_4060.yaml`
- `config/profiles/train_a100.yaml`
- `src/dg_twfd/__init__.py`
- `src/dg_twfd/config.py`：配置 dataclass + 从 yaml 合并 profile
- `src/dg_twfd/utils/logging.py`
- `src/dg_twfd/utils/seed.py`
- `src/dg_twfd/data/teacher.py`：teacher 轨迹生成器接口与 Dummy 实现
- `src/dg_twfd/data/dataset.py`：轨迹对数据集
- `src/dg_twfd/data/dataloader.py`：构造 dataloader 的函数
- `tests/test_data.py`

## 2. 配置中心规范（强制）
实现 `DGConfig`（dataclass）并提供 `load_config(profile: str, overrides: Optional[List[str]])`：
- YAML 合并优先级：default < profile < CLI overrides
- `debug_4060` profile：极小 batch、num_workers=0 或 2、开启 AMP、gradient_accumulation 较大、限制预取
- `train_a100` profile：较大 batch、num_workers>4、pin_memory=True、prefetch_factor 合理，AMP 仍默认开启
- 必须支持：数据维度（channels, image_size）、时间离散点数（用于 teacher 轨迹缓存的 m）、轨迹对采样策略（随机 t,s 还是从缓存中挑）

## 3. Teacher 轨迹生成器与数据集
本阶段不需要真实扩散 teacher。你要实现一个可插拔接口：
- `class TeacherTrajectory` 抽象类：
  - `sample_x0(batch, device) -> x0`
  - `forward_map(x_t, t, s) -> x_s` 代表 \(\Phi_T(t\to s, x_t)\)
  - `make_trajectory(x0, t_grid) -> Dict[t, x_t]`（可选缓存）
- `DummyTeacherTrajectory` 作为默认实现，要求：
  - 使用一个简单、可控的“非线性但可计算”的 ODE/flow 作为 teacher 映射，确保存在弯曲，从而后续 defect 与 warp 有意义
  - 例如：定义一个可解析的 time-dependent 速度场 \(v(x,t)=a(t)\cdot x + b(t)\cdot \tanh(x)\)，用小步 Euler 在 teacher 内部积分得到 forward_map（只需少量步，且可缓存）
  - 注意：这里只是工程占位，必须在注释里写明“Dummy teacher 仅用于单测与 debug，不代表最终论文 teacher”
- Dataset：
  - `TrajectoryPairDataset` 每次 __getitem__ 返回 `(x_t, x_s, t, s)`，其中 `t>s`
  - 支持两种模式：on-the-fly（在线 teacher 积分）与 cached（先生成固定 t_grid 的轨迹再随机抽点）
  - `debug_4060` 默认 cached=False 或极小缓存，避免占显存

## 4. DataLoader
实现 `build_dataloader(cfg, teacher, split)`：
- `split` 支持 train/val
- 支持 `persistent_workers`、`pin_memory`、`prefetch_factor` 根据 profile 自动设置
- 输出 batch 字典，键名固定：`x_t, x_s, t, s`

## 5. tests/test_data.py（必须可跑）
测试逻辑：
1. 加载 `debug_4060` 配置
2. 构造 DummyTeacherTrajectory
3. 构造 dataloader，取 2 个 batch
4. 断言：
   - `x_t.shape == x_s.shape == (B,C,H,W)`
   - `t,s` shape 为 `(B,)`，且 `torch.all(t > s)`
   - dtype 在 AMP 下合理（可在 CPU 上用 float32）
5. 若有 CUDA：打印 max_memory_allocated；否则打印 RAM 占用的粗略信息即可
6. 测试结束后退出

## 6. 输出要求
- 你必须一次性给出所有文件的完整代码内容（逐文件分块输出）。
- 所有关键类/函数必须有 docstring，解释其对应的数学定义。
- 完成本阶段后立刻停止生成，并提示“等待下一阶段指令”。