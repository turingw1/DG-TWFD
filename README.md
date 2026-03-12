# DG-TWFD 项目说明与开发手册

本仓库是 **Defect-Guided Time-Warp Flow Distillation (DG-TWFD)** 的工程化参考实现，当前已经覆盖：

- 数据与 teacher 轨迹生成
- student、time-warp、boundary 三个核心模型组件
- match / defect / warp / boundary 四类损失
- defect-adaptive scheduler
- 训练、验证、checkpoint、日志
- 推理采样与 profiling

仓库的目标不是只“跑通一个 demo”，而是提供一个便于你继续替换 teacher、数据集、优化器、损失、训练策略和推理 schedule 的可扩展骨架。

## 1. 项目目标与核心思想

DG-TWFD 的核心对象在代码里对应如下：

- `Phi_T(t->s, x_t)`：teacher 的时间映射，给定 `t > s`，从 `x_t` 推到 `x_s`
- `M_theta(t, s, x_t)`：student 的跨时间映射
- `u = g_phi(t)`：单调 time-warp，把原时间 `t` 重参数化到 `u`
- `B_psi(x_tmax)`：边界校正器，解决高噪声端第一跳不稳定
- `p_eta(t)`：基于 defect 统计的时间采样分布

当前实现里的 `DummyTeacherTrajectory` 只是占位 teacher，用于把整个 pipeline 跑通并验证工程设计。后续接入真实 diffusion / ODE / consistency teacher 时，推荐只替换 teacher 接口层，而尽量复用下游 dataset、loss、trainer 和 sampler。

## 2. 仓库结构

```text
.
├── config/
│   ├── default.yaml
│   └── profiles/
│       ├── debug_4060.yaml
│       └── train_a100.yaml
├── scripts/
│   └── profile_infer.py
├── src/dg_twfd/
│   ├── config.py
│   ├── data/
│   │   ├── dataloader.py
│   │   ├── dataset.py
│   │   └── teacher.py
│   ├── engine/
│   │   ├── amp.py
│   │   ├── checkpoint.py
│   │   ├── metrics.py
│   │   └── trainer.py
│   ├── infer/
│   │   ├── sampler.py
│   │   └── schedules.py
│   ├── losses/
│   │   ├── boundary.py
│   │   ├── defect.py
│   │   └── warp.py
│   ├── models/
│   │   ├── boundary.py
│   │   ├── embeddings.py
│   │   ├── student.py
│   │   └── timewarp.py
│   ├── schedule/
│   │   └── defect_adaptive.py
│   └── utils/
│       ├── logging.py
│       └── seed.py
├── tests/
│   ├── test_data.py
│   ├── test_loss.py
│   └── test_models.py
├── train.py
└── sample.py
```

## 3. 环境与快速开始

统一使用 `consistency` conda 环境：

```bash
conda activate consistency
python -m pip install -e .
```

本地 4060 调试训练：

```bash
python train.py --mode debug_4060
```

使用训练好的 checkpoint 采样：

```bash
python sample.py --mode debug_4060 --checkpoint checkpoints/best.pt --steps 4 --batch-size 2
```

推理 profiling：

```bash
python scripts/profile_infer.py
```

运行测试：

```bash
pytest tests/test_data.py -q -s
pytest tests/test_models.py -q -s
pytest tests/test_loss.py -q -s
```

## 4. 项目整体 Pipeline

完整训练流程可以概括为：

```text
1. 配置载入
   load_config(profile, overrides)

2. teacher 与数据构建
   teacher.sample_x0(...)
   teacher.forward_map(...)
   dataset -> (x_t, x_s, t, s)
   dataset.sample_triplet_batch() -> (x_t3, x_t2, x_t1, t3, t2, t1)

3. 模型前向
   student(x_t, t, s) -> x_s_pred
   timewarp(t) -> u
   boundary(x) -> corrected_x

4. 损失构建
   match:    M_theta(t,s,x_t) 对齐 x_s
   defect:   direct path vs composed path
   warp:     teacher 有限差分 + warped step balance
   boundary: B_psi(x_tmax) 对齐 teacher 的近邻目标

5. 调度更新
   scheduler.update(t, defect)
   scheduler.sample(batch_size)

6. 训练器更新
   optimizer(student+boundary)
   optimizer(timewarp)
   checkpoint / log / validate

7. 推理
   等间隔 u-grid
   t_i = g_phi^{-1}(u_i)
   多步复合 student
   可选 boundary correction
```

## 5. 按文件与函数对应的完整 Pipeline 解析

这一节是代码审查时最重要的索引。你可以顺着这一节逐文件核对代码和论文想法是否一致。

### 5.1 配置入口

文件：[src/dg_twfd/config.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/config.py)

关键对象与函数：

- `DGConfig`：总配置树
- `load_config(profile, overrides)`：配置总入口
- `_merge_dicts()`：实现 `default < profile < CLI override`
- `_apply_overrides()`：支持 `train.learning_rate=1e-4` 这类命令行覆盖

作用：

- 所有模块都不应该各自硬编码参数，应从 `DGConfig` 取值
- 如果你要加新模块，优先先加配置，再实现模块

### 5.2 Teacher 轨迹层

文件：[src/dg_twfd/data/teacher.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/data/teacher.py)

关键类与函数：

- `TeacherTrajectory`
  - `sample_x0(batch_size, device)`
  - `forward_map(x_t, t, s)`
  - `make_trajectory(x0, t_grid)`
- `DummyTeacherTrajectory`
  - `_velocity()`
  - `forward_map()`
  - `make_trajectory()`

对应论文对象：

- `forward_map()` 对应 `Phi_T(t->s, x_t)`
- `make_trajectory()` 用于构造 teacher 轨迹缓存，后续用于 warp loss 和 defect 分析

如何审查：

- 看 `forward_map()` 是否真的实现了从较大时间到较小时间的映射
- 看 `make_trajectory()` 是否保持了时间网格上的一致性
- 如果你替换成真实 teacher，这个文件是第一入口

### 5.3 数据集与 dataloader

文件：

- [src/dg_twfd/data/dataset.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/data/dataset.py)
- [src/dg_twfd/data/dataloader.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/data/dataloader.py)

关键类与函数：

- `TrajectoryPairDataset`
  - `__getitem__()`：返回 `(x_t, x_s, t, s)`
  - `sample_triplet_batch()`：返回 warp loss 所需三点
  - `_build_cache()`：建立 cached trajectory
  - `_on_the_fly_item()` / `_cached_item()`：两种采样模式
- `build_dataloader(cfg, teacher, split)`

对应论文对象：

- `x_s = Phi_T(t->s, x_t)` 是监督目标
- triplet `(t3, t2, t1)` 用于 warp 相关学习

如何审查：

- 看 `__getitem__()` 是否始终保证 `t > s`
- 看 `sample_triplet_batch()` 是否来自同一条轨迹
- 看 cached 与 on-the-fly 是否语义一致

### 5.4 模型层

文件：

- [src/dg_twfd/models/embeddings.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/models/embeddings.py)
- [src/dg_twfd/models/student.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/models/student.py)
- [src/dg_twfd/models/timewarp.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/models/timewarp.py)
- [src/dg_twfd/models/boundary.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/models/boundary.py)

关键对象与函数：

- `TimeEmbedding.forward(time_values)`：标量时间编码
- `PairTimeConditioner.forward(t, s)`：编码 `(t, s, delta=t-s)`
- `FlowStudent.forward(x_t, t, s)`：实现 `M_theta(t, s, x_t)`
- `TimeWarpMonotone.forward(t)`：实现 `u = g_phi(t)`
- `TimeWarpMonotone.inverse(u)`：实现 `t = g_phi^{-1}(u)`
- `BoundaryCorrector.forward(x, enabled, gate_weight)`：实现 `x + w * f(x)`

对应论文对象：

- `FlowStudent` 对应主 student 映射
- `TimeWarpMonotone` 对应单调 time-warp
- `BoundaryCorrector` 对应高噪声边界修正器

如何审查：

- 看 `TimeWarpMonotone` 是否严格单调
- 看 `inverse()` 是否足够稳定，能给推理构造 schedule
- 看 `FlowStudent` 是否真的接收了 `(t,s,delta)`
- 看 `BoundaryCorrector` 是否支持显式开关和 gate

### 5.5 损失层

文件：

- [src/dg_twfd/losses/defect.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/losses/defect.py)
- [src/dg_twfd/losses/warp.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/losses/warp.py)
- [src/dg_twfd/losses/boundary.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/losses/boundary.py)

关键对象与函数：

- `MatchLoss.forward(prediction, target)`
- `SemigroupDefectLoss.forward(student, x_t, t, s, scheduler)`
- `WarpLoss.forward(timewarp, triplet_batch)`
- `BoundaryLoss.forward(boundary_model, x_boundary, target, gate_weight, enabled)`

对应论文对象：

- `L_match`
- `L_def`
- `L_warp`
- `L_boundary`

如何审查：

- `SemigroupDefectLoss` 是否构造了 direct path 和 composed path
- `WarpLoss` 是否显式依赖同一 teacher 轨迹上的三点
- `WarpLoss` 当前是工程近似实现，不使用 JVP；如果你追求更严格论文形式，可以优先改这个文件

### 5.6 调度与统计

文件：

- [src/dg_twfd/schedule/defect_adaptive.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/schedule/defect_adaptive.py)
- [src/dg_twfd/engine/metrics.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/engine/metrics.py)

关键对象与函数：

- `DefectAdaptiveScheduler.update(t, defect_value)`
- `DefectAdaptiveScheduler.sample(batch_size)`
- `DefectAdaptiveScheduler.set_eta(eta)`
- `MetricTracker.update(**metrics)`

对应论文对象：

- `p_eta(t) ∝ exp(eta * d_hat(t))`

如何审查：

- 看 defect 的 EMA 是否按时间桶更新
- 看采样分布是否真的依赖 defect_ema

### 5.7 训练器

文件：

- [src/dg_twfd/engine/amp.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/engine/amp.py)
- [src/dg_twfd/engine/checkpoint.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/engine/checkpoint.py)
- [src/dg_twfd/engine/trainer.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/engine/trainer.py)
- [train.py](~/workspace/Zhengwei/DG-TWFD/train.py)

关键对象与函数：

- `build_grad_scaler()`
- `save_checkpoint()` / `load_checkpoint()`
- `Trainer._compute_losses()`
- `Trainer.train_epoch()`
- `Trainer.validate()`
- `Trainer.fit()`
- `train.py: main()`

训练期实际调用链：

```text
train.py
  -> load_config()
  -> build teacher / dataloader / models / losses / scheduler
  -> Trainer(...)
  -> Trainer.fit()
     -> train_epoch()
        -> _compute_losses()
        -> backward()
        -> optimizer step
     -> validate()
     -> maybe_save_checkpoint()
```

如何审查：

- `Trainer._compute_losses()` 是整个训练数学逻辑最核心的位置
- `train_epoch()` 里有 student/boundary 与 timewarp 的交替优化逻辑
- 如果你想换优化器，最直接的入口在 `Trainer.__init__()`

### 5.8 推理与采样

文件：

- [src/dg_twfd/infer/schedules.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/infer/schedules.py)
- [src/dg_twfd/infer/sampler.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/infer/sampler.py)
- [sample.py](~/workspace/Zhengwei/DG-TWFD/sample.py)
- [scripts/profile_infer.py](~/workspace/Zhengwei/DG-TWFD/scripts/profile_infer.py)

关键对象与函数：

- `build_u_schedule(steps)`
- `build_t_schedule_from_u(timewarp, u_schedule)`
- `sample_dg_twfd(...)`
- `profile_sampling(...)`

推理期实际调用链：

```text
sample.py
  -> load checkpoint
  -> build models
  -> sample_dg_twfd()
     -> build_u_schedule()
     -> timewarp.inverse()
     -> boundary(optional)
     -> repeated student(x, t_i, t_{i+1})
  -> save artifacts
```

如何审查：

- 看 schedule 是否先在 `u` 空间等间隔，再 inverse 到 `t`
- 看 boundary 校正是否只在采样起点启用
- 看 NFE 和 latency 是否由 `profile_sampling()` 正确统计

## 6. 配置系统与参数解释

配置文件：

- [config/default.yaml](~/workspace/Zhengwei/DG-TWFD/config/default.yaml)
- [config/profiles/debug_4060.yaml](~/workspace/Zhengwei/DG-TWFD/config/profiles/debug_4060.yaml)
- [config/profiles/train_a100.yaml](~/workspace/Zhengwei/DG-TWFD/config/profiles/train_a100.yaml)

优先级：

```text
default.yaml < profiles/*.yaml < --override key=value
```

### 6.1 `experiment`

- `name`：实验名，影响日志和 wandb run 名称
- `seed`：随机种子

### 6.2 `data`

- `channels`：输入通道数，例如 RGB 为 3
- `image_size`：图像边长
- `dataset_size`：训练集样本数
- `val_dataset_size`：验证集样本数
- `batch_size`：每个 batch 的样本数
- `num_workers`：DataLoader worker 数
- `pin_memory`：是否固定页内存
- `persistent_workers`：是否复用 worker
- `prefetch_factor`：预取倍数
- `drop_last`：训练时是否丢弃最后不足 batch
- `trajectory_cache_mode`：是否启用 teacher 轨迹缓存
- `num_cached_trajectories`：缓存轨迹条数
- `time_grid_size`：teacher 轨迹离散时间点数
- `sample_strategy`：当前保留字段，后续可扩展不同采样策略
- `teacher_integration_steps`：Dummy teacher 的 Euler 积分步数

### 6.3 `runtime`

- `device`：`auto` / `cpu` / `cuda`
- `amp`：是否启用 AMP
- `gradient_accumulation`：梯度累积步数

### 6.4 `teacher`

- `velocity_scale`：Dummy teacher 线性速度项强度
- `nonlinearity_scale`：Dummy teacher 非线性项强度
- `x0_std`：初始采样标准差

### 6.5 `model`

- `time_embed_dim`：时间 embedding 维度
- `cond_dim`：条件向量维度
- `hidden_channels`：student 隐层通道数
- `boundary_hidden_channels`：boundary 隐层通道数
- `boundary_num_blocks`：boundary block 数
- `student_num_blocks`：student 残差块数
- `timewarp_num_bins`：time-warp 的离散 bin 数
- `timewarp_init_bias`：time-warp 初始化偏置
- `predict_residual`：student 输出 residual 还是直接输出目标

### 6.6 `loss`

- `match_loss_type`：`l2` 或 `huber`
- `huber_delta`：Huber loss 的 delta
- `defect_weight`：defect loss 权重
- `warp_weight`：warp loss 权重
- `boundary_weight`：boundary loss 权重
- `per_pixel_mean`：是否按像素平均

### 6.7 `schedule`

- `num_bins`：defect 统计桶数
- `ema_decay`：EMA 衰减系数
- `eta`：采样分布锐度
- `eps`：数值稳定项
- `seed`：scheduler 随机种子

### 6.8 `boundary`

- `gate_weight`：边界校正残差缩放系数
- `enable_until_step`：训练前多少 step 启用 boundary

### 6.9 `train`

- `epochs`：训练轮数
- `learning_rate`：学习率
- `weight_decay`：权重衰减
- `grad_clip_norm`：梯度裁剪阈值
- `log_every`：日志打印间隔
- `save_every`：checkpoint 保存间隔
- `checkpoint_dir`：checkpoint 目录
- `resume_path`：恢复训练的 checkpoint 路径
- `warp_update_every`：timewarp 优化器更新频率
- `max_train_steps`：用于 smoke test 的最大 step 限制

## 7. 常用命令与 override 示例

### 7.1 4060 本地调试

```bash
python train.py --mode debug_4060
```

### 7.2 更小显存模式

```bash
python train.py --mode debug_4060 \
  --override data.batch_size=1 \
  --override runtime.gradient_accumulation=16
```

### 7.3 切换更大的时间网格

```bash
python train.py --mode train_a100 \
  --override data.time_grid_size=64 \
  --override data.num_cached_trajectories=512
```

### 7.4 切换损失形式

```bash
python train.py --mode debug_4060 \
  --override loss.match_loss_type='huber' \
  --override loss.huber_delta=0.05
```

### 7.5 恢复训练

```bash
python train.py --mode debug_4060 \
  --override train.resume_path='checkpoints/best.pt'
```

### 7.6 改变采样步数

```bash
python sample.py --mode debug_4060 --checkpoint checkpoints/best.pt --steps 8
```

## 8. 如何替换不同模块

### 8.1 替换数据集

当前项目本质上是 teacher-supervised pipeline，所以“数据集”不只是传统图像数据，还包括 teacher 如何生成轨迹监督。

如果你想换数据来源，优先考虑两种方式：

1. 保持 `TeacherTrajectory` 不变，只改 `sample_x0()`
2. 同时改 `sample_x0()` 和 `make_trajectory()`，让 teacher 直接从真实数据或 latent 出发

通常修改入口：

- [src/dg_twfd/data/teacher.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/data/teacher.py)
- [src/dg_twfd/data/dataset.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/data/dataset.py)

如果你要接真实图像数据集，可以增加一个 dataset reader，再在 teacher 内部决定如何把图像映射为初值 `x0`。

### 8.2 替换 teacher 模型

最推荐的改法是保留 `TeacherTrajectory` 抽象接口，仅实现一个新的子类，例如：

```python
class RealDiffusionTeacher(TeacherTrajectory):
    def sample_x0(self, batch_size, device):
        ...

    def forward_map(self, x_t, t, s):
        ...

    def make_trajectory(self, x0, t_grid):
        ...
```

替换位置：

- [src/dg_twfd/data/teacher.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/data/teacher.py)
- [train.py](~/workspace/Zhengwei/DG-TWFD/train.py)

你只需要把 `DummyTeacherTrajectory(cfg)` 换成新 teacher，即可沿用现有训练与推理框架。

### 8.3 替换 student 网络

修改文件：

- [src/dg_twfd/models/student.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/models/student.py)

建议保持接口不变：

```python
forward(x_t, t, s) -> x_s_pred
```

这样可以避免改 loss、trainer 和 sampler。

### 8.4 替换 time-warp 实现

修改文件：

- [src/dg_twfd/models/timewarp.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/models/timewarp.py)

最重要的是保留三个方法：

- `forward(t)`
- `inverse(u)`
- `grid_cache()`

如果你要换成样条或更复杂的单调变换，优先保证 `inverse()` 在推理期稳定。

### 8.5 替换 boundary 模块

修改文件：

- [src/dg_twfd/models/boundary.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/models/boundary.py)

建议保持：

```python
forward(x, enabled=True, gate_weight=1.0)
```

### 8.6 替换损失

修改文件：

- [src/dg_twfd/losses/defect.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/losses/defect.py)
- [src/dg_twfd/losses/warp.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/losses/warp.py)
- [src/dg_twfd/losses/boundary.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/losses/boundary.py)

其中最可能被论文迭代影响的是：

- `WarpLoss`
- `SemigroupDefectLoss`
- `Trainer._compute_losses()`

### 8.7 替换优化器

当前优化器定义在：

- [src/dg_twfd/engine/trainer.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/engine/trainer.py)

具体在 `Trainer.__init__()` 里：

- `self.student_optimizer`
- `self.warp_optimizer`

你可以直接换成：

- `torch.optim.Adam`
- `torch.optim.AdamW`
- `torch.optim.SGD`
- 自定义 optimizer / scheduler 组合

如果你还要接 LR scheduler，建议也放在 `Trainer.__init__()` 和 `Trainer.fit()` 中统一管理。

## 9. 训练与推理的实际改造建议

### 9.1 如果你想让代码更贴论文

优先改：

1. `WarpLoss`，让其更严格贴近你想要的理论形式
2. `TeacherTrajectory`，接入真实 teacher
3. `DefectAdaptiveScheduler`，让时间采样更强依赖真实 defect 估计

### 9.2 如果你想先把实验做稳

优先改：

1. `TrajectoryPairDataset` 的 cached 逻辑
2. `Trainer._compute_losses()` 的 loss 组织方式
3. checkpoint 和日志的实验记录粒度

### 9.3 如果你想快速试超参数

最常改的项：

- `data.batch_size`
- `runtime.gradient_accumulation`
- `train.learning_rate`
- `loss.defect_weight`
- `loss.warp_weight`
- `schedule.eta`
- `model.hidden_channels`
- `model.timewarp_num_bins`

## 10. 测试与验收建议

测试文件：

- [tests/test_data.py](~/workspace/Zhengwei/DG-TWFD/tests/test_data.py)
- [tests/test_models.py](~/workspace/Zhengwei/DG-TWFD/tests/test_models.py)
- [tests/test_loss.py](~/workspace/Zhengwei/DG-TWFD/tests/test_loss.py)

建议的审查顺序：

1. 先看 `test_data.py`，确认 teacher + dataset + dataloader 的监督张量是否合理
2. 再看 `test_models.py`，确认模型 shape、time-warp 反演和 boundary gate 是否正确
3. 最后看 `test_loss.py`，确认 total loss、backward 和 timewarp 梯度链路是否真的打通

## 11. 当前实现的边界与已知注意事项

- 当前 `DummyTeacherTrajectory` 不是最终论文 teacher，只是工程占位
- `WarpLoss` 是无 JVP 的工程近似实现，适合调试和快速实验，但未必是论文最终形式
- 推理 profiling 目前只输出 latency / NFE / 显存，真实 FID / Inception 评估接口仍需后续补充
- checkpoint 在 PyTorch 2.6 下显式使用 `weights_only=False` 读取，因为当前 checkpoint 不只是权重，还包含 optimizer、scheduler 和 scaler 状态

## 12. 建议的阅读路径

如果你第一次接手这个项目，建议按下面顺序阅读代码：

1. [config/default.yaml](~/workspace/Zhengwei/DG-TWFD/config/default.yaml)
2. [src/dg_twfd/config.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/config.py)
3. [src/dg_twfd/data/teacher.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/data/teacher.py)
4. [src/dg_twfd/data/dataset.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/data/dataset.py)
5. [src/dg_twfd/models/student.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/models/student.py)
6. [src/dg_twfd/models/timewarp.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/models/timewarp.py)
7. [src/dg_twfd/losses/defect.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/losses/defect.py)
8. [src/dg_twfd/losses/warp.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/losses/warp.py)
9. [src/dg_twfd/engine/trainer.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/engine/trainer.py)
10. [src/dg_twfd/infer/sampler.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/infer/sampler.py)

这样读，你会最容易建立“论文对象 -> 代码模块 -> 实验命令”的对应关系。

<!-- 019ca8ab-7294-7e71-9d1d-35a6aea33158 -->
<!-- i2p 019cc679-8826-7d60-9204-b58649dec87d -->