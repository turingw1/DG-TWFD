# Teacher 与数据集对接指导

本文档只覆盖 `Guide/teacher_dataset.md` 相关的 teacher / trajectory dataset 对接工作，并明确区分：

- 我已经直接在仓库里完成的部分
- 你必须手动完成的部分
- 推荐的最短落地路径

## 1. 我已经完成的部分

### 1.1 Teacher 适配层

已实现：

- `TeacherTrajectory` 统一抽象接口
- `DummyTeacherTrajectory` 保留原有 debug 路径
- `DiffusersDDPMTeacher` 适配器骨架
- `build_teacher(cfg)` 工厂函数

相关文件：

- [src/dg_twfd/data/teacher.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/data/teacher.py)

当前可直接用的 `teacher_type`：

- `dummy`
- `diffusers_ddpm`

### 1.2 数据集与离线轨迹 shard

已实现：

- `TrajectoryPairDataset`：原有在线 teacher 数据路径
- `TrajectoryShardDataset`：读取离线 teacher 轨迹 shard
- `build_dataset(cfg, teacher, split)`：按配置选择数据集
- dataloader 支持附加字段，例如 `y`

相关文件：

- [src/dg_twfd/data/dataset.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/data/dataset.py)
- [src/dg_twfd/data/dataloader.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/data/dataloader.py)

### 1.3 轨迹采集脚本

已实现：

- `scripts/collect_teacher.py`

用途：

- 用 teacher 在给定 `t_grid` 上 rollout
- 按 shard 保存成 `.pt`
- 后续训练时由 `TrajectoryShardDataset` 直接读取

### 1.4 训练器兼容性

已修改：

- `Trainer` 不再要求数据集必须是 `TrajectoryPairDataset`
- 只要数据集实现 `sample_triplet_batch()`，就能进入 Phase 3/4 的 loss 与 trainer

### 1.5 依赖入口

已添加可选依赖：

```bash
python -m pip install -e '.[teacher]'
```

## 2. 你必须手动完成的部分

这些事情我不能直接替你完成，因为需要外部权重、数据许可、网络下载或你本地路径决策。

### 2.1 安装真实 teacher 依赖

在 `consistency` 环境里执行：

```bash
cd ~/workspace/Zhengwei/DG-TWFD
conda activate consistency
python -m pip install -e '.[teacher]'
```

### 2.2 下载或准备 teacher 权重

推荐先从最短路径开始：

- `google/ddpm-cifar10-32`

你可以选两种方式：

1. 允许 `diffusers` 自动下载到本地缓存
2. 先手动下载到本地目录，再把路径写给 `teacher.pretrained_model_name_or_path`

### 2.3 确定 shard 输出目录

你需要自己决定一个真实路径，例如：

```bash
~/workspace/Zhengwei/DG-TWFD/data/teacher_shards/ddpm_cifar10_32
```

### 2.4 如果继续扩展到 ImageNet / guided-diffusion / DiT

这些都需要你自己准备：

- 数据许可与下载
- checkpoint 路径
- 可能新增的 adapter 代码

当前仓库已经把接口准备好了，但还没有替你实现 `guided-diffusion` / `EDM` / `DiT` 的 adapter。

## 3. 推荐的最短落地路径

建议你先走这一条：

1. 用 `diffusers_ddpm` 接 `google/ddpm-cifar10-32`
2. 先离线采集 `train/val` trajectory shards
3. 再切换训练配置，改用 `trajectory_shards`
4. 跑通训练与采样闭环

这是当前最稳、改动最少、最容易验证的路线。

首次真实采集时，建议不要直接用 1000 inference steps。对于 `google/ddpm-cifar10-32`，先用 `teacher.num_inference_steps=100` 或 `250` 做第一轮 shard 采集，确认链路正确后再逐步加大。

## 4. 第一步：采集 teacher 轨迹

先运行训练轨迹采集：

```bash
cd ~/workspace/Zhengwei/DG-TWFD
conda activate consistency
python scripts/collect_teacher.py \
  --mode debug_4060 \
  --split train \
  --num-samples 256 \
  --shard-size 16 \
  --output-dir ./data/teacher_shards/ddpm_cifar10_32 \
  --override teacher.teacher_type='diffusers_ddpm' \
  --override teacher.pretrained_model_name_or_path='google/ddpm-cifar10-32' \
  --override teacher.local_files_only=false \
  --override teacher.num_inference_steps=100 \
  --override data.time_grid_size=8
```

再采集验证集：

```bash
cd ~/workspace/Zhengwei/DG-TWFD
conda activate consistency
python scripts/collect_teacher.py \
  --mode debug_4060 \
  --split val \
  --num-samples 64 \
  --shard-size 16 \
  --output-dir ./data/teacher_shards/ddpm_cifar10_32 \
  --override teacher.teacher_type='diffusers_ddpm' \
  --override teacher.pretrained_model_name_or_path='google/ddpm-cifar10-32' \
  --override teacher.local_files_only=false \
  --override teacher.num_inference_steps=100 \
  --override data.time_grid_size=8
```

执行完以后，你需要把下面这些结果发给我：

1. `collect_teacher.py` 的完整输出
2. 生成的 shard 目录结构，例如 `find ./data/teacher_shards/ddpm_cifar10_32 -maxdepth 2 -type f | sort`
3. 如果报错，完整 traceback

补充说明：

- `no file named diffusion_pytorch_model.safetensors found ... Defaulting to unsafe serialization` 这条信息通常不是致命错误，表示模型回退到 `.bin` 权重加载。
- 如果采集速度仍然太慢，优先继续降低 `--shard-size`，其次再降低 `teacher.num_inference_steps`。

## 5. 第二步：切换训练到 shard 数据集

采集完成后，运行训练：

```bash
cd ~/workspace/Zhengwei/DG-TWFD
conda activate consistency
python train.py --mode debug_4060 \
  --override data.dataset_type='trajectory_shards' \
  --override data.trajectory_shard_dir='./data/teacher_shards/ddpm_cifar10_32' \
  --override teacher.teacher_type='diffusers_ddpm' \
  --override teacher.pretrained_model_name_or_path='google/ddpm-cifar10-32' \
  --override boundary.enable_until_step=0
```

这里把 `boundary.enable_until_step=0` 关掉，是为了先减少“离线 shard + 在线 boundary target teacher”之间的耦合。如果你后面确认 online teacher rollout 速度可接受，再把 boundary 打开。

你需要把下面这些结果发给我：

1. 每个 epoch 的 `train_loss` / `val_loss`
2. 如果是 GPU，`peak_mem`
3. 如果失败，完整 traceback

## 6. 第三步：验证采样

```bash
cd ~/workspace/Zhengwei/DG-TWFD
conda activate consistency
python sample.py \
  --mode debug_4060 \
  --checkpoint checkpoints/best.pt \
  --steps 4
```

如果这是 shard 训练得到的 checkpoint，也把输出发给我，我可以继续帮你判断当前 teacher 对接是否真的闭环。

## 7. shard 文件格式说明

当前 `TrajectoryShardDataset` 期待每个 `.pt` shard 是一个 `list[dict]`，每个 sample 至少包含：

```python
{
    "t_grid": Tensor[M],
    "x_grid": Tensor[M, C, H, W],
}
```

推荐附加字段：

```python
{
    "x0": Tensor[C, H, W],
    "seed": int,
    "y": Tensor[] or int,  # class-conditional 时使用
}
```

时间语义要求：

- `t` 越大表示越 noisy
- 训练时需要满足 `t > s > u`
- 数据集内部会自动重排 `t_grid`

## 8. 你后续最可能继续改的地方

### 8.1 如果你要接 guided-diffusion / EDM / DiT

建议新增 adapter 类，保持以下接口不变：

- `sample_x0(batch_size, device)`
- `forward_map(x_t, t, s)`
- `make_trajectory(x0, t_grid)`
- `sample_trajectory(batch_size, t_grid, device, labels=None)`

### 8.2 如果你要接 class-conditional teacher

优先修改：

- [src/dg_twfd/data/teacher.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/data/teacher.py)
- [src/dg_twfd/data/dataloader.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/data/dataloader.py)
- [src/dg_twfd/models/student.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/models/student.py)

### 8.3 如果你要完全切到离线 teacher 训练

优先考虑：

- 把 `boundary` 先关掉
- 先确保 `match/defect/warp` 三条链路稳定
- 再决定 boundary target 是否仍然依赖在线 teacher

## 9. 当前限制

- `DiffusersDDPMTeacher` 已经可作为接入入口，但没有替你下载外部权重
- 当前未直接实现 `guided-diffusion` / `EDM` / `DiT` adapter
- 当前未实现 ImageNet 原始图像读取与分类标签预处理流水线
- 当前最成熟的真实 teacher 路径是：`diffusers_ddpm + trajectory_shards`
