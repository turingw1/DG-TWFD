# A100: DDPM-CIFAR10-32 完整实验流水线（命令版）

本文档给你一条可直接执行的命令流水线，覆盖：

- teacher 轨迹数据生成
- 训练（含监督项配置）
- 训练结果汇总
- 采样与推理 profiling
- 结果产物管理

并且所有命令都尽量参数化，方便后续替换 teacher、数据集、损失和优化器。

---

## 0) 一次性参数区（先改这里）

```bash
export PROJ=/home/gzwlinux/vscode/gitProject/DG-TWFD
export ENV_NAME=consistency

# 本次实验名（建议每次改）
export EXP_NAME=ddpm_cifar10_a100_v1

# teacher 与数据
export TEACHER_ID=google/ddpm-cifar10-32
export SHARD_ROOT=$PROJ/data/teacher_shards/$EXP_NAME
export TRAIN_SAMPLES=50000
export VAL_SAMPLES=5000
export SHARD_SIZE=32
export TIME_GRID_SIZE=16
export TEACHER_STEPS=100

# 训练
export CKPT_DIR=$PROJ/checkpoints/$EXP_NAME
export EPOCHS=20
export BATCH_SIZE=64
export NUM_WORKERS=8

# 采样
export SAMPLE_STEPS=4
export SAMPLE_BATCH=16
```

---

## 1) 代码同步与环境准备

```bash
cd $PROJ
git pull --ff-only
conda activate $ENV_NAME
python -m pip install -e '.[dev,teacher]'
```

可选：先看 GPU 状态

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 2) 预检查（建议先跑）

```bash
cd $PROJ
conda activate $ENV_NAME
pytest tests/test_data.py tests/test_models.py tests/test_loss.py -q
```

---

## 3) 生成 teacher trajectory shards

### 3.1 先跑小规模 smoke（强烈建议）

```bash
cd $PROJ
conda activate $ENV_NAME
python scripts/collect_teacher.py \
  --mode train_a100 \
  --split train \
  --num-samples 64 \
  --shard-size 8 \
  --output-dir $SHARD_ROOT \
  --override teacher.teacher_type='diffusers_ddpm' \
  --override teacher.pretrained_model_name_or_path="$TEACHER_ID" \
  --override teacher.local_files_only=false \
  --override teacher.num_inference_steps=$TEACHER_STEPS \
  --override data.time_grid_size=$TIME_GRID_SIZE
```

检查输出：

```bash
find $SHARD_ROOT -maxdepth 2 -type f | sort
```

### 3.2 正式生成 train shards

```bash
cd $PROJ
conda activate $ENV_NAME
python scripts/collect_teacher.py \
  --mode train_a100 \
  --split train \
  --num-samples $TRAIN_SAMPLES \
  --shard-size $SHARD_SIZE \
  --output-dir $SHARD_ROOT \
  --override teacher.teacher_type='diffusers_ddpm' \
  --override teacher.pretrained_model_name_or_path="$TEACHER_ID" \
  --override teacher.local_files_only=false \
  --override teacher.num_inference_steps=$TEACHER_STEPS \
  --override data.time_grid_size=$TIME_GRID_SIZE
```

### 3.3 正式生成 val shards

```bash
cd $PROJ
conda activate $ENV_NAME
python scripts/collect_teacher.py \
  --mode train_a100 \
  --split val \
  --num-samples $VAL_SAMPLES \
  --shard-size $SHARD_SIZE \
  --output-dir $SHARD_ROOT \
  --override teacher.teacher_type='diffusers_ddpm' \
  --override teacher.pretrained_model_name_or_path="$TEACHER_ID" \
  --override teacher.local_files_only=false \
  --override teacher.num_inference_steps=$TEACHER_STEPS \
  --override data.time_grid_size=$TIME_GRID_SIZE
```

---

## 4) 训练：基线配置（shard 监督）

```bash
cd $PROJ
conda activate $ENV_NAME
python train.py --mode train_a100 --epochs $EPOCHS \
  --override experiment.name="$EXP_NAME" \
  --override data.dataset_type='trajectory_shards' \
  --override data.trajectory_shard_dir="$SHARD_ROOT" \
  --override data.batch_size=$BATCH_SIZE \
  --override data.num_workers=$NUM_WORKERS \
  --override data.time_grid_size=$TIME_GRID_SIZE \
  --override teacher.teacher_type='diffusers_ddpm' \
  --override teacher.pretrained_model_name_or_path="$TEACHER_ID" \
  --override teacher.num_inference_steps=$TEACHER_STEPS \
  --override train.checkpoint_dir="$CKPT_DIR" \
  --override train.log_every=50 \
  --override train.warp_update_every=1 \
  --override loss.defect_weight=0.5 \
  --override loss.warp_weight=0.25 \
  --override loss.boundary_weight=0.1
```

---

## 5) 训练监督与结果呈现（日志抽取）

训练日志中的关键字段：

- `train_loss`
- `val_loss`
- `peak_mem`
- `l_match`, `l_def`, `l_warp`, `l_boundary`

快速抽取 epoch 行（保存到文本）：

```bash
cd $PROJ
grep "Epoch " -n "$CKPT_DIR"/../logs/* 2>/dev/null || true
```

如果你是直接看终端日志，建议保存：

```bash
# 示例：用 tee 保存训练日志
python train.py ... 2>&1 | tee "$CKPT_DIR/train.log"
```

然后抽取：

```bash
grep "Epoch " "$CKPT_DIR/train.log"
```

---

## 6) 最终采样与分析验证

### 6.1 从 best checkpoint 采样

```bash
cd $PROJ
conda activate $ENV_NAME
python sample.py \
  --mode train_a100 \
  --checkpoint "$CKPT_DIR/best.pt" \
  --steps $SAMPLE_STEPS \
  --batch-size $SAMPLE_BATCH \
  --override data.image_size=32 \
  --override data.channels=3
```

默认会写入：

- `artifacts/samples_stepsK.pt`
- `artifacts/sample_diag_stepsK.pt`

### 6.2 推理 profile（NFE、latency、peak memory）

```bash
cd $PROJ
conda activate $ENV_NAME
python scripts/profile_infer.py
```

---

## 7) 生成模型效果参数（当前可直接得到）

当前仓库可直接得到：

- 训练损失曲线（终端日志）
- `L_match / L_def / L_warp / L_boundary`
- 采样步数 K 对 latency、NFE、peak memory 的影响
- 采样 schedule（`t_schedule`）与导出张量

当前仓库尚未内置（需要你后续补脚本）：

- FID / IS / Precision-Recall 等标准生成质量指标

建议后续新增 `scripts/eval_fid.py`，并保持输入接口：

```bash
python scripts/eval_fid.py \
  --samples artifacts/samples_steps4.pt \
  --ref-stats <cifar10_ref_stats.npz>
```

---

## 8) 可复用改造模板（后续模块变更）

你后续替换模块时，优先保持这几个配置键不变，整条 pipeline 可以复用：

- `teacher.teacher_type`
- `teacher.pretrained_model_name_or_path`
- `data.dataset_type`
- `data.trajectory_shard_dir`
- `loss.*`
- `train.*`

### 8.1 切换 teacher（仅示例）

```bash
--override teacher.teacher_type='diffusers_ddpm'
--override teacher.pretrained_model_name_or_path='<new_teacher_path_or_hf_id>'
```

### 8.2 切换监督权重

```bash
--override loss.defect_weight=0.8
--override loss.warp_weight=0.1
--override loss.boundary_weight=0.0
```

### 8.3 切换优化器策略（需要改代码）

当前优化器在 `src/dg_twfd/engine/trainer.py` 中定义。若你改 optimizer/scheduler，建议保持 CLI 命令不变，只在配置和 trainer 内部替换。

---

## 9) A100 日常增量实验命令模板

```bash
cd $PROJ
git pull --ff-only
conda activate $ENV_NAME

# 继续训练
python train.py --mode train_a100 \
  --override train.resume_path="$CKPT_DIR/best.pt" \
  --override train.checkpoint_dir="$CKPT_DIR" \
  --override data.dataset_type='trajectory_shards' \
  --override data.trajectory_shard_dir="$SHARD_ROOT" \
  --override teacher.teacher_type='diffusers_ddpm' \
  --override teacher.pretrained_model_name_or_path="$TEACHER_ID"
```

---

## 10) 执行顺序（最短版）

```text
1. 环境安装 -> 2. tests -> 3. collect train/val shards ->
4. train baseline -> 5. sample -> 6. profile -> 7. 记录结果并迭代权重/模块
```
