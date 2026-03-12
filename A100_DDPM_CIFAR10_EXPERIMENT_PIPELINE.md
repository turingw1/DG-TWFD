# A100 DDPM-CIFAR10 完整实验流水线（指令级）

本手册对应当前代码实现，目标是：  
`teacher trajectory 采集 -> shard 监督训练 -> 采样验证 -> 指标汇总 -> 迭代调参`

并满足你的循环工作流：本地改代码/文档 -> 你在 A100 跑 -> 回传日志 -> 本地继续优化。

---

## 0. 固定路径与实验变量

```bash
export PROJ=~/workspace/Zhengwei/DG-TWFD
export ENV_NAME=consistency
export EXP_NAME=ddpm_cifar10_a100_v2

# 大文件统一放 /cache/Zhengwei，按 dg_twfd_* 规范分目录
export SHARD_ROOT=/cache/Zhengwei/dg_twfd_shards/$EXP_NAME
export RUN_ROOT=/cache/Zhengwei/dg_twfd_runs/$EXP_NAME

export TEACHER_ID=google/ddpm-cifar10-32
export CKPT_DIR=$RUN_ROOT/checkpoints
export ARTIFACT_ROOT=$RUN_ROOT/samples
export TRAIN_LOG=$RUN_ROOT/train.log
```

---

## 1. 拉取最新代码并安装依赖

```bash
cd $PROJ
git pull --ff-only
conda activate $ENV_NAME
python -m pip install -e '.[dev,teacher]'
```

---

## 2. 采样逻辑对应（当前实现）

当前代码已切到以下策略：

- teacher sampler 可配 `ddim/ddpm`，推荐 `ddim`（deterministic）
- trajectory 缓存推荐 `num_inference_steps=128` + `time_grid_size=129`
- train/val split 使用不同 seed 偏移，避免重合
- Diffusers teacher 在 CUDA 上默认 FP16 推理，并减少逐步 CPU 拷贝瓶颈

---

## 3. teacher trajectory 采集（先 smoke，再正式）

### 3.1 Smoke：train split（先验证链路）

```bash
cd $PROJ
conda activate $ENV_NAME
CUDA_VISIBLE_DEVICES=1 python scripts/collect_teacher.py \
  --mode train_a100 \
  --split train \
  --num-samples 2048 \
  --shard-size 512 \
  --output-dir $SHARD_ROOT \
  --emit-supervision-overrides \
  --target-mem-util 0.70 \
  --override teacher.teacher_type='diffusers_ddpm' \
  --override teacher.pretrained_model_name_or_path="$TEACHER_ID" \
  --override teacher.solver='ddim' \
  --override teacher.num_inference_steps=128 \
  --override data.time_grid_size=129
```

### 3.2 Smoke：val split

```bash
cd $PROJ
conda activate $ENV_NAME
CUDA_VISIBLE_DEVICES=1 python scripts/collect_teacher.py \
  --mode train_a100 \
  --split val \
  --num-samples 512 \
  --shard-size 512 \
  --output-dir $SHARD_ROOT \
  --emit-supervision-overrides \
  --target-mem-util 0.70 \
  --override teacher.teacher_type='diffusers_ddpm' \
  --override teacher.pretrained_model_name_or_path="$TEACHER_ID" \
  --override teacher.solver='ddim' \
  --override teacher.num_inference_steps=128 \
  --override data.time_grid_size=129
```

Smoke 完成后会自动生成：

- `$SHARD_ROOT/supervision_overrides_train.yaml`
- `$SHARD_ROOT/supervision_overrides_train.txt`
- `$SHARD_ROOT/supervision_overrides_val.yaml`
- `$SHARD_ROOT/supervision_overrides_val.txt`

其中 `*.txt` 每行都是一个 `--override ...`，可直接复制到训练命令中。

### 3.3 正式采集（建议起步规模）

```bash
cd $PROJ
conda activate $ENV_NAME
CUDA_VISIBLE_DEVICES=1 python scripts/collect_teacher.py \
  --mode train_a100 \
  --split train \
  --num-samples 200000 \
  --shard-size 1024 \
  --output-dir $SHARD_ROOT \
  --override teacher.teacher_type='diffusers_ddpm' \
  --override teacher.pretrained_model_name_or_path="$TEACHER_ID" \
  --override teacher.solver='ddim' \
  --override teacher.num_inference_steps=128 \
  --override data.time_grid_size=129
```

```bash
cd $PROJ
conda activate $ENV_NAME
CUDA_VISIBLE_DEVICES=1 python scripts/collect_teacher.py \
  --mode train_a100 \
  --split val \
  --num-samples 10000 \
  --shard-size 1024 \
  --output-dir $SHARD_ROOT \
  --override teacher.teacher_type='diffusers_ddpm' \
  --override teacher.pretrained_model_name_or_path="$TEACHER_ID" \
  --override teacher.solver='ddim' \
  --override teacher.num_inference_steps=128 \
  --override data.time_grid_size=129
```

检查 shard：

```bash
find $SHARD_ROOT -maxdepth 2 -type f | sort | head -n 20
find $SHARD_ROOT -maxdepth 2 -type f | sort | tail -n 20
```

---

## 4. 训练逻辑对应（当前实现）

当前代码已对齐到如下抽样与监督逻辑：

- pair 抽样：短/中/长跨度默认 `4:4:2`
- semigroup 三元组链长分布：短/中/长默认 `3:5:2`
- warp triplet：local 邻域采样（默认 gap `2/4`）
- 训练日志新增 `steps_per_sec`，便于吞吐调优

---

## 5. A100 训练（shard 监督）

### 5.1 基线训练（推荐先关 boundary，加速稳定）

先查看 smoke 自动给出的监督参数：

```bash
cat $SHARD_ROOT/supervision_overrides_train.txt
```

```bash
cd $PROJ
conda activate $ENV_NAME
mkdir -p $CKPT_DIR $ARTIFACT_ROOT
DG_TWFD_COMPILE=1 CUDA_VISIBLE_DEVICES=1 python train.py --mode train_a100 --epochs 20 \
  --override experiment.name="$EXP_NAME" \
  --override data.dataset_type='trajectory_shards' \
  --override data.trajectory_shard_dir="$SHARD_ROOT" \
  --override data.batch_size=128 \
  --override data.num_workers=16 \
  --override data.prefetch_factor=8 \
  --override teacher.teacher_type='diffusers_ddpm' \
  --override teacher.pretrained_model_name_or_path="$TEACHER_ID" \
  --override teacher.solver='ddim' \
  --override teacher.num_inference_steps=128 \
  --override data.time_grid_size=129 \
  --override train.checkpoint_dir="$CKPT_DIR" \
  --override train.log_every=20 \
  --override train.warp_update_every=1 \
  --override loss.defect_weight=0.5 \
  --override loss.warp_weight=0.25 \
  --override loss.boundary_weight=0.1 \
  --override boundary.enable_until_step=0 \
  2>&1 | tee "$TRAIN_LOG"
```

### 5.2 若显存仍低（<60%），继续拉高 batch

```bash
# 依次尝试 192 -> 256
--override data.batch_size=192
```

---

## 6. 训练结果呈现（日志抽取）

抽取 epoch 结果：

```bash
grep "Epoch " "$TRAIN_LOG"
```

抽取吞吐与显存：

```bash
grep "sps=" "$TRAIN_LOG" | tail -n 50
```

抽取 loss 分量：

```bash
grep "match=" "$TRAIN_LOG" | tail -n 50
```

---

## 7. 采样与分析验证

### 7.1 从 best checkpoint 采样

```bash
cd $PROJ
conda activate $ENV_NAME
CUDA_VISIBLE_DEVICES=1 python sample.py \
  --mode train_a100 \
  --checkpoint "$CKPT_DIR/best.pt" \
  --output-dir "$ARTIFACT_ROOT" \
  --steps 4 \
  --batch-size 64 \
  --override data.image_size=32 \
  --override data.channels=3
```

### 7.2 推理 profile（1/2/4/8/16 steps）

```bash
cd $PROJ
conda activate $ENV_NAME
CUDA_VISIBLE_DEVICES=1 python scripts/profile_infer.py \
  --mode train_a100 \
  --checkpoint "$CKPT_DIR/best.pt"
```

---

## 8. 你每轮需要回传给我的结果

每轮实验后请回传：

1. `grep "Epoch " $TRAIN_LOG` 输出
2. `grep "sps=" $TRAIN_LOG | tail -n 30` 输出
3. `sample.py` 完整输出
4. `profile_infer.py` 完整输出
5. 若失败，完整 traceback

---

## 9. 增量迭代命令模板（继续实验）

```bash
cd $PROJ
git pull --ff-only
conda activate $ENV_NAME

DG_TWFD_COMPILE=1 CUDA_VISIBLE_DEVICES=1 python train.py --mode train_a100 \
  --override train.resume_path="$CKPT_DIR/best.pt" \
  --override train.checkpoint_dir="$CKPT_DIR" \
  --override data.dataset_type='trajectory_shards' \
  --override data.trajectory_shard_dir="$SHARD_ROOT" \
  --override teacher.teacher_type='diffusers_ddpm' \
  --override teacher.pretrained_model_name_or_path="$TEACHER_ID" \
  --override teacher.solver='ddim' \
  --override teacher.num_inference_steps=128
```

---

## 10. 本阶段固定要求（执行时请保持）

- 大文件与 shard 路径固定：
  - `SHARD_ROOT=/cache/Zhengwei/dg_twfd_shards/$EXP_NAME`
  - `RUN_ROOT=/cache/Zhengwei/dg_twfd_runs/$EXP_NAME`
- 代码改动后只更新本文件（阶段性要求）
- 工作流固定：
  - 本地修改 + 更新文档
  - 你在 A100 跑
  - 回传输出
  - 本地继续优化并同步 git
