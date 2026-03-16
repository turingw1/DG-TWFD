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
export TRAIN_MODE=train_a100_stable

# 大文件统一放 /cache/Zhengwei，按 dg_twfd_* 规范分目录
export SHARD_ROOT=/cache/Zhengwei/dg_twfd_shards/$EXP_NAME
export RUN_ROOT=/cache/Zhengwei/dg_twfd_runs/$EXP_NAME

export TEACHER_ID=google/ddpm-cifar10-32
export CKPT_DIR=$RUN_ROOT/checkpoints
export ARTIFACT_ROOT=$RUN_ROOT/samples
export TRAIN_LOG=$RUN_ROOT/train.log
```

### 0.1 正式实验版本入口

- `train_a100_base`
  - A100 DDPM-CIFAR10 的公共基础配置，负责 teacher、数据、路径、A100 训练骨架
- `train_a100_stable`
  - 当前推荐正式版本；在 `base` 基础上固定多步稳定性相关设置
- `train_a100`
  - 向后兼容入口，等价于 `train_a100_stable`
- `train_a100_ablate_*`
  - 单损失 ablation 入口，全部继承自 `train_a100_base`

原则：

- 正式实验优先切换 `TRAIN_MODE`
- 文档命令只调用 profile，不在命令行重复拼业务参数
- 命令行 `--override` 只保留给临时人工试验

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

当前 profile 已固定以下策略：

- teacher sampler 可配 `ddim/ddpm`，推荐 `ddim`（deterministic）
- trajectory 缓存推荐 `num_inference_steps=128` + `time_grid_size=129`
- train/val split 使用不同 seed 偏移，避免重合
- Diffusers teacher 在 CUDA 上默认 FP16 推理，并减少逐步 CPU 拷贝瓶颈
- `teacher.pretrained_model_name_or_path` 直接从 `$TEACHER_ID` 注入
- `trajectory_shard_dir` 与 `checkpoint_dir` 直接从 `$SHARD_ROOT/$CKPT_DIR` 注入

---

## 3. teacher trajectory 采集（先 smoke，再正式）

### 3.1 Smoke：train split（先验证链路）

```bash
cd $PROJ
conda activate $ENV_NAME
CUDA_VISIBLE_DEVICES=1 python scripts/collect_teacher.py \
  --mode "$TRAIN_MODE" \
  --split train \
  --num-samples 2048 \
  --shard-size 512 \
  --output-dir $SHARD_ROOT \
  --emit-supervision-overrides \
  --target-mem-util 0.70
```

### 3.2 Smoke：val split

```bash
cd $PROJ
conda activate $ENV_NAME
CUDA_VISIBLE_DEVICES=1 python scripts/collect_teacher.py \
  --mode "$TRAIN_MODE" \
  --split val \
  --num-samples 512 \
  --shard-size 512 \
  --output-dir $SHARD_ROOT \
  --emit-supervision-overrides \
  --target-mem-util 0.70
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
  --mode "$TRAIN_MODE" \
  --split train \
  --num-samples 200000 \
  --shard-size 1024 \
  --output-dir $SHARD_ROOT
```

```bash
cd $PROJ
conda activate $ENV_NAME
CUDA_VISIBLE_DEVICES=1 python scripts/collect_teacher.py \
  --mode "$TRAIN_MODE" \
  --split val \
  --num-samples 10000 \
  --shard-size 1024 \
  --output-dir $SHARD_ROOT
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
- student 输出改为“按 `delta=t-s` 缩放的有界残差”，用于抑制多步采样时每一步漂移累积
- 训练日志新增 `steps_per_sec`，便于吞吐调优

---

## 5. A100 训练（shard 监督）

### 5.1 正式训练（默认使用稳定版 profile）

```bash
cd $PROJ
conda activate $ENV_NAME
mkdir -p $CKPT_DIR $ARTIFACT_ROOT
DG_TWFD_COMPILE=1 CUDA_VISIBLE_DEVICES=1 python train.py --mode "$TRAIN_MODE" --epochs 20 \
  2>&1 | tee "$TRAIN_LOG"
```

### 5.2 若显存仍低（<60%），继续拉高 batch

```bash
# 临时手动试验才使用 override，不改文档主流程
DG_TWFD_COMPILE=1 CUDA_VISIBLE_DEVICES=1 python train.py --mode "$TRAIN_MODE" \
  --override data.batch_size=768 \
  --override train.learning_rate=6e-4
```

### 5.3 当前推荐的稳定性优先开关

这些设置已固化在 `train_a100_stable` profile 中：

- `model.residual_scale_by_delta=true`
- `model.residual_tanh_scale=0.75`
- `loss.composition_weight=0.25`
- `loss.composition_batch_size=8`
- `train.composition_update_every=16`

理由：

- `delta` 缩放保证小步长采样时单步更新幅度同步缩小，减少多步误差累积；
- `tanh` 限幅避免 student 在高噪声区输出过大残差；
- composition 监督保留但降频，避免 teacher runtime 成为训练主耗时。

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

重点检查多步稳定性：

```bash
grep "comp=" "$TRAIN_LOG" | tail -n 50
```

---

## 7. 采样与分析验证

### 7.1 从 best checkpoint 采样

```bash
cd $PROJ
conda activate $ENV_NAME
CUDA_VISIBLE_DEVICES=1 python sample.py \
  --mode "$TRAIN_MODE" \
  --checkpoint "$CKPT_DIR/best.pt" \
  --output-dir "$ARTIFACT_ROOT" \
  --steps 16 \
  --batch-size 64 \
  --disable-boundary
```

重点查看：

- `sampling_ema_student: True`
- `sample_stats`
- `step_stats`

判据：

- 若 `step_stats` 中 `std` 在后半程持续快速上升，同时 `|mean|` 单向漂移变大，说明 student 单步更新仍偏大；
- 若 1-step 质量尚可但 16-step 明显崩坏，优先继续调 `model.residual_tanh_scale` 到 `0.5~0.75`，其次再调 `loss.composition_weight`。

### 7.2 推理 profile（1/2/4/8/16 steps）

```bash
cd $PROJ
conda activate $ENV_NAME
CUDA_VISIBLE_DEVICES=1 python scripts/profile_infer.py \
  --mode "$TRAIN_MODE" \
  --checkpoint "$CKPT_DIR/best.pt" \
  --disable-boundary
```

### 7.3 采样图片拼接预览（直观查看结果）

```bash
cd $PROJ
conda activate $ENV_NAME
python scripts/preview_samples.py \
  --samples "$ARTIFACT_ROOT/samples_steps4.pt" \
  --output "$ARTIFACT_ROOT/samples_steps4_preview.png" \
  --nrow 8 \
  --max-images 64
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

DG_TWFD_COMPILE=1 CUDA_VISIBLE_DEVICES=1 python train.py --mode "$TRAIN_MODE" \
  --override train.resume_path="$CKPT_DIR/best.pt"
```

---

## 10. 新增正式版本的流程（不要直接堆 override）

若需要开一个新正式版本，不要直接在命令里堆超参数。做法是：

1. 复制一个现有 profile，例如从 `train_a100_stable.yaml` 派生
2. 给新版本起明确名字，例如 `train_a100_stable_v2`
3. 把该版本的差异固化在 profile
4. 文档与实验命令只切换 `TRAIN_MODE`

示例：

```yaml
# config/profiles/train_a100_stable_v2.yaml
extends: train_a100_base

model:
  residual_tanh_scale: 0.50

loss:
  composition_weight: 0.35
```

然后只需：

```bash
export TRAIN_MODE=train_a100_stable_v2
export EXP_NAME=ddpm_cifar10_a100_v2_stable_v2
export SHARD_ROOT=/cache/Zhengwei/dg_twfd_shards/$EXP_NAME
export RUN_ROOT=/cache/Zhengwei/dg_twfd_runs/$EXP_NAME
export CKPT_DIR=$RUN_ROOT/checkpoints
export ARTIFACT_ROOT=$RUN_ROOT/samples
export TRAIN_LOG=$RUN_ROOT/train.log
```

若新版本复用现有 teacher shards，只保留原 `SHARD_ROOT` 不变即可。

```bash
cd $PROJ
conda activate $ENV_NAME
mkdir -p "$CKPT_DIR" "$ARTIFACT_ROOT"
DG_TWFD_COMPILE=1 CUDA_VISIBLE_DEVICES=1 python train.py --mode "$TRAIN_MODE" --epochs 20 \
  2>&1 | tee "$TRAIN_LOG"
```

对应采样：

```bash
cd $PROJ
conda activate $ENV_NAME
CUDA_VISIBLE_DEVICES=1 python sample.py \
  --mode "$TRAIN_MODE" \
  --checkpoint "$CKPT_DIR/best.pt" \
  --output-dir "$ARTIFACT_ROOT" \
  --steps 16 \
  --batch-size 64 \
  --disable-boundary
```

说明：`sample.py` 与 `profile_infer.py` 默认优先加载 checkpoint 里的 EMA student；如需对照可加 `--no-ema`。
同时 `sample.py` 会输出每一步的 `mean/std/min/max`，并把中间态序列保存在 `sample_diag_steps*.pt` 的 `diagnostics.x_steps` 中。

查看第 0 步或最后一步预览：

```bash
python scripts/preview_samples.py \
  --samples "$ARTIFACT_ROOT/sample_diag_steps16.pt" \
  --step-index 0 \
  --output "$ARTIFACT_ROOT/step0_preview.png"
```

```bash
python scripts/preview_samples.py \
  --samples "$ARTIFACT_ROOT/sample_diag_steps16.pt" \
  --step-index 16 \
  --output "$ARTIFACT_ROOT/step16_preview.png"
```

---

## 11. 本阶段固定要求（执行时请保持）

- 大文件与 shard 路径固定：
  - `SHARD_ROOT=/cache/Zhengwei/dg_twfd_shards/$EXP_NAME`
  - `RUN_ROOT=/cache/Zhengwei/dg_twfd_runs/$EXP_NAME`
- 代码改动后只更新本文件（阶段性要求）
- 工作流固定：
  - 本地修改 + 更新文档
  - 你在 A100 跑
  - 回传输出
  - 本地继续优化并同步 git

---

## 12. CIFAR 单损失 Ablation（先不做对比分析）

### 12.1 可用 profile

- `train_a100_ablate_match`：只开 match loss
- `train_a100_ablate_defect`：只开 defect loss
- `train_a100_ablate_warp`：只开 warp loss
- `train_a100_ablate_boundary`：只开 boundary loss

> 说明：`warp-only` 与 `boundary-only` 的生成质量通常会显著弱于 match-based 方案，本阶段先完成流程与产物留档。

### 12.2 单个实验通用变量模板

```bash
export ABLATION_NAME=ablate_match
export ABLATION_PROFILE=train_a100_ablate_match
export EXP_ABL=ddpm_cifar10_${ABLATION_NAME}
export RUN_ROOT_ABL=/cache/Zhengwei/dg_twfd_runs/$EXP_ABL
export CKPT_DIR_ABL=$RUN_ROOT_ABL/checkpoints
export ARTIFACT_ROOT_ABL=$RUN_ROOT_ABL/samples
export TRAIN_LOG_ABL=$RUN_ROOT_ABL/train.log
mkdir -p $CKPT_DIR_ABL $ARTIFACT_ROOT_ABL
```

### 12.3 训练（单损失）

```bash
cd $PROJ
conda activate $ENV_NAME
DG_TWFD_COMPILE=1 CUDA_VISIBLE_DEVICES=1 python train.py --mode $ABLATION_PROFILE --epochs 20 \
  2>&1 | tee "$TRAIN_LOG_ABL"
```

### 12.4 采样 + profile + 预览（每个 ablation 都执行）

```bash
cd $PROJ
conda activate $ENV_NAME
CUDA_VISIBLE_DEVICES=1 python sample.py \
  --mode $ABLATION_PROFILE \
  --checkpoint "$CKPT_DIR_ABL/best.pt" \
  --output-dir "$ARTIFACT_ROOT_ABL" \
  --steps 16 \
  --batch-size 64 \
  --disable-boundary
```

```bash
cd $PROJ
conda activate $ENV_NAME
CUDA_VISIBLE_DEVICES=1 python scripts/profile_infer.py \
  --mode $ABLATION_PROFILE \
  --checkpoint "$CKPT_DIR_ABL/best.pt" \
  --disable-boundary
```

```bash
cd $PROJ
conda activate $ENV_NAME
python scripts/preview_samples.py \
  --samples "$ARTIFACT_ROOT_ABL/samples_steps16.pt" \
  --output "$ARTIFACT_ROOT_ABL/samples_steps16_preview.png" \
  --nrow 8 \
  --max-images 64
```

### 12.5 产物检查（每个 ablation）

```bash
ls -lh "$CKPT_DIR_ABL"/best.pt
ls -lh "$ARTIFACT_ROOT_ABL"/samples_steps16.pt
ls -lh "$ARTIFACT_ROOT_ABL"/sample_diag_steps16.pt
ls -lh "$ARTIFACT_ROOT_ABL"/samples_steps16_preview.png
grep "Epoch " "$TRAIN_LOG_ABL"
grep "match=" "$TRAIN_LOG_ABL" | tail -n 30
```

### 12.6 四个实验顺序执行示例

```bash
for pair in \
  "ablate_match train_a100_ablate_match" \
  "ablate_defect train_a100_ablate_defect" \
  "ablate_warp train_a100_ablate_warp" \
  "ablate_boundary train_a100_ablate_boundary"
do
  set -- $pair
  export ABLATION_NAME=$1
  export ABLATION_PROFILE=$2
  export EXP_ABL=ddpm_cifar10_${ABLATION_NAME}
  export RUN_ROOT_ABL=/cache/Zhengwei/dg_twfd_runs/$EXP_ABL
  export CKPT_DIR_ABL=$RUN_ROOT_ABL/checkpoints
  export ARTIFACT_ROOT_ABL=$RUN_ROOT_ABL/samples
  export TRAIN_LOG_ABL=$RUN_ROOT_ABL/train.log
  mkdir -p "$CKPT_DIR_ABL" "$ARTIFACT_ROOT_ABL"

  DG_TWFD_COMPILE=1 CUDA_VISIBLE_DEVICES=1 python train.py --mode $ABLATION_PROFILE --epochs 20 \
    2>&1 | tee "$TRAIN_LOG_ABL"

  CUDA_VISIBLE_DEVICES=1 python sample.py --mode $ABLATION_PROFILE \
    --checkpoint "$CKPT_DIR_ABL/best.pt" \
    --output-dir "$ARTIFACT_ROOT_ABL" \
    --steps 16 --batch-size 64 --disable-boundary

  CUDA_VISIBLE_DEVICES=1 python scripts/profile_infer.py --mode $ABLATION_PROFILE \
    --checkpoint "$CKPT_DIR_ABL/best.pt" --disable-boundary

  python scripts/preview_samples.py \
    --samples "$ARTIFACT_ROOT_ABL/samples_steps16.pt" \
    --output "$ARTIFACT_ROOT_ABL/samples_steps16_preview.png" \
    --nrow 8 --max-images 64
done
```
