# DGTD v3 Round-4 Online Mainline Server Smoke

## 用途

以后每次 server eval，只保留三部分：

1. 环境激活
2. 算法验证命令
3. 日志回收命令

我会在你回传服务器结果后，直接把关键观察写到这份文档里。

## 1. 环境激活

```bash
export PROJ=/data2/yl7622/Zhengwei/DG-TWFD
cd "$PROJ"

source /data2/yl7622/anaconda/etc/profile.d/conda.sh
conda activate "$PROJ/.conda_envs/dgfm_map"

source scripts/experiments/activate_fm_cifar10.sh dgtd_v3_smoke round4_online_mainline

export DGTD_V3_R4_ROOT=/tmp/dgtd_v3_round4_online_mainline_smoke
rm -rf "$DGTD_V3_R4_ROOT"
mkdir -p "$DGTD_V3_R4_ROOT"
```

## 2. 算法验证命令

### 2.1 Online teacher preflight

```bash
cd "$PROJ"
python - <<'PY'
from dgfm.config import load_experiment_config
from dgtd.teacher import build_teacher_adapter

cfg = load_experiment_config(
    "configs/experiment/dgtd_cifar10_v3_smoke.yaml",
    overrides=[
        "dgtd.disable_online_teacher=false",
        "dgtd.use_online_teacher_data=true",
        "teacher.local_files_only=false",
    ],
)
adapter = build_teacher_adapter(cfg)
print("online_teacher_built", adapter.online_teacher is not None)
print("teacher_type", type(adapter.online_teacher).__name__ if adapter.online_teacher is not None else None)
adapter.prepare("cpu")
print("online_teacher_prepare", "ok")
PY
```

### 2.2 Train smoke

```bash
cd "$PROJ"
python scripts/run_train.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --run-root "$DGTD_V3_R4_ROOT" \
  --set dgtd.disable_online_teacher=false \
  --set dgtd.use_online_teacher_data=true \
  --set teacher.local_files_only=false \
  2>&1 | tee "$DGTD_V3_R4_ROOT/train.stdout_stderr.txt"
```

### 2.3 Sample smoke

```bash
cd "$PROJ"
python scripts/run_sample_dgtd.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --checkpoint "$DGTD_V3_R4_ROOT/checkpoints/last.pt" \
  --output-dir "$DGTD_V3_R4_ROOT/sample" \
  --steps 4 \
  2>&1 | tee "$DGTD_V3_R4_ROOT/sample.stdout_stderr.txt"
```

### 2.4 Eval smoke

```bash
cd "$PROJ"
python scripts/run_eval.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --checkpoint "$DGTD_V3_R4_ROOT/checkpoints/last.pt" \
  --eval-root "$DGTD_V3_R4_ROOT/eval" \
  --steps 1 2 4 \
  2>&1 | tee "$DGTD_V3_R4_ROOT/eval.stdout_stderr.txt"
```

### 2.5 Diagnostics export

```bash
cd "$PROJ"
python scripts/plot_dgtd_diagnostics.py \
  --log "$DGTD_V3_R4_ROOT/logs/train.jsonl" \
  --output-dir "$DGTD_V3_R4_ROOT/diag" \
  2>&1 | tee "$DGTD_V3_R4_ROOT/diag.stdout_stderr.txt"
```

## 3. 日志回收命令

### 3.1 只回收算法判断最需要的字段

```bash
echo "==== tail train.jsonl ===="
tail -n 5 "$DGTD_V3_R4_ROOT/logs/train.jsonl"

echo "==== checkpoint files ===="
find "$DGTD_V3_R4_ROOT/checkpoints" -maxdepth 2 -type f | sort

echo "==== sample files ===="
find "$DGTD_V3_R4_ROOT/sample" -maxdepth 3 -type f | sort

echo "==== eval files ===="
find "$DGTD_V3_R4_ROOT/eval" -maxdepth 4 -type f | sort

echo "==== diag files ===="
find "$DGTD_V3_R4_ROOT/diag" -maxdepth 3 -type f | sort
```

### 3.2 如果 train 成功，优先看这些字段

从 `train.jsonl` 里重点判断：

- `online_teacher_data`
- `continuation_sources`
- `online_anchor_used_rate`
- `online_continuation_rate`
- `cached_fallback_rate`
- `exact_mask_hit_rate`
- `alpha_online_mean/min/max`
- `train_bridge_state_teacher_error`
- `train_bridge_u_teacher_error`
- `train_teacher_rel_error_mean`
- `train_direct_bridge_gap`
- `eta`
- `stage`

## 4. 本轮已回收的真实结果

### 4.1 Online teacher preflight

- `online_teacher_built=True`
- `teacher_type=DiffusersDDPMTeacher`
- `online_teacher_prepare=ok`

### 4.2 Train smoke 关键观察

来自你回传的 `train.jsonl`：

- `online_teacher_data=true`
- `stage=warmup`
- `eta=0.95`
- `continuation_sources.online=1.0`
- `continuation_sources.cached_affine=0.0`
- `continuation_sources.cached_exact=0.0`
- `continuation_sources.bootstrap=0.0`
- `online_anchor_used_rate=1.0`
- `online_continuation_rate=1.0`
- `cached_fallback_rate=0.0`
- `exact_mask_hit_rate=0.0`
- `alpha_online_mean=0.9322488903999329`
- `alpha_online_min=0.8432849049568176`
- `alpha_online_max=0.9902514219284058`
- `train_bridge_state_teacher_error=0.02299138531088829`
- `train_bridge_u_teacher_error=0.029789606109261513`
- `train_teacher_rel_error_mean=0.05534832552075386`
- `train_direct_bridge_gap=0.002555012470111251`

算法含义：

- online continuation 已经真正成为主源，不再只是 online trajectory data path
- warmup 阶段没有退回 cached fallback
- alpha 分布接近 1 但没有爆掉，当前这条路是可走的
- bridge-side diagnostics 数值正常，没有出现明显异常发散

### 4.3 Checkpoint

已回传：

- `/tmp/dgtd_v3_round4_online_mainline_smoke/checkpoints/best.pt`
- `/tmp/dgtd_v3_round4_online_mainline_smoke/checkpoints/last.pt`

### 4.4 Sample / Eval / Diag

当前已确认 sample / eval / diag 都通过。

eval 结果：

- `step_count=1`: `fid=446.6433`
- `step_count=2`: `fid=419.2015`
- `step_count=4`: `fid=419.2667`

算法含义：

- eval 闭环是通的
- 1 步到 2 步有改善
- 2 步到 4 步基本持平
- 这些仍是 smoke 级 approximate FID，只能说明管线和趋势，不代表最终质量结论
