# Mac Local Development Migration Assessment

## 1. 结论

**可行，但只适合作为本地开发前端，不适合作为正式训练环境。**

推荐模式是：

- macOS：读代码、改代码、跑 import / pytest / CPU 或 MPS 小规模 smoke、生成服务器实验命令
- Linux 服务器：正式训练、DDP/NCCL、DDPM online teacher 大批量 rollout、FID/eval、长时间 GPU 实验

本地已验证的最小事实：

```text
import_forward_backward_ok (2, 3, 32, 32) 0.003341549774631858
```

即：在 CPU 上能 import 核心模块，构建 `local_map_resnet`，跑一次 DGTD residual
forward/backward。

## 2. 主要迁移障碍

### 可直接在 macOS 上运行

- `src/dgtd/sigma.py`
- `src/dgtd/warp.py`
- `src/dgtd/metrics.py`
- `src/dgtd/defect.py`
- `src/dgtd/cache.py` 的小型本地 shard / mock trajectory 路径
- `src/dgtd/teacher.py` 的 `teacher.type=none` / dummy teacher / cached affine 逻辑
- `src/dgfm/config`
- `src/dgfm/models/map.py` 的 `local_map_resnet`
- 大部分纯 Python pytest，包括 DGTD 单测和配置单测

### 需要轻微修改或安装额外依赖

- `torchvision`：数据集、sample/eval 保存图片依赖它，但 `pyproject.toml` 没列出。
- `numpy/scipy/matplotlib/pillow`：多处测试、诊断图、FID 工具实际需要。
- `torch-fidelity`：`dgfm.evaluators` 当前 import 时会触发 FID 相关 import；本地若不装，会影响 eval 入口 import。
- 设备选择：多处仍是 `cuda if available else cpu`，不会用 MPS。
- `scripts/experiments/create_map_branch_env.sh`：固定安装 CUDA 12.8 PyTorch wheel，不适合 Mac。
- `scripts/experiments/activate_fm_cifar10.sh`：默认 `/data2/...`、CUDA env、服务器路径；Mac 本地不应依赖它。

### 必须保留在 Linux 服务器运行

- `scripts/run_ddp_smoke.py`：显式要求 CUDA，并用 `nccl`。
- 正式 `scripts/run_train.py` / DGTD full run / distributed run。
- online `DiffusersDDPMTeacher` 的大批量 rollout。
- 大规模 sample/eval/FID。
- `scripts/prepare_teacher_trajectories.py`、`scripts/collect_teacher.py` 的真实 teacher 轨迹采集。
- 所有依赖 `/data2/...`、多 GPU、`CUDA_VISIBLE_DEVICES`、`MASTER_ADDR`、`MASTER_PORT` 的实验脚本。

## 3. 依赖审计

主工程 `src/`、`scripts/`、`tests/` 中没有发现硬依赖：

- `flash-attn`
- `xformers`
- `bitsandbytes`
- `apex`
- `triton`
- 自定义 CUDA extension

这些依赖主要出现在 `refs/` 外部参考代码里，不应纳入 Mac 本地开发路径。

主要真实依赖分层：

- 最小核心：`torch`, `PyYAML`, `pytest`
- 本地 smoke 推荐：`torchvision`, `numpy`, `scipy`, `matplotlib`, `pillow`
- 本地 eval import 兼容：`torch-fidelity`
- 可选 teacher：`diffusers`, `transformers`, `accelerate`, `safetensors`
- 可选感知损失：`piq`

## 4. 最小本地开发方案

### 4.1 创建 Mac Python 环境

推荐用 venv 或 conda，先不要安装 CUDA wheel：

```bash
cd /path/to/DG-TWFD

python3.10 -m venv .venv-mac
source .venv-mac/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision
python -m pip install -e '.[dev]'
python -m pip install numpy scipy matplotlib pillow torch-fidelity
```

如果只做 DGTD 代码/单测，不跑 online diffusers teacher，可先不装 teacher extra。

需要 online teacher import / 小模型调试时再装：

```bash
python -m pip install -e '.[teacher]'
```

### 4.2 最小 import / pytest

```bash
python - <<'PY'
import torch
import dgtd
from dgfm.config import load_experiment_config
from dgfm.models import build_map_model
print("torch", torch.__version__)
print("mps_available", torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False)
print("imports_ok")
PY

pytest tests/test_dgtd.py tests/test_dgfm_config.py tests/test_dgfm_overrides.py -q
```

### 4.3 最小 CPU forward/backward smoke

```bash
python - <<'PY'
import torch
from dgfm.config import load_experiment_config
from dgfm.models import build_map_model
from dgtd.teacher import build_teacher_adapter
from dgtd.defect import compute_dgtd_residual

cfg = load_experiment_config(
    "configs/experiment/dgtd_cifar10_v3_smoke.yaml",
    overrides=[
        "runtime.device=cpu",
        "runtime.amp=false",
        "train.batch_size=2",
        "train.num_workers=0",
        "teacher.type=none",
        "dgtd.disable_online_teacher=true",
        "dgtd.use_online_teacher_data=false",
        "model.hidden_channels=8",
        "model.time_embed_dim=8",
        "model.cond_dim=8",
        "model.num_res_blocks=1",
    ],
)

model = build_map_model(cfg)
adapter = build_teacher_adapter(cfg)
x_t = torch.randn(2, 3, 32, 32)
t = torch.tensor([0.1, 0.2])
s = torch.tensor([0.4, 0.5])
u = torch.tensor([0.8, 0.9])
out = compute_dgtd_residual(model, adapter, x_t, t, s, u, eta=0.0, trajectory=None, extra={})
loss = out["residual"].square().mean()
loss.backward()
print("ok", tuple(out["residual"].shape), float(loss.item()))
PY
```

## 5. 最小侵入式改造建议

不建议大改代码结构。建议只做以下小改造：

1. 增加统一设备 helper，例如 `src/dgfm/device.py`：

```python
def auto_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

2. 替换这些文件里的 `cuda if available else cpu`：

- `src/dgfm/trainers/baseline.py`
- `src/dgfm/trainers/map.py`
- `src/dgtd/train_dgtd.py`
- `src/dgfm/evaluators/common.py`
- `train.py`
- `sample.py`
- `scripts/collect_teacher.py`
- `scripts/prepare_teacher_trajectories.py`

3. 保持 AMP 仅 CUDA 启用；MPS/CPU 默认 `runtime.amp=false`。

4. 将 `torch-fidelity` 的 import 延迟到真正计算 FID 时，避免本地只 import evaluator 就需要 FID 依赖。

5. 给 `pyproject.toml` 增加一个轻量 extra，例如：

```toml
[project.optional-dependencies]
local-dev = [
  "pytest>=8.0",
  "torchvision",
  "numpy",
  "scipy",
  "matplotlib",
  "pillow",
]
eval = [
  "torch-fidelity==0.4.0",
]
```

6. 新增 Mac 本地 smoke config，避免误触 online teacher / FID / cache shard：

- `configs/experiment/dgtd_cifar10_v3_local_smoke.yaml`

建议覆盖：

```yaml
runtime:
  device: cpu
  amp: false
train:
  batch_size: 2
  num_workers: 0
  max_train_batches: 1
  max_val_batches: 1
teacher:
  type: none
dgtd:
  disable_online_teacher: true
  use_online_teacher_data: false
eval:
  metrics: []
```

7. 不要改服务器脚本；另写本地文档或 `scripts/experiments/create_local_dev_env.sh`，避免破坏现有 A100/A6000 workflow。

## 6. 本地测试命令示例

只测代码结构和算法单元，不测训练质量：

```bash
pytest tests/test_dgtd.py -q
pytest tests/test_dgfm_config.py tests/test_dgfm_overrides.py -q
pytest tests/test_dgfm_map_branch.py tests/test_dgfm_teacher_trajectory.py -q
```

如果已安装 `torchvision` / `torch-fidelity`，可以再跑：

```bash
pytest tests/test_dgfm_fid.py tests/test_dgfm_external_eval.py -q
```

不建议在 Mac 默认跑：

```bash
pytest
```

原因是全量测试可能触发 FID、dataset、teacher、外部参考路径或较慢模型构建。

## 7. 服务器正式实验命令模板

Mac 本地只生成命令文档；正式运行仍放服务器。

```bash
export PROJ=/data2/yl7622/Zhengwei/DG-TWFD
cd "$PROJ"

source /data2/yl7622/anaconda/etc/profile.d/conda.sh
conda activate "$PROJ/.conda_envs/dgfm_map"
source scripts/experiments/activate_fm_cifar10.sh dgtd_v3_smoke roundX_tag

export RUN_ROOT=/tmp/dgtd_roundX_smoke
rm -rf "$RUN_ROOT"
mkdir -p "$RUN_ROOT"

python scripts/run_train.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --run-root "$RUN_ROOT" \
  --set dgtd.disable_online_teacher=false \
  --set dgtd.use_online_teacher_data=true \
  --set teacher.local_files_only=false \
  2>&1 | tee "$RUN_ROOT/train.stdout_stderr.txt"

python scripts/run_sample_dgtd.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --checkpoint "$RUN_ROOT/checkpoints/last.pt" \
  --output-dir "$RUN_ROOT/sample" \
  --steps 4 \
  2>&1 | tee "$RUN_ROOT/sample.stdout_stderr.txt"

python scripts/run_eval.py \
  --config configs/experiment/dgtd_cifar10_v3_smoke.yaml \
  --checkpoint "$RUN_ROOT/checkpoints/last.pt" \
  --eval-root "$RUN_ROOT/eval" \
  --steps 1 2 4 \
  2>&1 | tee "$RUN_ROOT/eval.stdout_stderr.txt"

tail -n 5 "$RUN_ROOT/logs/train.jsonl"
find "$RUN_ROOT/checkpoints" "$RUN_ROOT/sample" "$RUN_ROOT/eval" -maxdepth 3 -type f | sort
```

## 8. 最终判断

Mac 作为本地开发前端是可行的。  
最小改造重点不是迁移训练，而是隔离三类服务器专属路径：

- CUDA/DDP/NCCL
- diffusers teacher 大规模 rollout
- FID / official eval / 长时间实验

只要本地默认使用 `teacher.type=none` 或 dummy teacher、`runtime.device=cpu/mps`、
`runtime.amp=false`、小 batch、小模型配置，就能满足“看、改、测、产出服务器命令”的目标。

