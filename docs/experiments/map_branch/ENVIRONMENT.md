# map_branch environment

This branch now assumes one server-local working root:

```bash
/data2/yl7622/Zhengwei/DG-TWFD
```

Everything lives under this directory:

- code
- datasets
- runs
- eval outputs
- refs
- teacher trajectories
- HuggingFace cache
- torch cache
- conda environments

There is no default archive mirror and no default backup directory.

## 1. Default roots

After sourcing
[activate_fm_cifar10.sh](/home/gzwlinux/vscode/gitProject/DG-TWFD/scripts/experiments/activate_fm_cifar10.sh),
the default layout is:

```bash
PROJ=/data2/yl7622/Zhengwei/DG-TWFD
DATA_ROOT=$PROJ/datasets
RUNS_ROOT=$PROJ/runs
EVAL_ROOT=$PROJ/eval
REF_ROOT=$PROJ/refs
TRAJ_ROOT=$PROJ/teacher_traj/cifar10_ddpm128_p33
IMAGENET_RAW_ROOT=$DATA_ROOT/imagenet_raw
IMAGENET64_PREPROCESSED=$DATA_ROOT/imagenet64
HF_HOME=$PROJ/.hf_home
TORCH_HOME=$PROJ/.torch
```

This keeps the workflow simple:

- experiment outputs stay under `runs/` and `eval/`
- model/data caches stay under the same project root
- no `/cache/...` split
- no `/temp/...` split

## 2. Environment creation

Use the branch-local helper:

```bash
cd /data2/yl7622/Zhengwei/DG-TWFD
bash scripts/experiments/create_map_branch_env.sh dgfm_map
conda activate /data2/yl7622/Zhengwei/DG-TWFD/.conda_envs/dgfm_map
```

This script now keeps a pure workflow:

- no pip mirror configuration
- no HuggingFace mirror configuration
- no GitHub asset mirror configuration
- no wheel cache staging

It installs:

- Python `3.10`
- PyTorch `2.10.0`
- torchvision `0.25.0`
- `diffusers`
- `transformers`
- `accelerate`
- `torch-fidelity`
- `piq`
- project editable install

## 3. Manual environment creation

If you do not want the helper script:

```bash
conda create -p /data2/yl7622/Zhengwei/DG-TWFD/.conda_envs/dgfm_map python=3.10 -y
conda activate /data2/yl7622/Zhengwei/DG-TWFD/.conda_envs/dgfm_map

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0
python -m pip install \
  PyYAML==6.0.3 \
  numpy==2.2.3 \
  scipy==1.15.3 \
  torch-fidelity==0.4.0 \
  'diffusers>=0.30' \
  'transformers>=4.40' \
  'accelerate>=0.30' \
  'safetensors>=0.4' \
  'piq>=0.8' \
  matplotlib \
  pillow \
  pytest
python -m pip install -e .
```

## 4. Validation

Minimal checks:

```bash
cd /data2/yl7622/Zhengwei/DG-TWFD
conda activate /data2/yl7622/Zhengwei/DG-TWFD/.conda_envs/dgfm_map
pytest tests/test_dgfm_config.py tests/test_dgfm_map_branch.py tests/test_dgfm_teacher_sampler.py -q
```

## 5. Distributed-training prep

The current workflow remains single-process by default, but activation now also
exports these placeholders:

```bash
NNODES
NODE_RANK
NPROC_PER_NODE
MASTER_ADDR
MASTER_PORT
```

Current commands in
[A100_PIPELINE.md](/home/gzwlinux/vscode/gitProject/DG-TWFD/docs/experiments/map_branch/A100_PIPELINE.md)
do not switch to `torchrun` yet. These variables are only reserved so the
server branch can grow into distributed training later without another path
cleanup pass.
