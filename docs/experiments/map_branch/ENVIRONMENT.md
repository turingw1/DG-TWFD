# map_branch environment

This branch does not assume that the server already has a usable
`consistency` environment. The recommended environment root is:

```bash
/cache/$USER/conda_envs
```

The recommended wheel cache root is:

```bash
/cache/$USER/wheels
```

## 1. Reference environment

Current local reference environment for this branch:

- Python `3.10.19`
- PyTorch `2.10.0+cu128`
- torchvision `0.25.0+cu128`
- numpy `2.2.3`
- scipy `1.15.3`
- PyYAML `6.0.3`
- torch-fidelity `0.4.0`
- diffusers `>=0.30`
- transformers `>=4.40`
- accelerate `>=0.30`
- safetensors `>=0.4`
- piq `>=0.8`

Minimum practical package set for current `map_branch`:

- `python=3.10`
- `torch==2.10.0`
- `torchvision==0.25.0`
- `PyYAML==6.0.3`
- `numpy==2.2.3`
- `scipy==1.15.3`
- `torch-fidelity==0.4.0`
- `diffusers>=0.30`
- `transformers>=4.40`
- `accelerate>=0.30`
- `safetensors>=0.4`
- `piq>=0.8`
- `matplotlib`
- `pillow`
- `pytest`

## 2. Mirror strategy

Different assets come from different upstreams. Use different mirrors for
different classes of downloads.

### 2.1 pip packages

Recommended default:

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

Optional alternatives:

```bash
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export PIP_INDEX_URL=https://pypi.mirrors.ustc.edu.cn/simple
```

If your server already has a global pip config, check it first:

```bash
python -m pip config list
```

### 2.2 HuggingFace

Recommended:

```bash
export HF_HOME=/cache/huggingface
export HF_HUB_CACHE=/cache/huggingface/hub
export HF_ENDPOINT=https://hf-mirror.com
```

This matters for:

- `diffusers`
- `transformers`
- online teacher loading

### 2.3 GitHub assets

For GitHub-hosted binary assets, use:

```bash
export DGFM_TORCH_FIDELITY_MIRROR_PREFIX=https://githubfast.com/
```

This currently affects:

- `torch_fidelity` Inception weights during FID evaluation

### 2.4 PyTorch wheels

This is the important exception.

Your current slow download:

```text
https://download-r2.pytorch.org/whl/cu128/torch-2.10.0+cu128-...
```

does **not** come from PyPI, HuggingFace, or GitHub. Therefore:

- `PIP_INDEX_URL` does not solve this
- `HF_ENDPOINT` does not solve this
- `githubfast` does not solve this

The recommended solution is:

1. download the large Torch wheels once into `/cache/$USER/wheels`
2. install from local wheel files
3. reuse that wheel cache across future environment creation

## 3. Recommended creation method

Use the branch-local setup script:

```bash
git clone https://github.com/turingw1/DG-TWFD.git
git checkout map_branch_ctm_explicit_map
cd ~/workspace/Zhengwei/DG-TWFD
bash scripts/experiments/create_map_branch_env.sh dgfm_map
conda activate /cache/$USER/conda_envs/dgfm_map
```

By default, the script now:

- creates the environment under `/cache/$USER/conda_envs/<env_name>`
- writes pip mirror config to the new environment
- caches Torch wheels under `/cache/$USER/wheels/torch-cu128`
- installs Torch from local wheel files when available

If your server should use a different env root:

```bash
bash scripts/experiments/create_map_branch_env.sh dgfm_map /cache/custom_user/conda_envs
conda activate /cache/custom_user/conda_envs/dgfm_map
```

## 4. Fastest path for slow Torch downloads

If downloading `torch-2.10.0+cu128` from `download-r2.pytorch.org` is too slow,
do not repeatedly reinstall directly from the network.

### Option A. Pre-download once to local wheel cache

```bash
mkdir -p /cache/$USER/wheels/torch-cu128
cd /cache/$USER/wheels/torch-cu128

wget -c https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl
wget -c https://download.pytorch.org/whl/cu128/torchvision-0.25.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl
```

Then create the env:

```bash
cd ~/workspace/Zhengwei/DG-TWFD
bash scripts/experiments/create_map_branch_env.sh dgfm_map
```

The script will detect the local wheel files and skip the slow online Torch
download.

### Option B. Use a custom Torch wheel mirror URL

If you already have an internal mirror or object store that serves the same
wheel filenames, set:

```bash
export TORCH_WHL_URL=https://<your-mirror>/whl/cu128
```

Then run:

```bash
cd ~/workspace/Zhengwei/DG-TWFD
bash scripts/experiments/create_map_branch_env.sh dgfm_map
```

The script will download:

- `torch-2.10.0+cu128-...whl`
- `torchvision-0.25.0+cu128-...whl`

into `/cache/$USER/wheels/torch-cu128` and then install locally.

### Option C. Download elsewhere, copy once, install locally

If the target server network is poor:

1. download the two wheel files on a faster machine
2. copy them to:

```bash
/cache/$USER/wheels/torch-cu128
```

3. rerun:

```bash
bash scripts/experiments/create_map_branch_env.sh dgfm_map
```

This is usually the most robust path on unstable servers.

## 5. Manual creation commands

If you do not want to use the helper script, use the following sequence.

### 5.1 Prepare mirrors and caches

```bash
mkdir -p /cache/$USER/conda_envs
mkdir -p /cache/$USER/wheels/torch-cu128

export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export HF_HOME=/cache/huggingface
export HF_HUB_CACHE=/cache/huggingface/hub
export HF_ENDPOINT=https://hf-mirror.com
export DGFM_TORCH_FIDELITY_MIRROR_PREFIX=https://githubfast.com/
```

### 5.2 Create environment

```bash
conda create -p /cache/$USER/conda_envs/dgfm_map python=3.10 -y
conda activate /cache/$USER/conda_envs/dgfm_map
```

### 5.3 Upgrade basic packaging tools

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip config set global.index-url "$PIP_INDEX_URL"
```

### 5.4 Install Torch from local wheels

If local wheels already exist:

```bash
python -m pip install \
  /cache/$USER/wheels/torch-cu128/torch-2.10.0+cu128-*.whl \
  /cache/$USER/wheels/torch-cu128/torchvision-0.25.0+cu128-*.whl
```

If they do not exist yet, download them first:

```bash
wget -c -P /cache/$USER/wheels/torch-cu128 \
  https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl
wget -c -P /cache/$USER/wheels/torch-cu128 \
  https://download.pytorch.org/whl/cu128/torchvision-0.25.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl
```

Then install locally:

```bash
python -m pip install \
  /cache/$USER/wheels/torch-cu128/torch-2.10.0+cu128-*.whl \
  /cache/$USER/wheels/torch-cu128/torchvision-0.25.0+cu128-*.whl
```

### 5.5 Install the remaining Python packages

```bash
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

## 6. Validation

Check environment versions:

```bash
python - <<'PY'
import torch, torchvision, yaml, numpy, scipy, torch_fidelity
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda_available", torch.cuda.is_available())
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("yaml", yaml.__version__)
print("torch_fidelity", torch_fidelity.__version__)
PY
```

Run the branch smoke tests:

```bash
pytest tests/test_dgfm_map_branch.py tests/test_dgfm_teacher_trajectory.py tests/test_dgfm_teacher_sampler.py tests/test_dgfm_velocity_model.py tests/test_dgfm_config.py tests/test_dgfm_overrides.py -q
```

## 7. Practical recommendation

On unstable A100 servers, use this order:

1. set `PIP_INDEX_URL`
2. set `HF_ENDPOINT=https://hf-mirror.com`
3. set `DGFM_TORCH_FIDELITY_MIRROR_PREFIX=https://githubfast.com/`
4. pre-download Torch wheels into `/cache/$USER/wheels/torch-cu128`
5. run `create_map_branch_env.sh`

That avoids repeatedly paying the slowest download cost.
