# map_branch environment

This branch no longer assumes that a pre-existing `consistency` environment is
available on the server.

## Reference environment

The current working local environment used for this branch is:

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

For the current `map_branch v2` experiment, the minimum practical package set is:

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
- `matplotlib`
- `pillow`
- `pytest`

## Recommended creation method

Use the branch-local setup script:

```bash
bash scripts/experiments/create_map_branch_env.sh dgfm_map
conda activate /cache/$USER/conda_envs/dgfm_map
```

By default, the script creates the environment under:

```bash
/cache/$USER/conda_envs/<env_name>
```

The experiment activation script also defaults the HuggingFace cache to:

```bash
/cache/huggingface
```

This is important for offline teacher rollout, because the current teacher
configuration uses:

- `teacher.local_files_only=true`

If you want a different writable cache root:

```bash
bash scripts/experiments/create_map_branch_env.sh dgfm_map /cache/custom_user/conda_envs
conda activate /cache/custom_user/conda_envs/dgfm_map
```

If your server has a different CUDA/PyTorch compatibility requirement, edit the
Torch install line in that script first.

## Manual creation commands

```bash
mkdir -p /cache/$USER/conda_envs
conda create -p /cache/$USER/conda_envs/dgfm_map python=3.10 -y
conda activate /cache/$USER/conda_envs/dgfm_map

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  PyYAML==6.0.3 \
  numpy==2.2.3 \
  scipy==1.15.3 \
  torch-fidelity==0.4.0 \
  'diffusers>=0.30' \
  'transformers>=4.40' \
  'accelerate>=0.30' \
  'safetensors>=0.4' \
  matplotlib \
  pillow \
  pytest

python -m pip install -e .
```

## Validation

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

Then run the branch smoke tests:

```bash
pytest tests/test_dgfm_map_branch.py tests/test_dgfm_teacher_trajectory.py tests/test_dgfm_velocity_model.py tests/test_dgfm_config.py tests/test_dgfm_overrides.py -q
```
