# Git 同步工作流（本机 + GitHub + A100）

## 1. 首次绑定 GitHub 远程仓库

在本机项目目录执行：

```bash
cd ~/workspace/Zhengwei/DG-TWFD
git branch -M main
git remote add origin <你的github仓库SSH或HTTPS地址>
git push -u origin main
```

如果已存在远程地址，改用：

```bash
git remote set-url origin <新的github仓库地址>
git push -u origin main
```

## 2. 日常提交与推送

```bash
cd ~/workspace/Zhengwei/DG-TWFD
git status
git add -A
git commit -m "简明描述本次改动"
git push
```

## 3. A100 服务器同步最新代码

首次：

```bash
git clone <你的github仓库地址> DG-TWFD
cd DG-TWFD
git checkout main
```

后续每次更新：

```bash
cd /path/to/DG-TWFD
git pull --ff-only
```

## 4. 建议分支策略

- `main`：可稳定训练/采样的主分支
- `exp/*`：实验分支，例如 `exp/teacher-edm`
- 合并前先本地通过关键测试：
  - `pytest tests/test_data.py -q`
  - `pytest tests/test_models.py -q`
  - `pytest tests/test_loss.py -q`

## 5. 当前已配置的忽略策略

项目已通过 `.gitignore` 忽略以下高频产物，避免污染提交：

- Python cache（`__pycache__/`、`*.pyc`）
- 训练与推理产物（`checkpoints*/`、`artifacts/`、`vis_out/`）
- 大体积 teacher 数据（`data/teacher_shards/`）
- 工具缓存（`.pytest_cache/` 等）

如果你需要长期保存大模型或大数据，建议后续接入 Git LFS 或外部对象存储，而不是直接提交到 Git 仓库。
