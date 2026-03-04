## 1) Teacher 模型评估与选择建议

你现在的框架非常适合做“多 teacher 体系”的系统实验，我建议按三个层级选 teacher，并且这些 teacher 本身也作为 baseline 参与对比。

### A. 最快打通链路的 teacher（Debug 与单卡验证）

**Teacher 1: DDPM CIFAR-10 32x32（diffusers 直接可用）**
优点是下载和推理最省事，适合先把“轨迹采集 数据管线 time-warp defect 训练 sampling”全跑通。
模型来源：HuggingFace `google/ddpm-cifar10-32`。([Hugging Face][1])

下载与使用方式（你可以直接做成 `teacher_type: diffusers_ddpm`）：

```python
from diffusers import DDPMPipeline
import torch

pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to("cuda")
pipe.unet.eval()
```

对接建议
在 `src/dg_twfd/data/teacher.py` 里做一个适配器 `DiffusersDDPMTeacher`，把 diffusers 的 UNet 和 scheduler 包装成你统一的 teacher API，支持：

* `forward_eps(x_t, t, cond=None) -> eps_pred`
* `sample_trajectory(batch, steps, return_intermediates=True) -> {times, states, x0, eps0/seed}`

你后面所有 teacher 都对齐这套 API，训练与采样代码就不需要分支。

---

### B. ImageNet 64 的主 teacher（与你 review 叙事最贴合）

**Teacher 2: OpenAI guided-diffusion（ImageNet 64 128 256 全套 checkpoint）**
优点是经典、可控、审稿人熟悉，且官方 README 给出了 checkpoint 下载链接。([GitHub][2])
这套 teacher 很适合作为“高曲率 teacher”的代表，用来验证 time-warp 和 defect 学习的必要性。

下载方式
`openai/guided-diffusion` README 里列了 64x64, 128x128, 256x256 的 diffusion 与 classifier checkpoint 链接。([GitHub][2])

对接建议
做一个 `GuidedDiffusionTeacher`，内部沿用它的 UNet forward 形式与时间步编码方式。你只需要保证 teacher adapter 能输出 eps 或 v 的预测，并且能在给定 timestep grid 下执行一次 solver rollout，把每个中间 `x_t` 存出来。

**Teacher 3: EDM 系列 teacher（ImageNet-64 高质量参考，训练与 distill 社区常用）**
你有两条路都很稳：

1. 直接用 NVlabs/edm 的基线网络与采样脚本，它明确提供了 ImageNet-64 条件模型 `baseline-imagenet-64x64-cond-adm.pkl` 等权重信息。([GitHub][3])
2. 用 OpenAI Consistency Models 提供的 ImageNet-64 EDM teacher checkpoint，后续还可以对比 CD/CT 的快速模型。其预训练模型发布信息在官方 repo 与 pypi 页面都能查到。([GitHub][4])

如果你现在优先“快速接入”，建议先接入 OpenAI Consistency Models 这一套，因为它同时给你 teacher 和若干 1-step, few-step baseline 模型（CT, CD），对比组天然齐全。([GitHub][4])

---

### C. ImageNet 256 的强 teacher（更接近 SOTA 论文语境）

**Teacher 4: DiT-XL/2（ImageNet 256，官方权重直链）**
优点是公认的强 teacher，并且官方提供了 256 分辨率权重直链。([Hugging Face][5])
这能作为“强 teacher”去验证 DG-TWFD 在更强 teacher 下是否仍能提高跨步稳定性。

下载方式（权重直链在 repo 文档里给出）：([Hugging Face][5])
你可以在 adapter 里写成自动下载缓存，或要求用户先下载放到 `checkpoints/dit/`。

对接建议
DiT 的采样本质上也是用模型预测噪声或速度项配合 scheduler 更新 `x_t`，你同样包装成 `DiTTeacher.forward_eps` 与 `sample_trajectory` 即可。
你现在已经有 `models/embeddings.py`，如果你打算做 class-conditional，teacher adapter 里把 `y` 透传进去，然后 student 也用 label embedding 保持条件一致。

---

### 建议你最终保留的 teacher/baseline 组合

最少但足够写论文的组合：

* CIFAR-10: `google/ddpm-cifar10-32` 作为 debug teacher。([Hugging Face][1])
* ImageNet-64: `guided-diffusion 64x64 diffusion` 作为高曲率 teacher。([GitHub][2])
* ImageNet-64: `EDM teacher` 作为高质量 teacher。([GitHub][3])
* ImageNet-256: `DiT-XL/2 256` 作为强 teacher。([Hugging Face][5])
* 额外 baseline（不做 teacher）：OpenAI 的 CT/CD ImageNet-64 diffusers pipeline 直接跑 1-step 2-step 作对比。([Hugging Face][6])

---

## 2) Teacher 轨迹采集的具体实现方式

你的 `src/dg_twfd/data/teacher.py` 最好定义一个抽象基类，所有 teacher 统一输出“离散时间网格上的轨迹”。这样你在 `losses/defect.py` 和 `losses/warp.py` 中就完全不关心 teacher 细节。

### 2.1 统一数据结构与存储格式

建议每条样本存这些字段：

* `seed` 或 `eps`：用于复现初始噪声
* `y`：类别标签，class-conditional 才需要
* `t_grid`：长度 M 的时间数组，单调递减
* `x_grid`：形状 `[M, C, H, W]` 的轨迹状态
* `x0`：终点图像，也可用 `x_grid[-1]` 代替

建议写成 shard：

* 每个 shard 存 N 条样本，`.pt` 或者 WebDataset tar 都行
* `dataset.py` 读取 shard 后随机抽 `(t,s,u)` 三元组来训练 defect loss

### 2.2 轨迹采集算法（teacher rollout）

你只需要一套通用流程，solver 可以替换：

1. 初始化
   采 `eps ~ N(0,I)`，构造 `x_T`。在 VP/DDPM 体系是 `x_T=eps`，在 EDM 噪声参数体系通常是 `x = sigma_max * eps`。你在 adapter 内部处理即可。

2. 选时间网格
   为了复现实验与对齐 schedule，建议把 `t_grid` 作为配置项，例如：

* `grid_type: teacher_native` 用 teacher scheduler 的离散步
* `grid_type: edm_karras` 用 Karras 风格 sigma 或 t 采样
* `M=64` 做高密轨迹，训练时再抽子段

3. 逐步更新并存储
   每一步计算 teacher 的预测量并更新 `x`，把每个 `x` 保存到 `x_grid`。

你现在有 `scripts/profile_infer.py`，可以加一个 `scripts/collect_teacher.py`，专门离线生成 shards，训练时读取 shards，这样速度与稳定性最好。

---

## 3) 时间重参数化 time-warp 的实现建议（你框架里的 `models/timewarp.py`）

你要的核心性质只有三个：

* 单调
* 可逆或可近似求逆
* 便宜，训练时不要引入 JVP

最简单可控的工程实现方案是“分段线性单调映射”。

### 3.1 参数化

设有 K 个 segment，学习正增量：

* `a in R^K` 为可训练参数
* `w = softplus(a) + eps`
* `p = w / sum(w)`，每段长度为 `p_k`

定义累计节点：

* `u_0=0`
* `u_k = sum_{i<k} p_i`
* `u_K=1`

则 `g_phi` 是把原始时间 `t` 映射到 `u` 的分段线性函数。
反函数 `g^{-1}` 也可用 `searchsorted` 在节点上做分段线性回推。

### 3.2 为什么它适合你现在的离线轨迹

你训练时实际上只需要把“在 u 上均匀取点”映射回“t 上非均匀取点”，从而把更多步长预算给高 defect 或高曲率区间。分段线性足够表达这种“步长重分配”，实现又非常快。

---

## 4) 数据集选择与接入方案

你问的是“主流数据集评估 选择 下载方式 接入方式 分类实验方案”。我按你当前 DG-TWFD 的叙事给一条从易到难的路径。

### A. CIFAR-10（32x32，最快验证）

* 下载：torchvision 支持 `download=True` 自动下载。([PyTorch Docs][7])
* 接入：`dataset.py` 里用 torchvision dataset，输出 `x in [-1,1]`。
* 实验价值：快速验证 step-generalization 曲线和 defect 下降曲线，调好训练超参后再去 ImageNet。

### B. ImageNet ILSVRC2012（64 与 256 两个分辨率都从这来）

* 官方获取方式：需要登录并同意 Terms of Access。([image-net.org][8])
* HuggingFace 也提供了镜像式的 dataset 接入入口，同样要求同意 ImageNet Terms。([Hugging Face][9])

接入策略建议

* 你在 `dataset.py` 做一个 `ImageNetFolderDataset`，读取已经解压并按类目录组织的 `train/` 和 `val/`。
* 预处理：

  * ImageNet-64：`center crop` 后 `resize 64`，训练时可加轻量随机翻转
  * ImageNet-256：`center crop` 后 `resize 256`

分类实验方案建议

* 版本 1：class-conditional ImageNet-64，对齐 DMD 等一类工作的常用设定。DMD 的论文报告就包含 class-conditional ImageNet-64 训练。([arXiv][10])
* 版本 2：class-conditional ImageNet-256，teacher 用 DiT-XL/2 256，验证你方法的可扩展性。([Hugging Face][5])
* 如果你担心算力，先做 ImageNet-64，把方法讲清楚再扩展到 256。

### C. LSUN Bedroom 256（无条件，经典生成数据集）

下载方式：官方给了 python 脚本下载 lmdb。([GitHub][11])
torchvision 也提供 LSUN dataset 读取接口，提示需要 `lmdb`。([PyTorch Docs][12])
优势是无条件，模型对接更简单。

### D. FFHQ（人脸，高分辨率，生成任务常用）

NVlabs 提供了官方 repo 与说明。([GitHub][13])
这类数据集很适合用来做“边界稳定化是否浪费容量”的对照实验，因为结构单一，分布更集中。

---

## 5) 你现在立刻能落地的对接清单

你不需要一次性把所有 teacher 都接上，推荐按顺序推进，每一步都能产出可用的 ablation。

1. 先接入 `google/ddpm-cifar10-32`，跑通 `collect_teacher.py -> shards -> train.py -> sample.py` 的完整闭环。([Hugging Face][1])
2. 接入 `openai/guided-diffusion` 的 64x64 diffusion checkpoint，做 ImageNet-64 的高曲率 teacher。([GitHub][2])
3. 接入 DiT-XL/2 256，做 ImageNet-256 强 teacher。([Hugging Face][5])
4. 加入 OpenAI CT/CD ImageNet-64 作为 1-step, few-step baseline，直接调用 diffusers pipeline。([Hugging Face][6])


[1]: https://huggingface.co/google/ddpm-cifar10-32?utm_source=chatgpt.com "google/ddpm-cifar10-32"
[2]: https://github.com/openai/guided-diffusion?utm_source=chatgpt.com "openai/guided-diffusion"
[3]: https://github.com/NVlabs/edm?utm_source=chatgpt.com "NVlabs/edm: Elucidating the Design Space of Diffusion- ..."
[4]: https://github.com/openai/consistency_models?utm_source=chatgpt.com "openai/consistency_models: Official repo for consistency ..."
[5]: https://huggingface.co/Dragunflie-420/Delphi-Oracle-DiT/commit/a6257584aefb84cae3a26615ced3da58524382e7?utm_source=chatgpt.com "Update README.md · Dragunflie-420/Delphi-Oracle-DiT at a625758"
[6]: https://huggingface.co/openai/diffusers-ct_imagenet64/blame/e0aaf194b4862a05cd45d8ec5d54d94e4cc2bae0/README.md?utm_source=chatgpt.com "README.md · openai/diffusers-ct_imagenet64 at ..."
[7]: https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html?utm_source=chatgpt.com "CIFAR10 — Torchvision main documentation"
[8]: https://www.image-net.org/download.php?utm_source=chatgpt.com "Download"
[9]: https://huggingface.co/datasets/ILSVRC/imagenet-1k?utm_source=chatgpt.com "ILSVRC/imagenet-1k · Datasets at Hugging Face"
[10]: https://arxiv.org/html/2311.18828v4?utm_source=chatgpt.com "One-step Diffusion with Distribution Matching Distillation"
[11]: https://github.com/fyu/lsun?utm_source=chatgpt.com "fyu/lsun: LSUN Dataset Documentation and Demo Code"
[12]: https://docs.pytorch.org/vision/0.20/generated/torchvision.datasets.LSUN.html?utm_source=chatgpt.com "LSUN — Torchvision 0.20 documentation"
[13]: https://github.com/NVlabs/ffhq-dataset?utm_source=chatgpt.com "NVlabs/ffhq-dataset: Flickr-Faces-HQ Dataset (FFHQ)"
