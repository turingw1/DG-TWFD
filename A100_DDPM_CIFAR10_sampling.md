# CIFAR-10 DDPM 轨迹采集策略（train / val）

## 目标
为后续的A100训练，离线构建 **按整条轨迹划分** 的 teacher 轨迹数据集。

---

## 1. 基本原则

- **按整条轨迹切分 train / val**
  - 不按单个 \((t,s)\) pair 随机切
  - 不让同一个初始噪声 seed 产生的轨迹同时出现在 train 和 val
- **轨迹样本单位**
  - 一条轨迹 = 一个初始高斯噪声 \(x_T\) 经过 teacher sampler 得到的全程状态序列
- **验证目标分两层**
  - trajectory-level：单跳误差、semigroup defect
  - generation-level：1 / 2 / 4 / 8 step 生成质量

---

## 2. teacher 设定

## 模型
- CIFAR-10 unconditional DDPM teacher

## 采样器
- **优先：DDIM deterministic sampler**
  - 轨迹稳定
  - 便于构造确定性 teacher target
  - 更适合蒸馏数据缓存
- 若已有高质量 ODE teacher，也可换成 DPM-Solver teacher，但初版不推荐，工程复杂度更高

## teacher 采样总步数
- **离线缓存步数：128**
  - 对 CIFAR-10 足够细
  - 后续可从中抽稀得到不同训练跨度
  - 对 80G A100 来说吞吐和精度比较平衡

---

## 3. 时间点设计

## 缓存时间网格
在 \([1,0]\) 上存 **129 个状态点**
\[
t_0=1 > t_1 > \cdots > t_{128}=0
\]

推荐两种方案：

### 方案 A：均匀 index 网格
- 直接用 DDPM / DDIM 的 128 个离散反推步
- 最简单，和 teacher 原始离散过程完全对齐

### 方案 B：前密后疏网格
- 高噪声端更密，低噪声端略疏
- 更适合 few-step 蒸馏
- 初版建议先不用，先保持简单

**推荐最终选择：方案 A**

---

## 4. train / val 划分

## 切分单位
- 按 **seed** 切
- 每个 seed 对应一条完整轨迹

## 数量建议
为了兼顾速度和后续训练覆盖：

- **train trajectories: 200,000**
- **val trajectories: 10,000**

这对 CIFAR-10 已经很够用，且 80G A100 离线采样压力不大。

## 切分方式
- 固定随机种子列表
- 前 200k 个 seed 作为 train
- 后 10k 个 seed 作为 val
- 永远不要混用

---

## 5. 单条轨迹保存内容

每条轨迹建议保存：

- `seed`
- `x_T` 对应初始噪声
- `t_grid` 长度 129
- `x_t` 序列，shape 为 `(129, 3, 32, 32)`

可选保存：

- `pred_x0` 或 teacher 每步预测
- 但初版没必要，节省存储即可

---

## 6. 采样算法

## 对每个 seed
1. 采样初始噪声
   \[
   x_T \sim \mathcal N(0,I)
   \]
2. 用 teacher 的 **DDIM deterministic 128-step** 从 \(t=1\) 反推到 \(t=0\)
3. 每一步保存当前状态 \(x_t\)
4. 得到完整轨迹后写盘

---

## 7. 后续训练时如何从轨迹里取样

缓存轨迹后，训练不直接重新跑 teacher，而是从轨迹中抽三类样本：

### 单跳 pair
\[
(x_t, t, s, x_s), \quad t>s
\]

### semigroup 三元组
\[
(x_t, t, s, u, x_s, x_u), \quad t>s>u
\]

### 局部 triplet
\[
(x_{t_3}, x_{t_2}, x_{t_1}), \quad t_3>t_2>t_1
\]

---


## 8. 针对 80G A100 的最高速采样建议

## batch size
- **离线采样 batch size: 1024**
- 若 teacher 较轻，且使用 fp16 / bf16，可尝试 **2048**
- 先从 1024 起测

## 精度
- **bf16 优先**
- 若 teacher 只支持 fp16，也可用 fp16
- 保存轨迹时建议转成 **fp16**

## 并行
- 单卡 A100 80G 直接离线采
- 如果多卡，按 seed shard 分片采样，最后合并

## 存储格式
- 推荐 `npz` / `pt` 分 shard 保存
- 每个 shard 保存 5k 到 10k 条轨迹
- 不要一个文件塞全部数据

---

## 9. 验证集的使用方式

val 不参与训练，只做两件事：

### 轨迹级验证
- 单跳 MSE
- semigroup defect
- 局部一致性指标

### 生成级验证
固定同一组 val seeds，测试：
- 1-step
- 2-step
- 4-step
- 8-step

看质量是否稳定，而不是只盯一个步数。

---

## 11. 最终推荐配置

## 轨迹采集配置
- teacher: CIFAR-10 DDPM
- sampler: **DDIM deterministic**
- total cached steps: **128**
- train seeds: **200,000**
- val seeds: **10,000**
- precision: **bf16**
- save dtype: **fp16**
- batch size on A100 80G: **1024**
- storage unit: **按 shard 分文件，每 shard 10k trajectories**

## 训练抽样配置
- 短跳 / 中跳 / 长跳 = **4 : 4 : 2**
- 单跳、三元组、局部 triplet 同时抽
- train / val 严格按 seed 隔离

---

## 12. 一句话总结

最稳妥、最快速的策略是：**用 DDIM deterministic teacher，从独立高斯 seed 出发离线采 128-step 全轨迹，按整条轨迹划分 200k train 与 10k val，并在训练时从缓存轨迹中混合抽取短中长跨度 pair 与 semigroup 三元组。**