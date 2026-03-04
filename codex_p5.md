# Prompt Phase 5: 推理采样与架构可视化 

你将基于 Phase 1-4 的工程，实现 DG-TWFD 推理采样与最终文档收尾。你必须在本阶段产出“完整可运行代码”，并在完成后立刻停止生成，等待我 Code Review 指令再进入下一阶段。

## 0. 总体要求
1. 推理函数：实现 `sample_dg_twfd`：
   - 支持步数 K in {1,2,4,8,16}
   - 使用 u 坐标等间隔采样 \(u_i\)，并通过 \(t_i=g_\phi^{-1}(u_i)\) 得到时间表
   - 逐步复合 \(x_{t_{i+1}} = M_\theta(t_i,t_{i+1},x_{t_i})\)
   - 边界校正器 Bψ 可选启用
2. 分析脚本：输出生成延迟、NFE、峰值显存、并导出 defect 曲线与 warp 映射表。
3. README 完善：写清楚云端 A100 训练步骤、推荐 profile、wandb 配置、复现说明。

## 1. 新增文件结构（必须创建）
- `src/dg_twfd/infer/sampler.py`
- `src/dg_twfd/infer/schedules.py`（u-grid 与 t-grid 构造）
- `src/dg_twfd/cli/sample.py` 或根目录 `sample.py`
- `scripts/profile_infer.py`（可选，用于速度显存 profiling）
- 更新 `README.md` 增加推理与部署部分

## 2. sample_dg_twfd 细节
接口建议：
`sample_dg_twfd(models, timewarp, boundary, noise, steps: int, device, enable_boundary: bool) -> x0`
要求：
- noise 默认从 N(0,I) 采样
- 返回 x0（或最后时刻的图像张量）
- 支持 batch
- 全程 `torch.no_grad()` 与 AMP autocast（可配置）

## 3. 评估与可视化
实现一个简单的 profiler：
- 统计每次采样耗时
- 统计峰值显存
- 输出每个 steps 的耗时与显存表到控制台
若没有真实 Inception 网络与 FID，本阶段用占位评估即可，但要把接口留好，注释写明如何替换成真实 FID 评估。

## 4. README 最终化（必须）
必须包含：
- 最终目录树
- pipeline 图（更新为包含 timewarp、defect-adaptive schedule、boundary gate）
- 本地 debug（4060）一键运行命令：训练与采样
- 云端 A100 训练命令（train_a100 profile）与建议参数
- 常见问题 FAQ：OOM 处理、AMP、grad accumulation、恢复训练
- 说明哪些部分是 Dummy teacher 占位，如何接入真实 diffusion teacher（接口与 TODO）

## 5. 输出要求
- 给出所有新增/修改文件完整代码
- 给出可执行示例命令
- 完成本阶段后必须停止生成并等待我的最终指令