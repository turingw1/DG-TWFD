# Prompt Phase 4: 训练验证 Pipeline 与日志分析 

你将基于 Phase 1-3 的工程，实现完整训练验证 pipeline。你必须在本阶段产出“完整可运行代码”，并在完成后立刻停止生成，等待我 Code Review 指令再进入下一阶段。

## 0. 总体要求
1. 训练框架：实现 `Trainer` 类，支持：
   - AMP 混合精度
   - 梯度累积
   - gradient clipping
   - checkpoint 保存与恢复（模型、优化器、scheduler、scaler、全局 step）
2. 日志：集成 wandb（若无则自动 fallback 到 TensorBoard），必须记录：
   - L_match, L_def, L_warp, L_boundary, total_loss
   - defect 的 EMA 曲线
   - 学到的 time-warp 映射的可视化数据（例如导出 t->u 的离散表作为 artifact）
3. Profile 自适应：
   - debug_4060：自动开启显存保护策略（更小 batch、AMP、accumulation、禁用多余缓存）
   - train_a100：高吞吐设置（更大 batch、多 workers、pin_memory、prefetch）
4. 测试：提供 `train.py --mode debug_4060` 在 Dummy 数据上跑通 5 个 epoch，并在终端打印 loss 下降趋势。

## 1. 新增文件结构（必须创建）
- `src/dg_twfd/engine/trainer.py`
- `src/dg_twfd/engine/checkpoint.py`
- `src/dg_twfd/engine/amp.py`
- `src/dg_twfd/engine/loops.py`（可选，若 trainer 太大可拆）
- `src/dg_twfd/cli/train.py` 或项目根目录 `train.py`
- `scripts/run_debug.sh`（可选）
- 更新 `README.md` 增加训练命令与调参说明

## 2. Trainer 关键设计要求
- 组件注入：Trainer 构造时接收 teacher、models（student, timewarp, boundary）、losses、scheduler、dataloaders
- 两阶段优化（可选但推荐）：
  - student θ 用 match + defect + boundary 更新
  - timewarp φ 用 warp + defect（或 warp 主导）更新
  - 若用交替优化，请写清楚每 N step 更新哪个参数组，并在注释解释避免不稳定的原因
- 在线 defect-adaptive schedule：
  - 每 step 用当前 batch 的 defect 更新 scheduler
  - 下一个 batch 的 t 采样策略可通过 dataset hook 或 trainer 控制（实现一个简单可用版本即可）

## 3. Debug 运行要求
`train.py --mode debug_4060`：
- 使用 DummyTeacherTrajectory
- dataset 用小规模（比如 256 样本）
- 跑 5 epoch
- 每个 epoch 打印平均 loss
- 若 CUDA 可用，定期打印 peak memory
- wandb 在 debug 模式可用 offline 或 disabled，但代码必须可运行

## 4. Checkpoint
实现：
- `save_checkpoint(path, state)`
- `load_checkpoint(path)`
并支持恢复训练（继续 global_step）。

## 5. 输出要求
- 给出所有新增/修改文件完整代码
- 训练脚本必须可直接运行
- 完成本阶段后必须停止生成并等待下一阶段指令
