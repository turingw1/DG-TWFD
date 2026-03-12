# Changelog

## 2026-03-04

- 扩展配置系统，加入 `data.dataset_type`、`data.trajectory_shard_dir`、`teacher.teacher_type`、`teacher.pretrained_model_name_or_path` 等 teacher / 数据集对接参数。
- 重构 [teacher.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/data/teacher.py)，加入 `build_teacher()` 工厂和 `DiffusersDDPMTeacher` 适配器骨架。
- 重构 [dataset.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/data/dataset.py)，新增 `TrajectoryShardDataset` 与 `build_dataset()`。
- 更新 [dataloader.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/data/dataloader.py)，使其支持 shard 数据集与附加字段。
- 更新 [trainer.py](~/workspace/Zhengwei/DG-TWFD/src/dg_twfd/engine/trainer.py)，取消对 `TrajectoryPairDataset` 的硬编码依赖。
- 更新 [train.py](~/workspace/Zhengwei/DG-TWFD/train.py)，改为通过配置构建 teacher。
- 新增 [collect_teacher.py](~/workspace/Zhengwei/DG-TWFD/scripts/collect_teacher.py)，用于离线采集 teacher 轨迹 shard。
- 新增 [test_teacher_dataset_integration.py](~/workspace/Zhengwei/DG-TWFD/tests/test_teacher_dataset_integration.py)，验证 teacher 构建和 shard 数据集读取。
- 新增 [TEACHER_DATASET_INTEGRATION_GUIDE.md](~/workspace/Zhengwei/DG-TWFD/TEACHER_DATASET_INTEGRATION_GUIDE.md)，明确哪些步骤已完成、哪些步骤需要手动执行。
- 优化 `DiffusersDDPMTeacher` 的 rollout 逻辑，改为基于 scheduler inference timesteps 的 batched 推进，避免逐样本逐步的极慢采集路径。
