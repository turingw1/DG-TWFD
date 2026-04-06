from __future__ import annotations

from pathlib import Path

import torch
import yaml

from dgfm.datasets import build_map_training_dataloaders
from dgfm.targets import build_target_builder


def _trajectory_config(tmp_path: Path) -> dict:
    shard_root = tmp_path / "teacher_traj"
    train_root = shard_root / "train"
    val_root = shard_root / "val"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)
    t_grid = torch.linspace(0.0, 1.0, steps=5, dtype=torch.float32)
    x_grid = torch.randn(5, 3, 32, 32)
    sample = [{"sample_id": 0, "t_grid": t_grid, "x_grid": x_grid.to(dtype=torch.float16)}]
    torch.save(sample, train_root / "train_00000.pt")
    torch.save(sample, val_root / "val_00000.pt")
    manifest = {"shards": [{"file": "train_00000.pt", "count": 1}]}
    (train_root / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    manifest = {"shards": [{"file": "val_00000.pt", "count": 1}]}
    (val_root / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return {
        "experiment": {"seed": 42},
        "runtime": {"device": "cpu", "amp": False},
        "train": {"objective": "explicit_map", "batch_size": 1, "num_workers": 0},
        "dataset": {"channels": 3, "image_size": 32, "name": "cifar10", "data_root": str(tmp_path)},
        "target": {
            "builder": "trajectory_shard",
            "shard_root": str(shard_root),
            "trajectory_file_glob": "*.pt",
            "cache_limit": 1,
            "pair_short_max": 2,
            "pair_mid_max": 3,
            "pair_long_max": 4,
            "pair_short_weight": 1.0,
            "pair_mid_weight": 0.0,
            "pair_long_weight": 0.0,
            "pair_endpoint_weight": 0.0,
            "high_noise_t_weight": 1.0,
            "high_noise_t_fraction": 0.5,
        },
    }


def test_trajectory_shard_target_builder_returns_ordered_pair(tmp_path: Path) -> None:
    cfg = _trajectory_config(tmp_path)
    loaders = build_map_training_dataloaders(cfg)
    batch = next(iter(loaders["train"]))
    builder = build_target_builder(cfg)
    target = builder.build_from_batch(batch, device=torch.device("cpu"), path=None)
    assert target.x_t.shape == target.x_s_target.shape == (1, 3, 32, 32)
    assert torch.all(target.s > target.t)

