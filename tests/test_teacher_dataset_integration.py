from __future__ import annotations

import torch

from dg_twfd.config import load_config
from dg_twfd.data import TrajectoryShardDataset, build_teacher


def test_build_dummy_teacher_from_config() -> None:
    cfg = load_config("debug_4060")
    teacher = build_teacher(cfg)
    x = teacher.sample_x0(cfg.data.batch_size, "cpu")
    assert x.shape == (
        cfg.data.batch_size,
        cfg.data.channels,
        cfg.data.image_size,
        cfg.data.image_size,
    )


def test_trajectory_shard_dataset_reads_pairs_and_triplets(tmp_path) -> None:
    cfg = load_config(
        "debug_4060",
        overrides=[
            "data.dataset_type='trajectory_shards'",
            f"data.trajectory_shard_dir='{tmp_path}'",
        ],
    )
    split_dir = tmp_path / "train"
    split_dir.mkdir(parents=True, exist_ok=True)
    t_grid = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    x_grid = torch.randn(3, cfg.data.channels, cfg.data.image_size, cfg.data.image_size)
    torch.save(
        [
            {
                "t_grid": t_grid,
                "x_grid": x_grid,
                "x0": x_grid[-1],
                "y": torch.tensor(3),
            }
        ],
        split_dir / "train_00000.pt",
    )

    dataset = TrajectoryShardDataset(cfg, split="train")
    item = dataset[0]
    assert item["x_t"].shape == item["x_s"].shape
    assert item["t"] > item["s"]
    assert item["y"].item() == 3

    triplet = dataset.sample_triplet_batch(batch_size=1, device="cpu")
    assert triplet["x_t3"].shape == (
        1,
        cfg.data.channels,
        cfg.data.image_size,
        cfg.data.image_size,
    )
    assert torch.all(triplet["t3"] > triplet["t2"])
    assert torch.all(triplet["t2"] > triplet["t1"])
