"""DataLoader factory for DG-TWFD teacher supervision."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from dg_twfd.config import DGConfig
from dg_twfd.data.dataset import build_dataset
from dg_twfd.data.teacher import TeacherTrajectory


def _collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    collated: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = torch.tensor(values)
    return collated


def build_dataloader(
    cfg: DGConfig,
    teacher: TeacherTrajectory,
    split: str,
) -> DataLoader:
    """Build a split-aware DataLoader returning fixed-key batch dictionaries."""

    dataset = build_dataset(cfg=cfg, teacher=teacher, split=split)
    num_workers = cfg.data.num_workers
    prefetch_factor = cfg.data.prefetch_factor
    shuffle = split == "train"
    if cfg.data.dataset_type == "trajectory_shards":
        # For large shard files, multi-worker loading duplicates shard reads and
        # can become I/O-bound. Keep a single dataset instance for stable throughput.
        num_workers = 0
        prefetch_factor = None
        shuffle = False
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": cfg.data.batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": cfg.data.pin_memory,
        "drop_last": cfg.data.drop_last if split == "train" else False,
        "persistent_workers": cfg.data.persistent_workers and num_workers > 0,
        "collate_fn": _collate_fn,
    }
    if prefetch_factor is not None and num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)
