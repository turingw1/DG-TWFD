from __future__ import annotations

import bisect
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import yaml


def _resolve_root(config: dict) -> Path:
    target_cfg = config.get("target", {})
    root = Path(str(target_cfg.get("shard_root", "")))
    if not root.exists():
        raise FileNotFoundError(f"Teacher trajectory shard root not found: {root}")
    return root


def _sort_sample(sample: dict[str, Any]) -> tuple[Tensor, Tensor]:
    times = torch.as_tensor(sample["t_grid"]).float().view(-1)
    states = torch.as_tensor(sample["x_grid"]).float()
    if states.ndim == 5 and states.shape[0] == 1:
        states = states.squeeze(0)
    if states.ndim != 4:
        raise ValueError(f"x_grid must have shape [M,C,H,W], got {tuple(states.shape)}")
    order = torch.argsort(times)
    return times[order], states[order]


def _curvature_from_sorted(times: Tensor, states: Tensor) -> Tensor:
    if times.numel() < 3:
        return torch.zeros(0, dtype=states.dtype)
    delta_t = torch.clamp(times[1:] - times[:-1], min=1.0e-6)
    velocity = (states[1:] - states[:-1]) / delta_t.view(-1, 1, 1, 1)
    vel_norm = velocity[:-1].flatten(1).square().mean(dim=1)
    delta_v = (velocity[1:] - velocity[:-1]).flatten(1).square().mean(dim=1)
    return delta_v / torch.clamp(vel_norm, min=1.0e-6)


class TrajectoryCacheDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, config: dict, split: str = "train") -> None:
        self.config = config
        self.target_cfg = config.get("target", {})
        self.split = split
        self.root = _resolve_root(config)
        split_root = self.root / split
        self.split_root = split_root if split_root.exists() else self.root
        pattern = str(self.target_cfg.get("trajectory_file_glob", "*.pt"))
        self.shard_files = sorted(self.split_root.glob(pattern))
        if not self.shard_files:
            raise FileNotFoundError(f"No trajectory shards found under {self.split_root}")
        self.cache_limit = max(1, int(self.target_cfg.get("cache_limit", 2)))
        self._cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        self.shard_lengths = self._resolve_shard_lengths()
        self.cumulative_sizes: list[int] = []
        running = 0
        for length in self.shard_lengths:
            running += int(length)
            self.cumulative_sizes.append(running)
        self.total_samples = running

    def _resolve_shard_lengths(self) -> list[int]:
        manifest_path = self.split_root / "manifest.yaml"
        if manifest_path.exists():
            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            counts = {
                str(item["file"]): int(item["count"])
                for item in payload.get("shards", [])
                if isinstance(item, dict) and "file" in item and "count" in item
            }
            lengths = [counts.get(path.name, -1) for path in self.shard_files]
            if lengths and all(length > 0 for length in lengths):
                return lengths
        first_samples = torch.load(self.shard_files[0], map_location="cpu", weights_only=False)
        last_samples = torch.load(self.shard_files[-1], map_location="cpu", weights_only=False)
        if not isinstance(first_samples, list) or not isinstance(last_samples, list):
            raise TypeError("Trajectory shard must contain a list of sample dicts")
        lengths = [len(first_samples)] * len(self.shard_files)
        lengths[-1] = len(last_samples)
        return lengths

    def _load_shard(self, shard_index: int) -> list[dict[str, Any]]:
        cached = self._cache.get(shard_index)
        if cached is not None:
            self._cache.move_to_end(shard_index)
            return cached
        samples = torch.load(self.shard_files[shard_index], map_location="cpu", weights_only=False)
        if not isinstance(samples, list):
            raise TypeError(f"Trajectory shard must contain a list of sample dicts: {self.shard_files[shard_index]}")
        self._cache[shard_index] = samples
        self._cache.move_to_end(shard_index)
        while len(self._cache) > self.cache_limit:
            self._cache.popitem(last=False)
        return samples

    def _get_sample(self, index: int) -> dict[str, Any]:
        if index < 0:
            index = self.total_samples + index
        shard_index = bisect.bisect_right(self.cumulative_sizes, index)
        shard_start = 0 if shard_index == 0 else self.cumulative_sizes[shard_index - 1]
        sample_index = index - shard_start
        return self._load_shard(shard_index)[sample_index]

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = self._get_sample(index)
        times, states = _sort_sample(sample)
        payload = {
            "times": times,
            "states": states,
            "curvature": _curvature_from_sorted(times, states),
        }
        if "y" in sample:
            payload["cond"] = torch.as_tensor(sample["y"]).long()
        elif "label" in sample:
            payload["cond"] = torch.as_tensor(sample["label"]).long()
        return payload


def _ensure_batch(traj: dict[str, Tensor]) -> dict[str, Tensor]:
    times = traj["times"]
    states = traj["states"]
    curvature = traj.get("curvature")
    cond = traj.get("cond")
    if times.ndim == 1:
        times = times.unsqueeze(0)
        states = states.unsqueeze(0)
        if curvature is not None:
            curvature = curvature.unsqueeze(0)
        if cond is not None and cond.ndim == 0:
            cond = cond.unsqueeze(0)
    return {
        "times": times,
        "states": states,
        "curvature": curvature,
        "cond": cond,
    }


def interpolate_state(traj: dict[str, Tensor], t: Tensor) -> Tensor:
    batch = _ensure_batch(traj)
    times = batch["times"].to(device=t.device, dtype=t.dtype)
    states = batch["states"].to(device=t.device, dtype=batch["states"].dtype)
    query = t.view(-1)
    if query.shape[0] != times.shape[0]:
        raise ValueError(f"Batch mismatch: {query.shape[0]} queries vs {times.shape[0]} trajectories")
    outputs: list[Tensor] = []
    for idx in range(times.shape[0]):
        time_row = times[idx]
        state_row = states[idx]
        q = query[idx].clamp(float(time_row[0].item()), float(time_row[-1].item()))
        upper = int(torch.searchsorted(time_row, q, right=False).item())
        if upper <= 0:
            outputs.append(state_row[0])
            continue
        if upper >= time_row.shape[0]:
            outputs.append(state_row[-1])
            continue
        lower = upper - 1
        t0 = time_row[lower]
        t1 = time_row[upper]
        alpha = (q - t0) / torch.clamp(t1 - t0, min=1.0e-6)
        outputs.append(state_row[lower] + alpha * (state_row[upper] - state_row[lower]))
    return torch.stack(outputs, dim=0)


def interpolate_curvature(traj: dict[str, Tensor], t: Tensor) -> Tensor:
    batch = _ensure_batch(traj)
    curvature = batch.get("curvature")
    if curvature is None:
        return torch.zeros_like(t)
    times = batch["times"].to(device=t.device, dtype=t.dtype)
    curvature = curvature.to(device=t.device, dtype=t.dtype)
    query = t.view(-1)
    outputs: list[Tensor] = []
    for idx in range(times.shape[0]):
        if curvature[idx].numel() == 0:
            outputs.append(torch.zeros((), device=t.device, dtype=t.dtype))
            continue
        anchors = times[idx, 1:-1]
        q = query[idx].clamp(float(anchors[0].item()), float(anchors[-1].item()))
        upper = int(torch.searchsorted(anchors, q, right=False).item())
        if upper <= 0:
            outputs.append(curvature[idx, 0])
            continue
        if upper >= anchors.shape[0]:
            outputs.append(curvature[idx, -1])
            continue
        lower = upper - 1
        a0 = anchors[lower]
        a1 = anchors[upper]
        alpha = (q - a0) / torch.clamp(a1 - a0, min=1.0e-6)
        outputs.append(curvature[idx, lower] + alpha * (curvature[idx, upper] - curvature[idx, lower]))
    return torch.stack(outputs, dim=0)


def get_teacher_pair(traj: dict[str, Tensor], t: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
    return interpolate_state(traj, t), interpolate_state(traj, u)


def build_cache_dataloaders(config: dict) -> dict[str, DataLoader]:
    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 4))
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(train_cfg.get("pin_memory", True)),
    }
    if num_workers > 0:
        common["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        common["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 4))
    train_set = TrajectoryCacheDataset(config=config, split="train")
    val_set = TrajectoryCacheDataset(config=config, split="val")
    distributed = int(__import__("os").environ.get("WORLD_SIZE", "1")) > 1
    rank = int(__import__("os").environ.get("RANK", "0"))
    world_size = int(__import__("os").environ.get("WORLD_SIZE", "1"))
    train_sampler = None
    val_sampler = None
    if distributed and world_size > 1:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    return {
        "train": DataLoader(train_set, shuffle=train_sampler is None, sampler=train_sampler, drop_last=True, **common),
        "val": DataLoader(val_set, shuffle=False, sampler=val_sampler, drop_last=False, **common),
    }
