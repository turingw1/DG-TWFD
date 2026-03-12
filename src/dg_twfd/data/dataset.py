"""Datasets for teacher-generated supervision."""

from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset
import yaml

from dg_twfd.config import DGConfig
from dg_twfd.data.teacher import TeacherTrajectory


@dataclass(slots=True)
class CachedTrajectory:
    x0: Tensor
    t_grid: Tensor
    trajectory: dict[float, Tensor]


class TrajectoryPairDataset(Dataset[dict[str, Tensor]]):
    """Online teacher dataset used by the dummy Phase 1 path."""

    def __init__(
        self,
        cfg: DGConfig,
        teacher: TeacherTrajectory,
        split: str = "train",
        cached: bool | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.teacher = teacher
        self.split = split
        self.device = torch.device(device)
        self.cached = cfg.data.trajectory_cache_mode if cached is None else cached
        self.length = cfg.data.dataset_size if split == "train" else cfg.data.val_dataset_size
        self._cache: list[CachedTrajectory] = []
        if self.cached:
            self._build_cache()

    def __len__(self) -> int:
        return self.length

    def _sample_pair_times(self) -> tuple[Tensor, Tensor]:
        pair = torch.rand(2)
        t, s = torch.sort(pair, descending=True).values
        if torch.isclose(t, s):
            s = torch.clamp(t - 1e-3, min=0.0)
        return t, s

    def _sample_grid_pair(self, t_grid: Tensor) -> tuple[float, float]:
        indices = torch.randperm(len(t_grid))[:2]
        values = t_grid[indices]
        ordered = torch.sort(values, descending=True).values
        t_value = float(ordered[0].item())
        s_value = float(ordered[1].item())
        if t_value == s_value:
            s_value = max(0.0, t_value - 1.0 / max(2, self.cfg.data.time_grid_size))
        return t_value, s_value

    def _build_cache(self) -> None:
        t_grid = torch.linspace(0.0, 1.0, steps=self.cfg.data.time_grid_size, dtype=torch.float32)
        for _ in range(self.cfg.data.num_cached_trajectories):
            x0 = self.teacher.sample_x0(1, self.device)
            trajectory = self.teacher.make_trajectory(x0, t_grid)
            self._cache.append(CachedTrajectory(x0=x0, t_grid=t_grid.clone(), trajectory=trajectory))

    def sample_triplet_batch(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
    ) -> dict[str, Tensor]:
        target_device = self.device if device is None else torch.device(device)
        if self.cached and self._cache:
            return self._sample_triplet_from_cache(batch_size, target_device)
        return self._sample_triplet_on_the_fly(batch_size, target_device)

    def _sample_triplet_times(self) -> tuple[float, float, float]:
        values = torch.sort(torch.rand(3), descending=True).values
        t3, t2, t1 = [float(value.item()) for value in values]
        t2 = min(t3 - 1e-3, t2)
        t1 = min(t2 - 1e-3, t1)
        return t3, max(t2, 1e-3), max(t1, 0.0)

    def _sample_triplet_on_the_fly(
        self,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, Tensor]:
        x0 = self.teacher.sample_x0(batch_size, device)
        t3, t2, t1 = self._sample_triplet_times()
        t_grid = torch.tensor([0.0, t1, t2, t3], device=device, dtype=torch.float32)
        trajectory = self.teacher.make_trajectory(x0, torch.unique(torch.sort(t_grid).values))
        return {
            "x_t3": self._trajectory_value(trajectory, t3).to(device),
            "x_t2": self._trajectory_value(trajectory, t2).to(device),
            "x_t1": self._trajectory_value(trajectory, t1).to(device),
            "t3": torch.full((batch_size,), t3, device=device, dtype=torch.float32),
            "t2": torch.full((batch_size,), t2, device=device, dtype=torch.float32),
            "t1": torch.full((batch_size,), t1, device=device, dtype=torch.float32),
        }

    def _sample_triplet_from_cache(
        self,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, Tensor]:
        x_t3_list = []
        x_t2_list = []
        x_t1_list = []
        t3_list = []
        t2_list = []
        t1_list = []
        for _ in range(batch_size):
            cached = self._cache[torch.randint(0, len(self._cache), (1,)).item()]
            indices = torch.sort(torch.randperm(len(cached.t_grid))[:3], descending=True).values
            t3 = float(cached.t_grid[indices[0]].item())
            t2 = float(cached.t_grid[indices[1]].item())
            t1 = float(cached.t_grid[indices[2]].item())
            x_t3_list.append(cached.trajectory[t3].squeeze(0).to(device))
            x_t2_list.append(cached.trajectory[t2].squeeze(0).to(device))
            x_t1_list.append(cached.trajectory[t1].squeeze(0).to(device))
            t3_list.append(t3)
            t2_list.append(t2)
            t1_list.append(t1)
        return {
            "x_t3": torch.stack(x_t3_list, dim=0),
            "x_t2": torch.stack(x_t2_list, dim=0),
            "x_t1": torch.stack(x_t1_list, dim=0),
            "t3": torch.tensor(t3_list, device=device, dtype=torch.float32),
            "t2": torch.tensor(t2_list, device=device, dtype=torch.float32),
            "t1": torch.tensor(t1_list, device=device, dtype=torch.float32),
        }

    def _trajectory_value(self, trajectory: dict[float, Tensor], time_value: float) -> Tensor:
        closest_key = min(trajectory.keys(), key=lambda key: abs(key - time_value))
        return trajectory[closest_key]

    def _on_the_fly_item(self) -> dict[str, Tensor]:
        x0 = self.teacher.sample_x0(1, self.device)
        t_scalar, s_scalar = self._sample_pair_times()
        full_grid = torch.tensor([0.0, float(s_scalar.item()), float(t_scalar.item())], device=self.device)
        trajectory = self.teacher.make_trajectory(x0, torch.unique(torch.sort(full_grid).values))
        x_t = trajectory[float(t_scalar.item())].squeeze(0).cpu()
        x_s = trajectory[float(s_scalar.item())].squeeze(0).cpu()
        return {
            "x_t": x_t,
            "x_s": x_s,
            "t": torch.tensor(float(t_scalar.item()), dtype=torch.float32),
            "s": torch.tensor(float(s_scalar.item()), dtype=torch.float32),
        }

    def _cached_item(self) -> dict[str, Tensor]:
        cached = self._cache[torch.randint(0, len(self._cache), (1,)).item()]
        t_value, s_value = self._sample_grid_pair(cached.t_grid)
        x_t = cached.trajectory[t_value].squeeze(0).cpu()
        x_s = cached.trajectory[s_value].squeeze(0).cpu()
        return {
            "x_t": x_t,
            "x_s": x_s,
            "t": torch.tensor(t_value, dtype=torch.float32),
            "s": torch.tensor(s_value, dtype=torch.float32),
        }

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        del index
        if self.cached:
            return self._cached_item()
        return self._on_the_fly_item()


class TrajectoryShardDataset(Dataset[dict[str, Tensor]]):
    """Dataset backed by precomputed teacher trajectory shards.

    Each shard is expected to contain a list of samples, where every sample is a
    dictionary with at least:
    - `t_grid`: `[M]`
    - `x_grid`: `[M, C, H, W]` or `[1, M, C, H, W]`
    Optional fields such as `y`, `seed`, and `x0` are preserved when possible.
    """

    def __init__(self, cfg: DGConfig, split: str = "train") -> None:
        self.cfg = cfg
        self.split = split
        self.root = Path(cfg.data.trajectory_shard_dir or "")
        if not self.root.exists():
            raise FileNotFoundError(
                "trajectory_shard_dir is missing. Set data.trajectory_shard_dir to the shard root."
            )
        split_root = self.root / split
        if split_root.exists():
            self.shard_files = sorted(split_root.glob(cfg.data.trajectory_file_glob))
        else:
            self.shard_files = sorted(self.root.glob(cfg.data.trajectory_file_glob))
        if not self.shard_files:
            raise FileNotFoundError(
                f"No shard files found under {self.root} with pattern {cfg.data.trajectory_file_glob}"
            )
        self.cache_limit = 4
        self._shard_cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        self._cache_misses = 0

        self.shard_lengths = self._resolve_shard_lengths(split_root)
        if len(self.shard_lengths) != len(self.shard_files):
            raise RuntimeError("shard_lengths and shard_files mismatch")
        self.cumulative_sizes: list[int] = []
        running = 0
        for length in self.shard_lengths:
            running += int(length)
            self.cumulative_sizes.append(running)
        self.total_samples = running
        print(
            "[TrajectoryShardDataset] split=%s shards=%d samples=%d cache_limit=%d"
            % (self.split, len(self.shard_files), self.total_samples, self.cache_limit),
            flush=True,
        )

    def __len__(self) -> int:
        return self.total_samples

    def _get_sample(self, index: int) -> dict[str, Any]:
        if index < 0:
            index = self.total_samples + index
        if index < 0 or index >= self.total_samples:
            raise IndexError(f"Index out of range: {index}")
        shard_index = bisect.bisect_right(self.cumulative_sizes, index)
        shard_start = 0 if shard_index == 0 else self.cumulative_sizes[shard_index - 1]
        sample_index = index - shard_start
        shard_samples = self._get_shard_samples(shard_index)
        return shard_samples[sample_index]

    def _manifest_path(self, split_root: Path) -> Path:
        return split_root / "manifest.yaml"

    def _resolve_shard_lengths(self, split_root: Path) -> list[int]:
        manifest_path = self._manifest_path(split_root)
        if manifest_path.exists():
            try:
                with manifest_path.open("r", encoding="utf-8") as handle:
                    payload = yaml.safe_load(handle) or {}
                shard_items = payload.get("shards", [])
                count_by_file = {
                    str(item["file"]): int(item["count"])
                    for item in shard_items
                    if isinstance(item, dict) and "file" in item and "count" in item
                }
                lengths = [count_by_file.get(path.name, -1) for path in self.shard_files]
                if lengths and all(length > 0 for length in lengths):
                    return lengths
                print(
                    "[TrajectoryShardDataset] manifest exists but incomplete, fallback to shard probing: %s"
                    % manifest_path,
                    flush=True,
                )
            except Exception as exc:
                print(
                    "[TrajectoryShardDataset] failed to parse manifest %s (%s), fallback to probing"
                    % (manifest_path, exc),
                    flush=True,
                )
        return self._probe_shard_lengths()

    def _probe_shard_lengths(self) -> list[int]:
        # Fast fallback without loading every shard:
        # infer middle shard lengths from the first shard and only read first/last.
        if len(self.shard_files) == 1:
            samples = torch.load(self.shard_files[0], map_location="cpu", weights_only=False)
            if not isinstance(samples, list):
                raise TypeError(f"Shard must contain a list of samples: {self.shard_files[0]}")
            return [len(samples)]
        first_samples = torch.load(self.shard_files[0], map_location="cpu", weights_only=False)
        last_samples = torch.load(self.shard_files[-1], map_location="cpu", weights_only=False)
        if not isinstance(first_samples, list):
            raise TypeError(f"Shard must contain a list of samples: {self.shard_files[0]}")
        if not isinstance(last_samples, list):
            raise TypeError(f"Shard must contain a list of samples: {self.shard_files[-1]}")
        first_len = len(first_samples)
        last_len = len(last_samples)
        if first_len <= 0 or last_len <= 0:
            raise ValueError("Shard probing found empty shards")
        lengths = [first_len] * len(self.shard_files)
        lengths[-1] = last_len
        print(
            "[TrajectoryShardDataset] using probed shard lengths (first=%d, last=%d); "
            "for precise startup metadata, keep manifest.yaml generated by collect_teacher.py"
            % (first_len, last_len),
            flush=True,
        )
        return lengths

    def _get_shard_samples(self, shard_index: int) -> list[dict[str, Any]]:
        cached = self._shard_cache.get(shard_index)
        if cached is not None:
            self._shard_cache.move_to_end(shard_index)
            return cached
        shard_path = self.shard_files[shard_index]
        samples = torch.load(shard_path, map_location="cpu", weights_only=False)
        if not isinstance(samples, list):
            raise TypeError(f"Shard must contain a list of samples: {shard_path}")
        self._cache_misses += 1
        if self._cache_misses <= 5 or self._cache_misses % 50 == 0:
            print(
                "[TrajectoryShardDataset] cache miss=%d loaded shard=%s samples=%d"
                % (self._cache_misses, shard_path.name, len(samples)),
                flush=True,
            )
        self._shard_cache[shard_index] = samples
        self._shard_cache.move_to_end(shard_index)
        while len(self._shard_cache) > self.cache_limit:
            self._shard_cache.popitem(last=False)
        return samples

    def _sorted_trajectory(self, sample: dict[str, Any]) -> tuple[Tensor, Tensor]:
        t_grid = sample["t_grid"].float().view(-1)
        x_grid = sample["x_grid"].float()
        if x_grid.ndim == 5 and x_grid.shape[0] == 1:
            x_grid = x_grid.squeeze(0)
        if x_grid.ndim != 4:
            raise ValueError("x_grid must have shape [M, C, H, W] or [1, M, C, H, W]")
        order = torch.argsort(t_grid, descending=True)
        return t_grid[order], x_grid[order]

    def _sample_jump_delta(self, max_delta: int) -> int:
        short_hi = min(self.cfg.data.pair_short_max, max_delta)
        mid_lo = min(max_delta, self.cfg.data.pair_short_max + 1)
        mid_hi = min(self.cfg.data.pair_mid_max, max_delta)
        long_lo = min(max_delta, self.cfg.data.pair_mid_max + 1)
        long_hi = min(self.cfg.data.pair_long_max, max_delta)

        choices: list[tuple[int, int, float]] = []
        if short_hi >= 1:
            choices.append((1, short_hi, self.cfg.data.pair_short_weight))
        if mid_hi >= mid_lo:
            choices.append((mid_lo, mid_hi, self.cfg.data.pair_mid_weight))
        if long_hi >= long_lo:
            choices.append((long_lo, long_hi, self.cfg.data.pair_long_weight))
        if not choices:
            return 1

        weights = torch.tensor([weight for _, _, weight in choices], dtype=torch.float32)
        weights = weights / weights.sum()
        bucket_idx = int(torch.multinomial(weights, 1).item())
        low, high, _ = choices[bucket_idx]
        return int(torch.randint(low, high + 1, (1,)).item())

    def _sample_pair_from_sorted(self, t_grid: Tensor, x_grid: Tensor) -> dict[str, Tensor]:
        max_delta = len(t_grid) - 1
        delta = self._sample_jump_delta(max_delta)
        t_index = int(torch.randint(0, len(t_grid) - delta, (1,)).item())
        s_index = t_index + delta
        payload = {
            "x_t": x_grid[t_index],
            "x_s": x_grid[s_index],
            "t": t_grid[t_index],
            "s": t_grid[s_index],
        }
        return payload

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = self._get_sample(index)
        t_grid, x_grid = self._sorted_trajectory(sample)
        payload = self._sample_pair_from_sorted(t_grid, x_grid)
        if "y" in sample:
            payload["y"] = torch.as_tensor(sample["y"]).long()
        return payload

    def sample_triplet_batch(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
    ) -> dict[str, Tensor]:
        target_device = torch.device("cpu" if device is None else device)
        x_t3_list = []
        x_t2_list = []
        x_t1_list = []
        t3_list = []
        t2_list = []
        t1_list = []
        y_list = []
        for _ in range(batch_size):
            sample = self._get_sample(torch.randint(0, len(self), (1,)).item())
            t_grid, x_grid = self._sorted_trajectory(sample)
            gap1 = max(1, self.cfg.data.triplet_local_gap1)
            gap2 = max(gap1 + 1, self.cfg.data.triplet_local_gap2)
            max_gap = len(t_grid) - 1
            if gap2 > max_gap:
                i3, i2, i1 = 0, min(1, max_gap), max_gap
            else:
                i3 = int(torch.randint(0, len(t_grid) - gap2, (1,)).item())
                i2 = i3 + gap1
                i1 = i3 + gap2
            x_t3_list.append(x_grid[i3].to(target_device))
            x_t2_list.append(x_grid[i2].to(target_device))
            x_t1_list.append(x_grid[i1].to(target_device))
            t3_list.append(float(t_grid[i3].item()))
            t2_list.append(float(t_grid[i2].item()))
            t1_list.append(float(t_grid[i1].item()))
            if "y" in sample:
                y_list.append(int(torch.as_tensor(sample["y"]).item()))
        payload = {
            "x_t3": torch.stack(x_t3_list, dim=0),
            "x_t2": torch.stack(x_t2_list, dim=0),
            "x_t1": torch.stack(x_t1_list, dim=0),
            "t3": torch.tensor(t3_list, device=target_device, dtype=torch.float32),
            "t2": torch.tensor(t2_list, device=target_device, dtype=torch.float32),
            "t1": torch.tensor(t1_list, device=target_device, dtype=torch.float32),
        }
        if y_list:
            payload["y"] = torch.tensor(y_list, device=target_device, dtype=torch.long)
        return payload


def build_dataset(
    cfg: DGConfig,
    teacher: TeacherTrajectory,
    split: str,
) -> Dataset[dict[str, Tensor]]:
    """Build the configured dataset implementation."""

    if cfg.data.dataset_type == "teacher_online":
        return TrajectoryPairDataset(
            cfg=cfg,
            teacher=teacher,
            split=split,
            cached=cfg.data.trajectory_cache_mode,
        )
    if cfg.data.dataset_type == "trajectory_shards":
        return TrajectoryShardDataset(cfg=cfg, split=split)
    raise ValueError(f"Unsupported data.dataset_type: {cfg.data.dataset_type}")
