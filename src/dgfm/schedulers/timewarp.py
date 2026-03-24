from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


DEFAULT_STRATEGIES = (
    "uniform",
    "source_dense_power2",
    "data_dense_power2",
    "random_dirichlet",
)


def list_timewarp_strategies() -> tuple[str, ...]:
    return DEFAULT_STRATEGIES


def _validate_time_grid(time_grid: torch.Tensor, step_count: int) -> torch.Tensor:
    if time_grid.ndim != 1:
        raise ValueError(f"time_grid must be 1D, got shape={tuple(time_grid.shape)}")
    if time_grid.shape[0] != step_count + 1:
        raise ValueError(f"time_grid length must be {step_count + 1}, got {time_grid.shape[0]}")
    if not torch.all(time_grid[1:] >= time_grid[:-1]):
        raise ValueError("time_grid must be monotone non-decreasing")
    if abs(float(time_grid[0].item())) > 1.0e-8 or abs(float(time_grid[-1].item()) - 1.0) > 1.0e-8:
        raise ValueError("time_grid must start at 0 and end at 1")
    return time_grid


def _uniform_grid(step_count: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, steps=step_count + 1, device=device, dtype=dtype)


def _source_dense_power_grid(step_count: int, *, gamma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    base = _uniform_grid(step_count, device=device, dtype=dtype)
    return base.pow(gamma)


def _data_dense_power_grid(step_count: int, *, gamma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    base = _uniform_grid(step_count, device=device, dtype=dtype)
    return 1.0 - (1.0 - base).pow(gamma)


def _random_dirichlet_grid(
    step_count: int,
    *,
    concentration: float,
    random_seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if concentration <= 0.0:
        raise ValueError(f"concentration must be positive, got {concentration}")
    rng = np.random.default_rng(random_seed)
    increments_np = rng.dirichlet(np.full(step_count, float(concentration), dtype=np.float64))
    increments = torch.tensor(increments_np, device=device, dtype=dtype)
    time_grid = torch.cat(
        [
            torch.zeros(1, device=device, dtype=dtype),
            torch.cumsum(increments, dim=0),
        ],
        dim=0,
    )
    time_grid[-1] = torch.tensor(1.0, device=device, dtype=dtype)
    return time_grid


def build_time_grid(
    step_count: int,
    strategy: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    power_gamma: float = 2.0,
    random_concentration: float = 1.0,
    random_seed: int = 123,
) -> torch.Tensor:
    if step_count <= 0:
        raise ValueError(f"step_count must be positive, got {step_count}")

    if strategy == "uniform":
        return _validate_time_grid(_uniform_grid(step_count, device=device, dtype=dtype), step_count)
    if strategy == "source_dense_power2":
        return _validate_time_grid(
            _source_dense_power_grid(step_count, gamma=power_gamma, device=device, dtype=dtype),
            step_count,
        )
    if strategy == "data_dense_power2":
        return _validate_time_grid(
            _data_dense_power_grid(step_count, gamma=power_gamma, device=device, dtype=dtype),
            step_count,
        )
    if strategy == "random_dirichlet":
        return _validate_time_grid(
            _random_dirichlet_grid(
                step_count,
                concentration=random_concentration,
                random_seed=random_seed,
                device=device,
                dtype=dtype,
            ),
            step_count,
        )
    raise ValueError(f"Unsupported timewarp strategy: {strategy}")
