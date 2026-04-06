from __future__ import annotations

import torch


def rollout_with_map(
    model: torch.nn.Module,
    x_init: torch.Tensor,
    step_count: int,
    time_grid: torch.Tensor | None = None,
) -> torch.Tensor:
    if step_count <= 0:
        raise ValueError(f"step_count must be positive, got {step_count}")
    if time_grid is None:
        time_grid = torch.linspace(0.0, 1.0, steps=step_count + 1, device=x_init.device, dtype=x_init.dtype)
    else:
        time_grid = time_grid.to(device=x_init.device, dtype=x_init.dtype)
        if time_grid.ndim != 1 or time_grid.shape[0] != step_count + 1:
            raise ValueError(f"time_grid must have shape ({step_count + 1},), got {tuple(time_grid.shape)}")
    x = x_init
    batch = x.shape[0]
    for idx in range(step_count):
        t = torch.full((batch,), float(time_grid[idx].item()), device=x.device, dtype=x.dtype)
        s = torch.full((batch,), float(time_grid[idx + 1].item()), device=x.device, dtype=x.dtype)
        x = model(x, t, s, extra={})
    return x
