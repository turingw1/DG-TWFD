from __future__ import annotations

import json
from pathlib import Path

import torch

from dgfm.samplers import rollout_with_map
from dgfm.schedulers import summarize_time_grid


@torch.no_grad()
def build_mode_a_time_grid(
    *,
    warp,
    step_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if step_count <= 0:
        raise ValueError(f"step_count must be positive, got {step_count}")
    if not hasattr(warp, "r_to_t"):
        raise TypeError("DGTD Mode A sampling requires a warp with r_to_t()")
    r_grid = torch.linspace(0.0, 1.0, steps=step_count + 1, device=device, dtype=dtype)
    t_grid = warp.r_to_t(r_grid).to(device=device, dtype=dtype)
    t_grid[0] = torch.tensor(0.0, device=device, dtype=dtype)
    t_grid[-1] = torch.tensor(1.0, device=device, dtype=dtype)
    if not torch.all(t_grid[1:] >= t_grid[:-1]):
        raise ValueError("DGTD Mode A generated a non-monotone time grid")
    return t_grid


@torch.no_grad()
def rollout_mode_a(
    *,
    model: torch.nn.Module,
    x_init: torch.Tensor,
    warp,
    step_count: int,
    extra: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    time_grid = build_mode_a_time_grid(
        warp=warp,
        step_count=step_count,
        device=x_init.device,
        dtype=x_init.dtype,
    )
    samples = rollout_with_map(
        model=model,
        x_init=x_init,
        step_count=step_count,
        time_grid=time_grid,
        extra=extra,
    )
    return samples, time_grid


def export_dp_schedule_stub(
    *,
    out_path: str | Path,
    step_count: int,
    note: str = "TODO: estimate interval costs C(i,j) and run DP node selection.",
) -> dict[str, object]:
    payload = {
        "mode": "dp_schedule_todo",
        "step_count": int(step_count),
        "status": "not_implemented",
        "note": note,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def summarize_mode_a(time_grid: torch.Tensor) -> dict[str, float | list[float]]:
    return summarize_time_grid(time_grid)
