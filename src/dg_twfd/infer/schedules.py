"""Sampling schedules in warped and original time coordinates."""

from __future__ import annotations

import torch
from torch import Tensor

from dg_twfd.models.timewarp import TimeWarpMonotone


def build_u_schedule(
    steps: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build an equal-spaced `u` grid from 1 to 0 for `K` sampling steps."""

    if steps not in {1, 2, 4, 8, 16}:
        raise ValueError("steps must be one of {1, 2, 4, 8, 16}")
    return torch.linspace(1.0, 0.0, steps=steps + 1, device=device, dtype=dtype)


@torch.no_grad()
def build_t_schedule_from_u(
    timewarp: TimeWarpMonotone,
    u_schedule: Tensor,
) -> Tensor:
    """Map an equal-spaced `u` grid to the original time grid `t = g_phi^{-1}(u)`."""

    return timewarp.inverse(u_schedule).clamp(0.0, 1.0)
