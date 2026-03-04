"""Boundary correction network for the noisy endpoint."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.block(x))


class BoundaryCorrector(nn.Module):
    """Gateable boundary corrector `B_psi(x_{t_max})`.

    The module targets the unstable first jump near the high-noise endpoint.
    During training, a gate weight can anneal the residual `x + w * f(x)` so
    the corrector only activates when needed and does not waste capacity.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 32,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[_ConvBlock(hidden_channels) for _ in range(num_blocks)])
        self.out_proj = nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: Tensor,
        enabled: bool = True,
        gate_weight: float | Tensor = 1.0,
    ) -> Tensor:
        if not enabled:
            return x

        if not torch.is_tensor(gate_weight):
            gate_weight = torch.tensor(gate_weight, device=x.device, dtype=x.dtype)
        gate_weight = gate_weight.to(device=x.device, dtype=x.dtype)
        while gate_weight.ndim < x.ndim:
            gate_weight = gate_weight.unsqueeze(-1)

        residual = self.out_proj(self.blocks(self.in_proj(x)))
        return x + gate_weight * residual
