"""Flow student network `M_theta(t, s, x_t)`."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from dg_twfd.models.embeddings import PairTimeConditioner


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ConditionedResBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, channels * 2)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale_shift = self.cond_proj(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        hidden = self.norm1(x)
        hidden = hidden * (1.0 + scale) + shift
        hidden = self.act(hidden)
        hidden = self.conv1(hidden)
        hidden = self.norm2(hidden)
        hidden = self.act(hidden)
        hidden = self.conv2(hidden)
        return x + hidden


class FlowStudent(nn.Module):
    """Student map `M_theta(t, s, x_t) -> x_s`.

    This implementation uses a compact residual convolutional backbone. The
    conditioner encodes `(t, s, delta=t-s)` so the model can predict either the
    target state directly or a residual update relative to `x_t`.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        time_embed_dim: int,
        cond_dim: int,
        num_blocks: int = 4,
        predict_residual: bool = True,
    ) -> None:
        super().__init__()
        self.predict_residual = predict_residual
        self.conditioner = PairTimeConditioner(time_embed_dim, cond_dim)
        self.in_proj = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [_ConditionedResBlock(hidden_channels, cond_dim) for _ in range(num_blocks)]
        )
        self.out_norm = nn.GroupNorm(_group_count(hidden_channels), hidden_channels)
        self.out_act = nn.SiLU()
        self.out_proj = nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1)

    def forward(self, x_t: Tensor, t: Tensor, s: Tensor) -> Tensor:
        cond = self.conditioner(t, s)
        hidden = self.in_proj(x_t)
        for block in self.blocks:
            hidden = block(hidden, cond)
        predicted = self.out_proj(self.out_act(self.out_norm(hidden)))
        if self.predict_residual:
            return x_t + predicted
        return predicted
