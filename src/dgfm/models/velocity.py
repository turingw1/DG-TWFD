from __future__ import annotations

import torch
from torch import Tensor, nn

from dg_twfd.models.embeddings import TimeEmbedding


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _TimeResBlock(nn.Module):
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
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = self.norm1(x)
        h = h * (1.0 + scale) + shift
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return x + h


class VelocityConvNet(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, time_embed_dim: int, num_blocks: int) -> None:
        super().__init__()
        self.time_embedding = TimeEmbedding(time_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim + 1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.in_proj = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([_TimeResBlock(hidden_channels, hidden_channels) for _ in range(num_blocks)])
        self.out_norm = nn.GroupNorm(_group_count(hidden_channels), hidden_channels)
        self.out_act = nn.SiLU()
        self.out_proj = nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, t: Tensor, extra: dict | None = None) -> Tensor:
        cond = torch.cat([self.time_embedding(t), t.float().view(-1, 1)], dim=-1)
        cond = self.time_proj(cond)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(self.out_act(self.out_norm(h)))


def build_velocity_model(config: dict) -> nn.Module:
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    return VelocityConvNet(
        channels=int(dataset_cfg["channels"]),
        hidden_channels=int(model_cfg.get("hidden_channels", 128)),
        time_embed_dim=int(model_cfg.get("time_embed_dim", 64)),
        num_blocks=int(model_cfg.get("num_res_blocks", 4)),
    )
