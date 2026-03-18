"""Patch-transformer student backbone inspired by DiT/pMF design choices."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from dg_twfd.models.embeddings import PairTimeConditioner


class _PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, hidden_size: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class _DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.modulation = nn.Linear(cond_dim, hidden_size * 4)
        self.attn_gate = nn.Parameter(torch.zeros(hidden_size))
        self.mlp_gate = nn.Parameter(torch.zeros(hidden_size))

    def _modulate(self, x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
        return x * (1.0 + scale[:, None, :]) + shift[:, None, :]

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale1, shift1, scale2, shift2 = self.modulation(cond).chunk(4, dim=-1)
        attn_in = self._modulate(self.norm1(x), scale1, shift1)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out * self.attn_gate
        mlp_in = self._modulate(self.norm2(x), scale2, shift2)
        x = x + self.mlp(mlp_in) * self.mlp_gate
        return x


class PatchDiTStudent(nn.Module):
    """Student map `M_theta(t, s, x_t)` with patch tokenization + self-attention."""

    def __init__(
        self,
        image_size: int,
        channels: int,
        hidden_size: int,
        time_embed_dim: int,
        cond_dim: int,
        num_blocks: int,
        num_heads: int,
        patch_size: int,
        mlp_ratio: float = 4.0,
        predict_residual: bool = True,
        residual_scale_by_delta: bool = True,
        residual_tanh_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.predict_residual = predict_residual
        self.residual_scale_by_delta = residual_scale_by_delta
        self.residual_tanh_scale = residual_tanh_scale
        self.channels = channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        self.conditioner = PairTimeConditioner(time_embed_dim, cond_dim)
        self.patch_embed = _PatchEmbed(image_size, patch_size, channels, hidden_size)
        self.cond_proj = nn.Linear(cond_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, hidden_size))
        self.blocks = nn.ModuleList(
            [_DiTBlock(hidden_size, num_heads, mlp_ratio, hidden_size) for _ in range(num_blocks)]
        )
        self.out_norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, patch_size * patch_size * channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.normal_(self.pos_embed, std=0.02)

    def _unpatchify(self, patch_tokens: Tensor) -> Tensor:
        batch = patch_tokens.shape[0]
        grid = self.image_size // self.patch_size
        x = patch_tokens.view(batch, grid, grid, self.patch_size, self.patch_size, self.channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(batch, self.channels, self.image_size, self.image_size)

    def forward(self, x_t: Tensor, t: Tensor, s: Tensor) -> Tensor:
        cond = self.cond_proj(self.conditioner(t, s))
        hidden = self.patch_embed(x_t) + self.pos_embed
        for block in self.blocks:
            hidden = block(hidden, cond)
        predicted = self._unpatchify(self.out_proj(self.out_norm(hidden)))
        if self.predict_residual:
            if self.residual_tanh_scale > 0.0:
                predicted = torch.tanh(predicted / self.residual_tanh_scale) * self.residual_tanh_scale
            if self.residual_scale_by_delta:
                delta = torch.clamp(t - s, min=0.0).view(-1, 1, 1, 1)
                predicted = predicted * delta
            return x_t + predicted
        return predicted
