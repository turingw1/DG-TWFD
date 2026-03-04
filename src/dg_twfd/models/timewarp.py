"""Monotone time-warp module for DG-TWFD."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TimeWarpMonotone(nn.Module):
    """Monotone time-warp `u = g_phi(t)` on `[0, 1]`.

    This module changes the time measure to reduce composition error and future
    semigroup defect. It does not force trajectories in state space to become
    straight lines; it only reparameterizes time.

    The implementation learns positive densities on fixed bins, then normalizes
    their cumulative sum into a piecewise-linear CDF.
    """

    def __init__(self, num_bins: int = 64, init_bias: float = 0.0) -> None:
        super().__init__()
        if num_bins < 4:
            raise ValueError("num_bins must be at least 4")
        self.num_bins = num_bins
        self.logits = nn.Parameter(torch.full((num_bins,), float(init_bias)))

    def _bin_heights(self) -> Tensor:
        heights = F.softplus(self.logits) + 1e-4
        return heights / heights.sum()

    def _cdf(self) -> Tensor:
        heights = self._bin_heights()
        cdf = torch.cat(
            [
                torch.zeros(1, device=heights.device, dtype=heights.dtype),
                torch.cumsum(heights, dim=0),
            ],
            dim=0,
        )
        cdf[-1] = 1.0
        return cdf

    def _grid(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.linspace(0.0, 1.0, steps=self.num_bins + 1, device=device, dtype=dtype)

    def forward(self, t: Tensor) -> Tensor:
        """Map `t` to `u` with a piecewise-linear monotone CDF."""

        t = t.clamp(0.0, 1.0)
        cdf = self._cdf()
        grid = self._grid(t.device, t.dtype)
        flat_t = t.reshape(-1)
        scaled = flat_t * self.num_bins
        indices = torch.clamp(scaled.floor().long(), max=self.num_bins - 1)
        local_alpha = scaled - indices.float()
        u0 = cdf[indices]
        u1 = cdf[indices + 1]
        warped = u0 + local_alpha * (u1 - u0)
        return warped.reshape_as(t)

    @torch.no_grad()
    def inverse(self, u: Tensor) -> Tensor:
        """Approximate `t = g_phi^{-1}(u)` with torch-native interpolation."""

        u = u.clamp(0.0, 1.0)
        cdf = self._cdf().to(device=u.device, dtype=u.dtype)
        grid = self._grid(u.device, u.dtype)
        flat_u = u.reshape(-1)
        indices = torch.searchsorted(cdf, flat_u, right=True) - 1
        indices = indices.clamp(0, self.num_bins - 1)
        u0 = cdf[indices]
        u1 = cdf[indices + 1]
        t0 = grid[indices]
        t1 = grid[indices + 1]
        denom = torch.clamp(u1 - u0, min=1e-6)
        alpha = (flat_u - u0) / denom
        restored = t0 + alpha * (t1 - t0)
        return restored.reshape_as(u)

    @torch.no_grad()
    def grid_cache(self) -> tuple[Tensor, Tensor]:
        """Return a discrete `(t_grid, u_grid)` cache for diagnostics or inverse acceleration."""

        t_grid = self._grid(self.logits.device, self.logits.dtype)
        u_grid = self._cdf().to(dtype=t_grid.dtype)
        return t_grid, u_grid
