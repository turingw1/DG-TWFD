from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MonotoneDensityWarp(nn.Module):
    """Monotone density warp over the current dgfm time convention.

    The project keeps time on `[0, 1]` with:
    - `0.0`: noisiest
    - `1.0`: cleanest

    This module parameterizes a positive density `q_phi(t)` on that interval and
    exposes both:
    - `t_to_r(t)`: cumulative density coordinate
    - `r_to_t(r)`: inverse-CDF map used for warped-time sampling
    """

    def __init__(self, num_bins: int, eps: float = 1.0e-6, init: str = "uniform") -> None:
        super().__init__()
        if num_bins < 4:
            raise ValueError("num_bins must be at least 4")
        self.num_bins = int(num_bins)
        self.eps = float(eps)
        self.density_raw = nn.Parameter(torch.zeros(self.num_bins))
        self.reset_parameters(init=init)

    def reset_parameters(self, *, init: str = "uniform") -> None:
        if init == "uniform":
            self.density_raw.data.zero_()
            return
        if init == "logit_normal":
            centers = torch.linspace(0.5 / self.num_bins, 1.0 - 0.5 / self.num_bins, steps=self.num_bins)
            p_mean = -1.2
            p_std = 1.2
            sigma = (torch.log(torch.clamp(1.0 / torch.clamp(centers, min=self.eps) - 1.0, min=self.eps)) - p_mean) / max(p_std, self.eps)
            density = torch.exp(-0.5 * sigma.square()) + self.eps
            density = density / density.mean()
            self.density_raw.data.copy_(torch.log(torch.expm1(torch.clamp(density, min=self.eps))))
            return
        raise ValueError(f"Unsupported warp init: {init}")

    def density(self) -> Tensor:
        density = F.softplus(self.density_raw) + self.eps
        return density / density.sum()

    def cdf(self) -> Tensor:
        density = self.density()
        cdf = torch.cat(
            [
                torch.zeros(1, device=density.device, dtype=density.dtype),
                torch.cumsum(density, dim=0),
            ],
            dim=0,
        )
        cdf[-1] = 1.0
        return cdf

    def _grid(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.linspace(0.0, 1.0, steps=self.num_bins + 1, device=device, dtype=dtype)

    def _interp_forward(self, values: Tensor, coords: Tensor) -> Tensor:
        flat = coords.reshape(-1).clamp(0.0, 1.0)
        scaled = flat * self.num_bins
        indices = torch.clamp(scaled.floor().long(), max=self.num_bins - 1)
        alpha = scaled - indices.to(dtype=coords.dtype)
        v0 = values[indices]
        v1 = values[indices + 1]
        return (v0 + alpha * (v1 - v0)).reshape_as(coords)

    def t_to_r(self, t: Tensor) -> Tensor:
        cdf = self.cdf().to(device=t.device, dtype=t.dtype)
        return self._interp_forward(cdf, t)

    @torch.no_grad()
    def r_to_t(self, r: Tensor) -> Tensor:
        flat = r.reshape(-1).clamp(0.0, 1.0)
        cdf = self.cdf().to(device=r.device, dtype=r.dtype)
        t_grid = self._grid(device=r.device, dtype=r.dtype)
        indices = torch.searchsorted(cdf, flat, right=True) - 1
        indices = indices.clamp(0, self.num_bins - 1)
        r0 = cdf[indices]
        r1 = cdf[indices + 1]
        t0 = t_grid[indices]
        t1 = t_grid[indices + 1]
        alpha = (flat - r0) / torch.clamp(r1 - r0, min=self.eps)
        restored = t0 + alpha * (t1 - t0)
        return restored.reshape_as(r)

    def density_at(self, t: Tensor) -> Tensor:
        flat = t.reshape(-1).clamp(0.0, 1.0 - self.eps)
        indices = torch.clamp((flat * self.num_bins).floor().long(), min=0, max=self.num_bins - 1)
        density = self.density().to(device=t.device, dtype=t.dtype)
        return density[indices].reshape_as(t)

    def sample_triplets(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        probs = torch.tensor([0.5, 0.3, 0.2], device=device)
        bucket = torch.multinomial(probs, num_samples=batch_size, replacement=True)
        delta_1 = torch.empty(batch_size, device=device)
        delta_2 = torch.empty(batch_size, device=device)
        ranges = (
            (1.0 / 64.0, 1.0 / 16.0),
            (1.0 / 16.0, 1.0 / 4.0),
            (1.0 / 4.0, 1.0),
        )
        for idx, (low, high) in enumerate(ranges):
            mask = bucket == idx
            if not mask.any():
                continue
            width = int(mask.sum().item())
            d1 = torch.empty(width, device=device).uniform_(low, high)
            d2 = torch.empty(width, device=device).uniform_(low, high)
            swap = d2 <= d1
            if swap.any():
                d2[swap] = torch.minimum(d1[swap] + torch.empty_like(d1[swap]).uniform_(self.eps, high - low + self.eps), torch.full_like(d1[swap], high))
            d2 = torch.maximum(d2, d1 + self.eps)
            delta_1[mask] = d1
            delta_2[mask] = d2
        max_start = torch.clamp(1.0 - delta_2, min=self.eps)
        r_t = torch.rand(batch_size, device=device) * max_start
        r_s = torch.clamp(r_t + delta_1, max=1.0 - self.eps)
        r_u = torch.clamp(r_t + delta_2, max=1.0)
        assert torch.all(r_t < r_s)
        assert torch.all(r_s < r_u)
        return r_t, r_s, r_u

    def kl_to_target_density(self, q_D: Tensor) -> Tensor:
        q_phi = self.density().to(device=q_D.device, dtype=q_D.dtype)
        q_D = q_D / torch.clamp(q_D.sum(), min=self.eps)
        return torch.sum(q_D * (torch.log(torch.clamp(q_D, min=self.eps)) - torch.log(torch.clamp(q_phi, min=self.eps))))

    def entropy(self) -> Tensor:
        q_phi = self.density()
        return -(q_phi * torch.log(torch.clamp(q_phi, min=self.eps))).sum()

    def forward(self, t: Tensor) -> Tensor:
        return self.t_to_r(t)

    @torch.no_grad()
    def grid_cache(self) -> tuple[Tensor, Tensor]:
        t_grid = self._grid(device=self.density_raw.device, dtype=self.density_raw.dtype)
        r_grid = self.cdf().to(dtype=t_grid.dtype)
        return t_grid, r_grid


class MonotoneRationalQuadraticSplineWarp(nn.Module):
    """Strictly monotone rational-quadratic spline warp on `[0, 1]`.

    Compared with `MonotoneDensityWarp`, this keeps a positive mass per interval
    but also learns positive knot derivatives. The induced map is smooth inside
    each interval instead of a piecewise-linear CDF.
    """

    def __init__(
        self,
        num_bins: int,
        eps: float = 1.0e-6,
        init: str = "uniform",
        min_bin_mass: float = 1.0e-4,
        min_derivative: float = 1.0e-3,
        inverse_iters: int = 36,
        center_kl_weight: float = 0.25,
        derivative_smoothness_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if num_bins < 4:
            raise ValueError("num_bins must be at least 4")
        if min_bin_mass * num_bins >= 1.0:
            raise ValueError("min_bin_mass * num_bins must be smaller than 1")
        self.num_bins = int(num_bins)
        self.eps = float(eps)
        self.min_bin_mass = float(min_bin_mass)
        self.min_derivative = float(min_derivative)
        self.inverse_iters = int(inverse_iters)
        self.center_kl_weight = float(center_kl_weight)
        self.derivative_smoothness_weight = float(derivative_smoothness_weight)
        self.height_raw = nn.Parameter(torch.zeros(self.num_bins))
        self.derivative_raw = nn.Parameter(torch.zeros(self.num_bins + 1))
        self.reset_parameters(init=init)

    def reset_parameters(self, *, init: str = "uniform") -> None:
        if init == "uniform":
            self.height_raw.data.zero_()
        elif init == "logit_normal":
            centers = torch.linspace(0.5 / self.num_bins, 1.0 - 0.5 / self.num_bins, steps=self.num_bins)
            p_mean = -1.2
            p_std = 1.2
            sigma = (torch.log(torch.clamp(1.0 / torch.clamp(centers, min=self.eps) - 1.0, min=self.eps)) - p_mean) / max(p_std, self.eps)
            density = torch.exp(-0.5 * sigma.square()) + self.eps
            density = density / torch.clamp(density.sum(), min=self.eps)
            raw = torch.log(torch.clamp(density, min=self.eps))
            self.height_raw.data.copy_(raw - raw.mean())
        else:
            raise ValueError(f"Unsupported warp init: {init}")

        target = max(1.0 - self.min_derivative, self.eps)
        inv_softplus = math.log(math.expm1(target))
        self.derivative_raw.data.fill_(inv_softplus)

    def density(self) -> Tensor:
        free_mass = 1.0 - self.min_bin_mass * self.num_bins
        density = F.softmax(self.height_raw, dim=0)
        return self.min_bin_mass + free_mass * density

    def derivatives(self) -> Tensor:
        return F.softplus(self.derivative_raw) + self.min_derivative

    def cdf(self) -> Tensor:
        density = self.density()
        return torch.cat(
            [
                torch.zeros(1, device=density.device, dtype=density.dtype),
                torch.cumsum(density, dim=0),
            ],
            dim=0,
        )

    def _grid(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.linspace(0.0, 1.0, steps=self.num_bins + 1, device=device, dtype=dtype)

    def _spline_terms(self, t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        flat = t.reshape(-1).clamp(0.0, 1.0)
        scaled = flat * self.num_bins
        indices = torch.clamp(scaled.floor().long(), max=self.num_bins - 1)
        theta = scaled - indices.to(dtype=flat.dtype)

        cdf = self.cdf().to(device=t.device, dtype=t.dtype)
        derivatives = self.derivatives().to(device=t.device, dtype=t.dtype)
        y0 = cdf[indices]
        y1 = cdf[indices + 1]
        d0 = derivatives[indices]
        d1 = derivatives[indices + 1]
        height = torch.clamp(y1 - y0, min=self.eps)
        width = 1.0 / float(self.num_bins)
        delta = height / width
        return theta, y0, height, delta, d0, d1, flat

    def t_to_r(self, t: Tensor) -> Tensor:
        theta, y0, height, delta, d0, d1, _ = self._spline_terms(t)
        one_minus_theta = 1.0 - theta
        theta_one_minus_theta = theta * one_minus_theta
        denominator = delta + (d0 + d1 - 2.0 * delta) * theta_one_minus_theta
        numerator = height * (delta * theta.square() + d0 * theta_one_minus_theta)
        r = y0 + numerator / torch.clamp(denominator, min=self.eps)
        return r.reshape_as(t).clamp(0.0, 1.0)

    @torch.no_grad()
    def r_to_t(self, r: Tensor) -> Tensor:
        flat = r.reshape(-1).clamp(0.0, 1.0)
        lo = torch.zeros_like(flat)
        hi = torch.ones_like(flat)
        for _ in range(max(8, self.inverse_iters)):
            mid = 0.5 * (lo + hi)
            value = self.t_to_r(mid)
            lo = torch.where(value < flat, mid, lo)
            hi = torch.where(value >= flat, mid, hi)
        restored = 0.5 * (lo + hi)
        return restored.reshape_as(r)

    def density_at(self, t: Tensor) -> Tensor:
        theta, _, _, delta, d0, d1, _ = self._spline_terms(t)
        one_minus_theta = 1.0 - theta
        theta_one_minus_theta = theta * one_minus_theta
        denominator = delta + (d0 + d1 - 2.0 * delta) * theta_one_minus_theta
        numerator = delta.square() * (
            d1 * theta.square()
            + 2.0 * delta * theta_one_minus_theta
            + d0 * one_minus_theta.square()
        )
        density = numerator / torch.clamp(denominator.square(), min=self.eps)
        return density.reshape_as(t)

    def sample_triplets(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        probs = torch.tensor([0.5, 0.3, 0.2], device=device)
        bucket = torch.multinomial(probs, num_samples=batch_size, replacement=True)
        delta_1 = torch.empty(batch_size, device=device)
        delta_2 = torch.empty(batch_size, device=device)
        ranges = (
            (1.0 / 64.0, 1.0 / 16.0),
            (1.0 / 16.0, 1.0 / 4.0),
            (1.0 / 4.0, 1.0),
        )
        for idx, (low, high) in enumerate(ranges):
            mask = bucket == idx
            if not mask.any():
                continue
            width = int(mask.sum().item())
            d1 = torch.empty(width, device=device).uniform_(low, high)
            d2 = torch.empty(width, device=device).uniform_(low, high)
            swap = d2 <= d1
            if swap.any():
                d2[swap] = torch.minimum(d1[swap] + torch.empty_like(d1[swap]).uniform_(self.eps, high - low + self.eps), torch.full_like(d1[swap], high))
            d2 = torch.maximum(d2, d1 + self.eps)
            delta_1[mask] = d1
            delta_2[mask] = d2
        max_start = torch.clamp(1.0 - delta_2, min=self.eps)
        r_t = torch.rand(batch_size, device=device) * max_start
        r_s = torch.clamp(r_t + delta_1, max=1.0 - self.eps)
        r_u = torch.clamp(r_t + delta_2, max=1.0)
        assert torch.all(r_t < r_s)
        assert torch.all(r_s < r_u)
        return r_t, r_s, r_u

    def kl_to_target_density(self, q_D: Tensor) -> Tensor:
        q_mass = self.density().to(device=q_D.device, dtype=q_D.dtype)
        q_D = q_D / torch.clamp(q_D.sum(), min=self.eps)
        mass_kl = torch.sum(q_D * (torch.log(torch.clamp(q_D, min=self.eps)) - torch.log(torch.clamp(q_mass, min=self.eps))))

        loss = mass_kl
        if self.center_kl_weight > 0.0:
            centers = (torch.arange(self.num_bins, device=q_D.device, dtype=q_D.dtype) + 0.5) / float(self.num_bins)
            center_density = self.density_at(centers)
            center_mass = center_density / torch.clamp(center_density.sum(), min=self.eps)
            center_kl = torch.sum(q_D * (torch.log(torch.clamp(q_D, min=self.eps)) - torch.log(torch.clamp(center_mass, min=self.eps))))
            loss = loss + self.center_kl_weight * center_kl
        if self.derivative_smoothness_weight > 0.0:
            log_d = torch.log(torch.clamp(self.derivatives().to(device=q_D.device, dtype=q_D.dtype), min=self.eps))
            smoothness = torch.diff(log_d).square().mean()
            loss = loss + self.derivative_smoothness_weight * smoothness
        return loss

    def entropy(self) -> Tensor:
        q_phi = self.density()
        return -(q_phi * torch.log(torch.clamp(q_phi, min=self.eps))).sum()

    def forward(self, t: Tensor) -> Tensor:
        return self.t_to_r(t)

    @torch.no_grad()
    def grid_cache(self) -> tuple[Tensor, Tensor]:
        t_grid = self._grid(device=self.height_raw.device, dtype=self.height_raw.dtype)
        r_grid = self.cdf().to(dtype=t_grid.dtype)
        return t_grid, r_grid
