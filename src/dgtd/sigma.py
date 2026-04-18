from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class SigmaSchedule:
    mode: str
    sigma_min: float
    sigma_max: float = 1.0
    alpha_mode: str = "clamped_ratio_sigma"
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    alpha_power: float = 1.0
    t_grid: Tensor | None = None
    sigma_grid: Tensor | None = None

    def sigma(self, t: Tensor) -> Tensor:
        if self.mode == "linear_1mt":
            sigma = float(self.sigma_max) * (1.0 - t)
            return torch.clamp(sigma, min=float(self.sigma_min))
        if self.mode == "explicit_grid":
            if self.t_grid is None or self.sigma_grid is None:
                raise ValueError("explicit_grid sigma schedule requires both t_grid and sigma_grid")
            return _interp_sigma_grid(
                t=t,
                t_grid=self.t_grid.to(device=t.device, dtype=t.dtype),
                sigma_grid=self.sigma_grid.to(device=t.device, dtype=t.dtype),
                sigma_min=float(self.sigma_min),
            )
        raise ValueError(f"Unsupported dgtd sigma_mode: {self.mode}")

    def alpha(self, s: Tensor, u: Tensor, *, eps: float = 1.0e-6) -> Tensor:
        sigma_s = self.sigma(s)
        sigma_u = self.sigma(u)
        if self.alpha_mode == "identity":
            return torch.ones_like(sigma_s)
        ratio = sigma_u / torch.clamp(sigma_s, min=float(eps))
        if self.alpha_mode == "ratio_sigma":
            return ratio
        if self.alpha_mode == "power_ratio_sigma":
            return torch.pow(torch.clamp(ratio, min=float(eps)), float(self.alpha_power))
        if self.alpha_mode == "clamped_ratio_sigma":
            return torch.clamp(ratio, min=float(self.alpha_min), max=float(self.alpha_max))
        raise ValueError(f"Unsupported dgtd alpha_mode: {self.alpha_mode}")


def _as_optional_tensor(values, *, dtype: torch.dtype) -> Tensor | None:
    if values is None:
        return None
    tensor = torch.as_tensor(values, dtype=dtype).view(-1)
    return tensor


def _interp_sigma_grid(
    *,
    t: Tensor,
    t_grid: Tensor,
    sigma_grid: Tensor,
    sigma_min: float,
) -> Tensor:
    if t_grid.ndim != 1 or sigma_grid.ndim != 1:
        raise ValueError("t_grid and sigma_grid must both be 1D")
    if t_grid.numel() != sigma_grid.numel():
        raise ValueError("t_grid and sigma_grid must have the same length")
    if t_grid.numel() < 2:
        raise ValueError("explicit sigma grid requires at least two points")
    if not torch.all(t_grid[1:] >= t_grid[:-1]):
        raise ValueError("explicit sigma t_grid must be monotone non-decreasing")
    query = t.reshape(-1).clamp(float(t_grid[0].item()), float(t_grid[-1].item()))
    upper = torch.searchsorted(t_grid, query, right=False)
    upper = upper.clamp(1, t_grid.numel() - 1)
    lower = upper - 1
    t0 = t_grid[lower]
    t1 = t_grid[upper]
    s0 = sigma_grid[lower]
    s1 = sigma_grid[upper]
    alpha = (query - t0) / torch.clamp(t1 - t0, min=1.0e-6)
    sigma = s0 + alpha * (s1 - s0)
    sigma = torch.clamp(sigma, min=float(sigma_min))
    return sigma.reshape_as(t)


def build_sigma_schedule(config: dict) -> SigmaSchedule:
    dgtd_cfg = config.get("dgtd", {})
    model_cfg = config.get("model", {})
    mode = str(dgtd_cfg.get("sigma_mode", "linear_1mt"))
    sigma_min = float(dgtd_cfg.get("sigma_min", model_cfg.get("sigma_min", 1.0e-3)))
    sigma_max = float(dgtd_cfg.get("sigma_max", 1.0))
    alpha_mode = str(dgtd_cfg.get("alpha_mode", "clamped_ratio_sigma"))
    alpha_min = float(dgtd_cfg.get("alpha_min", 0.0))
    alpha_max = float(dgtd_cfg.get("alpha_max", 1.0))
    alpha_power = float(dgtd_cfg.get("alpha_power", 1.0))
    dtype = torch.float32
    t_grid = _as_optional_tensor(dgtd_cfg.get("sigma_t_grid"), dtype=dtype)
    sigma_grid = _as_optional_tensor(dgtd_cfg.get("sigma_value_grid"), dtype=dtype)
    if mode == "explicit_grid" and (t_grid is None or sigma_grid is None):
        raise ValueError("dgtd sigma_mode=explicit_grid requires sigma_t_grid and sigma_value_grid")
    return SigmaSchedule(
        mode=mode,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        alpha_mode=alpha_mode,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_power=alpha_power,
        t_grid=t_grid,
        sigma_grid=sigma_grid,
    )
