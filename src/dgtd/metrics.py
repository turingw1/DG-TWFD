from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def laplacian_filter(x: Tensor) -> Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected image tensor [B,C,H,W], got {tuple(x.shape)}")
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=x.shape[1])


def high_frequency_norm(x: Tensor) -> Tensor:
    return x.flatten(1).square().mean(dim=1)


def _resolve_sigma(t_or_sigma: Tensor, sigma_fn=None) -> Tensor:
    if sigma_fn is None:
        return torch.clamp(1.0 - t_or_sigma, min=1.0e-3)
    sigma = sigma_fn(t_or_sigma)
    if not isinstance(sigma, Tensor):
        sigma = torch.as_tensor(sigma, device=t_or_sigma.device, dtype=t_or_sigma.dtype)
    return sigma.to(device=t_or_sigma.device, dtype=t_or_sigma.dtype)


def metric_norm(
    R: Tensor,
    u: Tensor,
    sigma_fn=None,
    *,
    lambda_hf_max: float = 0.1,
    sigma_detail: float = 0.2,
    disable_hf_metric: bool = False,
) -> Tensor:
    sigma = _resolve_sigma(u, sigma_fn=sigma_fn)
    l2_term = R.flatten(1).square().mean(dim=1)
    if disable_hf_metric:
        return l2_term
    hf_term = high_frequency_norm(laplacian_filter(R))
    lambda_hf = float(lambda_hf_max) * torch.exp(-(sigma.square()) / max(float(sigma_detail) ** 2, 1.0e-6))
    return l2_term + lambda_hf * hf_term


def min_snr_weight(
    t_or_sigma: Tensor,
    sigma_fn=None,
    *,
    gamma: float = 5.0,
    eps: float = 1.0e-6,
) -> Tensor:
    sigma = _resolve_sigma(t_or_sigma, sigma_fn=sigma_fn)
    snr = 1.0 / torch.clamp(sigma.square(), min=eps)
    return torch.minimum(snr, torch.full_like(snr, float(gamma))) / torch.clamp(snr, min=eps)


def edm_weight(
    t_or_sigma: Tensor,
    sigma_fn=None,
    *,
    sigma_data: float = 0.5,
    eps: float = 1.0e-6,
) -> Tensor:
    sigma = _resolve_sigma(t_or_sigma, sigma_fn=sigma_fn)
    sigma_data_tensor = torch.full_like(sigma, float(sigma_data))
    return (sigma.square() + sigma_data_tensor.square()) / torch.clamp((sigma * sigma_data_tensor).square(), min=eps)
