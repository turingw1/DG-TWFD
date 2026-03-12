"""Match and semigroup defect losses."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from dg_twfd.schedule.defect_adaptive import DefectAdaptiveScheduler


def _reduce_loss(loss_tensor: Tensor, per_pixel_mean: bool = True) -> Tensor:
    if per_pixel_mean and loss_tensor.ndim > 1:
        return loss_tensor.mean()
    return loss_tensor.mean()


class MatchLoss(nn.Module):
    """Teacher regression loss for `M_theta(t, s, x_t) ~= x_s`.

    `L_match = ||M_theta(t, s, x_t) - x_s||^2`, with optional Huber fallback.
    """

    def __init__(self, loss_type: str = "l2", huber_delta: float = 0.1) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.huber_delta = huber_delta

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if self.loss_type == "huber":
            return F.huber_loss(prediction, target, delta=self.huber_delta)
        return F.mse_loss(prediction, target)


class SemigroupDefectLoss(nn.Module):
    """Semigroup defect loss based on a composed student path.

    For `t > s > u`, the loss compares the direct student map `M_theta(t, u, x_t)`
    against the composed path `M_theta(s, u, M_theta(t, s, x_t))`.
    """

    def __init__(
        self,
        per_pixel_mean: bool = True,
        eps: float = 1e-4,
        short_weight: float = 0.3,
        mid_weight: float = 0.5,
        long_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.per_pixel_mean = per_pixel_mean
        self.eps = eps
        self.chain_weights = torch.tensor([short_weight, mid_weight, long_weight], dtype=torch.float32)

    def _sample_u_from_chain_buckets(self, s: Tensor) -> Tensor:
        device = s.device
        weights = (self.chain_weights / self.chain_weights.sum()).to(device=device)
        bucket = torch.multinomial(weights, num_samples=s.shape[0], replacement=True)
        ratio = torch.zeros_like(s)
        short_mask = bucket == 0
        mid_mask = bucket == 1
        long_mask = bucket == 2
        if short_mask.any():
            ratio[short_mask] = torch.empty(short_mask.sum(), device=device).uniform_(0.75, 0.95)
        if mid_mask.any():
            ratio[mid_mask] = torch.empty(mid_mask.sum(), device=device).uniform_(0.35, 0.75)
        if long_mask.any():
            ratio[long_mask] = torch.empty(long_mask.sum(), device=device).uniform_(0.05, 0.35)
        return torch.clamp(ratio * s, min=0.0, max=1.0 - self.eps)

    def _sample_u(
        self,
        s: Tensor,
        scheduler: DefectAdaptiveScheduler | None = None,
    ) -> Tensor:
        if scheduler is None:
            u = self._sample_u_from_chain_buckets(s)
        else:
            sampled = scheduler.sample(s.shape[0], device=s.device, dtype=s.dtype)
            u = torch.minimum(sampled, torch.clamp(s - self.eps, min=0.0))
        return torch.clamp(u, min=0.0, max=1.0 - self.eps)

    def forward(
        self,
        student: nn.Module,
        x_t: Tensor,
        t: Tensor,
        s: Tensor,
        scheduler: DefectAdaptiveScheduler | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        u = self._sample_u(s, scheduler)
        x_u_direct = student(x_t, t, u)
        x_s_mid = student(x_t, t, s)
        x_u_comp = student(x_s_mid, s, u)
        per_element = (x_u_direct - x_u_comp) ** 2
        defect_value = _reduce_loss(per_element, per_pixel_mean=self.per_pixel_mean)
        per_sample = per_element.flatten(1).mean(dim=1)
        return defect_value, u, per_sample
