"""Boundary correction loss."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    """Loss for the gateable boundary corrector `B_psi`.

    When enabled, the model learns
    `L_boundary = ||B_psi(x_{t_max}) - x_{t_max-delta}^T||^2`.
    """

    def __init__(self, loss_type: str = "l2", huber_delta: float = 0.1) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.huber_delta = huber_delta

    def forward(
        self,
        boundary_model: nn.Module,
        x_boundary: Tensor,
        target: Tensor,
        gate_weight: float | Tensor = 1.0,
        enabled: bool = True,
    ) -> tuple[Tensor, Tensor]:
        prediction = boundary_model(x_boundary, enabled=enabled, gate_weight=gate_weight)
        if not enabled:
            zero = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
            return zero, prediction
        if self.loss_type == "huber":
            loss = F.huber_loss(prediction, target, delta=self.huber_delta)
        else:
            loss = F.mse_loss(prediction, target)
        return loss, prediction
