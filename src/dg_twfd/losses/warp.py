"""Warp loss built from teacher trajectory finite differences."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from dg_twfd.data.dataset import TrajectoryPairDataset
from dg_twfd.data.teacher import TeacherTrajectory
from dg_twfd.models.timewarp import TimeWarpMonotone


class WarpLoss(nn.Module):
    """Finite-difference warp loss for updating `g_phi`.

    Base term:
    `L_warp_base = ||Phi_T(t3->t2, x_t3) - Phi_T(t2->t1, x_t2)||^2`

    To connect the loss to the learnable time-warp without JVP, this
    implementation multiplies the base mismatch by a warped step-balance term
    computed from `u_i = g_phi(t_i)`. The intent is still to learn a time
    measure that makes teacher dynamics more uniform across composed steps.
    """

    def __init__(self, per_pixel_mean: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        self.per_pixel_mean = per_pixel_mean
        self.eps = eps

    def forward(
        self,
        timewarp: TimeWarpMonotone,
        triplet_batch: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        x_32 = triplet_batch["x_t2"]
        x_21 = triplet_batch["x_t1"]
        t3 = triplet_batch["t3"]
        t2 = triplet_batch["t2"]
        t1 = triplet_batch["t1"]

        base = (x_32 - x_21) ** 2
        base_loss = base.mean() if self.per_pixel_mean else base.flatten(1).sum(dim=1).mean()

        u3 = timewarp(t3)
        u2 = timewarp(t2)
        u1 = timewarp(t1)
        step_32 = torch.clamp(u3 - u2, min=self.eps)
        step_21 = torch.clamp(u2 - u1, min=self.eps)
        balance = ((step_32 - step_21) ** 2) / (step_32 + step_21 + self.eps)
        loss = base_loss * (1.0 + balance.mean())
        stats = {
            "base_loss": base_loss.detach(),
            "balance": balance.mean().detach(),
            "u3": u3.detach(),
            "u2": u2.detach(),
            "u1": u1.detach(),
        }
        return loss, stats

    @torch.no_grad()
    def sample_triplet_batch(
        self,
        dataset: TrajectoryPairDataset,
        batch_size: int,
        device: torch.device | str,
    ) -> dict[str, Tensor]:
        return dataset.sample_triplet_batch(batch_size=batch_size, device=device)
