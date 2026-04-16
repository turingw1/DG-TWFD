from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from dgfm.teachers import build_teacher

from .cache import interpolate_state


@dataclass(slots=True)
class TeacherAdapter:
    config: dict
    online_teacher: object | None = None
    proximity_rtol: float = 0.05

    def prepare(self, device: torch.device) -> None:
        if self.online_teacher is not None and hasattr(self.online_teacher, "prepare"):
            self.online_teacher.prepare(device)

    def cached_state(self, traj: dict[str, Tensor], t: Tensor) -> Tensor:
        return interpolate_state(traj, t)

    def local_flow(
        self,
        traj: dict[str, Tensor] | None,
        s: Tensor,
        u: Tensor,
        z: Tensor,
        *,
        extra: dict | None = None,
    ) -> Tensor | None:
        del extra
        if traj is not None:
            x_s_cached = self.cached_state(traj, s).to(device=z.device, dtype=z.dtype)
            x_u_cached = self.cached_state(traj, u).to(device=z.device, dtype=z.dtype)
            rel = (z.detach() - x_s_cached).flatten(1).square().mean(dim=1)
            denom = torch.clamp(x_s_cached.flatten(1).square().mean(dim=1), min=1.0e-6)
            near_mask = (rel / denom) <= float(self.proximity_rtol)
            if bool(near_mask.all()):
                return x_u_cached.detach()
        if self.online_teacher is not None and hasattr(self.online_teacher, "local_flow"):
            return self.online_teacher.local_flow(s, u, z)
        return None

    def sigma(self, t: Tensor) -> Tensor:
        sigma_min = float(self.config.get("model", {}).get("sigma_min", 1.0e-3))
        return torch.clamp(1.0 - t, min=sigma_min)

    def precondition(self, x: Tensor, t: Tensor) -> Tensor:
        del t
        return x

    def get_condition(self, batch: dict[str, Tensor]) -> Tensor | None:
        cond = batch.get("cond")
        if cond is None:
            return None
        return cond


def build_teacher_adapter(config: dict) -> TeacherAdapter:
    teacher_cfg = config.get("teacher", {})
    online_teacher = None
    dgtd_cfg = config.get("dgtd", {})
    if not bool(dgtd_cfg.get("disable_online_teacher", False)) and teacher_cfg.get("type") not in {None, "none"}:
        try:
            online_teacher = build_teacher(config)
        except Exception:
            online_teacher = None
    return TeacherAdapter(
        config=config,
        online_teacher=online_teacher,
        proximity_rtol=float(dgtd_cfg.get("teacher_proximity_rtol", 0.05)),
    )
