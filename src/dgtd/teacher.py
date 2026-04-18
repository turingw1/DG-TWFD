from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from dgfm.teachers import build_teacher

from .cache import build_trajectory_payload, interpolate_state
from .sigma import SigmaSchedule, build_sigma_schedule


CONTINUATION_SOURCE_NONE = 0
CONTINUATION_SOURCE_ONLINE = 1
CONTINUATION_SOURCE_CACHED_AFFINE = 2
CONTINUATION_SOURCE_CACHED_EXACT = 3
CONTINUATION_SOURCE_BOOTSTRAP = 4

CONTINUATION_SOURCE_NAMES = {
    CONTINUATION_SOURCE_NONE: "none",
    CONTINUATION_SOURCE_ONLINE: "online",
    CONTINUATION_SOURCE_CACHED_AFFINE: "cached_affine",
    CONTINUATION_SOURCE_CACHED_EXACT: "cached_exact",
    CONTINUATION_SOURCE_BOOTSTRAP: "bootstrap",
}


@dataclass(slots=True)
class TeacherContinuation:
    continuation: Tensor | None
    source_ids: Tensor
    teacher_s: Tensor | None = None
    teacher_u: Tensor | None = None
    rel_error: Tensor | None = None
    alpha: Tensor | None = None
    exact_mask: Tensor | None = None
    used_online_anchor: Tensor | None = None


@dataclass(slots=True)
class TeacherAdapter:
    config: dict
    sigma_schedule: SigmaSchedule
    online_teacher: object | None = None
    proximity_rtol: float = 0.05
    continuation_mode: str = "affine_fallback"
    online_continuation_mode: str = "affine_mainline"
    keep_cached_exact: bool = True

    def prepare(self, device: torch.device) -> None:
        if self.online_teacher is not None and hasattr(self.online_teacher, "prepare"):
            self.online_teacher.prepare(device)

    def has_online_teacher(self) -> bool:
        return self.online_teacher is not None

    def online_u_grid(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        num_points = int(self.config.get("teacher", {}).get("retain_num_points", 33))
        if num_points < 2:
            raise ValueError("teacher.retain_num_points must be at least 2 for online trajectories")
        return torch.linspace(0.0, 1.0, steps=num_points, device=device, dtype=dtype)

    def online_trajectory_from_x0(
        self,
        x_0: Tensor,
        *,
        cond: Tensor | None = None,
        device: torch.device,
    ) -> dict[str, Tensor]:
        if self.online_teacher is None or not hasattr(self.online_teacher, "sample_trajectory_from_x0"):
            raise RuntimeError("Online teacher trajectory sampling is not available")
        u_grid = self.online_u_grid(device=device, dtype=x_0.dtype)
        traj = self.online_teacher.sample_trajectory_from_x0(x_0=x_0, u_grid=u_grid, device=device)
        return build_trajectory_payload(
            times=traj.t_grid,
            states=traj.x_grid,
            cond=cond,
            device=device,
            online_anchor=True,
        )

    def cached_state(self, traj: dict[str, Tensor], t: Tensor) -> Tensor:
        return interpolate_state(traj, t)

    def local_gain(self, s: Tensor, u: Tensor) -> Tensor:
        return self.sigma_schedule.alpha(s, u)

    def _source_none(self, z: Tensor) -> Tensor:
        return torch.full(
            (z.shape[0],),
            CONTINUATION_SOURCE_NONE,
            device=z.device,
            dtype=torch.long,
        )

    def _online_anchor_mask(self, traj: dict[str, Tensor] | None, batch_size: int, device: torch.device) -> Tensor:
        if traj is None:
            return torch.zeros(batch_size, device=device, dtype=torch.bool)
        value = traj.get("teacher_anchor_online")
        if value is None:
            return torch.zeros(batch_size, device=device, dtype=torch.bool)
        mask = value.to(device=device, dtype=torch.bool).view(-1)
        if mask.numel() == 1 and batch_size > 1:
            mask = mask.expand(batch_size)
        if mask.numel() != batch_size:
            raise ValueError(f"teacher_anchor_online batch mismatch: {mask.numel()} vs {batch_size}")
        return mask

    def local_flow(
        self,
        traj: dict[str, Tensor] | None,
        s: Tensor,
        u: Tensor,
        z: Tensor,
        *,
        extra: dict | None = None,
    ) -> TeacherContinuation:
        del extra
        source_none = self._source_none(z)
        used_online_anchor = self._online_anchor_mask(traj, z.shape[0], z.device)
        if bool(used_online_anchor.any()) and self.online_continuation_mode == "affine_mainline":
            x_s_online = self.cached_state(traj, s).to(device=z.device, dtype=z.dtype)
            x_u_online = self.cached_state(traj, u).to(device=z.device, dtype=z.dtype)
            rel = (z.detach() - x_s_online).flatten(1).square().mean(dim=1)
            denom = torch.clamp(x_s_online.flatten(1).square().mean(dim=1), min=1.0e-6)
            rel_error = rel / denom
            alpha = self.local_gain(s, u).to(device=z.device, dtype=z.dtype)
            online = x_u_online.detach() + alpha.view(-1, 1, 1, 1) * (z - x_s_online.detach())
            return TeacherContinuation(
                continuation=online,
                source_ids=torch.full_like(source_none, CONTINUATION_SOURCE_ONLINE),
                teacher_s=x_s_online.detach(),
                teacher_u=x_u_online.detach(),
                rel_error=rel_error.detach(),
                alpha=alpha.detach(),
                exact_mask=torch.zeros_like(source_none, dtype=torch.bool),
                used_online_anchor=used_online_anchor,
            )
        if self.online_teacher is not None and hasattr(self.online_teacher, "local_flow"):
            with torch.no_grad():
                online = self.online_teacher.local_flow(s, u, z.detach())
            if online is not None:
                if not isinstance(online, Tensor):
                    online = torch.as_tensor(online, device=z.device, dtype=z.dtype)
                online = online.to(device=z.device, dtype=z.dtype).detach()
                return TeacherContinuation(
                    continuation=online,
                    source_ids=torch.full_like(source_none, CONTINUATION_SOURCE_ONLINE),
                    exact_mask=torch.zeros_like(source_none, dtype=torch.bool),
                    used_online_anchor=used_online_anchor,
                )
        if traj is not None:
            x_s_cached = self.cached_state(traj, s).to(device=z.device, dtype=z.dtype)
            x_u_cached = self.cached_state(traj, u).to(device=z.device, dtype=z.dtype)
            rel = (z.detach() - x_s_cached).flatten(1).square().mean(dim=1)
            denom = torch.clamp(x_s_cached.flatten(1).square().mean(dim=1), min=1.0e-6)
            rel_error = rel / denom
            exact_mask = rel_error <= float(self.proximity_rtol)
            alpha = self.local_gain(s, u).to(device=z.device, dtype=z.dtype)
            if self.continuation_mode not in {"affine_fallback", "cached_affine"}:
                cached_exact = x_u_cached.detach()
                return TeacherContinuation(
                    continuation=cached_exact,
                    source_ids=torch.full_like(source_none, CONTINUATION_SOURCE_CACHED_EXACT),
                    teacher_s=x_s_cached.detach(),
                    teacher_u=x_u_cached.detach(),
                    rel_error=rel_error.detach(),
                    alpha=alpha.detach(),
                    exact_mask=exact_mask.detach(),
                    used_online_anchor=used_online_anchor,
                )
            affine = x_u_cached.detach() + alpha.view(-1, 1, 1, 1) * (z - x_s_cached.detach())
            source_ids = torch.full_like(source_none, CONTINUATION_SOURCE_CACHED_AFFINE)
            if self.keep_cached_exact and bool(exact_mask.any()):
                affine = affine.clone()
                affine[exact_mask] = x_u_cached.detach()[exact_mask]
                source_ids[exact_mask] = CONTINUATION_SOURCE_CACHED_EXACT
            return TeacherContinuation(
                continuation=affine,
                source_ids=source_ids,
                teacher_s=x_s_cached.detach(),
                teacher_u=x_u_cached.detach(),
                rel_error=rel_error.detach(),
                alpha=alpha.detach(),
                exact_mask=exact_mask.detach(),
                used_online_anchor=used_online_anchor,
            )
        return TeacherContinuation(
            continuation=None,
            source_ids=source_none,
            exact_mask=torch.zeros_like(source_none, dtype=torch.bool),
            used_online_anchor=used_online_anchor,
        )

    def sigma(self, t: Tensor) -> Tensor:
        return self.sigma_schedule.sigma(t)

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
        sigma_schedule=build_sigma_schedule(config),
        online_teacher=online_teacher,
        proximity_rtol=float(dgtd_cfg.get("teacher_proximity_rtol", 0.05)),
        continuation_mode=str(dgtd_cfg.get("teacher_continuation_mode", "affine_fallback")),
        online_continuation_mode=str(dgtd_cfg.get("online_continuation_mode", "affine_mainline")),
        keep_cached_exact=bool(dgtd_cfg.get("teacher_keep_cached_exact", True)),
    )
