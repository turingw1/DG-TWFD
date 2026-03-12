"""Teacher trajectory interfaces and adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from dg_twfd.config import DGConfig


class TeacherTrajectory(ABC):
    """Abstract teacher API centered on `Phi_T(t->s, x_t)`.

    The project uses `t in [0, 1]` with larger values meaning noisier states.
    Real diffusion teachers should implement `forward_map()` directly instead of
    relying on the online dummy-trajectory assumptions used in Phase 1.
    """

    def __init__(self, cfg: DGConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def sample_x0(self, batch_size: int, device: torch.device | str) -> Tensor:
        """Sample the boundary state used to start or probe teacher trajectories."""

    @abstractmethod
    def forward_map(self, x_t: Tensor, t: Tensor, s: Tensor) -> Tensor:
        """Approximate the teacher map `Phi_T(t->s, x_t)`."""

    @abstractmethod
    def make_trajectory(self, x0: Tensor, t_grid: Tensor) -> dict[float, Tensor]:
        """Generate `{t: x_t}` along an ordered time grid."""

    def sample_trajectory(
        self,
        batch_size: int,
        t_grid: Tensor,
        device: torch.device | str,
        labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return a shard-friendly trajectory payload.

        The default implementation uses `sample_x0()` plus `make_trajectory()`.
        Real diffusion teachers may override this for a more native rollout.
        """

        del labels
        device = torch.device(device)
        x0 = self.sample_x0(batch_size, device)
        sorted_grid = torch.sort(t_grid.float()).values.to(device)
        trajectory = self.make_trajectory(x0, sorted_grid)
        x_grid = torch.stack([trajectory[float(t.item())] for t in sorted_grid], dim=1)
        return {
            "t_grid": sorted_grid.detach().cpu(),
            "x_grid": x_grid.detach().cpu(),
            "x0": x_grid[:, -1].detach().cpu(),
        }


class DummyTeacherTrajectory(TeacherTrajectory):
    """Cheap nonlinear teacher used only for unit tests and debug.

    This is not the final paper teacher. It simulates a curved flow with
    `v(x, t) = a(t) * x + b(t) * tanh(x)` and uses explicit Euler integration so
    later defect and warp logic has a nontrivial placeholder trajectory.
    """

    def sample_x0(self, batch_size: int, device: torch.device | str) -> Tensor:
        image_size = self.cfg.data.image_size
        channels = self.cfg.data.channels
        return (
            torch.randn(batch_size, channels, image_size, image_size, device=device)
            * self.cfg.teacher.x0_std
        )

    def _velocity(self, x: Tensor, time_value: Tensor) -> Tensor:
        a_t = self.cfg.teacher.velocity_scale * torch.cos(time_value * torch.pi)
        b_t = self.cfg.teacher.nonlinearity_scale * (1.0 + torch.sin(time_value * torch.pi))
        while a_t.ndim < x.ndim:
            a_t = a_t.unsqueeze(-1)
            b_t = b_t.unsqueeze(-1)
        return a_t * x + b_t * torch.tanh(x)

    def forward_map(self, x_t: Tensor, t: Tensor, s: Tensor) -> Tensor:
        if x_t.shape[0] != t.shape[0] or t.shape != s.shape:
            raise ValueError("Batch dimensions of x_t, t, and s must match")

        x_current = x_t.clone()
        steps = max(1, self.cfg.data.teacher_integration_steps)
        delta = (s - t) / steps
        current_t = t.clone()
        for _ in range(steps):
            x_current = x_current + delta.view(-1, 1, 1, 1) * self._velocity(x_current, current_t)
            current_t = current_t + delta
        return x_current

    def make_trajectory(self, x0: Tensor, t_grid: Tensor) -> dict[float, Tensor]:
        if t_grid.ndim != 1:
            raise ValueError("t_grid must be a 1D tensor")
        sorted_grid = torch.sort(t_grid).values.to(device=x0.device, dtype=x0.dtype)
        trajectory: dict[float, Tensor] = {}
        x_current = x0.clone()
        current_time = torch.zeros(x0.shape[0], device=x0.device, dtype=x0.dtype)
        for t_value in sorted_grid:
            target_time = torch.full_like(current_time, float(t_value.item()))
            x_current = self.forward_map(x_current, current_time, target_time)
            current_time = target_time
            trajectory[float(t_value.item())] = x_current.clone()
        return trajectory


class DiffusersDDPMTeacher(TeacherTrajectory):
    """Adapter for diffusers DDPM teachers such as `google/ddpm-cifar10-32`.

    This adapter is meant for real teacher rollout and shard collection. The
    repository does not auto-install `diffusers`, so loading this teacher
    requires the user to provide the dependency and model weights manually.
    """

    def __init__(self, cfg: DGConfig) -> None:
        super().__init__(cfg)
        self.pipeline = None
        self.unet = None
        self.scheduler = None
        self._loaded_device: str | None = None

    def _require_path(self) -> str:
        path = self.cfg.teacher.pretrained_model_name_or_path
        if not path:
            raise ValueError(
                "teacher.pretrained_model_name_or_path must be set for diffusers_ddpm"
            )
        return path

    def _ensure_pipeline(self, device: torch.device | str) -> None:
        device = torch.device(device)
        if self.pipeline is not None and self._loaded_device == str(device):
            return
        try:
            from diffusers import DDPMPipeline
            from diffusers.schedulers import DDIMScheduler
        except ImportError as exc:
            raise ImportError(
                "DiffusersDDPMTeacher requires `diffusers`. Install it in the "
                "`consistency` environment before using teacher_type=diffusers_ddpm."
            ) from exc

        path = self._require_path()
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.pipeline = DDPMPipeline.from_pretrained(
            path,
            local_files_only=self.cfg.teacher.local_files_only,
            torch_dtype=torch_dtype,
        ).to(device)
        self.unet = self.pipeline.unet.eval()
        if self.cfg.teacher.solver == "ddim":
            self.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        else:
            self.scheduler = self.pipeline.scheduler
        self.scheduler.set_timesteps(self.cfg.teacher.num_inference_steps, device=device)
        self._loaded_device = str(device)

    def sample_x0(self, batch_size: int, device: torch.device | str) -> Tensor:
        return torch.randn(
            batch_size,
            self.cfg.data.channels,
            self.cfg.data.image_size,
            self.cfg.data.image_size,
            device=device,
        ) * self.cfg.teacher.x0_std

    def _normalized_to_timestep(self, normalized_time: float) -> int:
        assert self.scheduler is not None
        total = int(self.scheduler.config.num_train_timesteps) - 1
        return int(round(float(normalized_time) * total))

    def _normalized_to_inference_index(self, normalized_time: float) -> int:
        assert self.scheduler is not None
        target_timestep = self._normalized_to_timestep(normalized_time)
        timesteps = self.scheduler.timesteps.detach().cpu().to(torch.long)
        index = int(torch.argmin(torch.abs(timesteps - target_timestep)).item())
        return index

    def forward_eps(
        self,
        x_t: Tensor,
        timestep_ids: Tensor,
        labels: Tensor | None = None,
    ) -> Tensor:
        self._ensure_pipeline(x_t.device)
        assert self.unet is not None
        timestep_ids = timestep_ids.to(device=x_t.device, dtype=torch.long).view(-1)
        kwargs: dict[str, Any] = {}
        if labels is not None and self.cfg.teacher.class_cond:
            kwargs["class_labels"] = labels.to(x_t.device)
        if x_t.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = self.unet(x_t, timestep_ids, **kwargs)
        else:
            output = self.unet(x_t, timestep_ids, **kwargs)
        return output.sample

    def _rollout_between_indices(
        self,
        x_t: Tensor,
        start_index: int,
        end_index: int,
        labels: Tensor | None = None,
    ) -> Tensor:
        self._ensure_pipeline(x_t.device)
        assert self.scheduler is not None
        if end_index < start_index:
            raise ValueError("end_index must be >= start_index for descending scheduler timesteps")
        current = x_t
        timesteps = self.scheduler.timesteps.to(x_t.device)
        for step_index in range(start_index, end_index):
            step_value = timesteps[step_index]
            step_batch = step_value.expand(current.shape[0])
            eps_pred = self.forward_eps(current, step_batch, labels=labels)
            current = self.scheduler.step(eps_pred, int(step_value.item()), current).prev_sample
        return current

    @torch.no_grad()
    def forward_map(self, x_t: Tensor, t: Tensor, s: Tensor) -> Tensor:
        self._ensure_pipeline(x_t.device)
        unique_t = torch.unique(t.detach())
        unique_s = torch.unique(s.detach())
        if unique_t.numel() == 1 and unique_s.numel() == 1:
            t_value = float(unique_t.item())
            s_value = float(unique_s.item())
            if s_value > t_value:
                raise ValueError("Diffusion teacher expects t >= s")
            start_index = self._normalized_to_inference_index(t_value)
            end_index = self._normalized_to_inference_index(s_value)
            if end_index <= start_index:
                return x_t.clone()
            return self._rollout_between_indices(x_t, start_index, end_index)

        outputs = []
        for index in range(x_t.shape[0]):
            current = x_t[index : index + 1]
            t_value = float(t[index].item())
            s_value = float(s[index].item())
            if s_value > t_value:
                raise ValueError("Diffusion teacher expects t >= s")
            start_index = self._normalized_to_inference_index(t_value)
            end_index = self._normalized_to_inference_index(s_value)
            if end_index <= start_index:
                outputs.append(current)
                continue
            outputs.append(self._rollout_between_indices(current, start_index, end_index))
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def make_trajectory(self, x0: Tensor, t_grid: Tensor) -> dict[float, Tensor]:
        if t_grid.ndim != 1:
            raise ValueError("t_grid must be a 1D tensor")
        descending = torch.sort(t_grid.float(), descending=True).values.to(x0.device)
        current = x0.clone()
        current_time = torch.full((x0.shape[0],), float(descending[0].item()), device=x0.device)
        trajectory: dict[float, Tensor] = {float(descending[0].item()): current.clone()}
        for next_t in descending[1:]:
            target_time = torch.full_like(current_time, float(next_t.item()))
            current = self.forward_map(current, current_time, target_time)
            current_time = target_time
            trajectory[float(next_t.item())] = current.clone()
        return trajectory

    @torch.no_grad()
    def sample_trajectory(
        self,
        batch_size: int,
        t_grid: Tensor,
        device: torch.device | str,
        labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        device = torch.device(device)
        self._ensure_pipeline(device)
        start_state = self.sample_x0(batch_size, device)
        descending = torch.sort(t_grid.float(), descending=True).values.to(device)
        index_grid = [self._normalized_to_inference_index(float(t.item())) for t in descending]
        current = start_state
        x_grid = [current.detach()]
        for idx in range(len(index_grid) - 1):
            start_index = index_grid[idx]
            end_index = index_grid[idx + 1]
            if end_index > start_index:
                current = self._rollout_between_indices(current, start_index, end_index, labels=labels)
            x_grid.append(current.detach())
        stacked = torch.stack(x_grid, dim=1).float().cpu()
        payload = {
            "t_grid": descending.detach().float().cpu(),
            "x_grid": stacked,
            "x0": stacked[:, -1],
        }
        if labels is not None:
            payload["y"] = labels.detach().cpu()
        return payload


def build_teacher(cfg: DGConfig) -> TeacherTrajectory:
    """Instantiate the configured teacher adapter."""

    teacher_type = cfg.teacher.teacher_type
    if teacher_type == "dummy":
        return DummyTeacherTrajectory(cfg)
    if teacher_type == "diffusers_ddpm":
        return DiffusersDDPMTeacher(cfg)
    raise ValueError(f"Unsupported teacher_type: {teacher_type}")
