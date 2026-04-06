from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class TargetBatch:
    x_t: torch.Tensor
    x_s_target: torch.Tensor
    t: torch.Tensor
    s: torch.Tensor
    x_0: torch.Tensor
    x_1: torch.Tensor


def _skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    p_mean = -1.2
    p_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * p_std + p_mean).exp()
    time = 1.0 / (1.0 + sigma)
    return torch.clamp(time, min=1.0e-4, max=1.0)


class AnalyticPathTargetBuilder:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.target_cfg = config.get("target", {})
        self.train_cfg = config.get("train", {})

    def sample_times(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        min_time = float(self.target_cfg.get("min_time", 1.0e-4))
        max_time = float(self.target_cfg.get("max_time", 0.999))
        min_gap = float(self.target_cfg.get("min_time_gap", 0.05))
        use_skewed = bool(self.train_cfg.get("skewed_timesteps", True))
        if use_skewed:
            s = _skewed_timestep_sample(batch_size, device=device)
        else:
            s = torch.rand(batch_size, device=device)
        s = torch.clamp(s, min=min_time + min_gap, max=max_time)
        fraction = torch.rand(batch_size, device=device)
        slack = torch.clamp(s - min_time - min_gap, min=0.0)
        t = s - min_gap - fraction * slack
        t = torch.maximum(t, torch.full_like(t, min_time))
        t = torch.minimum(t, s - min_gap)
        return t, s

    loader_mode = "images"
    needs_path = True

    def build(self, x_0: torch.Tensor, x_1: torch.Tensor, path) -> TargetBatch:
        t, s = self.sample_times(batch_size=x_0.shape[0], device=x_0.device)
        path_sample_t = path.sample(x_0=x_0, x_1=x_1, t=t)
        path_sample_s = path.sample(x_0=x_0, x_1=x_1, t=s)
        return TargetBatch(
            x_t=path_sample_t.x_t,
            x_s_target=path_sample_s.x_t,
            t=t,
            s=s,
            x_0=x_0,
            x_1=x_1,
        )

    def build_from_batch(self, batch, *, device: torch.device, path) -> TargetBatch:
        images, _labels = batch
        x_1 = images.to(device) * 2.0 - 1.0
        x_0 = torch.randn_like(x_1)
        return self.build(x_0=x_0, x_1=x_1, path=path)


class TrajectoryShardTargetBuilder:
    loader_mode = "trajectory_shard"
    needs_path = False

    def __init__(self, config: dict) -> None:
        self.config = config

    def build_from_batch(self, batch, *, device: torch.device, path=None) -> TargetBatch:
        del path
        x_t = batch["x_t"].to(device)
        x_s = batch["x_s"].to(device)
        t = batch["t"].to(device)
        s = batch["s"].to(device)
        return TargetBatch(
            x_t=x_t,
            x_s_target=x_s,
            t=t,
            s=s,
            x_0=x_t,
            x_1=x_s,
        )


def build_target_builder(config: dict):
    target_cfg = config.get("target", {})
    mode = str(target_cfg.get("builder", "analytic_path"))
    if mode == "analytic_path":
        return AnalyticPathTargetBuilder(config)
    if mode == "trajectory_shard":
        return TrajectoryShardTargetBuilder(config)
    if mode == "teacher_sampler":
        raise NotImplementedError("teacher_sampler target builder hook exists but is not implemented in the first map branch.")
    raise ValueError(f"Unsupported target builder mode: {mode}")
