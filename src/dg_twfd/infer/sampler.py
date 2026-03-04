"""DG-TWFD inference sampler."""

from __future__ import annotations

import time

import torch
from torch import Tensor

from dg_twfd.engine.amp import autocast_context
from dg_twfd.infer.schedules import build_t_schedule_from_u, build_u_schedule


@torch.no_grad()
def sample_dg_twfd(
    models: dict[str, torch.nn.Module],
    timewarp: torch.nn.Module,
    boundary: torch.nn.Module,
    noise: Tensor | None,
    steps: int,
    device: torch.device | str,
    enable_boundary: bool = True,
    amp: bool = True,
    gate_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Sample `x0` by composing `M_theta` over a warped time schedule.

    The sampler uses equal spacing in `u`, converts it to `t_i = g_phi^{-1}(u_i)`,
    and composes:
    `x_{t_{i+1}} = M_theta(t_i, t_{i+1}, x_{t_i})`.
    """

    student = models["student"]
    student.eval()
    timewarp.eval()
    boundary.eval()

    device = torch.device(device)
    dtype = next(student.parameters()).dtype
    if noise is None:
        sample_batch = 1
        channels = student.out_proj.out_channels
        image_size = 32
        noise = torch.randn(sample_batch, channels, image_size, image_size, device=device, dtype=dtype)
    else:
        noise = noise.to(device=device, dtype=dtype)

    u_schedule = build_u_schedule(steps, device=device, dtype=noise.dtype)
    t_schedule = build_t_schedule_from_u(timewarp, u_schedule)

    x = noise
    with autocast_context(device.type, amp):
        if enable_boundary:
            x = boundary(x, enabled=True, gate_weight=gate_weight)
        for idx in range(len(t_schedule) - 1):
            t_current = torch.full((x.shape[0],), float(t_schedule[idx].item()), device=device, dtype=noise.dtype)
            t_next = torch.full(
                (x.shape[0],),
                float(t_schedule[idx + 1].item()),
                device=device,
                dtype=noise.dtype,
            )
            x = student(x, t_current, t_next)

    diagnostics = {
        "u_schedule": u_schedule.detach().cpu(),
        "t_schedule": t_schedule.detach().cpu(),
    }
    return x, diagnostics


@torch.no_grad()
def profile_sampling(
    models: dict[str, torch.nn.Module],
    timewarp: torch.nn.Module,
    boundary: torch.nn.Module,
    noise: Tensor,
    steps_list: list[int],
    device: torch.device | str,
    amp: bool = True,
    enable_boundary: bool = True,
    gate_weight: float = 1.0,
) -> list[dict[str, float]]:
    """Profile inference latency, NFE, and peak memory across step counts."""

    device = torch.device(device)
    results: list[dict[str, float]] = []
    for steps in steps_list:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        _, diagnostics = sample_dg_twfd(
            models=models,
            timewarp=timewarp,
            boundary=boundary,
            noise=noise,
            steps=steps,
            device=device,
            enable_boundary=enable_boundary,
            amp=amp,
            gate_weight=gate_weight,
        )
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            peak_mem = 0.0
        latency_ms = (time.perf_counter() - start) * 1000.0
        results.append(
            {
                "steps": float(steps),
                "nfe": float(steps),
                "latency_ms": latency_ms,
                "peak_mem_mib": peak_mem,
                "t_start": float(diagnostics["t_schedule"][0].item()),
                "t_end": float(diagnostics["t_schedule"][-1].item()),
            }
        )
    return results
