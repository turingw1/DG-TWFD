from __future__ import annotations

from pathlib import Path

import torch

from dgfm.models import build_velocity_model


@torch.no_grad()
def device_from_config(config: dict) -> torch.device:
    requested = config.get("runtime", {}).get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@torch.no_grad()
def load_model_from_checkpoint(config: dict, checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint, map_location=device)
    model = build_velocity_model(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def sample_with_ode(model: torch.nn.Module, x_init: torch.Tensor, step_count: int, method: str = "midpoint") -> torch.Tensor:
    x = x_init
    time_grid = torch.linspace(0.0, 1.0, steps=step_count + 1, device=x_init.device, dtype=x_init.dtype)
    batch = x.shape[0]
    for idx in range(step_count):
        t0 = time_grid[idx]
        t1 = time_grid[idx + 1]
        dt = t1 - t0
        t0_vec = torch.full((batch,), float(t0.item()), device=x.device, dtype=x.dtype)
        if method == "euler":
            k1 = model(x, t0_vec)
            x = x + dt * k1
            continue
        if method == "heun2":
            k1 = model(x, t0_vec)
            t1_vec = torch.full((batch,), float(t1.item()), device=x.device, dtype=x.dtype)
            k2 = model(x + dt * k1, t1_vec)
            x = x + 0.5 * dt * (k1 + k2)
            continue
        if method == "midpoint":
            k1 = model(x, t0_vec)
            tmid = t0 + 0.5 * dt
            tmid_vec = torch.full((batch,), float(tmid.item()), device=x.device, dtype=x.dtype)
            k2 = model(x + 0.5 * dt * k1, tmid_vec)
            x = x + dt * k2
            continue
        raise ValueError(f"Unsupported solver method: {method}")
    return x


@torch.no_grad()
def to_unit_interval(images: torch.Tensor) -> torch.Tensor:
    return torch.clamp(images * 0.5 + 0.5, 0.0, 1.0)
