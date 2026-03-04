"""AMP helpers for training loops."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def amp_enabled(device_type: str, enabled: bool) -> bool:
    return enabled and device_type == "cuda" and torch.cuda.is_available()


def autocast_context(device_type: str, enabled: bool):
    if amp_enabled(device_type, enabled):
        return torch.autocast(device_type=device_type, enabled=True)
    return nullcontext()


def build_grad_scaler(device_type: str, enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device=device_type, enabled=amp_enabled(device_type, enabled))
