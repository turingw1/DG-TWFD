"""Checkpoint persistence for DG-TWFD training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _strip_orig_mod_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if not all(key.startswith("_orig_mod.") for key in state_dict):
        return state_dict
    return {key[len("_orig_mod.") :]: value for key, value in state_dict.items()}


def export_model_state_dict(model: torch.nn.Module) -> dict[str, Any]:
    base_model = getattr(model, "_orig_mod", model)
    return base_model.state_dict()


def load_model_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, Any],
    *,
    strict: bool = True,
) -> None:
    base_model = getattr(model, "_orig_mod", model)
    candidate = _strip_orig_mod_prefix(state_dict)
    base_model.load_state_dict(candidate, strict=strict)


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, target)
    return target


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)
