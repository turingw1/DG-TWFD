from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class RunRoots:
    run_root: Path
    checkpoint_dir: Path
    sample_dir: Path
    log_dir: Path


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _resolve_include(path_like: str) -> Path:
    include_path = ROOT / "configs" / path_like
    if include_path.suffix != ".yaml":
        include_path = include_path.with_suffix(".yaml")
    return include_path


def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    loaded = _read_yaml(path)

    base_path = loaded.pop("base", None)
    includes = loaded.pop("includes", [])

    merged: dict[str, Any] = {}
    if base_path:
        merged = _merge_dicts(merged, _read_yaml(ROOT / base_path))

    for include in includes:
        merged = _merge_dicts(merged, _read_yaml(_resolve_include(include)))

    merged = _merge_dicts(merged, loaded)
    return _expand_env_vars(merged)


def resolve_run_roots(run_root: str | Path) -> RunRoots:
    root = Path(run_root)
    return RunRoots(
        run_root=root,
        checkpoint_dir=root / "checkpoints",
        sample_dir=root / "samples",
        log_dir=root / "logs",
    )
