"""Configuration loading for the DG-TWFD scaffold."""

from __future__ import annotations

from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    seed: int


@dataclass(slots=True)
class DataConfig:
    dataset_type: str
    channels: int
    image_size: int
    dataset_size: int
    val_dataset_size: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: Optional[int]
    drop_last: bool
    trajectory_cache_mode: bool
    num_cached_trajectories: int
    time_grid_size: int
    sample_strategy: str
    pair_short_max: int
    pair_mid_max: int
    pair_long_max: int
    pair_short_weight: float
    pair_mid_weight: float
    pair_long_weight: float
    pair_endpoint_weight: float
    high_noise_t_weight: float
    high_noise_t_fraction: float
    triplet_local_gap1: int
    triplet_local_gap2: int
    teacher_integration_steps: int
    trajectory_shard_dir: Optional[str]
    trajectory_file_glob: str


@dataclass(slots=True)
class RuntimeConfig:
    device: str
    amp: bool
    gradient_accumulation: int


@dataclass(slots=True)
class TeacherConfig:
    teacher_type: str
    pretrained_model_name_or_path: Optional[str]
    local_files_only: bool
    num_inference_steps: int
    solver: str
    class_cond: bool
    num_classes: int
    velocity_scale: float
    nonlinearity_scale: float
    x0_std: float


@dataclass(slots=True)
class ModelConfig:
    time_embed_dim: int
    cond_dim: int
    hidden_channels: int
    boundary_hidden_channels: int
    boundary_num_blocks: int
    student_num_blocks: int
    timewarp_num_bins: int
    timewarp_init_bias: float
    predict_residual: bool


@dataclass(slots=True)
class LossConfig:
    match_loss_type: str
    huber_delta: float
    defect_weight: float
    warp_weight: float
    boundary_weight: float
    per_pixel_mean: bool
    semigroup_short_weight: float
    semigroup_mid_weight: float
    semigroup_long_weight: float


@dataclass(slots=True)
class ScheduleConfig:
    num_bins: int
    ema_decay: float
    eta: float
    eps: float
    seed: int


@dataclass(slots=True)
class BoundaryTrainConfig:
    gate_weight: float
    enable_until_step: int


@dataclass(slots=True)
class TrainConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    log_every: int
    save_every: int
    checkpoint_dir: str
    resume_path: Optional[str]
    warp_update_every: int
    max_train_steps: Optional[int]


@dataclass(slots=True)
class LoggingConfig:
    level: str


@dataclass(slots=True)
class DGConfig:
    """Configuration tree for Phase 1.

    The data section already carries the time discretization size `m`
    (`time_grid_size`) needed by future cached teacher trajectories.
    """

    experiment: ExperimentConfig
    data: DataConfig
    runtime: RuntimeConfig
    teacher: TeacherConfig
    model: ModelConfig
    loss: LossConfig
    schedule: ScheduleConfig
    boundary: BoundaryTrainConfig
    train: TrainConfig
    logging: LoggingConfig


_ROOT = Path(__file__).resolve().parents[2]


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


def _parse_override_value(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "null":
        return None
    try:
        return literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return raw_value


def _apply_overrides(config_dict: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    merged = dict(config_dict)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key_path, raw_value = override.split("=", 1)
        value = _parse_override_value(raw_value)
        target = merged
        keys = key_path.split(".")
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value
    return merged


def _build_dataclass(config_dict: dict[str, Any]) -> DGConfig:
    return DGConfig(
        experiment=ExperimentConfig(**config_dict["experiment"]),
        data=DataConfig(**config_dict["data"]),
        runtime=RuntimeConfig(**config_dict["runtime"]),
        teacher=TeacherConfig(**config_dict["teacher"]),
        model=ModelConfig(**config_dict["model"]),
        loss=LossConfig(**config_dict["loss"]),
        schedule=ScheduleConfig(**config_dict["schedule"]),
        boundary=BoundaryTrainConfig(**config_dict["boundary"]),
        train=TrainConfig(**config_dict["train"]),
        logging=LoggingConfig(**config_dict["logging"]),
    )


def load_config(profile: str, overrides: Optional[list[str]] = None) -> DGConfig:
    """Load config with priority `default < profile < CLI overrides`.

    This provides the Phase 1 configuration center that later phases can use to
    drive `M_theta(t, s, x_t)`, `Phi_T(t->s, x_t)`, and `u = g_phi(t)` modules.
    """

    base = _read_yaml(_ROOT / "config" / "default.yaml")
    profile_path = _ROOT / "config" / "profiles" / f"{profile}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Unknown profile: {profile}")
    merged = _merge_dicts(base, _read_yaml(profile_path))
    if overrides:
        merged = _apply_overrides(merged, overrides)
    return _build_dataclass(merged)
