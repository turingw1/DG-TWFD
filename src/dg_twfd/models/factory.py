"""Model factory helpers."""

from __future__ import annotations

from dg_twfd.config import DGConfig
from dg_twfd.models.student import FlowStudent
from dg_twfd.models.student_dit import PatchDiTStudent


def build_student_from_config(cfg: DGConfig):
    if cfg.model.student_backbone == "conv":
        return FlowStudent(
            channels=cfg.data.channels,
            hidden_channels=cfg.model.hidden_channels,
            time_embed_dim=cfg.model.time_embed_dim,
            cond_dim=cfg.model.cond_dim,
            num_blocks=cfg.model.student_num_blocks,
            predict_residual=cfg.model.predict_residual,
            residual_scale_by_delta=cfg.model.residual_scale_by_delta,
            residual_tanh_scale=cfg.model.residual_tanh_scale,
        )
    if cfg.model.student_backbone == "dit":
        return PatchDiTStudent(
            image_size=cfg.data.image_size,
            channels=cfg.data.channels,
            hidden_size=cfg.model.hidden_channels,
            time_embed_dim=cfg.model.time_embed_dim,
            cond_dim=cfg.model.cond_dim,
            num_blocks=cfg.model.student_num_blocks,
            num_heads=cfg.model.student_num_heads,
            patch_size=cfg.model.student_patch_size,
            mlp_ratio=cfg.model.student_mlp_ratio,
            predict_residual=cfg.model.predict_residual,
            residual_scale_by_delta=cfg.model.residual_scale_by_delta,
            residual_tanh_scale=cfg.model.residual_tanh_scale,
        )
    raise ValueError(f"Unsupported student_backbone: {cfg.model.student_backbone}")
