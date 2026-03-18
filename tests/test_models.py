from __future__ import annotations

import resource

import torch

from dg_twfd.config import load_config
from dg_twfd.models import BoundaryCorrector, TimeWarpMonotone, build_student_from_config
from dg_twfd.utils.seed import seed_everything


def _report_memory() -> None:
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"CUDA peak memory: {peak:.2f} MiB")
    else:
        usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"Approx RAM usage: {usage_kb / 1024:.2f} MiB")


def test_phase2_model_components() -> None:
    cfg = load_config("debug_4060")
    seed_everything(cfg.experiment.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warp = TimeWarpMonotone(
        num_bins=cfg.model.timewarp_num_bins,
        init_bias=cfg.model.timewarp_init_bias,
    ).to(device)
    boundary = BoundaryCorrector(
        channels=cfg.data.channels,
        hidden_channels=cfg.model.boundary_hidden_channels,
        num_blocks=cfg.model.boundary_num_blocks,
    ).to(device)
    student = build_student_from_config(cfg).to(device)

    batch_size = cfg.data.batch_size
    t = torch.rand(batch_size, device=device)
    s = t * 0.5

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.autocast(device_type=device.type, enabled=cfg.runtime.amp and device.type == "cuda"):
        u = warp(t)
        t_hat = warp.inverse(u)
        max_error = (t - t_hat).abs().max().item()
        print(f"timewarp inverse max error: {max_error:.6f}")
        assert max_error < 0.1

        x = torch.randn(
            batch_size,
            cfg.data.channels,
            cfg.data.image_size,
            cfg.data.image_size,
            device=device,
        )

        x_corr = boundary(x, enabled=True, gate_weight=0.5)
        x_passthrough = boundary(x, enabled=False)
        print(f"boundary enabled shape: {tuple(x_corr.shape)}")
        print(f"boundary disabled shape: {tuple(x_passthrough.shape)}")
        assert x_corr.shape == x.shape
        assert x_passthrough.shape == x.shape
        assert torch.allclose(x_passthrough, x)

        x_s_pred = student(x, t, s)
        print(f"student output shape: {tuple(x_s_pred.shape)}")
        assert x_s_pred.shape == x.shape


def test_student_residual_scales_with_delta() -> None:
    cfg = load_config("debug_4060")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = build_student_from_config(cfg).to(device)
    x = torch.randn(4, cfg.data.channels, cfg.data.image_size, cfg.data.image_size, device=device)
    t = torch.ones(4, device=device)
    s_small = torch.full((4,), 0.95, device=device)
    s_large = torch.zeros(4, device=device)
    out_small = student(x, t, s_small)
    out_large = student(x, t, s_large)
    delta_small = (out_small - x).abs().mean().item()
    delta_large = (out_large - x).abs().mean().item()
    assert delta_large > delta_small


def test_dit_student_backbone_forward() -> None:
    cfg = load_config(
        "debug_4060",
        overrides=[
            "model.student_backbone='dit'",
            "model.hidden_channels=64",
            "model.cond_dim=64",
            "model.student_num_blocks=2",
            "model.student_num_heads=4",
            "model.student_patch_size=4",
        ],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = build_student_from_config(cfg).to(device)
    x = torch.randn(2, cfg.data.channels, cfg.data.image_size, cfg.data.image_size, device=device)
    t = torch.rand(2, device=device)
    s = t * 0.5
    out = student(x, t, s)
    assert out.shape == x.shape

    _report_memory()
