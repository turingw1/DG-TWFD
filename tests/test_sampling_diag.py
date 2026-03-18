from __future__ import annotations

import torch

from dg_twfd.config import load_config
from dg_twfd.infer import sample_dg_twfd
from dg_twfd.models import BoundaryCorrector, TimeWarpMonotone, build_student_from_config


def test_sampling_diagnostics_include_step_history() -> None:
    cfg = load_config("debug_4060")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {
        "student": build_student_from_config(cfg).to(device),
        "timewarp": TimeWarpMonotone(
            num_bins=cfg.model.timewarp_num_bins,
            init_bias=cfg.model.timewarp_init_bias,
        ).to(device),
        "boundary": BoundaryCorrector(
            channels=cfg.data.channels,
            hidden_channels=cfg.model.boundary_hidden_channels,
            num_blocks=cfg.model.boundary_num_blocks,
        ).to(device),
    }
    noise = torch.randn(2, cfg.data.channels, cfg.data.image_size, cfg.data.image_size, device=device)
    samples, diagnostics = sample_dg_twfd(
        models=models,
        timewarp=models["timewarp"],
        boundary=models["boundary"],
        noise=noise,
        steps=4,
        device=device,
        enable_boundary=False,
        amp=False,
        gate_weight=cfg.boundary.gate_weight,
    )
    assert samples.shape == noise.shape
    assert diagnostics["x_steps"].shape[0] == 5
    assert diagnostics["x_steps"].shape[1:] == samples.cpu().shape
    assert len(diagnostics["step_stats"]) == 5
