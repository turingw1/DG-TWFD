from __future__ import annotations

import torch

from dgfm.evaluators.qualitative import build_multistep_panel


def test_build_multistep_panel_shape_with_noise_column() -> None:
    noise = torch.zeros(3, 3, 4, 4)
    samples_by_step = {
        1: torch.ones(3, 3, 4, 4),
        2: torch.ones(3, 3, 4, 4) * 2,
    }
    panel = build_multistep_panel(
        noise=noise,
        samples_by_step=samples_by_step,
        step_counts=[1, 2],
        include_noise=True,
    )
    assert panel.shape == (9, 3, 4, 4)


def test_build_multistep_panel_shape_without_noise_column() -> None:
    noise = torch.zeros(2, 3, 4, 4)
    samples_by_step = {
        4: torch.ones(2, 3, 4, 4),
        8: torch.ones(2, 3, 4, 4) * 2,
        16: torch.ones(2, 3, 4, 4) * 3,
    }
    panel = build_multistep_panel(
        noise=noise,
        samples_by_step=samples_by_step,
        step_counts=[4, 8, 16],
        include_noise=False,
    )
    assert panel.shape == (6, 3, 4, 4)
