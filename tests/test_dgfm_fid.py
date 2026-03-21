from __future__ import annotations

import numpy as np
import torch

from dgfm.evaluators.common import solver_nfe
from dgfm.evaluators.fid import GaussianStats, _resolve_inception_weights_path, frechet_distance, to_uint8


def test_frechet_distance_zero_for_identical_stats() -> None:
    mu = np.zeros(4, dtype=np.float64)
    sigma = np.eye(4, dtype=np.float64)
    stats = GaussianStats(mu=mu, sigma=sigma, count=16)
    fid = frechet_distance(stats, stats)
    assert abs(fid) < 1.0e-8


def test_solver_nfe_counts() -> None:
    assert solver_nfe(16, "euler") == 16
    assert solver_nfe(16, "midpoint") == 32
    assert solver_nfe(16, "heun2") == 32


def test_to_uint8_quantizes_unit_interval_images() -> None:
    image = torch.tensor([[[[0.0, 0.5, 1.0]]]], dtype=torch.float32).repeat(1, 3, 1, 1)
    image_u8 = to_uint8(image)
    assert image_u8.dtype == torch.uint8
    assert image_u8.shape == image.shape
    assert int(image_u8[0, 0, 0, 0]) == 0
    assert int(image_u8[0, 0, 0, 1]) in {127, 128}
    assert int(image_u8[0, 0, 0, 2]) == 255


def test_explicit_inception_weights_path_env(monkeypatch) -> None:
    monkeypatch.setenv("DGFM_TORCH_FIDELITY_WEIGHTS_PATH", "/tmp/custom_inception.pth")
    assert _resolve_inception_weights_path() == "/tmp/custom_inception.pth"


def test_no_mirror_env_returns_none(monkeypatch) -> None:
    monkeypatch.delenv("DGFM_TORCH_FIDELITY_WEIGHTS_PATH", raising=False)
    monkeypatch.delenv("DGFM_TORCH_FIDELITY_MIRROR_URL", raising=False)
    monkeypatch.delenv("DGFM_TORCH_FIDELITY_MIRROR_PREFIX", raising=False)
    assert _resolve_inception_weights_path() is None
