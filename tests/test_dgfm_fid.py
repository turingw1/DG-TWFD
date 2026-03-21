from __future__ import annotations

import numpy as np

from dgfm.evaluators.fid import GaussianStats, frechet_distance


def test_frechet_distance_zero_for_identical_stats() -> None:
    mu = np.zeros(4, dtype=np.float64)
    sigma = np.eye(4, dtype=np.float64)
    stats = GaussianStats(mu=mu, sigma=sigma, count=16)
    fid = frechet_distance(stats, stats)
    assert abs(fid) < 1.0e-8
