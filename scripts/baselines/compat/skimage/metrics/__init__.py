"""Minimal metrics shim for CTM no-GAN training imports."""

from __future__ import annotations

import numpy as np


def structural_similarity(src1, src2, *_, data_range=255, **__) -> float:
    """Return a lightweight similarity proxy when skimage is unavailable."""
    arr1 = np.asarray(src1, dtype=np.float64)
    arr2 = np.asarray(src2, dtype=np.float64)
    mse = np.mean((arr1 - arr2) ** 2)
    return float(max(0.0, 1.0 - mse / float(data_range**2)))
