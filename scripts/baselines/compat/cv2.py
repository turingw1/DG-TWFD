"""Small cv2 compatibility shim for CTM no-GAN training.

The upstream training utility imports OpenCV only for PSNR reporting in the
optional similarity evaluator. No-GAN FID training runs here disable that path,
but the import still has to resolve.
"""

from __future__ import annotations

import math

import numpy as np


def PSNR(src1, src2, max_pixel_value: float = 255.0) -> float:
    arr1 = np.asarray(src1, dtype=np.float64)
    arr2 = np.asarray(src2, dtype=np.float64)
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(max_pixel_value / math.sqrt(mse))
