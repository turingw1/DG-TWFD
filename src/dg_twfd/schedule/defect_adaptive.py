"""Defect-adaptive sampling over time bins."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class DefectAdaptiveScheduler:
    """Maintain EMA defect estimates and sample time bins from `exp(eta * d_hat)`."""

    def __init__(
        self,
        num_bins: int = 64,
        ema_decay: float = 0.9,
        eta: float = 1.5,
        eps: float = 1e-6,
        seed: int = 42,
    ) -> None:
        self.num_bins = num_bins
        self.ema_decay = ema_decay
        self.eta = eta
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        self.defect_ema = np.zeros(num_bins, dtype=np.float64)

    def _bin_indices(self, t: Tensor | np.ndarray) -> np.ndarray:
        values = t.detach().cpu().numpy() if isinstance(t, Tensor) else np.asarray(t)
        values = np.clip(values, 0.0, 1.0 - self.eps)
        return np.floor(values * self.num_bins).astype(np.int64)

    def update(self, t: Tensor | np.ndarray, defect_value: Tensor | np.ndarray | float) -> None:
        t_indices = self._bin_indices(t)
        defect_array = (
            defect_value.detach().cpu().numpy()
            if isinstance(defect_value, Tensor)
            else np.asarray(defect_value)
        )
        if defect_array.ndim == 0:
            defect_array = np.full_like(t_indices, float(defect_array), dtype=np.float64)
        for index, value in zip(t_indices, defect_array):
            self.defect_ema[index] = (
                self.ema_decay * self.defect_ema[index] + (1.0 - self.ema_decay) * float(value)
            )

    def probabilities(self) -> np.ndarray:
        logits = self.eta * (self.defect_ema - self.defect_ema.max())
        weights = np.exp(logits)
        probs = weights / np.clip(weights.sum(), a_min=self.eps, a_max=None)
        return probs

    def sample(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        probs = self.probabilities()
        indices = self.rng.choice(self.num_bins, size=batch_size, p=probs)
        offsets = self.rng.random(batch_size)
        samples = (indices + offsets) / self.num_bins
        tensor = torch.tensor(samples, device=device, dtype=dtype)
        return tensor.clamp(0.0, 1.0 - self.eps)

    def set_eta(self, eta: float) -> None:
        self.eta = eta
