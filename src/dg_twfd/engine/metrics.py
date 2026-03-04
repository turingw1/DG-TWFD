"""Metric tracking for DG-TWFD training loops."""

from __future__ import annotations

from collections import defaultdict


class MetricTracker:
    """Append-only metric storage for defect and loss curves."""

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = defaultdict(list)

    def update(self, **metrics: float) -> None:
        for key, value in metrics.items():
            self.history[key].append(float(value))

    def latest(self, key: str) -> float | None:
        values = self.history.get(key)
        if not values:
            return None
        return values[-1]
