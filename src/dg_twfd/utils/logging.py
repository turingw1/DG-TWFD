"""Logging helpers with timestamps and source locations."""

from __future__ import annotations

import logging
from typing import Optional


def setup_logger(level: str = "INFO", name: Optional[str] = None) -> logging.Logger:
    """Configure a logger for debugging the DG-TWFD data pipeline."""

    logger = logging.getLogger(name if name is not None else "dg_twfd")
    if logger.handlers:
        logger.setLevel(level.upper())
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger
