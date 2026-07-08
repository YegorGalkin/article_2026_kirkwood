"""Kernel and periodic-geometry helpers for one-dimensional simulations."""

from __future__ import annotations

import numpy as np


def periodic_displacement(
    x: np.ndarray | float, y: np.ndarray | float, length: float
) -> np.ndarray | float:
    """Return the signed shortest displacement from ``y`` to ``x`` on a periodic interval."""

    return (np.asarray(x) - np.asarray(y) + 0.5 * length) % length - 0.5 * length


def periodic_distance(
    x: np.ndarray | float, y: np.ndarray | float, length: float
) -> np.ndarray | float:
    """Return the shortest absolute distance between two periodic one-dimensional positions."""

    return np.abs(periodic_displacement(x, y, length))


def gaussian_kernel(distance: np.ndarray | float, sigma: float) -> np.ndarray | float:
    """Evaluate a normalized one-dimensional Gaussian kernel at ``distance``."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    distance_array = np.asarray(distance)
    return np.exp(-0.5 * (distance_array / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)
