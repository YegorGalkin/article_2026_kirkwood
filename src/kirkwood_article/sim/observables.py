"""Simulation observables used by the scaling experiment."""

from __future__ import annotations

import numpy as np


def density(positions: np.ndarray, length: float) -> float:
    """Return particle density on a one-dimensional interval."""

    return float(len(positions) / length)


def pair_correlation_1d(
    positions: np.ndarray, length: float, dr: float, r_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the radial 1D pair correlation for nonzero periodic separations.

    Self-pairs are excluded explicitly. For a uniform Poisson process, nonzero
    bins should fluctuate around one.
    """

    if dr <= 0 or r_max <= 0:
        raise ValueError("dr and r_max must be positive")
    n = len(positions)
    edges = np.arange(0.0, r_max + dr, dr)
    radii = 0.5 * (edges[:-1] + edges[1:])
    if n < 2:
        return radii, np.full_like(radii, np.nan, dtype=float)

    diff = np.abs(positions[:, None] - positions[None, :])
    dist = np.minimum(diff, length - diff)
    mask = ~np.eye(n, dtype=bool)
    counts, _ = np.histogram(dist[mask], bins=edges)
    shell_width = 2.0 * dr
    expected = n * (n - 1) * shell_width / length
    return radii, counts.astype(float) / expected


def first_spatial_moment(positions: np.ndarray, length: float) -> float:
    """Return the first spatial moment, here the particle density."""

    return density(positions, length)


def second_spatial_moment(
    positions: np.ndarray, length: float, dr: float, r_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return a pair-correlation representation of the second spatial moment."""

    return pair_correlation_1d(positions, length, dr, r_max)
