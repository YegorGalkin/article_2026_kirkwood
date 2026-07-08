"""Simulation observables used by the scaling experiment."""

from __future__ import annotations

import numpy as np


def density(positions: np.ndarray, length: float) -> float:
    """Return particle density on a one-dimensional interval."""

    return float(len(positions) / length)


def pair_correlation_fft_1d(
    positions: np.ndarray, length: float, dr: float, r_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate periodic 1D pair correlation with an FFT convolution.

    The density is histogrammed on a grid of width approximately ``dr`` and
    circularly autocorrelated via FFT. Self-pairs are explicitly subtracted from
    the zero-lag bin before normalization by ``N * (N - 1)``.
    """

    if dr <= 0 or r_max <= 0:
        raise ValueError("dr and r_max must be positive")
    n = len(positions)
    n_r = int(r_max / dr) + 1
    if n < 2:
        return np.arange(n_r, dtype=float) * dr, np.full(n_r, np.nan, dtype=float)

    n_bins = max(int(round(length / dr)), 1)
    bin_width = length / n_bins
    counts, _ = np.histogram(positions % length, bins=n_bins, range=(0.0, length))
    rho_hat = np.fft.fft(counts.astype(float))
    corr = np.fft.ifft(np.abs(rho_hat) ** 2).real
    corr[0] -= n

    g_r = corr * length / (n * (n - 1) * bin_width)
    n_keep = min(n_r, n_bins // 2 + 1)
    radii = np.arange(n_keep, dtype=float) * bin_width
    return radii, g_r[:n_keep]


def pair_correlation_1d(
    positions: np.ndarray, length: float, dr: float, r_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Alias for the FFT pair-correlation estimator used in long experiments."""

    return pair_correlation_fft_1d(positions, length, dr, r_max)


def first_spatial_moment(positions: np.ndarray, length: float) -> float:
    """Return the first spatial moment, here the particle density."""

    return density(positions, length)


def second_spatial_moment(
    positions: np.ndarray, length: float, dr: float, r_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return a pair-correlation representation of the second spatial moment."""

    return pair_correlation_fft_1d(positions, length, dr, r_max)
