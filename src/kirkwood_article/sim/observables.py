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



def triplet_correlation_ordered_1d(
    positions: np.ndarray, length: float, dr: float, r_max: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate periodic ordered 1D triplet correlation ``x <- r1 -> y <- r2 -> z``.

    Particles are binned by their distance to each center particle ``y``: ``r1``
    is the periodic distance from ``y`` leftward to ``x`` and ``r2`` is the
    periodic distance from ``y`` rightward to ``z``. The estimator counts
    ordered triplets of distinct particles and normalizes by the third
    factorial count and two bin widths, matching the pair-correlation density
    convention used by :func:`pair_correlation_fft_1d`.
    """

    if dr <= 0 or r_max <= 0:
        raise ValueError("dr and r_max must be positive")
    n = len(positions)
    n_r = int(r_max / dr) + 1
    r_values = np.arange(n_r, dtype=float) * dr
    if n < 3:
        return r_values, r_values.copy(), np.full((n_r, n_r), np.nan, dtype=float)

    n_bins = max(int(round(length / dr)), 1)
    bin_width = length / n_bins
    n_keep = min(n_r, n_bins // 2 + 1)
    r_values = np.arange(n_keep, dtype=float) * bin_width
    counts = np.zeros((n_keep, n_keep), dtype=float)
    wrapped = np.asarray(positions, dtype=float) % length

    for center_index, center in enumerate(wrapped):
        deltas_right = (wrapped - center) % length
        deltas_left = (center - wrapped) % length
        other = np.arange(n) != center_index
        right_bins = np.floor(deltas_right[other] / bin_width + 0.5).astype(int)
        left_bins = np.floor(deltas_left[other] / bin_width + 0.5).astype(int)
        right_counts = np.bincount(
            right_bins[(0 <= right_bins) & (right_bins < n_keep)], minlength=n_keep
        ).astype(float)
        left_counts = np.bincount(
            left_bins[(0 <= left_bins) & (left_bins < n_keep)], minlength=n_keep
        ).astype(float)
        counts += np.outer(left_counts, right_counts)

        # Remove cases where the same non-center particle supplied both x and z.
        valid_same = (left_bins == right_bins) & (0 <= left_bins) & (left_bins < n_keep)
        if np.any(valid_same):
            same_counts = np.bincount(left_bins[valid_same], minlength=n_keep).astype(float)
            counts[np.arange(n_keep), np.arange(n_keep)] -= same_counts

    g3 = counts * length**2 / (n * (n - 1) * (n - 2) * bin_width**2)
    return r_values, r_values.copy(), g3


def third_spatial_moment(
    positions: np.ndarray, length: float, dr: float, r_max: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a triplet-correlation representation of the third spatial moment."""

    return triplet_correlation_ordered_1d(positions, length, dr, r_max)


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
