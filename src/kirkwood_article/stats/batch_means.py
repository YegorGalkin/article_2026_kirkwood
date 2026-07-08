"""Batch and replicate-level uncertainty estimators."""

from __future__ import annotations

import numpy as np


def mean_and_se(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return sample mean and unbiased standard error along ``axis``."""

    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=axis)
    n = np.sum(~np.isnan(values), axis=axis)
    se = np.full_like(mean, np.nan, dtype=float)
    centered = values - np.expand_dims(mean, axis)
    squared = np.where(np.isnan(centered), 0.0, centered**2)
    sum_squared = np.sum(squared, axis=axis)
    valid = n > 1
    variance = np.full_like(mean, np.nan, dtype=float)
    np.divide(sum_squared, n - 1, out=variance, where=valid)
    std = np.sqrt(variance)
    se = np.where(valid, std / np.sqrt(n), se)
    return mean, se


def batch_mean(values: np.ndarray, batch_size: int) -> np.ndarray:
    """Split a 1D series into complete batches and return batch means."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    values = np.asarray(values, dtype=float)
    n_batches = len(values) // batch_size
    if n_batches == 0:
        return np.array([], dtype=float)
    trimmed = values[: n_batches * batch_size]
    return trimmed.reshape(n_batches, batch_size).mean(axis=1)


def batch_mean_and_se(values: np.ndarray, batch_size: int) -> tuple[float, float]:
    """Estimate a mean and standard error from complete batch means."""

    batches = batch_mean(values, batch_size)
    if len(batches) == 0:
        return float("nan"), float("nan")
    mean, se = mean_and_se(batches)
    return float(mean), float(se)
