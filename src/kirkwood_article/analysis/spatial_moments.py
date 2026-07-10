"""Spatial moment estimators and result containers for saved coordinate traces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kirkwood_article.sim.observables import density, pair_correlation_1d


@dataclass(frozen=True)
class FirstMomentResult:
    """Estimated first spatial moment for one coordinate sample."""

    density: float
    population: int
    length: float


@dataclass(frozen=True)
class SecondMomentResult:
    """Pair-correlation representation of a second spatial moment estimate."""

    radii: np.ndarray
    values: np.ndarray
    dr: float
    r_max: float


@dataclass(frozen=True)
class SpatialMomentSummary:
    """Spatial moment estimates computed from one saved coordinate sample."""

    first: FirstMomentResult
    second: SecondMomentResult | None = None


def first_spatial_moment_result(positions: np.ndarray, length: float) -> FirstMomentResult:
    """Compute the first spatial moment result from one coordinate sample."""

    return FirstMomentResult(density=density(positions, length), population=len(positions), length=length)


def second_spatial_moment_result(
    positions: np.ndarray, length: float, dr: float, r_max: float
) -> SecondMomentResult:
    """Compute the second spatial moment result from one coordinate sample."""

    radii, values = pair_correlation_1d(positions, length, dr, r_max)
    return SecondMomentResult(radii=radii, values=values, dr=dr, r_max=r_max)


def summarize_spatial_moments(
    positions: np.ndarray, length: float, dr: float | None = None, r_max: float | None = None
) -> SpatialMomentSummary:
    """Compute available spatial moment summaries from one saved coordinate sample."""

    first = first_spatial_moment_result(positions, length)
    second = None
    if dr is not None and r_max is not None:
        second = second_spatial_moment_result(positions, length, dr, r_max)
    return SpatialMomentSummary(first=first, second=second)
