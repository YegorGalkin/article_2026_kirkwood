"""Lightweight placeholders for Kirkwood-closure numerical predictions.

The functions provide a stable interface for experiments while the detailed
closure solver is developed.
"""

from __future__ import annotations

import numpy as np

from kirkwood_article.sim.kernels import gaussian_kernel
from kirkwood_article.sim.ssa_1d import SSAParams


def first_moment_mean_field(params: SSAParams) -> float:
    """Return the spatially homogeneous mean-field density prediction."""

    if params.competition_rate <= 0:
        return np.inf
    return max(params.birth_rate - params.death_rate, 0.0) / params.competition_rate


def second_moment_independent(params: SSAParams, radii: np.ndarray) -> np.ndarray:
    """Return an independent-particles baseline ``g(r)=1`` for comparison plots."""

    return np.ones_like(radii, dtype=float)


def gaussian_competition_kernel(params: SSAParams, radii: np.ndarray) -> np.ndarray:
    """Expose the normalized death kernel used by the simulator."""

    return gaussian_kernel(radii, params.death_sigma)
