"""One-dimensional periodic spatial SSA simulator API.

The public API is intentionally small, while the event loop is delegated to the
Numba cell-list implementation vendored in :mod:`kirkwood_article.sim.numba_sim_normal`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kirkwood_article.sim.numba_sim_normal import SSANormalState, make_normal_ssa_1d


@dataclass(frozen=True)
class SSAParams:
    """Parameters for a one-species 1D birth-death-competition process."""

    length: float = 100.0
    birth_rate: float = 1.0
    death_rate: float = 0.0
    competition_rate: float = 0.01
    birth_sigma: float = 1.0
    death_sigma: float = 1.0
    seed: int = 0
    death_cull_sigmas: float = 5.0
    cell_count: int | None = None

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.birth_rate < 0 or self.death_rate < 0 or self.competition_rate < 0:
            raise ValueError("rates must be non-negative")
        if self.birth_sigma <= 0 or self.death_sigma <= 0:
            raise ValueError("kernel sigmas must be positive")
        if self.death_cull_sigmas <= 0:
            raise ValueError("death_cull_sigmas must be positive")


@dataclass
class SSAState:
    """Mutable simulator state backed by the Numba normal-kernel SSA."""

    params: SSAParams
    backend: SSANormalState

    @property
    def positions(self) -> np.ndarray:
        """Return a defensive copy of current one-dimensional coordinates."""

        return get_positions_1d(self)

    @property
    def time(self) -> float:
        """Current simulation time."""

        return self.backend.current_time()

    @property
    def events(self) -> int:
        """Current event count."""

        return int(self.backend.event_count)


def initialize(params: SSAParams, initial_population: int) -> SSAState:
    """Create a reproducible Numba-backed state with uniform initial particles."""

    if initial_population < 0:
        raise ValueError("initial_population must be non-negative")
    backend = make_normal_ssa_1d(
        M=1,
        area_len=params.length,
        birth_rates=[params.birth_rate],
        death_rates=[params.death_rate],
        dd_matrix=[[params.competition_rate]],
        birth_std=[params.birth_sigma],
        death_std=[[params.death_sigma]],
        death_cull_sigmas=params.death_cull_sigmas,
        cell_count=params.cell_count,
        is_periodic=True,
        seed=params.seed,
    )
    backend.spawn_random(0, initial_population)
    return SSAState(params=params, backend=backend)


def population(state: SSAState) -> int:
    """Return the current number of particles."""

    return state.backend.current_population()


def get_positions_1d(state: SSAState) -> np.ndarray:
    """Return a defensive copy of current one-dimensional coordinates."""

    n = population(state)
    return np.asarray(state.backend.positions[:n, 0], dtype=float).copy()


def run_events(state: SSAState, n_events: int) -> int:
    """Run up to ``n_events`` events and return the number actually performed."""

    return state.backend.run_events(int(n_events))


def run_until(state: SSAState, t_end: float, max_events: int | None = None) -> int:
    """Run until absolute simulation time reaches ``t_end`` or ``max_events`` is reached."""

    performed = 0
    while state.backend.current_time() < t_end and (max_events is None or performed < max_events):
        duration = t_end - state.backend.current_time()
        if max_events is None:
            done = state.backend.run_until_time(duration)
        else:
            done = state.backend.run_events(min(10_000, max_events - performed))
        performed += done
        if done == 0:
            break
    return performed
