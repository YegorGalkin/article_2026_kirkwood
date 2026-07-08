"""Minimal one-dimensional periodic spatial SSA simulator.

The implementation favors a compact, transparent API for reproducible article
experiments. It is intentionally O(N^2) for competition-rate recomputation and
therefore best suited to small and medium validation runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from kirkwood_article.sim.kernels import gaussian_kernel, periodic_distance


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

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.birth_rate < 0 or self.death_rate < 0 or self.competition_rate < 0:
            raise ValueError("rates must be non-negative")
        if self.birth_sigma <= 0 or self.death_sigma <= 0:
            raise ValueError("kernel sigmas must be positive")


@dataclass
class SSAState:
    """Mutable simulator state."""

    params: SSAParams
    positions: np.ndarray
    time: float
    rng: Generator
    events: int = 0


def initialize(params: SSAParams, initial_population: int) -> SSAState:
    """Create a reproducible state with uniformly distributed initial particles."""

    if initial_population < 0:
        raise ValueError("initial_population must be non-negative")
    rng = np.random.default_rng(params.seed)
    positions = rng.uniform(0.0, params.length, size=initial_population)
    return SSAState(params=params, positions=positions.astype(float), time=0.0, rng=rng)


def population(state: SSAState) -> int:
    """Return the current number of particles."""

    return int(state.positions.size)


def get_positions_1d(state: SSAState) -> np.ndarray:
    """Return a defensive copy of the current one-dimensional coordinates."""

    return state.positions.copy()


def _competition_hazards(state: SSAState) -> np.ndarray:
    n = population(state)
    if n == 0 or state.params.competition_rate == 0:
        return np.zeros(n)
    distances = periodic_distance(
        state.positions[:, None], state.positions[None, :], state.params.length
    )
    np.fill_diagonal(distances, np.inf)
    kernel_sum = gaussian_kernel(distances, state.params.death_sigma).sum(axis=1)
    return state.params.competition_rate * kernel_sum


def step(state: SSAState) -> bool:
    """Perform one SSA event, returning ``False`` if no event can occur."""

    n = population(state)
    birth_total = state.params.birth_rate * n
    death_hazards = state.params.death_rate + _competition_hazards(state)
    death_total = float(death_hazards.sum())
    total_rate = birth_total + death_total
    if n == 0 or total_rate <= 0:
        return False

    state.time += float(state.rng.exponential(1.0 / total_rate))
    if state.rng.random() < birth_total / total_rate:
        parent = state.rng.integers(n)
        child = (
            state.positions[parent] + state.rng.normal(0.0, state.params.birth_sigma)
        ) % state.params.length
        state.positions = np.append(state.positions, child)
    else:
        victim = int(state.rng.choice(n, p=death_hazards / death_total))
        state.positions = np.delete(state.positions, victim)
    state.events += 1
    return True


def run_events(state: SSAState, n_events: int) -> int:
    """Run up to ``n_events`` events and return the number actually performed."""

    performed = 0
    for _ in range(n_events):
        if not step(state):
            break
        performed += 1
    return performed


def run_until(state: SSAState, t_end: float, max_events: int | None = None) -> int:
    """Run until simulation time reaches ``t_end`` or ``max_events`` is reached."""

    performed = 0
    while state.time < t_end and (max_events is None or performed < max_events):
        if not step(state):
            break
        performed += 1
    return performed
