"""Generate a quick visual check of dual-start PPP population convergence."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from kirkwood_article.sim.ssa_1d import SSAParams, SSAState, get_positions_1d, initialize, run_events

DOMAIN_LENGTH = 100.0
BIRTH_RATE = 1.0
DEATH_RATE = 0.0
COMPETITION_RATE = 0.1
PPP_INTENSITY = BIRTH_RATE / COMPETITION_RATE
PPP_EQUILIBRIUM_POPULATION = int(DOMAIN_LENGTH * PPP_INTENSITY)
BATCH_SIZE = 3
EFFECTIVE_SAMPLE_COUNT = 50
MAX_SECONDS = 300.0


def _batch_means(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    n_batches = len(values_array) // BATCH_SIZE
    if n_batches == 0:
        return np.array([], dtype=float)
    return values_array[: n_batches * BATCH_SIZE].reshape(n_batches, BATCH_SIZE).mean(axis=1)


def _has_zero_trend_at_95_percent(samples: np.ndarray, max_abs_slope: float = 0.01) -> bool:
    if len(samples) < 3:
        return False
    x_values = np.arange(len(samples), dtype=float)
    result = stats.linregress(x_values, samples)
    if not np.isfinite(result.stderr):
        return False
    slope_ci = abs(result.slope) + stats.t.ppf(0.975, len(samples) - 2) * result.stderr
    return bool(slope_ci < max_abs_slope)


def _make_state(initial_population: int, seed: int) -> SSAState:
    params = SSAParams(
        length=DOMAIN_LENGTH,
        birth_rate=BIRTH_RATE,
        death_rate=DEATH_RATE,
        competition_rate=COMPETITION_RATE,
        birth_sigma=1.0,
        death_sigma=1.0,
        seed=seed,
        cell_count=25,
    )
    return initialize(params, initial_population=initial_population)


def _statistics_step(state: SSAState) -> int:
    n_population = max(len(get_positions_1d(state)), 1)
    performed = run_events(state, n_population)
    if performed <= 0:
        msg = "SSA backend did not perform any events"
        raise RuntimeError(msg)
    return len(get_positions_1d(state))


def _has_matched_equilibrium(lower: list[int], upper: list[int]) -> bool:
    lower_density = [population / DOMAIN_LENGTH for population in lower]
    upper_density = [population / DOMAIN_LENGTH for population in upper]
    lower_batches = _batch_means(lower_density)
    upper_batches = _batch_means(upper_density)
    if len(lower_batches) < EFFECTIVE_SAMPLE_COUNT or len(upper_batches) < EFFECTIVE_SAMPLE_COUNT:
        return False
    lower_recent = lower_batches[-EFFECTIVE_SAMPLE_COUNT:]
    upper_recent = upper_batches[-EFFECTIVE_SAMPLE_COUNT:]
    if not _has_zero_trend_at_95_percent(lower_recent):
        return False
    if not _has_zero_trend_at_95_percent(upper_recent):
        return False
    lower_mean = float(np.mean(lower_recent))
    upper_mean = float(np.mean(upper_recent))
    diff_half_width = float(
        stats.t.ppf(0.975, 98)
        * np.sqrt(
            np.var(lower_recent, ddof=1) / EFFECTIVE_SAMPLE_COUNT
            + np.var(upper_recent, ddof=1) / EFFECTIVE_SAMPLE_COUNT
        )
    )
    relative_density_gap = (abs(lower_mean - upper_mean) + diff_half_width) / PPP_INTENSITY
    return relative_density_gap < 0.01


def collect_population_dynamics() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run lower/upper starts until the shared equilibrium criterion is met."""

    states = [
        _make_state(initial_population=1, seed=101),
        _make_state(initial_population=2 * PPP_EQUILIBRIUM_POPULATION, seed=202),
    ]
    populations: list[list[int]] = [[1], [2 * PPP_EQUILIBRIUM_POPULATION]]
    started_at = time.monotonic()

    while time.monotonic() - started_at < MAX_SECONDS:
        for index, state in enumerate(states):
            populations[index].append(_statistics_step(state))
        if _has_matched_equilibrium(populations[0], populations[1]):
            elapsed = time.monotonic() - started_at
            steps = np.arange(len(populations[0]), dtype=int)
            return steps, np.asarray(populations[0]), np.asarray(populations[1]), elapsed

    msg = f"population dynamics did not equilibrate within {MAX_SECONDS:.0f}s"
    raise TimeoutError(msg)


def plot_population_dynamics(output: Path) -> None:
    """Save the lower/upper population dynamics plot."""

    steps, lower, upper, elapsed = collect_population_dynamics()
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, lower, label="lower start: N=1", color="tab:blue")
    ax.plot(steps, upper, label=f"upper start: N={2 * PPP_EQUILIBRIUM_POPULATION}", color="tab:orange")
    ax.axhline(
        PPP_EQUILIBRIUM_POPULATION,
        color="black",
        linestyle="--",
        linewidth=1.25,
        label=f"PPP mean N={PPP_EQUILIBRIUM_POPULATION}",
    )
    ax.set_xlabel("statistics step (current-population events per step)")
    ax.set_ylabel("population N")
    ax.set_title(f"Dual-start SSA convergence to PPP equilibrium ({elapsed:.1f}s)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    plot_population_dynamics(Path("article/figures/ppp_population_dynamics.png"))


if __name__ == "__main__":
    main()
