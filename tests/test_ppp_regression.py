from __future__ import annotations

import os
import time

import numpy as np
import pytest
from scipy import stats

from kirkwood_article.sim.observables import pair_correlation_fft_1d
from kirkwood_article.sim.ssa_1d import SSAParams, SSAState, get_positions_1d, initialize, run_events

RUN_SLOW = os.environ.get("RUN_SLOW_SSA_TESTS") == "1"
SLOW_TIMEOUT_SECONDS = float(os.environ.get("SLOW_SSA_TIMEOUT_SECONDS", "300"))
EFFECTIVE_SAMPLE_COUNT = 50
DOMAIN_LENGTH = 100.0
COMPETITION_RATE = 0.1
PPP_INTENSITY = 1.0 / COMPETITION_RATE
PPP_EQUILIBRIUM_POPULATION = int(DOMAIN_LENGTH * PPP_INTENSITY)


def _mean_pcf_error(positions: np.ndarray, length: float) -> float:
    radii, g_r = pair_correlation_fft_1d(positions, length=length, dr=0.5, r_max=8.0)
    mask = (radii >= 1.0) & np.isfinite(g_r)
    return float(np.abs(g_r[mask].mean() - 1.0))


def test_equal_gaussian_kernel_simulator_runs_near_ppp_smoke():
    params = SSAParams(
        length=DOMAIN_LENGTH,
        birth_rate=1.0,
        death_rate=0.0,
        competition_rate=COMPETITION_RATE,
        birth_sigma=1.0,
        death_sigma=1.0,
        seed=7,
        cell_count=25,
    )
    state = initialize(params, initial_population=PPP_EQUILIBRIUM_POPULATION)
    performed = run_events(state, 1_000)
    positions = get_positions_1d(state)

    assert performed == 1_000
    assert abs(len(positions) / params.length - PPP_INTENSITY) < 1.0
    assert _mean_pcf_error(positions, params.length) < 0.05


def _batch_means(values: list[float], batch_size: int = 3) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    n_batches = len(values_array) // batch_size
    if n_batches == 0:
        return np.array([], dtype=float)
    return values_array[: n_batches * batch_size].reshape(n_batches, batch_size).mean(axis=1)


def _mean_ci_half_width(samples: np.ndarray, confidence: float = 0.95) -> float:
    if len(samples) < 2:
        return float("inf")
    standard_error = stats.sem(samples)
    if not np.isfinite(standard_error):
        return float("inf")
    return float(stats.t.ppf((1.0 + confidence) / 2.0, len(samples) - 1) * standard_error)


def _has_zero_trend_at_95_percent(samples: np.ndarray, max_abs_slope: float = 0.01) -> bool:
    if len(samples) < 3:
        return False
    x_values = np.arange(len(samples), dtype=float)
    result = stats.linregress(x_values, samples)
    if not np.isfinite(result.stderr):
        return False
    slope_ci = abs(result.slope) + stats.t.ppf(0.975, len(samples) - 2) * result.stderr
    return bool(slope_ci < max_abs_slope)


def _poisson_bin_chi_square_pvalue(positions: np.ndarray, length: float, n_bins: int = 100) -> float:
    counts, _ = np.histogram(positions % length, bins=n_bins, range=(0.0, length))
    expected = np.full(n_bins, len(positions) / n_bins, dtype=float)
    return float(stats.chisquare(counts, expected).pvalue)


def _make_ppp_limit_state(initial_population: int, seed: int) -> SSAState:
    params = SSAParams(
        length=DOMAIN_LENGTH,
        birth_rate=1.0,
        death_rate=0.0,
        competition_rate=COMPETITION_RATE,
        birth_sigma=1.0,
        death_sigma=1.0,
        seed=seed,
        cell_count=25,
    )
    return initialize(params, initial_population=initial_population)


def _statistics_step(state: SSAState) -> float:
    n_population = max(len(get_positions_1d(state)), 1)
    performed = run_events(state, n_population)
    assert performed > 0
    return len(get_positions_1d(state)) / state.params.length


def _estimate_required_runtime(
    elapsed: float,
    statistics_steps: int,
    latest_lower_density: float,
) -> float:
    """Estimate wall time needed when the fixed-timeout slow test has not converged."""

    if statistics_steps <= 0:
        return float("inf")
    seconds_per_step = elapsed / statistics_steps
    minimum_steps_for_trend_test = EFFECTIVE_SAMPLE_COUNT * 3
    observed_density_fraction = max(latest_lower_density / PPP_INTENSITY, 0.05)
    lower_start_penalty = 1.0 / min(observed_density_fraction, 1.0)
    projected_steps = max(statistics_steps, minimum_steps_for_trend_test) * lower_start_penalty
    return float(projected_steps * seconds_per_step)


@pytest.mark.slow
def test_equal_gaussian_kernel_dual_start_converges_to_ppp_intensity_10():
    """Long regression mirroring extinction-scaling upper/lower convergence.

    Equal standard normal birth and competition kernels with b=1, d=0, and
    d'=0.1 have the Poisson point process limit with intensity b/d'=10.
    The two coupled-in-parameter simulations start far below and far above that
    intensity, are sampled once per current-population events, and are accepted
    only after batch means indicate no trend and matching density means.
    """

    if not RUN_SLOW:
        pytest.skip("set RUN_SLOW_SSA_TESTS=1 to run the 5-minute SSA convergence regression")

    timeout_seconds = SLOW_TIMEOUT_SECONDS
    started_at = time.monotonic()
    states = [
        _make_ppp_limit_state(initial_population=1, seed=101),
        _make_ppp_limit_state(initial_population=2 * PPP_EQUILIBRIUM_POPULATION, seed=202),
    ]
    density_series: list[list[float]] = [[], []]

    while time.monotonic() - started_at < timeout_seconds:
        for index, state in enumerate(states):
            density_series[index].append(_statistics_step(state))

        batch_series = [_batch_means(series) for series in density_series]
        if all(
            len(batches) >= 50 and _has_zero_trend_at_95_percent(batches[-50:])
            for batches in batch_series
        ):
            lower_recent = batch_series[0][-50:]
            upper_recent = batch_series[1][-50:]
            lower_mean = float(np.mean(lower_recent))
            upper_mean = float(np.mean(upper_recent))
            diff_half_width = float(
                stats.t.ppf(0.975, 98)
                * np.sqrt(np.var(lower_recent, ddof=1) / 50 + np.var(upper_recent, ddof=1) / 50)
            )
            relative_density_gap = (abs(lower_mean - upper_mean) + diff_half_width) / PPP_INTENSITY
            if relative_density_gap < 0.01:
                break
    else:
        elapsed = time.monotonic() - started_at
        lower_batches, upper_batches = (_batch_means(series) for series in density_series)
        lower_latest = float(np.mean(lower_batches[-50:])) if len(lower_batches) >= 50 else float("nan")
        upper_latest = float(np.mean(upper_batches[-50:])) if len(upper_batches) >= 50 else float("nan")
        estimated_runtime = _estimate_required_runtime(
            elapsed=elapsed,
            statistics_steps=len(density_series[0]),
            latest_lower_density=lower_latest,
        )
        pytest.fail(
            "dual-start PPP convergence test exceeded "
            f"{timeout_seconds:.0f}s after {len(density_series[0])} statistics steps "
            f"({elapsed:.1f}s elapsed); latest 50-batch means were "
            f"{lower_latest:.3f} and {upper_latest:.3f}. Estimated required runtime: "
            f"{estimated_runtime / 60.0:.1f} minutes"
        )

    for state, batches in zip(states, (_batch_means(series)[-50:] for series in density_series)):
        mean_density = float(np.mean(batches))
        density_half_width = _mean_ci_half_width(batches)
        positions = get_positions_1d(state)
        assert abs(mean_density - PPP_INTENSITY) + density_half_width < 0.01 * PPP_INTENSITY
        assert _poisson_bin_chi_square_pvalue(positions, state.params.length) > 0.01
        assert _mean_pcf_error(positions, state.params.length) < 0.05
