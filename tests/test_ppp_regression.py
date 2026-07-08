import numpy as np

from kirkwood_article.sim.observables import pair_correlation_fft_1d
from kirkwood_article.sim.ssa_1d import SSAParams, get_positions_1d, initialize, run_events


def _mean_pcf_error(positions: np.ndarray, length: float) -> float:
    radii, g_r = pair_correlation_fft_1d(positions, length=length, dr=0.5, r_max=8.0)
    mask = (radii >= 1.0) & np.isfinite(g_r)
    return float(np.abs(g_r[mask].mean() - 1.0))


def test_equal_gaussian_kernel_simulator_runs_near_ppp_smoke():
    params = SSAParams(
        length=100.0,
        birth_rate=1.0,
        death_rate=0.0,
        competition_rate=0.01,
        birth_sigma=1.0,
        death_sigma=1.0,
        seed=7,
        cell_count=25,
    )
    state = initialize(params, initial_population=10_000)
    performed = run_events(state, 1_000)
    positions = get_positions_1d(state)

    assert performed == 1_000
    assert abs(len(positions) / params.length - 100.0) < 1.0
    assert _mean_pcf_error(positions, params.length) < 0.01
