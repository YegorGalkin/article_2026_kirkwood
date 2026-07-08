"""Config-driven scaling experiment runner."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kirkwood_article.experiments.configs import ScalingConfig
from kirkwood_article.sim.observables import density, pair_correlation_1d
from kirkwood_article.sim.ssa_1d import SSAParams, get_positions_1d, initialize, run_events
from kirkwood_article.stats.batch_means import mean_and_se
from kirkwood_article.theory.kirkwood_1d import first_moment_mean_field, second_moment_independent


def _params_for_d(config: ScalingConfig, d_value: float, seed: int) -> SSAParams:
    return SSAParams(
        length=config.length,
        birth_rate=config.birth_rate,
        death_rate=config.death_rate + d_value,
        competition_rate=config.competition_rate,
        birth_sigma=config.birth_sigma,
        death_sigma=config.death_sigma,
        seed=seed,
    )


def run_replicate(
    config: ScalingConfig, d_value: float, seed: int
) -> dict[str, np.ndarray | float | int]:
    """Run one independent replicate and return summary observables."""

    params = _params_for_d(config, d_value, seed)
    state = initialize(params, config.initial_population)
    warmup_performed = run_events(state, config.warmup_events)

    densities: list[float] = []
    pcfs: list[np.ndarray] = []
    performed = 0
    radii: np.ndarray | None = None
    for _ in range(config.samples_per_replicate):
        performed += run_events(state, config.events_per_sample)
        positions = get_positions_1d(state)
        densities.append(density(positions, params.length))
        radii, pcf = pair_correlation_1d(positions, params.length, config.dr, config.r_max)
        pcfs.append(pcf)

    density_mean, density_se = mean_and_se(np.array(densities))
    pcf_mean, pcf_se = mean_and_se(np.vstack(pcfs), axis=0)
    return {
        "seed": seed,
        "d": d_value,
        "warmup_events_performed": warmup_performed,
        "measurement_events_performed": performed,
        "density_mean": float(density_mean),
        "density_se": float(density_se),
        "radii": np.asarray(radii),
        "pcf_mean": pcf_mean,
        "pcf_se": pcf_se,
    }


def run_scaling_grid(config: ScalingConfig) -> dict[str, np.ndarray]:
    """Run all ``d`` values and aggregate uncertainty across independent replicates."""

    d_values = np.asarray(config.d_values, dtype=float)
    density_means = []
    density_ses = []
    pcf_means = []
    pcf_ses = []
    all_seeds = []
    radii = None

    for d_index, d_value in enumerate(d_values):
        replicate_results = []
        for rep in range(config.n_replicates):
            seed = config.seed + 10_000 * d_index + rep
            all_seeds.append(seed)
            replicate_results.append(run_replicate(config, float(d_value), seed))

        replicate_density = np.array([r["density_mean"] for r in replicate_results], dtype=float)
        d_mean, d_se = mean_and_se(replicate_density)
        density_means.append(d_mean)
        density_ses.append(d_se)

        replicate_pcf = np.vstack([r["pcf_mean"] for r in replicate_results])
        p_mean, p_se = mean_and_se(replicate_pcf, axis=0)
        pcf_means.append(p_mean)
        pcf_ses.append(p_se)
        radii = replicate_results[0]["radii"]

    theory_density = np.array(
        [first_moment_mean_field(_params_for_d(config, float(d), config.seed)) for d in d_values]
    )
    theory_pcf = second_moment_independent(
        _params_for_d(config, float(d_values[0]), config.seed), np.asarray(radii)
    )

    return {
        "d_values": d_values,
        "seeds": np.asarray(all_seeds, dtype=int),
        "radii": np.asarray(radii, dtype=float),
        "density_mean": np.asarray(density_means, dtype=float),
        "density_se": np.asarray(density_ses, dtype=float),
        "pcf_mean": np.asarray(pcf_means, dtype=float),
        "pcf_se": np.asarray(pcf_ses, dtype=float),
        "theory_density": theory_density,
        "theory_pcf": theory_pcf,
    }


def save_results(results: dict[str, np.ndarray], output: Path) -> None:
    """Save experiment arrays in a compact NumPy archive."""

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **results)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/minimal_scaling.npz"))
    args = parser.parse_args(argv)
    save_results(run_scaling_grid(ScalingConfig()), args.output)


if __name__ == "__main__":
    main()
