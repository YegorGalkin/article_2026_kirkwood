"""Adaptive mean-field ``d``-scaling experiments with coordinate traces."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

from kirkwood_article.analysis.density_bias import (
    analyze_density_bias,
    ci_half_width_to_se as _ci_half_width_to_se,
    weighted_zero_bias_fit as _weighted_zero_bias_fit,
)
from kirkwood_article.io.coordinate_traces import CoordinateTraceWriter, iter_coordinate_shards
from kirkwood_article.sim.observables import pair_correlation_fft_1d, triplet_correlation_ordered_1d
from kirkwood_article.sim.ssa_1d import (
    SSAParams,
    SSAState,
    get_positions_1d,
    initialize,
    run_events,
)
from kirkwood_article.stats.batch_means import batch_mean, mean_and_se

DEFAULT_DATA_DIR = Path("data")

__all__ = [
    "AdaptiveDScalingConfig",
    "analyze_density_bias",
    "mean_field_density",
    "run_d_value",
    "run_scaling_grid",
    "save_convergence_diagnostics_plot",
    "save_summary_plot",
    "_ci_half_width_to_se",
    "save_pcf_fit_parameter_plot",
    "save_pcf_grid_plot",
    "save_pcf_posthoc_analysis",
    "save_triplet_posthoc_analysis",
    "save_triplet_difference_line_plots",
    "save_triplet_difference_surface_plots",
    "save_triplet_g3_surface_plots",
    "_weighted_zero_bias_fit",
]


@dataclass(frozen=True)
class AdaptiveDScalingConfig:
    """Parameters for adaptive death-rate scaling against mean-field density."""

    length: float = 1000.0
    birth_rate: float = 1.0
    competition_rate: float = 1.0
    birth_sigma: float = 1.0
    death_sigma: float = 1.0
    d_values: tuple[float, ...] = tuple(np.round(np.arange(0.0, 0.1000001, 0.01), 2))
    seed: int = 12345
    cell_count: int = 250
    batch_size: int = 5
    effective_sample_count: int = 50
    density_ci_cutoff: float = 0.005
    relative_density_ci_cutoff: float = 0.05
    alpha_total: float = 0.05
    min_batches: int = 50
    look_interval_batches: int = 5
    min_batch_size: int = 5
    measurement_check_interval_steps: int = 25
    tau_safety_factor: float = 5.0
    autocorr_window_size: int = 1000
    warmup_consecutive_windows: int = 2
    warmup_window_batches: int = 50
    warmup_check_interval_steps: int = 50
    warmup_slope_tol: float = 0.01
    warmup_gap_tol: float = 0.002
    max_equilibration_steps: int = 20_000
    max_measurement_steps: int = 50_000
    coordinate_stride: int = 1


def mean_field_density(config: AdaptiveDScalingConfig, d_value: float) -> float:
    """Return the mean-field density ``(b - d) / d_prime``."""

    return (config.birth_rate - d_value) / config.competition_rate


def _params_for_d(config: AdaptiveDScalingConfig, d_value: float, seed: int) -> SSAParams:
    return SSAParams(
        length=config.length,
        birth_rate=config.birth_rate,
        death_rate=d_value,
        competition_rate=config.competition_rate,
        birth_sigma=config.birth_sigma,
        death_sigma=config.death_sigma,
        seed=seed,
        cell_count=config.cell_count,
    )


def _make_state(
    config: AdaptiveDScalingConfig, d_value: float, seed: int, initial_population: int
) -> SSAState:
    return initialize(_params_for_d(config, d_value, seed), initial_population=initial_population)


def _statistics_step(state: SSAState) -> np.ndarray:
    positions = get_positions_1d(state)
    performed = run_events(state, max(len(positions), 1))
    if performed <= 0:
        raise RuntimeError("SSA backend did not perform any events")
    return get_positions_1d(state)


def _density(positions: np.ndarray, length: float) -> float:
    return float(len(positions) / length)


def _has_zero_trend_at_95_percent(samples: np.ndarray, max_abs_slope: float = 0.001) -> bool:
    if len(samples) < 3:
        return False
    x_values = np.arange(len(samples), dtype=float)
    result = stats.linregress(x_values, samples)
    if not np.isfinite(result.stderr):
        return False
    slope_ci = abs(result.slope) + stats.t.ppf(0.975, len(samples) - 2) * result.stderr
    return bool(slope_ci < max_abs_slope)


def _batch_mean_ci(samples: list[float], batch_size: int) -> tuple[float, float, int]:
    batches = batch_mean(np.asarray(samples, dtype=float), batch_size)
    if len(batches) < 2:
        return float("nan"), float("inf"), len(batches)
    mean, se = mean_and_se(batches)
    half_width = float(stats.t.ppf(0.975, len(batches) - 1) * se)
    return float(mean), half_width, len(batches)


def _autocorr_time(values: list[float] | np.ndarray) -> float:
    """Estimate integrated autocorrelation time for a scalar chain."""

    series = np.asarray(values, dtype=float)
    n = len(series)
    if n < 4:
        return 1.0
    centered = series - float(np.mean(series))
    variance = float(np.dot(centered, centered) / n)
    if not np.isfinite(variance) or variance <= 0.0:
        return 1.0
    max_lag = min(n // 2, 200)
    tau = 1.0
    previous_pair_sum = np.inf
    for lag in range(1, max_lag + 1, 2):
        corr_1 = float(np.dot(centered[:-lag], centered[lag:]) / ((n - lag) * variance))
        if lag + 1 <= max_lag:
            corr_2 = float(
                np.dot(centered[: -(lag + 1)], centered[lag + 1 :]) / ((n - lag - 1) * variance)
            )
        else:
            corr_2 = 0.0
        pair_sum = corr_1 + corr_2
        if not np.isfinite(pair_sum) or pair_sum <= 0.0:
            break
        pair_sum = min(pair_sum, previous_pair_sum)
        tau += 2.0 * pair_sum
        previous_pair_sum = pair_sum
    return float(max(tau, 1.0))


def _autocorr_batch_len(
    values: list[float] | np.ndarray, config: AdaptiveDScalingConfig
) -> tuple[int, float]:
    """Choose a batch length safely larger than the estimated autocorrelation time."""

    recent_values = np.asarray(values, dtype=float)[-config.autocorr_window_size :]
    tau_int = _autocorr_time(recent_values)
    batch_len = max(config.min_batch_size, int(np.ceil(config.tau_safety_factor * tau_int)))
    return batch_len, tau_int


def _autocorr_corrected_mean_ci(
    samples: list[float],
    config: AdaptiveDScalingConfig,
    look_index: int,
) -> dict[str, float | int]:
    """Return an autocorrelation-corrected mean and sequential CI diagnostics."""

    batch_len, tau_int = _autocorr_batch_len(samples, config)
    batches = batch_mean(np.asarray(samples, dtype=float), batch_len)
    n_batches = len(batches)
    if n_batches < 2:
        return {
            "mean_density": float(np.mean(samples)) if samples else float("nan"),
            "mcse": float("inf"),
            "density_ci_half_width": float("inf"),
            "tau_int": tau_int,
            "n_eff": 0.0,
            "batch_len": batch_len,
            "batch_count": n_batches,
            "look_index": look_index,
            "alpha_at_look": float("nan"),
        }
    mean, mcse = mean_and_se(batches)
    alpha_at_look = _alpha_for_look(config.alpha_total, look_index)
    half_width = float(stats.t.ppf(1.0 - alpha_at_look / 2.0, n_batches - 1) * mcse)
    return {
        "mean_density": float(mean),
        "mcse": float(mcse),
        "density_ci_half_width": half_width,
        "tau_int": float(tau_int),
        "n_eff": float(len(samples) / tau_int),
        "batch_len": int(batch_len),
        "batch_count": int(n_batches),
        "look_index": int(look_index),
        "alpha_at_look": float(alpha_at_look),
    }


def _alpha_for_look(alpha_total: float, look_index: int) -> float:
    """Conservative alpha spending sequence for repeated interim looks."""

    if look_index <= 0:
        raise ValueError("look_index must be positive")
    return float(alpha_total / (look_index * (look_index + 1)))


def _density_stop_reached(
    diagnostics: dict[str, float | int], config: AdaptiveDScalingConfig
) -> bool:
    """Return whether a sequential density stopping boundary is satisfied."""

    mean_density = abs(float(diagnostics["mean_density"]))
    half_width = float(diagnostics["density_ci_half_width"])
    absolute_ok = half_width <= config.density_ci_cutoff
    relative_ok = (
        mean_density > 0.0 and half_width <= config.relative_density_ci_cutoff * mean_density
    )
    return bool(absolute_ok or relative_ok)


def _chains_statistically_compatible(
    lower_densities: list[float], upper_densities: list[float], config: AdaptiveDScalingConfig
) -> bool:
    """Return whether lower/upper chains have met within autocorrelation-sized windows."""

    lower_batch_len, _ = _autocorr_batch_len(lower_densities, config)
    upper_batch_len, _ = _autocorr_batch_len(upper_densities, config)
    batch_len = max(lower_batch_len, upper_batch_len)
    lower_batches = batch_mean(np.asarray(lower_densities), batch_len)
    upper_batches = batch_mean(np.asarray(upper_densities), batch_len)
    window_len = max(config.look_interval_batches, 3)
    if len(lower_batches) < window_len or len(upper_batches) < window_len:
        return False
    lower_window = lower_batches[-window_len:]
    upper_window = upper_batches[-window_len:]
    lower_mean, lower_se = mean_and_se(lower_window)
    upper_mean, upper_se = mean_and_se(upper_window)
    tcrit = stats.t.ppf(0.975, window_len - 1)
    chain_gap = abs(float(lower_mean) - float(upper_mean))
    chain_gap_se = float(np.sqrt(lower_se**2 + upper_se**2))
    return bool(chain_gap <= config.warmup_gap_tol + tcrit * chain_gap_se)


def _equilibrium_reached(
    lower_densities: list[float], upper_densities: list[float], config: AdaptiveDScalingConfig
) -> bool:
    min_samples = config.min_batch_size * config.min_batches * config.warmup_consecutive_windows
    if len(lower_densities) < min_samples or len(upper_densities) < min_samples:
        return False
    lower_batch_len, lower_tau = _autocorr_batch_len(lower_densities, config)
    upper_batch_len, upper_tau = _autocorr_batch_len(upper_densities, config)
    batch_len = max(lower_batch_len, upper_batch_len)
    lower_batches = batch_mean(np.asarray(lower_densities), batch_len)
    upper_batches = batch_mean(np.asarray(upper_densities), batch_len)
    window_len = config.warmup_window_batches
    required_batches = config.warmup_consecutive_windows * window_len
    if len(lower_batches) < required_batches or len(upper_batches) < required_batches:
        return False
    if not np.isfinite(lower_tau) or not np.isfinite(upper_tau):
        return False

    for window_index in range(config.warmup_consecutive_windows):
        end = len(lower_batches) - window_index * window_len
        start = end - window_len
        lower_window = lower_batches[start:end]
        upper_window = upper_batches[start:end]
        lower_mean, lower_se = mean_and_se(lower_window)
        upper_mean, upper_se = mean_and_se(upper_window)
        tcrit = stats.t.ppf(0.975, max(window_len - 1, 1))
        chain_gap = abs(float(lower_mean) - float(upper_mean))
        chain_gap_se = float(np.sqrt(lower_se**2 + upper_se**2))
        gap_ok = chain_gap <= config.warmup_gap_tol + tcrit * chain_gap_se
        trend_ok = _has_zero_trend_at_95_percent(
            lower_window, config.warmup_slope_tol
        ) and _has_zero_trend_at_95_percent(upper_window, config.warmup_slope_tol)
        if not (gap_ok and trend_ok):
            return False
    return True


def run_d_value(
    config: AdaptiveDScalingConfig, d_value: float, output_dir: Path, seed: int | None = None
) -> dict[str, float | int | bool]:
    """Run one adaptive scaling experiment and persist all requested coordinates."""

    run_seed = config.seed if seed is None else seed
    mf_density = mean_field_density(config, d_value)
    mf_population = int(round(config.length * mf_density))
    upper_initial = max(2 * mf_population, 1)
    lower_state = _make_state(config, d_value, run_seed, 1)
    upper_state = _make_state(config, d_value, run_seed + 100_000, upper_initial)
    writer = CoordinateTraceWriter(output_dir / f"d_{d_value:.2f}", config.coordinate_stride)

    lower_densities: list[float] = [1 / config.length]
    upper_densities: list[float] = [upper_initial / config.length]
    intersected = False
    restarted_lower = False
    started_at = time.monotonic()

    for step in range(1, config.max_equilibration_steps + 1):
        lower_positions = _statistics_step(lower_state)
        if len(lower_positions) == 0 and not restarted_lower:
            lower_state = _make_state(config, d_value, run_seed + 1, 1)
            lower_positions = get_positions_1d(lower_state)
            restarted_lower = True
        upper_positions = _statistics_step(upper_state)
        lower_density = _density(lower_positions, config.length)
        upper_density = _density(upper_positions, config.length)
        lower_densities.append(lower_density)
        upper_densities.append(upper_density)
        writer.write("equilibration_lower", step, lower_state, lower_positions)
        writer.write("equilibration_upper", step, upper_state, upper_positions)

        if lower_density >= upper_density:
            intersected = True
        is_warmup_look = step % config.warmup_check_interval_steps == 0
        if not intersected and is_warmup_look:
            intersected = _chains_statistically_compatible(lower_densities, upper_densities, config)
        if (
            intersected
            and is_warmup_look
            and _equilibrium_reached(lower_densities, upper_densities, config)
        ):
            break
    else:
        raise TimeoutError(
            f"d={d_value:.2f} did not equilibrate in {config.max_equilibration_steps} steps"
        )

    equilibration_time = float(lower_state.time)
    equilibration_events = int(lower_state.events)
    measurement_densities: list[float] = []
    look_index = 0
    last_look_batch_count = 0
    for measurement_step in range(1, config.max_measurement_steps + 1):
        positions = _statistics_step(lower_state)
        measurement_densities.append(_density(positions, config.length))
        writer.write("measurement", measurement_step, lower_state, positions)
        if (
            measurement_step < config.min_batch_size * config.min_batches
            or measurement_step % config.measurement_check_interval_steps != 0
        ):
            continue
        batch_len, _ = _autocorr_batch_len(measurement_densities, config)
        n_batches = len(batch_mean(np.asarray(measurement_densities, dtype=float), batch_len))
        is_new_look = (
            n_batches >= config.min_batches
            and n_batches % config.look_interval_batches == 0
            and n_batches != last_look_batch_count
        )
        if not is_new_look:
            continue
        look_index += 1
        last_look_batch_count = n_batches
        diagnostics = _autocorr_corrected_mean_ci(measurement_densities, config, look_index)
        if _density_stop_reached(diagnostics, config):
            elapsed = time.monotonic() - started_at
            summary = {
                "d": float(d_value),
                "seed": int(run_seed),
                "mean_field_density": float(mf_density),
                "mean_field_population": int(mf_population),
                "equilibration_steps": int(step),
                "equilibration_time": equilibration_time,
                "equilibration_events": equilibration_events,
                "measurement_steps": int(measurement_step),
                "measurement_time": float(lower_state.time),
                "measurement_events": int(lower_state.events),
                "density_mean": float(diagnostics["mean_density"]),
                "density_ci_half_width": float(diagnostics["density_ci_half_width"]),
                "density_ci_cutoff": float(config.density_ci_cutoff),
                "relative_density_ci_cutoff": float(config.relative_density_ci_cutoff),
                "mcse": float(diagnostics["mcse"]),
                "tau_int": float(diagnostics["tau_int"]),
                "n_eff": float(diagnostics["n_eff"]),
                "batch_len": int(diagnostics["batch_len"]),
                "batch_count": int(diagnostics["batch_count"]),
                "look_index": int(diagnostics["look_index"]),
                "alpha_at_look": float(diagnostics["alpha_at_look"]),
                "intersected": bool(intersected),
                "restarted_lower": bool(restarted_lower),
                "elapsed_seconds": float(elapsed),
            }
            with (output_dir / f"d_{d_value:.2f}" / "summary.json").open(
                "w", encoding="utf-8"
            ) as fh:
                json.dump(summary, fh, indent=2, sort_keys=True)
            return summary

    raise TimeoutError(
        f"d={d_value:.2f} did not reach CI cutoff in {config.max_measurement_steps} samples"
    )


def run_scaling_grid(
    config: AdaptiveDScalingConfig, output_dir: Path
) -> list[dict[str, float | int | bool]]:
    """Run all configured death-rate values and write aggregate metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        run_d_value(config, float(d), output_dir, seed=config.seed + i * 10_000)
        for i, d in enumerate(config.d_values)
    ]
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"config": asdict(config), "runs": summaries}, fh, indent=2, sort_keys=True)
    return summaries


def load_run_summaries(output_dir: Path) -> list[dict[str, float | int | bool]]:
    """Load per-``d`` run summaries from an experiment output directory."""

    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        return list(payload["runs"])
    summaries = []
    for path in sorted(output_dir.glob("d_*/summary.json")):
        with path.open(encoding="utf-8") as fh:
            summaries.append(json.load(fh))
    if not summaries:
        raise FileNotFoundError(f"no run summaries found under {output_dir}")
    return summaries


def save_summary_plot(output_dir: Path, plot_path: Path | None = None) -> Path:
    """Save a summary plot for the full scaling grid and regression diagnostics."""

    summaries = load_run_summaries(output_dir)
    analysis = analyze_density_bias(summaries)
    d_values = np.asarray(analysis["d_values"], dtype=float)
    observed = np.asarray(analysis["observed"], dtype=float)
    expected = np.asarray(analysis["expected"], dtype=float)
    ci_half_width = np.asarray(analysis["ci_half_width"], dtype=float)
    residuals = np.asarray(analysis["residuals"], dtype=float)
    linear = analysis["linear"]
    quadratic = analysis["quadratic"]
    linear_coefficient = float(linear["coefficients"][0])
    linear_density = expected + linear_coefficient * d_values

    if plot_path is None:
        plot_path = output_dir / "density_scaling_summary.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (density_ax, residual_ax) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    density_ax.fill_between(
        d_values,
        observed - ci_half_width,
        observed + ci_half_width,
        color="tab:blue",
        alpha=0.2,
        label="observed 95% CI",
    )
    density_ax.plot(d_values, observed, "o-", color="tab:blue", label="observed mean density")
    density_ax.plot(d_values, expected, "--", color="black", label="mean field (zero bias)")
    density_ax.plot(
        d_values,
        linear_density,
        color="tab:orange",
        label="zero-intercept linear bias fit",
    )
    density_ax.set_ylabel("density")
    density_ax.set_title(
        "Adaptive d-scaling density summary\n"
        f"linear density residual coefficient = {linear_coefficient:.3g}"
    )
    density_ax.grid(True, alpha=0.25)
    density_ax.legend(loc="best")

    residual_ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, label="zero bias")
    residual_ax.errorbar(
        d_values,
        residuals,
        yerr=ci_half_width,
        fmt="o",
        color="tab:blue",
        capsize=3,
        label="observed - mean field",
    )
    residual_ax.plot(
        d_values,
        linear_coefficient * d_values,
        color="tab:orange",
        label="zero-intercept linear fit",
    )
    residual_ax.set_xlabel("death rate d")
    residual_ax.set_ylabel("density residual")
    residual_ax.grid(True, alpha=0.25)
    residual_ax.legend(loc="best")
    residual_ax.text(
        0.02,
        0.98,
        "quadratic p-value = "
        f"{float(quadratic['quadratic_p_value']):.3g}; "
        f"significant: {bool(quadratic['quadratic_significant_95_percent'])}",
        transform=residual_ax.transAxes,
        ha="left",
        va="top",
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    serializable = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in analysis.items()
    }
    with (output_dir / "density_scaling_regression.json").open("w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2, sort_keys=True)
    return plot_path


def save_convergence_diagnostics_plot(output_dir: Path, plot_path: Path | None = None) -> Path:
    """Save convergence time and event-count diagnostics over the ``d`` grid."""

    summaries = sorted(load_run_summaries(output_dir), key=lambda item: float(item["d"]))
    d_values = np.asarray([float(item["d"]) for item in summaries], dtype=float)
    equilibration_time = np.asarray(
        [float(item.get("equilibration_time", np.nan)) for item in summaries]
    )
    measurement_time = np.asarray(
        [float(item.get("measurement_time", np.nan)) for item in summaries]
    )
    equilibration_events = np.asarray(
        [
            int(item.get("equilibration_events", item.get("equilibration_steps", 0)))
            for item in summaries
        ]
    )
    measurement_events = np.asarray(
        [
            int(item.get("measurement_events", item.get("measurement_steps", 0)))
            for item in summaries
        ]
    )
    if plot_path is None:
        plot_path = output_dir / "convergence_diagnostics.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (time_ax, events_ax) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    time_ax.plot(d_values, equilibration_time, "o-", label="equilibration simulation time")
    time_ax.plot(d_values, measurement_time, "o-", label="final simulation time")
    time_ax.set_ylabel("simulation time")
    time_ax.set_title("Adaptive d-scaling convergence diagnostics")
    time_ax.grid(True, alpha=0.25)
    time_ax.legend(loc="best")

    events_ax.plot(d_values, equilibration_events, "o-", label="equilibration events")
    events_ax.plot(d_values, measurement_events, "o-", label="final events")
    events_ax.set_xlabel("death rate d")
    events_ax.set_ylabel("events")
    events_ax.grid(True, alpha=0.25)
    events_ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def _load_output_metadata(
    output_dir: Path,
) -> tuple[list[dict[str, float | int | bool]], dict[str, object]]:
    """Load aggregate summaries plus optional config metadata."""

    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        return list(payload["runs"]), dict(payload.get("config", {}))
    return load_run_summaries(output_dir), {}


def _pointwise_autocorr_corrected_mean_ci(
    samples: np.ndarray, config: AdaptiveDScalingConfig
) -> dict[str, np.ndarray]:
    """Return point-wise autocorrelation-corrected 95% PCF confidence bands."""

    values = np.asarray(samples, dtype=float)
    if values.ndim != 2:
        raise ValueError("samples must have shape (time, radius)")
    n_radii = values.shape[1]
    mean = np.full(n_radii, np.nan, dtype=float)
    half_width = np.full(n_radii, np.inf, dtype=float)
    mcse = np.full(n_radii, np.inf, dtype=float)
    tau_int = np.full(n_radii, np.nan, dtype=float)
    n_eff = np.zeros(n_radii, dtype=float)
    batch_len = np.zeros(n_radii, dtype=int)
    batch_count = np.zeros(n_radii, dtype=int)

    for radius_index in range(n_radii):
        column = values[:, radius_index]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            continue
        tau = _autocorr_time(finite)
        batches_len = max(config.min_batch_size, int(np.ceil(config.tau_safety_factor * tau)))
        batches = batch_mean(finite, batches_len)
        tau_int[radius_index] = tau
        n_eff[radius_index] = finite.size / tau
        batch_len[radius_index] = batches_len
        batch_count[radius_index] = len(batches)
        if len(batches) < 2:
            mean[radius_index] = float(np.mean(finite))
            continue
        mean_value, se = mean_and_se(batches)
        mean[radius_index] = float(mean_value)
        mcse[radius_index] = float(se)
        half_width[radius_index] = float(stats.t.ppf(0.975, len(batches) - 1) * se)

    return {
        "mean": mean,
        "half_width": half_width,
        "lower": mean - half_width,
        "upper": mean + half_width,
        "mcse": mcse,
        "tau_int": tau_int,
        "n_eff": n_eff,
        "batch_len": batch_len,
        "batch_count": batch_count,
    }




def _pointwise_autocorr_corrected_mean_ci_nd(
    samples: np.ndarray, config: AdaptiveDScalingConfig
) -> dict[str, np.ndarray]:
    """Return point-wise autocorrelation-corrected 95% CIs for any sample grid."""

    values = np.asarray(samples, dtype=float)
    if values.ndim < 2:
        raise ValueError("samples must have shape (time, *grid_shape)")
    grid_shape = values.shape[1:]
    flat_result = _pointwise_autocorr_corrected_mean_ci(values.reshape(values.shape[0], -1), config)
    return {
        key: np.asarray(value).reshape(grid_shape)
        for key, value in flat_result.items()
    }


def _lookup_grid_values(radii: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return values at nearest grid locations for target radii."""

    indices = np.searchsorted(radii, targets)
    indices = np.clip(indices, 0, len(radii) - 1)
    previous = np.clip(indices - 1, 0, len(radii) - 1)
    choose_previous = np.abs(radii[previous] - targets) < np.abs(radii[indices] - targets)
    indices[choose_previous] = previous[choose_previous]
    return values[indices]


def save_triplet_posthoc_analysis(output_dir: Path, dr: float = 0.1, r_max: float = 5.0) -> Path:
    """Compute post-equilibrium ordered triplet ``g3`` and closure-difference summaries."""

    summaries, config_payload = _load_output_metadata(output_dir)
    length = float(config_payload.get("length", AdaptiveDScalingConfig.length))
    config_kwargs = {
        k: v for k, v in config_payload.items() if k in AdaptiveDScalingConfig.__dataclass_fields__
    }
    config = AdaptiveDScalingConfig(**config_kwargs)
    target = np.round(np.arange(0.0, r_max + dr / 2.0, dr), 10)
    g2_target = np.round(np.arange(0.0, 2.0 * r_max + dr / 2.0, dr), 10)
    r1_grid, r2_grid = np.meshgrid(target, target, indexing="ij")
    rsum = (r1_grid + r2_grid).ravel()

    d_values = []
    g3_means = []
    g3_half_widths = []
    difference_means = []
    difference_half_widths = []
    difference_mcses = []
    difference_batch_counts = []
    sample_counts = []

    for summary in sorted(summaries, key=lambda item: float(item["d"])):
        d_value = float(summary["d"])
        g3_samples = []
        difference_samples = []
        for shard in iter_coordinate_shards(output_dir / f"d_{d_value:.2f}", phase="measurement"):
            r1_values, r2_values, g3 = triplet_correlation_ordered_1d(
                shard.positions, length=length, dr=dr, r_max=r_max
            )
            radii, g2 = pair_correlation_fft_1d(
                shard.positions, length=length, dr=dr, r_max=2.0 * r_max
            )
            if len(r1_values) != len(target) or not np.allclose(r1_values, target):
                raise ValueError("triplet grid does not match requested target grid")
            g2_on_target = np.interp(g2_target, radii, g2, left=np.nan, right=np.nan)
            g2_r1 = _lookup_grid_values(g2_target, g2_on_target, r1_grid.ravel()).reshape(g3.shape)
            g2_r2 = _lookup_grid_values(g2_target, g2_on_target, r2_grid.ravel()).reshape(g3.shape)
            g2_sum = _lookup_grid_values(g2_target, g2_on_target, rsum).reshape(g3.shape)
            pair_product = g2_r1 * g2_r2 * g2_sum
            g3_samples.append(g3)
            difference_samples.append(g3 - pair_product)
        if not difference_samples:
            continue
        g3_ci = _pointwise_autocorr_corrected_mean_ci_nd(np.stack(g3_samples), config)
        difference_ci = _pointwise_autocorr_corrected_mean_ci_nd(np.stack(difference_samples), config)
        d_values.append(d_value)
        g3_means.append(g3_ci["mean"])
        g3_half_widths.append(g3_ci["half_width"])
        difference_means.append(difference_ci["mean"])
        difference_half_widths.append(difference_ci["half_width"])
        difference_mcses.append(difference_ci["mcse"])
        difference_batch_counts.append(difference_ci["batch_count"])
        sample_counts.append(len(difference_samples))

    mean_difference = np.stack(difference_means)
    half_difference = np.stack(difference_half_widths)
    mean_g3 = np.stack(g3_means)
    half_g3 = np.stack(g3_half_widths)
    payload = {
        "d_values": np.asarray(d_values, dtype=float),
        "r1_values": target,
        "r2_values": target,
        "g2_radii": g2_target,
        "g3_mean": mean_g3,
        "g3_ci_half_width": half_g3,
        "g3_lower": mean_g3 - half_g3,
        "g3_upper": mean_g3 + half_g3,
        "difference_mean": mean_difference,
        "difference_ci_half_width": half_difference,
        "difference_lower": mean_difference - half_difference,
        "difference_upper": mean_difference + half_difference,
        "difference_mcse": np.stack(difference_mcses),
        "difference_batch_count": np.stack(difference_batch_counts),
        "sample_count": np.asarray(sample_counts, dtype=int),
    }
    array_payload = {k: v for k, v in payload.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(output_dir / "triplet_posthoc_analysis.npz", **array_payload)
    json_path = output_dir / "triplet_posthoc_analysis.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_ready(payload), fh, indent=2, sort_keys=True)
    return json_path


def _load_triplet_posthoc(output_dir: Path) -> dict[str, object]:
    path = output_dir / "triplet_posthoc_analysis.json"
    if not path.exists():
        save_triplet_posthoc_analysis(output_dir)
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _positive_triplet_plot_window(
    r1_values: np.ndarray, r2_values: np.ndarray, *arrays: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Return r1/r2 grids and arrays with zero-radius rows/columns removed."""

    r1_mask = r1_values > 0.0
    r2_mask = r2_values > 0.0
    trimmed = [array[np.ix_(r1_mask, r2_mask)] for array in arrays]
    return r1_values[r1_mask], r2_values[r2_mask], trimmed


def save_triplet_g3_surface_plots(output_dir: Path, plot_dir: Path | None = None) -> list[Path]:
    """Save one positive-radius g3 point-estimate surface plot per death-rate value."""

    analysis = _load_triplet_posthoc(output_dir)
    d_values = np.asarray(analysis["d_values"], dtype=float)
    r1_values = np.asarray(analysis["r1_values"], dtype=float)
    r2_values = np.asarray(analysis["r2_values"], dtype=float)
    means = np.asarray(analysis["g3_mean"], dtype=float)
    if plot_dir is None:
        plot_dir = output_dir
    plot_dir.mkdir(parents=True, exist_ok=True)
    positive_values = means[:, 1:, 1:]
    vmax = float(np.nanmax(positive_values)) if np.any(np.isfinite(positive_values)) else 1.0
    paths = []
    for d_value, mean in zip(d_values, means, strict=False):
        r1_plot, r2_plot, (mean_plot,) = _positive_triplet_plot_window(r1_values, r2_values, mean)
        fig, ax = plt.subplots(figsize=(7, 6))
        mesh = ax.pcolormesh(r2_plot, r1_plot, mean_plot, cmap="viridis", vmin=0.0, vmax=vmax, shading="auto")
        ax.set_xlabel("r2")
        ax.set_ylabel("r1")
        ax.set_title(f"g3 point estimate, d = {d_value:.2f}")
        fig.colorbar(mesh, ax=ax, label="g3")
        fig.tight_layout()
        path = plot_dir / f"triplet_g3_surface_d_{d_value:.2f}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


def save_triplet_difference_surface_plots(output_dir: Path, plot_dir: Path | None = None) -> list[Path]:
    """Save one positive-radius closure-difference surface plot per death-rate value."""

    analysis = _load_triplet_posthoc(output_dir)
    d_values = np.asarray(analysis["d_values"], dtype=float)
    r1_values = np.asarray(analysis["r1_values"], dtype=float)
    r2_values = np.asarray(analysis["r2_values"], dtype=float)
    means = np.asarray(analysis["difference_mean"], dtype=float)
    if plot_dir is None:
        plot_dir = output_dir
    plot_dir.mkdir(parents=True, exist_ok=True)
    positive_values = means[:, 1:, 1:]
    vmax = float(np.nanmax(np.abs(positive_values))) if np.any(np.isfinite(positive_values)) else 1.0
    paths = []
    for d_value, mean in zip(d_values, means, strict=False):
        r1_plot, r2_plot, (mean_plot,) = _positive_triplet_plot_window(r1_values, r2_values, mean)
        fig, ax = plt.subplots(figsize=(7, 6))
        mesh = ax.pcolormesh(r2_plot, r1_plot, mean_plot, cmap="bwr", vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_xlabel("r2")
        ax.set_ylabel("r1")
        ax.set_title(f"g3 - g2(r1)g2(r2)g2(r1+r2), d = {d_value:.2f}")
        fig.colorbar(mesh, ax=ax, label="closure difference")
        fig.tight_layout()
        path = plot_dir / f"triplet_difference_surface_d_{d_value:.2f}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


def save_triplet_difference_line_plots(output_dir: Path, plot_path: Path | None = None) -> Path:
    """Save positive-radius closure-difference slice plots with confidence bands."""

    analysis = _load_triplet_posthoc(output_dir)
    d_values = np.asarray(analysis["d_values"], dtype=float)
    r1_values = np.asarray(analysis["r1_values"], dtype=float)
    r2_values = np.asarray(analysis["r2_values"], dtype=float)
    means = np.asarray(analysis["difference_mean"], dtype=float)
    lower = np.asarray(analysis["difference_lower"], dtype=float)
    upper = np.asarray(analysis["difference_upper"], dtype=float)
    r_mask = r1_values > 0.0
    if plot_path is None:
        plot_path = output_dir / "triplet_difference_lines.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    ncols = 4
    nrows = int(np.ceil(len(d_values) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)
    flat_axes = np.atleast_1d(axes).ravel()
    fixed_r2 = (0.1, 0.5, 1.0, 2.0)
    for ax, d_value, mean, lo, hi in zip(flat_axes, d_values, means, lower, upper, strict=False):
        for fixed in fixed_r2:
            index = int(np.argmin(np.abs(r2_values - fixed)))
            if r2_values[index] <= 0.0:
                continue
            label = f"r2={r2_values[index]:.1f}"
            ax.plot(r1_values[r_mask], mean[r_mask, index], label=label)
            ax.fill_between(r1_values[r_mask], lo[r_mask, index], hi[r_mask, index], alpha=0.12)
        diag = np.diag(mean)[r_mask]
        diag_lo = np.diag(lo)[r_mask]
        diag_hi = np.diag(hi)[r_mask]
        ax.plot(r1_values[r_mask], diag, color="black", linestyle="--", label="r1=r2")
        ax.fill_between(r1_values[r_mask], diag_lo, diag_hi, color="black", alpha=0.08)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"d = {d_value:.2f}")
        ax.grid(True, alpha=0.25)
    for ax in flat_axes[len(d_values):]:
        ax.axis("off")
    flat_axes[0].legend(loc="best", fontsize="small")
    fig.supxlabel("r")
    fig.supylabel("g3 - g2(r1)g2(r2)g2(r1+r2)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def _exponential_excess_model(radii: np.ndarray, amplitude: float, lambda_: float) -> np.ndarray:
    return amplitude * np.exp(-lambda_ * radii)


def _weighted_polyfit_with_ci(
    x_values: np.ndarray, y_values: np.ndarray, y_se: np.ndarray, degree: int
) -> dict[str, object]:
    """Fit weighted polynomial and return coefficient covariance diagnostics."""

    finite = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(y_se) & (y_se > 0.0)
    x = x_values[finite]
    y = y_values[finite]
    se = y_se[finite]
    design = np.vstack([x**power for power in range(degree + 1)]).T
    weights = 1.0 / np.square(se)
    xtw = design.T * weights
    xtwx = xtw @ design
    xtwy = xtw @ y
    coefficients = np.linalg.solve(xtwx, xtwy)
    residuals = y - design @ coefficients
    dof = max(len(y) - design.shape[1], 1)
    chi_square = float(np.sum(weights * np.square(residuals)))
    scale = max(chi_square / dof, 1.0)
    covariance = np.linalg.inv(xtwx) * scale
    se_coefficients = np.sqrt(np.diag(covariance))
    tcrit = stats.t.ppf(0.975, dof)
    return {
        "coefficients": coefficients,
        "covariance": covariance,
        "standard_errors": se_coefficients,
        "ci_half_width": tcrit * se_coefficients,
        "chi_square": chi_square,
        "dof": int(dof),
    }


def _weighted_zero_intercept_polyfit_with_ci(
    x_values: np.ndarray, y_values: np.ndarray, y_se: np.ndarray, degree: int
) -> dict[str, object]:
    """Fit weighted zero-intercept polynomial with powers 1..degree."""

    if degree < 1:
        raise ValueError("degree must be at least 1 for zero-intercept polynomial fits")
    finite = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(y_se) & (y_se > 0.0)
    x = x_values[finite]
    y = y_values[finite]
    se = y_se[finite]
    design = np.vstack([x**power for power in range(1, degree + 1)]).T
    weights = 1.0 / np.square(se)
    xtw = design.T * weights
    xtwx = xtw @ design
    xtwy = xtw @ y
    coefficients = np.linalg.solve(xtwx, xtwy)
    residuals = y - design @ coefficients
    dof = max(len(y) - design.shape[1], 1)
    chi_square = float(np.sum(weights * np.square(residuals)))
    scale = max(chi_square / dof, 1.0)
    covariance = np.linalg.inv(xtwx) * scale
    se_coefficients = np.sqrt(np.diag(covariance))
    tcrit = stats.t.ppf(0.975, dof)
    return {
        "coefficients": coefficients,
        "covariance": covariance,
        "standard_errors": se_coefficients,
        "ci_half_width": tcrit * se_coefficients,
        "chi_square": chi_square,
        "dof": int(dof),
    }


def _fit_exponential_pcfs(
    radii: np.ndarray,
    pcf_excess_mean: np.ndarray,
    pcf_excess_mcse: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit PCF - 1 to A exp(-lambda x), excluding the unstable zero bin."""

    d_count = pcf_excess_mean.shape[0]
    amplitude = np.full(d_count, np.nan, dtype=float)
    amplitude_se = np.full(d_count, np.nan, dtype=float)
    lambda_ = np.full(d_count, np.nan, dtype=float)
    lambda_se = np.full(d_count, np.nan, dtype=float)
    fitted = np.full_like(pcf_excess_mean, np.nan, dtype=float)
    fit_mask = radii > 0.0

    for d_index in range(d_count):
        y = pcf_excess_mean[d_index, fit_mask]
        sigma = pcf_excess_mcse[d_index, fit_mask]
        x = radii[fit_mask]
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0.0)
        if np.count_nonzero(finite) < 3:
            finite = np.isfinite(x) & np.isfinite(y)
            sigma_arg = None
        else:
            sigma_arg = sigma[finite]
        if np.count_nonzero(finite) < 3:
            continue
        initial_amplitude = float(y[finite][0])
        initial_lambda = 1.0
        try:
            params, covariance = optimize.curve_fit(
                _exponential_excess_model,
                x[finite],
                y[finite],
                p0=(initial_amplitude, initial_lambda),
                sigma=sigma_arg,
                absolute_sigma=sigma_arg is not None,
                bounds=([-np.inf, 0.0], [np.inf, np.inf]),
                maxfev=20_000,
            )
        except (RuntimeError, ValueError):
            continue
        amplitude[d_index], lambda_[d_index] = params
        errors = np.sqrt(np.diag(covariance))
        amplitude_se[d_index], lambda_se[d_index] = errors
        fitted[d_index, :] = _exponential_excess_model(radii, amplitude[d_index], lambda_[d_index])

    return {
        "amplitude": amplitude,
        "amplitude_se": amplitude_se,
        "lambda": lambda_,
        "lambda_se": lambda_se,
        "fitted_pcf_excess": fitted,
    }


def _test_pcf_fit_hypotheses(
    d_values: np.ndarray,
    amplitude: np.ndarray,
    amplitude_se: np.ndarray,
    lambda_: np.ndarray,
    lambda_se: np.ndarray,
) -> dict[str, object]:
    """Test constant lambda(d) and linear A(d) with weighted regressions."""

    lambda_constant = _weighted_polyfit_with_ci(d_values, lambda_, lambda_se, degree=0)
    lambda_linear = _weighted_polyfit_with_ci(d_values, lambda_, lambda_se, degree=1)
    lambda_slope = float(lambda_linear["coefficients"][1])
    lambda_slope_se = float(lambda_linear["standard_errors"][1])
    lambda_t = lambda_slope / lambda_slope_se if lambda_slope_se > 0.0 else np.nan
    lambda_p = float(2.0 * stats.t.sf(abs(lambda_t), int(lambda_linear["dof"])))

    amplitude_analysis_mask = d_values > 0.0
    amplitude_d_values = d_values[amplitude_analysis_mask]
    amplitude_values = amplitude[amplitude_analysis_mask]
    amplitude_standard_errors = amplitude_se[amplitude_analysis_mask]
    amplitude_linear = _weighted_zero_intercept_polyfit_with_ci(
        amplitude_d_values, amplitude_values, amplitude_standard_errors, degree=1
    )
    amplitude_quadratic = _weighted_zero_intercept_polyfit_with_ci(
        amplitude_d_values, amplitude_values, amplitude_standard_errors, degree=2
    )
    quad = float(amplitude_quadratic["coefficients"][1])
    quad_se = float(amplitude_quadratic["standard_errors"][1])
    quad_t = quad / quad_se if quad_se > 0.0 else np.nan
    quad_p = float(2.0 * stats.t.sf(abs(quad_t), int(amplitude_quadratic["dof"])))
    return {
        "lambda_constant": lambda_constant,
        "lambda_linear": lambda_linear,
        "lambda_slope_p_value": lambda_p,
        "lambda_constant_not_rejected_95_percent": bool(lambda_p >= 0.05),
        "amplitude_analysis_mask": amplitude_analysis_mask,
        "amplitude_analysis_d_values": amplitude_d_values,
        "amplitude_linear": amplitude_linear,
        "amplitude_quadratic": amplitude_quadratic,
        "amplitude_quadratic_p_value": quad_p,
        "amplitude_linear_not_rejected_95_percent": bool(quad_p >= 0.05),
    }


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(v) for v in value]
    return value


def save_pcf_posthoc_analysis(output_dir: Path, dr: float = 0.1, r_max: float = 5.0) -> Path:
    """Compute post-equilibrium PCF - 1 summaries, fits, and hypothesis tests."""

    summaries, config_payload = _load_output_metadata(output_dir)
    length = float(config_payload.get("length", AdaptiveDScalingConfig.length))
    config_kwargs = {
        k: v for k, v in config_payload.items() if k in AdaptiveDScalingConfig.__dataclass_fields__
    }
    config = AdaptiveDScalingConfig(**config_kwargs)
    target_radii = np.round(np.arange(0.0, r_max + dr / 2.0, dr), 10)
    sorted_summaries = sorted(summaries, key=lambda item: float(item["d"]))

    d_values = []
    means = []
    half_widths = []
    mcses = []
    batch_counts = []
    sample_counts = []
    for summary in sorted_summaries:
        d_value = float(summary["d"])
        samples = []
        for shard in iter_coordinate_shards(output_dir / f"d_{d_value:.2f}", phase="measurement"):
            radii, pcf = pair_correlation_fft_1d(shard.positions, length=length, dr=dr, r_max=r_max)
            pcf_excess = pcf - 1.0
            if len(radii) != len(target_radii) or not np.allclose(radii, target_radii):
                finite = np.isfinite(radii) & np.isfinite(pcf_excess)
                pcf_excess = np.interp(
                    target_radii, radii[finite], pcf_excess[finite], left=np.nan, right=np.nan
                )
            samples.append(pcf_excess)
        if not samples:
            continue
        ci = _pointwise_autocorr_corrected_mean_ci(np.vstack(samples), config)
        d_values.append(d_value)
        means.append(ci["mean"])
        half_widths.append(ci["half_width"])
        mcses.append(ci["mcse"])
        batch_counts.append(ci["batch_count"])
        sample_counts.append(len(samples))

    d_array = np.asarray(d_values, dtype=float)
    mean_array = np.vstack(means)
    half_width_array = np.vstack(half_widths)
    mcse_array = np.vstack(mcses)
    fits = _fit_exponential_pcfs(target_radii, mean_array, mcse_array)
    tests = _test_pcf_fit_hypotheses(
        d_array, fits["amplitude"], fits["amplitude_se"], fits["lambda"], fits["lambda_se"]
    )
    payload = {
        "d_values": d_array,
        "radii": target_radii,
        "pcf_excess_mean": mean_array,
        "pcf_excess_ci_half_width": half_width_array,
        "pcf_excess_mcse": mcse_array,
        "pcf_excess_lower": mean_array - half_width_array,
        "pcf_excess_upper": mean_array + half_width_array,
        "batch_count": np.vstack(batch_counts),
        "sample_count": np.asarray(sample_counts, dtype=int),
        "fits": fits,
        "hypothesis_tests": tests,
    }
    array_payload = {k: v for k, v in payload.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(output_dir / "pcf_posthoc_analysis.npz", **array_payload)
    json_path = output_dir / "pcf_posthoc_analysis.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_ready(payload), fh, indent=2, sort_keys=True)
    return json_path


def _load_pcf_posthoc(output_dir: Path) -> dict[str, object]:
    path = output_dir / "pcf_posthoc_analysis.json"
    if not path.exists():
        save_pcf_posthoc_analysis(output_dir)
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def save_pcf_grid_plot(output_dir: Path, plot_path: Path | None = None) -> Path:
    """Save a 4 x 3 grid of PCF - 1 curves with point-wise confidence bands."""

    analysis = _load_pcf_posthoc(output_dir)
    d_values = np.asarray(analysis["d_values"], dtype=float)
    radii = np.asarray(analysis["radii"], dtype=float)
    means = np.asarray(analysis["pcf_excess_mean"], dtype=float)
    lower = np.asarray(analysis["pcf_excess_lower"], dtype=float)
    upper = np.asarray(analysis["pcf_excess_upper"], dtype=float)
    fitted = np.asarray(analysis["fits"]["fitted_pcf_excess"], dtype=float)
    if plot_path is None:
        plot_path = output_dir / "pcf_grid.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True, sharey=True)
    flat_axes = axes.ravel()
    for axis, d_value, mean, lo, hi, fit in zip(
        flat_axes, d_values, means, lower, upper, fitted, strict=False
    ):
        axis.fill_between(radii, lo, hi, color="tab:blue", alpha=0.2, label="point-wise 95% CI")
        axis.plot(radii, mean, color="tab:blue", label="PCF - 1")
        axis.plot(
            radii[radii > 0.0],
            fit[radii > 0.0],
            color="tab:orange",
            linestyle="--",
            label="exp fit",
        )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title(f"d = {d_value:.2f}")
        axis.grid(True, alpha=0.25)
    for axis in flat_axes[len(d_values) :]:
        axis.axis("off")
    flat_axes[0].legend(loc="best", fontsize="small")
    fig.supxlabel("distance x")
    fig.supylabel("PCF(x) - 1")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def save_pcf_fit_parameter_plot(output_dir: Path, plot_path: Path | None = None) -> Path:
    """Save fitted exponential amplitude and decay-rate summaries versus d."""

    analysis = _load_pcf_posthoc(output_dir)
    d_values = np.asarray(analysis["d_values"], dtype=float)
    fits = analysis["fits"]
    amplitude = np.asarray(fits["amplitude"], dtype=float)
    amplitude_se = np.asarray(fits["amplitude_se"], dtype=float)
    lambda_ = np.asarray(fits["lambda"], dtype=float)
    lambda_se = np.asarray(fits["lambda_se"], dtype=float)
    tests = analysis["hypothesis_tests"]
    if plot_path is None:
        plot_path = output_dir / "pcf_fit_parameters.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    d_grid = np.linspace(float(np.min(d_values)), float(np.max(d_values)), 200)
    lambda_const = float(tests["lambda_constant"]["coefficients"][0])
    lambda_band = float(tests["lambda_constant"]["ci_half_width"][0])
    amplitude_analysis_d_values = np.asarray(tests["amplitude_analysis_d_values"], dtype=float)
    amp_grid = np.linspace(
        float(np.min(amplitude_analysis_d_values)), float(np.max(amplitude_analysis_d_values)), 200
    )
    amp_coef = np.asarray(tests["amplitude_linear"]["coefficients"], dtype=float)
    amp_cov = np.asarray(tests["amplitude_linear"]["covariance"], dtype=float)
    design = amp_grid[:, np.newaxis]
    amp_line = design @ amp_coef
    amp_band = stats.t.ppf(0.975, int(tests["amplitude_linear"]["dof"])) * np.sqrt(
        np.sum((design @ amp_cov) * design, axis=1)
    )
    fig, (lambda_ax, amp_ax) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    lambda_ax.errorbar(d_values, lambda_, yerr=1.96 * lambda_se, fmt="o", capsize=3)
    lambda_ax.plot(d_grid, np.full_like(d_grid, lambda_const), color="tab:orange")
    lambda_ax.fill_between(
        d_grid,
        lambda_const - lambda_band,
        lambda_const + lambda_band,
        color="tab:orange",
        alpha=0.2,
    )
    lambda_ax.set_ylabel("lambda")
    lambda_ax.set_title(f"constant lambda p = {float(tests['lambda_slope_p_value']):.3g}")
    lambda_ax.grid(True, alpha=0.25)
    amp_ax.errorbar(d_values, amplitude, yerr=1.96 * amplitude_se, fmt="o", capsize=3)
    amp_ax.plot(amp_grid, amp_line, color="tab:orange")
    amp_ax.fill_between(
        amp_grid, amp_line - amp_band, amp_line + amp_band, color="tab:orange", alpha=0.2
    )
    amp_ax.set_xlabel("death rate d")
    amp_ax.set_ylabel("A")
    amp_ax.set_title(
        f"zero-intercept linear A(d), d>0 quadratic p = {float(tests['amplitude_quadratic_p_value']):.3g}"
    )
    amp_ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR / "adaptive_d_scaling")
    parser.add_argument("--seed", type=int, default=AdaptiveDScalingConfig.seed)
    parser.add_argument("--length", type=float, default=AdaptiveDScalingConfig.length)
    parser.add_argument("--d-min", type=float, default=0.0)
    parser.add_argument("--d-max", type=float, default=0.1)
    parser.add_argument("--d-step", type=float, default=0.01)
    parser.add_argument("--only-d", type=float, default=None)
    parser.add_argument(
        "--max-equilibration-steps",
        type=int,
        default=AdaptiveDScalingConfig.max_equilibration_steps,
    )
    parser.add_argument(
        "--max-measurement-steps", type=int, default=AdaptiveDScalingConfig.max_measurement_steps
    )
    parser.add_argument(
        "--coordinate-stride", type=int, default=AdaptiveDScalingConfig.coordinate_stride
    )
    parser.add_argument("--summary-plot", type=Path, default=None)
    parser.add_argument("--convergence-plot", type=Path, default=None)
    parser.add_argument("--summary-plot-only", action="store_true")
    parser.add_argument("--pcf-posthoc-only", action="store_true")
    parser.add_argument("--triplet-posthoc-only", action="store_true")
    parser.add_argument("--triplet-dr", type=float, default=0.1)
    parser.add_argument("--triplet-r-max", type=float, default=5.0)
    args = parser.parse_args(argv)

    if args.pcf_posthoc_only:
        save_pcf_posthoc_analysis(args.output_dir)
        save_pcf_grid_plot(args.output_dir)
        save_pcf_fit_parameter_plot(args.output_dir)
        return

    if args.triplet_posthoc_only:
        save_triplet_posthoc_analysis(args.output_dir, dr=args.triplet_dr, r_max=args.triplet_r_max)
        save_triplet_g3_surface_plots(args.output_dir)
        save_triplet_difference_surface_plots(args.output_dir)
        save_triplet_difference_line_plots(args.output_dir)
        return

    if args.summary_plot_only:
        save_summary_plot(args.output_dir, args.summary_plot)
        save_convergence_diagnostics_plot(args.output_dir, args.convergence_plot)
        return

    if args.only_d is None:
        d_values = tuple(
            float(x)
            for x in np.round(np.arange(args.d_min, args.d_max + args.d_step / 2, args.d_step), 2)
        )
    else:
        d_values = (float(args.only_d),)
    config = AdaptiveDScalingConfig(
        length=args.length,
        d_values=d_values,
        seed=args.seed,
        max_equilibration_steps=args.max_equilibration_steps,
        max_measurement_steps=args.max_measurement_steps,
        coordinate_stride=args.coordinate_stride,
    )
    run_scaling_grid(config, args.output_dir)
    save_summary_plot(args.output_dir, args.summary_plot)
    save_convergence_diagnostics_plot(args.output_dir, args.convergence_plot)


if __name__ == "__main__":
    main()
