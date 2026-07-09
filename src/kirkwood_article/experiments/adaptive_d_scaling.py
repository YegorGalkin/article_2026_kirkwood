"""Adaptive mean-field ``d``-scaling experiments with coordinate traces."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from kirkwood_article.sim.ssa_1d import SSAParams, SSAState, get_positions_1d, initialize, run_events
from kirkwood_article.stats.batch_means import batch_mean, mean_and_se


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


def _make_state(config: AdaptiveDScalingConfig, d_value: float, seed: int, initial_population: int) -> SSAState:
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
                np.dot(centered[: -(lag + 1)], centered[lag + 1 :])
                / ((n - lag - 1) * variance)
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


def _autocorr_batch_len(values: list[float] | np.ndarray, config: AdaptiveDScalingConfig) -> tuple[int, float]:
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


def _density_stop_reached(diagnostics: dict[str, float | int], config: AdaptiveDScalingConfig) -> bool:
    """Return whether a sequential density stopping boundary is satisfied."""

    mean_density = abs(float(diagnostics["mean_density"]))
    half_width = float(diagnostics["density_ci_half_width"])
    absolute_ok = half_width <= config.density_ci_cutoff
    relative_ok = mean_density > 0.0 and half_width <= config.relative_density_ci_cutoff * mean_density
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


def _ci_half_width_to_se(half_width: np.ndarray, batch_count: np.ndarray) -> np.ndarray:
    """Convert 95% CI half-widths back to standard errors."""

    critical_values = stats.t.ppf(0.975, np.maximum(batch_count - 1, 1))
    return np.divide(
        half_width,
        critical_values,
        out=np.full_like(half_width, np.nan, dtype=float),
        where=critical_values > 0,
    )


def _weighted_zero_bias_fit(
    d_values: np.ndarray, residuals: np.ndarray, standard_errors: np.ndarray, quadratic: bool
) -> dict[str, float | list[float]]:
    """Fit residuals against ``d`` with intercept constrained to zero."""

    columns = [d_values]
    if quadratic:
        columns.append(d_values**2)
    design = np.column_stack(columns)
    weights = np.divide(
        1.0,
        standard_errors**2,
        out=np.ones_like(standard_errors, dtype=float),
        where=standard_errors > 0,
    )
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_residuals = residuals * np.sqrt(weights)
    coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_residuals, rcond=None)
    fitted = design @ coefficients
    weighted_sse = float(np.sum(weights * (residuals - fitted) ** 2))
    degrees_of_freedom = len(residuals) - len(coefficients)
    if degrees_of_freedom > 0:
        covariance = np.linalg.pinv(weighted_design.T @ weighted_design) * (
            weighted_sse / degrees_of_freedom
        )
        coefficient_se = np.sqrt(np.diag(covariance))
        t_values = np.divide(
            coefficients,
            coefficient_se,
            out=np.full_like(coefficients, np.nan, dtype=float),
            where=coefficient_se > 0,
        )
        p_values = 2.0 * stats.t.sf(np.abs(t_values), degrees_of_freedom)
    else:
        coefficient_se = np.full_like(coefficients, np.nan, dtype=float)
        p_values = np.full_like(coefficients, np.nan, dtype=float)
    result: dict[str, float | list[float]] = {
        "coefficients": [float(value) for value in coefficients],
        "standard_errors": [float(value) for value in coefficient_se],
        "p_values": [float(value) for value in p_values],
        "degrees_of_freedom": float(degrees_of_freedom),
        "weighted_sse": weighted_sse,
    }
    if quadratic:
        result["quadratic_coefficient"] = float(coefficients[1])
        result["quadratic_p_value"] = float(p_values[1])
        result["quadratic_significant_95_percent"] = bool(np.isfinite(p_values[1]) and p_values[1] < 0.05)
    return result


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


class CoordinateTraceWriter:
    """Write step-wise particle coordinates into compressed shard files."""

    def __init__(self, output_dir: Path, stride: int = 1) -> None:
        self.output_dir = output_dir
        self.stride = max(int(stride), 1)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, phase: str, step: int, state: SSAState, positions: np.ndarray) -> None:
        if step % self.stride != 0:
            return
        path = self.output_dir / f"{phase}_step_{step:07d}.npz"
        np.savez_compressed(
            path,
            phase=phase,
            step=step,
            time=float(state.time),
            events=int(state.events),
            population=len(positions),
            positions=np.asarray(positions, dtype=float),
        )


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
        if intersected and is_warmup_look and _equilibrium_reached(lower_densities, upper_densities, config):
            break
    else:
        raise TimeoutError(f"d={d_value:.2f} did not equilibrate in {config.max_equilibration_steps} steps")

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
            with (output_dir / f"d_{d_value:.2f}" / "summary.json").open("w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, sort_keys=True)
            return summary

    raise TimeoutError(f"d={d_value:.2f} did not reach CI cutoff in {config.max_measurement_steps} samples")


def run_scaling_grid(config: AdaptiveDScalingConfig, output_dir: Path) -> list[dict[str, float | int | bool]]:
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


def analyze_density_bias(
    summaries: list[dict[str, float | int | bool]],
) -> dict[str, np.ndarray | dict[str, float | list[float]]]:
    """Analyze mean-density residuals relative to the zero-bias mean-field prediction."""

    ordered = sorted(summaries, key=lambda item: float(item["d"]))
    d_values = np.asarray([float(item["d"]) for item in ordered], dtype=float)
    observed = np.asarray([float(item["density_mean"]) for item in ordered], dtype=float)
    expected = np.asarray([float(item["mean_field_density"]) for item in ordered], dtype=float)
    ci_half_width = np.asarray([float(item["density_ci_half_width"]) for item in ordered], dtype=float)
    batch_count = np.asarray([int(item["batch_count"]) for item in ordered], dtype=float)
    standard_errors = _ci_half_width_to_se(ci_half_width, batch_count)
    residuals = observed - expected
    linear = _weighted_zero_bias_fit(d_values, residuals, standard_errors, quadratic=False)
    quadratic = _weighted_zero_bias_fit(d_values, residuals, standard_errors, quadratic=True)
    return {
        "d_values": d_values,
        "observed": observed,
        "expected": expected,
        "ci_half_width": ci_half_width,
        "standard_errors": standard_errors,
        "residuals": residuals,
        "linear": linear,
        "quadratic": quadratic,
    }


def save_summary_plot(output_dir: Path, plot_path: Path | None = None) -> Path:
    """Save a summary plot for the full scaling grid and regression diagnostics."""

    summaries = load_run_summaries(output_dir)
    analysis = analyze_density_bias(summaries)
    d_values = np.asarray(analysis["d_values"], dtype=float)
    observed = np.asarray(analysis["observed"], dtype=float)
    expected = np.asarray(analysis["expected"], dtype=float)
    ci_half_width = np.asarray(analysis["ci_half_width"], dtype=float)
    standard_errors = np.asarray(analysis["standard_errors"], dtype=float)
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
    density_ax.set_title("Adaptive d-scaling density summary")
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
    for d_value, standard_error in zip(d_values, standard_errors, strict=True):
        if not np.isfinite(standard_error) or standard_error <= 0:
            continue
        y_grid = np.linspace(-3.0 * standard_error, 3.0 * standard_error, 121)
        pdf = stats.norm.pdf(y_grid, loc=0.0, scale=standard_error)
        width = 0.003 * pdf / pdf.max()
        residual_ax.plot(d_value + width, y_grid, color="0.6", alpha=0.45, linewidth=0.8)
        residual_ax.plot(d_value - width, y_grid, color="0.6", alpha=0.45, linewidth=0.8)
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
    measurement_time = np.asarray([float(item.get("measurement_time", np.nan)) for item in summaries])
    equilibration_events = np.asarray(
        [int(item.get("equilibration_events", item.get("equilibration_steps", 0))) for item in summaries]
    )
    measurement_events = np.asarray(
        [int(item.get("measurement_events", item.get("measurement_steps", 0))) for item in summaries]
    )
    measurement_steps = np.asarray([int(item["measurement_steps"]) for item in summaries])

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
    events_ax.plot(d_values, measurement_steps, "o-", label="measurement statistic steps")
    events_ax.set_xlabel("death rate d")
    events_ax.set_ylabel("simulation steps / events")
    events_ax.grid(True, alpha=0.25)
    events_ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/adaptive_d_scaling"))
    parser.add_argument("--seed", type=int, default=AdaptiveDScalingConfig.seed)
    parser.add_argument("--length", type=float, default=AdaptiveDScalingConfig.length)
    parser.add_argument("--d-min", type=float, default=0.0)
    parser.add_argument("--d-max", type=float, default=0.1)
    parser.add_argument("--d-step", type=float, default=0.01)
    parser.add_argument("--only-d", type=float, default=None)
    parser.add_argument("--max-equilibration-steps", type=int, default=AdaptiveDScalingConfig.max_equilibration_steps)
    parser.add_argument("--max-measurement-steps", type=int, default=AdaptiveDScalingConfig.max_measurement_steps)
    parser.add_argument("--coordinate-stride", type=int, default=AdaptiveDScalingConfig.coordinate_stride)
    parser.add_argument("--summary-plot", type=Path, default=None)
    parser.add_argument("--convergence-plot", type=Path, default=None)
    parser.add_argument("--summary-plot-only", action="store_true")
    args = parser.parse_args(argv)

    if args.summary_plot_only:
        save_summary_plot(args.output_dir, args.summary_plot)
        save_convergence_diagnostics_plot(args.output_dir, args.convergence_plot)
        return

    if args.only_d is None:
        d_values = tuple(float(x) for x in np.round(np.arange(args.d_min, args.d_max + args.d_step / 2, args.d_step), 2))
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
