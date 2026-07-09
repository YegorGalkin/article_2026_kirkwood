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
    density_ci_cutoff: float = 0.01
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
    if len(lower_densities) < config.batch_size * config.effective_sample_count:
        return False
    lower_batches = batch_mean(np.asarray(lower_densities), config.batch_size)
    upper_batches = batch_mean(np.asarray(upper_densities), config.batch_size)
    if len(lower_batches) < config.effective_sample_count or len(upper_batches) < config.effective_sample_count:
        return False
    return _has_zero_trend_at_95_percent(lower_batches[-config.effective_sample_count :]) and _has_zero_trend_at_95_percent(
        upper_batches[-config.effective_sample_count :]
    )


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
        if intersected and _equilibrium_reached(lower_densities, upper_densities, config):
            break
    else:
        raise TimeoutError(f"d={d_value:.2f} did not equilibrate in {config.max_equilibration_steps} steps")

    measurement_densities: list[float] = []
    for measurement_step in range(1, config.max_measurement_steps + 1):
        positions = _statistics_step(lower_state)
        measurement_densities.append(_density(positions, config.length))
        writer.write("measurement", measurement_step, lower_state, positions)
        mean_density, half_width, n_batches = _batch_mean_ci(measurement_densities, config.batch_size)
        if n_batches >= config.effective_sample_count and half_width <= config.density_ci_cutoff:
            elapsed = time.monotonic() - started_at
            summary = {
                "d": float(d_value),
                "seed": int(run_seed),
                "mean_field_density": float(mf_density),
                "mean_field_population": int(mf_population),
                "equilibration_steps": int(step),
                "measurement_steps": int(measurement_step),
                "density_mean": float(mean_density),
                "density_ci_half_width": float(half_width),
                "density_ci_cutoff": float(config.density_ci_cutoff),
                "batch_count": int(n_batches),
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
    parser.add_argument("--summary-plot-only", action="store_true")
    args = parser.parse_args(argv)

    if args.summary_plot_only:
        save_summary_plot(args.output_dir, args.summary_plot)
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


if __name__ == "__main__":
    main()
