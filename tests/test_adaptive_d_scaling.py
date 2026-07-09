from __future__ import annotations

import json

import numpy as np

from kirkwood_article.experiments.adaptive_d_scaling import (
    AdaptiveDScalingConfig,
    _alpha_for_look,
    _autocorr_corrected_mean_ci,
    _batch_mean_ci,
    _density_stop_reached,
    analyze_density_bias,
    mean_field_density,
    save_convergence_diagnostics_plot,
    save_summary_plot,
)


def test_default_d_grid_and_mean_field_density() -> None:
    config = AdaptiveDScalingConfig()
    assert config.length == 1000.0
    assert config.density_ci_cutoff == 0.005
    assert config.relative_density_ci_cutoff == 0.05
    assert config.d_values == tuple(np.round(np.arange(0.0, 0.1000001, 0.01), 2))
    assert mean_field_density(config, 0.1) == 0.9


def test_batch_mean_ci_reaches_absolute_density_cutoff() -> None:
    samples = [0.9 + 0.001 * ((-1) ** i) for i in range(250)]
    mean, half_width, n_batches = _batch_mean_ci(samples, batch_size=5)
    assert n_batches == 50
    assert abs(mean - 0.9) < 1e-12
    assert half_width <= 0.01


def test_sequential_autocorr_stop_uses_alpha_spending() -> None:
    config = AdaptiveDScalingConfig()
    samples = [1.0 + 0.0001 * ((-1) ** i) for i in range(300)]

    diagnostics = _autocorr_corrected_mean_ci(samples, config, look_index=2)

    assert diagnostics["alpha_at_look"] == _alpha_for_look(config.alpha_total, 2)
    assert diagnostics["batch_count"] >= config.min_batches
    assert _density_stop_reached(diagnostics, config) is True


def test_density_bias_analysis_detects_quadratic_term() -> None:
    summaries = []
    for d_value in np.linspace(0.01, 0.1, 10):
        expected = 1.0 - d_value
        summaries.append(
            {
                "d": float(d_value),
                "density_mean": float(expected + 0.5 * d_value**2),
                "mean_field_density": float(expected),
                "density_ci_half_width": 0.001,
                "batch_count": 50,
            }
        )

    analysis = analyze_density_bias(summaries)

    assert analysis["quadratic"]["quadratic_coefficient"] > 0.0
    assert analysis["quadratic"]["quadratic_significant_95_percent"] is True


def test_summary_and_convergence_plots_write_outputs(tmp_path) -> None:
    runs = []
    for d_value in np.linspace(0.0, 0.1, 11):
        expected = 1.0 - d_value
        runs.append(
            {
                "d": float(d_value),
                "density_mean": float(expected + 0.02 * d_value),
                "mean_field_density": float(expected),
                "density_ci_half_width": 0.01,
                "batch_count": 50,
                "equilibration_time": float(10.0 + d_value),
                "measurement_time": float(20.0 + d_value),
                "equilibration_events": int(1000 + 10 * d_value),
                "measurement_events": int(2000 + 10 * d_value),
                "measurement_steps": 250,
            }
        )
    (tmp_path / "summary.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")

    plot_path = save_summary_plot(tmp_path)
    convergence_path = save_convergence_diagnostics_plot(tmp_path)

    assert plot_path.exists()
    assert convergence_path.exists()
    assert (tmp_path / "density_scaling_regression.json").exists()
