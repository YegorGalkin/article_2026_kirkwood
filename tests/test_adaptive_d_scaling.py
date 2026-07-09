from __future__ import annotations

import json

import numpy as np

from kirkwood_article.experiments.adaptive_d_scaling import (
    AdaptiveDScalingConfig,
    _batch_mean_ci,
    analyze_density_bias,
    mean_field_density,
    save_summary_plot,
)


def test_default_d_grid_and_mean_field_density() -> None:
    config = AdaptiveDScalingConfig()
    assert config.length == 1000.0
    assert config.d_values == tuple(np.round(np.arange(0.0, 0.1000001, 0.01), 2))
    assert mean_field_density(config, 0.1) == 0.9


def test_batch_mean_ci_reaches_absolute_density_cutoff() -> None:
    samples = [0.9 + 0.001 * ((-1) ** i) for i in range(250)]
    mean, half_width, n_batches = _batch_mean_ci(samples, batch_size=5)
    assert n_batches == 50
    assert abs(mean - 0.9) < 1e-12
    assert half_width <= 0.01


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


def test_save_summary_plot_writes_plot_and_regression_json(tmp_path) -> None:
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
            }
        )
    (tmp_path / "summary.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")

    plot_path = save_summary_plot(tmp_path)

    assert plot_path.exists()
    assert (tmp_path / "density_scaling_regression.json").exists()
