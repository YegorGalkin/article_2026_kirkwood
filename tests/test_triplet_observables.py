import numpy as np

from kirkwood_article.sim.observables import triplet_correlation_ordered_1d
from kirkwood_article.experiments.adaptive_d_scaling import (
    AdaptiveDScalingConfig,
    _pointwise_autocorr_corrected_mean_ci_nd,
)


def test_triplet_correlation_shape_and_self_exclusion():
    r1, r2, g3 = triplet_correlation_ordered_1d(
        np.array([0.0, 1.0, 2.0]), length=10.0, dr=1.0, r_max=2.0
    )

    assert np.allclose(r1, [0.0, 1.0, 2.0])
    assert np.allclose(r2, [0.0, 1.0, 2.0])
    assert g3.shape == (3, 3)
    assert g3[0, 0] == 0.0
    assert g3[1, 1] > 0.0


def test_triplet_correlation_requires_three_particles():
    _, _, g3 = triplet_correlation_ordered_1d(
        np.array([0.0, 1.0]), length=10.0, dr=1.0, r_max=2.0
    )

    assert np.isnan(g3).all()


def test_pointwise_autocorr_ci_nd_preserves_grid_shape():
    samples = np.arange(4 * 2 * 3, dtype=float).reshape(4, 2, 3)
    config = AdaptiveDScalingConfig(min_batch_size=1)

    ci = _pointwise_autocorr_corrected_mean_ci_nd(samples, config)

    assert ci["mean"].shape == (2, 3)
    assert ci["batch_count"].shape == (2, 3)


def test_save_triplet_posthoc_analysis_writes_outputs(tmp_path):
    import json

    from kirkwood_article.experiments.adaptive_d_scaling import (
        save_triplet_difference_line_plots,
        save_triplet_difference_surface_plots,
        save_triplet_g3_surface_plots,
        save_triplet_posthoc_analysis,
    )

    output_dir = tmp_path / "run"
    d_dir = output_dir / "d_0.00"
    d_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {"length": 20.0, "min_batch_size": 1},
                "runs": [{"d": 0.0}],
            }
        ),
        encoding="utf-8",
    )
    for step in range(1, 5):
        np.savez_compressed(
            d_dir / f"measurement_step_{step:07d}.npz",
            phase="measurement",
            step=step,
            time=float(step),
            events=step,
            population=5,
            positions=np.array([0.0, 1.0, 2.0, 4.0, 8.0]) + 0.01 * step,
        )

    json_path = save_triplet_posthoc_analysis(output_dir, dr=1.0, r_max=2.0)

    assert json_path.exists()
    assert (output_dir / "triplet_posthoc_analysis.npz").exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["r1_values"] == [0.0, 1.0, 2.0]
    assert payload["r2_values"] == [0.0, 1.0, 2.0]
    assert payload["sample_count"] == [4]
    assert "difference_mean" in payload
    assert not any(key.startswith("log_") for key in payload)
    assert not any("imputation" in key for key in payload)
    assert save_triplet_g3_surface_plots(output_dir)[0].name == "triplet_g3_surface_d_0.00.png"
    assert (output_dir / "triplet_g3_surface_d_0.00.png").exists()
    assert (
        save_triplet_difference_surface_plots(output_dir)[0].name
        == "triplet_difference_surface_d_0.00.png"
    )
    assert save_triplet_difference_line_plots(output_dir).name == "triplet_difference_lines.png"
