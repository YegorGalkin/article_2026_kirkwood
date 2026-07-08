import numpy as np

from kirkwood_article.experiments.configs import ScalingConfig
from kirkwood_article.experiments.scaling import run_scaling_grid


def test_scaling_grid_is_reproducible_for_same_seed():
    config = ScalingConfig(
        n_replicates=1, samples_per_replicate=2, warmup_events=2, events_per_sample=2
    )
    first = run_scaling_grid(config)
    second = run_scaling_grid(config)
    np.testing.assert_allclose(first["density_mean"], second["density_mean"])
