import numpy as np

from kirkwood_article.sim.observables import density, pair_correlation_1d


def test_density_counts_particles_per_length():
    assert density(np.array([1.0, 2.0, 3.0]), 6.0) == 0.5


def test_pair_correlation_excludes_self_pairs():
    radii, g = pair_correlation_1d(np.array([0.0, 5.0]), length=10.0, dr=1.0, r_max=2.0)
    assert radii[0] == 0.5
    assert g[0] == 0.0
