import numpy as np

from kirkwood_article.sim.kernels import gaussian_kernel, periodic_distance


def test_periodic_distance_wraps_short_way():
    np.testing.assert_allclose(periodic_distance(0.1, 9.9, 10.0), 0.2)


def test_gaussian_kernel_positive():
    assert gaussian_kernel(0.0, 1.0) > gaussian_kernel(2.0, 1.0) > 0.0
