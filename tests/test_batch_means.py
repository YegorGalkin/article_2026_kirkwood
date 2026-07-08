import numpy as np

from kirkwood_article.stats.batch_means import batch_mean, mean_and_se


def test_batch_mean_uses_complete_batches():
    np.testing.assert_allclose(batch_mean(np.arange(5), 2), np.array([0.5, 2.5]))


def test_mean_and_se_uses_unbiased_standard_error():
    mean, se = mean_and_se(np.array([1.0, 3.0]))
    assert mean == 2.0
    np.testing.assert_allclose(se, 1.0)
