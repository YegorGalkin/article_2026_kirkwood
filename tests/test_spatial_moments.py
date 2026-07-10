from __future__ import annotations

import numpy as np

from kirkwood_article.analysis.spatial_moments import summarize_spatial_moments


def test_summarize_spatial_moments_can_use_saved_coordinates():
    summary = summarize_spatial_moments(
        np.array([0.0, 5.0]), length=10.0, dr=1.0, r_max=2.0
    )

    assert summary.first.density == 0.2
    assert summary.first.population == 2
    assert summary.second is not None
    assert summary.second.radii[0] == 0.0
    assert summary.second.values[0] == 0.0
