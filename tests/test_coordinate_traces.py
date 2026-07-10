from __future__ import annotations

import numpy as np

from kirkwood_article.io.coordinate_traces import iter_coordinate_shards, load_coordinate_shard


def test_coordinate_shard_round_trips_metadata_and_positions(tmp_path):
    path = tmp_path / "measurement_step_0000001.npz"
    positions = np.array([0.25, 1.5, 3.75])
    np.savez_compressed(
        path,
        phase="measurement",
        step=1,
        time=2.5,
        events=10,
        population=len(positions),
        positions=positions,
    )

    shard = load_coordinate_shard(path)

    assert shard.phase == "measurement"
    assert shard.step == 1
    assert shard.time == 2.5
    assert shard.events == 10
    assert shard.population == 3
    np.testing.assert_array_equal(shard.positions, positions)
    assert shard.path == path


def test_iter_coordinate_shards_filters_by_phase(tmp_path):
    np.savez_compressed(
        tmp_path / "measurement_step_0000002.npz",
        phase="measurement",
        step=2,
        time=1.0,
        events=2,
        population=1,
        positions=np.array([0.0]),
    )
    np.savez_compressed(
        tmp_path / "equilibration_lower_step_0000001.npz",
        phase="equilibration_lower",
        step=1,
        time=0.5,
        events=1,
        population=1,
        positions=np.array([0.0]),
    )

    shards = list(iter_coordinate_shards(tmp_path, phase="measurement"))

    assert [shard.phase for shard in shards] == ["measurement"]
    assert [shard.step for shard in shards] == [2]
