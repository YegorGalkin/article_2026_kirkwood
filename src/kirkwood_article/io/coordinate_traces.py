"""Read and write persisted one-dimensional coordinate trace shards."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kirkwood_article.sim.ssa_1d import SSAState


@dataclass(frozen=True)
class CoordinateShard:
    """One persisted coordinate trace sample and its simulation metadata."""

    phase: str
    step: int
    time: float
    events: int
    population: int
    positions: np.ndarray
    path: Path | None = None


class CoordinateTraceWriter:
    """Write step-wise particle coordinates into compressed shard files."""

    def __init__(self, output_dir: Path, stride: int = 1) -> None:
        self.output_dir = output_dir
        self.stride = max(int(stride), 1)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, phase: str, step: int, state: SSAState, positions: np.ndarray) -> None:
        """Persist coordinates for a sampled simulation step when the stride permits it."""

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


def load_coordinate_shard(path: Path) -> CoordinateShard:
    """Load one compressed coordinate trace shard."""

    with np.load(path, allow_pickle=False) as data:
        return CoordinateShard(
            phase=str(data["phase"]),
            step=int(data["step"]),
            time=float(data["time"]),
            events=int(data["events"]),
            population=int(data["population"]),
            positions=np.asarray(data["positions"], dtype=float),
            path=path,
        )


def iter_coordinate_shards(root: Path, phase: str | None = None) -> Iterator[CoordinateShard]:
    """Yield coordinate shards under ``root``, optionally filtering by phase prefix."""

    pattern = "*_step_*.npz" if phase is None else f"{phase}_step_*.npz"
    for path in sorted(root.glob(pattern)):
        yield load_coordinate_shard(path)
