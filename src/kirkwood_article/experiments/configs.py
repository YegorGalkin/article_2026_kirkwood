"""Configuration objects for reproducible scaling experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScalingConfig:
    """Validated parameters for a minimal 1D scaling experiment."""

    length: float = 100.0
    birth_rate: float = 1.0
    death_rate: float = 0.0
    competition_rate: float = 0.05
    birth_sigma: float = 1.0
    death_sigma: float = 1.0
    d_values: tuple[float, ...] = (0.0, 0.02, 0.04)
    n_replicates: int = 3
    initial_population: int = 20
    warmup_events: int = 100
    samples_per_replicate: int = 20
    events_per_sample: int = 20
    dr: float = 0.5
    r_max: float = 10.0
    seed: int = 12345
    metadata: dict[str, str] = field(default_factory=dict)
