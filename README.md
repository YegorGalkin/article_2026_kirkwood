# Kirkwood article reproducibility package

This repository contains a minimal, reproducible Python package for comparing
numerical Kirkwood-closure moment predictions against one-dimensional spatial
stochastic simulation algorithm (SSA) experiments.

## Quick start

```bash
uv sync --extra dev
uv run pytest
uv run run-scaling --output results/minimal_scaling.npz
uv run make-figures results/minimal_scaling.npz --output article/figures/scaling.png
```

The package intentionally keeps experiment scripts thin. Reusable code lives in
`src/kirkwood_article/`, while generated stochastic outputs should be written to
`results/` and regenerated from saved metadata rather than committed.
