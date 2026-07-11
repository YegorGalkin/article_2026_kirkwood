# Kirkwood article reproducibility package

This repository contains a minimal, reproducible Python package for comparing
numerical Kirkwood-closure moment predictions against one-dimensional spatial
stochastic simulation algorithm (SSA) experiments.

## Quick start

```bash
uv sync --extra dev
uv run pytest
uv run run-scaling --output data/minimal_scaling.npz
uv run make-figures data/minimal_scaling.npz --output data/figures/scaling.png
```

The package uses `uv run` console entry points for reproducible execution. Reusable
code lives in `src/kirkwood_article/`, while generated stochastic outputs should
be written to `data/` and regenerated from saved metadata rather than
committed. Post-processing code for saved outputs lives under
`src/kirkwood_article/analysis/`, and coordinate trace I/O lives under
`src/kirkwood_article/io/`.

## Adaptive death-rate scaling experiment

The adaptive scaling runner estimates the equilibrium density bias relative to
the mean-field prediction

```text
rho_mf(d) = (b - d) / d'
```

for the grid `d = 0.00, 0.01, ..., 0.10`, with the default article parameters
`b = 1`, `d' = 1`, standard-normal birth/death kernels, and area length `1000`.
Each `d` value uses paired low/high starts, waits for the population traces to
intersect, checks that recent batch means have no detectable trend, and then
collects density samples with autocorrelation-corrected batch means. Sequential
stopping is evaluated only at planned alpha-spending looks and stops when the
95% confidence interval half-width is either below `0.005` in absolute density
or within `±5%` of the observed mean density.

Run the full grid with:

```bash
uv run run-adaptive-d-scaling --output-dir data/adaptive_d_scaling
```

For a faster smoke test on the quickest grid point:

```bash
uv run run-adaptive-d-scaling \
  --only-d 0.1 \
  --output-dir data/adaptive_d_scaling_d01_smoke \
  --max-equilibration-steps 10000 \
  --max-measurement-steps 15000
```

Outputs are written under the selected `data/` subdirectory:

- `summary.json` contains all per-`d` density estimates, confidence half-widths,
  seeds, and run metadata.
- `d_*/summary.json` stores each individual grid-point summary.
- `d_*/*_step_*.npz` coordinate shards store the phase, step, simulation time,
  event count, population, and current one-dimensional particle coordinates.
  These saved shards are the source of truth for future second and third spatial
  moment post-processing.
- `density_scaling_summary.png` compares observed densities and confidence bands
  against mean field, overlays a zero-intercept linear residual fit, and shows
  residual diagnostics.
- `density_scaling_regression.json` records the zero-bias linear and quadratic
  residual regressions, including whether the quadratic term is significant at
  the 95% level.
- `convergence_diagnostics.png` plots final/equilibration simulation time and
  event-count diagnostics against `d`.

To regenerate only the summary plot and regression JSON from an existing output
directory:

```bash
uv run run-adaptive-d-scaling \
  --output-dir data/adaptive_d_scaling \
  --summary-plot-only
```

To run post-hoc pair-correlation analysis from saved measurement coordinate shards:

```bash
uv run run-adaptive-d-scaling \
  --output-dir data/adaptive_d_scaling \
  --pcf-posthoc-only
```

To run post-hoc ordered triplet-correlation analysis and log-Q plots from the
same saved measurement coordinate shards:

```bash
uv run run-adaptive-d-scaling \
  --output-dir data/adaptive_d_scaling \
  --triplet-posthoc-only
```

## Simulator backend

The public `kirkwood_article.sim.ssa_1d` API delegates the main event loop to a
vendored Numba cell-list implementation adapted from `SBDPP_sim/SSA/numba_sim_normal.py`.
Pair-correlation estimates use an FFT convolution with explicit zero-lag
self-pair subtraction.
