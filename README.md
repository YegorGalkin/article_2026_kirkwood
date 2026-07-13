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

To run post-hoc ordered triplet-correlation analysis and closure-difference plots from the
same saved measurement coordinate shards:

```bash
uv run run-adaptive-d-scaling \
  --output-dir data/adaptive_d_scaling \
  --triplet-posthoc-only
```

## iMPS pair-analysis workflow

The iMPS linear-response workflow depends on PyTorch, which is kept behind the
optional `imps` extra because it is a large dependency. Install the full
development/iMPS environment with:

```bash
uv sync --extra dev --extra imps
```

The packaged console entry points are:

- `uv run run-adaptive-d-scaling` for simulation runs and post-hoc PCF/triplet
  analysis.
- `uv run run-imps-linear-response` for the thermodynamic-limit iMPS derivative
  prediction.
- `uv run run-pair-analysis` for overlaying iMPS density and pair-correlation
  predictions on adaptive simulation outputs.

To generate simulation data comparable to the exponential-kernel iMPS theory,
run adaptive scaling with equal birth/death exponential kernels and variance 1:

```bash
uv run run-adaptive-d-scaling \
  --kernel exponential \
  --kernel-variance 1 \
  --output-dir data/adaptive_d_scaling_exp_var1
```

Run the iMPS linear-response prediction with the matching one-dimensional
exponential kernel convention `K(r)=(lambda/2) exp(-lambda |r|)`. Variance 1
corresponds to `lambda=sqrt(2)`:

```bash
uv run run-imps-linear-response \
  --lam 1.4142135623730951 \
  --a 0.25 \
  --bond-dim 3 \
  --projection-length 7 \
  --d-values 0.00125,0.0025,0.005,0.01 \
  --pairs 1,2,3,4,5,6,8,10,12,16,20 \
  --triplets '1,2;1,3;2,4;2,6' \
  --fit-degree 2 \
  --output data/imps_exp_var1_D3_l7
```

After the adaptive run has measurement coordinate shards, compute PCF summaries
and the iMPS overlay in one step from the adaptive entry point:

```bash
uv run run-adaptive-d-scaling \
  --output-dir data/adaptive_d_scaling_exp_var1 \
  --pcf-posthoc-only \
  --imps-summary data/imps_exp_var1_D3_l7/summary.json
```

Alternatively, call the dedicated pair-analysis entry point directly:

```bash
uv run run-pair-analysis \
  --simulation-dir data/adaptive_d_scaling_exp_var1 \
  --imps-summary data/imps_exp_var1_D3_l7/summary.json
```

The pair-analysis workflow writes `pair_analysis.json`, `pair_analysis.npz`, and
`pair_analysis_summary.png` under the simulation output directory. It validates
that simulation and iMPS kernel metadata match by default; use
`--allow-kernel-mismatch` only for exploratory comparisons.

## Simulator backend

The public `kirkwood_article.sim.ssa_1d` API delegates the main event loop to a
vendored Numba cell-list implementation adapted from `SBDPP_sim/SSA/numba_sim_normal.py`.
Pair-correlation estimates use an FFT convolution with explicit zero-lag
self-pair subtraction.
