# Data directory

Generated experiment outputs should be written under this repo-root `data/`
directory. The files are reproducible from configuration, seeds, and code, so
large generated artifacts should not be committed unless explicitly required.

The adaptive death-rate scaling runner writes to `data/adaptive_d_scaling/` by
default. Post-hoc PCF analysis writes `pcf_posthoc_analysis.json`,
`pcf_posthoc_analysis.npz`, `pcf_grid.png`, and `pcf_fit_parameters.png` into the
same experiment output directory. Post-hoc triplet analysis writes
`triplet_posthoc_analysis.json`, `triplet_posthoc_analysis.npz`,
`triplet_g3_surface_d_*.png`, `triplet_difference_surface_d_*.png`, and
`triplet_difference_lines.png` into that
directory.
