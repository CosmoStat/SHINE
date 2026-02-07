# shine.validation.cli

CLI entry points for the three-stage bias measurement pipeline.

- **Stage 1** (`shine-bias-run`): Generate data + run MCMC
- **Stage 2** (`shine-bias-extract`): Load posteriors, extract diagnostics, write CSV
- **Stage 3** (`shine-bias-stats`): Read CSV, compute bias, check acceptance, plot

::: shine.validation.cli
