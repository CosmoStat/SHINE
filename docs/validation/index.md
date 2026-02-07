# Validation Pipeline

SHINE includes a built-in bias measurement infrastructure to verify that the
inference pipeline correctly recovers known shear values. The validation framework
is organized as a three-stage pipeline with dedicated CLI tools at each step.

## Overview

The core question: if we generate data with a known shear $g_\text{true}$, does
the posterior from SHINE recover it?

Bias is quantified via the standard multiplicative ($m$) and additive ($c$)
parameterization:

$$
\hat{g} = (1 + m) \, g_\text{true} + c
$$

A perfect estimator has $m = 0$ and $c = 0$.

## Three-Stage Pipeline

```
Stage 1 (Run)       Stage 2 (Extract)     Stage 3 (Stats)
─────────────       ─────────────────     ───────────────
Config + shear  →   posterior.nc      →   summary.csv      →  bias_results.json
  ↓                   ↓                     ↓
Generate data       Extract diagnostics   Compute m, c
Run MCMC            Check convergence     Check acceptance
Save posterior      Write CSV             Generate plots
```

### Stage 1: Run (`shine-bias-run`)

Generates synthetic data with an explicit shear override and runs MCMC inference.

**Outputs** (per realization):

- `posterior.nc` -- ArviZ InferenceData with posterior samples
- `truth.json` -- ground truth shear values and seed
- `convergence.json` -- R-hat, ESS, divergences, BFMI

### Stage 2: Extract (`shine-bias-extract`)

Scans a directory of Stage 1 outputs, loads each posterior, extracts convergence
diagnostics and shear summary statistics, and writes a CSV.

**Output**: a summary CSV with columns for true shear, estimated shear (mean,
median, std), convergence diagnostics, and pass/fail status.

### Stage 3: Stats (`shine-bias-stats`)

Reads the summary CSV, computes bias ($m$, $c$), checks acceptance criteria,
and optionally generates diagnostic plots.

**Outputs**:

- `bias_results.json` -- bias values, pass/fail status
- `plots/` -- trace plots, marginal posteriors, pair plots

## Bias Levels

| Level | Description | Noise | Galaxies |
|-------|-------------|-------|----------|
| **Level 0** | Noiseless sanity check | Very low | Single, fixed morphology |
| Level 1 | Low-noise regression | Low | Single, varied morphology |
| Level 2 | Realistic noise | Survey-like | Population with priors |
| Level 3 | Full survey simulation | Realistic | Multi-galaxy scenes |

!!! note
    Currently only **Level 0** is fully implemented. Higher levels are defined
    in the configuration schema but their statistical routines are stubs.

## Configuration

Validation campaigns are configured with a `BiasTestConfig` YAML:

```yaml
level: level_0
shine_config_path: configs/validation/level0_base.yaml

shear_grid:
  values: [0.02]

n_realizations: 1
paired: false

convergence:
  rhat_max: 1.05
  ess_min: 100
  divergence_frac_max: 0.0
  bfmi_min: 0.3

acceptance:
  max_offset_sigma: 1.0
  max_posterior_width: 0.01
  max_abs_m: 0.01
  coverage_levels: [0.68, 0.95]

output_dir: results/validation/level0
```

## Next Steps

- [Level 0 Walkthrough](level0.md) -- step-by-step tutorial
- [GPU-Batched Inference](batched.md) -- running multiple realizations efficiently
- [API Reference](../api/validation/index.md) -- full module documentation
