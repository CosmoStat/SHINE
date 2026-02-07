# Level 0 Walkthrough

Level 0 is a noiseless sanity check: a single galaxy with fixed morphology,
very low noise, and a known shear. The posterior should collapse tightly
around the true shear values.

## Prerequisites

```bash
pip install -e .
```

## Step 1: Run a single realization

Use `shine-bias-run` to generate data with a known shear and run MCMC:

```bash
shine-bias-run \
  --shine-config configs/validation/level0_base.yaml \
  --g1-true 0.02 \
  --g2-true -0.01 \
  --seed 42 \
  --output-dir results/validation/level0/r0001 \
  --run-id r0001
```

This produces:

```
results/validation/level0/r0001/
├── posterior.nc       # ArviZ InferenceData
├── truth.json         # {"g1": 0.02, "g2": -0.01}
└── convergence.json   # R-hat, ESS, divergences
```

## Step 2: Extract results

Scan the output directory and extract shear estimates into a CSV:

```bash
shine-bias-extract \
  --input-dir results/validation/level0 \
  --output results/validation/level0/summary.csv
```

The CSV contains one row per realization with columns for true shear, estimated
shear (mean, median, std), and convergence diagnostics.

## Step 3: Compute statistics

Compute bias and check acceptance criteria:

```bash
shine-bias-stats \
  --input results/validation/level0/summary.csv \
  --output-dir results/validation/level0/stats \
  --level level_0 \
  --posterior-dir results/validation/level0
```

This produces:

- `stats/bias_results.json` -- bias values and overall pass/fail
- `stats/plots/` -- diagnostic plots (trace, marginals, pair plot)

## Acceptance Criteria

Level 0 checks three conditions:

| Criterion | Threshold | Meaning |
|-----------|-----------|---------|
| Posterior width | $\sigma < 0.01$ | Posterior should be tight |
| Offset from truth | $< 1\sigma$ | Mean should be near truth |
| Multiplicative bias | $\|m\| < 0.01$ | Less than 1% bias |

## Inspecting Results

Load the posterior in Python for custom analysis:

```python
import arviz as az

idata = az.from_netcdf("results/validation/level0/r0001/posterior.nc")

# Summary table
print(az.summary(idata, var_names=["g1", "g2"]))

# Trace plot
az.plot_trace(idata, var_names=["g1", "g2"])

# Pair plot
az.plot_pair(idata, var_names=["g1", "g2"], kind="kde")
```

## Running Multiple Realizations

For more robust validation, loop over seeds:

```bash
for seed in 42 43 44 45 46; do
  shine-bias-run \
    --shine-config configs/validation/level0_base.yaml \
    --g1-true 0.02 --g2-true -0.01 \
    --seed $seed \
    --output-dir results/validation/level0/r_s${seed} \
    --run-id r_s${seed}
done

shine-bias-extract \
  --input-dir results/validation/level0 \
  --output results/validation/level0/summary.csv
```

Or use the batched approach for GPU efficiency --
see [GPU-Batched Inference](batched.md).
