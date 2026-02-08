# Level 0 Walkthrough

Level 0 is a noiseless sanity check: a single galaxy with fixed morphology,
very low noise, and a known shear. Since there is effectively no noise, MAP
estimation is sufficient -- the point estimate should land directly on the
true shear values.

## Prerequisites

```bash
pip install -e .
```

## Step 1: Run a single realization

Use `shine-bias-run` to generate data with a known shear and run MAP inference.
The default Level 0 config (`configs/validation/level0_base.yaml`) uses
`method: map`:

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
├── posterior.nc       # ArviZ InferenceData (MAP point estimate)
├── truth.json         # {"g1": 0.02, "g2": -0.01}
└── convergence.json   # Convergence diagnostics (sentinels for MAP)
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
- `stats/plots/` -- diagnostic plots (MAP estimate vs truth)

## Acceptance Criteria

Level 0 checks three conditions:

| Criterion | Threshold | Meaning |
|-----------|-----------|---------|
| Posterior width | $\sigma < 0.01$ | Posterior should be tight (0 for MAP) |
| Offset from truth | $< 1\sigma$ | Estimate should be near truth |
| Multiplicative bias | $\|m\| < 0.01$ | Less than 1% bias |

## Inspecting Results

Load the posterior in Python for custom analysis:

```python
import arviz as az

idata = az.from_netcdf("results/validation/level0/r0001/posterior.nc")

# Check the inference method
print(idata.posterior.attrs.get("inference_method"))  # "map"

# Point estimates
print(f"g1 = {float(idata.posterior.g1.values.flatten()[0]):.6f}")
print(f"g2 = {float(idata.posterior.g2.values.flatten()[0]):.6f}")
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
