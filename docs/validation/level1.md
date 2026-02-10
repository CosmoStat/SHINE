# Level 1 Walkthrough

Level 1 introduces **noise bias** -- the systematic shear offset caused by pixel
noise breaking the likelihood symmetry. Unlike Level 0 (noiseless), Level 1 uses
realistic noise levels, random intrinsic ellipticities, and the **paired-shear
method** to cancel shape noise.

## Key Concepts

### Noise Bias

When pixel noise is added, the posterior mode shifts slightly from the true
shear. This systematic offset is the noise bias, and it depends on SNR
(flux/noise ratio). Level 1 measures this bias across a grid of flux and noise
values.

### Paired-Shear Method

Shape noise from intrinsic galaxy ellipticity dominates over shear. The paired
method cancels it:

1. Draw a random ellipticity $(e_1, e_2)$
2. Generate two observations with the **same** ellipticity and noise seed but
   **opposite** shear: $+g$ and $-g$
3. The per-pair shear response is:

$$
R_i = \frac{\hat{g}(+g) - \hat{g}(-g)}{2\,g_\text{true}}
$$

A perfect estimator has $R = 1$. The multiplicative bias is $m = \langle R
\rangle - 1$.

### Random Ellipticity

Intrinsic ellipticities are drawn from $\mathcal{N}(0, 0.2)$ per component,
with rejection sampling to enforce $|e| < 0.7$. Each paired realization uses
the same ellipticity for both the $+g$ and $-g$ observations.

## Prerequisites

```bash
pip install -e ".[dev,test]"
```

## Step 1: Run paired realizations

Use `--paired` to generate $+g/-g$ observation pairs:

```bash
shine-bias-run \
  --shine-config configs/validation/level1_base.yaml \
  --g1-true 0.02 \
  --g2-true 0.0 \
  --seed 42 \
  --paired \
  --output-dir results/validation/level1/r0001 \
  --run-id r0001
```

This produces paired directories:

```
results/validation/level1/r0001/
├── r0001_plus/
│   ├── posterior.nc
│   ├── truth.json       # includes e1, e2, sign="plus", pair_id
│   └── convergence.json
└── r0001_minus/
    ├── posterior.nc
    ├── truth.json       # same e1, e2; sign="minus", same pair_id
    └── convergence.json
```

### Overriding flux and noise

Level 1 sweeps over flux/noise combinations. Use `--flux` and `--noise-sigma` to
override the config values:

```bash
shine-bias-run \
  --shine-config configs/validation/level1_base.yaml \
  --g1-true 0.02 --g2-true 0.0 \
  --seed 42 --paired \
  --flux 500 --noise-sigma 0.5 \
  --output-dir results/validation/level1/f500_n0.5/r0001 \
  --run-id r0001
```

### Running a full campaign

Loop over the shear grid and flux/noise grid:

```bash
for g1 in 0.01 0.02 0.05; do
  for flux in 100 500 1000 5000; do
    for noise in 0.01 0.1 0.5 1.0; do
      for seed in $(seq 42 141); do  # 100 realizations
        shine-bias-run \
          --shine-config configs/validation/level1_base.yaml \
          --g1-true $g1 --g2-true 0.0 \
          --seed $seed --paired \
          --flux $flux --noise-sigma $noise \
          --output-dir results/validation/level1/f${flux}_n${noise}/g1_${g1}/r_s${seed} \
          --run-id r_s${seed}
      done
    done
  done
done
```

!!! tip
    For HPC clusters, each `shine-bias-run` invocation is independent and can
    be submitted as a separate SLURM job. Use `--batch-size` to pack multiple
    realizations per GPU.

## Step 2: Extract results

```bash
shine-bias-extract \
  --input-dir results/validation/level1 \
  --output results/validation/level1/summary.csv
```

The Level 1 CSV includes additional columns beyond Level 0:

| Column | Description |
|--------|-------------|
| `e1_true`, `e2_true` | Random intrinsic ellipticity |
| `sign` | `"plus"` or `"minus"` |
| `pair_id` | Links $+g$/$-g$ pairs |
| `flux` | Galaxy flux |
| `noise_sigma` | Noise standard deviation |
| `g1_p16`, `g1_p84` | 16th/84th percentiles (for coverage) |
| `g2_p16`, `g2_p84` | 16th/84th percentiles (for coverage) |

## Step 3: Compute Level 1 statistics

```bash
shine-bias-stats \
  --input results/validation/level1/summary.csv \
  --output-dir results/validation/level1/stats \
  --level level_1 \
  --posterior-dir results/validation/level1
```

Level 1 statistics are computed per (flux, noise) grid point:

1. **Paired response** -- match $+g/-g$ pairs, compute $R_i$ per shear grid
   point
2. **Bias regression** -- fit $\hat{g} = (1+m)\,g_\text{true} + c$ across shear
   grid points using `compute_bias_regression()`
3. **Coverage** -- check that credible intervals contain the truth at the
   nominal rate
4. **SBC ranks** -- simulation-based calibration for posterior quality

### Diagnostic plots

Level 1 generates the following plots in `stats/plots/`:

- `bias_vs_shear_g1.png` / `bias_vs_shear_g2.png` -- estimated vs true shear
  with regression line and $m$/$c$ annotation
- `coverage.png` -- observed vs nominal coverage with $3\sigma$ binomial bands
- `sbc_ranks_g1.png` / `sbc_ranks_g2.png` -- SBC rank histograms with expected
  uniform level

## Acceptance Criteria

| Criterion | Threshold | Condition |
|-----------|-----------|-----------|
| Multiplicative bias | $\|m\| < 0.01$ | For SNR $> 20$ grid points |
| Additive bias | $\|c\| < 5 \times 10^{-4}$ | All grid points |
| Coverage (68%) | Within 3% of nominal | $0.65 \leq \text{obs} \leq 0.71$ |
| Coverage (95%) | Within 3% of nominal | $0.92 \leq \text{obs} \leq 0.98$ |
| SBC calibration | KS $p > 0.01$ | Rank uniformity |

## Configuration Reference

### SHINE config (`level1_base.yaml`)

```yaml
gal:
  type: Exponential
  flux: 1000.0
  half_light_radius: 0.5
  ellipticity:
    e1: { type: Normal, mean: 0.0, std: 0.2 }
    e2: { type: Normal, mean: 0.0, std: 0.2 }

psf:
  type: Gaussian
  sigma: 0.1

image:
  pixel_scale: 0.2
  stamp_size: 51
  noise:
    type: gaussian
    sigma: 0.1

inference:
  method: nuts
  num_warmup: 500
  num_samples: 1000
  num_chains: 2
  init_strategy: map
```

### Campaign config (`level1_test.yaml`)

```yaml
level: level_1
shine_config_path: configs/validation/level1_base.yaml
n_realizations: 1000
paired: true

shear_grid:
  values: [0.01, 0.02, 0.05]

flux_noise_grid:
  flux_values: [100, 500, 1000, 5000]
  noise_values: [0.01, 0.1, 0.5, 1.0]

acceptance:
  max_abs_m: 0.01
  max_abs_c: 0.0005
  coverage_tolerance: 0.03
  sbc_ks_pvalue_min: 0.01
```

## Inspecting Results

```python
import json
import pandas as pd

# Load bias results
with open("results/validation/level1/stats/bias_results.json") as f:
    bias = json.load(f)

print(f"Overall pass: {bias['pass']}")

# Load CSV for custom analysis
df = pd.read_csv("results/validation/level1/summary.csv")

# Filter to a specific grid point
subset = df[(df["flux"] == 1000) & (df["noise_sigma"] == 0.1)]
print(f"Realizations at flux=1000, noise=0.1: {len(subset)}")
```
