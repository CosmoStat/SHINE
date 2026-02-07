# GPU-Batched Inference

When running many bias realizations, batching packs multiple independent
realizations into a single MCMC run. This amortizes JAX compilation overhead
and makes better use of GPU parallelism.

## How It Works

Instead of running N separate MCMC jobs, batched inference:

1. Generates N synthetic observations and stacks them into a single array
2. Builds a batched NumPyro model that `vmap`s over the batch dimension
3. Runs one MCMC chain that samples all N shear posteriors simultaneously
4. Splits the combined posterior back into per-realization outputs

Each realization gets its own shear latent variables (`g1_0`, `g1_1`, ...) so
they are independent despite sharing the same MCMC chain.

## Usage

Use `--batch-size` with `shine-bias-run`:

```bash
shine-bias-run \
  --shine-config configs/validation/level0_base.yaml \
  --batch-size 4 \
  --shear-grid 0.01 0.02 0.05 \
  --n-realizations 2 \
  --base-seed 42 \
  --output-dir results/validation/batched
```

This creates `3 shear points x 2 realizations = 6` realizations, processed
in chunks of 4.

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | `1` | Realizations per GPU job |
| `--shear-grid` | -- | Space-separated $g_1$ values |
| `--n-realizations` | `1` | Realizations per shear grid point |
| `--base-seed` | `42` | Starting seed (incremented per realization) |
| `--g1-true` | `0.0` | Single $g_1$ value (if no `--shear-grid`) |
| `--g2-true` | `0.0` | $g_2$ value (same for all realizations) |

## Output Structure

Batched runs produce the same per-realization directory structure as single runs:

```
results/validation/batched/
├── g1_+0.0100_g2_+0.0000_s42/
│   ├── posterior.nc
│   ├── truth.json
│   └── convergence.json
├── g1_+0.0100_g2_+0.0000_s43/
│   ├── ...
└── g1_+0.0200_g2_+0.0000_s44/
    ├── ...
```

Run IDs encode the true shear and seed: `g1_{g1:+.4f}_g2_{g2:+.4f}_s{seed}`.

## Downstream Stages

Extraction and statistics work identically on batched output:

```bash
# Extract
shine-bias-extract \
  --input-dir results/validation/batched \
  --output results/validation/batched/summary.csv

# Stats
shine-bias-stats \
  --input results/validation/batched/summary.csv \
  --output-dir results/validation/batched/stats \
  --level level_0 \
  --posterior-dir results/validation/batched
```

## Choosing Batch Size

- **GPU memory** is the main constraint -- each realization adds an image to the
  forward model. Monitor with `nvidia-smi`.
- Start with `--batch-size 4` and increase if GPU memory allows.
- Compilation time grows with batch size (first run only).
- If a batch fails, all realizations in that batch are lost. Smaller batches
  provide more granular fault tolerance.

!!! tip
    For Level 0 (small images, low noise), batch sizes of 8--16 work well on
    a single GPU. For realistic image sizes, start with 2--4.
