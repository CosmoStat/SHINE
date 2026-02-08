# Getting Started

## Installation

### Prerequisites

- Python >= 3.11
- A JAX-compatible GPU is recommended but not required

### Install from PyPI

```bash
pip install shine-wl
```

### Install from source (development)

For contributing or working on SHINE itself:

```bash
git clone https://github.com/CosmoStat/SHINE.git
cd SHINE
pip install -e ".[dev,test]"
```

Optional extras:

```bash
pip install -e ".[docs]"   # documentation tools (mkdocs)
```

## Your First Run

SHINE is driven by YAML configuration files. Any parameter specified as a
distribution (e.g. `type: Normal`) becomes a latent variable for Bayesian
inference; everything else is fixed.

### 1. Write a config

Create a file `my_config.yaml`:

```yaml
image:
  pixel_scale: 0.2
  size_x: 32
  size_y: 32
  n_objects: 1
  fft_size: 128
  noise:
    type: Gaussian
    sigma: 0.01

psf:
  type: Gaussian
  sigma: 0.1

gal:
  type: Exponential
  flux: 100.0
  half_light_radius: 0.5
  shear:
    type: G1G2
    g1:
      type: Normal
      mean: 0.02
      sigma: 0.05
    g2:
      type: Normal
      mean: -0.01
      sigma: 0.05

inference:
  method: nuts                # "nuts", "map", or "vi"
  nuts_config:
    warmup: 200
    samples: 500
    chains: 1
    dense_mass: false
    map_init:
      enabled: true
      num_steps: 500
      learning_rate: 0.01
  rng_seed: 42
```

Here, `flux` and `half_light_radius` are fixed values. The shear components
`g1` and `g2` are defined as Normal distributions and become the latent
variables to infer.

### 2. Run inference

```bash
python -m shine.main --config my_config.yaml
```

This will:

1. Generate synthetic data from the config (since no `data_path` is specified)
2. Build the NumPyro probabilistic model
3. Run inference using the configured method (NUTS with MAP init in this example)
4. Save the posterior as `results/posterior.nc` (ArviZ NetCDF format)

Override the output directory with `--output`:

```bash
python -m shine.main --config my_config.yaml --output my_output/
```

### 3. Inspect results

The output is an [ArviZ](https://arviz-devs.github.io/arviz/) `InferenceData`
object saved in NetCDF format. Load and explore it in Python:

```python
import arviz as az

idata = az.from_netcdf("results/posterior.nc")
print(az.summary(idata, var_names=["g1", "g2"]))
az.plot_trace(idata, var_names=["g1", "g2"])
```

### 4. Pedagogical example

For a step-by-step walkthrough that builds the config inline and plots
diagnostics:

```bash
python examples/shear_inference.py
```

## Next Steps

- [Configuration reference](configuration.md) -- full YAML specification
- [Architecture overview](architecture.md) -- how the pieces fit together
- [Validation pipeline](validation/index.md) -- bias measurement and testing
