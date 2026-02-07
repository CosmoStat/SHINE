# Configuration Reference

SHINE uses GalSim-compatible YAML configuration with a probabilistic extension:
any parameter defined as a distribution becomes a **latent variable** for
inference rather than a fixed simulation value.

## Top-Level Structure

A SHINE config has four top-level sections:

```yaml
image:      # Image rendering parameters
psf:        # Point Spread Function model
gal:        # Galaxy morphology and shear priors
inference:  # MCMC sampler settings
```

Plus optional paths:

```yaml
data_path: null         # Path to observed FITS data (null = generate synthetic)
output_path: results/   # Where to save posteriors
```

## Image Section

Controls the pixel grid and noise model.

```yaml
image:
  pixel_scale: 0.1      # arcsec/pixel
  size_x: 48            # image width in pixels
  size_y: 48            # image height in pixels
  n_objects: 1           # number of galaxies in the scene
  fft_size: 128          # FFT pad size for JAX-GalSim (must be power of 2)
  noise:
    type: Gaussian
    sigma: 0.1           # noise standard deviation
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `pixel_scale` | float > 0 | Pixel scale in arcsec/pixel |
| `size_x`, `size_y` | int > 0 | Image dimensions in pixels |
| `n_objects` | int > 0 | Number of galaxies to render |
| `fft_size` | int (power of 2) | FFT padding for convolution |
| `noise.type` | str | Noise model (`Gaussian`) |
| `noise.sigma` | float > 0 | Noise standard deviation |

## PSF Section

Defines the Point Spread Function. Supports Gaussian and Moffat profiles.

```yaml
# Gaussian PSF
psf:
  type: Gaussian
  sigma: 0.1           # arcsec

# Moffat PSF
psf:
  type: Moffat
  sigma: 0.1           # arcsec (FWHM)
  beta: 4.0            # Moffat beta parameter
```

## Galaxy Section

Defines galaxy morphology, intrinsic shape, shear, and position priors.

### Profile types

```yaml
gal:
  type: Sersic           # or Exponential (Sersic n=1)
  n: 4.0                 # Sersic index (fixed or distribution)
  flux: 1000.0           # total flux
  half_light_radius: 0.5 # arcsec
```

### Distribution syntax

Any numeric parameter can be replaced with a distribution block to make it
a latent variable:

```yaml
# Fixed value
flux: 1000.0

# Prior distribution (becomes a latent variable)
flux:
  type: LogNormal
  mean: 1000.0
  sigma: 0.5
```

Supported distributions:

| Type | Parameters | Description |
|------|-----------|-------------|
| `Normal` | `mean`, `sigma` | Gaussian prior |
| `LogNormal` | `mean`, `sigma` | Log-normal prior |
| `Uniform` | `min`, `max` | Uniform prior |

### Shear

Gravitational shear is defined as two components:

```yaml
gal:
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
```

### Ellipticity

Intrinsic galaxy ellipticity before shearing:

```yaml
gal:
  ellipticity:
    type: E1E2
    e1:
      type: Normal
      mean: 0.0
      sigma: 0.2
    e2:
      type: Normal
      mean: 0.0
      sigma: 0.2
```

### Position

Galaxy position priors. Values < 1 are treated as fractions of image size;
values >= 1 are absolute pixel coordinates.

```yaml
gal:
  position:
    type: Uniform
    x_min: 0.3    # 30% from left edge
    x_max: 0.7    # 70% from left edge
    y_min: 0.3
    y_max: 0.7
```

## Inference Section

Controls the MCMC sampler and optional MAP initialization.

```yaml
inference:
  warmup: 500           # NUTS warmup steps
  samples: 1000         # posterior samples
  chains: 2             # number of parallel chains
  dense_mass: false     # use dense mass matrix
  rng_seed: 42          # reproducibility seed
  map_init:
    enabled: true       # run MAP before MCMC
    num_steps: 1000     # Adam optimization steps
    learning_rate: 0.01
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warmup` | int > 0 | -- | NUTS warmup iterations |
| `samples` | int > 0 | -- | Number of posterior samples |
| `chains` | int > 0 | -- | Number of MCMC chains |
| `dense_mass` | bool | `false` | Dense mass matrix for correlated parameters |
| `rng_seed` | int >= 0 | -- | JAX PRNG seed |
| `map_init.enabled` | bool | `false` | Enable MAP pre-initialization |
| `map_init.num_steps` | int > 0 | -- | Optimization steps for MAP |
| `map_init.learning_rate` | float > 0 | -- | Adam learning rate for MAP |

## Complete Example

```yaml
image:
  pixel_scale: 0.1
  size_x: 48
  size_y: 48
  n_objects: 1
  fft_size: 128
  noise:
    type: Gaussian
    sigma: 0.1

psf:
  type: Gaussian
  sigma: 0.1

gal:
  type: Exponential
  flux: 1000.0
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
  position:
    type: Uniform
    x_min: 23.5
    x_max: 24.5
    y_min: 23.5
    y_max: 24.5

inference:
  warmup: 500
  samples: 1000
  chains: 2
  dense_mass: false
  rng_seed: 42
  map_init:
    enabled: true
    num_steps: 1000
    learning_rate: 0.01
```
