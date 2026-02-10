#!/usr/bin/env python3
"""Bayesian shear inference with SHINE — a step-by-step tutorial.

This script walks through the full SHINE pipeline:
  1. Define a configuration describing the scene (galaxy, PSF, noise, shear priors)
  2. Generate a synthetic observation with known ground-truth shear
  3. Build a probabilistic forward model in NumPyro
  4. Run MCMC (NUTS) to recover the posterior over shear
  5. Inspect the results and plot diagnostics

Run it with:
    python examples/shear_inference.py
"""

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import yaml

from shine.config import ConfigHandler
from shine.data import DataLoader, get_mean
from shine.inference import Inference
from shine.scene import SceneBuilder

# ── Step 1: Set up configuration ────────────────────────────────────────────
# The config dict mirrors a SHINE YAML file. Any parameter given as a
# distribution (type: Normal / Uniform / …) becomes a latent variable that
# the sampler will explore; everything else is fixed.

config_dict = {
    "image": {
        "pixel_scale": 0.1,      # arcsec / pixel
        "size_x": 48,
        "size_y": 48,
        "n_objects": 1,
        "noise": {"type": "Gaussian", "sigma": 0.5},
        "fft_size": 128,
    },
    "psf": {
        "type": "Gaussian",
        "sigma": 0.1,            # arcsec
    },
    "gal": {
        "type": "Exponential",
        "flux": 1000.0,          # fixed
        "half_light_radius": 0.5,  # fixed, arcsec
        "shear": {
            "type": "G1G2",
            # ↓ These are the parameters we want to infer.
            # "mean" doubles as the ground truth for synthetic data generation,
            # and "sigma" sets the width of the Normal prior.
            "g1": {"type": "Normal", "mean":  0.02, "sigma": 0.05},
            "g2": {"type": "Normal", "mean": -0.01, "sigma": 0.05},
        },
    },
    "inference": {
        "method": "nuts",
        "nuts_config": {
            "warmup": 500,
            "samples": 1000,
            "chains": 2,
            "dense_mass": False,
            "map_init": {
                "enabled": True,
                "num_steps": 1000,
                "learning_rate": 0.01,
            },
        },
        "rng_seed": 42,
    },
    "data_path": None,           # None → generate synthetic data
    "output_path": "examples/output",
}

# Write the dict to YAML so ConfigHandler can validate it.
output_dir = Path(config_dict["output_path"])
output_dir.mkdir(exist_ok=True)

config_path = output_dir / "config.yaml"
with open(config_path, "w") as f:
    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

config = ConfigHandler.load(str(config_path))
print(f"Configuration loaded from {config_path}")

# ── Step 2: Generate synthetic observation ──────────────────────────────────
# DataLoader renders a noiseless galaxy image with GalSim, then adds Gaussian
# noise. The observation also carries a pre-built JAX-GalSim PSF object that
# the sampler will reuse on every MCMC iteration (no re-construction overhead).

observation = DataLoader.generate_synthetic(config)
print(f"Synthetic image: {observation.image.shape}, noise σ = {config.image.noise.sigma}")

# ── Step 3: Build the probabilistic scene model ────────────────────────────
# SceneBuilder translates the config into a NumPyro model. Distribution
# parameters become `numpyro.sample(...)` sites; fixed parameters stay
# constant. The model renders a galaxy image with JAX-GalSim and defines a
# Gaussian likelihood comparing the rendered image to the observation.

scene = SceneBuilder(config)
model_fn = scene.build_model()
print("Probabilistic model built")

# ── Step 4: Run Bayesian inference ──────────────────────────────────────────
# The Inference engine first runs MAP estimation (SVI with an AutoDelta guide)
# to find a good starting point, then launches NUTS chains from that point.

rng_key = jax.random.PRNGKey(config.inference.rng_seed)
engine = Inference(model=model_fn, config=config.inference)

print(f"Running inference: {config.inference.nuts_config.warmup} warmup + "
      f"{config.inference.nuts_config.samples} samples × {config.inference.nuts_config.chains} chains")
idata = engine.run(
    rng_key=rng_key,
    observed_data=observation.image,
    extra_args={"psf": observation.psf_model},
)

# Save the full posterior as a NetCDF file (ArviZ / xarray format).
idata.to_netcdf(output_dir / "posterior.nc")

# ── Step 5: Examine the results ────────────────────────────────────────────
# Ground-truth values come from the "mean" field of each shear distribution.
true_g1 = get_mean(config.gal.shear.g1)
true_g2 = get_mean(config.gal.shear.g2)

g1_samples = idata.posterior.g1.values.reshape(-1)
g2_samples = idata.posterior.g2.values.reshape(-1)

# Print a compact summary.
print("\n" + "=" * 50)
print("POSTERIOR SUMMARY")
print("=" * 50)
print(f"  g1:  truth = {true_g1:+.4f}  |  posterior = {g1_samples.mean():+.4f} ± {g1_samples.std():.4f}")
print(f"  g2:  truth = {true_g2:+.4f}  |  posterior = {g2_samples.mean():+.4f} ± {g2_samples.std():.4f}")
print("=" * 50)

# Simple diagnostic figure: trace plots (top) + marginal posteriors (bottom).
g1_chains = idata.posterior.g1.values   # (n_chains, n_samples)
g2_chains = idata.posterior.g2.values

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# ── g1 trace ──
for i, chain in enumerate(g1_chains):
    axes[0, 0].plot(chain, alpha=0.7, label=f"Chain {i + 1}")
axes[0, 0].axhline(true_g1, color="red", ls="--", lw=2, label="Truth")
axes[0, 0].set(xlabel="Sample", ylabel="g1", title="Trace: g1")
axes[0, 0].legend()

# ── g2 trace ──
for i, chain in enumerate(g2_chains):
    axes[0, 1].plot(chain, alpha=0.7, label=f"Chain {i + 1}")
axes[0, 1].axhline(true_g2, color="red", ls="--", lw=2, label="Truth")
axes[0, 1].set(xlabel="Sample", ylabel="g2", title="Trace: g2")
axes[0, 1].legend()

# ── g1 posterior ──
axes[1, 0].hist(g1_samples, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="black")
axes[1, 0].axvline(true_g1, color="red", ls="--", lw=2, label="Truth")
axes[1, 0].axvline(g1_samples.mean(), color="green", lw=2, label="Mean")
axes[1, 0].set(xlabel="g1", ylabel="Density",
               title=f"g1 = {g1_samples.mean():.4f} ± {g1_samples.std():.4f}")
axes[1, 0].legend()

# ── g2 posterior ──
axes[1, 1].hist(g2_samples, bins=50, density=True, alpha=0.7, color="coral", edgecolor="black")
axes[1, 1].axvline(true_g2, color="red", ls="--", lw=2, label="Truth")
axes[1, 1].axvline(g2_samples.mean(), color="green", lw=2, label="Mean")
axes[1, 1].set(xlabel="g2", ylabel="Density",
               title=f"g2 = {g2_samples.mean():.4f} ± {g2_samples.std():.4f}")
axes[1, 1].legend()

fig.tight_layout()
fig_path = output_dir / "shear_posterior.png"
fig.savefig(fig_path, dpi=150)
print(f"\nPlot saved to {fig_path}")
plt.close(fig)
