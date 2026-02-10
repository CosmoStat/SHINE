"""Standalone script to test MAP fitting on Euclid VIS data.

Reproduces the notebook pipeline to verify end-to-end inference.
"""

import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from shine.config import InferenceConfig, MAPConfig
from shine.euclid.config import (
    EuclidDataConfig,
    EuclidInferenceConfig,
    SourceSelectionConfig,
)
from shine.euclid.data_loader import EuclidDataLoader
from shine.euclid.scene import MultiExposureScene, render_model_images
from shine.inference import Inference

logging.basicConfig(level=logging.INFO)

DATA_DIR = Path("data/EUC_VIS_SWL")

exposure_paths = sorted(
    str(p) for p in DATA_DIR.glob("EUC_VIS_SWL-DET-*_3-4-F.fits.gz")
)
bkg_paths = sorted(
    str(p) for p in DATA_DIR.glob("EUC_VIS_SWL-BKG-*_3-4-F.fits.gz")
)

config = EuclidInferenceConfig(
    data=EuclidDataConfig(
        exposure_paths=exposure_paths,
        psf_path=str(DATA_DIR / "PSF_3-4-F.fits.gz"),
        catalog_path=str(DATA_DIR / "catalogue_3-4-F.fits.gz"),
        background_paths=bkg_paths,
    ),
    sources=SourceSelectionConfig(max_sources=10, min_snr=10.0),
    inference=InferenceConfig(
        method="map",
        map_config=MAPConfig(enabled=True, num_steps=500),
    ),
)

print("Loading data...")
data = EuclidDataLoader(config).load()
print(f"Loaded {data.n_sources} sources, {data.n_exposures} exposures")

print("Building model...")
scene = MultiExposureScene(config, data)
model = scene.build_model()

# Catalog-based initial parameters: safe values where JAX-GalSim rendering
# is always well-behaved (zero ellipticity/shear, catalog flux/hlr).
init_params = {
    "g1": jnp.float32(0.0),
    "g2": jnp.float32(0.0),
    "flux": jnp.asarray(data.catalog_flux_adu),
    "hlr": jnp.asarray(data.catalog_hlr_arcsec),
    "e1": jnp.zeros(data.n_sources),
    "e2": jnp.zeros(data.n_sources),
    "dx": jnp.zeros(data.n_sources),
    "dy": jnp.zeros(data.n_sources),
}

print("Running MAP inference...")
engine = Inference(model, config.inference)
rng = jax.random.PRNGKey(42)
idata = engine.run(rng, observed_data=data.images, init_params=init_params)

print("\nMAP results:")
for name in ["g1", "g2", "flux", "hlr", "e1", "e2", "dx", "dy"]:
    val = idata.posterior[name].values.squeeze()
    print(f"  {name} = {val}")

print("\nRendering model images...")
map_params = {}
for name in ["g1", "g2", "flux", "hlr", "e1", "e2", "dx", "dy"]:
    map_params[name] = np.squeeze(idata.posterior[name].values)

model_images = render_model_images(
    map_params, data,
    pixel_scale=config.data.pixel_scale,
    stamp_sizes=config.galaxy_stamp_sizes,
)
print(f"Model images shape: {model_images.shape}")
print("Done.")
