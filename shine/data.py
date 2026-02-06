import jax.numpy as jnp
import jax
from dataclasses import dataclass
from typing import Optional, Any, Dict
from shine.config import ShineConfig
from shine import psf_utils, galaxy_utils

@dataclass
class Observation:
    """Container for observational data."""
    image: jnp.ndarray
    noise_map: jnp.ndarray
    psf_config: Dict[str, Any]
    wcs: Any = None

class DataLoader:
    @staticmethod
    def load(config: ShineConfig) -> Observation:
        if config.data_path and config.data_path != "None":
            # TODO: Implement real data loading (Fits/HDF5)
            raise NotImplementedError("Real data loading not yet implemented. Use synthetic generation.")
        else:
            print("No data path provided. Generating synthetic data...")
            return DataLoader.generate_synthetic(config)

    @staticmethod
    def generate_synthetic(config: ShineConfig) -> Observation:
        """
        Generate synthetic galaxy observations using GalSim.

        This function generates synthetic data for testing purposes.
        Uses mean values from distribution configs for ground truth parameters.
        """
        import galsim

        # 1. Define PSF using PSF utilities
        psf = psf_utils.get_psf(config.psf)

        # 2. Define Galaxy (using mean values from config for "truth")
        def get_mean(param):
            if isinstance(param, (float, int)):
                return float(param)
            # If it's a distribution config
            if param.mean is not None:
                return param.mean
            # Handle Uniform
            if param.type == 'Uniform' and param.min is not None and param.max is not None:
                return (param.min + param.max) / 2.0
            return param.mean # Fallback (might still be None if not handled)

        gal_flux = get_mean(config.gal.flux)
        gal_hlr = get_mean(config.gal.half_light_radius)

        # Intrinsic ellipticity
        e1 = 0.0
        e2 = 0.0
        if config.gal.ellipticity is not None:
            e1 = get_mean(config.gal.ellipticity.e1)
            e2 = get_mean(config.gal.ellipticity.e2)

        # Shear
        g1 = get_mean(config.gal.shear.g1)
        g2 = get_mean(config.gal.shear.g2)
        shear = galsim.Shear(g1=g1, g2=g2)

        # Create Galaxy Object using galaxy utilities
        gal = galaxy_utils.get_galaxy(config.gal, gal_flux, gal_hlr, e1, e2)
        gal = gal.shear(shear)
        
        # Convolve
        final = galsim.Convolve([gal, psf])

        # 3. Draw Image
        image = final.drawImage(nx=config.image.size_x, 
                                ny=config.image.size_y, 
                                scale=config.image.pixel_scale).array
        
        # 4. Add Noise
        rng = galsim.BaseDeviate(0)
        noise_sigma = config.image.noise.sigma
        noise = galsim.GaussianNoise(rng, sigma=noise_sigma)
        
        # GalSim image for noise addition
        gs_image = galsim.Image(image)
        gs_image.addNoise(noise)
        noisy_image = gs_image.array
        
        noise_map = jnp.ones_like(noisy_image) * (noise_sigma**2)

        # Return JAX arrays and PSF config
        return Observation(
            image=jnp.array(noisy_image),
            noise_map=noise_map,
            psf_config={'type': config.psf.type, 'sigma': config.psf.sigma}
        )
