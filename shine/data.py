import jax.numpy as jnp
import jax
from dataclasses import dataclass
from typing import Optional, Any, Dict
from shine.config import ShineConfig

@dataclass
class Observation:
    image: jnp.ndarray
    noise_map: jnp.ndarray
    psf_config: Dict[str, Any]  # Store PSF config instead of object
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
        import galsim
        
        # 1. Define PSF
        if config.psf.type == "Gaussian":
            psf = galsim.Gaussian(sigma=config.psf.sigma)
        else:
            raise NotImplementedError(f"PSF type {config.psf.type} not supported for synthetic gen")

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
        
        # Shear
        g1 = get_mean(config.gal.shear.g1)
        g2 = get_mean(config.gal.shear.g2)
        shear = galsim.Shear(g1=g1, g2=g2)

        # Create Galaxy Object - Use Exponential (Sersic n=1)
        gal = galsim.Exponential(half_light_radius=gal_hlr, flux=gal_flux)
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
