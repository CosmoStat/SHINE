import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import jax
import jax_galsim as galsim
from shine.config import ShineConfig, DistributionConfig

class SceneBuilder:
    def __init__(self, config: ShineConfig):
        self.config = config

    def _parse_prior(self, name: str, param_config):
        """Helper to create NumPyro distributions from config or return fixed value."""
        if isinstance(param_config, (float, int)):
            return float(param_config)
        
        if isinstance(param_config, DistributionConfig):
            if param_config.type == 'Normal':
                return numpyro.sample(name, dist.Normal(param_config.mean, param_config.sigma))
            elif param_config.type == 'LogNormal':
                 return numpyro.sample(name, dist.LogNormal(jnp.log(param_config.mean), param_config.sigma))
            elif param_config.type == 'Uniform':
                return numpyro.sample(name, dist.Uniform(param_config.min, param_config.max))
            else:
                raise ValueError(f"Unknown distribution type: {param_config.type}")
        
        return param_config

    def build_model(self):
        """Returns a callable model function for NumPyro."""
        
        def model(observed_data=None, psf_config=None):
            # Define GSParams with fixed FFT size to avoid dynamic shape issues in JAX
            fft_size = 128
            gsparams = galsim.GSParams(maximum_fft_size=fft_size, minimum_fft_size=fft_size)

            # --- 0. Build PSF from config using JAX-GalSim ---
            if psf_config['type'] == 'Gaussian':
                psf = galsim.Gaussian(sigma=psf_config['sigma'], gsparams=gsparams)
            else:
                raise NotImplementedError(f"PSF type {psf_config['type']} not implemented")
            
            # --- 1. Global Parameters (Shear) ---
            g1 = self._parse_prior("g1", self.config.gal.shear.g1)
            g2 = self._parse_prior("g2", self.config.gal.shear.g2)
            shear = galsim.Shear(g1=g1, g2=g2)
            
            # --- 2. Galaxy Population ---
            n_galaxies = self.config.image.n_objects
            
            with numpyro.plate("galaxies", n_galaxies):
                flux = self._parse_prior("flux", self.config.gal.flux)
                hlr = self._parse_prior("hlr", self.config.gal.half_light_radius)
                
                x = self.config.image.size_x / 2.0
                y = self.config.image.size_y / 2.0

            # --- 3. Differentiable Rendering ---
            def render_one_galaxy(flux, hlr, x, y):
                # Use Exponential instead of Sersic (not available in jax_galsim yet)
                gal = galsim.Exponential(half_light_radius=hlr, flux=flux, gsparams=gsparams)
                gal = gal.shear(shear)
                gal = galsim.Convolve([gal, psf], gsparams=gsparams)
                return gal.drawImage(nx=self.config.image.size_x, 
                                     ny=self.config.image.size_y, 
                                     scale=self.config.image.pixel_scale,
                                     offset=(x - self.config.image.size_x/2 + 0.5, y - self.config.image.size_y/2 + 0.5)
                                     ).array

            flux = jnp.atleast_1d(flux)
            hlr = jnp.atleast_1d(hlr)
            x = jnp.atleast_1d(x)
            y = jnp.atleast_1d(y)
            
            galaxy_images = jax.vmap(render_one_galaxy)(flux, hlr, x, y)
            model_image = jnp.sum(galaxy_images, axis=0)
            
            # --- 4. Likelihood ---
            sigma = self.config.image.noise.sigma
            numpyro.sample("obs", dist.Normal(model_image, sigma), obs=observed_data)
            
        return model
