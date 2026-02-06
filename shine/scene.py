import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import jax
import jax_galsim as galsim
from shine.config import ShineConfig, DistributionConfig
from shine import psf_utils, galaxy_utils

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
        """
        Returns a callable forward model function for NumPyro inference.

        This function constructs the generative model including priors,
        differentiable rendering, and likelihood.
        """

        def model(observed_data=None, psf_config=None):
            # Define GSParams with fixed FFT size to avoid dynamic shape issues in JAX
            fft_size = 128
            gsparams = galsim.GSParams(maximum_fft_size=fft_size, minimum_fft_size=fft_size)

            # --- 0. Build PSF from config using JAX-GalSim ---
            # Reconstruct PSF config object for utility function
            from shine.config import PSFConfig
            psf_cfg = PSFConfig(**psf_config)
            psf = psf_utils.get_jax_psf(psf_cfg, gsparams=gsparams)

            # --- 1. Global Parameters (Shear) ---
            g1 = self._parse_prior("g1", self.config.gal.shear.g1)
            g2 = self._parse_prior("g2", self.config.gal.shear.g2)
            shear = galsim.Shear(g1=g1, g2=g2)

            # --- 2. Galaxy Population ---
            n_galaxies = self.config.image.n_objects

            with numpyro.plate("galaxies", n_galaxies):
                flux = self._parse_prior("flux", self.config.gal.flux)
                hlr = self._parse_prior("hlr", self.config.gal.half_light_radius)

                # Intrinsic ellipticity
                e1 = 0.0
                e2 = 0.0
                if self.config.gal.ellipticity is not None:
                    e1 = self._parse_prior("e1", self.config.gal.ellipticity.e1)
                    e2 = self._parse_prior("e2", self.config.gal.ellipticity.e2)

                # Position priors - sample positions instead of fixing at center
                # For individual postage stamps, positions should be near center
                # For a wider field, positions should span the image
                # TODO: Add configuration for position priors
                x = numpyro.sample("x", dist.Uniform(
                    self.config.image.size_x * 0.3,
                    self.config.image.size_x * 0.7
                ))
                y = numpyro.sample("y", dist.Uniform(
                    self.config.image.size_y * 0.3,
                    self.config.image.size_y * 0.7
                ))

            # --- 3. Differentiable Rendering ---
            def render_one_galaxy(flux, hlr, e1, e2, x, y):
                # Create galaxy using utility function
                gal = galaxy_utils.get_jax_galaxy(
                    self.config.gal, flux, hlr, e1, e2, gsparams=gsparams
                )
                # Apply shear
                gal = gal.shear(shear)
                # Convolve with PSF
                gal = galsim.Convolve([gal, psf], gsparams=gsparams)
                return gal.drawImage(nx=self.config.image.size_x,
                                     ny=self.config.image.size_y,
                                     scale=self.config.image.pixel_scale,
                                     offset=(x - self.config.image.size_x/2 + 0.5, y - self.config.image.size_y/2 + 0.5)
                                     ).array

            flux = jnp.atleast_1d(flux)
            hlr = jnp.atleast_1d(hlr)
            e1 = jnp.atleast_1d(e1)
            e2 = jnp.atleast_1d(e2)
            x = jnp.atleast_1d(x)
            y = jnp.atleast_1d(y)

            galaxy_images = jax.vmap(render_one_galaxy)(flux, hlr, e1, e2, x, y)
            model_image = jnp.sum(galaxy_images, axis=0)

            # --- 4. Likelihood ---
            sigma = self.config.image.noise.sigma
            numpyro.sample("obs", dist.Normal(model_image, sigma), obs=observed_data)

        return model
