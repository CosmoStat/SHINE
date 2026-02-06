from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
import jax_galsim as galsim
import numpyro
import numpyro.distributions as dist

from shine import galaxy_utils, psf_utils
from shine.config import DistributionConfig, ShineConfig


class SceneBuilder:
    """Builder for NumPyro probabilistic scene models.

    This class constructs the forward generative model for Bayesian shear inference.
    It translates configuration parameters into NumPyro priors and builds a
    differentiable rendering pipeline using JAX-GalSim.

    Attributes:
        config: SHINE configuration object containing model specifications.
    """

    def __init__(self, config: ShineConfig):
        """Initialize the scene builder.

        Args:
            config: SHINE configuration object.
        """
        self.config = config

    def _parse_prior(
        self, name: str, param_config: Union[float, int, DistributionConfig]
    ) -> float:
        """Create NumPyro distributions from config or return fixed value.

        Args:
            name: Parameter name for NumPyro sampling.
            param_config: Either a fixed numeric value or a DistributionConfig object.

        Returns:
            Sampled value from the distribution or the fixed value.

        Raises:
            ValueError: If the distribution type is not recognized.
        """
        if isinstance(param_config, (float, int)):
            return float(param_config)

        if isinstance(param_config, DistributionConfig):
            if param_config.type == "Normal":
                return numpyro.sample(
                    name, dist.Normal(param_config.mean, param_config.sigma)
                )
            elif param_config.type == "LogNormal":
                return numpyro.sample(
                    name, dist.LogNormal(jnp.log(param_config.mean), param_config.sigma)
                )
            elif param_config.type == "Uniform":
                return numpyro.sample(
                    name, dist.Uniform(param_config.min, param_config.max)
                )
            else:
                raise ValueError(f"Unknown distribution type: {param_config.type}")

        return param_config

    def build_model(self) -> Callable:
        """Build the NumPyro forward generative model for inference.

        Constructs a callable model function that defines:
        - Prior distributions over latent parameters (shear, galaxy properties)
        - Differentiable forward rendering using JAX-GalSim
        - Likelihood function comparing model to observed data

        Returns:
            A NumPyro model function that can be passed to MCMC samplers.
        """

        def model(
            observed_data: Optional[jnp.ndarray] = None, psf: Optional[Any] = None
        ) -> None:
            """NumPyro probabilistic model for shear inference.

            Args:
                observed_data: Observed image data (used as obs in likelihood).
                psf: Pre-built JAX-GalSim PSF object to avoid reconstruction overhead.
            """
            # Define GSParams with configurable FFT size to avoid dynamic shape issues in JAX
            fft_size = self.config.image.fft_size
            gsparams = galsim.GSParams(
                maximum_fft_size=fft_size, minimum_fft_size=fft_size
            )

            # --- 0. PSF is pre-built and passed directly ---
            # No need to reconstruct from config on every MCMC iteration

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

                # Position priors - sample positions from configured distribution
                if self.config.gal.position is not None:
                    pos_cfg = self.config.gal.position
                    # Handle fractional vs pixel coordinates
                    x_min = pos_cfg.x_min if pos_cfg.x_min is not None else self.config.image.size_x * 0.3
                    x_max = pos_cfg.x_max if pos_cfg.x_max is not None else self.config.image.size_x * 0.7
                    y_min = pos_cfg.y_min if pos_cfg.y_min is not None else self.config.image.size_y * 0.3
                    y_max = pos_cfg.y_max if pos_cfg.y_max is not None else self.config.image.size_y * 0.7
                    # Convert fractions to pixels if needed
                    if x_min < 1:
                        x_min *= self.config.image.size_x
                    if x_max < 1:
                        x_max *= self.config.image.size_x
                    if y_min < 1:
                        y_min *= self.config.image.size_y
                    if y_max < 1:
                        y_max *= self.config.image.size_y
                else:
                    # Default: 30%-70% of image
                    x_min = self.config.image.size_x * 0.3
                    x_max = self.config.image.size_x * 0.7
                    y_min = self.config.image.size_y * 0.3
                    y_max = self.config.image.size_y * 0.7

                x = numpyro.sample("x", dist.Uniform(x_min, x_max))
                y = numpyro.sample("y", dist.Uniform(y_min, y_max))

            # --- 3. Differentiable Rendering ---
            def render_one_galaxy(
                flux: float, hlr: float, e1: float, e2: float, x: float, y: float
            ) -> jnp.ndarray:
                """Render a single galaxy image with JAX-GalSim.

                Args:
                    flux: Total flux of the galaxy.
                    hlr: Half-light radius in arcseconds.
                    e1: First component of intrinsic ellipticity.
                    e2: Second component of intrinsic ellipticity.
                    x: X position in pixel coordinates.
                    y: Y position in pixel coordinates.

                Returns:
                    Rendered galaxy image as a JAX array.
                """
                # Create galaxy using utility function
                gal = galaxy_utils.get_jax_galaxy(
                    self.config.gal, flux, hlr, e1, e2, gsparams=gsparams
                )
                # Apply shear
                gal = gal.shear(shear)
                # Convolve with PSF
                gal = galsim.Convolve([gal, psf], gsparams=gsparams)
                return gal.drawImage(
                    nx=self.config.image.size_x,
                    ny=self.config.image.size_y,
                    scale=self.config.image.pixel_scale,
                    offset=(
                        x - self.config.image.size_x / 2 + 0.5,
                        y - self.config.image.size_y / 2 + 0.5,
                    ),
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
