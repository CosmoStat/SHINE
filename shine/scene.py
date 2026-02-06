from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
import jax_galsim as galsim
import numpyro
import numpyro.distributions as dist

from shine import galaxy_utils
from shine.config import DistributionConfig, ShineConfig

# Default position prior bounds as fraction of image size
_DEFAULT_POS_MIN_FRAC = 0.3
_DEFAULT_POS_MAX_FRAC = 0.7


class SceneBuilder:
    """Builder for NumPyro probabilistic scene models.

    Constructs the forward generative model for Bayesian shear inference by
    translating configuration parameters into NumPyro priors and building a
    differentiable rendering pipeline using JAX-GalSim.

    Attributes:
        config: SHINE configuration object containing model specifications.
    """

    def __init__(self, config: ShineConfig) -> None:
        """Initialize the scene builder.

        Args:
            config: SHINE configuration object.
        """
        self.config = config

    def _parse_prior(
        self, name: str, param_config: Union[float, int, DistributionConfig]
    ) -> float:
        """Create a NumPyro sample site from config, or return a fixed value.

        Args:
            name: Parameter name for NumPyro sampling.
            param_config: Either a fixed numeric value or a DistributionConfig.

        Returns:
            Sampled value from the distribution or the fixed value.

        Raises:
            ValueError: If the distribution type is not recognized.
        """
        if isinstance(param_config, (float, int)):
            return float(param_config)

        if param_config.type == "Normal":
            return numpyro.sample(
                name, dist.Normal(param_config.mean, param_config.sigma)
            )
        if param_config.type == "LogNormal":
            return numpyro.sample(
                name, dist.LogNormal(jnp.log(param_config.mean), param_config.sigma)
            )
        if param_config.type == "Uniform":
            return numpyro.sample(
                name, dist.Uniform(param_config.min, param_config.max)
            )
        raise ValueError(f"Unknown distribution type: '{param_config.type}'")

    @staticmethod
    def _resolve_bound(
        value: Optional[float], image_size: int, default_frac: float
    ) -> float:
        """Resolve a single position bound to pixel coordinates.

        Values less than 1 are treated as fractions of the image size.
        None values fall back to the default fraction.

        Args:
            value: Position bound (fraction if < 1, pixels if >= 1, or None).
            image_size: Image dimension in pixels.
            default_frac: Default fraction of image size when value is None.

        Returns:
            Position bound in pixel coordinates.
        """
        if value is None:
            return image_size * default_frac
        if value < 1:
            return value * image_size
        return value

    def _resolve_position_bounds(
        self,
    ) -> tuple:
        """Resolve all position bounds from config to pixel coordinates.

        Returns:
            Tuple of (x_min, x_max, y_min, y_max) in pixel coordinates.
        """
        pos_cfg = self.config.gal.position
        img_cfg = self.config.image

        def _get(attr: str) -> Optional[float]:
            return getattr(pos_cfg, attr, None) if pos_cfg else None

        x_min = self._resolve_bound(_get("x_min"), img_cfg.size_x, _DEFAULT_POS_MIN_FRAC)
        x_max = self._resolve_bound(_get("x_max"), img_cfg.size_x, _DEFAULT_POS_MAX_FRAC)
        y_min = self._resolve_bound(_get("y_min"), img_cfg.size_y, _DEFAULT_POS_MIN_FRAC)
        y_max = self._resolve_bound(_get("y_max"), img_cfg.size_y, _DEFAULT_POS_MAX_FRAC)

        return x_min, x_max, y_min, y_max

    def build_model(self) -> Callable:
        """Build the NumPyro forward generative model for inference.

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
            fft_size = self.config.image.fft_size
            gsparams = galsim.GSParams(
                maximum_fft_size=fft_size, minimum_fft_size=fft_size
            )
            img_cfg = self.config.image
            gal_cfg = self.config.gal

            # 1. Global shear parameters
            g1 = self._parse_prior("g1", gal_cfg.shear.g1)
            g2 = self._parse_prior("g2", gal_cfg.shear.g2)
            shear = galsim.Shear(g1=g1, g2=g2)

            # 2. Galaxy population
            with numpyro.plate("galaxies", img_cfg.n_objects):
                flux = self._parse_prior("flux", gal_cfg.flux)
                hlr = self._parse_prior("hlr", gal_cfg.half_light_radius)

                # Intrinsic ellipticity
                e1 = 0.0
                e2 = 0.0
                if gal_cfg.ellipticity is not None:
                    e1 = self._parse_prior("e1", gal_cfg.ellipticity.e1)
                    e2 = self._parse_prior("e2", gal_cfg.ellipticity.e2)

                # Position priors
                x_min, x_max, y_min, y_max = self._resolve_position_bounds()

                x = numpyro.sample("x", dist.Uniform(x_min, x_max))
                y = numpyro.sample("y", dist.Uniform(y_min, y_max))

            # 3. Differentiable rendering
            def render_one_galaxy(
                flux: float, hlr: float, e1: float, e2: float, x: float, y: float
            ) -> jnp.ndarray:
                """Render a single galaxy image with JAX-GalSim."""
                gal = galaxy_utils.get_jax_galaxy(
                    gal_cfg, flux, hlr, e1, e2, gsparams=gsparams
                )
                gal = gal.shear(shear)
                gal = galsim.Convolve([gal, psf], gsparams=gsparams)
                return gal.drawImage(
                    nx=img_cfg.size_x,
                    ny=img_cfg.size_y,
                    scale=img_cfg.pixel_scale,
                    offset=(
                        x - img_cfg.size_x / 2 + 0.5,
                        y - img_cfg.size_y / 2 + 0.5,
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

            # 4. Likelihood
            sigma = img_cfg.noise.sigma
            numpyro.sample("obs", dist.Normal(model_image, sigma), obs=observed_data)

        return model
