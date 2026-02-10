import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

import jax.numpy as jnp

from shine import galaxy_utils, psf_utils
from shine.config import DistributionConfig, ShineConfig

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """Container for observational data.

    Attributes:
        image: Observed image as a JAX array.
        noise_map: Noise variance map corresponding to the image.
        psf_model: JAX-GalSim PSF object for differentiable convolutions.
        wcs: World Coordinate System information (optional, not yet implemented).
    """

    image: jnp.ndarray
    noise_map: jnp.ndarray
    psf_model: Any  # JAX-GalSim PSF object
    wcs: Any = None


def get_mean(param: Union[float, int, DistributionConfig]) -> float:
    """Extract mean value from a parameter that may be fixed or distributional.

    For fixed numeric values, returns the value directly. For distribution configs,
    returns the mean (Normal/LogNormal) or midpoint (Uniform).

    Args:
        param: Either a fixed numeric value or a DistributionConfig object.

    Returns:
        The representative central value of the parameter.

    Raises:
        ValueError: If distribution type is unsupported or missing required fields.
        TypeError: If parameter is neither numeric nor a DistributionConfig.
    """
    if isinstance(param, (float, int)):
        return float(param)

    if not isinstance(param, DistributionConfig):
        raise TypeError(
            f"Parameter must be a numeric value or DistributionConfig, "
            f"got {type(param).__name__}"
        )

    if param.mean is not None:
        return param.mean

    if param.type == "Uniform":
        if param.min is None or param.max is None:
            raise ValueError(
                f"Uniform distribution requires both 'min' and 'max', "
                f"got min={param.min}, max={param.max}"
            )
        return (param.min + param.max) / 2.0

    raise ValueError(
        f"Cannot extract mean from distribution type '{param.type}'. "
        f"Supported: 'Normal' (requires mean), 'Uniform' (requires min, max)"
    )


def _validate_magnitude(value: float, limit: float, name: str, components: str) -> None:
    """Validate that a magnitude is below a physical limit.

    Args:
        value: Magnitude value to check.
        limit: Upper limit (exclusive).
        name: Human-readable name for error messages (e.g., "Shear").
        components: Description of components for error messages.

    Raises:
        ValueError: If value >= limit.
    """
    if value >= limit:
        raise ValueError(
            f"{name} magnitude must be < {limit}, got {value:.4f} from {components}"
        )


class DataLoader:
    """Loader for observational data and synthetic data generation."""

    @staticmethod
    def load(config: ShineConfig) -> Observation:
        """Load observational data from file or generate synthetic data.

        Args:
            config: SHINE configuration object containing data paths and parameters.

        Returns:
            Observation object containing the loaded or generated data.

        Raises:
            NotImplementedError: If a data path is provided (real data loading not implemented).
        """
        # Guard against YAML parsing "None" as the string "None" instead of null
        if config.data_path and config.data_path != "None":
            raise NotImplementedError(
                "Real data loading not yet implemented. Use synthetic generation."
            )
        logger.info("No data path provided. Generating synthetic data...")
        return DataLoader.generate_synthetic(config)

    @staticmethod
    def generate_synthetic(
        config: ShineConfig,
        g1_true: Optional[float] = None,
        g2_true: Optional[float] = None,
        e1_true: Optional[float] = None,
        e2_true: Optional[float] = None,
        noise_seed: Optional[int] = None,
    ) -> Observation:
        """Generate synthetic galaxy observations using GalSim.

        Uses mean values from distribution configs for ground truth parameters,
        renders the galaxy image, and adds noise to simulate observations.
        The PSF is pre-built as a JAX-GalSim object to avoid reconstruction
        on each MCMC iteration during inference.

        Args:
            config: SHINE configuration object containing simulation parameters.
            g1_true: If provided, overrides the g1 shear from config.
            g2_true: If provided, overrides the g2 shear from config.
            e1_true: If provided, overrides the e1 ellipticity from config.
            e2_true: If provided, overrides the e2 ellipticity from config.
            noise_seed: If provided, overrides config.inference.rng_seed for noise RNG.

        Returns:
            Observation object with synthetic noisy image, noise map, and JAX-GalSim PSF.
        """
        import galsim
        import jax_galsim

        # Build GalSim PSF for rendering and JAX-GalSim PSF for inference
        psf = psf_utils.get_psf(config.psf)

        fft_size = config.image.fft_size
        gsparams = jax_galsim.GSParams(
            maximum_fft_size=fft_size, minimum_fft_size=fft_size
        )
        jax_psf = psf_utils.get_jax_psf(config.psf, gsparams=gsparams)

        # Extract ground truth galaxy parameters from config
        gal_flux = get_mean(config.gal.flux)
        gal_hlr = get_mean(config.gal.half_light_radius)

        if gal_flux <= 0:
            raise ValueError(f"Galaxy flux must be positive, got {gal_flux}")
        if gal_hlr <= 0:
            raise ValueError(f"Galaxy half-light radius must be positive, got {gal_hlr}")

        # Intrinsic ellipticity — use overrides if provided
        e1 = 0.0
        e2 = 0.0
        if e1_true is not None or e2_true is not None:
            e1 = e1_true if e1_true is not None else 0.0
            e2 = e2_true if e2_true is not None else 0.0
        elif config.gal.ellipticity is not None:
            e1 = get_mean(config.gal.ellipticity.e1)
            e2 = get_mean(config.gal.ellipticity.e2)
        e_mag = (e1**2 + e2**2) ** 0.5
        if e_mag > 0:
            _validate_magnitude(e_mag, 1.0, "Ellipticity", f"(e1={e1}, e2={e2})")

        # Shear — use overrides if provided, otherwise extract from config
        g1 = g1_true if g1_true is not None else get_mean(config.gal.shear.g1)
        g2 = g2_true if g2_true is not None else get_mean(config.gal.shear.g2)
        g_mag = (g1**2 + g2**2) ** 0.5
        _validate_magnitude(g_mag, 1.0, "Shear", f"(g1={g1}, g2={g2})")
        shear = galsim.Shear(g1=g1, g2=g2)

        # Create and shear galaxy
        gal = galaxy_utils.get_galaxy(config.gal, gal_flux, gal_hlr, e1, e2)
        gal = gal.shear(shear)

        # Convolve with PSF and draw image
        final = galsim.Convolve([gal, psf])
        image = final.drawImage(
            nx=config.image.size_x,
            ny=config.image.size_y,
            scale=config.image.pixel_scale,
        ).array

        # Add noise — use override seed if provided
        noise_sigma = config.image.noise.sigma
        seed = noise_seed if noise_seed is not None else config.inference.rng_seed
        rng = galsim.BaseDeviate(seed)
        gs_image = galsim.Image(image)
        gs_image.addNoise(galsim.GaussianNoise(rng, sigma=noise_sigma))
        noisy_image = gs_image.array

        noise_map = jnp.full_like(noisy_image, noise_sigma**2)

        return Observation(
            image=jnp.array(noisy_image),
            noise_map=noise_map,
            psf_model=jax_psf,
        )
