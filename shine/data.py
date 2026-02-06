from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jnp

from shine import galaxy_utils, psf_utils
from shine.config import DistributionConfig, ShineConfig


@dataclass
class Observation:
    """Container for observational data.

    This dataclass stores all observational data required for shear inference,
    including the observed image, noise characteristics, and PSF model.

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


class DataLoader:
    """Loader for observational data and synthetic data generation.

    This class provides static methods for loading observational data from files
    or generating synthetic observations for testing and validation.
    """

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
        if config.data_path and config.data_path != "None":
            # TODO: Implement real data loading (Fits/HDF5)
            raise NotImplementedError(
                "Real data loading not yet implemented. Use synthetic generation."
            )
        else:
            print("No data path provided. Generating synthetic data...")
            return DataLoader.generate_synthetic(config)

    @staticmethod
    def generate_synthetic(config: ShineConfig) -> Observation:
        """Generate synthetic galaxy observations using GalSim.

        This function generates synthetic data for testing and validation purposes.
        Uses mean values from distribution configs for ground truth parameters when
        generating the true galaxy image, then adds noise to simulate observations.

        The PSF is pre-built as a JAX-GalSim object to avoid reconstruction on
        each MCMC iteration during inference.

        Args:
            config: SHINE configuration object containing simulation parameters.

        Returns:
            Observation object with synthetic noisy image, noise map, and JAX-GalSim PSF.
        """
        try:
            import galsim
            import jax_galsim
        except ImportError as e:
            raise ImportError(
                f"Failed to import GalSim libraries. "
                f"Please install galsim and jax_galsim: {e}"
            )

        # 1. Define PSF using PSF utilities (for synthetic generation)
        try:
            psf = psf_utils.get_psf(config.psf)
        except Exception as e:
            raise ValueError(
                f"Failed to create PSF with type='{config.psf.type}', sigma={config.psf.sigma}: {e}"
            )

        # Build JAX-GalSim PSF object for inference (to avoid reconstruction on each MCMC iteration)
        try:
            fft_size = config.image.fft_size
            gsparams = jax_galsim.GSParams(
                maximum_fft_size=fft_size, minimum_fft_size=fft_size
            )
            jax_psf = psf_utils.get_jax_psf(config.psf, gsparams=gsparams)
        except Exception as e:
            raise ValueError(
                f"Failed to create JAX-GalSim PSF with fft_size={fft_size}: {e}"
            )

        # 2. Define Galaxy (using mean values from config for "truth")
        def get_mean(param: Union[float, int, DistributionConfig]) -> float:
            """Extract mean value from a parameter that may be fixed or distributional.

            Args:
                param: Either a fixed numeric value or a DistributionConfig object.

            Returns:
                The mean value: the parameter itself if fixed, otherwise the distribution mean.
                For Uniform distributions, returns the midpoint (min + max) / 2.

            Raises:
                ValueError: If parameter is a DistributionConfig with unsupported type or missing required fields.
                TypeError: If parameter is neither numeric nor a DistributionConfig.
            """
            if isinstance(param, (float, int)):
                return float(param)

            # Validate that it's a DistributionConfig
            if not isinstance(param, DistributionConfig):
                raise TypeError(
                    f"Parameter must be a numeric value or DistributionConfig, got {type(param).__name__}"
                )

            # If it's a distribution config with a mean field
            if param.mean is not None:
                return param.mean

            # Handle Uniform distribution
            if param.type == "Uniform":
                if param.min is None or param.max is None:
                    raise ValueError(
                        f"Uniform distribution requires both 'min' and 'max' parameters, "
                        f"got min={param.min}, max={param.max}"
                    )
                return (param.min + param.max) / 2.0

            # Handle other distribution types without explicit mean
            if param.type == "Normal":
                raise ValueError(
                    f"Normal distribution requires 'mean' parameter, but it was not provided"
                )

            # Unknown distribution type
            raise ValueError(
                f"Unsupported distribution type '{param.type}' or missing required parameters. "
                f"Supported types: 'Normal' (requires mean, sigma), 'Uniform' (requires min, max)"
            )

        try:
            gal_flux = get_mean(config.gal.flux)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid galaxy flux configuration: {e}")

        try:
            gal_hlr = get_mean(config.gal.half_light_radius)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid galaxy half-light radius configuration: {e}")

        # Validate physical constraints
        if gal_flux <= 0:
            raise ValueError(
                f"Galaxy flux must be positive, got {gal_flux}. "
                f"Check the 'flux' parameter in your configuration."
            )

        if gal_hlr <= 0:
            raise ValueError(
                f"Galaxy half-light radius must be positive, got {gal_hlr}. "
                f"Check the 'half_light_radius' parameter in your configuration."
            )

        # Intrinsic ellipticity
        e1 = 0.0
        e2 = 0.0
        if config.gal.ellipticity is not None:
            try:
                e1 = get_mean(config.gal.ellipticity.e1)
                e2 = get_mean(config.gal.ellipticity.e2)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid ellipticity configuration: {e}")

            # Validate ellipticity magnitude
            e_mag = (e1**2 + e2**2) ** 0.5
            if e_mag >= 1.0:
                raise ValueError(
                    f"Ellipticity magnitude must be < 1, got |e| = {e_mag:.4f} from (e1={e1}, e2={e2}). "
                    f"Check the 'ellipticity' parameters in your configuration."
                )

        # Shear
        try:
            g1 = get_mean(config.gal.shear.g1)
            g2 = get_mean(config.gal.shear.g2)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid shear configuration: {e}")

        # Validate shear magnitude
        g_mag = (g1**2 + g2**2) ** 0.5
        if g_mag >= 1.0:
            raise ValueError(
                f"Shear magnitude must be < 1, got |g| = {g_mag:.4f} from (g1={g1}, g2={g2}). "
                f"Check the 'shear' parameters in your configuration."
            )
        try:
            shear = galsim.Shear(g1=g1, g2=g2)
        except Exception as e:
            raise ValueError(f"Failed to create GalSim Shear object with g1={g1}, g2={g2}: {e}")

        # Create Galaxy Object using galaxy utilities
        try:
            gal = galaxy_utils.get_galaxy(config.gal, gal_flux, gal_hlr, e1, e2)
            gal = gal.shear(shear)
        except Exception as e:
            raise ValueError(
                f"Failed to create galaxy with type='{config.gal.type}', "
                f"flux={gal_flux}, hlr={gal_hlr}, e1={e1}, e2={e2}: {e}"
            )

        # Convolve
        try:
            final = galsim.Convolve([gal, psf])
        except Exception as e:
            raise ValueError(f"Failed to convolve galaxy with PSF: {e}")

        # 3. Draw Image
        try:
            image = final.drawImage(
                nx=config.image.size_x,
                ny=config.image.size_y,
                scale=config.image.pixel_scale,
            ).array
        except Exception as e:
            raise ValueError(
                f"Failed to draw image with size ({config.image.size_x}, {config.image.size_y}) "
                f"and pixel_scale={config.image.pixel_scale}: {e}"
            )

        # 4. Add Noise
        try:
            rng_seed = config.inference.rng_seed
            rng = galsim.BaseDeviate(rng_seed)
            noise_sigma = config.image.noise.sigma
            noise = galsim.GaussianNoise(rng, sigma=noise_sigma)

            # GalSim image for noise addition
            gs_image = galsim.Image(image)
            gs_image.addNoise(noise)
            noisy_image = gs_image.array
        except Exception as e:
            raise ValueError(f"Failed to add noise with sigma={noise_sigma}: {e}")

        try:
            noise_map = jnp.ones_like(noisy_image) * (noise_sigma**2)
        except Exception as e:
            raise ValueError(f"Failed to create noise map: {e}")

        # Return JAX arrays and pre-built JAX-GalSim PSF object
        try:
            return Observation(
                image=jnp.array(noisy_image),
                noise_map=noise_map,
                psf_model=jax_psf,
            )
        except Exception as e:
            raise ValueError(f"Failed to create Observation object: {e}")
