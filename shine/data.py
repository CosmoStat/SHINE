import logging
from dataclasses import dataclass
from typing import Any, Union

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
        # Check for catalog-based generation
        if config.catalog is not None:
            logger.info("Catalog configuration detected. Generating from catalog...")
            return DataLoader.generate_from_catalog(config)

        # Guard against YAML parsing "None" as the string "None" instead of null
        if config.data_path and config.data_path != "None":
            raise NotImplementedError(
                "Real data loading not yet implemented. Use synthetic generation."
            )
        logger.info("No data path provided. Generating synthetic data...")
        return DataLoader.generate_synthetic(config)

    @staticmethod
    def generate_synthetic(config: ShineConfig) -> Observation:
        """Generate synthetic galaxy observations using GalSim.

        Uses mean values from distribution configs for ground truth parameters,
        renders the galaxy image, and adds noise to simulate observations.
        The PSF is pre-built as a JAX-GalSim object to avoid reconstruction
        on each MCMC iteration during inference.

        Args:
            config: SHINE configuration object containing simulation parameters.

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

        # Intrinsic ellipticity
        e1 = 0.0
        e2 = 0.0
        if config.gal.ellipticity is not None:
            e1 = get_mean(config.gal.ellipticity.e1)
            e2 = get_mean(config.gal.ellipticity.e2)
            e_mag = (e1**2 + e2**2) ** 0.5
            _validate_magnitude(e_mag, 1.0, "Ellipticity", f"(e1={e1}, e2={e2})")

        # Shear
        g1 = get_mean(config.gal.shear.g1)
        g2 = get_mean(config.gal.shear.g2)
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

        # Add noise
        noise_sigma = config.image.noise.sigma
        rng = galsim.BaseDeviate(config.inference.rng_seed)
        gs_image = galsim.Image(image)
        gs_image.addNoise(galsim.GaussianNoise(rng, sigma=noise_sigma))
        noisy_image = gs_image.array

        noise_map = jnp.full_like(noisy_image, noise_sigma**2)

        return Observation(
            image=jnp.array(noisy_image),
            noise_map=noise_map,
            psf_model=jax_psf,
        )

    @staticmethod
    def generate_from_catalog(config: ShineConfig) -> Observation:
        """Generate scene from external galaxy catalog.

        Loads galaxies from a catalog (CosmoDC2, CatSim, etc.), renders them
        at their catalog positions with catalog morphologies, applies global
        shear, and adds noise.

        Args:
            config: SHINE configuration with catalog section.

        Returns:
            Observation object with catalog-based scene.

        Raises:
            ValueError: If catalog config is missing or invalid.
        """
        import galsim
        import jax_galsim

        from shine.catalogs import get_catalog_loader

        if config.catalog is None:
            raise ValueError("Catalog configuration is required")

        # Load catalog
        logger.info(f"Loading catalog: {config.catalog.type}")
        catalog_loader = get_catalog_loader(config.catalog.type)
        catalog_loader.load(config.catalog.path)

        # Sample galaxies for postage stamp
        galaxies = catalog_loader.sample_postage_stamp(
            center_ra=config.catalog.center_ra,
            center_dec=config.catalog.center_dec,
            size_arcmin=config.catalog.size_arcmin,
            pixel_scale=config.image.pixel_scale,
            image_size=(config.image.size_x, config.image.size_y),
            magnitude_limit=config.catalog.magnitude_limit,
        )

        logger.info(f"Rendering {len(galaxies)} galaxies from catalog")

        # Build PSF
        psf = psf_utils.get_psf(config.psf)
        fft_size = config.image.fft_size
        gsparams = jax_galsim.GSParams(
            maximum_fft_size=fft_size, minimum_fft_size=fft_size
        )
        jax_psf = psf_utils.get_jax_psf(config.psf, gsparams=gsparams)

        # Get global shear to apply
        g1 = get_mean(config.gal.shear.g1)
        g2 = get_mean(config.gal.shear.g2)
        global_shear = galsim.Shear(g1=g1, g2=g2)

        # Initialize blank image
        image = galsim.Image(
            config.image.size_x,
            config.image.size_y,
            scale=config.image.pixel_scale,
        )

        # Render each galaxy
        use_bulge_disk = config.catalog.use_bulge_disk and galaxies.has_bulge_disk()

        for i in range(len(galaxies)):
            # Get galaxy properties
            x = float(galaxies.x[i])
            y = float(galaxies.y[i])
            flux = float(galaxies.flux[i])
            hlr = float(galaxies.half_light_radius[i])
            e1 = float(galaxies.e1[i])
            e2 = float(galaxies.e2[i])

            # Skip if outside image bounds (with margin)
            margin = 5  # pixels
            if (
                x < margin
                or x > config.image.size_x - margin
                or y < margin
                or y > config.image.size_y - margin
            ):
                continue

            try:
                if use_bulge_disk:
                    gal = DataLoader._render_bulge_disk_galaxy(
                        galaxies, i, e1, e2, global_shear, psf
                    )
                else:
                    gal = DataLoader._render_single_component_galaxy(
                        flux, hlr, e1, e2, global_shear, psf, config.gal.type
                    )

                # Draw stamp at galaxy position
                stamp = gal.drawImage(
                    nx=32,
                    ny=32,
                    scale=config.image.pixel_scale,
                )

                # Calculate bounds (GalSim uses 1-indexed coordinates)
                ix = int(x) + 1  # Convert to 1-indexed
                iy = int(y) + 1
                stamp_center = galsim.PositionI(ix, iy)
                stamp.setCenter(stamp_center)

                # Add to image
                bounds = stamp.bounds & image.bounds
                if bounds.isDefined():
                    image[bounds] += stamp[bounds]

            except Exception as e:
                logger.warning(f"Failed to render galaxy {i}: {e}")
                continue

        # Add noise
        noise_sigma = config.image.noise.sigma
        rng = galsim.BaseDeviate(config.inference.rng_seed)
        image.addNoise(galsim.GaussianNoise(rng, sigma=noise_sigma))

        noise_map = jnp.full(
            (config.image.size_y, config.image.size_x), noise_sigma**2
        )

        return Observation(
            image=jnp.array(image.array),
            noise_map=noise_map,
            psf_model=jax_psf,
        )

    @staticmethod
    def _render_single_component_galaxy(
        flux: float,
        hlr: float,
        e1: float,
        e2: float,
        shear: Any,
        psf: Any,
        profile_type: str = "Exponential",
    ) -> Any:
        """Render a single-component galaxy.

        Args:
            flux: Total flux.
            hlr: Half-light radius in arcsec.
            e1: First ellipticity component.
            e2: Second ellipticity component.
            shear: GalSim Shear object to apply.
            psf: GalSim PSF object.
            profile_type: Galaxy profile type.

        Returns:
            GalSim object ready to draw.
        """
        import galsim

        # Create profile
        if profile_type == "Exponential":
            gal = galsim.Exponential(flux=flux, half_light_radius=hlr)
        elif profile_type == "DeVaucouleurs":
            gal = galsim.DeVaucouleurs(flux=flux, half_light_radius=hlr)
        elif profile_type == "Sersic":
            # Default to n=1 (Exponential) if not specified
            gal = galsim.Sersic(n=1.0, flux=flux, half_light_radius=hlr)
        else:
            # Fallback to Exponential
            gal = galsim.Exponential(flux=flux, half_light_radius=hlr)

        # Apply intrinsic ellipticity
        if e1 != 0 or e2 != 0:
            gal = gal.shear(e1=e1, e2=e2)

        # Apply global shear
        gal = gal.shear(shear)

        # Convolve with PSF
        return galsim.Convolve([gal, psf])

    @staticmethod
    def _render_bulge_disk_galaxy(
        galaxies: Any,
        index: int,
        e1: float,
        e2: float,
        shear: Any,
        psf: Any,
    ) -> Any:
        """Render a bulge+disk galaxy.

        Args:
            galaxies: GalaxyProperties object with bulge/disk info.
            index: Galaxy index.
            e1: Intrinsic ellipticity component 1.
            e2: Intrinsic ellipticity component 2.
            shear: GalSim Shear object to apply.
            psf: GalSim PSF object.

        Returns:
            GalSim object ready to draw.
        """
        import galsim

        # Extract bulge/disk properties
        bulge_flux = float(galaxies.bulge_flux[index])
        disk_flux = float(galaxies.disk_flux[index])
        bulge_hlr = float(galaxies.bulge_hlr[index])
        disk_hlr = float(galaxies.disk_hlr[index])

        # Sersic indices (default to standard values if not provided)
        if galaxies.bulge_n is not None:
            bulge_n = float(galaxies.bulge_n[index])
        else:
            bulge_n = 4.0  # DeVaucouleurs

        if galaxies.disk_n is not None:
            disk_n = float(galaxies.disk_n[index])
        else:
            disk_n = 1.0  # Exponential

        # Create bulge (typically DeVaucouleurs, n=4)
        if bulge_n == 4.0:
            bulge = galsim.DeVaucouleurs(flux=bulge_flux, half_light_radius=bulge_hlr)
        else:
            bulge = galsim.Sersic(n=bulge_n, flux=bulge_flux, half_light_radius=bulge_hlr)

        # Create disk (typically Exponential, n=1)
        if disk_n == 1.0:
            disk = galsim.Exponential(flux=disk_flux, half_light_radius=disk_hlr)
        else:
            disk = galsim.Sersic(n=disk_n, flux=disk_flux, half_light_radius=disk_hlr)

        # Apply intrinsic ellipticity to both components
        if e1 != 0 or e2 != 0:
            bulge = bulge.shear(e1=e1, e2=e2)
            disk = disk.shear(e1=e1, e2=e2)

        # Combine bulge and disk
        gal = bulge + disk

        # Apply global shear
        gal = gal.shear(shear)

        # Convolve with PSF
        return galsim.Convolve([gal, psf])
