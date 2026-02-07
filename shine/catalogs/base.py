"""Base classes and data structures for catalog loading.

This module provides the abstract interface for catalog loaders and standardized
data structures for galaxy properties.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict

import jax.numpy as jnp


@dataclass
class SkyRegion:
    """Defines a rectangular region on the sky.

    Attributes:
        center_ra: Center Right Ascension in degrees.
        center_dec: Center Declination in degrees.
        size_arcmin: Field size in arcminutes (assumes square field).
    """
    center_ra: float
    center_dec: float
    size_arcmin: float

    def get_bounds(self) -> tuple[float, float, float, float]:
        """Get RA/Dec bounds of the region.

        Returns:
            Tuple of (ra_min, ra_max, dec_min, dec_max) in degrees.
        """
        # Convert size to degrees
        size_deg = self.size_arcmin / 60.0
        half_size = size_deg / 2.0

        # Simple rectangular bounds (good approximation for small fields)
        ra_min = self.center_ra - half_size / jnp.cos(jnp.radians(self.center_dec))
        ra_max = self.center_ra + half_size / jnp.cos(jnp.radians(self.center_dec))
        dec_min = self.center_dec - half_size
        dec_max = self.center_dec + half_size

        return ra_min, ra_max, dec_min, dec_max


@dataclass
class GalaxyProperties:
    """Standardized galaxy properties for rendering.

    This dataclass provides a uniform interface between catalog loaders and the
    rendering pipeline. It supports both single-component and bulge+disk morphologies.

    Attributes:
        x: Pixel x-coordinates (array).
        y: Pixel y-coordinates (array).
        flux: Total flux in counts (array). Used for single-component fallback.
        half_light_radius: Half-light radius in arcseconds (array).
        e1: First ellipticity component (array).
        e2: Second ellipticity component (array).
        bulge_flux: Bulge component flux (optional array).
        disk_flux: Disk component flux (optional array).
        bulge_hlr: Bulge half-light radius in arcseconds (optional array).
        disk_hlr: Disk half-light radius in arcseconds (optional array).
        bulge_n: Bulge Sersic index (optional array).
        disk_n: Disk Sersic index (optional array).
        z: Redshift (optional array).
        g1_true: True shear component 1 (optional array, for validation).
        g2_true: True shear component 2 (optional array, for validation).
        flux_bands: Multi-band fluxes (optional dict, for chromatic rendering).
    """
    # Position (pixel coordinates)
    x: jnp.ndarray
    y: jnp.ndarray

    # Single-component properties
    flux: jnp.ndarray
    half_light_radius: jnp.ndarray
    e1: jnp.ndarray
    e2: jnp.ndarray

    # Bulge+disk decomposition (optional)
    bulge_flux: Optional[jnp.ndarray] = None
    disk_flux: Optional[jnp.ndarray] = None
    bulge_hlr: Optional[jnp.ndarray] = None
    disk_hlr: Optional[jnp.ndarray] = None
    bulge_n: Optional[jnp.ndarray] = None  # Sersic index (typically 4 for DeVaucouleurs)
    disk_n: Optional[jnp.ndarray] = None   # Sersic index (typically 1 for Exponential)

    # Additional properties
    z: Optional[jnp.ndarray] = None
    g1_true: Optional[jnp.ndarray] = None
    g2_true: Optional[jnp.ndarray] = None
    flux_bands: Optional[Dict[str, jnp.ndarray]] = None

    def __len__(self) -> int:
        """Return the number of galaxies."""
        return len(self.x)

    def has_bulge_disk(self) -> bool:
        """Check if bulge+disk decomposition is available."""
        return (
            self.bulge_flux is not None
            and self.disk_flux is not None
            and self.bulge_hlr is not None
            and self.disk_hlr is not None
        )


class CatalogLoader(ABC):
    """Abstract base class for catalog loaders.

    Concrete implementations should load catalogs from specific formats
    (e.g., CosmoDC2, CatSim, Flagship2) and convert them to the standardized
    GalaxyProperties format.
    """

    @abstractmethod
    def load(self, path: str) -> None:
        """Load catalog from file or directory.

        Args:
            path: Path to catalog file or directory.
        """
        pass

    @abstractmethod
    def sample_region(
        self,
        sky_region: SkyRegion,
        magnitude_limit: Optional[float] = None,
    ) -> GalaxyProperties:
        """Sample galaxies within a sky region.

        Args:
            sky_region: Sky region to sample from.
            magnitude_limit: Optional magnitude cut (e.g., i < 25).

        Returns:
            GalaxyProperties object containing sampled galaxies.
        """
        pass

    def sample_postage_stamp(
        self,
        center_ra: float,
        center_dec: float,
        size_arcmin: float,
        pixel_scale: float,
        image_size: tuple[int, int],
        magnitude_limit: Optional[float] = None,
    ) -> GalaxyProperties:
        """Sample galaxies for a postage stamp image.

        This is the main interface used by the rendering pipeline.

        Args:
            center_ra: Center RA in degrees.
            center_dec: Center Dec in degrees.
            size_arcmin: Field size in arcminutes.
            pixel_scale: Pixel scale in arcsec/pixel.
            image_size: Image dimensions (width, height) in pixels.
            magnitude_limit: Optional magnitude cut.

        Returns:
            GalaxyProperties with pixel coordinates relative to image center.
        """
        # Define sky region
        sky_region = SkyRegion(
            center_ra=center_ra,
            center_dec=center_dec,
            size_arcmin=size_arcmin,
        )

        # Sample galaxies
        galaxies = self.sample_region(sky_region, magnitude_limit)

        return galaxies
