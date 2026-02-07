"""Utility functions for catalog processing.

This module provides coordinate conversions, ellipticity calculations,
and magnitude-to-flux conversions used by catalog loaders.
"""

import jax.numpy as jnp
import numpy as np


def ra_dec_to_pixels(
    ra: np.ndarray,
    dec: np.ndarray,
    center_ra: float,
    center_dec: float,
    pixel_scale: float,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert RA/Dec coordinates to pixel coordinates using tangent plane projection.

    This is a simple gnomonic projection suitable for small fields (<1 degree).
    For larger fields, use a full WCS transformation.

    Args:
        ra: Right Ascension in degrees (array).
        dec: Declination in degrees (array).
        center_ra: Center RA in degrees.
        center_dec: Center Dec in degrees.
        pixel_scale: Pixel scale in arcsec/pixel.
        image_size: Image dimensions (width, height) in pixels.

    Returns:
        Tuple of (x, y) pixel coordinates (arrays). Origin at image center.
    """
    # Convert to radians
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    center_ra_rad = np.radians(center_ra)
    center_dec_rad = np.radians(center_dec)

    # Tangent plane projection (gnomonic)
    # See: https://mathworld.wolfram.com/GnomonicProjection.html
    cos_c = (
        np.sin(dec_rad) * np.sin(center_dec_rad)
        + np.cos(dec_rad) * np.cos(center_dec_rad) * np.cos(ra_rad - center_ra_rad)
    )

    # Angular offsets in radians
    xi = (
        np.cos(dec_rad) * np.sin(ra_rad - center_ra_rad) / cos_c
    )
    eta = (
        (np.sin(dec_rad) * np.cos(center_dec_rad)
         - np.cos(dec_rad) * np.sin(center_dec_rad) * np.cos(ra_rad - center_ra_rad))
        / cos_c
    )

    # Convert to arcseconds, then to pixels
    xi_arcsec = np.degrees(xi) * 3600.0  # radians -> degrees -> arcsec
    eta_arcsec = np.degrees(eta) * 3600.0

    x_pixels = xi_arcsec / pixel_scale
    y_pixels = eta_arcsec / pixel_scale

    # Offset to image center (0, 0) at center of image
    x_pixels += image_size[0] / 2.0
    y_pixels += image_size[1] / 2.0

    return x_pixels, y_pixels


def compute_ellipticity(
    semi_major: np.ndarray,
    semi_minor: np.ndarray,
    position_angle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert (a, b, PA) to ellipticity components (e1, e2).

    Uses the reduced shear definition:
        e = (a - b) / (a + b)
        e1 = e * cos(2 * PA)
        e2 = e * sin(2 * PA)

    Args:
        semi_major: Semi-major axis length (array).
        semi_minor: Semi-minor axis length (array).
        position_angle: Position angle in degrees, measured from +x axis (array).

    Returns:
        Tuple of (e1, e2) ellipticity components (arrays).
    """
    # Avoid division by zero
    denominator = semi_major + semi_minor
    ellipticity = np.where(
        denominator > 0,
        (semi_major - semi_minor) / denominator,
        0.0,
    )

    # Convert PA to radians
    pa_rad = np.radians(position_angle)

    # Ellipticity components
    e1 = ellipticity * np.cos(2.0 * pa_rad)
    e2 = ellipticity * np.sin(2.0 * pa_rad)

    return e1, e2


def magnitude_to_flux(
    magnitude: np.ndarray,
    zeropoint: float = 30.0,
) -> np.ndarray:
    """Convert AB magnitude to flux in counts.

    Uses the relation: flux = 10^((zeropoint - magnitude) / 2.5)

    Args:
        magnitude: AB magnitude (array).
        zeropoint: Magnitude zeropoint (default 30.0 for standard calibration).

    Returns:
        Flux in counts (array).
    """
    return 10.0 ** ((zeropoint - magnitude) / 2.5)


def flux_to_magnitude(
    flux: np.ndarray,
    zeropoint: float = 30.0,
) -> np.ndarray:
    """Convert flux in counts to AB magnitude.

    Uses the relation: magnitude = zeropoint - 2.5 * log10(flux)

    Args:
        flux: Flux in counts (array).
        zeropoint: Magnitude zeropoint (default 30.0 for standard calibration).

    Returns:
        AB magnitude (array).
    """
    return zeropoint - 2.5 * np.log10(flux)
