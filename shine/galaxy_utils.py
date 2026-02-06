"""Galaxy morphology modeling utilities."""
import logging
from typing import Optional, Any

import galsim
import jax_galsim

from shine.config import GalaxyConfig

logger = logging.getLogger(__name__)


def get_galaxy(
    gal_config: GalaxyConfig,
    flux: float,
    half_light_radius: float,
    e1: float = 0.0,
    e2: float = 0.0,
) -> galsim.GSObject:
    """
    Create a galaxy object based on the galaxy configuration.

    Args:
        gal_config: Galaxy configuration
        flux: Galaxy flux
        half_light_radius: Half-light radius
        e1: Intrinsic ellipticity component 1
        e2: Intrinsic ellipticity component 2

    Returns:
        GalSim galaxy object
    """
    if gal_config.type == "Exponential":
        gal = galsim.Exponential(half_light_radius=half_light_radius, flux=flux)
    elif gal_config.type == "DeVaucouleurs":
        gal = galsim.DeVaucouleurs(half_light_radius=half_light_radius, flux=flux)
    elif gal_config.type == "Sersic":
        if gal_config.n is None:
            raise ValueError("Sersic galaxy requires n parameter")
        # Handle if n is a distribution config
        n_value = (
            gal_config.n
            if isinstance(gal_config.n, (float, int))
            else gal_config.n.mean
        )
        gal = galsim.Sersic(n=n_value, half_light_radius=half_light_radius, flux=flux)
    else:
        raise NotImplementedError(f"Galaxy type {gal_config.type} not supported")

    # Apply intrinsic ellipticity
    if e1 != 0.0 or e2 != 0.0:
        gal = gal.shear(e1=e1, e2=e2)

    return gal


def get_jax_galaxy(
    gal_config: GalaxyConfig,
    flux: float,
    half_light_radius: float,
    e1: float = 0.0,
    e2: float = 0.0,
    gsparams: Optional[Any] = None,
) -> jax_galsim.GSObject:
    """
    Create a JAX-GalSim galaxy object based on the galaxy configuration.

    Args:
        gal_config: Galaxy configuration
        flux: Galaxy flux
        half_light_radius: Half-light radius
        e1: Intrinsic ellipticity component 1
        e2: Intrinsic ellipticity component 2
        gsparams: Optional GSParams for controlling rendering

    Returns:
        JAX-GalSim galaxy object
    """
    if gal_config.type == "Exponential":
        gal = jax_galsim.Exponential(
            half_light_radius=half_light_radius, flux=flux, gsparams=gsparams
        )
    elif gal_config.type == "DeVaucouleurs":
        gal = jax_galsim.DeVaucouleurs(
            half_light_radius=half_light_radius, flux=flux, gsparams=gsparams
        )
    elif gal_config.type == "Sersic":
        # Note: Sersic not yet available in jax_galsim, use Exponential as fallback
        # TODO: Replace with Sersic when available
        logger.warning(
            "Sersic profile not available in JAX-GalSim, falling back to Exponential profile"
        )
        gal = jax_galsim.Exponential(
            half_light_radius=half_light_radius, flux=flux, gsparams=gsparams
        )
    else:
        raise NotImplementedError(
            f"Galaxy type {gal_config.type} not supported in JAX-GalSim"
        )

    # Apply intrinsic ellipticity
    if e1 != 0.0 or e2 != 0.0:
        gal = gal.shear(e1=e1, e2=e2)

    return gal
