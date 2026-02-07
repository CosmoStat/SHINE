"""Galaxy morphology modeling utilities."""
import logging
from types import ModuleType
from typing import Any, Optional

import galsim
import jax_galsim

from shine.config import GalaxyConfig

logger = logging.getLogger(__name__)


def _resolve_sersic_index(gal_config: GalaxyConfig) -> float:
    """Extract the Sersic index from config, handling distribution configs.

    Args:
        gal_config: Galaxy configuration containing the Sersic index.

    Returns:
        Numeric Sersic index value.

    Raises:
        ValueError: If Sersic index is not provided.
    """
    if gal_config.n is None:
        raise ValueError("Sersic galaxy requires n parameter")
    if isinstance(gal_config.n, (float, int)):
        return float(gal_config.n)
    return gal_config.n.mean


def _apply_ellipticity(gal: Any, e1: float, e2: float) -> Any:
    """Apply intrinsic ellipticity to a galaxy object if non-zero.

    Args:
        gal: GalSim or JAX-GalSim galaxy object.
        e1: First ellipticity component.
        e2: Second ellipticity component.

    Returns:
        Galaxy object with ellipticity applied (or unchanged if both zero).
    """
    if isinstance(e1, (int, float)) and isinstance(e2, (int, float)):
        if e1 == 0.0 and e2 == 0.0:
            return gal
    return gal.shear(e1=e1, e2=e2)


def _build_galaxy(
    gal_config: GalaxyConfig,
    lib: ModuleType,
    flux: float,
    half_light_radius: float,
    e1: float = 0.0,
    e2: float = 0.0,
    **kwargs: Any,
) -> Any:
    """Create a galaxy object using the specified GalSim-compatible library.

    Args:
        gal_config: Galaxy configuration specifying profile type and parameters.
        lib: GalSim-compatible module (galsim or jax_galsim).
        flux: Galaxy flux.
        half_light_radius: Half-light radius in arcseconds.
        e1: First component of intrinsic ellipticity.
        e2: Second component of intrinsic ellipticity.
        **kwargs: Additional keyword arguments passed to the constructor (e.g., gsparams).

    Returns:
        Galaxy object with ellipticity applied.

    Raises:
        NotImplementedError: If the galaxy type is not supported.
    """
    common = dict(half_light_radius=half_light_radius, flux=flux, **kwargs)

    if gal_config.type == "Exponential":
        gal = lib.Exponential(**common)
    elif gal_config.type == "DeVaucouleurs":
        if lib is jax_galsim:
            raise NotImplementedError(
                "Galaxy type 'DeVaucouleurs' not supported in JAX-GalSim"
            )
        gal = lib.DeVaucouleurs(**common)
    elif gal_config.type == "Sersic":
        if lib is jax_galsim:
            raise NotImplementedError(
                "Galaxy type 'Sersic' not supported in JAX-GalSim"
            )
        else:
            n_value = _resolve_sersic_index(gal_config)
            gal = lib.Sersic(n=n_value, **common)
    else:
        lib_name = "JAX-GalSim" if lib is jax_galsim else "GalSim"
        raise NotImplementedError(
            f"Galaxy type '{gal_config.type}' not supported in {lib_name}"
        )

    return _apply_ellipticity(gal, e1, e2)


def get_galaxy(
    gal_config: GalaxyConfig,
    flux: float,
    half_light_radius: float,
    e1: float = 0.0,
    e2: float = 0.0,
) -> galsim.GSObject:
    """Create a GalSim galaxy object from configuration.

    Args:
        gal_config: Galaxy configuration specifying profile type and parameters.
        flux: Galaxy flux.
        half_light_radius: Half-light radius in arcseconds.
        e1: First component of intrinsic ellipticity.
        e2: Second component of intrinsic ellipticity.

    Returns:
        GalSim galaxy object with ellipticity applied.

    Raises:
        NotImplementedError: If the galaxy type is not supported.
    """
    return _build_galaxy(gal_config, galsim, flux, half_light_radius, e1, e2)


def get_jax_galaxy(
    gal_config: GalaxyConfig,
    flux: float,
    half_light_radius: float,
    e1: float = 0.0,
    e2: float = 0.0,
    gsparams: Optional[Any] = None,
) -> jax_galsim.GSObject:
    """Create a JAX-GalSim galaxy object from configuration.

    Args:
        gal_config: Galaxy configuration specifying profile type and parameters.
        flux: Galaxy flux.
        half_light_radius: Half-light radius in arcseconds.
        e1: First component of intrinsic ellipticity.
        e2: Second component of intrinsic ellipticity.
        gsparams: Optional GSParams for controlling rendering.

    Returns:
        JAX-GalSim galaxy object with ellipticity applied.

    Raises:
        NotImplementedError: If the galaxy type is not supported.
    """
    return _build_galaxy(
        gal_config, jax_galsim, flux, half_light_radius, e1, e2, gsparams=gsparams
    )
