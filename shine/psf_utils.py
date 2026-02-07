"""PSF modeling utilities."""
from types import ModuleType
from typing import Any, Optional

import galsim
import jax_galsim

from shine.config import PSFConfig


def _build_psf(
    psf_config: PSFConfig, lib: ModuleType, **kwargs: Any
) -> Any:
    """Create a PSF object using the specified GalSim-compatible library.

    Args:
        psf_config: PSF configuration specifying type and parameters.
        lib: GalSim-compatible module (galsim or jax_galsim).
        **kwargs: Additional keyword arguments passed to the constructor (e.g., gsparams).

    Returns:
        PSF object from the specified library.

    Raises:
        NotImplementedError: If the PSF type is not supported.
        ValueError: If Moffat PSF is missing the beta parameter.
    """
    if psf_config.type == "Gaussian":
        return lib.Gaussian(sigma=psf_config.sigma, **kwargs)

    if psf_config.type == "Moffat":
        if psf_config.beta is None:
            raise ValueError("Moffat PSF requires beta parameter")
        return lib.Moffat(beta=psf_config.beta, fwhm=psf_config.sigma, **kwargs)

    raise NotImplementedError(f"PSF type '{psf_config.type}' not supported")


def get_psf(psf_config: PSFConfig) -> galsim.GSObject:
    """Create a GalSim PSF object from configuration.

    Args:
        psf_config: PSF configuration specifying type and parameters.

    Returns:
        GalSim PSF object.

    Raises:
        NotImplementedError: If the PSF type is not supported.
    """
    return _build_psf(psf_config, galsim)


def get_jax_psf(
    psf_config: PSFConfig, gsparams: Optional[Any] = None
) -> jax_galsim.GSObject:
    """Create a JAX-GalSim PSF object from configuration.

    Args:
        psf_config: PSF configuration specifying type and parameters.
        gsparams: Optional GSParams for controlling rendering.

    Returns:
        JAX-GalSim PSF object.

    Raises:
        NotImplementedError: If the PSF type is not supported.
    """
    return _build_psf(psf_config, jax_galsim, gsparams=gsparams)
