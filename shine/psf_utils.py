"""PSF modeling utilities."""
import logging
from typing import Optional, Any

import galsim
import jax_galsim

from shine.config import PSFConfig

logger = logging.getLogger(__name__)


def get_psf(psf_config: PSFConfig) -> galsim.GSObject:
    """
    Create a PSF object based on the PSF configuration.

    Args:
        psf_config: PSF configuration

    Returns:
        GalSim PSF object
    """
    if psf_config.type == "Gaussian":
        return galsim.Gaussian(sigma=psf_config.sigma)
    elif psf_config.type == "Moffat":
        if psf_config.beta is None:
            logger.error("Moffat PSF requires beta parameter")
            raise ValueError("Moffat PSF requires beta parameter")
        return galsim.Moffat(beta=psf_config.beta, fwhm=psf_config.sigma)
    else:
        logger.error(f"PSF type {psf_config.type} not supported")
        raise NotImplementedError(f"PSF type {psf_config.type} not supported")


def get_jax_psf(
    psf_config: PSFConfig, gsparams: Optional[Any] = None
) -> jax_galsim.GSObject:
    """
    Create a JAX-GalSim PSF object based on the PSF configuration.

    Args:
        psf_config: PSF configuration
        gsparams: Optional GSParams for controlling rendering

    Returns:
        JAX-GalSim PSF object
    """
    if psf_config.type == "Gaussian":
        return jax_galsim.Gaussian(sigma=psf_config.sigma, gsparams=gsparams)
    elif psf_config.type == "Moffat":
        if psf_config.beta is None:
            logger.error("Moffat PSF requires beta parameter")
            raise ValueError("Moffat PSF requires beta parameter")
        return jax_galsim.Moffat(
            beta=psf_config.beta, fwhm=psf_config.sigma, gsparams=gsparams
        )
    else:
        logger.error(f"PSF type {psf_config.type} not supported")
        raise NotImplementedError(f"PSF type {psf_config.type} not supported")
