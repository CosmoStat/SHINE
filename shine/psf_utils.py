"""PSF modeling utilities."""
import galsim
import jax_galsim
from shine.config import PSFConfig


def get_psf(psf_config: PSFConfig):
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
            raise ValueError("Moffat PSF requires beta parameter")
        return galsim.Moffat(beta=psf_config.beta, fwhm=psf_config.sigma)
    else:
        raise NotImplementedError(f"PSF type {psf_config.type} not supported")


def get_jax_psf(psf_config: PSFConfig, gsparams=None):
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
            raise ValueError("Moffat PSF requires beta parameter")
        return jax_galsim.Moffat(beta=psf_config.beta, fwhm=psf_config.sigma, gsparams=gsparams)
    else:
        raise NotImplementedError(f"PSF type {psf_config.type} not supported")
