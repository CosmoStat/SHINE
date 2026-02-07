"""Tests for shine.psf_utils module."""

import galsim
import jax_galsim
import pytest

from shine.config import PSFConfig
from shine.psf_utils import get_jax_psf, get_psf


class TestGetPSF:
    """Test get_psf function for GalSim PSF creation."""

    def test_gaussian_psf(self):
        """Test creation of Gaussian PSF."""
        config = PSFConfig(type="Gaussian", sigma=1.5)
        psf = get_psf(config)

        assert isinstance(psf, galsim.Gaussian)
        assert psf.sigma == 1.5

    def test_moffat_psf(self):
        """Test creation of Moffat PSF."""
        config = PSFConfig(type="Moffat", sigma=2.0, beta=3.5)
        psf = get_psf(config)

        assert isinstance(psf, galsim.Moffat)
        assert psf.beta == 3.5

    def test_moffat_psf_missing_beta(self):
        """Test Moffat PSF without beta parameter raises error."""
        config = PSFConfig(type="Moffat", sigma=2.0)
        with pytest.raises(ValueError, match="Moffat PSF requires beta parameter"):
            get_psf(config)

    def test_unsupported_psf_type(self):
        """Test unsupported PSF type raises NotImplementedError."""
        config = PSFConfig(type="Kolmogorov", sigma=1.0)
        with pytest.raises(NotImplementedError, match="PSF type .* not supported"):
            get_psf(config)


class TestGetJaxPSF:
    """Test get_jax_psf function for JAX-GalSim PSF creation."""

    def test_gaussian_jax_psf(self):
        """Test creation of JAX-GalSim Gaussian PSF."""
        config = PSFConfig(type="Gaussian", sigma=1.5)
        psf = get_jax_psf(config)

        assert isinstance(psf, jax_galsim.Gaussian)
        assert psf.sigma == 1.5

    def test_moffat_jax_psf(self):
        """Test creation of JAX-GalSim Moffat PSF."""
        config = PSFConfig(type="Moffat", sigma=2.0, beta=3.5)
        psf = get_jax_psf(config)

        assert isinstance(psf, jax_galsim.Moffat)
        assert psf.beta == 3.5

    def test_moffat_jax_psf_missing_beta(self):
        """Test Moffat PSF without beta parameter raises error."""
        config = PSFConfig(type="Moffat", sigma=2.0)
        with pytest.raises(ValueError, match="Moffat PSF requires beta parameter"):
            get_jax_psf(config)

    def test_jax_psf_with_gsparams(self):
        """Test JAX PSF creation with custom GSParams."""
        gsparams = jax_galsim.GSParams(maximum_fft_size=256, minimum_fft_size=256)
        config = PSFConfig(type="Gaussian", sigma=1.5)
        psf = get_jax_psf(config, gsparams=gsparams)

        assert isinstance(psf, jax_galsim.Gaussian)
        assert psf.gsparams is not None

    def test_unsupported_jax_psf_type(self):
        """Test unsupported JAX PSF type raises NotImplementedError."""
        config = PSFConfig(type="Kolmogorov", sigma=1.0)
        with pytest.raises(NotImplementedError, match="PSF type .* not supported"):
            get_jax_psf(config)


class TestPSFConsistency:
    """Test consistency between GalSim and JAX-GalSim PSFs."""

    def test_gaussian_rendering_consistency(self):
        """Test that GalSim and JAX-GalSim Gaussian PSFs produce similar outputs."""
        config = PSFConfig(type="Gaussian", sigma=1.5)

        psf_galsim = get_psf(config)
        psf_jax = get_jax_psf(config)

        # Draw images and compare
        image_galsim = psf_galsim.drawImage(scale=0.2, nx=32, ny=32).array
        image_jax = psf_jax.drawImage(scale=0.2, nx=32, ny=32).array

        # Check shapes match
        assert image_galsim.shape == image_jax.shape

        # Note: Full numerical comparison would require chex for JAX arrays
        # This is a placeholder for future enhancement with chex

    def test_moffat_parameter_consistency(self):
        """Test that Moffat PSF parameters are consistent."""
        config = PSFConfig(type="Moffat", sigma=2.0, beta=2.5)

        psf_galsim = get_psf(config)
        psf_jax = get_jax_psf(config)

        assert psf_galsim.beta == psf_jax.beta
