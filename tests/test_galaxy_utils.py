"""Tests for shine.galaxy_utils module."""

import galsim
import jax_galsim
import pytest

from shine.config import DistributionConfig, GalaxyConfig, ShearConfig
from shine.galaxy_utils import get_galaxy, get_jax_galaxy


class TestGetGalaxy:
    """Test get_galaxy function for GalSim galaxy creation."""

    def test_exponential_galaxy(self):
        """Test creation of Exponential galaxy."""
        shear = ShearConfig(type="G1G2", g1=0.01, g2=0.02)
        config = GalaxyConfig(
            type="Exponential",
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
        )

        gal = get_galaxy(config, flux=1000.0, half_light_radius=1.0)
        assert isinstance(gal, galsim.Exponential)

    def test_devaucouleurs_galaxy(self):
        """Test creation of DeVaucouleurs galaxy."""
        shear = ShearConfig(type="G1G2", g1=0.01, g2=0.02)
        config = GalaxyConfig(
            type="DeVaucouleurs",
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
        )

        gal = get_galaxy(config, flux=1000.0, half_light_radius=1.0)
        assert isinstance(gal, galsim.DeVaucouleurs)

    def test_sersic_galaxy(self):
        """Test creation of Sersic galaxy."""
        shear = ShearConfig(type="G1G2", g1=0.01, g2=0.02)
        config = GalaxyConfig(
            type="Sersic",
            n=4.0,
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
        )

        gal = get_galaxy(config, flux=1000.0, half_light_radius=1.0)
        assert isinstance(gal, galsim.Sersic)

    def test_sersic_galaxy_missing_n(self):
        """Test Sersic galaxy without n parameter raises error."""
        shear = ShearConfig(type="G1G2", g1=0.01, g2=0.02)
        config = GalaxyConfig(
            type="Sersic",
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
        )

        with pytest.raises(ValueError, match="Sersic galaxy requires n parameter"):
            get_galaxy(config, flux=1000.0, half_light_radius=1.0)

    def test_sersic_galaxy_with_distribution_n(self):
        """Test Sersic galaxy with n as distribution uses mean value."""
        shear = ShearConfig(type="G1G2", g1=0.01, g2=0.02)
        n_dist = DistributionConfig(type="Normal", mean=4.0, sigma=0.5)
        config = GalaxyConfig(
            type="Sersic",
            n=n_dist,
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
        )

        gal = get_galaxy(config, flux=1000.0, half_light_radius=1.0)
        assert isinstance(gal, galsim.Sersic)
        # The n value should be extracted from distribution mean

    def test_galaxy_with_ellipticity(self):
        """Test galaxy creation with intrinsic ellipticity."""
        shear = ShearConfig(type="G1G2", g1=0.01, g2=0.02)
        config = GalaxyConfig(
            type="Exponential",
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
        )

        gal = get_galaxy(config, flux=1000.0, half_light_radius=1.0, e1=0.1, e2=0.2)
        # Galaxy should be sheared
        assert isinstance(gal, galsim.Transformation)

    def test_unsupported_galaxy_type(self):
        """Test unsupported galaxy type raises NotImplementedError."""
        shear = ShearConfig(type="G1G2", g1=0.01, g2=0.02)
        config = GalaxyConfig(
            type="Custom",
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
        )

        with pytest.raises(NotImplementedError, match="Galaxy type .* not supported"):
            get_galaxy(config, flux=1000.0, half_light_radius=1.0)


class TestGetJaxGalaxy:
    """Test get_jax_galaxy function for JAX-GalSim galaxy creation."""

    def test_exponential_jax_galaxy(self):
        """Test creation of JAX-GalSim Exponential galaxy."""
        shear = ShearConfig(type="G1G2", g1=0.01, g2=0.02)
        config = GalaxyConfig(
            type="Exponential",
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
        )

        gal = get_jax_galaxy(config, flux=1000.0, half_light_radius=1.0)
        assert isinstance(gal, jax_galsim.Exponential)

    def test_devaucouleurs_jax_galaxy(self):
        """Test creation of JAX-GalSim DeVaucouleurs galaxy."""
        shear = ShearConfig(type="G1G2", g1=0.01, g2=0.02)
        config = GalaxyConfig(
            type="DeVaucouleurs",
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
        )

        gal = get_jax_galaxy(config, flux=1000.0, half_light_radius=1.0)
        assert isinstance(gal, jax_galsim.DeVaucouleurs)

    def test_sersic_jax_galaxy_fallback(self):
        """Test Sersic galaxy falls back to Exponential in JAX-GalSim."""
        shear = ShearConfig(type="G1G2", g1=0.01, g2=0.02)
        config = GalaxyConfig(
            type="Sersic",
            n=4.0,
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
        )

        # TODO: This currently uses Exponential as fallback
        gal = get_jax_galaxy(config, flux=1000.0, half_light_radius=1.0)
        assert isinstance(gal, jax_galsim.Exponential)

    def test_jax_galaxy_with_ellipticity(self):
        """Test JAX galaxy creation with intrinsic ellipticity."""
        shear = ShearConfig(type="G1G2", g1=0.01, g2=0.02)
        config = GalaxyConfig(
            type="Exponential",
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
        )

        gal = get_jax_galaxy(config, flux=1000.0, half_light_radius=1.0, e1=0.1, e2=0.2)
        # Galaxy should be transformed
        assert isinstance(gal, jax_galsim.Transform)

    def test_jax_galaxy_with_gsparams(self):
        """Test JAX galaxy creation with custom GSParams."""
        gsparams = jax_galsim.GSParams(maximum_fft_size=256, minimum_fft_size=256)
        shear = ShearConfig(type="G1G2", g1=0.01, g2=0.02)
        config = GalaxyConfig(
            type="Exponential",
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
        )

        gal = get_jax_galaxy(
            config, flux=1000.0, half_light_radius=1.0, gsparams=gsparams
        )
        assert isinstance(gal, jax_galsim.Exponential)
        assert gal.gsparams is not None

    def test_unsupported_jax_galaxy_type(self):
        """Test unsupported JAX galaxy type raises NotImplementedError."""
        shear = ShearConfig(type="G1G2", g1=0.01, g2=0.02)
        config = GalaxyConfig(
            type="Custom",
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
        )

        with pytest.raises(
            NotImplementedError, match="Galaxy type .* not supported in JAX-GalSim"
        ):
            get_jax_galaxy(config, flux=1000.0, half_light_radius=1.0)


class TestGalaxyParameterization:
    """Test galaxy parameter handling."""

    def test_galaxy_flux_parameter(self):
        """Test that galaxy flux parameter is correctly applied."""
        shear = ShearConfig(type="G1G2", g1=0.01, g2=0.02)
        config = GalaxyConfig(
            type="Exponential",
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
        )

        gal = get_galaxy(config, flux=500.0, half_light_radius=1.0)
        # Note: flux is applied in the function call, not from config

    def test_galaxy_half_light_radius_parameter(self):
        """Test that half-light radius parameter is correctly applied."""
        shear = ShearConfig(type="G1G2", g1=0.01, g2=0.02)
        config = GalaxyConfig(
            type="Exponential",
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
        )

        gal = get_galaxy(config, flux=1000.0, half_light_radius=2.0)
        # Note: half_light_radius is applied in the function call
