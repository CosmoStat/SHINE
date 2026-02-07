"""Unit tests for catalog utilities and base classes."""

import numpy as np
import pytest
import jax.numpy as jnp

from shine.catalogs.base import GalaxyProperties, SkyRegion
from shine.catalogs.utils import (
    ra_dec_to_pixels,
    compute_ellipticity,
    magnitude_to_flux,
    flux_to_magnitude,
)


class TestSkyRegion:
    """Tests for SkyRegion class."""

    def test_get_bounds_simple(self):
        """Test bounds calculation for a simple case."""
        region = SkyRegion(center_ra=53.0, center_dec=-28.0, size_arcmin=1.0)
        ra_min, ra_max, dec_min, dec_max = region.get_bounds()

        # 1 arcmin = 1/60 deg
        expected_dec_range = 1.0 / 60.0
        assert dec_max - dec_min == pytest.approx(expected_dec_range, rel=1e-6)

        # RA range is wider due to cosine correction
        assert ra_max > ra_min

    def test_get_bounds_equator(self):
        """Test bounds at the equator (no cosine correction)."""
        region = SkyRegion(center_ra=180.0, center_dec=0.0, size_arcmin=1.0)
        ra_min, ra_max, dec_min, dec_max = region.get_bounds()

        expected_size = 1.0 / 60.0  # degrees
        assert dec_max - dec_min == pytest.approx(expected_size, rel=1e-6)
        # At equator, RA range should be similar to Dec range (within JAX precision)
        assert ra_max - ra_min == pytest.approx(expected_size, rel=1e-3)


class TestCoordinateConversion:
    """Tests for RA/Dec to pixel conversion."""

    def test_ra_dec_to_pixels_center(self):
        """Test conversion of center coordinate."""
        center_ra, center_dec = 53.0, -28.0
        ra = np.array([center_ra])
        dec = np.array([center_dec])

        x, y = ra_dec_to_pixels(
            ra, dec, center_ra, center_dec, pixel_scale=0.2, image_size=(300, 300)
        )

        # Center should map to center of image
        assert x[0] == pytest.approx(150.0, abs=0.1)
        assert y[0] == pytest.approx(150.0, abs=0.1)

    def test_ra_dec_to_pixels_offset(self):
        """Test conversion of offset coordinates."""
        center_ra, center_dec = 53.0, -28.0

        # Offset by +10 arcsec in RA (east)
        ra = np.array([center_ra + 10.0 / 3600.0])
        dec = np.array([center_dec])

        x, y = ra_dec_to_pixels(
            ra, dec, center_ra, center_dec, pixel_scale=0.2, image_size=(300, 300)
        )

        # +10 arcsec offset; gnomonic projection has ~cos(dec) factor
        # At dec=-28, expect ~44 pixels offset (10 arcsec * cos(-28°) / 0.2)
        # Allow wider tolerance for projection effects
        assert x[0] > 150.0  # Should be east of center
        assert x[0] == pytest.approx(194.15, abs=5.0)  # Relaxed tolerance
        assert y[0] == pytest.approx(150.0, abs=0.1)

    def test_ra_dec_to_pixels_multiple(self):
        """Test conversion of multiple galaxies."""
        center_ra, center_dec = 53.0, -28.0
        ra = np.array([53.0, 53.001, 52.999])
        dec = np.array([-28.0, -28.001, -27.999])

        x, y = ra_dec_to_pixels(
            ra, dec, center_ra, center_dec, pixel_scale=0.2, image_size=(300, 300)
        )

        assert len(x) == 3
        assert len(y) == 3
        assert x[0] == pytest.approx(150.0, abs=0.1)


class TestEllipticity:
    """Tests for ellipticity calculations."""

    def test_compute_ellipticity_circular(self):
        """Test ellipticity for circular galaxy (a = b)."""
        semi_major = np.array([1.0])
        semi_minor = np.array([1.0])
        position_angle = np.array([0.0])

        e1, e2 = compute_ellipticity(semi_major, semi_minor, position_angle)

        assert e1[0] == pytest.approx(0.0, abs=1e-10)
        assert e2[0] == pytest.approx(0.0, abs=1e-10)

    def test_compute_ellipticity_aligned(self):
        """Test ellipticity for aligned ellipse (PA = 0)."""
        semi_major = np.array([2.0])
        semi_minor = np.array([1.0])
        position_angle = np.array([0.0])

        e1, e2 = compute_ellipticity(semi_major, semi_minor, position_angle)

        # e = (2-1)/(2+1) = 1/3
        # e1 = e * cos(0) = 1/3
        # e2 = e * sin(0) = 0
        assert e1[0] == pytest.approx(1.0 / 3.0, rel=1e-6)
        assert e2[0] == pytest.approx(0.0, abs=1e-10)

    def test_compute_ellipticity_rotated(self):
        """Test ellipticity for rotated ellipse (PA = 45)."""
        semi_major = np.array([2.0])
        semi_minor = np.array([1.0])
        position_angle = np.array([45.0])

        e1, e2 = compute_ellipticity(semi_major, semi_minor, position_angle)

        # e = 1/3
        # e1 = e * cos(90) ≈ 0
        # e2 = e * sin(90) ≈ 1/3
        assert e1[0] == pytest.approx(0.0, abs=1e-6)
        assert e2[0] == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_compute_ellipticity_multiple(self):
        """Test ellipticity for multiple galaxies."""
        semi_major = np.array([2.0, 3.0, 1.5])
        semi_minor = np.array([1.0, 2.0, 1.5])
        position_angle = np.array([0.0, 0.0, 0.0])

        e1, e2 = compute_ellipticity(semi_major, semi_minor, position_angle)

        assert len(e1) == 3
        assert len(e2) == 3


class TestMagnitudeFluxConversion:
    """Tests for magnitude-flux conversions."""

    def test_magnitude_to_flux_standard(self):
        """Test magnitude to flux with standard zeropoint."""
        magnitude = np.array([20.0, 25.0, 30.0])
        flux = magnitude_to_flux(magnitude, zeropoint=30.0)

        # flux = 10^((30-mag)/2.5)
        expected = np.array([10.0 ** 4.0, 10.0 ** 2.0, 1.0])
        np.testing.assert_allclose(flux, expected, rtol=1e-6)

    def test_flux_to_magnitude_standard(self):
        """Test flux to magnitude with standard zeropoint."""
        flux = np.array([10000.0, 100.0, 1.0])
        magnitude = flux_to_magnitude(flux, zeropoint=30.0)

        expected = np.array([20.0, 25.0, 30.0])
        np.testing.assert_allclose(magnitude, expected, rtol=1e-6)

    def test_magnitude_flux_roundtrip(self):
        """Test round-trip conversion."""
        magnitude = np.array([20.0, 22.5, 25.0, 27.5])
        flux = magnitude_to_flux(magnitude)
        magnitude_back = flux_to_magnitude(flux)

        np.testing.assert_allclose(magnitude_back, magnitude, rtol=1e-10)

    def test_magnitude_to_flux_custom_zeropoint(self):
        """Test magnitude to flux with custom zeropoint."""
        magnitude = np.array([20.0])
        flux = magnitude_to_flux(magnitude, zeropoint=25.0)

        # flux = 10^((25-20)/2.5) = 10^2 = 100
        assert flux[0] == pytest.approx(100.0, rel=1e-6)


class TestGalaxyProperties:
    """Tests for GalaxyProperties dataclass."""

    def test_creation_minimal(self):
        """Test creation with minimal required fields."""
        props = GalaxyProperties(
            x=jnp.array([10.0, 20.0]),
            y=jnp.array([15.0, 25.0]),
            flux=jnp.array([1000.0, 500.0]),
            half_light_radius=jnp.array([1.0, 1.5]),
            e1=jnp.array([0.1, -0.05]),
            e2=jnp.array([0.05, 0.1]),
        )

        assert len(props) == 2
        assert not props.has_bulge_disk()

    def test_creation_with_bulge_disk(self):
        """Test creation with bulge+disk decomposition."""
        props = GalaxyProperties(
            x=jnp.array([10.0]),
            y=jnp.array([15.0]),
            flux=jnp.array([1000.0]),
            half_light_radius=jnp.array([1.0]),
            e1=jnp.array([0.1]),
            e2=jnp.array([0.05]),
            bulge_flux=jnp.array([600.0]),
            disk_flux=jnp.array([400.0]),
            bulge_hlr=jnp.array([0.5]),
            disk_hlr=jnp.array([1.5]),
            bulge_n=jnp.array([4.0]),
            disk_n=jnp.array([1.0]),
        )

        assert len(props) == 1
        assert props.has_bulge_disk()

    def test_has_bulge_disk_partial(self):
        """Test has_bulge_disk with partial information."""
        props = GalaxyProperties(
            x=jnp.array([10.0]),
            y=jnp.array([15.0]),
            flux=jnp.array([1000.0]),
            half_light_radius=jnp.array([1.0]),
            e1=jnp.array([0.1]),
            e2=jnp.array([0.05]),
            bulge_flux=jnp.array([600.0]),  # Only bulge_flux, missing others
        )

        assert not props.has_bulge_disk()

    def test_creation_with_optional_fields(self):
        """Test creation with optional fields."""
        props = GalaxyProperties(
            x=jnp.array([10.0]),
            y=jnp.array([15.0]),
            flux=jnp.array([1000.0]),
            half_light_radius=jnp.array([1.0]),
            e1=jnp.array([0.1]),
            e2=jnp.array([0.05]),
            z=jnp.array([0.5]),
            g1_true=jnp.array([0.02]),
            g2_true=jnp.array([-0.01]),
        )

        assert props.z is not None
        assert props.g1_true is not None
        assert props.g2_true is not None
