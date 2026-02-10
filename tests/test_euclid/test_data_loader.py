"""Tests for shine.euclid.data_loader module.

Tests PSF grid parsing, interpolation, exposure loading, WCS transforms,
image preparation, catalog source selection, and full ExposureSet assembly
using real Euclid VIS quadrant-level FITS data.
"""

from pathlib import Path

import numpy as np
import pytest

from shine.euclid.config import EuclidDataConfig, EuclidInferenceConfig

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "EUC_VIS_SWL"

# Skip all tests if data not available
pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists(), reason="Euclid test data not available"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def psf_model():
    """Load the PSF grid model from the test data."""
    from astropy.io import fits as pyfits

    from shine.euclid.data_loader import EuclidPSFModel

    with pyfits.open(DATA_DIR / "PSF_3-4-F.fits.gz") as hdul:
        psf_data = hdul["3-4.F"].data.astype(np.float32)
    return EuclidPSFModel(psf_data)


@pytest.fixture(scope="module")
def exposure():
    """Load a single Euclid VIS exposure."""
    from shine.euclid.data_loader import EuclidExposure

    path = sorted(DATA_DIR.glob("EUC_VIS_SWL-DET-*_3-4-F.fits.gz"))[0]
    return EuclidExposure(str(path))


@pytest.fixture(scope="module")
def exposure_with_background(exposure):
    """Provide the first exposure together with its background map."""
    from astropy.io import fits as pyfits

    bkg_path = sorted(DATA_DIR.glob("EUC_VIS_SWL-BKG-*_3-4-F.fits.gz"))[0]
    with pyfits.open(str(bkg_path)) as hdul:
        bkg_map = hdul["3-4.F"].data.astype(np.float32)
    return exposure, bkg_map


@pytest.fixture(scope="module")
def small_config():
    """Build a small EuclidInferenceConfig for 3 bright sources."""
    from shine.euclid.config import SourceSelectionConfig

    exposure_paths = sorted(
        str(p) for p in DATA_DIR.glob("EUC_VIS_SWL-DET-*_3-4-F.fits.gz")
    )
    bkg_paths = sorted(
        str(p) for p in DATA_DIR.glob("EUC_VIS_SWL-BKG-*_3-4-F.fits.gz")
    )
    return EuclidInferenceConfig(
        data=EuclidDataConfig(
            exposure_paths=exposure_paths,
            psf_path=str(DATA_DIR / "PSF_3-4-F.fits.gz"),
            catalog_path=str(DATA_DIR / "catalogue_3-4-F.fits.gz"),
            background_paths=bkg_paths,
        ),
        sources=SourceSelectionConfig(max_sources=3, min_snr=50.0),
    )


# ---------------------------------------------------------------------------
# PSF tests
# ---------------------------------------------------------------------------


class TestEuclidPSFModel:
    """Test PSF grid parsing and interpolation."""

    def test_psf_grid_shape(self, psf_model):
        """PSF stamps array should be (9, 9, 21, 21)."""
        assert psf_model.stamps.shape == (9, 9, 21, 21)

    def test_psf_interpolation(self, psf_model):
        """Bilinear interpolation at the quadrant centre should return a
        normalised stamp of the correct size."""
        stamp = psf_model.interpolate_at(1024, 1033)
        assert stamp.shape == (21, 21)
        np.testing.assert_allclose(stamp.sum(), 1.0, atol=1e-5)

    def test_psf_nearest(self, psf_model):
        """Nearest-neighbour lookup at a corner should return a normalised
        stamp of the correct size."""
        stamp = psf_model.get_nearest(100, 100)
        assert stamp.shape == (21, 21)
        np.testing.assert_allclose(stamp.sum(), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Exposure tests
# ---------------------------------------------------------------------------


class TestEuclidExposure:
    """Test single-exposure data loading and WCS transforms."""

    def test_exposure_loading(self, exposure):
        """Science, RMS, and flag arrays should have correct shapes."""
        assert exposure.sci.shape == (2066, 2048)
        assert exposure.rms.shape == (2066, 2048)
        assert exposure.flags.shape == (2066, 2048)

    def test_wcs_sky_to_pixel(self, exposure):
        """Converting a known RA/Dec should give pixel coords within the
        quadrant bounds (0..2048, 0..2066)."""
        # Use the WCS reference pixel as a safe known position.
        crval1 = exposure.wcs.wcs.crval[0]
        crval2 = exposure.wcs.wcs.crval[1]
        x, y = exposure.sky_to_pixel(crval1, crval2)
        assert 0 <= x < 2048, f"x={x} out of quadrant bounds"
        assert 0 <= y < 2066, f"y={y} out of quadrant bounds"

    def test_wcs_jacobian(self, exposure):
        """Local WCS Jacobian at the quadrant centre should have entries
        around 0.1 arcsec/pixel (VIS pixel scale), and the sign convention
        u-West / v-North should be consistent."""
        dudx, dudy, dvdx, dvdy = exposure.local_wcs_jacobian(1024, 1033)

        # Magnitude of the pixel scale should be ~0.1 arcsec/pixel.
        scale = np.sqrt(abs(dudx * dvdy - dudy * dvdx))
        np.testing.assert_allclose(scale, 0.1, atol=0.02)

        # u points West (positive RA decreases): dudx should contribute a
        # positive component when the standard orientation is RA increasing
        # with x and the sign flip is applied.  We just check the determinant
        # sign is positive (right-handed u-West, v-North).
        det = dudx * dvdy - dudy * dvdx
        assert det > 0, f"Jacobian determinant should be positive, got {det}"


# ---------------------------------------------------------------------------
# Image preparation tests
# ---------------------------------------------------------------------------


class TestImagePreparation:
    """Test bad-pixel masking, background subtraction, and noise arrays."""

    def test_flag_masking(self, exposure):
        """Mask should be boolean with some but not all pixels masked;
        noise_sigma at masked pixels should be 1e10."""
        image, noise_sigma, mask = exposure.prepare_image_data()

        assert mask.dtype == bool
        n_masked = (~mask).sum()
        n_total = mask.size
        assert n_masked > 0, "Expected some masked pixels"
        assert n_masked < n_total, "Expected some valid pixels"
        np.testing.assert_allclose(noise_sigma[~mask], 1e10)

    def test_background_subtraction_median(self, exposure):
        """With sigma-clipped median subtraction the valid-pixel mean
        should be approximately zero."""
        image, _, mask = exposure.prepare_image_data()
        mean_sky = np.mean(image[mask])
        assert abs(mean_sky) < 5.0, (
            f"Mean of valid pixels after background subtraction = {mean_sky}"
        )

    def test_background_subtraction_map(self, exposure_with_background):
        """With a provided background map the result should differ from
        the sigma-clipped fallback."""
        exp, bkg_map = exposure_with_background
        image_map, _, mask_map = exp.prepare_image_data(
            background_map=bkg_map,
        )
        image_med, _, mask_med = exp.prepare_image_data()

        # The two images should not be identical.
        assert not np.allclose(image_map, image_med)


# ---------------------------------------------------------------------------
# Catalog / source selection tests
# ---------------------------------------------------------------------------


class TestSourceSelection:
    """Test catalog loading and source filtering."""

    def test_source_selection(self, small_config):
        """Loading the catalog with max_sources=3 should return exactly
        3 sources after filtering."""
        from shine.euclid.data_loader import EuclidDataLoader

        loader = EuclidDataLoader(small_config)
        catalog = loader._load_catalog()
        sources = loader._select_sources(catalog)
        assert len(sources) == 3

    def test_flux_conversion(self, small_config):
        """Converted flux_adu values should be positive and finite."""
        from shine.euclid.data_loader import EuclidDataLoader, EuclidExposure

        loader = EuclidDataLoader(small_config)
        catalog = loader._load_catalog()
        sources = loader._select_sources(catalog)
        exp = EuclidExposure(
            small_config.data.exposure_paths[0],
            small_config.data.quadrant,
        )
        flux_adu = loader._flux_ujy_to_adu(sources, exp)

        assert flux_adu.shape == (3,)
        assert np.all(flux_adu > 0)
        assert np.all(np.isfinite(flux_adu))


# ---------------------------------------------------------------------------
# Full ExposureSet assembly
# ---------------------------------------------------------------------------


class TestExposureSet:
    """Test end-to-end data loading into ExposureSet."""

    def test_exposure_set(self, small_config):
        """Full EuclidDataLoader.load() should produce an ExposureSet with
        consistent array shapes for 3 exposures and 3 sources."""
        from shine.euclid.data_loader import EuclidDataLoader

        loader = EuclidDataLoader(small_config)
        eset = loader.load()

        n_exp = eset.n_exposures
        n_src = eset.n_sources
        ny = eset.image_ny
        nx = eset.image_nx

        assert n_exp == 3
        assert n_src == 3

        assert eset.images.shape == (n_exp, ny, nx)
        assert eset.noise_sigma.shape == (n_exp, ny, nx)
        assert eset.masks.shape == (n_exp, ny, nx)
        assert eset.backgrounds.shape == (n_exp,)

        assert eset.pixel_positions.shape == (n_src, n_exp, 2)
        assert eset.wcs_jacobians.shape == (n_src, n_exp, 4)
        assert eset.psf_images.shape == (n_src, n_exp, 21, 21)
        assert eset.source_visible.shape == (n_src, n_exp)

        assert eset.catalog_flux_adu.shape == (n_src,)
        assert eset.catalog_hlr_arcsec.shape == (n_src,)
        assert len(eset.source_ids) == n_src
