"""Integration tests for catalog-based scene generation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from shine.config import ConfigHandler


class TestCatalogConfig:
    """Tests for catalog configuration loading."""

    def test_load_config_with_catalog(self, tmp_path):
        """Test loading a config file with catalog section."""
        # Create a minimal mock catalog file
        catalog_path = tmp_path / "test_catalog.fits"
        self._create_mock_catalog(catalog_path)

        # Create config file
        config_yaml = f"""
image:
  pixel_scale: 0.2
  size_x: 100
  size_y: 100
  n_objects: 10
  noise:
    type: Gaussian
    sigma: 0.01

psf:
  type: Gaussian
  sigma: 0.7

gal:
  type: Exponential
  flux: 1000.0
  half_light_radius: 1.0
  shear:
    type: G1G2
    g1: 0.02
    g2: -0.01

catalog:
  type: cosmodc2
  path: {catalog_path}
  center_ra: 53.0
  center_dec: -28.0
  size_arcmin: 1.0
  magnitude_limit: 25.0
  use_bulge_disk: false

inference:
  warmup: 100
  samples: 100
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_yaml)

        # Load config
        config = ConfigHandler.load(str(config_file))

        # Verify catalog config
        assert config.catalog is not None
        assert config.catalog.type == "cosmodc2"
        assert config.catalog.center_ra == 53.0
        assert config.catalog.center_dec == -28.0
        assert config.catalog.size_arcmin == 1.0
        assert config.catalog.magnitude_limit == 25.0
        assert config.catalog.use_bulge_disk is False

    def test_config_mutual_exclusivity(self, tmp_path):
        """Test that data_path and catalog are mutually exclusive."""
        catalog_path = tmp_path / "test_catalog.fits"
        self._create_mock_catalog(catalog_path)

        config_yaml = f"""
image:
  pixel_scale: 0.2
  size_x: 100
  size_y: 100
  n_objects: 10
  noise:
    type: Gaussian
    sigma: 0.01

psf:
  type: Gaussian
  sigma: 0.7

gal:
  type: Exponential
  flux: 1000.0
  half_light_radius: 1.0
  shear:
    type: G1G2
    g1: 0.02
    g2: -0.01

data_path: /some/data/file.fits

catalog:
  type: cosmodc2
  path: {catalog_path}
  center_ra: 53.0
  center_dec: -28.0

inference:
  warmup: 100
  samples: 100
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_yaml)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Cannot specify both"):
            ConfigHandler.load(str(config_file))

    def test_catalog_path_validation(self, tmp_path):
        """Test that catalog path must exist."""
        config_yaml = """
image:
  pixel_scale: 0.2
  size_x: 100
  size_y: 100
  n_objects: 10
  noise:
    type: Gaussian
    sigma: 0.01

psf:
  type: Gaussian
  sigma: 0.7

gal:
  type: Exponential
  flux: 1000.0
  half_light_radius: 1.0
  shear:
    type: G1G2
    g1: 0.02
    g2: -0.01

catalog:
  type: cosmodc2
  path: /nonexistent/path/catalog.fits
  center_ra: 53.0
  center_dec: -28.0

inference:
  warmup: 100
  samples: 100
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_yaml)

        # Should raise ValueError about path not existing
        with pytest.raises(ValueError, match="does not exist"):
            ConfigHandler.load(str(config_file))

    @staticmethod
    def _create_mock_catalog(path: Path, center_ra: float = 53.0, center_dec: float = -28.0) -> None:
        """Create a minimal mock CosmoDC2 catalog for testing.

        Args:
            path: Path where to save the FITS catalog.
            center_ra: Center RA for galaxy distribution (default 53.0).
            center_dec: Center Dec for galaxy distribution (default -28.0).
        """
        # Create minimal catalog with required columns
        # Place galaxies in a ~0.2 degree region centered on (center_ra, center_dec)
        n_galaxies = 100

        # Generate positions around the center (0.1 deg = 6 arcmin radius)
        ra_offset = np.random.uniform(-0.1, 0.1, n_galaxies)
        dec_offset = np.random.uniform(-0.1, 0.1, n_galaxies)

        data = {
            "ra": center_ra + ra_offset,
            "dec": center_dec + dec_offset,
            "mag_i_lsst": np.random.uniform(20.0, 26.0, n_galaxies),
            "mag_r_lsst": np.random.uniform(20.0, 26.0, n_galaxies),
            "size": np.random.uniform(0.3, 2.0, n_galaxies),
            "size_minor": np.random.uniform(0.2, 1.8, n_galaxies),
            "position_angle": np.random.uniform(0, 180, n_galaxies),
            "ellipticity_1_true": np.random.uniform(-0.3, 0.3, n_galaxies),
            "ellipticity_2_true": np.random.uniform(-0.3, 0.3, n_galaxies),
            "redshift": np.random.uniform(0.1, 2.0, n_galaxies),
        }

        table = Table(data)
        table.write(path, format="fits", overwrite=True)


class TestCosmoDC2Loader:
    """Tests for CosmoDC2 catalog loader."""

    def test_load_and_sample(self, tmp_path):
        """Test loading CosmoDC2 catalog and sampling galaxies."""
        from shine.catalogs import get_catalog_loader

        # Create mock catalog
        catalog_path = tmp_path / "test_cosmodc2.fits"
        TestCatalogConfig._create_mock_catalog(catalog_path)

        # Load catalog
        loader = get_catalog_loader("cosmodc2")
        loader.load(str(catalog_path))

        # Sample postage stamp
        galaxies = loader.sample_postage_stamp(
            center_ra=53.0,
            center_dec=-28.0,
            size_arcmin=1.0,
            pixel_scale=0.2,
            image_size=(100, 100),
            magnitude_limit=25.0,
        )

        # Verify properties
        assert len(galaxies) > 0
        assert galaxies.x is not None
        assert galaxies.y is not None
        assert galaxies.flux is not None
        assert galaxies.half_light_radius is not None
        assert galaxies.e1 is not None
        assert galaxies.e2 is not None

        # Verify coordinates exist (they may be outside the image bounds,
        # which is expected for galaxies near the edge of the sampled region)
        assert np.all(np.isfinite(galaxies.x))
        assert np.all(np.isfinite(galaxies.y))

    def test_magnitude_cut(self, tmp_path):
        """Test that magnitude cut is applied correctly."""
        from shine.catalogs import get_catalog_loader

        # Create mock catalog with brighter galaxies for testing
        catalog_path = tmp_path / "test_cosmodc2.fits"
        self._create_mock_catalog_with_bright_galaxies(catalog_path)

        # Load catalog
        loader = get_catalog_loader("cosmodc2")
        loader.load(str(catalog_path))

        # Sample with different magnitude cuts
        try:
            galaxies_bright = loader.sample_postage_stamp(
                center_ra=53.0,
                center_dec=-28.0,
                size_arcmin=1.0,
                pixel_scale=0.2,
                image_size=(100, 100),
                magnitude_limit=22.0,
            )
            n_bright = len(galaxies_bright)
        except ValueError:
            # No galaxies found with strict cut
            n_bright = 0

        galaxies_all = loader.sample_postage_stamp(
            center_ra=53.0,
            center_dec=-28.0,
            size_arcmin=1.0,
            pixel_scale=0.2,
            image_size=(100, 100),
            magnitude_limit=26.0,
        )
        n_all = len(galaxies_all)

        # Should have fewer (or equal) galaxies with stricter cut
        assert n_bright <= n_all
        # At least some galaxies should be found with relaxed cut
        assert n_all > 0

    @staticmethod
    def _create_mock_catalog_with_bright_galaxies(path: Path) -> None:
        """Create mock catalog with brighter galaxies for magnitude cut testing."""
        n_galaxies = 100
        ra_offset = np.random.uniform(-0.1, 0.1, n_galaxies)
        dec_offset = np.random.uniform(-0.1, 0.1, n_galaxies)

        data = {
            "ra": 53.0 + ra_offset,
            "dec": -28.0 + dec_offset,
            "mag_i_lsst": np.random.uniform(19.0, 24.0, n_galaxies),  # Brighter range
            "mag_r_lsst": np.random.uniform(19.0, 24.0, n_galaxies),
            "size": np.random.uniform(0.3, 2.0, n_galaxies),
            "size_minor": np.random.uniform(0.2, 1.8, n_galaxies),
            "position_angle": np.random.uniform(0, 180, n_galaxies),
            "ellipticity_1_true": np.random.uniform(-0.3, 0.3, n_galaxies),
            "ellipticity_2_true": np.random.uniform(-0.3, 0.3, n_galaxies),
            "redshift": np.random.uniform(0.1, 2.0, n_galaxies),
        }

        table = Table(data)
        table.write(path, format="fits", overwrite=True)


@pytest.mark.slow
class TestEndToEndCatalogGeneration:
    """End-to-end tests for catalog-based scene generation."""

    def test_generate_from_catalog(self, tmp_path):
        """Test full scene generation from catalog."""
        from shine.data import DataLoader

        # Create mock catalog with denser galaxy distribution
        catalog_path = tmp_path / "test_catalog.fits"
        self._create_dense_mock_catalog(catalog_path)

        # Create config
        config_yaml = f"""
image:
  pixel_scale: 0.2
  size_x: 100
  size_y: 100
  n_objects: 50
  noise:
    type: Gaussian
    sigma: 0.01
  fft_size: 64

psf:
  type: Gaussian
  sigma: 0.7

gal:
  type: Exponential
  flux: 1000.0
  half_light_radius: 1.0
  shear:
    type: G1G2
    g1: 0.02
    g2: -0.01

catalog:
  type: cosmodc2
  path: {catalog_path}
  center_ra: 53.0
  center_dec: -28.0
  size_arcmin: 1.0
  magnitude_limit: 25.0
  use_bulge_disk: false

inference:
  warmup: 100
  samples: 100
  rng_seed: 42
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_yaml)

        # Load config and generate scene
        config = ConfigHandler.load(str(config_file))
        observation = DataLoader.load(config)

        # Verify observation properties
        assert observation.image is not None
        assert observation.noise_map is not None
        assert observation.psf_model is not None

        # Check image shape
        assert observation.image.shape == (100, 100)
        assert observation.noise_map.shape == (100, 100)

        # Check that image has signal (not all zeros)
        assert observation.image.sum() > 0

    @staticmethod
    def _create_dense_mock_catalog(path: Path) -> None:
        """Create a denser mock catalog for end-to-end testing."""
        # Create many galaxies tightly clustered around the center
        n_galaxies = 500

        # Generate positions very close to center (within 1 arcmin = 0.0167 deg)
        ra_offset = np.random.uniform(-0.02, 0.02, n_galaxies)
        dec_offset = np.random.uniform(-0.02, 0.02, n_galaxies)

        data = {
            "ra": 53.0 + ra_offset,
            "dec": -28.0 + dec_offset,
            "mag_i_lsst": np.random.uniform(20.0, 24.5, n_galaxies),
            "mag_r_lsst": np.random.uniform(20.0, 24.5, n_galaxies),
            "size": np.random.uniform(0.5, 1.5, n_galaxies),
            "size_minor": np.random.uniform(0.4, 1.4, n_galaxies),
            "position_angle": np.random.uniform(0, 180, n_galaxies),
            "ellipticity_1_true": np.random.uniform(-0.2, 0.2, n_galaxies),
            "ellipticity_2_true": np.random.uniform(-0.2, 0.2, n_galaxies),
            "redshift": np.random.uniform(0.2, 1.5, n_galaxies),
        }

        table = Table(data)
        table.write(path, format="fits", overwrite=True)
