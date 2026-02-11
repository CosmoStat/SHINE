"""Shared fixtures for Euclid VIS test suite."""

from pathlib import Path

import pytest

from shine.euclid.config import (
    EuclidDataConfig,
    EuclidInferenceConfig,
    SourceSelectionConfig,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "EUC_VIS_SWL"


def _exposure_paths() -> list[str]:
    return sorted(
        str(p) for p in DATA_DIR.glob("EUC_VIS_SWL-DET-*_3-4-F.fits.gz")
    )


def _background_paths() -> list[str]:
    return sorted(
        str(p) for p in DATA_DIR.glob("EUC_VIS_SWL-BKG-*_3-4-F.fits.gz")
    )


def _data_config() -> EuclidDataConfig:
    return EuclidDataConfig(
        exposure_paths=_exposure_paths(),
        psf_path=str(DATA_DIR / "PSF_3-4-F.fits.gz"),
        catalog_path=str(DATA_DIR / "catalogue_3-4-F.fits.gz"),
        background_paths=_background_paths(),
    )


@pytest.fixture(scope="module")
def small_config():
    """Build a small EuclidInferenceConfig for 3 bright sources."""
    return EuclidInferenceConfig(
        data=_data_config(),
        sources=SourceSelectionConfig(
            max_sources=3, min_snr=50.0, exclude_point_sources=False
        ),
    )


@pytest.fixture(scope="module")
def exposure_set(small_config):
    """Load a small ExposureSet (3 sources, 3 exposures)."""
    from shine.euclid.data_loader import EuclidDataLoader

    return EuclidDataLoader(small_config).load()
