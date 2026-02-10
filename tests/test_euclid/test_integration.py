"""End-to-end integration / smoke tests for Euclid VIS shear inference.

These tests exercise the full pipeline from data loading through MAP
inference. They are marked ``@pytest.mark.integration`` and
``@pytest.mark.slow`` so they can be easily skipped during fast
unit-test runs.
"""

from pathlib import Path

import jax
import pytest

from shine.config import InferenceConfig, MAPConfig
from shine.euclid.config import (
    EuclidDataConfig,
    EuclidInferenceConfig,
    SourceSelectionConfig,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "EUC_VIS_SWL"

pytestmark = [
    pytest.mark.skipif(
        not DATA_DIR.exists(), reason="Euclid test data not available"
    ),
    pytest.mark.integration,
    pytest.mark.slow,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_config():
    """Build a small EuclidInferenceConfig for MAP inference on 3 sources."""
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
        sources=SourceSelectionConfig(
            max_sources=3, min_snr=50.0, exclude_point_sources=False
        ),
        inference=InferenceConfig(
            method="map",
            map_config=MAPConfig(enabled=True, num_steps=50),
        ),
    )


@pytest.fixture(scope="module")
def exposure_set(small_config):
    """Load the small ExposureSet."""
    from shine.euclid.data_loader import EuclidDataLoader

    return EuclidDataLoader(small_config).load()


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_euclid_map_inference(small_config, exposure_set):
    """Smoke test: run MAP inference on 3 bright sources and verify
    the result contains g1 and g2 estimates."""
    from shine.euclid.scene import MultiExposureScene
    from shine.inference import Inference

    scene = MultiExposureScene(small_config, exposure_set)
    model = scene.build_model()

    engine = Inference(model, small_config.inference)
    rng = jax.random.PRNGKey(42)
    idata = engine.run(rng, observed_data=exposure_set.images)

    assert "posterior" in idata.groups()
    assert "g1" in idata.posterior
    assert "g2" in idata.posterior
