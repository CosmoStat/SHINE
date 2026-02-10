"""Tests for shine.euclid.scene module.

Verifies the NumPyro generative model structure produced by
MultiExposureScene, checking that all expected sample sites exist
with the correct shapes for both multi-exposure and single-exposure
model variants.
"""

from pathlib import Path

import jax
import numpy as np
import numpyro.handlers as handlers
import pytest

from shine.euclid.config import (
    EuclidDataConfig,
    EuclidInferenceConfig,
    SourceSelectionConfig,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "EUC_VIS_SWL"

pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists(), reason="Euclid test data not available"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_config():
    """Build a small EuclidInferenceConfig for 3 bright sources."""
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


@pytest.fixture(scope="module")
def exposure_set(small_config):
    """Load a small ExposureSet (3 sources, 3 exposures)."""
    from shine.euclid.data_loader import EuclidDataLoader

    return EuclidDataLoader(small_config).load()


# ---------------------------------------------------------------------------
# Model trace tests
# ---------------------------------------------------------------------------


class TestMultiExposureModel:
    """Test the multi-exposure NumPyro model structure."""

    def test_model_trace(self, small_config, exposure_set):
        """The model trace should contain g1, g2, flux, hlr, e1, e2, dx, dy
        sample sites and one obs_j site per exposure, all with correct
        shapes."""
        from shine.euclid.scene import MultiExposureScene

        scene = MultiExposureScene(small_config, exposure_set)
        model = scene.build_model()

        rng = jax.random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model, rng)).get_trace(
            observed_data=exposure_set.images,
        )

        # Scalar shear sites
        assert "g1" in trace
        assert "g2" in trace
        assert trace["g1"]["value"].shape == ()
        assert trace["g2"]["value"].shape == ()

        # Per-source parameter sites
        for name in ("flux", "hlr", "e1", "e2", "dx", "dy"):
            assert name in trace, f"Missing sample site: {name}"
            assert trace[name]["value"].shape == (exposure_set.n_sources,), (
                f"{name} shape mismatch"
            )

        # Observation sites for each exposure
        for j in range(exposure_set.n_exposures):
            site = f"obs_{j}"
            assert site in trace, f"Missing observation site: {site}"

    def test_single_exposure_model(self, small_config, exposure_set):
        """A single-exposure model should have only obs_0 and not obs_1
        or obs_2."""
        from shine.euclid.scene import MultiExposureScene

        scene = MultiExposureScene(small_config, exposure_set)
        model = scene.build_single_exposure_model(exposure_idx=0)

        rng = jax.random.PRNGKey(1)
        trace = handlers.trace(handlers.seed(model, rng)).get_trace(
            observed_data=exposure_set.images,
        )

        assert "obs_0" in trace
        assert "obs_1" not in trace
        assert "obs_2" not in trace
