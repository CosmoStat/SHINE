"""Integration tests for Level 1 bias measurement pipeline.

These tests verify the end-to-end paired-shear workflow including
random ellipticity, paired observation generation, and paired response
computation. Marked as slow since they require MCMC inference.
"""

import numpy as np
import pytest

from shine.validation.simulation import draw_ellipticity, generate_paired_observations
from shine.validation.statistics import compute_paired_response


class TestDrawEllipticity:
    """Tests for draw_ellipticity()."""

    def test_within_bounds(self):
        """Drawn ellipticities should satisfy |e| < 0.7."""
        rng = np.random.default_rng(42)
        # Use None for config since it's unused in current implementation
        for _ in range(100):
            e1, e2 = draw_ellipticity(None, rng)
            assert np.sqrt(e1**2 + e2**2) < 0.7

    def test_different_seeds_give_different_values(self):
        """Different RNG states produce different ellipticities."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)
        e1_a, e2_a = draw_ellipticity(None, rng1)
        e1_b, e2_b = draw_ellipticity(None, rng2)
        assert (e1_a, e2_a) != (e1_b, e2_b)

    def test_distribution_properties(self):
        """Drawn ellipticities should have ~Normal(0, 0.2) marginals."""
        rng = np.random.default_rng(42)
        e1s = []
        e2s = []
        for _ in range(5000):
            e1, e2 = draw_ellipticity(None, rng)
            e1s.append(e1)
            e2s.append(e2)
        e1s = np.array(e1s)
        e2s = np.array(e2s)
        # Mean should be near 0
        assert abs(np.mean(e1s)) < 0.02
        assert abs(np.mean(e2s)) < 0.02
        # Std should be near 0.2 (slightly less due to rejection)
        assert 0.15 < np.std(e1s) < 0.22
        assert 0.15 < np.std(e2s) < 0.22


@pytest.mark.slow
@pytest.mark.integration
class TestLevel1Integration:
    """Integration tests for Level 1 paired-shear pipeline.

    These tests are slow because they run MCMC inference.
    Use: pytest -m slow tests/test_validation/test_level1_integration.py
    """

    @pytest.fixture
    def shine_config(self):
        """Load the Level 1 base config."""
        from shine.config import ConfigHandler

        config = ConfigHandler.load("configs/validation/level1_base.yaml")
        # Override for fast test: reduce MCMC settings
        config.inference.method = "nuts"
        config.inference.num_warmup = 100
        config.inference.num_samples = 200
        config.inference.num_chains = 1
        # Use moderate noise for test
        config.image.noise.sigma = 0.1
        config.gal.flux = 1000.0
        return config

    def test_paired_observations_share_ellipticity(self, shine_config):
        """Paired +g/-g observations should use the same ellipticity."""
        rng = np.random.default_rng(42)
        plus_result, minus_result = generate_paired_observations(
            shine_config, g1_true=0.02, g2_true=0.0, seed=42, rng=rng,
        )
        # Same ellipticity for both
        assert plus_result.ground_truth["e1"] == minus_result.ground_truth["e1"]
        assert plus_result.ground_truth["e2"] == minus_result.ground_truth["e2"]

    def test_paired_observations_have_opposite_shear(self, shine_config):
        """Paired observations should have opposite shear signs."""
        rng = np.random.default_rng(42)
        plus_result, minus_result = generate_paired_observations(
            shine_config, g1_true=0.02, g2_true=0.01, seed=42, rng=rng,
        )
        assert plus_result.ground_truth["g1"] == pytest.approx(0.02)
        assert minus_result.ground_truth["g1"] == pytest.approx(-0.02)
        assert plus_result.ground_truth["g2"] == pytest.approx(0.01)
        assert minus_result.ground_truth["g2"] == pytest.approx(-0.01)

    def test_different_seeds_different_ellipticity(self, shine_config):
        """Different RNG seeds should produce different ellipticities."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)

        plus1, _ = generate_paired_observations(
            shine_config, g1_true=0.02, g2_true=0.0, seed=42, rng=rng1,
        )
        plus2, _ = generate_paired_observations(
            shine_config, g1_true=0.02, g2_true=0.0, seed=42, rng=rng2,
        )
        # Different RNG seeds → different ellipticities
        assert plus1.ground_truth["e1"] != plus2.ground_truth["e1"]

    def test_paired_inference_smoke(self, shine_config):
        """Smoke test: run paired inference and compute response.

        Generates 2 paired realizations at g1=0.02, runs inference on each,
        and computes the paired response.
        """
        import jax

        from shine.inference import Inference
        from shine.scene import SceneBuilder

        g1_true = 0.02
        g2_true = 0.0
        n_pairs = 2

        g1_est_plus = []
        g1_est_minus = []

        for i in range(n_pairs):
            rng = np.random.default_rng(42 + i)
            plus_result, minus_result = generate_paired_observations(
                shine_config, g1_true=g1_true, g2_true=g2_true,
                seed=42 + i, rng=rng,
            )

            for sign, sim_result in [("plus", plus_result), ("minus", minus_result)]:
                scene_builder = SceneBuilder(shine_config)
                model_fn = scene_builder.build_model()
                rng_key = jax.random.PRNGKey(shine_config.inference.rng_seed + i)
                engine = Inference(model=model_fn, config=shine_config.inference)
                idata = engine.run(
                    rng_key=rng_key,
                    observed_data=sim_result.observation.image,
                    extra_args={"psf": sim_result.observation.psf_model},
                )
                g1_mean = float(idata.posterior.g1.values.flatten().mean())
                if sign == "plus":
                    g1_est_plus.append(g1_mean)
                else:
                    g1_est_minus.append(g1_mean)

        g1_est_plus = np.array(g1_est_plus)
        g1_est_minus = np.array(g1_est_minus)

        # Compute paired response
        R = compute_paired_response(g1_est_plus, g1_est_minus, g1_true)
        assert R.shape == (n_pairs,)
        # Response should be ~1 for well-calibrated inference (allow wide margin)
        assert np.all(np.abs(R) < 5.0)  # Sanity: R shouldn't be wildly off
