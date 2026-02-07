"""Integration test for Level 0 bias measurement pipeline.

This test runs the full end-to-end pipeline: generate data → run MCMC →
extract diagnostics → check acceptance criteria.

Requires JAX, GalSim, NumPyro, and JAX-GalSim to be installed.
"""

import numpy as np
import pytest
import yaml

from shine.config import ConfigHandler
from shine.validation.bias_config import AcceptanceCriteria, ConvergenceThresholds
from shine.validation.extraction import (
    check_convergence,
    extract_convergence_diagnostics,
    extract_realization,
    extract_shear_estimates,
)
from shine.validation.simulation import generate_biased_observation
from shine.validation.statistics import compute_bias_single_point


@pytest.fixture
def level0_config(tmp_path):
    """Create a Level 0 config with reduced MCMC settings for fast testing."""
    config_dict = {
        "image": {
            "pixel_scale": 0.1,
            "size_x": 48,
            "size_y": 48,
            "n_objects": 1,
            "fft_size": 128,
            "noise": {"type": "Gaussian", "sigma": 0.1},
        },
        "psf": {"type": "Gaussian", "sigma": 0.1},
        "gal": {
            "type": "Exponential",
            "flux": 1000.0,
            "half_light_radius": 0.5,
            "shear": {
                "type": "G1G2",
                "g1": {"type": "Normal", "mean": 0.02, "sigma": 0.05},
                "g2": {"type": "Normal", "mean": -0.01, "sigma": 0.05},
            },
            "position": {
                "type": "Uniform",
                "x_min": 23.5,
                "x_max": 24.5,
                "y_min": 23.5,
                "y_max": 24.5,
            },
        },
        "inference": {
            "warmup": 200,
            "samples": 400,
            "chains": 1,
            "dense_mass": False,
            "rng_seed": 42,
            "map_init": {
                "enabled": True,
                "num_steps": 500,
                "learning_rate": 0.01,
            },
        },
    }

    config_path = tmp_path / "level0_test.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    return ConfigHandler.load(str(config_path))


@pytest.mark.slow
@pytest.mark.integration
class TestLevel0Integration:
    """End-to-end Level 0 bias measurement test."""

    def test_full_pipeline(self, level0_config):
        """Run generate → infer → extract → check acceptance."""
        import jax

        from shine.inference import Inference
        from shine.scene import SceneBuilder

        g1_true = 0.02
        g2_true = -0.01

        # Stage 1: Generate observation
        sim_result = generate_biased_observation(
            level0_config, g1_true, g2_true, seed=42
        )
        assert sim_result.ground_truth["g1"] == g1_true
        assert sim_result.ground_truth["g2"] == g2_true

        # Stage 1b: Run MCMC
        scene_builder = SceneBuilder(level0_config)
        model_fn = scene_builder.build_model()

        rng_key = jax.random.PRNGKey(level0_config.inference.rng_seed)
        engine = Inference(model=model_fn, config=level0_config.inference)
        idata = engine.run(
            rng_key=rng_key,
            observed_data=sim_result.observation.image,
            extra_args={"psf": sim_result.observation.psf_model},
        )

        # Stage 2: Extract results
        thresholds = ConvergenceThresholds(
            rhat_max=1.1,  # Relaxed for single chain
            ess_min=50,
            divergence_frac_max=0.05,
            bfmi_min=0.1,
        )
        result = extract_realization(
            idata, g1_true, g2_true,
            run_id="integration_test", seed=42,
            thresholds=thresholds,
        )

        # Stage 3: Check Level 0 acceptance
        # For near-zero noise, posterior should be close to truth
        g1_offset = abs(result.g1.mean - g1_true)
        g2_offset = abs(result.g2.mean - g2_true)

        # The posterior mean should be within a few sigma of truth
        # Use relaxed criteria for reduced MCMC settings
        if result.g1.std > 0:
            assert g1_offset / result.g1.std < 5.0, (
                f"g1 offset = {g1_offset / result.g1.std:.2f}σ"
            )
        if result.g2.std > 0:
            assert g2_offset / result.g2.std < 5.0, (
                f"g2 offset = {g2_offset / result.g2.std:.2f}σ"
            )
