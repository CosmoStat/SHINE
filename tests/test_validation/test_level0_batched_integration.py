"""Integration test for batched Level 0 bias measurement pipeline.

This test runs the full end-to-end batched pipeline: generate stacked data →
run batched MCMC → split posterior → extract diagnostics → verify each
realization recovers truth.

Requires JAX, GalSim, NumPyro, and JAX-GalSim to be installed.
"""

import numpy as np
import pytest
import yaml

from shine.config import ConfigHandler
from shine.validation.bias_config import ConvergenceThresholds
from shine.validation.extraction import (
    extract_realization,
    extract_shear_estimates,
    split_batched_idata,
)
from shine.validation.simulation import generate_batch_observations


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
            "noise": {"type": "Gaussian", "sigma": 0.001},
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

    config_path = tmp_path / "level0_batched_test.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    return ConfigHandler.load(str(config_path))


@pytest.mark.slow
@pytest.mark.integration
class TestLevel0BatchedIntegration:
    """End-to-end batched Level 0 bias measurement test."""

    def test_batched_pipeline(self, level0_config):
        """Generate 3 batched observations -> run batched MCMC -> split -> extract."""
        import jax

        from shine.inference import Inference
        from shine.scene import SceneBuilder

        n_batch = 3
        shear_pairs = [(0.02, -0.01), (0.03, 0.0), (0.01, 0.01)]
        seeds = [42, 43, 44]

        # Stage 1a: Generate batched observations
        batch_result = generate_batch_observations(
            level0_config,
            shear_pairs=shear_pairs,
            seeds=seeds,
        )
        assert batch_result.images.shape == (n_batch, 48, 48)
        assert len(batch_result.ground_truths) == n_batch

        # Stage 1b: Build batched model and run MCMC
        scene_builder = SceneBuilder(level0_config)
        model_fn = scene_builder.build_batched_model(n_batch)

        rng_key = jax.random.PRNGKey(level0_config.inference.rng_seed)
        engine = Inference(model=model_fn, config=level0_config.inference)

        idata = engine.run(
            rng_key=rng_key,
            observed_data=batch_result.images,
            extra_args={"psf": batch_result.psf_model},
        )

        # Verify posterior has batch dimension
        g1_shape = idata.posterior["g1"].values.shape
        assert len(g1_shape) == 3  # (chain, draw, batch)
        assert g1_shape[2] == n_batch

        # Stage 1c: Split posterior
        split_results = split_batched_idata(
            idata, n_batch, batch_result.run_ids
        )
        assert len(split_results) == n_batch

        # Stage 2: Extract and verify each realization
        for i, (run_id, single_idata) in enumerate(split_results):
            g1_true, g2_true = shear_pairs[i]

            g1_est = extract_shear_estimates(single_idata, "g1")
            g2_est = extract_shear_estimates(single_idata, "g2")

            # For near-zero noise, posterior should recover truth
            g1_offset = abs(g1_est.mean - g1_true)
            g2_offset = abs(g2_est.mean - g2_true)

            # Relaxed check for reduced MCMC settings
            if g1_est.std > 0:
                assert g1_offset / g1_est.std < 5.0, (
                    f"Realization {i}: g1 offset = "
                    f"{g1_offset / g1_est.std:.2f}sigma"
                )
            if g2_est.std > 0:
                assert g2_offset / g2_est.std < 5.0, (
                    f"Realization {i}: g2 offset = "
                    f"{g2_offset / g2_est.std:.2f}sigma"
                )
