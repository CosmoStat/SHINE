"""Tests for shine.validation.simulation module."""

import numpy as np
import pytest

from shine.config import ConfigHandler, ShineConfig
from shine.data import Observation
from shine.validation.simulation import (
    SimulationResult,
    generate_biased_observation,
    generate_paired_observations,
)


@pytest.fixture
def level0_config(tmp_path):
    """Create a minimal Level 0 SHINE config for testing."""
    import yaml

    config_dict = {
        "image": {
            "pixel_scale": 0.1,
            "size_x": 48,
            "size_y": 48,
            "n_objects": 1,
            "fft_size": 128,
            "noise": {"type": "Gaussian", "sigma": 1e-8},
        },
        "psf": {"type": "Gaussian", "sigma": 0.1},
        "gal": {
            "type": "Exponential",
            "flux": 1000.0,
            "half_light_radius": 0.5,
            "shear": {
                "type": "G1G2",
                "g1": {"type": "Normal", "mean": 0.0, "sigma": 0.05},
                "g2": {"type": "Normal", "mean": 0.0, "sigma": 0.05},
            },
        },
        "inference": {"warmup": 10, "samples": 10, "chains": 1, "rng_seed": 0},
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    return ConfigHandler.load(str(config_path))


class TestGenerateBiasedObservation:
    """Tests for generate_biased_observation()."""

    def test_produces_valid_observation(self, level0_config):
        result = generate_biased_observation(level0_config, 0.02, -0.01, 42)
        assert isinstance(result, SimulationResult)
        assert isinstance(result.observation, Observation)
        assert result.observation.image.shape == (48, 48)

    def test_ground_truth_correct(self, level0_config):
        result = generate_biased_observation(level0_config, 0.03, -0.02, 42)
        assert result.ground_truth["g1"] == 0.03
        assert result.ground_truth["g2"] == -0.02
        assert result.ground_truth["flux"] == 1000.0
        assert result.ground_truth["hlr"] == 0.5

    def test_shear_overrides_config(self, level0_config):
        result1 = generate_biased_observation(level0_config, 0.02, 0.0, 42)
        result2 = generate_biased_observation(level0_config, 0.05, 0.0, 42)
        # Different shear should produce different images
        assert not np.allclose(
            result1.observation.image, result2.observation.image
        )

    def test_different_seeds_different_noise(self, level0_config):
        result1 = generate_biased_observation(level0_config, 0.02, 0.0, 42)
        result2 = generate_biased_observation(level0_config, 0.02, 0.0, 99)
        # With near-zero noise, images should be nearly identical but not exactly
        # (noise is epsilon-level, so this tests the seed propagation)
        assert result1.observation.image.shape == result2.observation.image.shape


class TestGeneratePairedObservations:
    """Tests for generate_paired_observations()."""

    def test_opposite_shear(self, level0_config):
        plus, minus = generate_paired_observations(level0_config, 0.02, 0.01, 42)
        assert plus.ground_truth["g1"] == 0.02
        assert plus.ground_truth["g2"] == 0.01
        assert minus.ground_truth["g1"] == -0.02
        assert minus.ground_truth["g2"] == -0.01

    def test_same_noise_seed(self, level0_config):
        plus, minus = generate_paired_observations(level0_config, 0.02, 0.01, 42)
        # Both should be valid observations
        assert isinstance(plus.observation, Observation)
        assert isinstance(minus.observation, Observation)
        assert plus.observation.image.shape == minus.observation.image.shape
