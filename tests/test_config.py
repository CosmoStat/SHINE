"""Tests for shine.config module."""

import tempfile
from pathlib import Path

import pytest
import yaml

from shine.config import (
    ConfigHandler,
    DistributionConfig,
    EllipticityConfig,
    GalaxyConfig,
    ImageConfig,
    InferenceConfig,
    MAPConfig,
    NoiseConfig,
    NUTSConfig,
    PSFConfig,
    ShearConfig,
    ShineConfig,
    VIConfig,
)


class TestDistributionConfig:
    """Test DistributionConfig validation."""

    def test_normal_distribution_valid(self):
        """Test valid Normal distribution configuration."""
        config = DistributionConfig(type="Normal", mean=0.0, sigma=1.0)
        assert config.type == "Normal"
        assert config.mean == 0.0
        assert config.sigma == 1.0

    def test_normal_distribution_missing_params(self):
        """Test Normal distribution with missing parameters raises error."""
        with pytest.raises(ValueError, match="Normal distribution requires"):
            DistributionConfig(type="Normal", mean=0.0)

    def test_uniform_distribution_valid(self):
        """Test valid Uniform distribution configuration."""
        config = DistributionConfig(type="Uniform", min=0.0, max=1.0)
        assert config.type == "Uniform"
        assert config.min == 0.0
        assert config.max == 1.0

    def test_uniform_distribution_missing_params(self):
        """Test Uniform distribution with missing parameters raises error."""
        with pytest.raises(ValueError, match="Uniform distribution requires"):
            DistributionConfig(type="Uniform", min=0.0)

    def test_uniform_distribution_invalid_range(self):
        """Test Uniform distribution with min >= max raises error."""
        with pytest.raises(ValueError, match="min .* must be less than max"):
            DistributionConfig(type="Uniform", min=1.0, max=0.5)

    def test_negative_sigma_raises_error(self):
        """Test negative sigma raises validation error."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            DistributionConfig(type="Normal", mean=0.0, sigma=-1.0)

    def test_zero_sigma_raises_error(self):
        """Test zero sigma raises validation error."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            DistributionConfig(type="Normal", mean=0.0, sigma=0.0)


class TestNoiseConfig:
    """Test NoiseConfig validation."""

    def test_valid_noise_config(self):
        """Test valid noise configuration."""
        config = NoiseConfig(type="Gaussian", sigma=1.0)
        assert config.type == "Gaussian"
        assert config.sigma == 1.0

    def test_negative_sigma_raises_error(self):
        """Test negative noise sigma raises error."""
        with pytest.raises(ValueError, match="Noise sigma must be positive"):
            NoiseConfig(type="Gaussian", sigma=-1.0)


class TestImageConfig:
    """Test ImageConfig."""

    def test_valid_image_config(self):
        """Test valid image configuration."""
        noise = NoiseConfig(type="Gaussian", sigma=1.0)
        config = ImageConfig(
            pixel_scale=0.2, size_x=32, size_y=32, n_objects=1, noise=noise
        )
        assert config.pixel_scale == 0.2
        assert config.size_x == 32
        assert config.size_y == 32
        assert config.n_objects == 1


class TestPSFConfig:
    """Test PSFConfig."""

    def test_gaussian_psf_config(self):
        """Test Gaussian PSF configuration."""
        config = PSFConfig(type="Gaussian", sigma=1.0)
        assert config.type == "Gaussian"
        assert config.sigma == 1.0

    def test_moffat_psf_config(self):
        """Test Moffat PSF configuration."""
        config = PSFConfig(type="Moffat", sigma=1.0, beta=2.5)
        assert config.type == "Moffat"
        assert config.sigma == 1.0
        assert config.beta == 2.5


class TestShearConfig:
    """Test ShearConfig."""

    def test_fixed_shear_values(self):
        """Test shear with fixed values."""
        config = ShearConfig(type="G1G2", g1=0.01, g2=-0.02)
        assert config.type == "G1G2"
        assert config.g1 == 0.01
        assert config.g2 == -0.02

    def test_shear_with_distributions(self):
        """Test shear with distribution configurations."""
        g1_dist = DistributionConfig(type="Normal", mean=0.0, sigma=0.05)
        g2_dist = DistributionConfig(type="Normal", mean=0.0, sigma=0.05)
        config = ShearConfig(type="G1G2", g1=g1_dist, g2=g2_dist)
        assert config.type == "G1G2"
        assert isinstance(config.g1, DistributionConfig)
        assert isinstance(config.g2, DistributionConfig)


class TestEllipticityConfig:
    """Test EllipticityConfig."""

    def test_fixed_ellipticity_values(self):
        """Test ellipticity with fixed values."""
        config = EllipticityConfig(type="E1E2", e1=0.1, e2=0.2)
        assert config.type == "E1E2"
        assert config.e1 == 0.1
        assert config.e2 == 0.2


class TestGalaxyConfig:
    """Test GalaxyConfig."""

    def test_exponential_galaxy_config(self):
        """Test Exponential galaxy configuration."""
        shear = ShearConfig(type="G1G2", g1=0.01, g2=-0.02)
        config = GalaxyConfig(
            type="Exponential",
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
        )
        assert config.type == "Exponential"
        assert config.flux == 1000.0
        assert config.half_light_radius == 1.0
        assert config.n is None


class TestMAPConfig:
    """Test MAPConfig."""

    def test_map_config_defaults(self):
        """Test MAP configuration with defaults."""
        config = MAPConfig()
        assert config.enabled is False
        assert config.num_steps == 1000
        assert config.learning_rate == 1e-2

    def test_map_config_custom_values(self):
        """Test MAP configuration with custom values."""
        config = MAPConfig(enabled=True, num_steps=500, learning_rate=1e-3)
        assert config.enabled is True
        assert config.num_steps == 500
        assert config.learning_rate == 1e-3


class TestNUTSConfig:
    """Test NUTSConfig."""

    def test_nuts_config_defaults(self):
        """Test NUTS configuration with defaults."""
        config = NUTSConfig()
        assert config.warmup == 500
        assert config.samples == 1000
        assert config.chains == 1
        assert config.dense_mass is False
        assert config.map_init is None

    def test_nuts_config_with_map_init(self):
        """Test NUTS configuration with MAP initialization."""
        map_config = MAPConfig(enabled=True)
        config = NUTSConfig(map_init=map_config)
        assert config.map_init is not None
        assert config.map_init.enabled is True

    def test_nuts_config_custom_values(self):
        """Test NUTS configuration with custom values."""
        config = NUTSConfig(warmup=200, samples=500, chains=4, dense_mass=True)
        assert config.warmup == 200
        assert config.samples == 500
        assert config.chains == 4
        assert config.dense_mass is True

    def test_nuts_config_invalid_warmup(self):
        """Test that non-positive warmup raises error."""
        with pytest.raises(ValueError, match="warmup must be positive"):
            NUTSConfig(warmup=0)

    def test_nuts_config_invalid_chains(self):
        """Test that non-positive chains raises error."""
        with pytest.raises(ValueError, match="Number of chains must be positive"):
            NUTSConfig(chains=0)


class TestVIConfig:
    """Test VIConfig."""

    def test_vi_config_defaults(self):
        """Test VI configuration with defaults."""
        config = VIConfig()
        assert config.num_steps == 5000
        assert config.learning_rate == 1e-3
        assert config.num_samples == 1000

    def test_vi_config_custom_values(self):
        """Test VI configuration with custom values."""
        config = VIConfig(num_steps=10000, learning_rate=5e-4, num_samples=2000)
        assert config.num_steps == 10000
        assert config.learning_rate == 5e-4
        assert config.num_samples == 2000

    def test_vi_config_invalid_num_steps(self):
        """Test that non-positive num_steps raises error."""
        with pytest.raises(ValueError, match="num_steps must be positive"):
            VIConfig(num_steps=0)

    def test_vi_config_invalid_learning_rate(self):
        """Test that non-positive learning rate raises error."""
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            VIConfig(learning_rate=-0.01)

    def test_vi_config_invalid_num_samples(self):
        """Test that non-positive num_samples raises error."""
        with pytest.raises(ValueError, match="num_samples must be positive"):
            VIConfig(num_samples=0)


class TestInferenceConfig:
    """Test InferenceConfig."""

    def test_inference_config_defaults(self):
        """Test inference configuration with defaults."""
        config = InferenceConfig()
        assert config.method == "nuts"
        assert config.nuts_config is None
        assert config.map_config is None
        assert config.vi_config is None
        assert config.rng_seed == 0

    def test_inference_config_method_nuts(self):
        """Test inference configuration with NUTS method."""
        nuts = NUTSConfig(warmup=200, samples=500)
        config = InferenceConfig(method="nuts", nuts_config=nuts)
        assert config.method == "nuts"
        assert config.nuts_config.warmup == 200
        assert config.nuts_config.samples == 500

    def test_inference_config_method_map(self):
        """Test inference configuration with MAP method."""
        map_cfg = MAPConfig(enabled=True, num_steps=2000)
        config = InferenceConfig(method="map", map_config=map_cfg)
        assert config.method == "map"
        assert config.map_config.num_steps == 2000

    def test_inference_config_method_vi(self):
        """Test inference configuration with VI method."""
        vi = VIConfig(num_steps=5000, learning_rate=0.001, num_samples=2000)
        config = InferenceConfig(method="vi", vi_config=vi)
        assert config.method == "vi"
        assert config.vi_config.num_steps == 5000
        assert config.vi_config.num_samples == 2000

    def test_inference_config_invalid_method(self):
        """Test that invalid method raises validation error."""
        with pytest.raises(ValueError):
            InferenceConfig(method="invalid")

    def test_inference_config_no_method_defaults_to_nuts(self):
        """Test that no method field defaults to nuts."""
        config = InferenceConfig(rng_seed=42)
        assert config.method == "nuts"


class TestShineConfig:
    """Test full ShineConfig."""

    def test_complete_shine_config(self):
        """Test complete SHINE configuration."""
        noise = NoiseConfig(type="Gaussian", sigma=1.0)
        image = ImageConfig(pixel_scale=0.2, size_x=32, size_y=32, noise=noise)
        psf = PSFConfig(type="Gaussian", sigma=1.0)
        shear = ShearConfig(type="G1G2", g1=0.01, g2=-0.02)
        galaxy = GalaxyConfig(
            type="Exponential",
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
        )
        inference = InferenceConfig()

        config = ShineConfig(
            image=image,
            psf=psf,
            gal=galaxy,
            inference=inference,
        )

        assert config.image.pixel_scale == 0.2
        assert config.psf.type == "Gaussian"
        assert config.gal.type == "Exponential"
        assert config.output_path == "results"


class TestConfigHandler:
    """Test ConfigHandler for loading YAML configurations."""

    def test_load_valid_config(self):
        """Test loading a valid YAML configuration."""
        config_data = {
            "image": {
                "pixel_scale": 0.2,
                "size_x": 32,
                "size_y": 32,
                "n_objects": 1,
                "noise": {"type": "Gaussian", "sigma": 1.0},
            },
            "psf": {"type": "Gaussian", "sigma": 1.0},
            "gal": {
                "type": "Exponential",
                "flux": 1000.0,
                "half_light_radius": 1.0,
                "shear": {"type": "G1G2", "g1": 0.01, "g2": -0.02},
            },
            "inference": {
                "method": "nuts",
                "nuts_config": {"warmup": 500, "samples": 1000},
            },
            "output_path": "results",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config_data, tmp)
            tmp_path = tmp.name

        try:
            config = ConfigHandler.load(tmp_path)
            assert config.image.pixel_scale == 0.2
            assert config.psf.type == "Gaussian"
            assert config.gal.type == "Exponential"
        finally:
            Path(tmp_path).unlink()

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            ConfigHandler.load("/nonexistent/path/config.yaml")

    def test_load_config_with_distributions(self):
        """Test loading configuration with distribution priors."""
        config_data = {
            "image": {
                "pixel_scale": 0.2,
                "size_x": 32,
                "size_y": 32,
                "noise": {"type": "Gaussian", "sigma": 1.0},
            },
            "psf": {"type": "Gaussian", "sigma": 1.0},
            "gal": {
                "type": "Exponential",
                "flux": {"type": "Normal", "mean": 1000.0, "sigma": 100.0},
                "half_light_radius": {"type": "Uniform", "min": 0.5, "max": 2.0},
                "shear": {
                    "type": "G1G2",
                    "g1": {"type": "Normal", "mean": 0.0, "sigma": 0.05},
                    "g2": {"type": "Normal", "mean": 0.0, "sigma": 0.05},
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config_data, tmp)
            tmp_path = tmp.name

        try:
            config = ConfigHandler.load(tmp_path)
            assert isinstance(config.gal.flux, DistributionConfig)
            assert config.gal.flux.mean == 1000.0
            assert isinstance(config.gal.half_light_radius, DistributionConfig)
            assert config.gal.half_light_radius.min == 0.5
        finally:
            Path(tmp_path).unlink()

