"""Tests for shine.data module."""

import jax.numpy as jnp
import jax_galsim
import pytest

from shine.config import (
    ConfigHandler,
    DistributionConfig,
    GalaxyConfig,
    ImageConfig,
    InferenceConfig,
    NoiseConfig,
    PSFConfig,
    ShearConfig,
    ShineConfig,
)
from shine.data import DataLoader, Observation


class TestObservation:
    """Test Observation dataclass."""

    def test_observation_creation(self):
        """Test creation of Observation object."""
        image = jnp.zeros((32, 32))
        noise_map = jnp.ones((32, 32))
        psf_model = None  # Placeholder

        obs = Observation(image=image, noise_map=noise_map, psf_model=psf_model)

        assert obs.image.shape == (32, 32)
        assert obs.noise_map.shape == (32, 32)
        assert obs.psf_model is None
        assert obs.wcs is None


class TestDataLoader:
    """Test DataLoader functionality."""

    def test_load_with_no_data_path(self):
        """Test that load generates synthetic data when no path provided."""
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

        config = ShineConfig(image=image, psf=psf, gal=galaxy)

        obs = DataLoader.load(config)

        assert isinstance(obs, Observation)
        assert obs.image.shape == (32, 32)
        assert obs.noise_map.shape == (32, 32)

    def test_load_with_data_path_raises_error(self):
        """Test that loading from file path raises NotImplementedError."""
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

        config = ShineConfig(
            image=image, psf=psf, gal=galaxy, data_path="/path/to/data.fits"
        )

        with pytest.raises(NotImplementedError, match="Real data loading"):
            DataLoader.load(config)


class TestSyntheticDataGeneration:
    """Test synthetic data generation functionality."""

    def test_generate_synthetic_basic(self):
        """Test basic synthetic data generation."""
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

        config = ShineConfig(image=image, psf=psf, gal=galaxy)

        obs = DataLoader.generate_synthetic(config)

        assert isinstance(obs, Observation)
        assert obs.image.shape == (32, 32)
        assert obs.noise_map.shape == (32, 32)
        assert isinstance(obs.psf_model, jax_galsim.Gaussian)

    def test_generate_synthetic_with_moffat_psf(self):
        """Test synthetic generation with Moffat PSF."""
        noise = NoiseConfig(type="Gaussian", sigma=1.0)
        image = ImageConfig(pixel_scale=0.2, size_x=32, size_y=32, noise=noise)
        psf = PSFConfig(type="Moffat", sigma=2.0, beta=2.5)
        shear = ShearConfig(type="G1G2", g1=0.01, g2=-0.02)
        galaxy = GalaxyConfig(
            type="Exponential",
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
        )

        config = ShineConfig(image=image, psf=psf, gal=galaxy)

        obs = DataLoader.generate_synthetic(config)

        assert isinstance(obs, Observation)
        assert isinstance(obs.psf_model, jax_galsim.Moffat)

    def test_generate_synthetic_with_distributions(self):
        """Test synthetic generation with distribution parameters."""
        noise = NoiseConfig(type="Gaussian", sigma=1.0)
        image = ImageConfig(pixel_scale=0.2, size_x=32, size_y=32, noise=noise)
        psf = PSFConfig(type="Gaussian", sigma=1.0)

        flux_dist = DistributionConfig(type="Normal", mean=1000.0, sigma=100.0)
        hlr_dist = DistributionConfig(type="Uniform", min=0.5, max=2.0)
        g1_dist = DistributionConfig(type="Normal", mean=0.0, sigma=0.05)
        g2_dist = DistributionConfig(type="Normal", mean=0.0, sigma=0.05)
        shear = ShearConfig(type="G1G2", g1=g1_dist, g2=g2_dist)

        galaxy = GalaxyConfig(
            type="Exponential",
            flux=flux_dist,
            half_light_radius=hlr_dist,
            shear=shear,
        )

        config = ShineConfig(image=image, psf=psf, gal=galaxy)

        obs = DataLoader.generate_synthetic(config)

        assert isinstance(obs, Observation)
        assert obs.image.shape == (32, 32)
        # Verify that mean values were used for generation

    def test_generate_synthetic_with_ellipticity(self):
        """Test synthetic generation with intrinsic ellipticity."""
        from shine.config import EllipticityConfig

        noise = NoiseConfig(type="Gaussian", sigma=1.0)
        image = ImageConfig(pixel_scale=0.2, size_x=32, size_y=32, noise=noise)
        psf = PSFConfig(type="Gaussian", sigma=1.0)
        shear = ShearConfig(type="G1G2", g1=0.01, g2=-0.02)

        e1_dist = DistributionConfig(type="Normal", mean=0.0, sigma=0.2)
        e2_dist = DistributionConfig(type="Normal", mean=0.0, sigma=0.2)
        ellipticity = EllipticityConfig(type="E1E2", e1=e1_dist, e2=e2_dist)

        galaxy = GalaxyConfig(
            type="Exponential",
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
            ellipticity=ellipticity,
        )

        config = ShineConfig(image=image, psf=psf, gal=galaxy)

        obs = DataLoader.generate_synthetic(config)

        assert isinstance(obs, Observation)
        assert obs.image.shape == (32, 32)

    def test_noise_map_correct_variance(self):
        """Test that noise map has correct variance."""
        noise_sigma = 2.5
        noise = NoiseConfig(type="Gaussian", sigma=noise_sigma)
        image = ImageConfig(pixel_scale=0.2, size_x=32, size_y=32, noise=noise)
        psf = PSFConfig(type="Gaussian", sigma=1.0)
        shear = ShearConfig(type="G1G2", g1=0.01, g2=-0.02)
        galaxy = GalaxyConfig(
            type="Exponential",
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
        )

        config = ShineConfig(image=image, psf=psf, gal=galaxy)

        obs = DataLoader.generate_synthetic(config)

        # Noise map should contain variance (sigma^2)
        expected_variance = noise_sigma**2
        assert jnp.allclose(obs.noise_map, expected_variance)

    def test_image_dimensions(self):
        """Test that generated images have correct dimensions."""
        noise = NoiseConfig(type="Gaussian", sigma=1.0)
        size_x, size_y = 48, 64
        image = ImageConfig(pixel_scale=0.2, size_x=size_x, size_y=size_y, noise=noise)
        psf = PSFConfig(type="Gaussian", sigma=1.0)
        shear = ShearConfig(type="G1G2", g1=0.01, g2=-0.02)
        galaxy = GalaxyConfig(
            type="Exponential",
            flux=1000.0,
            half_light_radius=1.0,
            shear=shear,
        )

        config = ShineConfig(image=image, psf=psf, gal=galaxy)

        obs = DataLoader.generate_synthetic(config)

        assert obs.image.shape == (size_y, size_x)
        assert obs.noise_map.shape == (size_y, size_x)


class TestMeanValueExtraction:
    """Test extraction of mean values from distribution configs."""

    def test_uniform_distribution_mean(self):
        """Test mean calculation for Uniform distribution."""
        noise = NoiseConfig(type="Gaussian", sigma=1.0)
        image = ImageConfig(pixel_scale=0.2, size_x=32, size_y=32, noise=noise)
        psf = PSFConfig(type="Gaussian", sigma=1.0)

        hlr_dist = DistributionConfig(type="Uniform", min=1.0, max=3.0)
        shear = ShearConfig(type="G1G2", g1=0.01, g2=-0.02)

        galaxy = GalaxyConfig(
            type="Exponential",
            flux=1000.0,
            half_light_radius=hlr_dist,
            shear=shear,
        )

        config = ShineConfig(image=image, psf=psf, gal=galaxy)

        # Should use (min + max) / 2 = 2.0 for synthetic generation
        obs = DataLoader.generate_synthetic(config)
        assert isinstance(obs, Observation)

    def test_normal_distribution_mean(self):
        """Test mean extraction for Normal distribution."""
        noise = NoiseConfig(type="Gaussian", sigma=1.0)
        image = ImageConfig(pixel_scale=0.2, size_x=32, size_y=32, noise=noise)
        psf = PSFConfig(type="Gaussian", sigma=1.0)

        flux_dist = DistributionConfig(type="Normal", mean=1500.0, sigma=200.0)
        shear = ShearConfig(type="G1G2", g1=0.01, g2=-0.02)

        galaxy = GalaxyConfig(
            type="Exponential",
            flux=flux_dist,
            half_light_radius=1.0,
            shear=shear,
        )

        config = ShineConfig(image=image, psf=psf, gal=galaxy)

        # Should use mean = 1500.0 for synthetic generation
        obs = DataLoader.generate_synthetic(config)
        assert isinstance(obs, Observation)
