"""Tests for shine.prior_utils module."""

import jax.numpy as jnp
import numpyro
import numpyro.handlers as handlers
import pytest
from jax import random

from shine.config import DistributionConfig
from shine.prior_utils import parse_prior


class TestParsepriorFixedValues:
    """Test parse_prior with fixed numeric values."""

    def test_float_passthrough(self):
        """Fixed float values are returned directly."""
        assert parse_prior("x", 1.5) == 1.5

    def test_int_passthrough(self):
        """Fixed int values are returned as float."""
        result = parse_prior("x", 3)
        assert result == 3.0
        assert isinstance(result, float)


class TestParsePriorDistributions:
    """Test parse_prior with standard distribution configs."""

    def test_normal_distribution(self):
        """Normal distribution creates a sample site with correct params."""
        cfg = DistributionConfig(type="Normal", mean=1.0, sigma=0.5)

        def model():
            return parse_prior("x", cfg)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model, rng)).get_trace()
        assert "x" in trace
        assert trace["x"]["type"] == "sample"

    def test_lognormal_distribution(self):
        """LogNormal distribution creates a sample site."""
        cfg = DistributionConfig(type="LogNormal", mean=100.0, sigma=0.5)

        def model():
            return parse_prior("x", cfg)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model, rng)).get_trace()
        assert "x" in trace
        # LogNormal samples are always positive
        assert trace["x"]["value"] > 0

    def test_uniform_distribution(self):
        """Uniform distribution creates a sample site."""
        cfg = DistributionConfig(type="Uniform", min=0.0, max=10.0)

        def model():
            return parse_prior("x", cfg)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model, rng)).get_trace()
        assert "x" in trace
        val = float(trace["x"]["value"])
        assert 0.0 <= val <= 10.0

    def test_unknown_distribution_raises(self):
        """Unknown distribution type raises ValueError."""
        cfg = DistributionConfig.model_construct(type="Cauchy", sigma=1.0)
        with pytest.raises(ValueError, match="Unknown distribution type"):
            parse_prior("x", cfg)


class TestParsePriorCatalogCentered:
    """Test parse_prior with center='catalog' priors."""

    def test_lognormal_catalog_centered(self):
        """LogNormal with center='catalog' uses catalog values as median."""
        cfg = DistributionConfig(type="LogNormal", center="catalog", sigma=0.5)
        catalog = jnp.array([100.0, 200.0, 300.0])

        def model():
            with numpyro.plate("sources", 3):
                return parse_prior("flux", cfg, catalog_values=catalog)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model, rng)).get_trace()
        assert "flux" in trace
        assert trace["flux"]["value"].shape == (3,)
        # All samples should be positive (LogNormal)
        assert jnp.all(trace["flux"]["value"] > 0)

    def test_normal_catalog_centered(self):
        """Normal with center='catalog' uses catalog values as mean."""
        cfg = DistributionConfig(type="Normal", center="catalog", sigma=0.1)
        catalog = jnp.array([1.0, 2.0, 3.0])

        def model():
            with numpyro.plate("sources", 3):
                return parse_prior("pos", cfg, catalog_values=catalog)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model, rng)).get_trace()
        assert "pos" in trace
        assert trace["pos"]["value"].shape == (3,)

    def test_catalog_centered_without_values_raises(self):
        """center='catalog' without catalog_values raises ValueError."""
        cfg = DistributionConfig(type="LogNormal", center="catalog", sigma=0.5)
        with pytest.raises(ValueError, match="no catalog_values"):
            parse_prior("flux", cfg)

    def test_catalog_centered_none_values_raises(self):
        """center='catalog' with catalog_values=None raises ValueError."""
        cfg = DistributionConfig(type="LogNormal", center="catalog", sigma=0.5)
        with pytest.raises(ValueError, match="no catalog_values"):
            parse_prior("flux", cfg, catalog_values=None)
