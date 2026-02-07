"""Tests for shine.validation.bias_config Pydantic models."""

import pytest
from pydantic import ValidationError

from shine.validation.bias_config import (
    AcceptanceCriteria,
    BiasLevel,
    BiasRunConfig,
    BiasTestConfig,
    ConvergenceThresholds,
    ShearGrid,
)


class TestBiasLevel:
    """Tests for BiasLevel enum."""

    def test_all_levels(self):
        assert BiasLevel.level_0.value == "level_0"
        assert BiasLevel.level_1.value == "level_1"
        assert BiasLevel.level_2.value == "level_2"
        assert BiasLevel.level_3.value == "level_3"

    def test_from_string(self):
        assert BiasLevel("level_0") == BiasLevel.level_0


class TestConvergenceThresholds:
    """Tests for ConvergenceThresholds model."""

    def test_defaults(self):
        ct = ConvergenceThresholds()
        assert ct.rhat_max == 1.05
        assert ct.ess_min == 100
        assert ct.divergence_frac_max == 0.01
        assert ct.bfmi_min == 0.3

    def test_custom_values(self):
        ct = ConvergenceThresholds(rhat_max=1.1, ess_min=200, bfmi_min=0.5)
        assert ct.rhat_max == 1.1
        assert ct.ess_min == 200
        assert ct.bfmi_min == 0.5

    def test_rhat_below_one_raises(self):
        with pytest.raises(ValidationError, match="rhat_max must be >= 1.0"):
            ConvergenceThresholds(rhat_max=0.9)

    def test_ess_zero_raises(self):
        with pytest.raises(ValidationError, match="ess_min must be positive"):
            ConvergenceThresholds(ess_min=0)

    def test_ess_negative_raises(self):
        with pytest.raises(ValidationError, match="ess_min must be positive"):
            ConvergenceThresholds(ess_min=-10)


class TestShearGrid:
    """Tests for ShearGrid model."""

    def test_defaults(self):
        sg = ShearGrid()
        assert len(sg.values) == 7
        assert 0.0 in sg.values

    def test_custom_values(self):
        sg = ShearGrid(values=[0.01, 0.02, 0.03])
        assert sg.values == [0.01, 0.02, 0.03]

    def test_shear_out_of_range_raises(self):
        with pytest.raises(ValidationError, match="Shear values must have"):
            ShearGrid(values=[0.01, 1.5])


class TestAcceptanceCriteria:
    """Tests for AcceptanceCriteria model."""

    def test_defaults(self):
        ac = AcceptanceCriteria()
        assert ac.max_offset_sigma == 3.0
        assert ac.max_posterior_width is None
        assert ac.max_abs_m is None
        assert ac.coverage_levels == [0.68, 0.95]

    def test_custom(self):
        ac = AcceptanceCriteria(
            max_offset_sigma=1.0,
            max_posterior_width=1e-3,
            max_abs_m=0.003,
        )
        assert ac.max_offset_sigma == 1.0
        assert ac.max_posterior_width == 1e-3
        assert ac.max_abs_m == 0.003


class TestBiasRunConfig:
    """Tests for BiasRunConfig model."""

    def test_valid(self):
        rc = BiasRunConfig(
            shine_config_path="config.yaml",
            g1_true=0.02,
            g2_true=-0.01,
            seed=42,
            output_dir="results",
            run_id="r0001",
        )
        assert rc.g1_true == 0.02
        assert rc.paired is False

    def test_negative_seed_raises(self):
        with pytest.raises(ValidationError, match="seed must be non-negative"):
            BiasRunConfig(
                shine_config_path="config.yaml",
                g1_true=0.02,
                g2_true=-0.01,
                seed=-1,
                output_dir="results",
                run_id="r0001",
            )

    def test_shear_out_of_range_raises(self):
        with pytest.raises(ValidationError, match="Shear must have"):
            BiasRunConfig(
                shine_config_path="config.yaml",
                g1_true=1.5,
                g2_true=0.0,
                seed=0,
                output_dir="results",
                run_id="r0001",
            )


class TestBiasTestConfig:
    """Tests for BiasTestConfig model."""

    def test_defaults(self):
        btc = BiasTestConfig(shine_config_path="config.yaml")
        assert btc.level == BiasLevel.level_0
        assert btc.n_realizations == 1
        assert isinstance(btc.convergence, ConvergenceThresholds)
        assert isinstance(btc.acceptance, AcceptanceCriteria)

    def test_zero_realizations_raises(self):
        with pytest.raises(ValidationError, match="n_realizations must be positive"):
            BiasTestConfig(shine_config_path="config.yaml", n_realizations=0)

    def test_custom(self):
        btc = BiasTestConfig(
            shine_config_path="config.yaml",
            level="level_1",
            n_realizations=10,
            paired=True,
            shear_grid={"values": [0.01, 0.02]},
        )
        assert btc.level == BiasLevel.level_1
        assert btc.n_realizations == 10
        assert btc.paired is True
        assert btc.shear_grid.values == [0.01, 0.02]
