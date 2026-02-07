"""Tests for shine.validation.statistics module."""

import pytest

from shine.validation.statistics import (
    BiasResult,
    compute_bias_regression,
    compute_bias_single_point,
    compute_paired_response,
    jackknife_bias,
)


class TestComputeBiasSinglePoint:
    """Tests for compute_bias_single_point()."""

    def test_no_bias(self):
        """When g_est == g_true, m should be 0."""
        result = compute_bias_single_point(
            g_true=0.02, g_est_mean=0.02, g_est_std=0.001, component="g1"
        )
        assert isinstance(result, BiasResult)
        assert result.m == pytest.approx(0.0, abs=1e-10)
        assert result.method == "single_point"
        assert result.component == "g1"

    def test_multiplicative_bias(self):
        """When g_est = 1.01 * g_true, m should be 0.01."""
        g_true = 0.02
        g_est = 1.01 * g_true  # 1% multiplicative bias
        result = compute_bias_single_point(
            g_true=g_true, g_est_mean=g_est, g_est_std=0.001, component="g1"
        )
        assert result.m == pytest.approx(0.01, abs=1e-10)

    def test_negative_bias(self):
        """Negative multiplicative bias."""
        g_true = 0.02
        g_est = 0.99 * g_true  # -1% bias
        result = compute_bias_single_point(
            g_true=g_true, g_est_mean=g_est, g_est_std=0.001, component="g2"
        )
        assert result.m == pytest.approx(-0.01, abs=1e-10)
        assert result.component == "g2"

    def test_m_err_proportional_to_std(self):
        """m_err = g_est_std / |g_true|."""
        result = compute_bias_single_point(
            g_true=0.02, g_est_mean=0.02, g_est_std=0.001, component="g1"
        )
        assert result.m_err == pytest.approx(0.001 / 0.02, abs=1e-10)

    def test_zero_g_true_raises(self):
        """Cannot compute m for g_true=0."""
        with pytest.raises(ValueError, match="Cannot compute multiplicative bias"):
            compute_bias_single_point(
                g_true=0.0, g_est_mean=0.001, g_est_std=0.001, component="g1"
            )

    def test_c_is_zero_for_single_point(self):
        """For single-point, c = 0 by construction."""
        result = compute_bias_single_point(
            g_true=0.02, g_est_mean=0.021, g_est_std=0.001, component="g1"
        )
        assert result.c == 0.0


class TestStubs:
    """Tests that stub functions raise NotImplementedError."""

    def test_compute_bias_regression_raises(self):
        import numpy as np

        with pytest.raises(NotImplementedError, match="Level 1"):
            compute_bias_regression(
                np.array([0.01, 0.02]), np.array([0.01, 0.02])
            )

    def test_compute_paired_response_raises(self):
        import numpy as np

        with pytest.raises(NotImplementedError, match="Level 1"):
            compute_paired_response(np.array([0.02]), np.array([-0.02]), 0.02)

    def test_jackknife_bias_raises(self):
        import numpy as np

        with pytest.raises(NotImplementedError, match="Level 1"):
            jackknife_bias(np.array([0.01]), np.array([0.01]))
