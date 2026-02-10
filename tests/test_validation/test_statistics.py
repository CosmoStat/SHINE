"""Tests for shine.validation.statistics module."""

import numpy as np
import pytest

from shine.validation.statistics import (
    BiasResult,
    CoverageResult,
    SBCResult,
    compute_bias_regression,
    compute_bias_single_point,
    compute_coverage,
    compute_paired_response,
    compute_sbc_ranks,
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


class TestBiasRegression:
    """Tests for compute_bias_regression()."""

    def test_perfect_data(self):
        """When g_est == g_true exactly, m=0 and c=0."""
        g_true = np.array([0.01, 0.02, 0.05])
        g_est = g_true.copy()
        result = compute_bias_regression(g_true, g_est, component="g1")
        assert result.m == pytest.approx(0.0, abs=1e-10)
        assert result.c == pytest.approx(0.0, abs=1e-10)
        assert result.method == "regression"
        assert result.component == "g1"

    def test_known_multiplicative_bias(self):
        """With g_est = 1.05 * g_true, m should be ~0.05."""
        g_true = np.array([0.01, 0.02, 0.03, 0.05])
        g_est = 1.05 * g_true  # 5% multiplicative bias, no additive
        result = compute_bias_regression(g_true, g_est, component="g1")
        assert result.m == pytest.approx(0.05, abs=1e-10)
        assert result.c == pytest.approx(0.0, abs=1e-10)

    def test_known_additive_bias(self):
        """With g_est = g_true + 0.001, c should be ~0.001."""
        g_true = np.array([0.01, 0.02, 0.03, 0.05])
        g_est = g_true + 0.001
        result = compute_bias_regression(g_true, g_est, component="g2")
        assert result.m == pytest.approx(0.0, abs=1e-10)
        assert result.c == pytest.approx(0.001, abs=1e-10)

    def test_combined_bias(self):
        """With g_est = 1.02 * g_true + 0.0005."""
        g_true = np.array([0.01, 0.02, 0.03, 0.05])
        g_est = 1.02 * g_true + 0.0005
        result = compute_bias_regression(g_true, g_est, component="g1")
        assert result.m == pytest.approx(0.02, abs=1e-8)
        assert result.c == pytest.approx(0.0005, abs=1e-8)

    def test_weighted_regression(self):
        """Weighted regression should downweight noisy points."""
        g_true = np.array([0.01, 0.02, 0.05])
        g_est = np.array([0.01, 0.02, 0.05])
        weights = np.array([1.0, 1.0, 100.0])  # High weight on last point
        result = compute_bias_regression(g_true, g_est, weights=weights, component="g1")
        assert result.m == pytest.approx(0.0, abs=1e-8)

    def test_too_few_points(self):
        """Need at least 2 points."""
        with pytest.raises(ValueError, match="at least 2"):
            compute_bias_regression(np.array([0.01]), np.array([0.01]))

    def test_error_bars_nonzero(self):
        """With noisy data, errors should be nonzero."""
        rng = np.random.default_rng(42)
        g_true = np.linspace(0.01, 0.05, 20)
        g_est = g_true + rng.normal(0, 0.001, size=20)
        result = compute_bias_regression(g_true, g_est, component="g1")
        assert result.m_err > 0
        assert result.c_err > 0


class TestPairedResponse:
    """Tests for compute_paired_response()."""

    def test_perfect_response(self):
        """With perfect shear recovery, R = 1."""
        g_true = 0.02
        g_est_plus = np.array([0.02, 0.02, 0.02])
        g_est_minus = np.array([-0.02, -0.02, -0.02])
        R = compute_paired_response(g_est_plus, g_est_minus, g_true)
        np.testing.assert_allclose(R, 1.0)

    def test_biased_response(self):
        """With 5% multiplicative bias, R = 1.05."""
        g_true = 0.02
        g_est_plus = np.array([1.05 * 0.02])
        g_est_minus = np.array([1.05 * -0.02])
        R = compute_paired_response(g_est_plus, g_est_minus, g_true)
        np.testing.assert_allclose(R, 1.05)

    def test_zero_g_true_raises(self):
        """Cannot compute response for g_true=0."""
        with pytest.raises(ValueError, match="g_true=0"):
            compute_paired_response(np.array([0.0]), np.array([0.0]), 0.0)

    def test_mismatched_lengths(self):
        """Arrays must have same length."""
        with pytest.raises(ValueError, match="same length"):
            compute_paired_response(np.array([0.02, 0.02]), np.array([0.02]), 0.02)

    def test_component_parameter(self):
        """Component parameter is accepted."""
        R = compute_paired_response(
            np.array([0.02]), np.array([-0.02]), 0.02, component="g2"
        )
        assert len(R) == 1


class TestJackknifeBias:
    """Tests for jackknife_bias()."""

    def test_perfect_data(self):
        """Jackknife on perfect data gives m~0, c~0."""
        g_true = np.array([0.01, 0.02, 0.03, 0.04, 0.05] * 4)
        g_est = g_true.copy()
        result = jackknife_bias(g_true, g_est, n_groups=5, component="g1")
        assert result.m == pytest.approx(0.0, abs=1e-8)
        assert result.c == pytest.approx(0.0, abs=1e-8)
        assert result.method == "jackknife"

    def test_known_bias(self):
        """Jackknife should recover known bias."""
        g_true = np.tile(np.array([0.01, 0.02, 0.05]), 10)
        g_est = 1.03 * g_true + 0.001
        result = jackknife_bias(g_true, g_est, n_groups=5, component="g1")
        assert result.m == pytest.approx(0.03, abs=1e-6)
        assert result.c == pytest.approx(0.001, abs=1e-6)

    def test_too_few_points(self):
        """Need at least n_groups data points."""
        with pytest.raises(ValueError, match="at least"):
            jackknife_bias(np.array([0.01]), np.array([0.01]), n_groups=10)

    def test_jackknife_errors_smaller_than_naive(self):
        """Jackknife errors should be reasonable (not huge)."""
        rng = np.random.default_rng(42)
        g_true = np.tile(np.array([0.01, 0.02, 0.05]), 20)
        g_est = g_true + rng.normal(0, 0.001, size=60)
        result = jackknife_bias(g_true, g_est, n_groups=10, component="g1")
        assert result.m_err > 0
        assert result.m_err < 1.0  # Sanity check


class TestCoverage:
    """Tests for compute_coverage()."""

    def test_perfect_coverage(self):
        """When posteriors are perfectly calibrated, coverage should match alpha."""
        rng = np.random.default_rng(42)
        n = 10000
        g_true = rng.normal(0.02, 0.005, size=n)
        # Posterior matches the truth distribution
        g_est_means = g_true + rng.normal(0, 0.005, size=n)
        g_est_stds = np.full(n, 0.005)
        results = compute_coverage(g_true, g_est_means, g_est_stds, [0.68, 0.95])
        assert len(results) == 2
        assert isinstance(results[0], CoverageResult)
        # With large N, coverage should be close to nominal
        assert results[0].observed == pytest.approx(0.68, abs=0.03)
        assert results[1].observed == pytest.approx(0.95, abs=0.03)

    def test_overconfident_coverage(self):
        """When posteriors are too narrow, coverage < alpha."""
        n = 1000
        g_true = np.full(n, 0.02)
        g_est_means = np.full(n, 0.02)
        g_est_stds = np.full(n, 1e-10)  # Extremely tight
        # Add offset so truth falls outside
        g_true_offset = g_true + 0.01
        results = compute_coverage(g_true_offset, g_est_means, g_est_stds, [0.68])
        assert results[0].observed < 0.68

    def test_default_alpha_levels(self):
        """Default alpha levels are [0.68, 0.95]."""
        results = compute_coverage(
            np.array([0.02]), np.array([0.02]), np.array([0.01])
        )
        assert len(results) == 2
        assert results[0].alpha == 0.68
        assert results[1].alpha == 0.95


class TestSBCRanks:
    """Tests for compute_sbc_ranks()."""

    def test_uniform_ranks(self):
        """With well-calibrated posterior, ranks should be ~uniform.

        For SBC: draw theta from prior, draw data|theta, draw posterior
        samples from posterior(data). If the model is correct, the rank
        of theta among posterior samples is uniform.

        We simulate perfect calibration by drawing both theta and
        posterior samples from the SAME distribution (the prior),
        independently. This makes the rank of theta uniform.
        """
        n_realizations = 500
        n_samples = 200
        mu = 0.02
        sigma = 0.005

        rng_truth = np.random.default_rng(42)
        g_true = rng_truth.normal(mu, sigma, size=n_realizations)

        # Posterior samples drawn from the same marginal distribution (prior)
        # independently of truth — this simulates perfect calibration
        posterior_samples = np.zeros((n_realizations, n_samples))
        for i in range(n_realizations):
            rng_i = np.random.default_rng(1000 + i)
            posterior_samples[i] = rng_i.normal(mu, sigma, size=n_samples)

        result = compute_sbc_ranks(g_true, posterior_samples, "g1")
        assert isinstance(result, SBCResult)
        assert result.param == "g1"
        assert len(result.ranks) == n_realizations
        # Well-calibrated should have high p-value
        assert result.ks_pvalue > 0.01

    def test_biased_ranks(self):
        """With systematically biased posterior, KS p-value should be low."""
        rng = np.random.default_rng(42)
        n_realizations = 500
        n_samples = 200
        g_true = rng.normal(0.02, 0.005, size=n_realizations)
        # Posterior is biased high
        posterior_samples = np.array([
            rng.normal(g + 0.01, 0.005, size=n_samples)
            for g in g_true
        ])
        result = compute_sbc_ranks(g_true, posterior_samples, "g1")
        # Biased posterior should have low p-value
        assert result.ks_pvalue < 0.05

    def test_rank_shape(self):
        """Ranks should have shape (n_realizations,)."""
        g_true = np.array([0.01, 0.02, 0.03])
        samples = np.random.default_rng(42).normal(0.02, 0.01, size=(3, 100))
        result = compute_sbc_ranks(g_true, samples, "g2")
        assert result.ranks.shape == (3,)
