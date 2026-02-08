"""Tests for shine.validation.extraction module."""

import arviz as az
import numpy as np
import pytest

from shine.validation.bias_config import ConvergenceThresholds
from shine.validation.extraction import (
    ConvergenceDiagnostics,
    RealizationResult,
    ShearEstimates,
    check_convergence,
    extract_convergence_diagnostics,
    extract_realization,
    extract_shear_estimates,
)


def _make_mock_idata(
    g1_mean=0.02,
    g2_mean=-0.01,
    g1_std=0.001,
    g2_std=0.001,
    n_chains=2,
    n_samples=500,
    n_divergences=0,
    inference_method=None,
):
    """Create a mock InferenceData object for testing.

    Args:
        inference_method: If provided, sets the inference_method attr on posterior.
    """
    rng = np.random.default_rng(42)
    posterior = {
        "g1": rng.normal(g1_mean, g1_std, size=(n_chains, n_samples)),
        "g2": rng.normal(g2_mean, g2_std, size=(n_chains, n_samples)),
    }

    # Build sample_stats with diverging and energy
    diverging = np.zeros((n_chains, n_samples), dtype=bool)
    if n_divergences > 0:
        flat_indices = rng.choice(
            n_chains * n_samples, size=n_divergences, replace=False
        )
        diverging.flat[flat_indices] = True

    energy = rng.normal(100, 10, size=(n_chains, n_samples))

    sample_stats = {"diverging": diverging, "energy": energy}

    idata = az.from_dict(posterior=posterior, sample_stats=sample_stats)
    if inference_method is not None:
        idata.posterior.attrs["inference_method"] = inference_method
    return idata


class TestExtractConvergenceDiagnostics:
    """Tests for extract_convergence_diagnostics()."""

    def test_basic_extraction(self):
        idata = _make_mock_idata()
        diag = extract_convergence_diagnostics(idata)

        assert isinstance(diag, ConvergenceDiagnostics)
        assert "g1" in diag.rhat
        assert "g2" in diag.rhat
        assert "g1" in diag.ess
        assert "g2" in diag.ess
        assert diag.divergences == 0
        assert diag.divergence_frac == 0.0
        assert diag.n_chains == 2
        assert diag.n_samples == 1000  # 2 * 500

    def test_rhat_near_one(self):
        idata = _make_mock_idata()
        diag = extract_convergence_diagnostics(idata)
        # Well-mixed chains should have rhat close to 1
        assert diag.rhat["g1"] < 1.1
        assert diag.rhat["g2"] < 1.1

    def test_ess_positive(self):
        idata = _make_mock_idata()
        diag = extract_convergence_diagnostics(idata)
        assert diag.ess["g1"] > 0
        assert diag.ess["g2"] > 0

    def test_divergences_counted(self):
        idata = _make_mock_idata(n_divergences=10)
        diag = extract_convergence_diagnostics(idata)
        assert diag.divergences == 10
        assert diag.divergence_frac == pytest.approx(10 / 1000, abs=1e-6)

    def test_bfmi_computed(self):
        idata = _make_mock_idata()
        diag = extract_convergence_diagnostics(idata)
        assert isinstance(diag.bfmi, list)
        assert len(diag.bfmi) == 2  # 2 chains


class TestExtractShearEstimates:
    """Tests for extract_shear_estimates()."""

    def test_basic_extraction(self):
        idata = _make_mock_idata(g1_mean=0.02, g1_std=0.001)
        est = extract_shear_estimates(idata, "g1")

        assert isinstance(est, ShearEstimates)
        assert est.mean == pytest.approx(0.02, abs=0.005)
        assert est.std == pytest.approx(0.001, abs=0.005)
        assert est.median == pytest.approx(0.02, abs=0.005)

    def test_percentiles(self):
        idata = _make_mock_idata()
        est = extract_shear_estimates(idata, "g1")

        assert 2.5 in est.percentiles
        assert 16.0 in est.percentiles
        assert 50.0 in est.percentiles
        assert 84.0 in est.percentiles
        assert 97.5 in est.percentiles
        # 2.5 percentile < 97.5 percentile
        assert est.percentiles[2.5] < est.percentiles[97.5]


class TestCheckConvergence:
    """Tests for check_convergence()."""

    def test_pass_case(self):
        idata = _make_mock_idata()
        diag = extract_convergence_diagnostics(idata)
        thresholds = ConvergenceThresholds()
        assert check_convergence(diag, thresholds) is True

    def test_fail_rhat(self):
        diag = ConvergenceDiagnostics(
            rhat={"g1": 1.2, "g2": 1.0},
            ess={"g1": 500, "g2": 500},
            divergences=0,
            divergence_frac=0.0,
            bfmi=[0.8, 0.8],
            n_samples=1000,
            n_chains=2,
        )
        thresholds = ConvergenceThresholds(rhat_max=1.05)
        assert check_convergence(diag, thresholds) is False

    def test_fail_ess(self):
        diag = ConvergenceDiagnostics(
            rhat={"g1": 1.0, "g2": 1.0},
            ess={"g1": 50, "g2": 500},
            divergences=0,
            divergence_frac=0.0,
            bfmi=[0.8, 0.8],
            n_samples=1000,
            n_chains=2,
        )
        thresholds = ConvergenceThresholds(ess_min=100)
        assert check_convergence(diag, thresholds) is False

    def test_fail_divergences(self):
        diag = ConvergenceDiagnostics(
            rhat={"g1": 1.0, "g2": 1.0},
            ess={"g1": 500, "g2": 500},
            divergences=20,
            divergence_frac=0.02,
            bfmi=[0.8, 0.8],
            n_samples=1000,
            n_chains=2,
        )
        thresholds = ConvergenceThresholds(divergence_frac_max=0.01)
        assert check_convergence(diag, thresholds) is False

    def test_fail_bfmi(self):
        diag = ConvergenceDiagnostics(
            rhat={"g1": 1.0, "g2": 1.0},
            ess={"g1": 500, "g2": 500},
            divergences=0,
            divergence_frac=0.0,
            bfmi=[0.1, 0.8],
            n_samples=1000,
            n_chains=2,
        )
        thresholds = ConvergenceThresholds(bfmi_min=0.3)
        assert check_convergence(diag, thresholds) is False


class TestExtractRealization:
    """Tests for extract_realization()."""

    def test_full_extraction(self):
        idata = _make_mock_idata()
        thresholds = ConvergenceThresholds()
        result = extract_realization(
            idata, g1_true=0.02, g2_true=-0.01,
            run_id="test001", seed=42, thresholds=thresholds,
        )

        assert isinstance(result, RealizationResult)
        assert result.run_id == "test001"
        assert result.g1_true == 0.02
        assert result.g2_true == -0.01
        assert isinstance(result.g1, ShearEstimates)
        assert isinstance(result.g2, ShearEstimates)
        assert isinstance(result.diagnostics, ConvergenceDiagnostics)
        assert isinstance(result.passed_convergence, bool)
        assert result.seed == 42


class TestMethodAwareDiagnostics:
    """Tests for method-aware convergence diagnostics."""

    def test_map_diagnostics_sentinels(self):
        """MAP idata (1 chain, 1 draw) returns sentinel diagnostics."""
        idata = _make_mock_idata(
            n_chains=1, n_samples=1, inference_method="map"
        )
        diag = extract_convergence_diagnostics(idata)
        assert diag.rhat == {"g1": 1.0, "g2": 1.0}
        assert diag.ess == {"g1": 1.0, "g2": 1.0}
        assert diag.divergences == 0
        assert diag.bfmi == []
        assert diag.n_chains == 1
        assert diag.n_samples == 1

    def test_map_check_convergence_always_true(self):
        """MAP method always passes convergence."""
        diag = ConvergenceDiagnostics(
            rhat={"g1": 1.0, "g2": 1.0},
            ess={"g1": 1.0, "g2": 1.0},
            divergences=0,
            divergence_frac=0.0,
            bfmi=[],
            n_samples=1,
            n_chains=1,
        )
        thresholds = ConvergenceThresholds(ess_min=100)
        # MAP always returns True regardless of ESS
        assert check_convergence(diag, thresholds, method="map") is True

    def test_vi_diagnostics_ess_computed(self):
        """VI idata (1 chain, N draws) computes ESS, rhat=1.0."""
        idata = _make_mock_idata(
            n_chains=1, n_samples=1000, inference_method="vi"
        )
        diag = extract_convergence_diagnostics(idata)
        assert diag.rhat == {"g1": 1.0, "g2": 1.0}
        assert diag.ess["g1"] > 0
        assert diag.ess["g2"] > 0
        assert diag.divergences == 0
        assert diag.bfmi == []
        assert diag.n_chains == 1

    def test_vi_check_convergence_only_checks_ess(self):
        """VI convergence only checks ESS, not rhat/divergences/bfmi."""
        diag = ConvergenceDiagnostics(
            rhat={"g1": 2.0, "g2": 2.0},  # Would fail for NUTS
            ess={"g1": 500, "g2": 500},
            divergences=100,  # Would fail for NUTS
            divergence_frac=0.1,
            bfmi=[0.01],  # Would fail for NUTS
            n_samples=1000,
            n_chains=1,
        )
        thresholds = ConvergenceThresholds()
        # VI only checks ESS, so this should pass
        assert check_convergence(diag, thresholds, method="vi") is True

    def test_vi_check_convergence_fails_low_ess(self):
        """VI convergence fails when ESS is too low."""
        diag = ConvergenceDiagnostics(
            rhat={"g1": 1.0, "g2": 1.0},
            ess={"g1": 10, "g2": 500},
            divergences=0,
            divergence_frac=0.0,
            bfmi=[],
            n_samples=1000,
            n_chains=1,
        )
        thresholds = ConvergenceThresholds(ess_min=100)
        assert check_convergence(diag, thresholds, method="vi") is False

    def test_nuts_default_method(self):
        """Without inference_method attr, defaults to NUTS behavior."""
        idata = _make_mock_idata(n_chains=2, n_samples=500)
        diag = extract_convergence_diagnostics(idata)
        # Should compute full MCMC diagnostics
        assert diag.rhat["g1"] < 1.1
        assert diag.ess["g1"] > 0
        assert isinstance(diag.bfmi, list)
        assert len(diag.bfmi) == 2

    def test_extract_shear_estimates_map(self):
        """extract_shear_estimates works for MAP (1 chain, 1 draw)."""
        idata = _make_mock_idata(
            g1_mean=0.01, n_chains=1, n_samples=1, inference_method="map"
        )
        est = extract_shear_estimates(idata, "g1")
        assert isinstance(est, ShearEstimates)
        # For a single sample, mean == median == the value
        assert est.mean == est.median
        assert est.std == 0.0

    def test_extract_shear_estimates_vi(self):
        """extract_shear_estimates works for VI (1 chain, N draws)."""
        idata = _make_mock_idata(
            g1_mean=0.02, n_chains=1, n_samples=1000, inference_method="vi"
        )
        est = extract_shear_estimates(idata, "g1")
        assert isinstance(est, ShearEstimates)
        assert est.mean == pytest.approx(0.02, abs=0.005)

    def test_extract_realization_map(self):
        """extract_realization works end-to-end for MAP."""
        idata = _make_mock_idata(
            g1_mean=0.01, g2_mean=0.0,
            n_chains=1, n_samples=1, inference_method="map"
        )
        thresholds = ConvergenceThresholds()
        result = extract_realization(
            idata, g1_true=0.01, g2_true=0.0,
            run_id="map_test", seed=0, thresholds=thresholds,
        )
        assert result.passed_convergence is True
