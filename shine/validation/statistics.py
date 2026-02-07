"""Statistical analysis for bias measurement."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class BiasResult:
    """Result of a bias measurement.

    Attributes:
        m: Multiplicative bias.
        m_err: Uncertainty on m.
        c: Additive bias.
        c_err: Uncertainty on c.
        method: Method used to compute bias.
        component: Shear component ("g1" or "g2").
    """

    m: float
    m_err: float
    c: float
    c_err: float
    method: str
    component: str


@dataclass
class CoverageResult:
    """Result of a coverage test.

    Attributes:
        alpha: Nominal coverage level (e.g., 0.68, 0.95).
        observed: Observed coverage fraction.
        n_total: Total number of realizations tested.
        n_covered: Number of realizations where truth is within credible interval.
    """

    alpha: float
    observed: float
    n_total: int
    n_covered: int


@dataclass
class SBCResult:
    """Result of Simulation-Based Calibration.

    Attributes:
        ranks: Array of rank statistics.
        ks_pvalue: p-value from KS test of rank uniformity.
        param: Parameter name.
    """

    ranks: NDArray[np.int64]
    ks_pvalue: float
    param: str


def compute_bias_single_point(
    g_true: float,
    g_est_mean: float,
    g_est_std: float,
    component: str,
) -> BiasResult:
    """Compute multiplicative and additive bias from a single shear point.

    For a single g_true value, the bias model is:
        g_est = (1 + m) * g_true + c

    For Level 0 (single point), this simplifies to:
        m = (g_est / g_true) - 1  (when g_true != 0)
        c = g_est - (1 + m) * g_true = 0 by construction for single-point

    Args:
        g_true: True shear value.
        g_est_mean: Estimated (posterior mean) shear value.
        g_est_std: Posterior standard deviation of shear estimate.
        component: Shear component name ("g1" or "g2").

    Returns:
        BiasResult with multiplicative and additive bias.

    Raises:
        ValueError: If g_true is zero (cannot compute multiplicative bias).
    """
    if g_true == 0.0:
        raise ValueError(
            "Cannot compute multiplicative bias for g_true=0. "
            "Use compute_bias_regression() with multiple shear values."
        )

    m = (g_est_mean / g_true) - 1.0
    m_err = g_est_std / abs(g_true)

    # For single-point, c = g_est - (1+m)*g_true = 0 by construction
    c = 0.0
    c_err = g_est_std

    return BiasResult(
        m=m,
        m_err=m_err,
        c=c,
        c_err=c_err,
        method="single_point",
        component=component,
    )


def compute_bias_regression(
    g_true_values: NDArray[np.float64],
    g_est_means: NDArray[np.float64],
    weights: Optional[NDArray[np.float64]] = None,
) -> BiasResult:
    """Compute bias via weighted linear regression over multiple shear values.

    Fits the model: g_est = (1 + m) * g_true + c

    Args:
        g_true_values: Array of true shear values.
        g_est_means: Array of estimated (posterior mean) shear values.
        weights: Optional inverse-variance weights for regression.

    Returns:
        BiasResult with regression-fitted m and c.

    Raises:
        NotImplementedError: This function is a stub for Level 1+.
    """
    raise NotImplementedError(
        "compute_bias_regression() is planned for Level 1+. "
        "Use compute_bias_single_point() for Level 0."
    )


def compute_paired_response(
    g_est_plus: NDArray[np.float64],
    g_est_minus: NDArray[np.float64],
    g_true: float,
) -> NDArray[np.float64]:
    """Compute shear response from paired +g/-g observations.

    R_i = (g_est(+g) - g_est(-g)) / (2 * g_true)

    Args:
        g_est_plus: Shear estimates from +g realizations.
        g_est_minus: Shear estimates from -g realizations.
        g_true: Absolute value of true shear.

    Returns:
        Array of response values R_i.

    Raises:
        NotImplementedError: This function is a stub for Level 1+.
    """
    raise NotImplementedError(
        "compute_paired_response() is planned for Level 1+."
    )


def jackknife_bias(
    g_true_values: NDArray[np.float64],
    g_est_means: NDArray[np.float64],
    weights: Optional[NDArray[np.float64]] = None,
    n_groups: int = 10,
) -> BiasResult:
    """Compute bias with delete-one jackknife error estimation.

    Args:
        g_true_values: Array of true shear values.
        g_est_means: Array of estimated shear means.
        weights: Optional inverse-variance weights.
        n_groups: Number of jackknife groups.

    Returns:
        BiasResult with jackknife uncertainty estimates.

    Raises:
        NotImplementedError: This function is a stub for Level 1+.
    """
    raise NotImplementedError(
        "jackknife_bias() is planned for Level 1+."
    )


def compute_coverage(
    g_true_values: NDArray[np.float64],
    g_est_means: NDArray[np.float64],
    g_est_stds: NDArray[np.float64],
    alpha_levels: Optional[List[float]] = None,
) -> List[CoverageResult]:
    """Compute credible interval coverage.

    Args:
        g_true_values: Array of true shear values.
        g_est_means: Array of posterior means.
        g_est_stds: Array of posterior standard deviations.
        alpha_levels: Coverage levels to test (default: [0.68, 0.95]).

    Returns:
        List of CoverageResult for each alpha level.

    Raises:
        NotImplementedError: This function is a stub for Level 1+.
    """
    raise NotImplementedError(
        "compute_coverage() is planned for Level 1+."
    )


def compute_sbc_ranks(
    g_true_values: NDArray[np.float64],
    posterior_samples: NDArray[np.float64],
    param: str,
) -> SBCResult:
    """Compute Simulation-Based Calibration rank statistics.

    For each realization, the rank is the number of posterior samples
    less than the true value. Under correct calibration, ranks should
    be uniformly distributed.

    Args:
        g_true_values: Array of true parameter values.
        posterior_samples: Array of posterior samples (n_realizations, n_samples).
        param: Parameter name.

    Returns:
        SBCResult with rank histogram and KS test p-value.

    Raises:
        NotImplementedError: This function is a stub for Level 1+.
    """
    raise NotImplementedError(
        "compute_sbc_ranks() is planned for Level 1+."
    )
