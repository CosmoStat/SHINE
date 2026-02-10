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
    component: str = "g1",
) -> BiasResult:
    """Compute bias via weighted linear regression over multiple shear values.

    Fits the model: g_est = (1 + m) * g_true + c
    using weighted least squares.

    Args:
        g_true_values: Array of true shear values.
        g_est_means: Array of estimated (posterior mean) shear values.
        weights: Optional inverse-variance weights for regression.
        component: Shear component name ("g1" or "g2").

    Returns:
        BiasResult with regression-fitted m and c.

    Raises:
        ValueError: If fewer than 2 data points are provided.
    """
    g_true_values = np.asarray(g_true_values, dtype=np.float64)
    g_est_means = np.asarray(g_est_means, dtype=np.float64)
    n = len(g_true_values)

    if n < 2:
        raise ValueError(
            f"Need at least 2 data points for regression, got {n}"
        )

    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)

    # Weighted least squares: fit y = slope * x + intercept
    # where slope = 1 + m, intercept = c
    W = np.sum(w)
    Wx = np.sum(w * g_true_values)
    Wy = np.sum(w * g_est_means)
    Wxx = np.sum(w * g_true_values**2)
    Wxy = np.sum(w * g_true_values * g_est_means)

    denom = W * Wxx - Wx**2
    if abs(denom) < 1e-30:
        raise ValueError("Degenerate regression: all g_true values are identical")

    slope = (W * Wxy - Wx * Wy) / denom
    intercept = (Wxx * Wy - Wx * Wxy) / denom

    m = slope - 1.0
    c = intercept

    # Weighted residual variance for error estimation
    residuals = g_est_means - (slope * g_true_values + intercept)
    s2 = np.sum(w * residuals**2) / max(W - 2.0, 1.0)

    slope_var = W * s2 / denom
    intercept_var = Wxx * s2 / denom

    m_err = float(np.sqrt(max(slope_var, 0.0)))
    c_err = float(np.sqrt(max(intercept_var, 0.0)))

    return BiasResult(
        m=float(m),
        m_err=m_err,
        c=float(c),
        c_err=c_err,
        method="regression",
        component=component,
    )


def compute_paired_response(
    g_est_plus: NDArray[np.float64],
    g_est_minus: NDArray[np.float64],
    g_true: float,
    component: str = "g1",
) -> NDArray[np.float64]:
    """Compute shear response from paired +g/-g observations.

    R_i = (g_est(+g) - g_est(-g)) / (2 * g_true)

    Args:
        g_est_plus: Shear estimates from +g realizations.
        g_est_minus: Shear estimates from -g realizations.
        g_true: Absolute value of true shear.
        component: Shear component name ("g1" or "g2").

    Returns:
        Array of response values R_i.

    Raises:
        ValueError: If g_true is zero or arrays have different lengths.
    """
    g_est_plus = np.asarray(g_est_plus, dtype=np.float64)
    g_est_minus = np.asarray(g_est_minus, dtype=np.float64)

    if len(g_est_plus) != len(g_est_minus):
        raise ValueError(
            f"Plus and minus arrays must have same length, "
            f"got {len(g_est_plus)} and {len(g_est_minus)}"
        )

    if g_true == 0.0:
        raise ValueError("Cannot compute response for g_true=0")

    return (g_est_plus - g_est_minus) / (2.0 * g_true)


def jackknife_bias(
    g_true_values: NDArray[np.float64],
    g_est_means: NDArray[np.float64],
    weights: Optional[NDArray[np.float64]] = None,
    n_groups: int = 10,
    component: str = "g1",
) -> BiasResult:
    """Compute bias with delete-one-group jackknife error estimation.

    Divides data into n_groups groups, recomputes the bias regression
    leaving each group out, and estimates errors from the variance of
    the jackknife replicates.

    Args:
        g_true_values: Array of true shear values.
        g_est_means: Array of estimated shear means.
        weights: Optional inverse-variance weights.
        n_groups: Number of jackknife groups.
        component: Shear component name ("g1" or "g2").

    Returns:
        BiasResult with jackknife uncertainty estimates.

    Raises:
        ValueError: If fewer data points than groups.
    """
    g_true_values = np.asarray(g_true_values, dtype=np.float64)
    g_est_means = np.asarray(g_est_means, dtype=np.float64)
    n = len(g_true_values)

    if n < n_groups:
        raise ValueError(
            f"Need at least n_groups={n_groups} data points, got {n}"
        )

    # Full-sample estimate
    full_result = compute_bias_regression(
        g_true_values, g_est_means, weights, component=component
    )

    # Create group indices
    indices = np.arange(n)
    groups = np.array_split(indices, n_groups)

    m_jk = []
    c_jk = []
    for group in groups:
        mask = np.ones(n, dtype=bool)
        mask[group] = False

        w_sub = weights[mask] if weights is not None else None
        try:
            result_i = compute_bias_regression(
                g_true_values[mask], g_est_means[mask], w_sub, component=component
            )
            m_jk.append(result_i.m)
            c_jk.append(result_i.c)
        except ValueError:
            continue

    if len(m_jk) < 2:
        return full_result

    m_jk = np.array(m_jk)
    c_jk = np.array(c_jk)
    n_jk = len(m_jk)

    # Jackknife variance: (n-1)/n * sum((theta_i - theta_bar)^2)
    m_err = float(np.sqrt((n_jk - 1) / n_jk * np.sum((m_jk - m_jk.mean()) ** 2)))
    c_err = float(np.sqrt((n_jk - 1) / n_jk * np.sum((c_jk - c_jk.mean()) ** 2)))

    return BiasResult(
        m=full_result.m,
        m_err=m_err,
        c=full_result.c,
        c_err=c_err,
        method="jackknife",
        component=component,
    )


def compute_coverage(
    g_true_values: NDArray[np.float64],
    g_est_means: NDArray[np.float64],
    g_est_stds: NDArray[np.float64],
    alpha_levels: Optional[List[float]] = None,
) -> List[CoverageResult]:
    """Compute credible interval coverage.

    For each alpha level, checks what fraction of realizations have
    the true value within the symmetric credible interval
    [mean - z*std, mean + z*std], where z = norm.ppf((1+alpha)/2).

    Args:
        g_true_values: Array of true shear values.
        g_est_means: Array of posterior means.
        g_est_stds: Array of posterior standard deviations.
        alpha_levels: Coverage levels to test (default: [0.68, 0.95]).

    Returns:
        List of CoverageResult for each alpha level.
    """
    from scipy.stats import norm

    g_true_values = np.asarray(g_true_values, dtype=np.float64)
    g_est_means = np.asarray(g_est_means, dtype=np.float64)
    g_est_stds = np.asarray(g_est_stds, dtype=np.float64)

    if alpha_levels is None:
        alpha_levels = [0.68, 0.95]

    n_total = len(g_true_values)
    results = []

    for alpha in alpha_levels:
        z = norm.ppf((1.0 + alpha) / 2.0)
        lower = g_est_means - z * g_est_stds
        upper = g_est_means + z * g_est_stds
        covered = (g_true_values >= lower) & (g_true_values <= upper)
        n_covered = int(np.sum(covered))
        observed = n_covered / n_total if n_total > 0 else 0.0

        results.append(CoverageResult(
            alpha=alpha,
            observed=observed,
            n_total=n_total,
            n_covered=n_covered,
        ))

    return results


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
        g_true_values: Array of true parameter values, shape (n_realizations,).
        posterior_samples: Array of posterior samples, shape (n_realizations, n_samples).
        param: Parameter name.

    Returns:
        SBCResult with rank statistics and KS test p-value.
    """
    from scipy.stats import kstest

    g_true_values = np.asarray(g_true_values, dtype=np.float64)
    posterior_samples = np.asarray(posterior_samples, dtype=np.float64)

    n_realizations, n_samples = posterior_samples.shape

    # Compute ranks: number of posterior samples less than the true value
    ranks = np.sum(posterior_samples < g_true_values[:, np.newaxis], axis=1)

    # Normalize ranks to [0, 1] for KS test against uniform
    normalized_ranks = ranks / (n_samples + 1)
    ks_stat, ks_pvalue = kstest(normalized_ranks, "uniform")

    return SBCResult(
        ranks=ranks.astype(np.int64),
        ks_pvalue=float(ks_pvalue),
        param=param,
    )
