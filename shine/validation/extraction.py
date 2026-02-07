"""Extract structured results from ArviZ InferenceData."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import arviz as az
import numpy as np

from shine.validation.bias_config import ConvergenceThresholds

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceDiagnostics:
    """MCMC convergence diagnostic results.

    Attributes:
        rhat: R-hat statistic per parameter.
        ess: Effective sample size per parameter.
        divergences: Total number of divergent transitions.
        divergence_frac: Fraction of divergent transitions.
        bfmi: Bayesian Fraction of Missing Information per chain.
        n_samples: Total number of posterior samples.
        n_chains: Number of MCMC chains.
    """

    rhat: Dict[str, float]
    ess: Dict[str, float]
    divergences: int
    divergence_frac: float
    bfmi: List[float]
    n_samples: int
    n_chains: int


@dataclass
class ShearEstimates:
    """Summary statistics for a shear component posterior.

    Attributes:
        mean: Posterior mean.
        median: Posterior median.
        std: Posterior standard deviation.
        percentiles: Dictionary mapping percentile levels to values.
    """

    mean: float
    median: float
    std: float
    percentiles: Dict[float, float]


@dataclass
class RealizationResult:
    """Complete result for a single bias measurement realization.

    Attributes:
        run_id: Unique identifier for this realization.
        g1_true: True g1 shear value.
        g2_true: True g2 shear value.
        g1: Shear estimates for g1 component.
        g2: Shear estimates for g2 component.
        diagnostics: MCMC convergence diagnostics.
        passed_convergence: Whether convergence criteria were met.
        seed: Random seed used for this realization.
    """

    run_id: str
    g1_true: float
    g2_true: float
    g1: ShearEstimates
    g2: ShearEstimates
    diagnostics: ConvergenceDiagnostics
    passed_convergence: bool
    seed: int


def extract_convergence_diagnostics(
    idata: az.InferenceData,
    params: Optional[List[str]] = None,
) -> ConvergenceDiagnostics:
    """Extract convergence diagnostics from an InferenceData object.

    Args:
        idata: ArviZ InferenceData with posterior and sample_stats groups.
        params: Parameter names to compute diagnostics for (default: ["g1", "g2"]).

    Returns:
        ConvergenceDiagnostics with rhat, ess, divergences, bfmi.
    """
    if params is None:
        params = ["g1", "g2"]

    # R-hat
    rhat_data = az.rhat(idata, var_names=params)
    rhat = {p: float(rhat_data[p].values) for p in params}

    # ESS (bulk)
    ess_data = az.ess(idata, var_names=params)
    ess = {p: float(ess_data[p].values) for p in params}

    # Divergences
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        div_values = idata.sample_stats.diverging.values
        divergences = int(np.sum(div_values))
        total_samples = div_values.size
        divergence_frac = divergences / total_samples if total_samples > 0 else 0.0
    else:
        divergences = 0
        divergence_frac = 0.0
        total_samples = 0

    # BFMI
    try:
        bfmi_values = az.bfmi(idata)
        bfmi = [float(v) for v in bfmi_values]
    except Exception as exc:
        logger.debug(f"Could not compute BFMI: {exc}")
        bfmi = []

    # Chain/sample counts
    posterior = idata.posterior
    n_chains = posterior.sizes.get("chain", 1)
    n_samples_per_chain = posterior.sizes.get("draw", 0)
    n_samples = n_chains * n_samples_per_chain

    return ConvergenceDiagnostics(
        rhat=rhat,
        ess=ess,
        divergences=divergences,
        divergence_frac=divergence_frac,
        bfmi=bfmi,
        n_samples=n_samples,
        n_chains=n_chains,
    )


def extract_shear_estimates(
    idata: az.InferenceData,
    param: str,
) -> ShearEstimates:
    """Extract summary statistics for a shear parameter from posterior.

    Args:
        idata: ArviZ InferenceData with posterior group.
        param: Parameter name (e.g., "g1" or "g2").

    Returns:
        ShearEstimates with mean, median, std, and percentiles.
    """
    samples = idata.posterior[param].values.flatten()

    percentile_levels = [2.5, 16.0, 50.0, 84.0, 97.5]
    percentile_values = np.percentile(samples, percentile_levels)
    percentiles = {
        level: float(value)
        for level, value in zip(percentile_levels, percentile_values)
    }

    return ShearEstimates(
        mean=float(np.mean(samples)),
        median=float(np.median(samples)),
        std=float(np.std(samples)),
        percentiles=percentiles,
    )


def check_convergence(
    diagnostics: ConvergenceDiagnostics,
    thresholds: ConvergenceThresholds,
) -> bool:
    """Check if MCMC convergence diagnostics meet thresholds.

    Args:
        diagnostics: Computed convergence diagnostics.
        thresholds: Threshold criteria to check against.

    Returns:
        True if all diagnostics pass, False otherwise.
    """
    # Check R-hat (NaN/inf from degenerate posteriors are treated as failures)
    for param, rhat_val in diagnostics.rhat.items():
        if np.isnan(rhat_val) or np.isinf(rhat_val):
            logger.warning(
                f"R-hat for {param} = {rhat_val} (degenerate posterior)"
            )
            return False
        if rhat_val > thresholds.rhat_max:
            logger.warning(
                f"R-hat for {param} = {rhat_val:.4f} exceeds "
                f"threshold {thresholds.rhat_max}"
            )
            return False

    # Check ESS
    for param, ess_val in diagnostics.ess.items():
        if ess_val < thresholds.ess_min:
            logger.warning(
                f"ESS for {param} = {ess_val:.0f} below "
                f"threshold {thresholds.ess_min}"
            )
            return False

    # Check divergences
    if diagnostics.divergence_frac > thresholds.divergence_frac_max:
        logger.warning(
            f"Divergence fraction = {diagnostics.divergence_frac:.4f} exceeds "
            f"threshold {thresholds.divergence_frac_max}"
        )
        return False

    # Check BFMI
    for i, bfmi_val in enumerate(diagnostics.bfmi):
        if bfmi_val < thresholds.bfmi_min:
            logger.warning(
                f"BFMI for chain {i} = {bfmi_val:.4f} below "
                f"threshold {thresholds.bfmi_min}"
            )
            return False

    return True


def extract_realization(
    idata: az.InferenceData,
    g1_true: float,
    g2_true: float,
    run_id: str,
    seed: int,
    thresholds: ConvergenceThresholds,
) -> RealizationResult:
    """Extract a complete realization result from InferenceData.

    This is the main entry point for Stage 2 (extraction).

    Args:
        idata: ArviZ InferenceData with posterior and sample_stats groups.
        g1_true: True g1 shear value.
        g2_true: True g2 shear value.
        run_id: Unique identifier for this realization.
        seed: Random seed used for this realization.
        thresholds: Convergence diagnostic thresholds.

    Returns:
        RealizationResult with estimates, diagnostics, and pass/fail status.
    """
    diagnostics = extract_convergence_diagnostics(idata)
    g1_est = extract_shear_estimates(idata, "g1")
    g2_est = extract_shear_estimates(idata, "g2")
    passed = check_convergence(diagnostics, thresholds)

    return RealizationResult(
        run_id=run_id,
        g1_true=g1_true,
        g2_true=g2_true,
        g1=g1_est,
        g2=g2_est,
        diagnostics=diagnostics,
        passed_convergence=passed,
        seed=seed,
    )


def split_batched_idata(
    idata: az.InferenceData,
    n_batch: int,
    run_ids: List[str],
) -> List[Tuple[str, az.InferenceData]]:
    """Split a batched InferenceData into per-realization InferenceData objects.

    The batched posterior has variables with shape (n_chains, n_samples, n_batch).
    For each batch index i, slice [:, :, i] and create a new InferenceData with
    the standard (n_chains, n_samples) shape.

    Args:
        idata: ArviZ InferenceData from a batched MCMC run.
        n_batch: Number of batch elements.
        run_ids: List of run identifiers, one per batch element.

    Returns:
        List of (run_id, InferenceData) tuples, one per batch element.

    Raises:
        ValueError: If run_ids length doesn't match n_batch.
    """
    if len(run_ids) != n_batch:
        raise ValueError(
            f"run_ids length ({len(run_ids)}) must match n_batch ({n_batch})"
        )

    results = []
    posterior = idata.posterior

    # Identify batched variables (those with a "batch" dimension)
    batched_vars = [
        name for name in posterior.data_vars
        if "batch" in posterior[name].dims
    ]
    scalar_vars = [
        name for name in posterior.data_vars
        if "batch" not in posterior[name].dims
    ]

    for i in range(n_batch):
        # Build per-realization posterior dict
        post_dict = {}
        for name in batched_vars:
            # Shape: (chain, draw, batch) → (chain, draw)
            values = posterior[name].values
            post_dict[name] = values[:, :, i]

        for name in scalar_vars:
            post_dict[name] = posterior[name].values

        # Preserve sample_stats if available
        stats_dict = None
        if hasattr(idata, "sample_stats"):
            stats_dict = {}
            for name in idata.sample_stats.data_vars:
                stats_dict[name] = idata.sample_stats[name].values

        single_idata = az.from_dict(
            posterior=post_dict,
            sample_stats=stats_dict,
        )
        results.append((run_ids[i], single_idata))

    logger.info(f"Split batched InferenceData into {n_batch} per-realization objects")
    return results
