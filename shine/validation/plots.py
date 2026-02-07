"""Diagnostic plots for bias measurement using matplotlib and ArviZ."""

import logging
from pathlib import Path
from typing import List, Optional

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_level0_diagnostics(
    idata: az.InferenceData,
    g1_true: float,
    g2_true: float,
    output_dir: str,
) -> List[Path]:
    """Generate Level 0 diagnostic plots.

    Produces:
    - Trace plots for g1 and g2
    - Marginal posterior histograms with truth lines
    - Pair plot (g1 vs g2)

    Args:
        idata: ArviZ InferenceData with posterior samples.
        g1_true: True g1 shear value.
        g2_true: True g2 shear value.
        output_dir: Directory to save plots.

    Returns:
        List of saved plot file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved = []

    g1_chains = idata.posterior.g1.values  # (n_chains, n_samples)
    g2_chains = idata.posterior.g2.values
    g1_samples = g1_chains.flatten()
    g2_samples = g2_chains.flatten()

    def _safe_hist(ax, samples, color, edgecolor, alpha=0.7):
        """Plot histogram that handles near-zero-range (delta) posteriors."""
        data_range = float(np.ptp(samples))
        center = float(np.mean(samples))
        # If range is negligible relative to center value, treat as delta
        if (
            data_range == 0
            or not np.isfinite(data_range)
            or data_range < abs(center) * 1e-8
            or data_range < 1e-7
        ):
            width = max(abs(center) * 1e-4, 1e-10)
            ax.bar(center, 1.0, width=width,
                   alpha=alpha, color=color, edgecolor=edgecolor)
            ax.set_ylabel("(delta)")
            return
        n_bins = max(1, min(50, int(len(samples) ** 0.5)))
        try:
            ax.hist(samples, bins=n_bins, density=True,
                    alpha=alpha, color=color, edgecolor=edgecolor)
        except (ValueError, FloatingPointError):
            # Last resort: just show a bar at the mean
            width = max(abs(center) * 1e-4, 1e-10)
            ax.bar(center, 1.0, width=width,
                   alpha=alpha, color=color, edgecolor=edgecolor)

    # --- Trace plots + marginal posteriors ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # g1 trace
    for i, chain in enumerate(g1_chains):
        axes[0, 0].plot(chain, alpha=0.7, label=f"Chain {i + 1}")
    axes[0, 0].axhline(g1_true, color="red", ls="--", lw=2, label="Truth")
    axes[0, 0].set(xlabel="Sample", ylabel="g1", title="Trace: g1")
    axes[0, 0].legend()

    # g2 trace
    for i, chain in enumerate(g2_chains):
        axes[0, 1].plot(chain, alpha=0.7, label=f"Chain {i + 1}")
    axes[0, 1].axhline(g2_true, color="red", ls="--", lw=2, label="Truth")
    axes[0, 1].set(xlabel="Sample", ylabel="g2", title="Trace: g2")
    axes[0, 1].legend()

    # g1 posterior
    _safe_hist(axes[1, 0], g1_samples, "steelblue", "black")
    axes[1, 0].axvline(g1_true, color="red", ls="--", lw=2, label="Truth")
    axes[1, 0].axvline(g1_samples.mean(), color="green", lw=2, label="Mean")
    axes[1, 0].set(
        xlabel="g1",
        title=f"g1 = {g1_samples.mean():.4f} ± {g1_samples.std():.4f}",
    )
    axes[1, 0].legend()

    # g2 posterior
    _safe_hist(axes[1, 1], g2_samples, "coral", "black")
    axes[1, 1].axvline(g2_true, color="red", ls="--", lw=2, label="Truth")
    axes[1, 1].axvline(g2_samples.mean(), color="green", lw=2, label="Mean")
    axes[1, 1].set(
        xlabel="g2",
        title=f"g2 = {g2_samples.mean():.4f} ± {g2_samples.std():.4f}",
    )
    axes[1, 1].legend()

    fig.tight_layout()
    trace_path = output_path / "trace_posterior.png"
    fig.savefig(trace_path, dpi=150)
    plt.close(fig)
    saved.append(trace_path)
    logger.info(f"Saved trace/posterior plot to {trace_path}")

    # --- Pair plot (g1 vs g2) ---
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    ax2.scatter(g1_samples, g2_samples, alpha=0.1, s=2, color="steelblue")
    ax2.axvline(g1_true, color="red", ls="--", lw=1.5, label="g1 truth")
    ax2.axhline(g2_true, color="red", ls="--", lw=1.5, label="g2 truth")
    ax2.plot(g1_true, g2_true, "r*", ms=15, label="Truth")
    ax2.plot(g1_samples.mean(), g2_samples.mean(), "g*", ms=15, label="Mean")
    ax2.set(xlabel="g1", ylabel="g2", title="g1 vs g2 posterior")
    ax2.legend()
    fig2.tight_layout()
    pair_path = output_path / "pair_plot.png"
    fig2.savefig(pair_path, dpi=150)
    plt.close(fig2)
    saved.append(pair_path)
    logger.info(f"Saved pair plot to {pair_path}")

    return saved


def plot_bias_vs_shear(
    g_true_values: np.ndarray,
    g_est_means: np.ndarray,
    g_est_stds: np.ndarray,
    component: str,
    output_dir: str,
    m: Optional[float] = None,
    c: Optional[float] = None,
) -> Path:
    """Plot estimated shear vs true shear with bias regression line.

    Args:
        g_true_values: Array of true shear values.
        g_est_means: Array of estimated shear means.
        g_est_stds: Array of estimated shear standard deviations.
        component: Shear component name ("g1" or "g2").
        output_dir: Directory to save plot.
        m: Multiplicative bias (for regression line).
        c: Additive bias (for regression line).

    Returns:
        Path to saved plot.

    Raises:
        NotImplementedError: This function is a stub for Level 1+.
    """
    raise NotImplementedError(
        "plot_bias_vs_shear() is planned for Level 1+."
    )


def plot_coverage(
    alpha_levels: List[float],
    observed_coverage: List[float],
    output_dir: str,
) -> Path:
    """Plot observed vs expected coverage.

    Args:
        alpha_levels: Expected coverage levels.
        observed_coverage: Observed coverage fractions.
        output_dir: Directory to save plot.

    Returns:
        Path to saved plot.

    Raises:
        NotImplementedError: This function is a stub for Level 1+.
    """
    raise NotImplementedError(
        "plot_coverage() is planned for Level 1+."
    )


def plot_sbc_histogram(
    ranks: np.ndarray,
    param: str,
    output_dir: str,
) -> Path:
    """Plot SBC rank histogram.

    Args:
        ranks: Array of rank statistics.
        param: Parameter name.
        output_dir: Directory to save plot.

    Returns:
        Path to saved plot.

    Raises:
        NotImplementedError: This function is a stub for Level 1+.
    """
    raise NotImplementedError(
        "plot_sbc_histogram() is planned for Level 1+."
    )
