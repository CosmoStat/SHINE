"""Diagnostic plots for bias measurement using matplotlib and ArviZ."""

import logging
from pathlib import Path
from typing import List, Optional

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _is_delta_posterior(data_range: float, center: float) -> bool:
    """Check if a posterior has negligible range (effectively a delta function).

    Args:
        data_range: Peak-to-peak range of the samples.
        center: Mean of the samples.

    Returns:
        True if the posterior should be treated as a delta function.
    """
    if data_range == 0 or not np.isfinite(data_range):
        return True
    if data_range < abs(center) * 1e-8:
        return True
    return data_range < 1e-7


def _plot_delta_bar(ax: plt.Axes, center: float, color: str, edgecolor: str, alpha: float) -> None:
    """Plot a single bar for a delta-function posterior."""
    width = max(abs(center) * 1e-4, 1e-10)
    ax.bar(center, 1.0, width=width, alpha=alpha, color=color, edgecolor=edgecolor)
    ax.set_ylabel("(delta)")


def _safe_hist(ax: plt.Axes, samples: np.ndarray, color: str, edgecolor: str, alpha: float = 0.7) -> None:
    """Plot histogram that handles near-zero-range (delta) posteriors.

    For posteriors with negligible range, draws a single bar at the mean
    instead of attempting a histogram that would fail or be misleading.

    Args:
        ax: Matplotlib axes to plot on.
        samples: Posterior samples to histogram.
        color: Fill color for the histogram.
        edgecolor: Edge color for histogram bars.
        alpha: Transparency level.
    """
    data_range = float(np.ptp(samples))
    center = float(np.mean(samples))

    if _is_delta_posterior(data_range, center):
        _plot_delta_bar(ax, center, color, edgecolor, alpha)
        return

    n_bins = max(1, min(50, int(len(samples) ** 0.5)))
    try:
        ax.hist(samples, bins=n_bins, density=True,
                alpha=alpha, color=color, edgecolor=edgecolor)
    except (ValueError, FloatingPointError):
        _plot_delta_bar(ax, center, color, edgecolor, alpha)


def plot_level0_diagnostics(
    idata: az.InferenceData,
    g1_true: float,
    g2_true: float,
    output_dir: str,
) -> List[Path]:
    """Generate Level 0 diagnostic plots, dispatching on inference method.

    Reads ``inference_method`` from ``idata.posterior.attrs`` to select
    the appropriate plotting style:
    - NUTS: trace plots + histograms + pair plot
    - VI: histograms + pair plot (no trace since no chains)
    - MAP: bar at point estimate + truth line (2 panels: g1, g2)

    Args:
        idata: ArviZ InferenceData with posterior samples.
        g1_true: True g1 shear value.
        g2_true: True g2 shear value.
        output_dir: Directory to save plots.

    Returns:
        List of saved plot file paths.
    """
    method = idata.posterior.attrs.get("inference_method", "nuts")
    if method == "map":
        return _plot_map_diagnostics(idata, g1_true, g2_true, output_dir)
    elif method == "vi":
        return _plot_vi_diagnostics(idata, g1_true, g2_true, output_dir)
    else:
        return _plot_nuts_diagnostics(idata, g1_true, g2_true, output_dir)


def _plot_nuts_diagnostics(
    idata: az.InferenceData,
    g1_true: float,
    g2_true: float,
    output_dir: str,
) -> List[Path]:
    """Generate NUTS diagnostic plots: trace + histograms + pair plot."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved = []

    g1_chains = idata.posterior.g1.values  # (n_chains, n_samples)
    g2_chains = idata.posterior.g2.values
    g1_samples = g1_chains.flatten()
    g2_samples = g2_chains.flatten()

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


def _plot_vi_diagnostics(
    idata: az.InferenceData,
    g1_true: float,
    g2_true: float,
    output_dir: str,
) -> List[Path]:
    """Generate VI diagnostic plots: histograms + pair plot (no trace)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved = []

    g1_samples = idata.posterior.g1.values.flatten()
    g2_samples = idata.posterior.g2.values.flatten()

    # --- Marginal posteriors ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    _safe_hist(axes[0], g1_samples, "steelblue", "black")
    axes[0].axvline(g1_true, color="red", ls="--", lw=2, label="Truth")
    axes[0].axvline(g1_samples.mean(), color="green", lw=2, label="Mean")
    axes[0].set(
        xlabel="g1",
        title=f"VI: g1 = {g1_samples.mean():.4f} ± {g1_samples.std():.4f}",
    )
    axes[0].legend()

    _safe_hist(axes[1], g2_samples, "coral", "black")
    axes[1].axvline(g2_true, color="red", ls="--", lw=2, label="Truth")
    axes[1].axvline(g2_samples.mean(), color="green", lw=2, label="Mean")
    axes[1].set(
        xlabel="g2",
        title=f"VI: g2 = {g2_samples.mean():.4f} ± {g2_samples.std():.4f}",
    )
    axes[1].legend()

    fig.tight_layout()
    hist_path = output_path / "vi_posterior.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    saved.append(hist_path)
    logger.info(f"Saved VI posterior plot to {hist_path}")

    # --- Pair plot ---
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    ax2.scatter(g1_samples, g2_samples, alpha=0.1, s=2, color="steelblue")
    ax2.axvline(g1_true, color="red", ls="--", lw=1.5, label="g1 truth")
    ax2.axhline(g2_true, color="red", ls="--", lw=1.5, label="g2 truth")
    ax2.plot(g1_true, g2_true, "r*", ms=15, label="Truth")
    ax2.plot(g1_samples.mean(), g2_samples.mean(), "g*", ms=15, label="Mean")
    ax2.set(xlabel="g1", ylabel="g2", title="VI: g1 vs g2 posterior")
    ax2.legend()
    fig2.tight_layout()
    pair_path = output_path / "pair_plot.png"
    fig2.savefig(pair_path, dpi=150)
    plt.close(fig2)
    saved.append(pair_path)
    logger.info(f"Saved VI pair plot to {pair_path}")

    return saved


def _plot_map_diagnostics(
    idata: az.InferenceData,
    g1_true: float,
    g2_true: float,
    output_dir: str,
) -> List[Path]:
    """Generate MAP diagnostic plots: bar at point estimate + truth line."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved = []

    g1_val = float(idata.posterior.g1.values.flatten()[0])
    g2_val = float(idata.posterior.g2.values.flatten()[0])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # g1 MAP estimate
    bar_width = max(abs(g1_true) * 0.1, 1e-4)
    axes[0].bar(g1_val, 1.0, width=bar_width, alpha=0.7,
                color="steelblue", edgecolor="black", label="MAP")
    axes[0].axvline(g1_true, color="red", ls="--", lw=2, label="Truth")
    axes[0].set(xlabel="g1", title=f"MAP: g1 = {g1_val:.6f} (truth = {g1_true:.4f})")
    axes[0].set_ylabel("(point estimate)")
    axes[0].legend()

    # g2 MAP estimate
    bar_width = max(abs(g2_true) * 0.1, 1e-4)
    axes[1].bar(g2_val, 1.0, width=bar_width, alpha=0.7,
                color="coral", edgecolor="black", label="MAP")
    axes[1].axvline(g2_true, color="red", ls="--", lw=2, label="Truth")
    axes[1].set(xlabel="g2", title=f"MAP: g2 = {g2_val:.6f} (truth = {g2_true:.4f})")
    axes[1].set_ylabel("(point estimate)")
    axes[1].legend()

    fig.tight_layout()
    map_path = output_path / "map_estimate.png"
    fig.savefig(map_path, dpi=150)
    plt.close(fig)
    saved.append(map_path)
    logger.info(f"Saved MAP estimate plot to {map_path}")

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
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    ax.errorbar(
        g_true_values, g_est_means, yerr=g_est_stds,
        fmt="o", color="steelblue", ecolor="gray", capsize=3, label="Estimates",
    )

    xlim = ax.get_xlim()
    x_line = np.linspace(xlim[0], xlim[1], 100)
    ax.plot(x_line, x_line, ls="--", color="gray", lw=1.5, label="1:1")

    if m is not None and c is not None:
        y_reg = (1 + m) * x_line + c
        ax.plot(x_line, y_reg, ls="-", color="red", lw=2, label="Fit")
        ax.text(
            0.05, 0.95,
            f"m = {m:.4e}\nc = {c:.4e}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.set(
        xlabel=f"True shear {component}",
        ylabel=f"Estimated shear {component}",
        title=f"Bias: {component}",
    )
    ax.legend()
    fig.tight_layout()

    plot_path = output_path / f"bias_vs_shear_{component}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved bias vs shear plot to {plot_path}")

    return plot_path


def plot_coverage(
    alpha_levels: List[float],
    observed_coverage: List[float],
    output_dir: str,
    n_realizations: int = 1000,
) -> Path:
    """Plot observed vs expected coverage.

    Args:
        alpha_levels: Expected coverage levels.
        observed_coverage: Observed coverage fractions.
        output_dir: Directory to save plot.
        n_realizations: Number of realizations used (for confidence bands).

    Returns:
        Path to saved plot.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    alpha = np.asarray(alpha_levels)
    obs = np.asarray(observed_coverage)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.5, label="Perfect calibration")

    sigma = 3 * np.sqrt(alpha * (1 - alpha) / n_realizations)
    ax.fill_between(
        alpha, alpha - sigma, alpha + sigma,
        alpha=0.2, color="gray", label="3-sigma band",
    )

    ax.plot(alpha, obs, "o-", color="steelblue", lw=2, label="Observed")

    ax.set(
        xlabel="Expected coverage",
        ylabel="Observed coverage",
        title="Coverage calibration",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend()
    fig.tight_layout()

    plot_path = output_path / "coverage.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved coverage plot to {plot_path}")

    return plot_path


def plot_sbc_histogram(
    ranks: np.ndarray,
    param: str,
    output_dir: str,
    n_bins: int = 20,
) -> Path:
    """Plot SBC rank histogram.

    Args:
        ranks: Array of rank statistics.
        param: Parameter name.
        output_dir: Directory to save plot.
        n_bins: Number of histogram bins.

    Returns:
        Path to saved plot.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_realizations = len(ranks)
    expected = n_realizations / n_bins

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    ax.hist(ranks, bins=n_bins, color="steelblue", edgecolor="black", alpha=0.7)

    ax.axhline(expected, color="red", ls="--", lw=2, label="Expected")
    ax.fill_between(
        ax.get_xlim(),
        expected - 3 * np.sqrt(expected),
        expected + 3 * np.sqrt(expected),
        alpha=0.2, color="red", label="3-sigma band",
    )

    ax.set(
        xlabel="Rank",
        ylabel="Count",
        title=f"SBC rank histogram: {param}",
    )
    ax.legend()
    fig.tight_layout()

    plot_path = output_path / f"sbc_histogram_{param}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved SBC histogram to {plot_path}")

    return plot_path
