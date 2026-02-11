"""Visualization utilities for Euclid VIS shear inference.

Provides diagnostic plots comparing observed data, forward-model images,
and normalized residuals for multi-exposure scene models.
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_exposure_comparison(
    observed: np.ndarray,
    model: np.ndarray,
    noise_sigma: np.ndarray,
    mask: np.ndarray,
    exposure_idx: int = 0,
    residual_mask: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Plot observed, model, and normalized residual for one exposure.

    Creates a 3-panel figure showing:

    1. **Observed** — background-subtracted science image, with flagged
       pixels overlaid in grey.
    2. **Model** — rendered forward-model image (no mask overlay).
    3. **Residual** — pixel-wise ``(observed - model) / sigma``,
       i.e. chi-values, with ``residual_mask`` overlay.

    Args:
        observed: Observed image for this exposure, shape ``(ny, nx)``.
        model: Model image for this exposure, shape ``(ny, nx)``.
        noise_sigma: Per-pixel noise sigma, shape ``(ny, nx)``.
            Flagged pixels typically have value ``1e10``.
        mask: Boolean validity mask, shape ``(ny, nx)``.
            ``True`` = valid pixel, ``False`` = flagged.
        exposure_idx: Exposure index, used only for the figure title.
        residual_mask: Optional broader boolean mask for the residual
            panel, shape ``(ny, nx)``.  ``True`` = include in residual,
            ``False`` = mask out.  Useful for excluding bright-star
            halos and extended-object regions that are valid pixels
            but should not contribute to chi-squared.  If ``None``,
            falls back to ``mask``.
        output_path: If provided, save the figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    observed = np.asarray(observed)
    model = np.asarray(model)
    noise_sigma = np.asarray(noise_sigma)
    mask = np.asarray(mask, dtype=bool)

    if residual_mask is not None:
        residual_mask = np.asarray(residual_mask, dtype=bool)
    else:
        residual_mask = mask

    # Normalized residual (chi-values); mask out flagged pixels
    residual = np.where(residual_mask, (observed - model) / noise_sigma, np.nan)

    # Chi^2 per valid pixel (using the residual mask)
    n_valid = residual_mask.sum()
    chi2_per_pix = (
        np.nansum(residual[residual_mask] ** 2) / n_valid
        if n_valid > 0
        else np.nan
    )

    # Shared colour scale for observed and model (percentile-clipped)
    combined = np.concatenate(
        [observed[mask].ravel(), model[mask].ravel()]
    )
    if combined.size > 0:
        vmin = np.percentile(combined, 1)
        vmax = np.percentile(combined, 99)
    else:
        vmin, vmax = -1.0, 1.0

    # Residual symmetric colour scale
    rmax = (
        np.nanpercentile(np.abs(residual[residual_mask]), 99)
        if n_valid > 0
        else 5.0
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Observed panel (with pixel mask overlay) ---
    im0 = axes[0].imshow(
        observed, origin="lower", cmap="viridis",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    if (~mask).any():
        axes[0].contourf(
            ~mask, levels=[0.5, 1.5], colors="grey", alpha=0.4,
        )
    axes[0].set_title("Observed", fontsize=12)
    axes[0].set_xlabel("x [px]")
    axes[0].set_ylabel("y [px]")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # --- Model panel (no mask overlay) ---
    im1 = axes[1].imshow(
        model, origin="lower", cmap="viridis",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    axes[1].set_title("SHINE Model", fontsize=12)
    axes[1].set_xlabel("x [px]")
    axes[1].set_ylabel("y [px]")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # --- Residual panel (with residual_mask overlay) ---
    im2 = axes[2].imshow(
        residual, origin="lower", cmap="RdBu_r",
        vmin=-rmax, vmax=rmax, interpolation="nearest",
    )
    if (~residual_mask).any():
        axes[2].contourf(
            ~residual_mask, levels=[0.5, 1.5], colors="grey", alpha=0.4,
        )
    axes[2].set_title("Residual (chi)", fontsize=12)
    axes[2].set_xlabel("x [px]")
    axes[2].set_ylabel("y [px]")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Exposure {exposure_idx}  |  "
        f"chi2/pix = {chi2_per_pix:.2f}  |  "
        f"N_valid = {n_valid}",
        fontsize=13,
    )
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
