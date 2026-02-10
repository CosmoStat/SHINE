"""Multi-exposure multi-object scene model for Euclid VIS inference.

Builds a NumPyro forward generative model that samples global shear and
per-galaxy parameters, renders each galaxy through per-exposure PSFs and
WCS, scatter-adds the stamps onto full-sized model images, and evaluates
a per-pixel Gaussian likelihood weighted by the RMS noise map.

Sources are grouped into stamp-size tiers (e.g. 64, 128, 256 px) based
on their catalog half-light radius.  Each tier is rendered with its own
``jax.vmap`` pass and FFT size, preserving full parallelism within each
tier while avoiding expensive FFT convolutions for small galaxies.
"""

import logging
import math
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax_galsim as galsim
import numpy as np
import numpyro
import numpyro.distributions as dist

from shine.euclid.config import EuclidInferenceConfig

logger = logging.getLogger(__name__)


def _fft_size_for_stamp(stamp_size: int) -> int:
    """Return the FFT grid size (next power of 2 >= 2 * stamp_size)."""
    return int(2 ** math.ceil(math.log2(2 * stamp_size)))


def _compute_tier_indices(
    source_stamp_tier: jnp.ndarray, n_tiers: int
) -> list[jnp.ndarray]:
    """Partition source indices by stamp tier.

    Args:
        source_stamp_tier: Per-source tier index, shape ``(n_sources,)``.
        n_tiers: Number of available tiers.

    Returns:
        List of index arrays, one per tier.
    """
    tier_np = np.asarray(source_stamp_tier)
    return [
        jnp.array(np.where(tier_np == t)[0], dtype=jnp.int32)
        for t in range(n_tiers)
    ]


def _render_tier(
    tier_idx: int,
    stamp_size: int,
    exp_idx: int,
    g1: jnp.ndarray,
    g2: jnp.ndarray,
    flux: jnp.ndarray,
    hlr: jnp.ndarray,
    e1: jnp.ndarray,
    e2: jnp.ndarray,
    dx: jnp.ndarray,
    dy: jnp.ndarray,
    data: "ExposureSet",
    pixel_scale: float,
    tier_indices: list[jnp.ndarray],
    model_image: jnp.ndarray,
) -> jnp.ndarray:
    """Render one stamp-size tier for one exposure and scatter-add.

    Gathers sources belonging to this tier, renders them via
    ``jax.vmap`` at the tier's stamp/FFT size, and scatter-adds the
    resulting stamps onto ``model_image``.

    Args:
        tier_idx: Index of this tier in the stamp-sizes list.
        stamp_size: Stamp side length in pixels for this tier.
        exp_idx: Exposure index.
        g1: Global shear component 1 (scalar).
        g2: Global shear component 2 (scalar).
        flux: Per-source flux in ADU, shape ``(n_sources,)``.
        hlr: Per-source half-light radius in arcsec, shape ``(n_sources,)``.
        e1: Per-source intrinsic ellipticity component 1, shape ``(n_sources,)``.
        e2: Per-source intrinsic ellipticity component 2, shape ``(n_sources,)``.
        dx: Per-source position offset in arcsec (x), shape ``(n_sources,)``.
        dy: Per-source position offset in arcsec (y), shape ``(n_sources,)``.
        data: Packed exposure data.
        pixel_scale: Pixel scale in arcsec/pixel.
        tier_indices: Pre-computed index arrays, one per tier.
        model_image: Accumulated model image to scatter-add onto.

    Returns:
        Updated model image with this tier's contributions added.
    """
    indices = tier_indices[tier_idx]
    if indices.shape[0] == 0:
        return model_image

    fft_size = _fft_size_for_stamp(stamp_size)
    gsparams = galsim.GSParams(
        maximum_fft_size=fft_size, minimum_fft_size=fft_size
    )

    # Gather per-source data for this tier and exposure
    flux_t = flux[indices]
    hlr_t = hlr[indices]
    e1_t = e1[indices]
    e2_t = e2[indices]
    dx_t = dx[indices]
    dy_t = dy[indices]
    pos_t = data.pixel_positions[indices, exp_idx, :]
    wcs_t = data.wcs_jacobians[indices, exp_idx, :]
    psf_t = data.psf_images[indices, exp_idx, :, :]
    vis_t = data.source_visible[indices, exp_idx]

    # Safe PSF fallback for invisible sources
    psf_shape = psf_t.shape[-2:]
    safe_psf = jnp.zeros(psf_shape)
    safe_psf = safe_psf.at[psf_shape[0] // 2, psf_shape[1] // 2].set(1.0)

    # Use default-argument capture to bind stamp_size and gsparams
    # at definition time (Python loop is unrolled by JIT tracer).
    def render_one_galaxy(
        flux_i, hlr_i, e1_i, e2_i, dx_i, dy_i,
        psf_img, wcs_params, pix_pos, visible_i,
        _ss=stamp_size, _gsp=gsparams,
    ):
        v = visible_i
        flux_i = jnp.where(v, flux_i, 1.0)
        hlr_i = jnp.where(v, hlr_i, 0.5)
        e1_i = jnp.where(v, e1_i, 0.0)
        e2_i = jnp.where(v, e2_i, 0.0)
        dx_i = jnp.where(v, dx_i, 0.0)
        dy_i = jnp.where(v, dy_i, 0.0)
        psf_img = jnp.where(v, psf_img, safe_psf)
        wcs_params = jnp.where(
            v, wcs_params, jnp.array([pixel_scale, 0.0, 0.0, pixel_scale])
        )

        gal = galsim.Exponential(
            flux=flux_i, half_light_radius=hlr_i, gsparams=_gsp
        )
        gal = gal.shear(e1=e1_i, e2=e2_i)
        gal = gal.shear(g1=g1, g2=g2)

        psf = galsim.InterpolatedImage(
            galsim.Image(psf_img, scale=pixel_scale), gsparams=_gsp
        )
        final = galsim.Convolve([gal, psf], gsparams=_gsp)

        wcs = galsim.JacobianWCS(
            dudx=wcs_params[0], dudy=wcs_params[1],
            dvdx=wcs_params[2], dvdy=wcs_params[3],
        )

        pix_dx = dx_i / pixel_scale
        pix_dy = dy_i / pixel_scale

        stamp = final.drawImage(
            nx=_ss, ny=_ss, wcs=wcs,
            offset=galsim.PositionD(pix_dx, pix_dy),
        ).array

        return stamp * visible_i

    # Vectorise rendering over tier sources
    all_stamps = jax.vmap(render_one_galaxy)(
        flux_t, hlr_t, e1_t, e2_t, dx_t, dy_t,
        psf_t, wcs_t, pos_t, vis_t,
    )

    # Scatter-add stamps onto the model image
    stamp_half = stamp_size // 2
    corner_x = jnp.round(pos_t[:, 0]).astype(jnp.int32) - stamp_half
    corner_y = jnp.round(pos_t[:, 1]).astype(jnp.int32) - stamp_half

    def scatter_add(image, inputs, _ss=stamp_size):
        stamp, iy, ix = inputs
        iy = jnp.clip(iy, 0, data.image_ny - _ss)
        ix = jnp.clip(ix, 0, data.image_nx - _ss)
        current = jax.lax.dynamic_slice(image, (iy, ix), (_ss, _ss))
        return jax.lax.dynamic_update_slice(
            image, current + stamp, (iy, ix)
        ), None

    model_image, _ = jax.lax.scan(
        scatter_add, model_image, (all_stamps, corner_y, corner_x)
    )
    return model_image


def _render_exposure_image(
    exp_idx: int,
    g1: jnp.ndarray,
    g2: jnp.ndarray,
    flux: jnp.ndarray,
    hlr: jnp.ndarray,
    e1: jnp.ndarray,
    e2: jnp.ndarray,
    dx: jnp.ndarray,
    dy: jnp.ndarray,
    data: "ExposureSet",
    pixel_scale: float,
    stamp_sizes: list[int],
    tier_indices: list[jnp.ndarray],
) -> jnp.ndarray:
    """Render all galaxies for one exposure into a model image.

    Iterates over stamp-size tiers, rendering each tier's sources via
    ``jax.vmap`` at the appropriate stamp/FFT size and scatter-adding
    them onto a shared model image.

    Args:
        exp_idx: Index of the current exposure.
        g1: Global shear component 1 (scalar).
        g2: Global shear component 2 (scalar).
        flux: Per-source flux in ADU, shape ``(n_sources,)``.
        hlr: Per-source half-light radius in arcsec, shape ``(n_sources,)``.
        e1: Per-source intrinsic ellipticity component 1, shape ``(n_sources,)``.
        e2: Per-source intrinsic ellipticity component 2, shape ``(n_sources,)``.
        dx: Per-source position offset in arcsec (x), shape ``(n_sources,)``.
        dy: Per-source position offset in arcsec (y), shape ``(n_sources,)``.
        data: Packed exposure data (images, PSFs, WCS, etc.).
        pixel_scale: Pixel scale in arcsec/pixel.
        stamp_sizes: List of stamp side lengths, one per tier.
        tier_indices: Pre-computed index arrays, one per tier.

    Returns:
        Model image array of shape ``(image_ny, image_nx)``.
    """
    model_image = jnp.zeros((data.image_ny, data.image_nx))

    for tier_idx, stamp_size in enumerate(stamp_sizes):
        model_image = _render_tier(
            tier_idx, stamp_size, exp_idx,
            g1, g2, flux, hlr, e1, e2, dx, dy,
            data, pixel_scale, tier_indices, model_image,
        )

    return model_image


def _render_exposure_likelihood(
    exp_idx: int,
    g1: jnp.ndarray,
    g2: jnp.ndarray,
    flux: jnp.ndarray,
    hlr: jnp.ndarray,
    e1: jnp.ndarray,
    e2: jnp.ndarray,
    dx: jnp.ndarray,
    dy: jnp.ndarray,
    data: "ExposureSet",
    pixel_scale: float,
    stamp_sizes: list[int],
    tier_indices: list[jnp.ndarray],
    observed_data: Optional[jnp.ndarray],
    extra_args: dict,
) -> None:
    """Render all galaxies for one exposure and evaluate likelihood.

    Delegates rendering to :func:`_render_exposure_image` and registers
    the per-pixel Gaussian likelihood as a NumPyro sample site.

    Args:
        exp_idx: Index of the current exposure.
        g1: Global shear component 1 (scalar).
        g2: Global shear component 2 (scalar).
        flux: Per-source flux in ADU, shape ``(n_sources,)``.
        hlr: Per-source half-light radius in arcsec, shape ``(n_sources,)``.
        e1: Per-source intrinsic ellipticity component 1, shape ``(n_sources,)``.
        e2: Per-source intrinsic ellipticity component 2, shape ``(n_sources,)``.
        dx: Per-source position offset in arcsec (x), shape ``(n_sources,)``.
        dy: Per-source position offset in arcsec (y), shape ``(n_sources,)``.
        data: Packed exposure data (images, PSFs, WCS, etc.).
        pixel_scale: Pixel scale in arcsec/pixel.
        stamp_sizes: List of stamp side lengths, one per tier.
        tier_indices: Pre-computed index arrays, one per tier.
        observed_data: Observed images, shape ``(n_exp, ny, nx)``, or None
            for prior predictive sampling.
        extra_args: Additional keyword arguments forwarded from the model
            call (unused, reserved for future extensions).
    """
    model_image = _render_exposure_image(
        exp_idx, g1, g2, flux, hlr, e1, e2, dx, dy,
        data, pixel_scale, stamp_sizes, tier_indices,
    )

    # Likelihood: per-pixel Gaussian weighted by the RMS noise map
    noise_sigma_j = data.noise_sigma[exp_idx]

    obs_j = observed_data[exp_idx] if observed_data is not None else None

    numpyro.sample(
        f"obs_{exp_idx}",
        dist.Normal(model_image, noise_sigma_j).to_event(2),
        obs=obs_j,
    )


class MultiExposureScene:
    """Builder for multi-exposure, multi-object NumPyro scene models.

    Constructs a forward generative model that:

    1. Samples global shear ``(g1, g2)`` and per-galaxy parameters
       (flux, half-light radius, ellipticity, position offset).
    2. Renders each galaxy on a small internal stamp via JAX-GalSim,
       convolved with the interpolated per-source PSF.
    3. Scatter-adds all stamps onto the full model image per exposure.
    4. Evaluates per-pixel Gaussian likelihood weighted by the RMS
       noise map.

    Sources are grouped into stamp-size tiers so that small galaxies
    use cheap FFT sizes while large galaxies get bigger stamps.

    Attributes:
        config: Euclid inference configuration.
        data: Packed multi-exposure data (images, PSFs, WCS, noise maps).
    """

    def __init__(
        self, config: EuclidInferenceConfig, exposure_set: "ExposureSet"
    ) -> None:
        """Initialize the multi-exposure scene builder.

        Args:
            config: Euclid inference configuration including priors,
                stamp sizes, and pixel scale.
            exposure_set: Pre-built data structure holding all exposure
                images, noise maps, PSF stamps, WCS Jacobians, and source
                catalog information.
        """
        self.config = config
        self.data = exposure_set

    def build_model(self) -> Callable:
        """Build the multi-exposure NumPyro model.

        Returns:
            A NumPyro model function with signature
            ``model(observed_data=None, **extra_args)`` suitable for passing
            to ``numpyro.infer.MCMC`` or ``numpyro.infer.SVI``.
        """
        data = self.data
        priors = self.config.priors
        stamp_sizes = self.config.galaxy_stamp_sizes
        pixel_scale = self.config.data.pixel_scale

        # Pre-compute tier indices (concrete arrays, captured by closure).
        tier_indices = _compute_tier_indices(
            data.source_stamp_tier, len(stamp_sizes)
        )

        tier_str = ", ".join(
            f"{stamp_sizes[t]}px: {tier_indices[t].shape[0]}"
            for t in range(len(stamp_sizes))
        )
        logger.info(
            "Building multi-exposure model: %d sources, %d exposures, "
            "tiers=[%s]",
            data.n_sources,
            data.n_exposures,
            tier_str,
        )

        def model(
            observed_data: Optional[jnp.ndarray] = None, **extra_args
        ) -> None:
            """NumPyro probabilistic model for multi-exposure shear inference.

            Args:
                observed_data: Observed images, shape ``(n_exp, ny, nx)``.
                    Pass ``None`` for prior predictive sampling.
                **extra_args: Reserved for future use.
            """
            # 1. Global shear
            g1 = numpyro.sample(
                "g1", dist.Normal(0.0, priors.shear_prior_sigma)
            )
            g2 = numpyro.sample(
                "g2", dist.Normal(0.0, priors.shear_prior_sigma)
            )

            # 2. Per-source parameters
            with numpyro.plate("sources", data.n_sources):
                flux = numpyro.sample(
                    "flux",
                    dist.LogNormal(
                        jnp.log(data.catalog_flux_adu),
                        priors.flux_prior_log_sigma,
                    ),
                )
                hlr = numpyro.sample(
                    "hlr",
                    dist.LogNormal(
                        jnp.log(data.catalog_hlr_arcsec),
                        priors.hlr_prior_log_sigma,
                    ),
                )
                e1 = numpyro.sample(
                    "e1", dist.Normal(0.0, priors.ellipticity_prior_sigma)
                )
                e2 = numpyro.sample(
                    "e2", dist.Normal(0.0, priors.ellipticity_prior_sigma)
                )
                dx = numpyro.sample(
                    "dx", dist.Normal(0.0, priors.position_prior_sigma)
                )
                dy = numpyro.sample(
                    "dy", dist.Normal(0.0, priors.position_prior_sigma)
                )

            # 3. Per-exposure rendering and likelihood
            for j in range(data.n_exposures):
                _render_exposure_likelihood(
                    j, g1, g2, flux, hlr, e1, e2, dx, dy,
                    data, pixel_scale, stamp_sizes, tier_indices,
                    observed_data, extra_args,
                )

        return model

    def build_single_exposure_model(
        self, exposure_idx: int = 0
    ) -> Callable:
        """Build a model for a single exposure (useful for debugging).

        Identical to :meth:`build_model` but only renders and evaluates
        the likelihood for one exposure, which makes JIT compilation and
        gradient evaluation much faster during development.

        Args:
            exposure_idx: Zero-based index of the exposure to model.

        Returns:
            A NumPyro model function with the same signature as
            :meth:`build_model`.

        Raises:
            IndexError: If ``exposure_idx`` is out of range.
        """
        if exposure_idx < 0 or exposure_idx >= self.data.n_exposures:
            raise IndexError(
                f"exposure_idx {exposure_idx} out of range for "
                f"{self.data.n_exposures} exposures"
            )

        data = self.data
        priors = self.config.priors
        stamp_sizes = self.config.galaxy_stamp_sizes
        pixel_scale = self.config.data.pixel_scale

        tier_indices = _compute_tier_indices(
            data.source_stamp_tier, len(stamp_sizes)
        )

        tier_str = ", ".join(
            f"{stamp_sizes[t]}px: {tier_indices[t].shape[0]}"
            for t in range(len(stamp_sizes))
        )
        logger.info(
            "Building single-exposure model: %d sources, exposure %d, "
            "tiers=[%s]",
            data.n_sources,
            exposure_idx,
            tier_str,
        )

        def model(
            observed_data: Optional[jnp.ndarray] = None, **extra_args
        ) -> None:
            """NumPyro model for single-exposure shear inference.

            Args:
                observed_data: Observed images, shape ``(n_exp, ny, nx)``.
                    Pass ``None`` for prior predictive sampling.
                **extra_args: Reserved for future use.
            """
            g1 = numpyro.sample(
                "g1", dist.Normal(0.0, priors.shear_prior_sigma)
            )
            g2 = numpyro.sample(
                "g2", dist.Normal(0.0, priors.shear_prior_sigma)
            )

            with numpyro.plate("sources", data.n_sources):
                flux = numpyro.sample(
                    "flux",
                    dist.LogNormal(
                        jnp.log(data.catalog_flux_adu),
                        priors.flux_prior_log_sigma,
                    ),
                )
                hlr = numpyro.sample(
                    "hlr",
                    dist.LogNormal(
                        jnp.log(data.catalog_hlr_arcsec),
                        priors.hlr_prior_log_sigma,
                    ),
                )
                e1 = numpyro.sample(
                    "e1", dist.Normal(0.0, priors.ellipticity_prior_sigma)
                )
                e2 = numpyro.sample(
                    "e2", dist.Normal(0.0, priors.ellipticity_prior_sigma)
                )
                dx = numpyro.sample(
                    "dx", dist.Normal(0.0, priors.position_prior_sigma)
                )
                dy = numpyro.sample(
                    "dy", dist.Normal(0.0, priors.position_prior_sigma)
                )

            _render_exposure_likelihood(
                exposure_idx, g1, g2, flux, hlr, e1, e2, dx, dy,
                data, pixel_scale, stamp_sizes, tier_indices,
                observed_data, extra_args,
            )

        return model


def render_model_images(
    params: dict,
    data: "ExposureSet",
    pixel_scale: float = 0.1,
    stamp_sizes: Optional[list[int]] = None,
) -> jnp.ndarray:
    """Render model images for all exposures from parameter values.

    Takes a dictionary of MAP (or sampled) parameters and produces the
    corresponding forward-model images for every exposure by reusing the
    same tiered vmap + scan rendering pipeline as the NumPyro model.

    Args:
        params: Parameter dictionary with keys ``"g1"``, ``"g2"``,
            ``"flux"``, ``"hlr"``, ``"e1"``, ``"e2"``, ``"dx"``, ``"dy"``.
            Scalars for shear, arrays of shape ``(n_sources,)`` for the rest.
        data: Packed multi-exposure data (images, PSFs, WCS, etc.).
        pixel_scale: Pixel scale in arcsec/pixel (default 0.1).
        stamp_sizes: Stamp tier sizes.  If ``None``, defaults to
            ``[64, 128, 256]``.

    Returns:
        Model images array of shape ``(n_exp, image_ny, image_nx)``.
    """
    if stamp_sizes is None:
        stamp_sizes = [64, 128, 256]

    tier_indices = _compute_tier_indices(
        data.source_stamp_tier, len(stamp_sizes)
    )

    g1 = jnp.asarray(params["g1"])
    g2 = jnp.asarray(params["g2"])
    flux = jnp.asarray(params["flux"])
    hlr = jnp.asarray(params["hlr"])
    e1 = jnp.asarray(params["e1"])
    e2 = jnp.asarray(params["e2"])
    dx = jnp.asarray(params["dx"])
    dy = jnp.asarray(params["dy"])

    images = []
    for j in range(data.n_exposures):
        img = _render_exposure_image(
            j, g1, g2, flux, hlr, e1, e2, dx, dy,
            data, pixel_scale, stamp_sizes, tier_indices,
        )
        images.append(img)

    return jnp.stack(images, axis=0)
