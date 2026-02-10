"""Multi-exposure multi-object scene model for Euclid VIS inference.

Builds a NumPyro forward generative model that samples global shear and
per-galaxy parameters, renders each galaxy through per-exposure PSFs and
WCS, scatter-adds the stamps onto full-sized model images, and evaluates
a per-pixel Gaussian likelihood weighted by the RMS noise map.
"""

import logging
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax_galsim as galsim
import numpyro
import numpyro.distributions as dist

from shine.euclid.config import EuclidInferenceConfig

logger = logging.getLogger(__name__)


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
    gsparams: galsim.GSParams,
    pixel_scale: float,
    stamp_size: int,
    observed_data: Optional[jnp.ndarray],
    extra_args: dict,
) -> None:
    """Render all galaxies for one exposure and evaluate likelihood.

    Generates stamps for every source via ``jax.vmap``, scatter-adds them
    onto a blank model image using ``jax.lax.scan``, and registers the
    per-pixel Gaussian likelihood as a NumPyro sample site.

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
        gsparams: JAX-GalSim rendering parameters.
        pixel_scale: Pixel scale in arcsec/pixel.
        stamp_size: Side length of the internal rendering stamp in pixels.
        observed_data: Observed images, shape ``(n_exp, ny, nx)``, or None
            for prior predictive sampling.
        extra_args: Additional keyword arguments forwarded from the model
            call (unused, reserved for future extensions).
    """
    # Per-source data for this exposure
    pos_j = data.pixel_positions[:, exp_idx, :]   # (n_src, 2)
    wcs_j = data.wcs_jacobians[:, exp_idx, :]     # (n_src, 4)
    psf_j = data.psf_images[:, exp_idx, :, :]     # (n_src, psf_ny, psf_nx)
    visible_j = data.source_visible[:, exp_idx]    # (n_src,)

    def render_one_galaxy(
        flux_i: jnp.ndarray,
        hlr_i: jnp.ndarray,
        e1_i: jnp.ndarray,
        e2_i: jnp.ndarray,
        dx_i: jnp.ndarray,
        dy_i: jnp.ndarray,
        psf_img: jnp.ndarray,
        wcs_params: jnp.ndarray,
        pix_pos: jnp.ndarray,
        visible_i: jnp.ndarray,
    ) -> jnp.ndarray:
        """Render a single galaxy onto a local stamp.

        Args:
            flux_i: Galaxy flux in ADU.
            hlr_i: Half-light radius in arcsec.
            e1_i: Intrinsic ellipticity component 1.
            e2_i: Intrinsic ellipticity component 2.
            dx_i: Position offset in arcsec (x).
            dy_i: Position offset in arcsec (y).
            psf_img: Interpolated PSF stamp for this source/exposure.
            wcs_params: Local WCS Jacobian ``[dudx, dudy, dvdx, dvdy]``.
            pix_pos: Catalog pixel position ``[x, y]`` (unused inside
                rendering, kept for vmap signature consistency).
            visible_i: Visibility flag (1 = visible, 0 = masked).

        Returns:
            Rendered stamp array of shape ``(stamp_size, stamp_size)``.
        """
        gal = galsim.Exponential(
            flux=flux_i, half_light_radius=hlr_i, gsparams=gsparams
        )
        gal = gal.shear(e1=e1_i, e2=e2_i)
        gal = gal.shear(g1=g1, g2=g2)

        psf = galsim.InterpolatedImage(
            galsim.Image(psf_img, scale=pixel_scale),
            gsparams=gsparams,
        )
        final = galsim.Convolve([gal, psf], gsparams=gsparams)

        wcs = galsim.JacobianWCS(
            dudx=wcs_params[0],
            dudy=wcs_params[1],
            dvdx=wcs_params[2],
            dvdy=wcs_params[3],
        )

        # dx, dy are small arcsec offsets; convert to pixel offset
        pix_dx = dx_i / pixel_scale
        pix_dy = dy_i / pixel_scale

        stamp = final.drawImage(
            nx=stamp_size,
            ny=stamp_size,
            wcs=wcs,
            offset=galsim.PositionD(pix_dx, pix_dy),
        ).array

        # Zero out if source not visible in this exposure
        stamp = stamp * visible_i
        return stamp

    # Vectorise rendering over all sources
    all_stamps = jax.vmap(render_one_galaxy)(
        flux, hlr, e1, e2, dx, dy, psf_j, wcs_j, pos_j, visible_j
    )  # (n_src, stamp_size, stamp_size)

    # Scatter-add stamps onto the full model image
    stamp_half = stamp_size // 2
    corner_x = jnp.round(pos_j[:, 0]).astype(jnp.int32) - stamp_half
    corner_y = jnp.round(pos_j[:, 1]).astype(jnp.int32) - stamp_half

    def scatter_add(
        image: jnp.ndarray,
        inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, None]:
        """Add one stamp onto the running model image via dynamic slicing."""
        stamp, iy, ix = inputs
        iy = jnp.clip(iy, 0, data.image_ny - stamp_size)
        ix = jnp.clip(ix, 0, data.image_nx - stamp_size)
        current = jax.lax.dynamic_slice(
            image, (iy, ix), (stamp_size, stamp_size)
        )
        return jax.lax.dynamic_update_slice(
            image, current + stamp, (iy, ix)
        ), None

    model_image, _ = jax.lax.scan(
        scatter_add,
        jnp.zeros((data.image_ny, data.image_nx)),
        (all_stamps, corner_y, corner_x),
    )

    # Likelihood: per-pixel Gaussian weighted by the RMS noise map
    noise_sigma_j = data.noise_sigma[exp_idx]  # (ny, nx); 1e10 at flagged pixels

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

    Attributes:
        config: Euclid inference configuration.
        data: Packed multi-exposure data (images, PSFs, WCS, noise maps).
    """

    def __init__(
        self, config: EuclidInferenceConfig, exposure_set: "ExposureSet"
    ) -> None:
        """Initialize the multi-exposure scene builder.

        Args:
            config: Euclid inference configuration including priors, stamp
                size, FFT size, and pixel scale.
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
        stamp_size = self.config.galaxy_stamp_size
        fft_size = self.config.fft_size
        pixel_scale = self.config.data.pixel_scale

        logger.info(
            "Building multi-exposure model: %d sources, %d exposures, "
            "stamp=%dpx, fft=%d",
            data.n_sources,
            data.n_exposures,
            stamp_size,
            fft_size,
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
            gsparams = galsim.GSParams(
                maximum_fft_size=fft_size, minimum_fft_size=fft_size
            )

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
                    j,
                    g1,
                    g2,
                    flux,
                    hlr,
                    e1,
                    e2,
                    dx,
                    dy,
                    data,
                    gsparams,
                    pixel_scale,
                    stamp_size,
                    observed_data,
                    extra_args,
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
        stamp_size = self.config.galaxy_stamp_size
        fft_size = self.config.fft_size
        pixel_scale = self.config.data.pixel_scale

        logger.info(
            "Building single-exposure model: %d sources, exposure %d, "
            "stamp=%dpx, fft=%d",
            data.n_sources,
            exposure_idx,
            stamp_size,
            fft_size,
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
            gsparams = galsim.GSParams(
                maximum_fft_size=fft_size, minimum_fft_size=fft_size
            )

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
                exposure_idx,
                g1,
                g2,
                flux,
                hlr,
                e1,
                e2,
                dx,
                dy,
                data,
                gsparams,
                pixel_scale,
                stamp_size,
                observed_data,
                extra_args,
            )

        return model
