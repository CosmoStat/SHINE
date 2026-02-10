"""Euclid VIS multi-exposure data loader for SHINE.

Loads Euclid VIS quadrant-level FITS data, PSF grids, background maps,
and source catalogs, preparing them for probabilistic shear inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from shine.euclid.config import EuclidInferenceConfig

logger = logging.getLogger(__name__)


def _sigma_clipped_median(
    data: np.ndarray, sigma: float = 3.0, maxiters: int = 5
) -> float:
    """Compute sigma-clipped median for background estimation.

    Iteratively clips outliers beyond ``sigma`` standard deviations from
    the median until convergence or ``maxiters`` is reached.

    Args:
        data: 1-D array of pixel values.
        sigma: Clipping threshold in standard deviations.
        maxiters: Maximum number of clipping iterations.

    Returns:
        The sigma-clipped median value.
    """
    d = data.copy()
    for _ in range(maxiters):
        med = np.median(d)
        std = np.std(d)
        mask = np.abs(d - med) < sigma * std
        if mask.all():
            break
        d = d[mask]
    return float(np.median(d))


class EuclidPSFModel:
    """Tiled PSF grid for a single Euclid VIS quadrant.

    The PSF FITS extension stores a single 2-D image that tiles
    ``grid_ny x grid_nx`` individual stamps of size
    ``stamp_size x stamp_size``.  This class splits the tile into a
    4-D array and provides nearest-neighbour and bilinear interpolation
    at arbitrary pixel positions.

    Args:
        psf_data: Raw 2-D PSF tile array (e.g. 189 x 189).
        stamp_size: Side length of each individual PSF stamp.
        grid_nx: Number of stamps along the x (column) axis.
        grid_ny: Number of stamps along the y (row) axis.
        quad_nx: Quadrant width in pixels.
        quad_ny: Quadrant height in pixels.
    """

    def __init__(
        self,
        psf_data: np.ndarray,
        stamp_size: int = 21,
        grid_nx: int = 9,
        grid_ny: int = 9,
        quad_nx: int = 2048,
        quad_ny: int = 2066,
    ) -> None:
        # Split (grid_ny*stamp_size, grid_nx*stamp_size) into
        # (grid_ny, stamp_size, grid_nx, stamp_size) then transpose to
        # (grid_ny, grid_nx, stamp_size, stamp_size).
        self.stamps = psf_data.reshape(
            grid_ny, stamp_size, grid_nx, stamp_size
        ).transpose(0, 2, 1, 3)

        self.stamp_size = stamp_size
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny

        # Grid cell centres in pixel coordinates.
        self.grid_x = np.linspace(
            quad_nx / (2 * grid_nx),
            quad_nx - quad_nx / (2 * grid_nx),
            grid_nx,
        )
        self.grid_y = np.linspace(
            quad_ny / (2 * grid_ny),
            quad_ny - quad_ny / (2 * grid_ny),
            grid_ny,
        )

    def get_nearest(self, x_pix: float, y_pix: float) -> np.ndarray:
        """Return the nearest-neighbour PSF stamp.

        Args:
            x_pix: Source x position in detector pixels (0-indexed).
            y_pix: Source y position in detector pixels (0-indexed).

        Returns:
            PSF stamp of shape ``(stamp_size, stamp_size)`` normalised to
            unit sum.
        """
        ix = int(np.argmin(np.abs(self.grid_x - x_pix)))
        iy = int(np.argmin(np.abs(self.grid_y - y_pix)))
        stamp = self.stamps[iy, ix].copy()
        total = stamp.sum()
        if total > 0:
            stamp /= total
        return stamp

    def interpolate_at(self, x_pix: float, y_pix: float) -> np.ndarray:
        """Bilinear interpolation of the four nearest PSF stamps.

        Args:
            x_pix: Source x position in detector pixels (0-indexed).
            y_pix: Source y position in detector pixels (0-indexed).

        Returns:
            Interpolated PSF stamp of shape ``(stamp_size, stamp_size)``
            normalised to unit sum.
        """
        # Find bounding grid indices along x.
        ix = np.searchsorted(self.grid_x, x_pix) - 1
        ix = int(np.clip(ix, 0, self.grid_nx - 2))

        # Find bounding grid indices along y.
        iy = np.searchsorted(self.grid_y, y_pix) - 1
        iy = int(np.clip(iy, 0, self.grid_ny - 2))

        # Fractional distances within the bounding cell.
        dx = self.grid_x[ix + 1] - self.grid_x[ix]
        dy = self.grid_y[iy + 1] - self.grid_y[iy]

        wx = (x_pix - self.grid_x[ix]) / dx if dx > 0 else 0.5
        wy = (y_pix - self.grid_y[iy]) / dy if dy > 0 else 0.5

        wx = float(np.clip(wx, 0.0, 1.0))
        wy = float(np.clip(wy, 0.0, 1.0))

        # Bilinear combination.
        stamp = (
            (1 - wx) * (1 - wy) * self.stamps[iy, ix]
            + wx * (1 - wy) * self.stamps[iy, ix + 1]
            + (1 - wx) * wy * self.stamps[iy + 1, ix]
            + wx * wy * self.stamps[iy + 1, ix + 1]
        )

        total = stamp.sum()
        if total > 0:
            stamp = stamp / total
        return stamp


class EuclidExposure:
    """Single-quadrant Euclid VIS exposure.

    Reads the science, RMS, and flag extensions from a quadrant-level
    FITS file and exposes WCS transforms and image preparation utilities.

    Args:
        fits_path: Path to the quadrant FITS file (may be gzipped).
        quadrant: CCD quadrant identifier (e.g. ``"3-4.F"``).
    """

    def __init__(self, fits_path: str, quadrant: str = "3-4.F") -> None:
        logger.info("Loading exposure: %s", fits_path)
        with fits.open(fits_path) as hdul:
            sci_hdu = hdul[f"{quadrant}.SCI"]
            self.sci = sci_hdu.data.astype(np.float32)
            self.rms = hdul[f"{quadrant}.RMS"].data.astype(np.float32)
            self.flags = hdul[f"{quadrant}.FLG"].data.astype(np.int32)
            self.wcs = WCS(sci_hdu.header)
            self.gain = float(sci_hdu.header.get("GAIN", 3.48))
            self.exptime = float(sci_hdu.header.get("EXPTIME", 560.52))
            self.magzeropoint = float(sci_hdu.header.get("MAGZEROP", 24.57))
        logger.info(
            "  shape=%s  gain=%.2f  exptime=%.2f  magzp=%.2f",
            self.sci.shape,
            self.gain,
            self.exptime,
            self.magzeropoint,
        )

    def sky_to_pixel(self, ra: float, dec: float) -> tuple[float, float]:
        """Convert sky coordinates to 0-indexed pixel coordinates.

        Uses the full TPV / SIP WCS transform via astropy.

        Args:
            ra: Right ascension in degrees.
            dec: Declination in degrees.

        Returns:
            Tuple ``(x, y)`` of 0-indexed pixel coordinates.
        """
        x, y = self.wcs.all_world2pix(ra, dec, 0)
        return float(x), float(y)

    def is_within_bounds(self, x: float, y: float, margin: int = 32) -> bool:
        """Check whether a pixel position is within the quadrant with margin.

        Args:
            x: Pixel x coordinate (0-indexed).
            y: Pixel y coordinate (0-indexed).
            margin: Required distance from the detector edge in pixels.

        Returns:
            ``True`` if the position is safely inside the quadrant.
        """
        ny, nx = self.sci.shape
        return margin <= x < nx - margin and margin <= y < ny - margin

    def local_wcs_jacobian(
        self, x_pix: float, y_pix: float
    ) -> tuple[float, float, float, float]:
        """Compute the local WCS Jacobian at a pixel position.

        Returns the derivatives ``(dudx, dudy, dvdx, dvdy)`` in
        arcsec/pixel, where *u = -RA cos(dec)* points West and
        *v = Dec* points North.

        The astropy ``pixel_scale_matrix`` gives ``d(sky_deg)/d(pixel)``
        and already includes ``cos(dec)`` on the RA axis for TAN-family
        projections.  We flip the RA sign and convert to arcsec.

        Args:
            x_pix: Pixel x coordinate.
            y_pix: Pixel y coordinate.

        Returns:
            Tuple ``(dudx, dudy, dvdx, dvdy)`` in arcsec/pixel.
        """
        # pixel_scale_matrix is [[dRA/dx, dRA/dy], [dDec/dx, dDec/dy]]
        # in degrees/pixel (RA axis already includes cos(dec)).
        psm = self.wcs.pixel_scale_matrix  # (2, 2) degrees/pixel

        # u = -RA*cos(dec) -> dudx = -dRA/dx * 3600 (cos(dec) already in)
        dudx = -psm[0, 0] * 3600.0
        dudy = -psm[0, 1] * 3600.0
        dvdx = psm[1, 0] * 3600.0
        dvdy = psm[1, 1] * 3600.0

        return (float(dudx), float(dudy), float(dvdx), float(dvdy))

    def prepare_image_data(
        self,
        bad_pixel_mask: int = 0x01EAF5FF,
        background_map: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare image data for inference.

        Applies the bad-pixel mask, subtracts the background, and
        constructs the noise-sigma array.

        Args:
            bad_pixel_mask: Bitmask of defective-pixel flags to exclude.
            background_map: If provided, 2-D background map to subtract.
                Otherwise a sigma-clipped median is used.

        Returns:
            Tuple ``(image, noise_sigma, mask)`` where:

            - *image*: background-subtracted science image (float32).
            - *noise_sigma*: per-pixel sigma; ``1e10`` at masked pixels.
            - *mask*: boolean array, ``True`` for valid pixels.
        """
        mask = (self.flags & bad_pixel_mask) == 0

        if background_map is not None:
            image = self.sci - background_map
        else:
            valid = self.sci[mask]
            bg = _sigma_clipped_median(valid)
            image = self.sci - bg
            logger.info("  sigma-clipped background = %.2f", bg)

        noise_sigma = np.where(mask, self.rms, 1e10)

        return (
            image.astype(np.float32),
            noise_sigma.astype(np.float32),
            mask,
        )


@dataclass
class ExposureSet:
    """All data needed for multi-exposure shear inference, as JAX arrays.

    Attributes:
        images: Background-subtracted science images, shape
            ``(n_exp, ny, nx)``.
        noise_sigma: Per-pixel noise sigma, shape ``(n_exp, ny, nx)``.
            Masked pixels have value ``1e10``.
        masks: Boolean validity masks, shape ``(n_exp, ny, nx)``.
        backgrounds: Estimated background levels, shape ``(n_exp,)``.
        pixel_positions: Source pixel positions per exposure, shape
            ``(n_sources, n_exp, 2)``.
        wcs_jacobians: Local WCS Jacobians per source per exposure,
            shape ``(n_sources, n_exp, 4)`` as
            ``(dudx, dudy, dvdx, dvdy)``.
        psf_images: Interpolated PSF stamps per source per exposure,
            shape ``(n_sources, n_exp, psf_size, psf_size)``.
        source_visible: Visibility flag per source per exposure, shape
            ``(n_sources, n_exp)``.
        catalog_flux_adu: Catalog flux in ADU per source, shape
            ``(n_sources,)``.
        catalog_hlr_arcsec: Catalog half-light radius in arcsec per
            source, shape ``(n_sources,)``.
        source_ids: Object identifiers from the catalog.
        n_exposures: Number of exposures.
        n_sources: Number of selected sources.
        image_ny: Image height in pixels.
        image_nx: Image width in pixels.
    """

    images: jnp.ndarray
    noise_sigma: jnp.ndarray
    masks: jnp.ndarray
    backgrounds: jnp.ndarray

    pixel_positions: jnp.ndarray
    wcs_jacobians: jnp.ndarray
    psf_images: jnp.ndarray
    source_visible: jnp.ndarray

    catalog_flux_adu: jnp.ndarray
    catalog_hlr_arcsec: jnp.ndarray
    source_ids: list[int]

    n_exposures: int
    n_sources: int
    image_ny: int
    image_nx: int


class EuclidDataLoader:
    """Load Euclid VIS multi-exposure data for shear inference.

    Orchestrates reading of exposure FITS files, PSF grid, background
    maps, and the source catalog, then assembles a single
    :class:`ExposureSet` ready for the inference engine.

    Args:
        config: Top-level Euclid inference configuration.
    """

    def __init__(self, config: EuclidInferenceConfig) -> None:
        self.config = config

    def load(self) -> ExposureSet:
        """Load all data and return an :class:`ExposureSet`.

        Returns:
            Fully assembled multi-exposure data structure.
        """
        exposures = self._load_exposures()
        psf_model = self._load_psf()
        backgrounds = self._load_backgrounds()
        catalog = self._load_catalog()
        sources = self._select_sources(catalog)

        logger.info(
            "Selected %d sources from catalog (%d total)",
            len(sources),
            len(catalog),
        )

        # Prepare image data per exposure.
        images_list: list[np.ndarray] = []
        sigma_list: list[np.ndarray] = []
        mask_list: list[np.ndarray] = []
        bg_list: list[float] = []

        for i, exp in enumerate(exposures):
            bkg_map = backgrounds[i] if backgrounds is not None else None
            img, sigma, mask = exp.prepare_image_data(
                self.config.data.bad_pixel_mask, bkg_map
            )
            images_list.append(img)
            sigma_list.append(sigma)
            mask_list.append(mask)
            if bkg_map is not None:
                bg_list.append(float(np.median(bkg_map)))
            else:
                bg_list.append(float(np.median(exp.sci[mask])))

        # Per-source, per-exposure metadata.
        metadata = self._compute_source_metadata(sources, exposures, psf_model)

        ny, nx = exposures[0].sci.shape
        n_exp = len(exposures)
        n_src = len(sources)

        logger.info(
            "ExposureSet: %d exposures, %d sources, image %dx%d",
            n_exp,
            n_src,
            nx,
            ny,
        )

        return ExposureSet(
            images=jnp.array(np.stack(images_list)),
            noise_sigma=jnp.array(np.stack(sigma_list)),
            masks=jnp.array(np.stack(mask_list)),
            backgrounds=jnp.array(bg_list),
            pixel_positions=jnp.array(metadata["pixel_positions"]),
            wcs_jacobians=jnp.array(metadata["wcs_jacobians"]),
            psf_images=jnp.array(metadata["psf_images"]),
            source_visible=jnp.array(metadata["source_visible"]),
            catalog_flux_adu=jnp.array(metadata["flux_adu"]),
            catalog_hlr_arcsec=jnp.array(metadata["hlr_arcsec"]),
            source_ids=metadata["source_ids"],
            n_exposures=n_exp,
            n_sources=n_src,
            image_ny=ny,
            image_nx=nx,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_exposures(self) -> list[EuclidExposure]:
        """Load all exposure FITS files.

        Returns:
            List of :class:`EuclidExposure` instances.
        """
        return [
            EuclidExposure(p, self.config.data.quadrant)
            for p in self.config.data.exposure_paths
        ]

    def _load_psf(self) -> EuclidPSFModel:
        """Load the tiled PSF grid.

        Returns:
            :class:`EuclidPSFModel` instance.
        """
        logger.info("Loading PSF: %s", self.config.data.psf_path)
        with fits.open(self.config.data.psf_path) as hdul:
            psf_data = hdul[self.config.data.quadrant].data.astype(np.float32)
        logger.info("  PSF tile shape: %s", psf_data.shape)
        return EuclidPSFModel(psf_data)

    def _load_backgrounds(self) -> Optional[list[np.ndarray]]:
        """Load pipeline background maps if provided.

        Returns:
            List of 2-D background arrays, or ``None`` if no paths are
            configured.
        """
        if self.config.data.background_paths is None:
            logger.info("No background maps provided; will use sigma-clipped median")
            return None
        backgrounds: list[np.ndarray] = []
        for path in self.config.data.background_paths:
            logger.info("Loading background: %s", path)
            with fits.open(path) as hdul:
                backgrounds.append(
                    hdul[self.config.data.quadrant].data.astype(np.float32)
                )
        return backgrounds

    def _load_catalog(self) -> Table:
        """Load the source catalog.

        Returns:
            Astropy :class:`~astropy.table.Table` with catalog columns.
        """
        logger.info("Loading catalog: %s", self.config.data.catalog_path)
        catalog = Table.read(self.config.data.catalog_path)
        logger.info("  Catalog contains %d sources", len(catalog))
        return catalog

    def _select_sources(self, catalog: Table) -> Table:
        """Filter catalog based on :class:`SourceSelectionConfig`.

        Args:
            catalog: Full source catalog.

        Returns:
            Filtered catalog table.
        """
        src_cfg = self.config.sources
        mask = np.ones(len(catalog), dtype=bool)

        # SNR filter.
        if (
            "flux_vis_psf" in catalog.colnames
            and "fluxerr_vis_psf" in catalog.colnames
        ):
            snr = catalog["flux_vis_psf"] / catalog["fluxerr_vis_psf"]
            mask &= snr >= src_cfg.min_snr
            logger.info(
                "  SNR >= %.1f: %d / %d pass",
                src_cfg.min_snr,
                mask.sum(),
                len(catalog),
            )

        # VIS detection.
        if src_cfg.require_vis_detected and "vis_det" in catalog.colnames:
            mask &= catalog["vis_det"] > 0

        # Spurious flag.
        if src_cfg.exclude_spurious and "spurious_flag" in catalog.colnames:
            mask &= catalog["spurious_flag"] == 0

        # Deblended flag.
        if src_cfg.exclude_deblended and "deblended_flag" in catalog.colnames:
            mask &= catalog["deblended_flag"] == 0

        selected = catalog[mask]
        logger.info("  After all filters: %d sources", len(selected))

        if src_cfg.max_sources is not None and len(selected) > src_cfg.max_sources:
            # Sort by SNR descending, take top N.
            if (
                "flux_vis_psf" in selected.colnames
                and "fluxerr_vis_psf" in selected.colnames
            ):
                snr = selected["flux_vis_psf"] / selected["fluxerr_vis_psf"]
                idx = np.argsort(snr)[::-1][: src_cfg.max_sources]
                selected = selected[idx]
            else:
                selected = selected[: src_cfg.max_sources]
            logger.info("  Capped to %d sources", src_cfg.max_sources)

        return selected

    def _compute_source_metadata(
        self,
        sources: Table,
        exposures: list[EuclidExposure],
        psf_model: EuclidPSFModel,
    ) -> dict:
        """Compute per-source, per-exposure metadata.

        For each source and each exposure this computes pixel positions
        (via WCS), the local WCS Jacobian, the interpolated PSF stamp,
        and a visibility flag.

        Args:
            sources: Selected source catalog.
            exposures: List of loaded exposures.
            psf_model: PSF grid model.

        Returns:
            Dictionary with keys ``pixel_positions``,
            ``wcs_jacobians``, ``psf_images``, ``source_visible``,
            ``flux_adu``, ``hlr_arcsec``, ``source_ids``.
        """
        n_src = len(sources)
        n_exp = len(exposures)
        stamp_size = self.config.galaxy_stamp_size
        margin = stamp_size // 2
        psf_stamp_size = psf_model.stamp_size

        pixel_positions = np.zeros((n_src, n_exp, 2))
        wcs_jacobians = np.zeros((n_src, n_exp, 4))
        psf_images = np.zeros((n_src, n_exp, psf_stamp_size, psf_stamp_size))
        source_visible = np.zeros((n_src, n_exp), dtype=bool)

        for i, src in enumerate(sources):
            ra = float(src["right_ascension"])
            dec = float(src["declination"])

            for j, exp in enumerate(exposures):
                x, y = exp.sky_to_pixel(ra, dec)
                pixel_positions[i, j] = [x, y]

                if exp.is_within_bounds(x, y, margin):
                    source_visible[i, j] = True
                    wcs_jacobians[i, j] = exp.local_wcs_jacobian(x, y)
                    psf_images[i, j] = psf_model.interpolate_at(x, y)

        # Flux conversion: microJansky -> ADU.
        flux_adu = self._flux_ujy_to_adu(sources, exposures[0])

        # Half-light radius from catalog.
        hlr_arcsec = np.zeros(n_src)
        if (
            "kron_radius" in sources.colnames
            and "semimajor_axis" in sources.colnames
        ):
            hlr_arcsec = np.array(
                sources["kron_radius"]
                * sources["semimajor_axis"]
                * self.config.data.pixel_scale
            )
        elif "fwhm" in sources.colnames:
            hlr_arcsec = (
                np.array(sources["fwhm"]) * self.config.data.pixel_scale * 0.5
            )

        hlr_arcsec = np.clip(hlr_arcsec, 0.05, 5.0)

        source_ids = list(sources["object_id"])

        n_visible = source_visible.sum()
        logger.info(
            "  Source visibility: %d / %d source-exposure pairs",
            n_visible,
            n_src * n_exp,
        )

        return {
            "pixel_positions": pixel_positions,
            "wcs_jacobians": wcs_jacobians,
            "psf_images": psf_images,
            "source_visible": source_visible,
            "flux_adu": flux_adu,
            "hlr_arcsec": hlr_arcsec,
            "source_ids": source_ids,
        }

    def _flux_ujy_to_adu(
        self, sources: Table, reference_exposure: EuclidExposure
    ) -> np.ndarray:
        """Convert microJansky fluxes to ADU.

        Uses the formula::

            flux_adu = flux_ujy * 10^((MAGZEROP + 2.5*log10(EXPTIME) - 23.9) / 2.5)

        where 23.9 is the AB magnitude zero-point for 1 micro-Jansky.

        Args:
            sources: Source catalog containing ``flux_vis_psf``.
            reference_exposure: Exposure used for ``MAGZEROP`` and
                ``EXPTIME`` header values.

        Returns:
            Array of fluxes in ADU, clipped to a minimum of 1.0.
        """
        magzp = reference_exposure.magzeropoint
        exptime = reference_exposure.exptime

        conversion = 10 ** ((magzp + 2.5 * np.log10(exptime) - 23.9) / 2.5)

        flux_ujy = np.array(sources["flux_vis_psf"], dtype=np.float64)
        flux_adu = flux_ujy * conversion
        logger.info(
            "  Flux conversion: magzp=%.2f, exptime=%.2f, factor=%.4e",
            magzp,
            exptime,
            conversion,
        )
        return np.clip(flux_adu, 1.0, None).astype(np.float32)
