"""Pydantic configuration models for Euclid VIS shear inference.

Provides structured, validated configuration for Euclid VIS data paths,
source selection criteria, prior distributions, and inference settings.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, field_validator

from shine.config import InferenceConfig


class EuclidDataConfig(BaseModel):
    """Paths and settings for Euclid VIS data products.

    Attributes:
        exposure_paths: Paths to quadrant-level FITS exposure files.
        psf_path: Path to the PSF grid FITS file.
        catalog_path: Path to the source catalog FITS file.
        background_paths: Optional paths to background estimation FITS files
            (e.g., from NoiseChisel).
        quadrant: CCD quadrant identifier (default "3-4.F").
        pixel_scale: Pixel scale in arcsec/pixel (default 0.1).
        bad_pixel_mask: Bitmask of defective-pixel flags to exclude.
            Default ``0x1`` matches the ``INVALID`` convenience bit
            (bit 0) from the VIS flag map, which is the OR of HOT,
            COLD, SAT, COSMIC, GHOST, CHARINJ, SATXTALKGHOST, ADCMAX,
            and other flags that mark pixels whose values should not be
            used. See the `VIS Flag Map DPDD
            <https://euclid.esac.esa.int/msp/dpdd/v1.1/visdpd/dpcards/vis_flagmap.html>`_.
    """

    exposure_paths: List[str]
    psf_path: str
    catalog_path: str
    background_paths: Optional[List[str]] = None
    quadrant: str = "3-4.F"
    pixel_scale: float = 0.1
    bad_pixel_mask: int = 0x1

    @field_validator("pixel_scale")
    @classmethod
    def validate_pixel_scale_positive(cls, v: float) -> float:
        """Validate that pixel scale is positive.

        Args:
            v: Pixel scale value to validate.

        Returns:
            The validated pixel scale.

        Raises:
            ValueError: If pixel scale is not positive.
        """
        if v <= 0:
            raise ValueError(f"Pixel scale must be positive, got {v}")
        return v


class SourceSelectionConfig(BaseModel):
    """Catalog filtering criteria for source selection.

    Controls which sources from the Euclid catalog are included in
    the inference. Sources can be filtered by signal-to-noise ratio,
    detection flags, blending status, and morphological classification.

    Attributes:
        min_snr: Minimum signal-to-noise ratio for source inclusion
            (default 10.0).
        require_vis_detected: Only include sources detected in VIS
            (default True).
        exclude_spurious: Exclude sources flagged as spurious
            (default True).
        exclude_deblended: Exclude deblended sources. False by default
            because SHINE can model blended sources jointly.
        exclude_point_sources: Exclude sources classified as point-like
            (stars). Uses the ``point_like_flag`` column from the MER
            catalogue (default True).
        det_quality_exclude_mask: Bitmask of ``det_quality_flag`` bits
            that trigger source exclusion.  Default ``0x78C`` excludes
            saturated (bit 3), border (bit 4), VIS bright-star mask
            (bit 8), NIR bright-star mask (bit 9), extended-object area
            (bit 10), and deblending-skipped (bit 11).  Bits 1
            (neighbor contamination) and 2 (blended) are kept because
            SHINE can handle these.  Set to 0 to disable.
        max_sources: Maximum number of sources to process. None means
            no limit.
    """

    min_snr: float = 10.0
    require_vis_detected: bool = True
    exclude_spurious: bool = True
    exclude_deblended: bool = False
    exclude_point_sources: bool = True
    det_quality_exclude_mask: int = 0x78C
    max_sources: Optional[int] = None

    @field_validator("min_snr")
    @classmethod
    def validate_min_snr_positive(cls, v: float) -> float:
        """Validate that minimum SNR is positive.

        Args:
            v: Minimum SNR value to validate.

        Returns:
            The validated minimum SNR.

        Raises:
            ValueError: If min_snr is not positive.
        """
        if v <= 0:
            raise ValueError(f"min_snr must be positive, got {v}")
        return v

    @field_validator("max_sources")
    @classmethod
    def validate_max_sources_positive(cls, v: Optional[int]) -> Optional[int]:
        """Validate that max_sources is positive when provided.

        Args:
            v: Maximum number of sources to validate.

        Returns:
            The validated max_sources value.

        Raises:
            ValueError: If max_sources is not positive.
        """
        if v is not None and v <= 0:
            raise ValueError(f"max_sources must be positive, got {v}")
        return v


class PriorConfig(BaseModel):
    """Prior distribution parameters for Bayesian inference.

    Defines the width (sigma) of prior distributions on galaxy and
    shear parameters. All priors are centered on catalog values or
    zero (for shear).

    Attributes:
        shear_prior_sigma: Width of the shear prior (default 0.05).
        flux_prior_log_sigma: Width of the log-flux prior (default 0.5).
        hlr_prior_log_sigma: Width of the log-half-light-radius prior
            (default 0.3).
        ellipticity_prior_sigma: Width of the ellipticity prior
            (default 0.3).
        position_prior_sigma: Width of the position prior in arcsec
            (default 0.05).
    """

    shear_prior_sigma: float = 0.05
    flux_prior_log_sigma: float = 0.5
    hlr_prior_log_sigma: float = 0.3
    ellipticity_prior_sigma: float = 0.3
    position_prior_sigma: float = 0.05

    @field_validator(
        "shear_prior_sigma",
        "flux_prior_log_sigma",
        "hlr_prior_log_sigma",
        "ellipticity_prior_sigma",
        "position_prior_sigma",
    )
    @classmethod
    def validate_sigma_positive(cls, v: float, info) -> float:
        """Validate that all prior sigma values are positive.

        Args:
            v: Sigma value to validate.
            info: Pydantic field validation info.

        Returns:
            The validated sigma value.

        Raises:
            ValueError: If sigma is not positive.
        """
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive, got {v}")
        return v


class EuclidInferenceConfig(BaseModel):
    """Top-level configuration for Euclid VIS shear inference.

    Combines data paths, source selection, priors, and inference settings
    into a single validated configuration. Reuses the base
    ``shine.config.InferenceConfig`` for MCMC/VI settings.

    Attributes:
        data: Euclid data file paths and pixel settings.
        sources: Source selection and filtering criteria.
        priors: Prior distribution parameters.
        inference: Base SHINE inference configuration (NUTS/MAP/VI).
        galaxy_stamp_sizes: Available rendering stamp tiers in pixels,
            sorted ascending (default ``[64, 128, 256]``).  Each source
            is assigned the smallest tier whose stamp can contain its
            light profile.  The FFT size for each tier is ``2 * stamp``.
        background: Background estimation strategy: "fit" estimates
            background jointly, "median" uses the median of the image,
            "fixed" uses a provided background map (default "median").
        output_dir: Directory for saving inference results
            (default "results/euclid").
    """

    data: EuclidDataConfig
    sources: SourceSelectionConfig = SourceSelectionConfig()
    priors: PriorConfig = PriorConfig()
    inference: InferenceConfig = InferenceConfig()
    galaxy_stamp_sizes: List[int] = [64, 128, 256]
    background: Literal["fit", "median", "fixed"] = "median"
    output_dir: str = "results/euclid"

    @field_validator("galaxy_stamp_sizes")
    @classmethod
    def validate_stamp_sizes(cls, v: List[int]) -> List[int]:
        """Validate that stamp sizes are positive and sorted ascending.

        Args:
            v: List of stamp sizes to validate.

        Returns:
            The validated stamp sizes.

        Raises:
            ValueError: If list is empty, sizes are not positive, or
                not sorted ascending.
        """
        if not v:
            raise ValueError("galaxy_stamp_sizes must not be empty")
        for s in v:
            if s <= 0:
                raise ValueError(
                    f"All stamp sizes must be positive, got {s}"
                )
        if v != sorted(v):
            raise ValueError(
                f"galaxy_stamp_sizes must be sorted ascending, got {v}"
            )
        return v
