"""Pydantic configuration models for Euclid VIS shear inference.

Provides structured, validated configuration for Euclid VIS data paths,
source selection criteria, galaxy model specification, and inference
settings.  The galaxy model (priors, profile type) is specified via the
shared :class:`~shine.config.GalaxyConfig`, making the probabilistic
model explicit in the YAML configuration file.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from shine.config import (
    DistributionConfig,
    EllipticityConfig,
    GalaxyConfig,
    InferenceConfig,
    PositionConfig,
    ShearConfig,
)


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


def _default_euclid_galaxy_config() -> GalaxyConfig:
    """Build the default Euclid galaxy model specification.

    The default model uses catalog-centered LogNormal priors for flux and
    half-light radius, Normal priors for shear and ellipticity, and
    Normal offset priors for position.  This matches the priors that were
    previously hard-coded in ``MultiExposureScene._sample_parameters()``.
    """
    return GalaxyConfig(
        type="Exponential",
        flux=DistributionConfig(type="LogNormal", center="catalog", sigma=0.5),
        half_light_radius=DistributionConfig(
            type="LogNormal", center="catalog", sigma=0.3
        ),
        shear=ShearConfig(
            type="G1G2",
            g1=DistributionConfig(type="Normal", mean=0.0, sigma=0.05),
            g2=DistributionConfig(type="Normal", mean=0.0, sigma=0.05),
        ),
        ellipticity=EllipticityConfig(
            type="E1E2",
            e1=DistributionConfig(type="Normal", mean=0.0, sigma=0.3),
            e2=DistributionConfig(type="Normal", mean=0.0, sigma=0.3),
        ),
        position=PositionConfig(
            type="Offset",
            dx=DistributionConfig(type="Normal", mean=0.0, sigma=0.05),
            dy=DistributionConfig(type="Normal", mean=0.0, sigma=0.05),
        ),
    )


class EuclidInferenceConfig(BaseModel):
    """Top-level configuration for Euclid VIS shear inference.

    Combines data paths, source selection, galaxy model specification,
    and inference settings into a single validated configuration.  The
    galaxy model (profile type, priors) is specified via the shared
    :class:`~shine.config.GalaxyConfig`, making the probabilistic model
    explicit in the YAML file.

    Attributes:
        data: Euclid data file paths and pixel settings.
        sources: Source selection and filtering criteria.
        gal: Galaxy model specification (profile type, priors).
            Defaults match the previously hard-coded Euclid priors:
            catalog-centered LogNormal for flux/hlr, Normal for
            shear/ellipticity, Normal offsets for position.
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
    gal: GalaxyConfig = Field(default_factory=_default_euclid_galaxy_config)
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
