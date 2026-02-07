from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DistributionConfig(BaseModel):
    """Configuration for probability distributions used as priors.

    Supports Normal, LogNormal, and Uniform distributions with appropriate parameters.

    Attributes:
        type: Distribution type (e.g., 'Normal', 'LogNormal', 'Uniform').
        mean: Mean parameter for Normal/LogNormal distributions.
        sigma: Standard deviation for Normal/LogNormal distributions.
        min: Lower bound for Uniform distributions.
        max: Upper bound for Uniform distributions.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    mean: Optional[float] = None
    sigma: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None

    @field_validator("sigma")
    @classmethod
    def validate_sigma_positive(cls, v: Optional[float]) -> Optional[float]:
        """Validate that sigma is positive when provided."""
        if v is not None and v <= 0:
            raise ValueError(f"sigma must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def validate_distribution_params(self) -> "DistributionConfig":
        """Validate distribution type has required parameters."""
        if self.type == "Normal" and (self.mean is None or self.sigma is None):
            raise ValueError(
                "Normal distribution requires 'mean' and 'sigma' parameters"
            )

        if self.type == "Uniform" and (self.min is None or self.max is None):
            raise ValueError(
                "Uniform distribution requires 'min' and 'max' parameters"
            )

        if self.type == "Uniform" and self.min >= self.max:
            raise ValueError(
                f"For Uniform distribution, min ({self.min}) must be less than max ({self.max})"
            )

        return self


class NoiseConfig(BaseModel):
    """Configuration for noise properties in observations.

    Attributes:
        type: Noise distribution type (currently only 'Gaussian' is supported).
        sigma: Standard deviation of the Gaussian noise.
    """

    type: str = "Gaussian"
    sigma: float

    @field_validator("sigma")
    @classmethod
    def validate_sigma_positive(cls, v: float) -> float:
        """Validate that noise sigma is positive."""
        if v <= 0:
            raise ValueError(f"Noise sigma must be positive, got {v}")
        return v


class ImageConfig(BaseModel):
    """Configuration for image properties and rendering parameters.

    Attributes:
        pixel_scale: Physical size of one pixel in arcseconds.
        size_x: Image width in pixels.
        size_y: Image height in pixels.
        n_objects: Number of galaxy objects to simulate/infer (default 1).
        noise: Noise configuration for the observation.
        fft_size: FFT size for JAX-GalSim rendering, must be a power of 2 (default 128).
    """

    pixel_scale: float
    size_x: int
    size_y: int
    n_objects: int = 1
    noise: NoiseConfig
    fft_size: int = 128

    @field_validator("pixel_scale")
    @classmethod
    def validate_pixel_scale_positive(cls, v: float) -> float:
        """Validate that pixel scale is positive."""
        if v <= 0:
            raise ValueError(f"Pixel scale must be positive, got {v}")
        return v

    @field_validator("size_x", "size_y")
    @classmethod
    def validate_dimensions_positive(cls, v: int) -> int:
        """Validate that image dimensions are positive."""
        if v <= 0:
            raise ValueError(f"Image dimensions must be positive, got {v}")
        return v

    @field_validator("n_objects")
    @classmethod
    def validate_n_objects_positive(cls, v: int) -> int:
        """Validate that number of objects is positive."""
        if v <= 0:
            raise ValueError(f"Number of objects must be positive, got {v}")
        return v

    @field_validator("fft_size")
    @classmethod
    def validate_fft_size(cls, v: int) -> int:
        """Validate that fft_size is a positive power of 2."""
        if v <= 0:
            raise ValueError(f"fft_size must be positive, got {v}")
        if v & (v - 1) != 0:
            raise ValueError(f"fft_size must be a power of 2, got {v}")
        return v


class PSFConfig(BaseModel):
    """Configuration for Point Spread Function (PSF) models.

    Supports Gaussian and Moffat profiles.

    Attributes:
        type: PSF model type ('Gaussian' or 'Moffat').
        sigma: Width parameter for the PSF in arcseconds.
        beta: Moffat beta parameter (only used for Moffat PSF, default 2.5).
    """

    type: str = "Gaussian"
    sigma: float
    beta: Optional[float] = 2.5

    @field_validator("sigma")
    @classmethod
    def validate_sigma_positive(cls, v: float) -> float:
        """Validate that PSF sigma is positive."""
        if v <= 0:
            raise ValueError(f"PSF sigma must be positive, got {v}")
        return v

    @field_validator("beta")
    @classmethod
    def validate_beta_positive(cls, v: Optional[float]) -> Optional[float]:
        """Validate that Moffat beta is positive when provided."""
        if v is not None and v <= 0:
            raise ValueError(f"Moffat beta must be positive, got {v}")
        return v


class ShearComponentConfig(BaseModel):
    """Configuration for individual shear components (g1 or g2).

    Allows specification of shear components as either fixed values or distributions.

    Attributes:
        type: Distribution type (optional, None for fixed values).
        mean: Mean value for the shear component (default 0.0).
        sigma: Standard deviation for the shear component distribution (default 0.05).
    """

    type: Optional[str] = None
    mean: Optional[float] = 0.0
    sigma: Optional[float] = 0.05


class ShearConfig(BaseModel):
    """Configuration for gravitational shear parameters.

    Defines the shear as two components (g1, g2) which can be either fixed values
    or distributions for Bayesian inference.

    Attributes:
        type: Shear parameterization type (e.g., 'G1G2' for reduced shear).
        g1: First shear component (fixed value or distribution).
        g2: Second shear component (fixed value or distribution).
    """

    type: str = "G1G2"
    g1: Union[float, DistributionConfig]
    g2: Union[float, DistributionConfig]


class EllipticityConfig(BaseModel):
    """Configuration for intrinsic galaxy ellipticity.

    Defines the intrinsic ellipticity of galaxies before shearing, parameterized
    as two components (e1, e2).

    Attributes:
        type: Ellipticity parameterization type (e.g., 'E1E2').
        e1: First ellipticity component (fixed value or distribution).
        e2: Second ellipticity component (fixed value or distribution).
    """

    type: str = "E1E2"
    e1: Union[float, DistributionConfig]
    e2: Union[float, DistributionConfig]


class PositionConfig(BaseModel):
    """Configuration for galaxy position priors.

    Defines the prior distribution over galaxy positions in the image.
    Values less than 1 are treated as fractions of image size, values >= 1 as pixels.

    Attributes:
        type: Distribution type for positions (default 'Uniform').
        x_min: Minimum x position (fraction if < 1, pixels if >= 1).
        x_max: Maximum x position (fraction if < 1, pixels if >= 1).
        y_min: Minimum y position (fraction if < 1, pixels if >= 1).
        y_max: Maximum y position (fraction if < 1, pixels if >= 1).
    """

    type: str = "Uniform"
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None

    @model_validator(mode="after")
    def validate_position_bounds(self) -> "PositionConfig":
        """Validate that position bounds are consistent."""
        if self.x_min is not None and self.x_max is not None and self.x_min >= self.x_max:
            raise ValueError(
                f"x_min ({self.x_min}) must be less than x_max ({self.x_max})"
            )
        if self.y_min is not None and self.y_max is not None and self.y_min >= self.y_max:
            raise ValueError(
                f"y_min ({self.y_min}) must be less than y_max ({self.y_max})"
            )
        return self


class CatalogConfig(BaseModel):
    """Configuration for catalog-based scene generation.

    Attributes:
        type: Catalog type ('cosmodc2', 'catsim', 'openuniverse', 'flagship2').
        path: Path to catalog file or directory.
        center_ra: Center Right Ascension in degrees.
        center_dec: Center Declination in degrees.
        size_arcmin: Field size in arcminutes (default 1.0).
        magnitude_limit: Optional magnitude cut (e.g., i < 25).
        use_bulge_disk: Whether to render bulge+disk separately (default True).
    """

    type: str
    path: str
    center_ra: float
    center_dec: float
    size_arcmin: float = 1.0
    magnitude_limit: Optional[float] = None
    use_bulge_disk: bool = True

    @field_validator("type")
    @classmethod
    def validate_catalog_type(cls, v: str) -> str:
        """Validate that catalog type is supported."""
        supported_types = ["cosmodc2", "catsim", "openuniverse", "flagship2"]
        if v not in supported_types:
            raise ValueError(
                f"Unsupported catalog type: {v}. "
                f"Supported types: {', '.join(supported_types)}"
            )
        return v

    @field_validator("size_arcmin")
    @classmethod
    def validate_size_positive(cls, v: float) -> float:
        """Validate that field size is positive."""
        if v <= 0:
            raise ValueError(f"Field size must be positive, got {v}")
        return v

    @field_validator("path")
    @classmethod
    def validate_path_exists(cls, v: str) -> str:
        """Validate that catalog path exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Catalog path does not exist: {v}")
        return v


class GalaxyConfig(BaseModel):
    """Configuration for galaxy morphology and properties.

    Attributes:
        type: Galaxy profile type (e.g., 'Exponential', 'Sersic').
        n: Sersic index (optional, not used for Exponential profiles).
        flux: Total flux of the galaxy (fixed value or distribution).
        half_light_radius: Half-light radius in arcseconds (fixed value or distribution).
        ellipticity: Intrinsic ellipticity configuration (optional).
        shear: Gravitational shear configuration.
        position: Position prior configuration (optional).
    """

    type: str = "Exponential"
    n: Optional[Union[float, DistributionConfig]] = None
    flux: Union[float, DistributionConfig]
    half_light_radius: Union[float, DistributionConfig]
    ellipticity: Optional[EllipticityConfig] = None
    shear: ShearConfig
    position: Optional[PositionConfig] = None


class MAPConfig(BaseModel):
    """Configuration for Maximum A Posteriori (MAP) initialization.

    Attributes:
        enabled: Whether MAP initialization is enabled (default False).
        num_steps: Number of optimization steps for MAP estimation (default 1000).
        learning_rate: Learning rate for MAP optimization (default 1e-2).
    """

    enabled: bool = False
    num_steps: int = 1000
    learning_rate: float = 1e-2

    @field_validator("num_steps")
    @classmethod
    def validate_num_steps_positive(cls, v: int) -> int:
        """Validate that number of MAP steps is positive."""
        if v <= 0:
            raise ValueError(f"Number of MAP steps must be positive, got {v}")
        return v

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate_positive(cls, v: float) -> float:
        """Validate that learning rate is positive."""
        if v <= 0:
            raise ValueError(f"Learning rate must be positive, got {v}")
        return v


class InferenceConfig(BaseModel):
    """Configuration for Bayesian inference settings.

    Attributes:
        warmup: Number of warmup/burn-in steps for MCMC (default 500).
        samples: Number of posterior samples to draw (default 1000).
        chains: Number of independent MCMC chains to run (default 1).
        dense_mass: Whether to use dense mass matrix in NUTS (default False).
        map_init: Optional MAP initialization configuration.
        rng_seed: Random number generator seed for reproducibility (default 0).
    """

    warmup: int = 500
    samples: int = 1000
    chains: int = 1
    dense_mass: bool = False
    map_init: Optional[MAPConfig] = None
    rng_seed: int = 0

    @field_validator("warmup", "samples")
    @classmethod
    def validate_positive_integers(cls, v: int, info) -> int:
        """Validate that warmup and samples are positive."""
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive, got {v}")
        return v

    @field_validator("chains")
    @classmethod
    def validate_chains_positive(cls, v: int) -> int:
        """Validate that number of chains is positive."""
        if v <= 0:
            raise ValueError(f"Number of chains must be positive, got {v}")
        return v

    @field_validator("rng_seed")
    @classmethod
    def validate_rng_seed_non_negative(cls, v: int) -> int:
        """Validate that RNG seed is non-negative."""
        if v < 0:
            raise ValueError(f"RNG seed must be non-negative, got {v}")
        return v


class ShineConfig(BaseModel):
    """Top-level configuration for SHINE inference pipeline.

    Attributes:
        image: Image and rendering configuration.
        psf: Point Spread Function configuration.
        gal: Galaxy morphology and properties configuration.
        inference: Bayesian inference settings (default factory creates default config).
        data_path: Path to observational data file (optional, None for synthetic data).
        catalog: Catalog configuration for scene generation (optional, mutually exclusive with data_path).
        output_path: Directory path for saving results (default 'results').
    """

    image: ImageConfig
    psf: PSFConfig
    gal: GalaxyConfig
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    data_path: Optional[str] = None
    catalog: Optional[CatalogConfig] = None
    output_path: str = "results"

    @model_validator(mode="after")
    def validate_data_source(self) -> "ShineConfig":
        """Validate that only one data source is specified."""
        if self.data_path is not None and self.catalog is not None:
            raise ValueError(
                "Cannot specify both 'data_path' and 'catalog'. "
                "Please use only one data source."
            )
        return self


class ConfigHandler:
    """Handler for loading and validating SHINE configuration files."""

    @staticmethod
    def load(path: str) -> ShineConfig:
        """Load and validate configuration from YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated ShineConfig object.

        Raises:
            FileNotFoundError: If config file does not exist.
            yaml.YAMLError: If YAML parsing fails.
            ValueError: If configuration validation fails.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}\n"
                f"Please provide a valid path to a YAML configuration file."
            )

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Failed to parse YAML configuration file '{path}':\n{e}"
            ) from e

        if data is None:
            raise ValueError(
                f"Configuration file '{path}' is empty or contains only comments."
            )

        try:
            return ShineConfig(**data)
        except Exception as e:
            raise ValueError(
                f"Configuration validation failed for '{path}':\n{e}"
            ) from e
