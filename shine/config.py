from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, root_validator, validator

# --- Distribution Models (for Priors) ---


class DistributionConfig(BaseModel):
    """Configuration for probability distributions used as priors.

    This model represents statistical distributions that can be used to define
    priors over model parameters in the Bayesian inference framework. Supports
    Normal, LogNormal, and Uniform distributions with appropriate parameters.

    Attributes:
        type: Distribution type (e.g., 'Normal', 'LogNormal', 'Uniform').
        mean: Mean parameter for Normal/LogNormal distributions.
        sigma: Standard deviation for Normal/LogNormal distributions.
        min: Lower bound for Uniform distributions.
        max: Upper bound for Uniform distributions.
    """
    type: str
    mean: Optional[float] = None
    sigma: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None

    # Allow extra fields for other distributions
    class Config:
        extra = "allow"

    @validator("sigma")
    def validate_sigma_positive(cls, v):
        """Validate that sigma is positive when provided.

        Args:
            v: Sigma value to validate.

        Returns:
            The validated sigma value.

        Raises:
            ValueError: If sigma is not positive.
        """
        if v is not None and v <= 0:
            raise ValueError(f"sigma must be positive, got {v}")
        return v

    @root_validator
    def validate_distribution_params(cls, values):
        """Validate distribution type has required parameters.

        Args:
            values: Dictionary of field values.

        Returns:
            The validated values dictionary.

        Raises:
            ValueError: If required parameters for the distribution type are missing
                or if min >= max for Uniform distributions.
        """
        dist_type = values.get("type")
        mean = values.get("mean")
        sigma = values.get("sigma")
        min_val = values.get("min")
        max_val = values.get("max")

        if dist_type == "Normal" and (mean is None or sigma is None):
            raise ValueError(
                f"Normal distribution requires 'mean' and 'sigma' parameters"
            )

        if dist_type == "Uniform" and (min_val is None or max_val is None):
            raise ValueError(
                f"Uniform distribution requires 'min' and 'max' parameters"
            )

        if dist_type == "Uniform" and min_val is not None and max_val is not None:
            if min_val >= max_val:
                raise ValueError(
                    f"For Uniform distribution, min ({min_val}) must be less than max ({max_val})"
                )

        return values


# --- Component Models ---


class NoiseConfig(BaseModel):
    """Configuration for noise properties in observations.

    Defines the statistical properties of noise added to synthetic observations
    or expected in real data.

    Attributes:
        type: Noise distribution type (currently only 'Gaussian' is supported).
        sigma: Standard deviation of the Gaussian noise.
    """
    type: str = "Gaussian"
    sigma: float

    @validator("sigma")
    def validate_sigma_positive(cls, v):
        """Validate that noise sigma is positive.

        Args:
            v: Sigma value to validate.

        Returns:
            The validated sigma value.

        Raises:
            ValueError: If sigma is not positive.
        """
        if v <= 0:
            raise ValueError(f"Noise sigma must be positive, got {v}")
        return v


class ImageConfig(BaseModel):
    """Configuration for image properties and rendering parameters.

    Defines the image dimensions, pixel scale, number of objects, noise properties,
    and FFT parameters for JAX-GalSim rendering.

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
    n_objects: int = 1  # Default to 1 for simple tests
    noise: NoiseConfig
    fft_size: int = 128  # FFT size for JAX-GalSim rendering (must be power of 2)

    @validator("pixel_scale")
    def validate_pixel_scale_positive(cls, v):
        """Validate that pixel scale is positive.

        Args:
            v: Pixel scale value to validate.

        Returns:
            The validated pixel scale value.

        Raises:
            ValueError: If pixel scale is not positive.
        """
        if v <= 0:
            raise ValueError(f"Pixel scale must be positive, got {v}")
        return v

    @validator("size_x", "size_y")
    def validate_dimensions_positive(cls, v):
        """Validate that image dimensions are positive.

        Args:
            v: Dimension value to validate.

        Returns:
            The validated dimension value.

        Raises:
            ValueError: If dimension is not positive.
        """
        if v <= 0:
            raise ValueError(f"Image dimensions must be positive, got {v}")
        return v

    @validator("n_objects")
    def validate_n_objects_positive(cls, v):
        """Validate that number of objects is positive.

        Args:
            v: Number of objects to validate.

        Returns:
            The validated number of objects.

        Raises:
            ValueError: If number of objects is not positive.
        """
        if v <= 0:
            raise ValueError(f"Number of objects must be positive, got {v}")
        return v

    @validator("fft_size")
    def validate_fft_size(cls, v):
        """Validate that fft_size is a positive power of 2.

        Args:
            v: FFT size to validate.

        Returns:
            The validated FFT size.

        Raises:
            ValueError: If FFT size is not positive or not a power of 2.
        """
        if v <= 0:
            raise ValueError(f"fft_size must be positive, got {v}")
        if v & (v - 1) != 0:
            raise ValueError(f"fft_size must be a power of 2, got {v}")
        return v


class PSFConfig(BaseModel):
    """Configuration for Point Spread Function (PSF) models.

    Defines the PSF model type and its parameters. Supports Gaussian and Moffat profiles.

    Attributes:
        type: PSF model type ('Gaussian' or 'Moffat').
        sigma: Width parameter for the PSF in arcseconds.
        beta: Moffat beta parameter (only used for Moffat PSF, default 2.5).
    """
    type: str = "Gaussian"
    sigma: float
    beta: Optional[float] = 2.5  # For Moffat

    @validator("sigma")
    def validate_sigma_positive(cls, v):
        """Validate that PSF sigma is positive.

        Args:
            v: Sigma value to validate.

        Returns:
            The validated sigma value.

        Raises:
            ValueError: If sigma is not positive.
        """
        if v <= 0:
            raise ValueError(f"PSF sigma must be positive, got {v}")
        return v

    @validator("beta")
    def validate_beta_positive(cls, v):
        """Validate that Moffat beta is positive when provided.

        Args:
            v: Beta value to validate.

        Returns:
            The validated beta value.

        Raises:
            ValueError: If beta is not positive.
        """
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
    # Can be a fixed float or a distribution
    type: Optional[str] = (
        None  # If None, assume fixed value in parent or handled elsewhere
    )
    mean: Optional[float] = 0.0
    sigma: Optional[float] = 0.05

    # To handle the case where it's just a float in YAML, we might need a custom validator
    # but for now let's assume structured input as per design doc


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

    type: str = "Uniform"  # Distribution type for positions
    x_min: Optional[float] = None  # Minimum x position (fraction of image if < 1, pixels if >= 1)
    x_max: Optional[float] = None  # Maximum x position (fraction of image if < 1, pixels if >= 1)
    y_min: Optional[float] = None  # Minimum y position (fraction of image if < 1, pixels if >= 1)
    y_max: Optional[float] = None  # Maximum y position (fraction of image if < 1, pixels if >= 1)

    @root_validator
    def validate_position_bounds(cls, values):
        """Validate that position bounds are consistent.

        Args:
            values: Dictionary of field values.

        Returns:
            The validated values dictionary.

        Raises:
            ValueError: If min values are greater than or equal to max values.
        """
        x_min = values.get("x_min")
        x_max = values.get("x_max")
        y_min = values.get("y_min")
        y_max = values.get("y_max")

        if x_min is not None and x_max is not None and x_min >= x_max:
            raise ValueError(f"x_min ({x_min}) must be less than x_max ({x_max})")

        if y_min is not None and y_max is not None and y_min >= y_max:
            raise ValueError(f"y_min ({y_min}) must be less than y_max ({y_max})")

        return values


class GalaxyConfig(BaseModel):
    """Configuration for galaxy morphology and properties.

    Defines the galaxy profile type and its parameters, including flux, size,
    ellipticity, shear, and position.

    Attributes:
        type: Galaxy profile type (e.g., 'Exponential', 'Sersic').
        n: Sersic index (optional, not used for Exponential profiles).
        flux: Total flux of the galaxy (fixed value or distribution).
        half_light_radius: Half-light radius in arcseconds (fixed value or distribution).
        ellipticity: Intrinsic ellipticity configuration (optional).
        shear: Gravitational shear configuration.
        position: Position prior configuration (optional).
    """
    type: str = "Exponential"  # Changed default from Sersic to Exponential
    n: Optional[Union[float, DistributionConfig]] = (
        None  # Make optional for Exponential
    )
    flux: Union[float, DistributionConfig]
    half_light_radius: Union[float, DistributionConfig]
    ellipticity: Optional[EllipticityConfig] = None  # Intrinsic ellipticity
    shear: ShearConfig
    position: Optional[PositionConfig] = None  # Position prior configuration


class MAPConfig(BaseModel):
    """Configuration for Maximum A Posteriori (MAP) initialization.

    Controls whether to use MAP estimation to initialize MCMC samplers,
    which can improve convergence for complex posteriors.

    Attributes:
        enabled: Whether MAP initialization is enabled (default False).
        num_steps: Number of optimization steps for MAP estimation (default 1000).
        learning_rate: Learning rate for MAP optimization (default 1e-2).
    """

    enabled: bool = False
    num_steps: int = 1000
    learning_rate: float = 1e-2

    @validator("num_steps")
    def validate_num_steps_positive(cls, v):
        """Validate that number of MAP steps is positive.

        Args:
            v: Number of steps to validate.

        Returns:
            The validated number of steps.

        Raises:
            ValueError: If number of steps is not positive.
        """
        if v <= 0:
            raise ValueError(f"Number of MAP steps must be positive, got {v}")
        return v

    @validator("learning_rate")
    def validate_learning_rate_positive(cls, v):
        """Validate that learning rate is positive.

        Args:
            v: Learning rate to validate.

        Returns:
            The validated learning rate.

        Raises:
            ValueError: If learning rate is not positive.
        """
        if v <= 0:
            raise ValueError(f"Learning rate must be positive, got {v}")
        return v


class InferenceConfig(BaseModel):
    """Configuration for Bayesian inference settings.

    Controls the MCMC sampling parameters including warmup, number of samples,
    chains, and optional MAP initialization.

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
    map_init: Optional[MAPConfig] = None  # Optional MAP initialization
    rng_seed: int = 0  # RNG seed for reproducibility

    @validator("warmup", "samples")
    def validate_positive_integers(cls, v, field):
        """Validate that warmup and samples are positive.

        Args:
            v: Value to validate.
            field: Field information.

        Returns:
            The validated value.

        Raises:
            ValueError: If value is not positive.
        """
        if v <= 0:
            raise ValueError(f"{field.name} must be positive, got {v}")
        return v

    @validator("chains")
    def validate_chains_positive(cls, v):
        """Validate that number of chains is positive.

        Args:
            v: Number of chains to validate.

        Returns:
            The validated number of chains.

        Raises:
            ValueError: If number of chains is not positive.
        """
        if v <= 0:
            raise ValueError(f"Number of chains must be positive, got {v}")
        return v

    @validator("rng_seed")
    def validate_rng_seed_non_negative(cls, v):
        """Validate that RNG seed is non-negative.

        Args:
            v: RNG seed to validate.

        Returns:
            The validated RNG seed.

        Raises:
            ValueError: If RNG seed is negative.
        """
        if v < 0:
            raise ValueError(f"RNG seed must be non-negative, got {v}")
        return v


class ShineConfig(BaseModel):
    """Top-level configuration for SHINE inference pipeline.

    Aggregates all configuration components for image properties, PSF, galaxy
    morphology, inference settings, and I/O paths.

    Attributes:
        image: Image and rendering configuration.
        psf: Point Spread Function configuration.
        gal: Galaxy morphology and properties configuration.
        inference: Bayesian inference settings (default factory creates default config).
        data_path: Path to observational data file (optional, None for synthetic data).
        output_path: Directory path for saving results (default 'results').
    """
    image: ImageConfig
    psf: PSFConfig
    gal: GalaxyConfig
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    data_path: Optional[str] = None
    output_path: str = "results"


class ConfigHandler:
    """Handler for loading and validating SHINE configuration files.

    Provides static methods for loading YAML configuration files and validating
    them against the ShineConfig schema.
    """

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
            )

        if data is None:
            raise ValueError(
                f"Configuration file '{path}' is empty or contains only comments."
            )

        # Basic validation and type conversion via Pydantic
        try:
            return ShineConfig(**data)
        except Exception as e:
            raise ValueError(
                f"Configuration validation failed for '{path}':\n{e}"
            )
