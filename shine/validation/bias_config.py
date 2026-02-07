"""Pydantic configuration models for bias testing."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class BiasLevel(str, Enum):
    """Bias testing levels, from noiseless sanity check to survey-realistic."""

    level_0 = "level_0"
    level_1 = "level_1"
    level_2 = "level_2"
    level_3 = "level_3"


class ConvergenceThresholds(BaseModel):
    """Thresholds for MCMC convergence diagnostics.

    Attributes:
        rhat_max: Maximum acceptable R-hat statistic.
        ess_min: Minimum acceptable effective sample size.
        divergence_frac_max: Maximum acceptable fraction of divergent transitions.
        bfmi_min: Minimum acceptable Bayesian Fraction of Missing Information.
    """

    rhat_max: float = 1.05
    ess_min: int = 100
    divergence_frac_max: float = 0.01
    bfmi_min: float = 0.3

    @field_validator("rhat_max")
    @classmethod
    def validate_rhat(cls, v: float) -> float:
        """Validate that R-hat threshold is >= 1."""
        if v < 1.0:
            raise ValueError(f"rhat_max must be >= 1.0, got {v}")
        return v

    @field_validator("ess_min")
    @classmethod
    def validate_ess(cls, v: int) -> int:
        """Validate that ESS threshold is positive."""
        if v <= 0:
            raise ValueError(f"ess_min must be positive, got {v}")
        return v


class ShearGrid(BaseModel):
    """Grid of true shear values for bias regression.

    Attributes:
        values: List of g_true values to test.
    """

    values: List[float] = Field(
        default=[-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05]
    )

    @field_validator("values")
    @classmethod
    def validate_values(cls, v: List[float]) -> List[float]:
        """Validate that shear grid values are within physical bounds."""
        for val in v:
            if abs(val) >= 1.0:
                raise ValueError(
                    f"Shear values must have |g| < 1, got {val}"
                )
        return v


class AcceptanceCriteria(BaseModel):
    """Level-specific acceptance criteria for bias tests.

    Attributes:
        max_offset_sigma: Maximum offset from truth in units of posterior sigma.
        max_posterior_width: Maximum posterior standard deviation.
        max_abs_m: Maximum acceptable |m| (multiplicative bias).
        max_abs_c: Maximum acceptable |c| (additive bias).
        coverage_levels: Expected coverage levels to check (e.g., 68%, 95%).
    """

    max_offset_sigma: float = 3.0
    max_posterior_width: Optional[float] = None
    max_abs_m: Optional[float] = None
    max_abs_c: Optional[float] = None
    coverage_levels: List[float] = Field(default=[0.68, 0.95])


class BiasRunConfig(BaseModel):
    """Configuration for a single bias measurement realization.

    This is the per-GPU-job unit. The campaign orchestrator creates one
    BiasRunConfig per realization.

    Attributes:
        shine_config_path: Path to the base SHINE YAML config.
        g1_true: True g1 shear value for this realization.
        g2_true: True g2 shear value for this realization.
        seed: Random seed for noise generation.
        paired: Whether this is a paired-shear realization.
        output_dir: Directory for output files.
        run_id: Unique identifier for this realization.
    """

    shine_config_path: str
    g1_true: float
    g2_true: float
    seed: int
    paired: bool = False
    output_dir: str
    run_id: str

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v: int) -> int:
        """Validate that seed is non-negative."""
        if v < 0:
            raise ValueError(f"seed must be non-negative, got {v}")
        return v

    @field_validator("g1_true", "g2_true")
    @classmethod
    def validate_shear(cls, v: float) -> float:
        """Validate that shear components are within physical bounds."""
        if abs(v) >= 1.0:
            raise ValueError(f"Shear must have |g| < 1, got {v}")
        return v


class BiasTestConfig(BaseModel):
    """Full campaign configuration for bias testing.

    Attributes:
        level: Bias testing level.
        shine_config_path: Path to the base SHINE YAML config.
        shear_grid: Grid of true shear values.
        n_realizations: Number of realizations per shear grid point.
        paired: Whether to use paired-shear method.
        convergence: Convergence diagnostic thresholds.
        acceptance: Acceptance criteria for bias results.
        output_dir: Top-level output directory.
    """

    level: BiasLevel = BiasLevel.level_0
    shine_config_path: str
    shear_grid: ShearGrid = Field(default_factory=ShearGrid)
    n_realizations: int = 1
    paired: bool = False
    convergence: ConvergenceThresholds = Field(
        default_factory=ConvergenceThresholds
    )
    acceptance: AcceptanceCriteria = Field(default_factory=AcceptanceCriteria)
    output_dir: str = "results/validation"

    @field_validator("n_realizations")
    @classmethod
    def validate_n_realizations(cls, v: int) -> int:
        """Validate that number of realizations is positive."""
        if v <= 0:
            raise ValueError(f"n_realizations must be positive, got {v}")
        return v
