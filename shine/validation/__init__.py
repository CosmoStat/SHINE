"""SHINE validation module — bias measurement infrastructure."""

from shine.validation.bias_config import (
    AcceptanceCriteria,
    BiasLevel,
    BiasRunConfig,
    BiasTestConfig,
    ConvergenceThresholds,
    ShearGrid,
)
from shine.validation.extraction import (
    ConvergenceDiagnostics,
    RealizationResult,
    ShearEstimates,
    check_convergence,
    extract_convergence_diagnostics,
    extract_realization,
    extract_shear_estimates,
)
from shine.validation.plots import plot_level0_diagnostics
from shine.validation.simulation import (
    SimulationResult,
    generate_biased_observation,
    generate_paired_observations,
)
from shine.validation.statistics import (
    BiasResult,
    CoverageResult,
    SBCResult,
    compute_bias_single_point,
)

__all__ = [
    # Config
    "AcceptanceCriteria",
    "BiasLevel",
    "BiasRunConfig",
    "BiasTestConfig",
    "ConvergenceThresholds",
    "ShearGrid",
    # Simulation
    "SimulationResult",
    "generate_biased_observation",
    "generate_paired_observations",
    # Extraction
    "ConvergenceDiagnostics",
    "RealizationResult",
    "ShearEstimates",
    "check_convergence",
    "extract_convergence_diagnostics",
    "extract_realization",
    "extract_shear_estimates",
    # Statistics
    "BiasResult",
    "CoverageResult",
    "SBCResult",
    "compute_bias_single_point",
    # Plots
    "plot_level0_diagnostics",
]
