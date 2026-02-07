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
    split_batched_idata,
)
from shine.validation.plots import plot_level0_diagnostics
from shine.validation.simulation import (
    BatchSimulationResult,
    SimulationResult,
    generate_batch_observations,
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
    "BatchSimulationResult",
    "SimulationResult",
    "generate_batch_observations",
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
    "split_batched_idata",
    # Statistics
    "BiasResult",
    "CoverageResult",
    "SBCResult",
    "compute_bias_single_point",
    # Plots
    "plot_level0_diagnostics",
]
