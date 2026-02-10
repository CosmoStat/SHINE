"""SHINE validation module — bias measurement infrastructure."""

from shine.validation.bias_config import (
    AcceptanceCriteria,
    BiasLevel,
    BiasRunConfig,
    BiasTestConfig,
    ConvergenceThresholds,
    FluxNoiseGrid,
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
from shine.validation.plots import (
    plot_bias_vs_shear,
    plot_coverage,
    plot_level0_diagnostics,
    plot_sbc_histogram,
)
from shine.validation.simulation import (
    BatchSimulationResult,
    SimulationResult,
    draw_ellipticity,
    generate_batch_observations,
    generate_biased_observation,
    generate_paired_observations,
)
from shine.validation.statistics import (
    BiasResult,
    CoverageResult,
    SBCResult,
    compute_bias_regression,
    compute_bias_single_point,
    compute_coverage,
    compute_paired_response,
    compute_sbc_ranks,
    jackknife_bias,
)

__all__ = [
    # Config
    "AcceptanceCriteria",
    "BiasLevel",
    "BiasRunConfig",
    "BiasTestConfig",
    "ConvergenceThresholds",
    "FluxNoiseGrid",
    "ShearGrid",
    # Simulation
    "BatchSimulationResult",
    "SimulationResult",
    "draw_ellipticity",
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
    "compute_bias_regression",
    "compute_bias_single_point",
    "compute_coverage",
    "compute_paired_response",
    "compute_sbc_ranks",
    "jackknife_bias",
    # Plots
    "plot_bias_vs_shear",
    "plot_coverage",
    "plot_level0_diagnostics",
    "plot_sbc_histogram",
]
