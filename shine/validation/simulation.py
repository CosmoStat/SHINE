"""Paired-shear simulation driver for bias measurement."""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

from shine.config import ShineConfig
from shine.data import DataLoader, Observation, get_mean

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Result of a single bias simulation.

    Attributes:
        observation: The generated Observation (image + noise map + PSF).
        ground_truth: Dictionary of true parameter values used to generate data.
    """

    observation: Observation
    ground_truth: Dict[str, float]


def generate_biased_observation(
    config: ShineConfig,
    g1_true: float,
    g2_true: float,
    seed: int,
) -> SimulationResult:
    """Generate a synthetic observation with explicit shear overrides.

    Args:
        config: SHINE configuration object.
        g1_true: True g1 shear value.
        g2_true: True g2 shear value.
        seed: Random seed for noise generation.

    Returns:
        SimulationResult containing the observation and ground truth dict.
    """
    observation = DataLoader.generate_synthetic(
        config,
        g1_true=g1_true,
        g2_true=g2_true,
        noise_seed=seed,
    )

    # Build ground truth dict from config + overrides
    e1 = 0.0
    e2 = 0.0
    if config.gal.ellipticity is not None:
        e1 = get_mean(config.gal.ellipticity.e1)
        e2 = get_mean(config.gal.ellipticity.e2)

    ground_truth = {
        "g1": g1_true,
        "g2": g2_true,
        "flux": get_mean(config.gal.flux),
        "hlr": get_mean(config.gal.half_light_radius),
        "e1": e1,
        "e2": e2,
    }

    logger.info(
        f"Generated observation: g1={g1_true:.4f}, g2={g2_true:.4f}, seed={seed}"
    )

    return SimulationResult(observation=observation, ground_truth=ground_truth)


def generate_paired_observations(
    config: ShineConfig,
    g1_true: float,
    g2_true: float,
    seed: int,
) -> Tuple[SimulationResult, SimulationResult]:
    """Generate paired +g/-g observations with the same noise seed.

    The paired-shear method cancels shape noise by averaging responses
    from +g and -g images generated with identical noise realizations.

    Args:
        config: SHINE configuration object.
        g1_true: Positive g1 shear value.
        g2_true: Positive g2 shear value.
        seed: Random seed for noise generation (same for both).

    Returns:
        Tuple of (plus_result, minus_result) SimulationResults.
    """
    plus_result = generate_biased_observation(config, g1_true, g2_true, seed)
    minus_result = generate_biased_observation(config, -g1_true, -g2_true, seed)

    logger.info(
        f"Generated paired observations: +g=({g1_true:.4f}, {g2_true:.4f}), "
        f"-g=({-g1_true:.4f}, {-g2_true:.4f}), seed={seed}"
    )

    return plus_result, minus_result
