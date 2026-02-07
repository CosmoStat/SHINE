"""Paired-shear simulation driver for bias measurement."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import jax.numpy as jnp

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


@dataclass
class BatchSimulationResult:
    """Result of batched bias simulation.

    Attributes:
        images: Stacked observed images, shape (n_batch, nx, ny).
        psf_model: Shared JAX-GalSim PSF object.
        ground_truths: List of ground truth dicts, one per realization.
        run_ids: List of run identifiers.
    """

    images: jnp.ndarray
    psf_model: Any
    ground_truths: List[Dict[str, float]]
    run_ids: List[str]


def generate_batch_observations(
    config: ShineConfig,
    shear_pairs: List[Tuple[float, float]],
    seeds: List[int],
    run_id_prefix: str = "batch",
) -> BatchSimulationResult:
    """Generate N observations and stack them into arrays.

    Args:
        config: SHINE configuration object.
        shear_pairs: List of (g1, g2) pairs, one per realization.
        seeds: List of random seeds, one per realization.
        run_id_prefix: Prefix for run identifiers.

    Returns:
        BatchSimulationResult with stacked images and shared PSF.

    Raises:
        ValueError: If shear_pairs and seeds have different lengths.
    """
    if len(shear_pairs) != len(seeds):
        raise ValueError(
            f"shear_pairs ({len(shear_pairs)}) and seeds ({len(seeds)}) "
            f"must have the same length"
        )

    n_batch = len(shear_pairs)
    images = []
    ground_truths = []
    run_ids = []
    psf_model = None

    for i, ((g1, g2), seed) in enumerate(zip(shear_pairs, seeds)):
        run_id = f"{run_id_prefix}_{i:04d}"
        sim_result = generate_biased_observation(config, g1, g2, seed)
        images.append(sim_result.observation.image)
        ground_truths.append(sim_result.ground_truth)
        run_ids.append(run_id)

        # Use PSF from first observation (shared across batch)
        if psf_model is None:
            psf_model = sim_result.observation.psf_model

    stacked_images = jnp.stack(images, axis=0)

    logger.info(
        f"Generated batch of {n_batch} observations, "
        f"stacked shape: {stacked_images.shape}"
    )

    return BatchSimulationResult(
        images=stacked_images,
        psf_model=psf_model,
        ground_truths=ground_truths,
        run_ids=run_ids,
    )
