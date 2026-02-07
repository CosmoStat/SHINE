"""CLI entry points for the three-stage bias measurement pipeline.

Stage 1 (run): Generate data + run MCMC → save posterior + truth
Stage 2 (extract): Load posteriors → extract diagnostics + estimates → CSV
Stage 3 (stats): Read CSV → compute bias → check acceptance → plots + JSON
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import arviz as az
import jax
import yaml

from shine.config import ConfigHandler
from shine.validation.bias_config import (
    AcceptanceCriteria,
    BiasLevel,
    BiasRunConfig,
    ConvergenceThresholds,
)
from shine.validation.extraction import extract_realization, split_batched_idata
from shine.validation.plots import plot_level0_diagnostics
from shine.validation.simulation import (
    generate_batch_observations,
    generate_biased_observation,
)
from shine.validation.statistics import compute_bias_single_point

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Stage 1: Run a single bias realization
# --------------------------------------------------------------------------- #


def run_bias_realization() -> None:
    """CLI entry point: shine-bias-run.

    Generates synthetic data with explicit shear, runs MCMC, and saves outputs.
    Supports batched execution (--batch-size > 1) to pack N independent
    realizations into a single MCMC run for GPU efficiency.
    """
    parser = argparse.ArgumentParser(
        description="SHINE bias measurement — run a single realization"
    )
    parser.add_argument("--config", type=str, default=None, help="BiasRunConfig YAML")
    parser.add_argument("--shine-config", type=str, default=None, help="SHINE config YAML")
    parser.add_argument("--g1-true", type=float, default=None)
    parser.add_argument("--g2-true", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)

    # Batched inference arguments
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Number of realizations per GPU job (default: 1)",
    )
    parser.add_argument(
        "--shear-grid", type=float, nargs="+", default=None,
        help="g1 values for shear grid (e.g., 0.01 0.02 0.05)",
    )
    parser.add_argument(
        "--n-realizations", type=int, default=1,
        help="Number of realizations per shear grid point",
    )
    parser.add_argument(
        "--base-seed", type=int, default=42,
        help="Starting seed, incremented per realization",
    )
    args = parser.parse_args()

    # Batched path: --batch-size > 1
    if args.batch_size > 1:
        _run_batched(args)
        return

    # Original single-realization path
    # Build BiasRunConfig from YAML and/or CLI overrides
    run_config_dict = {}
    if args.config:
        with open(args.config) as f:
            run_config_dict = yaml.safe_load(f) or {}

    # CLI args override YAML
    if args.shine_config:
        run_config_dict["shine_config_path"] = args.shine_config
    if args.g1_true is not None:
        run_config_dict["g1_true"] = args.g1_true
    if args.g2_true is not None:
        run_config_dict["g2_true"] = args.g2_true
    if args.seed is not None:
        run_config_dict["seed"] = args.seed
    if args.output_dir:
        run_config_dict["output_dir"] = args.output_dir
    if args.run_id:
        run_config_dict["run_id"] = args.run_id

    # Defaults
    run_config_dict.setdefault("g1_true", 0.0)
    run_config_dict.setdefault("g2_true", 0.0)
    run_config_dict.setdefault("seed", 42)
    run_config_dict.setdefault("output_dir", "results/validation")
    run_config_dict.setdefault("run_id", "r0001")

    if "shine_config_path" not in run_config_dict:
        logger.error("--shine-config is required")
        sys.exit(1)

    try:
        run_cfg = BiasRunConfig(**run_config_dict)
    except Exception as e:
        logger.error(f"Invalid run config: {e}")
        sys.exit(1)

    # Load SHINE config
    shine_config = ConfigHandler.load(run_cfg.shine_config_path)

    # Create output dir
    output_dir = Path(run_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1a: Generate observation
    logger.info(
        f"Generating observation: g1={run_cfg.g1_true}, g2={run_cfg.g2_true}, "
        f"seed={run_cfg.seed}"
    )
    sim_result = generate_biased_observation(
        shine_config, run_cfg.g1_true, run_cfg.g2_true, run_cfg.seed
    )

    # Save ground truth
    truth_path = output_dir / "truth.json"
    with open(truth_path, "w") as f:
        json.dump(sim_result.ground_truth, f, indent=2)
    logger.info(f"Ground truth saved to {truth_path}")

    # Stage 1b: Build model and run inference
    from shine.inference import Inference
    from shine.scene import SceneBuilder

    scene_builder = SceneBuilder(shine_config)
    model_fn = scene_builder.build_model()

    rng_key = jax.random.PRNGKey(shine_config.inference.rng_seed)
    engine = Inference(model=model_fn, config=shine_config.inference)

    logger.info("Running MCMC inference...")
    idata = engine.run(
        rng_key=rng_key,
        observed_data=sim_result.observation.image,
        extra_args={"psf": sim_result.observation.psf_model},
    )

    # Save posterior
    posterior_path = output_dir / "posterior.nc"
    idata.to_netcdf(str(posterior_path))
    logger.info(f"Posterior saved to {posterior_path}")

    # Save convergence summary
    from shine.validation.extraction import extract_convergence_diagnostics

    diagnostics = extract_convergence_diagnostics(idata)
    conv_dict = {
        "rhat": diagnostics.rhat,
        "ess": diagnostics.ess,
        "divergences": diagnostics.divergences,
        "divergence_frac": diagnostics.divergence_frac,
        "bfmi": diagnostics.bfmi,
        "n_samples": diagnostics.n_samples,
        "n_chains": diagnostics.n_chains,
    }
    conv_path = output_dir / "convergence.json"
    with open(conv_path, "w") as f:
        json.dump(conv_dict, f, indent=2)
    logger.info(f"Convergence diagnostics saved to {conv_path}")

    logger.info("Stage 1 (run) complete.")


def _run_batched(args: argparse.Namespace) -> None:
    """Execute the batched inference path.

    Builds a list of (g1, g2, seed, run_id) tuples, generates stacked
    observations, runs batched MCMC, splits the posterior, and saves
    per-realization outputs in the same format as the single-realization path.

    Args:
        args: Parsed CLI arguments (must include batch-size > 1).
    """
    if not args.shine_config:
        logger.error("--shine-config is required")
        sys.exit(1)

    shine_config = ConfigHandler.load(args.shine_config)
    output_dir = Path(args.output_dir or "results/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    g2_true = args.g2_true if args.g2_true is not None else 0.0

    # Build the list of (g1, g2, seed, run_id) tuples
    shear_pairs = []
    seeds = []
    run_ids = []

    if args.shear_grid is not None:
        # Expand shear grid x n_realizations
        for g1_val in args.shear_grid:
            for r in range(args.n_realizations):
                shear_pairs.append((g1_val, g2_true))
                seeds.append(args.base_seed + len(seeds))
                run_ids.append(
                    f"g1_{g1_val:+.4f}_g2_{g2_true:+.4f}_s{args.base_seed + len(run_ids)}"
                )
    else:
        # Single shear point x n_realizations
        g1_true = args.g1_true if args.g1_true is not None else 0.0
        for r in range(args.n_realizations):
            shear_pairs.append((g1_true, g2_true))
            seeds.append(args.base_seed + r)
            run_ids.append(
                f"g1_{g1_true:+.4f}_g2_{g2_true:+.4f}_s{args.base_seed + r}"
            )

    n_total = len(shear_pairs)
    if n_total == 0:
        logger.error("No realizations to run")
        sys.exit(1)

    # Process in chunks of batch_size
    batch_size = args.batch_size

    from shine.inference import Inference
    from shine.scene import SceneBuilder
    from shine.validation.extraction import extract_convergence_diagnostics

    for batch_start in range(0, n_total, batch_size):
        batch_end = min(batch_start + batch_size, n_total)
        batch_shears = shear_pairs[batch_start:batch_end]
        batch_seeds = seeds[batch_start:batch_end]
        batch_run_ids = run_ids[batch_start:batch_end]
        n_batch = len(batch_shears)

        logger.info(
            f"Processing batch [{batch_start}:{batch_end}] "
            f"({n_batch} realizations)"
        )

        # Stage 1a: Generate stacked observations
        batch_result = generate_batch_observations(
            shine_config,
            shear_pairs=batch_shears,
            seeds=batch_seeds,
            run_id_prefix="batch",
        )
        # Use the run_ids we computed, not the auto-generated ones
        batch_result.run_ids = batch_run_ids

        # Stage 1b: Build batched model and run inference
        scene_builder = SceneBuilder(shine_config)
        model_fn = scene_builder.build_batched_model(n_batch)

        rng_key = jax.random.PRNGKey(shine_config.inference.rng_seed)
        engine = Inference(model=model_fn, config=shine_config.inference)

        logger.info(f"Running batched MCMC ({n_batch} realizations)...")
        idata = engine.run(
            rng_key=rng_key,
            observed_data=batch_result.images,
            extra_args={"psf": batch_result.psf_model},
        )

        # Stage 1c: Split posterior and save per-realization outputs
        split_results = split_batched_idata(idata, n_batch, batch_run_ids)

        for i, (run_id, single_idata) in enumerate(split_results):
            run_dir = output_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            # Save ground truth (with seed)
            truth = dict(batch_result.ground_truths[i])
            truth["seed"] = batch_seeds[i]
            truth_path = run_dir / "truth.json"
            with open(truth_path, "w") as f:
                json.dump(truth, f, indent=2)

            # Save posterior
            posterior_path = run_dir / "posterior.nc"
            single_idata.to_netcdf(str(posterior_path))

            # Save convergence summary
            diagnostics = extract_convergence_diagnostics(single_idata)
            conv_dict = {
                "rhat": diagnostics.rhat,
                "ess": diagnostics.ess,
                "divergences": diagnostics.divergences,
                "divergence_frac": diagnostics.divergence_frac,
                "bfmi": diagnostics.bfmi,
                "n_samples": diagnostics.n_samples,
                "n_chains": diagnostics.n_chains,
            }
            conv_path = run_dir / "convergence.json"
            with open(conv_path, "w") as f:
                json.dump(conv_dict, f, indent=2)

            logger.info(f"Saved realization {run_id} to {run_dir}")

    logger.info(
        f"Stage 1 (batched run) complete. "
        f"{n_total} realizations saved to {output_dir}"
    )


# --------------------------------------------------------------------------- #
# Stage 2: Extract results from posteriors
# --------------------------------------------------------------------------- #


def extract_bias_results() -> None:
    """CLI entry point: shine-bias-extract.

    Loads posterior files, extracts diagnostics and estimates, writes CSV.
    """
    parser = argparse.ArgumentParser(
        description="SHINE bias measurement — extract results from posteriors"
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--convergence-config", type=str, default=None)
    args = parser.parse_args()

    # Load convergence thresholds
    if args.convergence_config:
        with open(args.convergence_config) as f:
            conv_dict = yaml.safe_load(f) or {}
        thresholds = ConvergenceThresholds(**conv_dict.get("convergence", conv_dict))
    else:
        thresholds = ConvergenceThresholds()

    input_dir = Path(args.input_dir)
    posterior_files = sorted(input_dir.glob("*/posterior.nc"))

    if not posterior_files:
        logger.warning(f"No posterior.nc files found in {input_dir}/*/")
        sys.exit(0)

    results = []
    for posterior_path in posterior_files:
        run_dir = posterior_path.parent
        run_id = run_dir.name

        # Load truth
        truth_path = run_dir / "truth.json"
        if not truth_path.exists():
            logger.warning(f"Skipping {run_id}: no truth.json found")
            continue

        with open(truth_path) as f:
            truth = json.load(f)

        # Load posterior
        idata = az.from_netcdf(str(posterior_path))

        # Extract seed from run_id or default
        seed = truth.get("seed", 0)

        result = extract_realization(
            idata,
            g1_true=truth["g1"],
            g2_true=truth["g2"],
            run_id=run_id,
            seed=seed,
            thresholds=thresholds,
        )
        results.append(result)
        logger.info(
            f"Extracted {run_id}: g1={result.g1.mean:.6f}, g2={result.g2.mean:.6f}, "
            f"convergence={'PASS' if result.passed_convergence else 'FAIL'}"
        )

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_id", "g1_true", "g2_true",
        "g1_mean", "g1_median", "g1_std",
        "g2_mean", "g2_median", "g2_std",
        "rhat_g1", "rhat_g2", "ess_g1", "ess_g2",
        "divergences", "divergence_frac",
        "passed_convergence", "seed",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "run_id": r.run_id,
                "g1_true": r.g1_true,
                "g2_true": r.g2_true,
                "g1_mean": r.g1.mean,
                "g1_median": r.g1.median,
                "g1_std": r.g1.std,
                "g2_mean": r.g2.mean,
                "g2_median": r.g2.median,
                "g2_std": r.g2.std,
                "rhat_g1": r.diagnostics.rhat.get("g1", ""),
                "rhat_g2": r.diagnostics.rhat.get("g2", ""),
                "ess_g1": r.diagnostics.ess.get("g1", ""),
                "ess_g2": r.diagnostics.ess.get("g2", ""),
                "divergences": r.diagnostics.divergences,
                "divergence_frac": r.diagnostics.divergence_frac,
                "passed_convergence": r.passed_convergence,
                "seed": r.seed,
            })

    logger.info(f"Summary CSV written to {output_path} ({len(results)} realizations)")
    logger.info("Stage 2 (extract) complete.")


# --------------------------------------------------------------------------- #
# Stage 3: Compute bias statistics
# --------------------------------------------------------------------------- #


def _check_offset(
    run_id: str,
    comp: str,
    true_val: float,
    mean: float,
    std: float,
    level: BiasLevel,
    acceptance: AcceptanceCriteria,
) -> bool:
    """Check whether the posterior offset from truth is acceptable.

    For Level 0 with collapsed posteriors (std below max_posterior_width),
    uses absolute offset instead of sigma-based offset, since the latter
    is meaningless for delta-like posteriors.

    Args:
        run_id: Realization identifier for logging.
        comp: Shear component name ("g1" or "g2").
        true_val: True shear value.
        mean: Posterior mean.
        std: Posterior standard deviation.
        level: Bias testing level.
        acceptance: Acceptance criteria.

    Returns:
        True if the offset passes acceptance criteria, False otherwise.
    """
    abs_offset = abs(mean - true_val)

    is_collapsed_level0 = (
        level == BiasLevel.level_0
        and acceptance.max_posterior_width is not None
        and std < acceptance.max_posterior_width
    )

    if is_collapsed_level0:
        if abs_offset > acceptance.max_posterior_width:
            logger.warning(
                f"{run_id}: {comp} absolute offset = "
                f"{abs_offset:.2e} exceeds {acceptance.max_posterior_width}"
            )
            return False
        logger.info(
            f"{run_id}: {comp} offset = {abs_offset:.2e} "
            f"(posterior width = {std:.2e}, collapsed on truth)"
        )
        return True

    if std > 0:
        offset_sigma = abs_offset / std
        if offset_sigma > acceptance.max_offset_sigma:
            logger.warning(
                f"{run_id}: {comp} offset = {offset_sigma:.2f}σ "
                f"exceeds {acceptance.max_offset_sigma}σ"
            )
            return False

    return True


def compute_bias_statistics() -> None:
    """CLI entry point: shine-bias-stats.

    Reads summary CSV, computes bias, checks acceptance, generates plots.
    """
    parser = argparse.ArgumentParser(
        description="SHINE bias measurement — compute statistics"
    )
    parser.add_argument("--input", type=str, required=True, help="Summary CSV")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--level", type=str, default="level_0",
        choices=[lvl.value for lvl in BiasLevel],
    )
    parser.add_argument("--posterior-dir", type=str, default=None)
    parser.add_argument("--acceptance-config", type=str, default=None)
    args = parser.parse_args()

    level = BiasLevel(args.level)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load acceptance criteria
    if args.acceptance_config:
        with open(args.acceptance_config) as f:
            acc_dict = yaml.safe_load(f) or {}
        acceptance = AcceptanceCriteria(**acc_dict.get("acceptance", acc_dict))
    elif level == BiasLevel.level_0:
        # Level 0 defaults: posterior should collapse on truth
        acceptance = AcceptanceCriteria(
            max_offset_sigma=1.0,
            max_posterior_width=0.01,
            max_abs_m=0.01,
        )
    else:
        acceptance = AcceptanceCriteria()

    # Read CSV
    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        logger.error("No data in input CSV")
        sys.exit(1)

    # Filter to converged rows (Level 0 skips convergence filtering since
    # near-zero noise produces degenerate posteriors where R-hat/ESS are meaningless)
    if level == BiasLevel.level_0:
        converged = rows
        logger.info(
            f"Loaded {len(rows)} realizations (Level 0: skipping convergence filter)"
        )
    else:
        converged = [r for r in rows if r["passed_convergence"] == "True"]
        logger.info(
            f"Loaded {len(rows)} realizations, {len(converged)} passed convergence"
        )

    if not converged:
        logger.error("No converged realizations — cannot compute bias")
        sys.exit(1)

    # Compute bias for each row
    bias_results = {"g1": [], "g2": []}
    all_passed = True

    for row in converged:
        g1_true = float(row["g1_true"])
        g2_true = float(row["g2_true"])
        g1_mean = float(row["g1_mean"])
        g2_mean = float(row["g2_mean"])
        g1_std = float(row["g1_std"])
        g2_std = float(row["g2_std"])

        # Check posterior width first (needed to interpret offset check)
        if acceptance.max_posterior_width is not None:
            for comp, std in [("g1", g1_std), ("g2", g2_std)]:
                if std > acceptance.max_posterior_width:
                    logger.warning(
                        f"{row['run_id']}: {comp} width = {std:.6f} "
                        f"exceeds {acceptance.max_posterior_width}"
                    )
                    all_passed = False

        # Check offset from truth
        if acceptance.max_offset_sigma is not None:
            for comp, true_val, mean, std in [
                ("g1", g1_true, g1_mean, g1_std),
                ("g2", g2_true, g2_mean, g2_std),
            ]:
                if not _check_offset(
                    row["run_id"], comp, true_val, mean, std, level, acceptance
                ):
                    all_passed = False

        # Compute m, c for non-zero shear
        if level == BiasLevel.level_0:
            for comp, true, mean, std in [
                ("g1", g1_true, g1_mean, g1_std),
                ("g2", g2_true, g2_mean, g2_std),
            ]:
                if true != 0.0:
                    br = compute_bias_single_point(true, mean, std, comp)
                    bias_results[comp].append({
                        "run_id": row["run_id"],
                        "m": br.m,
                        "m_err": br.m_err,
                        "c": br.c,
                        "c_err": br.c_err,
                    })

                    if acceptance.max_abs_m is not None and abs(br.m) > acceptance.max_abs_m:
                        logger.warning(
                            f"{row['run_id']}: |m_{comp}| = {abs(br.m):.6f} "
                            f"exceeds {acceptance.max_abs_m}"
                        )
                        all_passed = False

    # Generate plots if posterior directory provided
    if args.posterior_dir:
        posterior_dir = Path(args.posterior_dir)
        for row in converged:
            run_id = row["run_id"]
            posterior_path = posterior_dir / run_id / "posterior.nc"
            if posterior_path.exists():
                idata = az.from_netcdf(str(posterior_path))
                plot_dir = output_dir / "plots" / run_id
                plot_level0_diagnostics(
                    idata,
                    float(row["g1_true"]),
                    float(row["g2_true"]),
                    str(plot_dir),
                )

    # Write results JSON
    results_dict = {
        "level": level.value,
        "n_realizations": len(rows),
        "n_converged": len(converged),
        "bias_g1": bias_results["g1"],
        "bias_g2": bias_results["g2"],
        "overall_passed": all_passed,
    }

    results_path = output_dir / "bias_results.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    logger.info(f"Bias results saved to {results_path}")
    logger.info(f"Overall passed: {all_passed}")
    logger.info("Stage 3 (stats) complete.")
