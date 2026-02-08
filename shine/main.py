import argparse
import logging
import sys
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import yaml

from shine.config import ConfigHandler
from shine.data import DataLoader
from shine.inference import Inference
from shine.scene import SceneBuilder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for SHINE inference pipeline.

    Handles configuration loading, data preparation, model building,
    inference execution, and results saving.
    """
    parser = argparse.ArgumentParser(description="SHINE: SHear INference Environment")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML"
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    try:
        # 1. Load and validate configuration
        logger.info(f"Loading configuration from {args.config}")
        try:
            config = ConfigHandler.load(args.config)
        except (yaml.YAMLError, FileNotFoundError, ValueError) as e:
            logger.error(f"Configuration error: {e}")
            sys.exit(1)

        if args.output:
            config.output_path = args.output

        # 2. Create output directory
        output_dir = Path(config.output_path)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            sys.exit(1)
        logger.info(f"Output directory: {output_dir}")

        # 3. Load or generate data
        logger.info("Loading observation data...")
        try:
            observation = DataLoader.load(config)
        except NotImplementedError as e:
            logger.error(f"Data loading not implemented: {e}")
            sys.exit(1)

        # 4. Save observation for reference (non-critical)
        try:
            jnp.savez(
                output_dir / "observation.npz",
                image=observation.image,
                noise_map=observation.noise_map,
            )
            logger.info("Observation data saved")
        except Exception as e:
            logger.warning(f"Failed to save observation data: {e}")

        # 5. Build probabilistic model
        logger.info("Building probabilistic scene model...")
        scene_builder = SceneBuilder(config)
        model_fn = scene_builder.build_model()

        # 6. Run Bayesian inference
        logger.info(f"Starting {config.inference.method.upper()} inference pipeline...")
        try:
            rng_key = jax.random.PRNGKey(config.inference.rng_seed)
            engine = Inference(model=model_fn, config=config.inference)
            results = engine.run(
                rng_key=rng_key,
                observed_data=observation.image,
                extra_args={"psf": observation.psf_model},
            )
        except RuntimeError as e:
            logger.error(f"Inference failed: {e}")
            logger.error("This may be due to numerical instability or model misspecification.")
            sys.exit(1)

        # 7. Save results
        output_file = output_dir / "posterior.nc"
        try:
            results.to_netcdf(output_file)
            logger.info(f"Posterior results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {e}")
            sys.exit(1)

        # 8. Summary
        logger.info("Inference completed successfully!")
        logger.info(f"Posterior Summary:\n{results.posterior}")

    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error in main pipeline: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
