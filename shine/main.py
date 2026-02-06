import argparse
import jax
import jax.numpy as jnp
from pathlib import Path
import pickle
import xarray as xr

from shine.config import ConfigHandler
from shine.data import DataLoader
from shine.scene import SceneBuilder
from shine.inference import Inference

def main():
    parser = argparse.ArgumentParser(description="SHINE: SHear INference Environment")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML")
    parser.add_argument("--output", type=str, default=None, help="Output directory")

    args = parser.parse_args()
    
    # 1. Load Config
    config = ConfigHandler.load(args.config)
    
    # Override output path if provided
    if args.output:
        config.output_path = args.output
        
    output_dir = Path(config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Load Data
    observation = DataLoader.load(config)
    
    # Save observation for reference
    jnp.savez(output_dir / "observation.npz", 
              image=observation.image, 
              noise_map=observation.noise_map)
    
    # 3. Build Model
    scene_builder = SceneBuilder(config)
    model_fn = scene_builder.build_model()

    # 4. Run Inference
    print("Starting inference pipeline...")
    rng_key = jax.random.PRNGKey(42)

    # Create inference engine
    engine = Inference(model=model_fn, config=config.inference)

    # Run inference (with optional MAP initialization based on config)
    results = engine.run(
        rng_key=rng_key,
        observed_data=observation.image,
        extra_args={"psf_config": observation.psf_config}
    )

    # Save results
    output_file = output_dir / "posterior.nc"
    results.to_netcdf(output_file)
    print(f"Posterior results saved to {output_file}")

    # Print summary
    print("\nPosterior Summary:")
    print(results.posterior)

if __name__ == "__main__":
    main()
