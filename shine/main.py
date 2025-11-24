import argparse
import jax
import jax.numpy as jnp
from pathlib import Path
import pickle
import xarray as xr

from shine.config import ConfigHandler
from shine.data import DataLoader
from shine.scene import SceneBuilder
from shine.inference import HMCInference, MAPInference

def main():
    parser = argparse.ArgumentParser(description="SHINE: SHear INference Environment")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML")
    parser.add_argument("--mode", type=str, default="hmc", choices=["hmc", "map"], help="Inference mode: hmc or map")
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
    print(f"Starting inference in {args.mode.upper()} mode...")
    rng_key = jax.random.PRNGKey(42)
    
    if args.mode == "hmc":
        engine = HMCInference(
            model=model_fn,
            num_warmup=config.inference.warmup,
            num_samples=config.inference.samples,
            num_chains=config.inference.chains,
            dense_mass=config.inference.dense_mass
        )
        results = engine.run(
            rng_key=rng_key,
            observed_data=observation.image,
            extra_args={"psf_config": observation.psf_config}
        )
        
        # Save results
        output_file = output_dir / "posterior.nc"
        results.to_netcdf(output_file)
        print(f"Results saved to {output_file}")
        
        # Print summary
        print(results.posterior)
        
    elif args.mode == "map":
        engine = MAPInference(model=model_fn)
        results = engine.run(
            rng_key=rng_key,
            observed_data=observation.image,
            extra_args={"psf_config": observation.psf_config}
        )
        
        # Save MAP estimates
        output_file = output_dir / "map_results.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        print(f"MAP estimates saved to {output_file}")
        
        print("MAP Estimates:")
        for k, v in results.items():
            print(f"{k}: {v}")
            
        # Calculate Residuals
        # We need to re-render the scene with MAP parameters
        # This is a bit tricky without exposing the render function directly from the model.
        # A clean way is to use numpyro.handlers.substitute and trace the model.
        
        from numpyro.handlers import substitute, trace, seed
        
        # We need to run the model with the MAP parameters
        # The model function expects (observed_data, psf)
        # We pass None for observed_data to get the model prediction (if the model returns it or we capture the deterministic site)
        # But our model samples 'obs' at the end.
        # We can capture the 'obs' site mean (which is the model image).
        
        # However, our model defines 'obs' as Normal(model_image, sigma).
        # We want 'model_image'.
        # We didn't expose 'model_image' as a deterministic site in the builder.
        # Let's modify the builder to expose it, OR we can just inspect the 'obs' distribution mean in the trace.
        
        model_with_params = substitute(model_fn, results)
        
        # Trace the model execution
        # We need to pass the same args as during inference
        traced_model = trace(seed(model_with_params, rng_key))
        trace_out = traced_model.get_trace(observed_data=None, psf_config=observation.psf_config)
        
        # The 'obs' site contains the distribution with the mean we want
        obs_node = trace_out['obs']
        model_image = obs_node['fn'].mean
        
        residual = observation.image - model_image
        chi2 = jnp.sum((residual**2) / observation.noise_map)
        dof = observation.image.size # approx
        reduced_chi2 = chi2 / dof
        
        print(f"Reduced Chi2: {reduced_chi2:.4f}")
        
        jnp.savez(output_dir / "residuals.npz", 
                  data=observation.image, 
                  model=model_image, 
                  residual=residual,
                  chi2=chi2)

if __name__ == "__main__":
    main()
