import jax
import numpyro
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from typing import Dict, Any, Callable, Optional
import arviz as az
from shine.config import InferenceConfig, MAPConfig


class Inference:
    """
    Main inference engine supporting multiple inference methods (HMC/MCMC, MAP).

    This class handles both full posterior inference via HMC and MAP estimation.
    MAP can be used as an initialization step before running MCMC chains.
    """

    def __init__(self, model: Callable, config: InferenceConfig):
        """
        Initialize the inference engine.

        Args:
            model: NumPyro model function (the forward generative model)
            config: Inference configuration
        """
        self.model = model
        self.config = config

    def run_map(self, rng_key, observed_data, extra_args=None, map_config: Optional[MAPConfig] = None) -> Dict[str, Any]:
        """
        Run MAP estimation to find maximum a posteriori parameters.

        Args:
            rng_key: JAX random key
            observed_data: Observed image data
            extra_args: Extra arguments to pass to the model (e.g., psf_config)
            map_config: MAP configuration (if None, uses default)

        Returns:
            Dictionary of MAP parameter estimates
        """
        if extra_args is None:
            extra_args = {}

        if map_config is None:
            map_config = MAPConfig()

        guide = AutoDelta(self.model)
        optimizer = numpyro.optim.Adam(step_size=map_config.learning_rate)
        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO())

        print(f"Running MAP estimation for {map_config.num_steps} steps...")
        svi_result = svi.run(rng_key, map_config.num_steps, observed_data=observed_data, **extra_args)

        params = svi_result.params
        map_estimates = guide.median(params)

        print("MAP estimation complete.")
        return map_estimates

    def run_mcmc(self, rng_key, observed_data, extra_args=None, init_params: Optional[Dict[str, Any]] = None) -> az.InferenceData:
        """
        Run MCMC inference using NUTS sampler.

        Args:
            rng_key: JAX random key
            observed_data: Observed image data
            extra_args: Extra arguments to pass to the model (e.g., psf_config)
            init_params: Optional initial parameters (e.g., from MAP)

        Returns:
            ArviZ InferenceData object with posterior samples
        """
        if extra_args is None:
            extra_args = {}

        kernel = NUTS(self.model, dense_mass=self.config.dense_mass, init_strategy=numpyro.infer.init_to_value(values=init_params) if init_params else numpyro.infer.init_to_median())
        mcmc = MCMC(kernel, num_warmup=self.config.warmup, num_samples=self.config.samples, num_chains=self.config.chains)

        print(f"Running MCMC inference: {self.config.warmup} warmup, {self.config.samples} samples, {self.config.chains} chains...")
        mcmc.run(rng_key, observed_data=observed_data, **extra_args)
        mcmc.print_summary()

        # Convert to ArviZ InferenceData
        return az.from_numpyro(mcmc)

    def run(self, rng_key, observed_data, extra_args=None) -> az.InferenceData:
        """
        Run full inference pipeline with optional MAP initialization.

        If MAP initialization is enabled in config, runs MAP first to find
        good starting points for MCMC chains.

        Args:
            rng_key: JAX random key
            observed_data: Observed image data
            extra_args: Extra arguments to pass to the model (e.g., psf_config)

        Returns:
            ArviZ InferenceData object with posterior samples
        """
        init_params = None

        # Run MAP initialization if enabled
        if self.config.map_init is not None and self.config.map_init.enabled:
            map_key, mcmc_key = jax.random.split(rng_key)
            init_params = self.run_map(map_key, observed_data, extra_args, self.config.map_init)
            rng_key = mcmc_key
        else:
            print("Skipping MAP initialization.")

        # Run MCMC
        return self.run_mcmc(rng_key, observed_data, extra_args, init_params)
