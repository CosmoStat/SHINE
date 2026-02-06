import logging
from typing import Any, Callable, Dict, Optional

import arviz as az
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta

from shine.config import InferenceConfig, MAPConfig

logger = logging.getLogger(__name__)


class Inference:
    """Inference engine supporting HMC/MCMC and optional MAP initialization.

    MAP estimation can be used as an initialization step before running
    MCMC chains to improve convergence.
    """

    def __init__(self, model: Callable, config: InferenceConfig) -> None:
        """Initialize the inference engine.

        Args:
            model: NumPyro model function (the forward generative model).
            config: Inference configuration.
        """
        self.model = model
        self.config = config

    def run_map(
        self,
        rng_key: jax.random.PRNGKey,
        observed_data: jnp.ndarray,
        extra_args: Optional[Dict[str, Any]] = None,
        map_config: Optional[MAPConfig] = None,
    ) -> Dict[str, Any]:
        """Run MAP estimation to find maximum a posteriori parameters.

        Args:
            rng_key: JAX random key.
            observed_data: Observed image data.
            extra_args: Extra keyword arguments passed to the model (e.g., psf).
            map_config: MAP configuration (defaults to MAPConfig() if None).

        Returns:
            Dictionary of MAP parameter estimates.
        """
        if extra_args is None:
            extra_args = {}
        if map_config is None:
            map_config = MAPConfig()

        guide = AutoDelta(self.model)
        optimizer = numpyro.optim.Adam(step_size=map_config.learning_rate)
        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO())

        logger.info(f"Running MAP estimation for {map_config.num_steps} steps...")
        svi_result = svi.run(
            rng_key, map_config.num_steps, observed_data=observed_data, **extra_args
        )

        map_estimates = guide.median(svi_result.params)
        logger.info("MAP estimation complete.")
        return map_estimates

    def run_mcmc(
        self,
        rng_key: jax.random.PRNGKey,
        observed_data: jnp.ndarray,
        extra_args: Optional[Dict[str, Any]] = None,
        init_params: Optional[Dict[str, Any]] = None,
    ) -> az.InferenceData:
        """Run MCMC inference using the NUTS sampler.

        Args:
            rng_key: JAX random key.
            observed_data: Observed image data.
            extra_args: Extra keyword arguments passed to the model (e.g., psf).
            init_params: Optional initial parameters (e.g., from MAP estimation).

        Returns:
            ArviZ InferenceData object with posterior samples.
        """
        if extra_args is None:
            extra_args = {}

        # init_to_uniform is robust for unbounded distributions where init_to_median may fail
        if init_params is not None:
            init_strategy = numpyro.infer.init_to_value(values=init_params)
        else:
            init_strategy = numpyro.infer.init_to_uniform()

        kernel = NUTS(
            self.model,
            dense_mass=self.config.dense_mass,
            init_strategy=init_strategy,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=self.config.warmup,
            num_samples=self.config.samples,
            num_chains=self.config.chains,
        )

        logger.info(
            f"Running MCMC: {self.config.warmup} warmup, "
            f"{self.config.samples} samples, {self.config.chains} chain(s)..."
        )
        mcmc.run(rng_key, observed_data=observed_data, **extra_args)
        mcmc.print_summary()

        return az.from_numpyro(mcmc)

    def run(
        self,
        rng_key: jax.random.PRNGKey,
        observed_data: jnp.ndarray,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> az.InferenceData:
        """Run full inference pipeline with optional MAP initialization.

        If MAP initialization is enabled in config, runs MAP first to find
        good starting points, then runs MCMC.

        Args:
            rng_key: JAX random key.
            observed_data: Observed image data.
            extra_args: Extra keyword arguments passed to the model (e.g., psf).

        Returns:
            ArviZ InferenceData object with posterior samples.
        """
        init_params = None

        map_init = self.config.map_init
        if map_init is not None and map_init.enabled:
            map_key, rng_key = jax.random.split(rng_key)
            init_params = self.run_map(
                map_key, observed_data, extra_args, map_init
            )
        else:
            logger.info("Skipping MAP initialization.")

        return self.run_mcmc(rng_key, observed_data, extra_args, init_params)
