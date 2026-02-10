import logging
from typing import Any, Callable, Dict, Optional

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta, AutoNormal

from shine.config import InferenceConfig, MAPConfig, NUTSConfig, VIConfig

logger = logging.getLogger(__name__)


class Inference:
    """Inference engine supporting NUTS/MCMC, MAP, and Variational Inference.

    All three methods return az.InferenceData so the downstream pipeline
    (extraction, diagnostics, plots) works uniformly.
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
        init_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run MAP estimation to find maximum a posteriori parameters.

        Args:
            rng_key: JAX random key.
            observed_data: Observed image data.
            extra_args: Extra keyword arguments passed to the model (e.g., psf).
            map_config: MAP configuration (defaults to MAPConfig() if None).
            init_params: Optional dictionary of initial parameter values.
                When provided, ``init_to_value`` is used instead of the
                default ``init_to_feasible`` strategy.  This is useful
                for models whose forward pass can produce NaN at random
                initial points (e.g. galaxy renderers with ellipticity
                constraints).

        Returns:
            Dictionary of MAP parameter estimates.
        """
        if extra_args is None:
            extra_args = {}
        if map_config is None:
            map_config = MAPConfig()

        if init_params is not None:
            init_loc_fn = numpyro.infer.init_to_value(values=init_params)
        else:
            init_loc_fn = numpyro.infer.init_to_feasible()

        guide = AutoDelta(self.model, init_loc_fn=init_loc_fn)
        optimizer = numpyro.optim.Adam(step_size=map_config.learning_rate)
        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO())

        logger.info(f"Running MAP estimation for {map_config.num_steps} steps...")
        svi_result = svi.run(
            rng_key, map_config.num_steps, observed_data=observed_data, **extra_args
        )

        map_estimates = guide.median(svi_result.params)
        logger.info("MAP estimation complete.")
        return map_estimates

    @staticmethod
    def _map_estimates_to_idata(map_estimates: Dict[str, Any]) -> az.InferenceData:
        """Wrap MAP point estimates as InferenceData (1 chain, 1 draw).

        Args:
            map_estimates: Dictionary of MAP parameter estimates.

        Returns:
            ArviZ InferenceData with posterior group containing point estimates.
        """
        posterior_dict = {}
        for name, value in map_estimates.items():
            arr = jnp.atleast_1d(jnp.asarray(value))
            posterior_dict[name] = np.array(arr)[None, None, ...]  # (1, 1, ...)
        idata = az.from_dict(posterior=posterior_dict)
        idata.posterior.attrs["inference_method"] = "map"
        return idata

    def run_vi(
        self,
        rng_key: jax.random.PRNGKey,
        observed_data: jnp.ndarray,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> az.InferenceData:
        """Run Variational Inference with AutoNormal guide.

        Args:
            rng_key: JAX random key.
            observed_data: Observed image data.
            extra_args: Extra keyword arguments passed to the model (e.g., psf).

        Returns:
            ArviZ InferenceData with posterior samples from the fitted guide.
        """
        if extra_args is None:
            extra_args = {}

        vi_config = self.config.vi_config or VIConfig()
        guide = AutoNormal(self.model)
        optimizer = numpyro.optim.Adam(step_size=vi_config.learning_rate)
        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO())

        logger.info(
            f"Running VI: {vi_config.num_steps} steps, "
            f"lr={vi_config.learning_rate}..."
        )
        svi_result = svi.run(
            rng_key, vi_config.num_steps, observed_data=observed_data, **extra_args
        )

        # Draw posterior samples from fitted guide
        sample_key, _ = jax.random.split(rng_key)
        predictive = numpyro.infer.Predictive(
            guide, params=svi_result.params, num_samples=vi_config.num_samples
        )
        vi_samples = predictive(
            sample_key, observed_data=observed_data, **extra_args
        )

        # Wrap as InferenceData (1 chain, N draws), filtering out "obs"
        posterior_dict = {
            k: np.array(v)[None, ...]
            for k, v in vi_samples.items()
            if k != "obs"
        }
        idata = az.from_dict(posterior=posterior_dict)
        idata.posterior.attrs["inference_method"] = "vi"
        idata.posterior.attrs["vi_final_loss"] = float(svi_result.losses[-1])

        logger.info(
            f"VI complete. Final ELBO loss: {svi_result.losses[-1]:.4f}, "
            f"{vi_config.num_samples} posterior samples drawn."
        )
        return idata

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

        nuts_cfg = self.config.nuts_config or NUTSConfig()

        # init_to_uniform is robust for unbounded distributions where init_to_median may fail
        if init_params is not None:
            init_strategy = numpyro.infer.init_to_value(values=init_params)
        else:
            init_strategy = numpyro.infer.init_to_uniform()

        kernel = NUTS(
            self.model,
            dense_mass=nuts_cfg.dense_mass,
            init_strategy=init_strategy,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=nuts_cfg.warmup,
            num_samples=nuts_cfg.samples,
            num_chains=nuts_cfg.chains,
        )

        logger.info(
            f"Running MCMC: {nuts_cfg.warmup} warmup, "
            f"{nuts_cfg.samples} samples, {nuts_cfg.chains} chain(s)..."
        )
        mcmc.run(rng_key, observed_data=observed_data, **extra_args)
        mcmc.print_summary()

        return az.from_numpyro(mcmc)

    def run(
        self,
        rng_key: jax.random.PRNGKey,
        observed_data: jnp.ndarray,
        extra_args: Optional[Dict[str, Any]] = None,
        init_params: Optional[Dict[str, Any]] = None,
    ) -> az.InferenceData:
        """Run inference pipeline, dispatching on the configured method.

        Args:
            rng_key: JAX random key.
            observed_data: Observed image data.
            extra_args: Extra keyword arguments passed to the model (e.g., psf).
            init_params: Optional initial parameter values for MAP estimation.
                Forwarded to :meth:`run_map` when ``method="map"``.

        Returns:
            ArviZ InferenceData object with posterior samples/estimates.
        """
        method = self.config.method

        if method == "map":
            map_cfg = self.config.map_config or MAPConfig()
            estimates = self.run_map(
                rng_key, observed_data, extra_args, map_cfg, init_params
            )
            return self._map_estimates_to_idata(estimates)

        if method == "vi":
            return self.run_vi(rng_key, observed_data, extra_args)

        # NUTS: optional MAP init then MCMC
        nuts_cfg = self.config.nuts_config or NUTSConfig()
        init_params = None
        if nuts_cfg.map_init is not None and nuts_cfg.map_init.enabled:
            map_key, rng_key = jax.random.split(rng_key)
            init_params = self.run_map(
                map_key, observed_data, extra_args, nuts_cfg.map_init
            )
        else:
            logger.info("Skipping MAP initialization.")

        idata = self.run_mcmc(rng_key, observed_data, extra_args, init_params)
        idata.posterior.attrs["inference_method"] = "nuts"
        return idata
