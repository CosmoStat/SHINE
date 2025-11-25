import jax
import numpyro
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from typing import Dict, Any
import arviz as az

class HMCInference:
    def __init__(self, model, num_warmup=500, num_samples=1000, num_chains=1, dense_mass=False):
        self.model = model
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.dense_mass = dense_mass

    def run(self, rng_key, observed_data, extra_args=None):
        if extra_args is None:
            extra_args = {}
            
        kernel = NUTS(self.model, dense_mass=self.dense_mass)
        mcmc = MCMC(kernel, num_warmup=self.num_warmup, num_samples=self.num_samples, num_chains=self.num_chains)
        
        mcmc.run(rng_key, observed_data=observed_data, **extra_args)
        mcmc.print_summary()
        
        # Convert to ArviZ InferenceData
        return az.from_numpyro(mcmc)

class MAPInference:
    def __init__(self, model, num_steps=1000, learning_rate=1e-2):
        self.model = model
        self.num_steps = num_steps
        self.learning_rate = learning_rate

    def run(self, rng_key, observed_data, extra_args=None):
        if extra_args is None:
            extra_args = {}

        guide = AutoDelta(self.model)
        optimizer = numpyro.optim.Adam(step_size=self.learning_rate)
        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO())
        
        svi_result = svi.run(rng_key, self.num_steps, observed_data=observed_data, **extra_args)
        
        params = svi_result.params
        # The params from AutoDelta are the MAP estimates (in unconstrained space usually, 
        # but AutoDelta returns constrained values if using `median` init or similar, 
        # actually AutoDelta parameters are the values themselves).
        
        # We need to sample from the guide to get the values in the proper structure if needed,
        # but for AutoDelta, params contains the values.
        # Note: AutoDelta names parameters with `_auto_loc` suffix sometimes or keeps original names depending on version.
        # Let's check the guide median.
        
        return guide.median(params)
