# shine.inference

Bayesian inference engine supporting NUTS/MCMC, MAP, and Variational Inference.

Dispatches on `InferenceConfig.method` to run one of three inference paths.
All methods return ArviZ `InferenceData` so the downstream pipeline works
uniformly.

::: shine.inference
