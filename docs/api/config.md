# shine.config

Configuration handling with Pydantic models.

Parses YAML configuration files and validates all parameters. Distribution
parameters (Normal, LogNormal, Uniform) are automatically treated as latent
variables for Bayesian inference. The `InferenceConfig` supports three
inference methods (NUTS, MAP, VI) with method-specific config blocks
(`NUTSConfig`, `MAPConfig`, `VIConfig`).

::: shine.config
