# shine.config

Configuration handling with Pydantic models.

Parses YAML configuration files and validates all parameters. Distribution
parameters (Normal, LogNormal, Uniform) are automatically treated as latent
variables for Bayesian inference. Distributions can use `center: "catalog"` to
resolve their location parameter from per-source catalog data at runtime.
Position priors support both `Uniform` (absolute pixel positions) and `Offset`
(small offsets from catalog positions) modes.

The `InferenceConfig` supports three inference methods (NUTS, MAP, VI) with
method-specific config blocks (`NUTSConfig`, `MAPConfig`, `VIConfig`).

::: shine.config
