#!/usr/bin/env python
"""JAX-level performance benchmark for SHINE inference.

Breaks down where time is spent during MAP inference:
  - JIT tracing / lowering / compilation vs. execution
  - Forward model (rendering) vs. gradient computation
  - SVI step overhead
  - GPU memory usage at each stage
  - Optional Perfetto trace for timeline analysis

Usage:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --trace /tmp/shine-trace
    python scripts/benchmark_inference.py --num-steps 500 --fft-size 256
"""

import argparse
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

# Enable 64-bit to match the notebook
jax.config.update("jax_enable_x64", True)

import jax_galsim as galsim
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta

from shine.config import (
    DistributionConfig,
    EllipticityConfig,
    GalaxyConfig,
    ImageConfig,
    InferenceConfig,
    MAPConfig,
    NoiseConfig,
    PositionConfig,
    PSFConfig,
    ShearConfig,
    ShineConfig,
)
from shine.data import DataLoader
from shine.scene import SceneBuilder
from shine.validation.simulation import generate_biased_observation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _device_info() -> str:
    """Return a string describing the active JAX device."""
    devs = jax.devices()
    info = [f"{d.platform.upper()}:{d.device_kind}" for d in devs]
    return ", ".join(info)


def _gpu_memory_stats() -> Optional[Dict[str, float]]:
    """Return GPU memory stats in MB, or None if not on GPU."""
    dev = jax.devices()[0]
    if dev.platform != "gpu":
        return None
    try:
        stats = dev.memory_stats()
        if stats is None:
            return None
        return {
            "peak_usage_mb": stats.get("peak_bytes_in_use", 0) / 1e6,
            "current_usage_mb": stats.get("bytes_in_use", 0) / 1e6,
            "bytes_limit_mb": stats.get("bytes_limit", 0) / 1e6,
        }
    except Exception:
        return None


def _print_memory(label: str) -> None:
    """Print GPU memory usage with a label."""
    stats = _gpu_memory_stats()
    if stats is not None:
        print(
            f"  [Memory @ {label}] "
            f"current={stats['current_usage_mb']:.1f} MB, "
            f"peak={stats['peak_usage_mb']:.1f} MB, "
            f"limit={stats['bytes_limit_mb']:.0f} MB"
        )


@contextmanager
def _timer(label: str):
    """Context manager that prints elapsed wall-clock time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  {label}: {elapsed:.4f} s")


def _time_fn(fn, *args, n_repeat: int = 5, label: str = "", **kwargs) -> float:
    """Time a function, returning median wall-clock seconds.

    The first call is treated as warmup (to absorb any residual lazy init).
    """
    times = []
    for i in range(n_repeat):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    times_no_warmup = times[1:]
    median = float(np.median(times_no_warmup))
    std = float(np.std(times_no_warmup))
    if label:
        print(
            f"  {label}: median={median:.4f} s, std={std:.4f} s "
            f"(over {len(times_no_warmup)} runs, warmup={times[0]:.4f} s)"
        )
    return median


# ---------------------------------------------------------------------------
# Build config & data (same as notebook)
# ---------------------------------------------------------------------------

def build_config(num_steps: int = 200, lr: float = 0.1, fft_size: int = 128,
                 image_size: int = 48) -> ShineConfig:
    return ShineConfig(
        image=ImageConfig(
            pixel_scale=0.263,
            size_x=image_size,
            size_y=image_size,
            n_objects=1,
            fft_size=fft_size,
            noise=NoiseConfig(type="Gaussian", sigma=1e-6),
        ),
        psf=PSFConfig(type="Moffat", sigma=0.9, beta=2.5),
        gal=GalaxyConfig(
            type="Exponential",
            flux=1.0,
            half_light_radius=0.5,
            ellipticity=EllipticityConfig(type="E1E2", e1=0.0, e2=0.0),
            shear=ShearConfig(
                type="G1G2",
                g1=DistributionConfig(type="Normal", mean=0.0, sigma=0.05),
                g2=DistributionConfig(type="Normal", mean=0.0, sigma=0.05),
            ),
            position=PositionConfig(
                type="Uniform", x_min=23.5, x_max=24.5, y_min=23.5, y_max=24.5,
            ),
        ),
        inference=InferenceConfig(
            method="map",
            map_config=MAPConfig(num_steps=num_steps, learning_rate=lr),
            rng_seed=42,
        ),
    )


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_forward_model(model_fn, observed_data, psf, rng_key):
    """Benchmark a single forward-model evaluation (no gradients)."""
    print("\n=== Forward Model (single evaluation) ===")

    # Build a callable that evaluates the model log-density
    from numpyro.infer.util import log_density

    guide = AutoDelta(model_fn)
    # Initialize guide to get valid parameter shapes
    svi = SVI(model_fn, guide, numpyro.optim.Adam(0.01), loss=Trace_ELBO())
    svi_state = svi.init(rng_key, observed_data=observed_data, psf=psf)
    params = svi.get_params(svi_state)

    def forward(params):
        # Evaluate ELBO loss (calls forward model internally)
        loss = svi.evaluate(svi_state, observed_data=observed_data, psf=psf)
        return loss

    # --- Trace / Lower / Compile / Execute breakdown (AOT API) ---
    print("\n  AOT Compilation Breakdown:")
    jitted_forward = jax.jit(forward)

    with _timer("1. Trace (jaxpr)"):
        traced = jitted_forward.trace(params)

    with _timer("2. Lower (StableHLO)"):
        lowered = traced.lower()

    with _timer("3. Compile (XLA)"):
        compiled = lowered.compile()

    # Show compiled cost analysis if available
    try:
        cost = compiled.cost_analysis()
        if cost and len(cost) > 0:
            c = cost[0]
            flops = c.get("flops", "N/A")
            print(f"  XLA cost analysis: flops={flops}")
    except Exception:
        pass

    with _timer("4. Execute (compiled)"):
        result = compiled(params)
        jax.block_until_ready(result)

    _print_memory("after forward")

    # --- Repeated execution timing ---
    print("\n  Steady-state execution:")
    _time_fn(compiled, params, n_repeat=10, label="Forward eval")

    return compiled, params


def bench_gradient(model_fn, observed_data, psf, rng_key):
    """Benchmark gradient computation (forward + backward)."""
    print("\n=== Gradient Computation ===")

    guide = AutoDelta(model_fn)
    svi = SVI(model_fn, guide, numpyro.optim.Adam(0.01), loss=Trace_ELBO())
    svi_state = svi.init(rng_key, observed_data=observed_data, psf=psf)
    params = svi.get_params(svi_state)

    loss_fn = lambda p: svi.evaluate(
        svi.init(rng_key, observed_data=observed_data, psf=psf),
        observed_data=observed_data, psf=psf,
    )

    grad_fn = jax.jit(jax.grad(lambda p: svi.evaluate(
        svi_state, observed_data=observed_data, psf=psf
    )))

    # Direct SVI update step (this is what MAP actually calls each iteration)
    update_fn = jax.jit(lambda state: svi.update(state, observed_data=observed_data, psf=psf))

    print("\n  AOT Compilation Breakdown (SVI update step):")

    with _timer("1. Trace"):
        traced = update_fn.trace(svi_state)

    with _timer("2. Lower"):
        lowered = traced.lower()

    with _timer("3. Compile"):
        compiled = lowered.compile()

    with _timer("4. Execute"):
        result = compiled(svi_state)
        jax.block_until_ready(result)

    _print_memory("after gradient")

    # Steady-state
    print("\n  Steady-state execution:")
    _time_fn(compiled, svi_state, n_repeat=10, label="SVI update step")

    return compiled, svi_state


def bench_full_map(model_fn, observed_data, psf, rng_key, config):
    """Benchmark the full MAP optimization end-to-end."""
    print("\n=== Full MAP Optimization ===")
    map_cfg = config.inference.map_config
    print(f"  Steps: {map_cfg.num_steps}, LR: {map_cfg.learning_rate}")

    _print_memory("before MAP")

    guide = AutoDelta(model_fn)
    optimizer = numpyro.optim.Adam(step_size=map_cfg.learning_rate)
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    # Time the full svi.run() which includes JIT compilation + all steps
    start = time.perf_counter()
    svi_result = svi.run(
        rng_key, map_cfg.num_steps,
        observed_data=observed_data, psf=psf,
        progress_bar=False,
    )
    jax.block_until_ready(svi_result.params)
    total = time.perf_counter() - start

    _print_memory("after MAP")

    print(f"\n  Total wall time: {total:.4f} s")
    print(f"  Per-step average: {total / map_cfg.num_steps * 1000:.2f} ms")
    print(f"  Final loss: {float(svi_result.losses[-1]):.6f}")

    # Estimate compilation vs execution overhead:
    # Run again (already compiled) to measure pure execution
    start2 = time.perf_counter()
    svi_result2 = svi.run(
        rng_key, map_cfg.num_steps,
        observed_data=observed_data, psf=psf,
        progress_bar=False,
    )
    jax.block_until_ready(svi_result2.params)
    exec_only = time.perf_counter() - start2

    jit_overhead = total - exec_only
    print(f"\n  Second run (cached): {exec_only:.4f} s")
    print(f"  Estimated JIT overhead: {jit_overhead:.4f} s "
          f"({jit_overhead / total * 100:.1f}% of first run)")
    print(f"  Per-step (steady state): {exec_only / map_cfg.num_steps * 1000:.2f} ms")

    return svi_result


def bench_rendering_isolation(config, psf):
    """Benchmark the JAX-GalSim rendering kernel in isolation."""
    print("\n=== Isolated Rendering Benchmark ===")

    img_cfg = config.image
    gal_cfg = config.gal
    fft_size = img_cfg.fft_size
    gsparams = galsim.GSParams(
        maximum_fft_size=fft_size, minimum_fft_size=fft_size
    )

    def render(g1, g2, flux, hlr, x, y):
        gal = galsim.Exponential(
            flux=flux, half_light_radius=hlr, gsparams=gsparams
        )
        shear = galsim.Shear(g1=g1, g2=g2)
        gal = gal.shear(shear)
        final = galsim.Convolve([gal, psf], gsparams=gsparams)
        return final.drawImage(
            nx=img_cfg.size_x, ny=img_cfg.size_y,
            scale=img_cfg.pixel_scale,
            offset=(x - img_cfg.size_x / 2 + 0.5,
                    y - img_cfg.size_y / 2 + 0.5),
        ).array

    # Typical parameter values
    g1, g2 = jnp.float64(0.01), jnp.float64(0.0)
    flux, hlr = jnp.float64(1.0), jnp.float64(0.5)
    x, y = jnp.float64(24.0), jnp.float64(24.0)

    jit_render = jax.jit(render)

    # AOT breakdown
    print("\n  AOT Compilation Breakdown:")
    with _timer("1. Trace"):
        traced = jit_render.trace(g1, g2, flux, hlr, x, y)
    with _timer("2. Lower"):
        lowered = traced.lower()
    with _timer("3. Compile"):
        compiled = lowered.compile()
    with _timer("4. Execute"):
        img = compiled(g1, g2, flux, hlr, x, y)
        jax.block_until_ready(img)

    print(f"  Output shape: {img.shape}, dtype: {img.dtype}")
    _print_memory("after render")

    # Steady-state
    print("\n  Steady-state execution:")
    _time_fn(compiled, g1, g2, flux, hlr, x, y, n_repeat=20, label="Render")

    # Gradient of rendering
    print("\n  Gradient of rendering w.r.t. (g1, g2):")
    grad_render = jax.jit(jax.grad(lambda g1, g2: jnp.sum(render(g1, g2, flux, hlr, x, y)),
                                   argnums=(0, 1)))
    # Warmup
    dg1, dg2 = grad_render(g1, g2)
    jax.block_until_ready((dg1, dg2))

    _time_fn(grad_render, g1, g2, n_repeat=20, label="Grad render")


def check_recompilation(model_fn, observed_data, psf, rng_key, config):
    """Run MAP with compile logging to detect unexpected recompilations."""
    print("\n=== Recompilation Check ===")
    print("  Running MAP with jax_log_compiles=True...")
    print("  (Compilations after the first few are suspicious)\n")

    map_cfg = config.inference.map_config

    with jax.log_compiles():
        guide = AutoDelta(model_fn)
        optimizer = numpyro.optim.Adam(step_size=map_cfg.learning_rate)
        svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())
        svi_result = svi.run(
            rng_key, min(map_cfg.num_steps, 20),  # Short run for checking
            observed_data=observed_data, psf=psf,
            progress_bar=False,
        )
        jax.block_until_ready(svi_result.params)

    print("  (Check stderr for compilation logs above)")


def generate_trace(model_fn, observed_data, psf, rng_key, config, trace_dir):
    """Generate a Perfetto trace of the full MAP run."""
    print(f"\n=== Generating Perfetto Trace → {trace_dir} ===")

    map_cfg = config.inference.map_config

    with jax.profiler.trace(trace_dir, create_perfetto_link=True):
        guide = AutoDelta(model_fn)
        optimizer = numpyro.optim.Adam(step_size=map_cfg.learning_rate)
        svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())
        svi_result = svi.run(
            rng_key, map_cfg.num_steps,
            observed_data=observed_data, psf=psf,
            progress_bar=False,
        )
        jax.block_until_ready(svi_result.params)

    print("  Trace written. Open the Perfetto link above, or run:")
    print(f"    tensorboard --logdir={trace_dir}")


def dump_memory_profile(model_fn, observed_data, psf, rng_key, config, out_path):
    """Save a device memory profile after a MAP run."""
    print(f"\n=== Device Memory Profile → {out_path} ===")

    map_cfg = config.inference.map_config
    guide = AutoDelta(model_fn)
    optimizer = numpyro.optim.Adam(step_size=map_cfg.learning_rate)
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(
        rng_key, map_cfg.num_steps,
        observed_data=observed_data, psf=psf,
        progress_bar=False,
    )
    jax.block_until_ready(svi_result.params)

    jax.profiler.save_device_memory_profile(out_path)
    print(f"  Written. Visualize with: pprof --web {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SHINE inference at the JAX level"
    )
    parser.add_argument("--num-steps", type=int, default=200,
                        help="MAP optimization steps (default: 200)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate (default: 0.1)")
    parser.add_argument("--fft-size", type=int, default=128,
                        help="FFT size for rendering (default: 128)")
    parser.add_argument("--image-size", type=int, default=48,
                        help="Image size in pixels (default: 48)")
    parser.add_argument("--trace", type=str, default=None,
                        help="Directory for Perfetto trace output")
    parser.add_argument("--memory-profile", type=str, default=None,
                        help="Path for device memory profile (.prof)")
    parser.add_argument("--skip-recompilation-check", action="store_true",
                        help="Skip the recompilation detection pass")
    args = parser.parse_args()

    # --- Setup ---
    print("=" * 70)
    print("SHINE Inference Benchmark")
    print("=" * 70)
    print(f"Device: {_device_info()}")
    print(f"JAX version: {jax.__version__}")
    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    print(f"Parameters: steps={args.num_steps}, lr={args.lr}, "
          f"fft={args.fft_size}, image={args.image_size}x{args.image_size}")

    _print_memory("startup")

    config = build_config(
        num_steps=args.num_steps, lr=args.lr,
        fft_size=args.fft_size, image_size=args.image_size,
    )

    # Generate a single observation
    print("\nGenerating synthetic observation...")
    sim = generate_biased_observation(config, g1_true=0.01, g2_true=0.0, seed=100)
    observed_data = sim.observation.image
    psf = sim.observation.psf_model
    print(f"  Image shape: {observed_data.shape}, dtype: {observed_data.dtype}")
    _print_memory("after data generation")

    # Build model
    scene = SceneBuilder(config)
    model_fn = scene.build_model()
    rng_key = jax.random.PRNGKey(42)

    # --- Run benchmarks ---
    bench_rendering_isolation(config, psf)
    bench_forward_model(model_fn, observed_data, psf, rng_key)
    bench_gradient(model_fn, observed_data, psf, rng_key)
    bench_full_map(model_fn, observed_data, psf, rng_key, config)

    if not args.skip_recompilation_check:
        check_recompilation(model_fn, observed_data, psf, rng_key, config)

    if args.trace:
        generate_trace(model_fn, observed_data, psf, rng_key, config, args.trace)

    if args.memory_profile:
        dump_memory_profile(model_fn, observed_data, psf, rng_key, config,
                            args.memory_profile)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    _print_memory("final")
    print("\nTips if inference is slow:")
    print("  1. Check the AOT breakdown — if Compile >> Execute, you're")
    print("     paying a large one-time JIT cost (normal for first run).")
    print("  2. Check recompilation logs — repeated compilations in the")
    print("     optimization loop indicate a tracing bug.")
    print("  3. If rendering dominates, try reducing --fft-size (power of 2).")
    print("  4. If gradients are slow relative to forward, consider VI")
    print("     (variational inference) instead of MAP/NUTS.")
    print("  5. Run with --trace /tmp/shine-trace for a full timeline.")
    print("  6. Disable x64 (remove jax_enable_x64) for ~2x speedup if")
    print("     precision allows.")


if __name__ == "__main__":
    main()
