# SHINE Performance Benchmarking & Profiling

This directory contains tooling for diagnosing JAX-level performance in SHINE's
inference pipeline. The benchmark script isolates each stage of the computation
so you can pinpoint exactly where time and memory are being spent.

## Quick Start

```bash
# From the repo root, using the project venv:
.venv/bin/python scripts/benchmark_inference.py

# Short smoke test (5 steps, skip recompilation check):
.venv/bin/python scripts/benchmark_inference.py --num-steps 5 --skip-recompilation-check

# Full profiling run with Perfetto trace + memory profile:
.venv/bin/python scripts/benchmark_inference.py \
    --trace /tmp/shine-trace \
    --memory-profile /tmp/shine-memory.prof
```

## What the Benchmark Measures

The script runs 5 independent benchmarks, each targeting a different layer of
the inference stack:

### 1. Isolated Rendering

Benchmarks a single JAX-GalSim `drawImage` call — galaxy creation, shear,
PSF convolution, and pixel rendering — completely outside NumPyro.

**Why it matters:** This is the innermost kernel. If this is slow, everything
built on top (gradients, SVI, MCMC) will be slow too.

### 2. Forward Model Evaluation

Evaluates the full NumPyro model (prior sampling + rendering + likelihood)
as a single compiled function.

**Why it matters:** Shows the overhead that NumPyro's probabilistic machinery
adds on top of the raw rendering.

### 3. Gradient / SVI Update Step

Times a single SVI optimizer step: forward pass + backward pass (gradient
through the JAX-GalSim renderer) + Adam parameter update.

**Why it matters:** This is the actual unit of work in MAP inference. The
ratio of gradient time to forward time tells you how expensive the backward
pass through the FFT convolution is.

### 4. Full MAP Optimization

Runs `svi.run()` end-to-end, then runs it again to separate JIT compilation
cost from steady-state execution.

**Why it matters:** Reveals overhead from NumPyro's Python-level loop
(loss logging, state management) on top of the raw compiled step time.

### 5. Recompilation Check

Runs a short MAP optimization with `jax.log_compiles()` enabled. Every
XLA compilation event is logged to stderr.

**Why it matters:** Unexpected recompilations inside the optimization loop
are a common JAX performance pitfall. After initial warmup, there should be
zero new compilations.

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num-steps` | 200 | Number of MAP optimization steps |
| `--lr` | 0.1 | Adam learning rate |
| `--fft-size` | 128 | FFT size for rendering (must be power of 2) |
| `--image-size` | 48 | Image size in pixels |
| `--trace DIR` | off | Write a Perfetto trace to DIR |
| `--memory-profile PATH` | off | Write a device memory profile to PATH |
| `--skip-recompilation-check` | off | Skip the recompilation detection pass |

## Reading the Output

### AOT Compilation Breakdown

Each benchmark prints a 4-stage breakdown using JAX's ahead-of-time API:

```
AOT Compilation Breakdown:
  1. Trace:   0.60 s    <-- Python -> jaxpr (tracing overhead)
  2. Lower:   0.22 s    <-- jaxpr -> StableHLO
  3. Compile: 0.99 s    <-- StableHLO -> GPU executable (XLA)
  4. Execute: 0.01 s    <-- actual GPU runtime
```

- **Trace** is Python-side: if this is large, the model has complex
  Python-level logic that runs during tracing (e.g. many `lax.cond` branches
  in JAX-GalSim).
- **Compile** is XLA: this is the one-time cost of compiling the GPU kernel.
  Larger FFT sizes and more complex models increase this.
- **Execute** is the actual GPU work. This is what you pay per step at
  steady state.

### Steady-State Timing

```
Steady-state execution:
  SVI update step: median=0.0248 s, std=0.0007 s (over 9 runs, warmup=0.0245 s)
```

The first call is discarded as warmup. The median of the remaining calls is
the true per-call cost.

### First Run vs Second Run

```
Total wall time:          5.83 s
Second run (cached):      4.10 s
Estimated JIT overhead:   1.73 s (29.7% of first run)
Per-step (steady state):  819 ms   <-- compare this to the isolated step time!
```

If the per-step steady-state time from `svi.run()` is much larger than the
isolated SVI update step, the overhead is in NumPyro's Python loop, not in
the GPU computation.

### GPU Memory

```
[Memory @ after gradient] current=0.0 MB, peak=4586.4 MB, limit=7766 MB
```

Peak memory is the high-water mark. For a 48x48 image this should be small;
if it's in the GB range, the XLA compiler is materializing large intermediate
buffers (common with FFT gradients).

## Perfetto Trace (Timeline Profiling)

Generate a full timeline of CPU and GPU activity:

```bash
.venv/bin/python scripts/benchmark_inference.py --trace /tmp/shine-trace
```

This prints a link to `ui.perfetto.dev` where you can view the trace
interactively. Alternatively:

```bash
# Using TensorBoard (requires: pip install tensorboard xprof)
tensorboard --logdir=/tmp/shine-trace
# Then open http://localhost:6006 -> Profile tab
```

**What to look for in the trace:**

- **Gaps between GPU ops:** These are Python/host overhead — the GPU is idle
  waiting for the next kernel launch.
- **HtoD / DtoH copies:** Data transfers between host and device. These
  should be rare during the optimization loop.
- **Repeated XLA compilation events:** If you see compilation happening
  inside the optimization loop (not just at startup), something is triggering
  retracing.
- **Kernel durations:** Click individual GPU kernels to see their duration.
  FFT kernels should dominate.

**Navigation:** WASD keys to pan and zoom (game-style controls). Click any
event to see its details.

## Device Memory Profile

Capture a snapshot of what's allocated on the GPU:

```bash
.venv/bin/python scripts/benchmark_inference.py \
    --memory-profile /tmp/shine-memory.prof
```

Visualize with pprof (requires Go):

```bash
go install github.com/google/pprof@latest
pprof --web /tmp/shine-memory.prof
```

This generates a call-graph where node sizes correspond to memory allocations.
Note: allocations inside `jax.jit`-compiled functions are attributed to the
function as a whole, not to individual operations within it.

## Useful JAX Debug Flags

These can be set in your script or notebook before running inference. They
add overhead, so use them for debugging, not production.

```python
import jax

# Log every JIT compilation (see if the optimization loop recompiles)
jax.config.update("jax_log_compiles", True)

# Explain WHY a cache miss happened (shape change? new function?)
jax.config.update("jax_explain_cache_misses", True)

# Warn on implicit host<->device transfers (print, np.array, if x > 0)
jax.config.update("jax_transfer_guard", "log")

# Detect leaked tracers (subtle bug causing silent recompilation)
jax.config.update("jax_check_tracer_leaks", True)

# Cache compiled XLA artifacts to disk (avoids recompilation across runs)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
```

## Common Performance Patterns

### Pattern: Compilation dominates (normal on first call)

```
Compile: 3.4 s
Execute: 0.03 s
```

This is expected. XLA compiles a specialized GPU kernel for the exact
shapes and dtypes of your inputs. Subsequent calls with the same shapes
hit the cache. If you run inference in a loop (e.g. over realizations),
only the first iteration pays this cost.

**Mitigation:** Use `jax_compilation_cache_dir` to persist the cache
across Python sessions.

### Pattern: `svi.run()` much slower than isolated step

```
Isolated SVI step:   25 ms
svi.run() per step: 400+ ms
```

NumPyro's `svi.run()` has Python-level overhead per step (loss array
bookkeeping, progress bar, state packing/unpacking). For fast inner
kernels this overhead can dominate.

**Mitigation:** For MAP with simple models, consider writing a manual
optimization loop with `svi.update()` inside a `jax.lax.scan`, which
compiles the entire loop into a single GPU kernel.

### Pattern: Gradient much slower than forward

```
Forward:  0.6 ms
Gradient: 25 ms  (40x)
```

The backward pass through FFT-based rendering (JAX-GalSim's
`drawImage` -> PSF convolution) requires materializing and transposing
large intermediate buffers. This is inherent to automatic differentiation
through FFTs.

**Mitigations:**
- Reduce `fft_size` (e.g. 64 instead of 128) if the image allows it.
- Disable `jax_enable_x64` for ~2x speedup if float32 precision is
  acceptable for your use case.
- For MCMC, consider variational inference (VI/SVI) which requires fewer
  gradient evaluations than NUTS.

### Pattern: High peak memory for small images

```
Image: 48x48
Peak GPU memory: 4.6 GB
```

XLA materializes intermediate FFT buffers during gradient computation.
With `fft_size=128` and float64, the gradient tape stores multiple
128x128 complex128 arrays.

**Mitigations:**
- Use float32 (`jax_enable_x64 = False`).
- Reduce `fft_size`.
- Use `jax.checkpoint` on the rendering function to trade recomputation
  for memory (not yet implemented in SHINE).

### Pattern: Many small `jit(cond)` compilations at startup

```
49 Compiling jit(cond) events
```

JAX-GalSim uses `jax.lax.cond` for runtime branching (e.g. choosing
FFT padding strategies). Each unique branch signature compiles
separately. These are cached after the first model evaluation and are
not a steady-state concern.

**Mitigation:** None needed — this is one-time startup cost. Use
`jax_compilation_cache_dir` to persist across sessions.

## Comparing Configurations

Run the benchmark with different parameters to find the sweet spot:

```bash
# Baseline
.venv/bin/python scripts/benchmark_inference.py --fft-size 128

# Smaller FFT
.venv/bin/python scripts/benchmark_inference.py --fft-size 64

# Larger image
.venv/bin/python scripts/benchmark_inference.py --image-size 64 --fft-size 256

# Without 64-bit (edit jax_enable_x64 in the script)
# Expect ~2x speedup and ~2x less memory
```

## Reference: Typical Numbers (RTX 3080, 48x48 image, fft=128, float64)

| Stage | Time |
|-------|------|
| Rendering (forward) | ~0.6 ms |
| Rendering (gradient w.r.t. g1,g2) | ~24 ms |
| Full SVI update step (compiled) | ~25 ms |
| `svi.run()` per step (Python loop) | ~400-800 ms |
| XLA compilation (SVI step) | ~3.4 s (one-time) |
| Peak GPU memory | ~4.6 GB |
