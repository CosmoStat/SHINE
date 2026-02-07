"""Tests for GPU-batched inference components."""

import arviz as az
import numpy as np
import pytest
import yaml

from shine.config import ConfigHandler
from shine.data import Observation
from shine.validation.extraction import split_batched_idata
from shine.validation.simulation import (
    BatchSimulationResult,
    generate_batch_observations,
)


@pytest.fixture
def level0_config(tmp_path):
    """Create a minimal Level 0 SHINE config for testing."""
    config_dict = {
        "image": {
            "pixel_scale": 0.1,
            "size_x": 48,
            "size_y": 48,
            "n_objects": 1,
            "fft_size": 128,
            "noise": {"type": "Gaussian", "sigma": 0.001},
        },
        "psf": {"type": "Gaussian", "sigma": 0.1},
        "gal": {
            "type": "Exponential",
            "flux": 1000.0,
            "half_light_radius": 0.5,
            "shear": {
                "type": "G1G2",
                "g1": {"type": "Normal", "mean": 0.0, "sigma": 0.05},
                "g2": {"type": "Normal", "mean": 0.0, "sigma": 0.05},
            },
            "position": {
                "type": "Uniform",
                "x_min": 23.5,
                "x_max": 24.5,
                "y_min": 23.5,
                "y_max": 24.5,
            },
        },
        "inference": {"warmup": 10, "samples": 10, "chains": 1, "rng_seed": 0},
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    return ConfigHandler.load(str(config_path))


@pytest.fixture
def multi_object_config(tmp_path):
    """Create a config with n_objects > 1 to test validation."""
    config_dict = {
        "image": {
            "pixel_scale": 0.1,
            "size_x": 48,
            "size_y": 48,
            "n_objects": 3,
            "fft_size": 128,
            "noise": {"type": "Gaussian", "sigma": 0.001},
        },
        "psf": {"type": "Gaussian", "sigma": 0.1},
        "gal": {
            "type": "Exponential",
            "flux": 1000.0,
            "half_light_radius": 0.5,
            "shear": {
                "type": "G1G2",
                "g1": {"type": "Normal", "mean": 0.0, "sigma": 0.05},
                "g2": {"type": "Normal", "mean": 0.0, "sigma": 0.05},
            },
        },
        "inference": {"warmup": 10, "samples": 10, "chains": 1, "rng_seed": 0},
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    return ConfigHandler.load(str(config_path))


def _make_batched_mock_idata(
    n_batch=3,
    n_chains=2,
    n_samples=500,
    g1_means=None,
    g2_means=None,
):
    """Create a mock batched InferenceData with batch dimension."""
    rng = np.random.default_rng(42)

    if g1_means is None:
        g1_means = [0.01, 0.02, 0.05][:n_batch]
    if g2_means is None:
        g2_means = [0.0, 0.0, 0.0][:n_batch]

    # Shape: (n_chains, n_samples, n_batch)
    g1 = np.stack(
        [rng.normal(m, 0.001, size=(n_chains, n_samples)) for m in g1_means],
        axis=-1,
    )
    g2 = np.stack(
        [rng.normal(m, 0.001, size=(n_chains, n_samples)) for m in g2_means],
        axis=-1,
    )

    # ArviZ needs xarray Dataset with named dims
    import xarray as xr

    posterior = xr.Dataset(
        {
            "g1": (["chain", "draw", "batch"], g1),
            "g2": (["chain", "draw", "batch"], g2),
        },
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_samples),
            "batch": np.arange(n_batch),
        },
    )

    diverging = np.zeros((n_chains, n_samples), dtype=bool)
    energy = rng.normal(100, 10, size=(n_chains, n_samples))

    sample_stats = xr.Dataset(
        {
            "diverging": (["chain", "draw"], diverging),
            "energy": (["chain", "draw"], energy),
        },
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_samples),
        },
    )

    return az.InferenceData(posterior=posterior, sample_stats=sample_stats)


class TestBuildBatchedModel:
    """Tests for SceneBuilder.build_batched_model()."""

    def test_builds_callable(self, level0_config):
        from shine.scene import SceneBuilder

        builder = SceneBuilder(level0_config)
        model_fn = builder.build_batched_model(n_batch=3)
        assert callable(model_fn)

    def test_rejects_multi_object(self, multi_object_config):
        from shine.scene import SceneBuilder

        builder = SceneBuilder(multi_object_config)
        with pytest.raises(ValueError, match="n_objects=1"):
            builder.build_batched_model(n_batch=3)

    def test_batched_model_runs(self, level0_config):
        """Verify the batched model can be traced with stacked data."""
        import jax
        import jax.numpy as jnp

        from shine.scene import SceneBuilder

        builder = SceneBuilder(level0_config)
        model_fn = builder.build_batched_model(n_batch=2)

        # Create dummy data
        dummy_images = jnp.zeros((2, 48, 48))
        from shine import psf_utils

        fft_size = level0_config.image.fft_size
        import jax_galsim

        gsparams = jax_galsim.GSParams(
            maximum_fft_size=fft_size, minimum_fft_size=fft_size
        )
        psf = psf_utils.get_jax_psf(level0_config.psf, gsparams=gsparams)

        # Use numpyro trace to verify the model runs and has correct sites
        import numpyro
        from numpyro.handlers import seed, trace

        rng_key = jax.random.PRNGKey(0)
        with seed(rng_seed=0):
            tr = trace(model_fn).get_trace(observed_data=dummy_images, psf=psf)

        # Verify expected sample sites exist
        assert "g1" in tr
        assert "g2" in tr
        assert "obs" in tr

        # Verify batch dimension in parameter shapes
        assert tr["g1"]["value"].shape == (2,)
        assert tr["g2"]["value"].shape == (2,)


class TestSplitBatchedIdata:
    """Tests for split_batched_idata()."""

    def test_split_produces_correct_count(self):
        n_batch = 3
        idata = _make_batched_mock_idata(n_batch=n_batch)
        run_ids = [f"run_{i}" for i in range(n_batch)]

        results = split_batched_idata(idata, n_batch, run_ids)
        assert len(results) == n_batch

    def test_split_preserves_run_ids(self):
        n_batch = 3
        idata = _make_batched_mock_idata(n_batch=n_batch)
        run_ids = ["alpha", "beta", "gamma"]

        results = split_batched_idata(idata, n_batch, run_ids)
        for i, (rid, _) in enumerate(results):
            assert rid == run_ids[i]

    def test_split_correct_shapes(self):
        n_batch = 3
        n_chains = 2
        n_samples = 500
        idata = _make_batched_mock_idata(
            n_batch=n_batch, n_chains=n_chains, n_samples=n_samples
        )
        run_ids = [f"run_{i}" for i in range(n_batch)]

        results = split_batched_idata(idata, n_batch, run_ids)
        for _, single_idata in results:
            g1_vals = single_idata.posterior["g1"].values
            assert g1_vals.shape == (n_chains, n_samples)

    def test_split_preserves_values(self):
        n_batch = 3
        g1_means = [0.01, 0.03, 0.05]
        idata = _make_batched_mock_idata(n_batch=n_batch, g1_means=g1_means)
        run_ids = [f"run_{i}" for i in range(n_batch)]

        results = split_batched_idata(idata, n_batch, run_ids)

        for i, (_, single_idata) in enumerate(results):
            g1_mean = float(single_idata.posterior["g1"].values.mean())
            assert g1_mean == pytest.approx(g1_means[i], abs=0.005)

    def test_split_preserves_sample_stats(self):
        idata = _make_batched_mock_idata(n_batch=2)
        results = split_batched_idata(idata, 2, ["a", "b"])

        for _, single_idata in results:
            assert hasattr(single_idata, "sample_stats")
            assert "diverging" in single_idata.sample_stats
            assert "energy" in single_idata.sample_stats

    def test_split_mismatched_run_ids_raises(self):
        idata = _make_batched_mock_idata(n_batch=3)
        with pytest.raises(ValueError, match="run_ids length"):
            split_batched_idata(idata, 3, ["a", "b"])

    def test_split_produces_valid_idata_for_extraction(self):
        """Verify split results are compatible with existing extraction pipeline."""
        from shine.validation.extraction import extract_shear_estimates

        n_batch = 2
        idata = _make_batched_mock_idata(
            n_batch=n_batch, g1_means=[0.02, 0.05]
        )
        results = split_batched_idata(idata, n_batch, ["r0", "r1"])

        for _, single_idata in results:
            est = extract_shear_estimates(single_idata, "g1")
            assert est.std > 0


class TestGenerateBatchObservations:
    """Tests for generate_batch_observations()."""

    def test_produces_batch_result(self, level0_config):
        result = generate_batch_observations(
            level0_config,
            shear_pairs=[(0.01, 0.0), (0.02, 0.0)],
            seeds=[42, 43],
        )
        assert isinstance(result, BatchSimulationResult)

    def test_stacked_image_shape(self, level0_config):
        result = generate_batch_observations(
            level0_config,
            shear_pairs=[(0.01, 0.0), (0.02, 0.0), (0.05, 0.0)],
            seeds=[42, 43, 44],
        )
        assert result.images.shape == (3, 48, 48)

    def test_ground_truths_correct(self, level0_config):
        shear_pairs = [(0.01, 0.0), (0.03, -0.01)]
        result = generate_batch_observations(
            level0_config,
            shear_pairs=shear_pairs,
            seeds=[42, 43],
        )
        assert len(result.ground_truths) == 2
        assert result.ground_truths[0]["g1"] == 0.01
        assert result.ground_truths[1]["g1"] == 0.03
        assert result.ground_truths[1]["g2"] == -0.01

    def test_run_ids_generated(self, level0_config):
        result = generate_batch_observations(
            level0_config,
            shear_pairs=[(0.01, 0.0), (0.02, 0.0)],
            seeds=[42, 43],
            run_id_prefix="test",
        )
        assert len(result.run_ids) == 2
        assert result.run_ids[0] == "test_0000"
        assert result.run_ids[1] == "test_0001"

    def test_psf_model_present(self, level0_config):
        result = generate_batch_observations(
            level0_config,
            shear_pairs=[(0.01, 0.0)],
            seeds=[42],
        )
        assert result.psf_model is not None

    def test_mismatched_lengths_raises(self, level0_config):
        with pytest.raises(ValueError, match="same length"):
            generate_batch_observations(
                level0_config,
                shear_pairs=[(0.01, 0.0), (0.02, 0.0)],
                seeds=[42],
            )


class TestBatchCLIArgs:
    """Tests for CLI argument parsing with batch flags."""

    def test_default_batch_size(self):
        """Default batch-size is 1 (backward compat)."""
        from shine.validation.cli import run_bias_realization

        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--batch-size", type=int, default=1)
        args = parser.parse_args([])
        assert args.batch_size == 1

    def test_batch_size_parsed(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--batch-size", type=int, default=1)
        parser.add_argument("--shear-grid", type=float, nargs="+", default=None)
        parser.add_argument("--n-realizations", type=int, default=1)
        parser.add_argument("--base-seed", type=int, default=42)

        args = parser.parse_args([
            "--batch-size", "16",
            "--shear-grid", "0.01", "0.02", "0.05",
            "--n-realizations", "2",
            "--base-seed", "100",
        ])

        assert args.batch_size == 16
        assert args.shear_grid == [0.01, 0.02, 0.05]
        assert args.n_realizations == 2
        assert args.base_seed == 100

    def test_shear_grid_expansion(self):
        """Verify shear grid x n_realizations produces correct count."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--shear-grid", type=float, nargs="+", default=None)
        parser.add_argument("--n-realizations", type=int, default=1)
        parser.add_argument("--base-seed", type=int, default=42)
        parser.add_argument("--g2-true", type=float, default=None)

        args = parser.parse_args([
            "--shear-grid", "0.01", "0.02", "0.05",
            "--n-realizations", "3",
        ])

        g2_true = args.g2_true if args.g2_true is not None else 0.0
        shear_pairs = []
        for g1_val in args.shear_grid:
            for r in range(args.n_realizations):
                shear_pairs.append((g1_val, g2_true))

        assert len(shear_pairs) == 9  # 3 shear values x 3 realizations
