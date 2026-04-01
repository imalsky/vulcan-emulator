"""Benchmark forward-pass throughput on CPU and GPU.

Times JIT compilation and steady-state inference for both model types at
several batch sizes. Reports latency per sample and total throughput.

Usage:
    python extras/benchmark.py
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# -- Configuration -----------------------------------------------------------
CHECKPOINT = _ROOT / "models" / "fastchem_mlp" / "best.pt"
DEVICES = ["cpu", "gpu"]
BATCH_SIZES = [1, 8, 32, 128]
WARMUP_ITERS = 3
BENCH_ITERS = 20
# ---------------------------------------------------------------------------

sys.path.insert(0, str(_ROOT))

from src.utils.numpy_compat import patch_numpy_asarray_copy

patch_numpy_asarray_copy()

import jax
import numpy as np

from src.models.jax_model import (
    MLPDimensions,
    TransformerDimensions,
    apply_mlp,
    apply_transformer_model,
)


def _uses_mlp(payload: dict) -> bool:
    return str(payload.get("config", {}).get("model_type", "")) == "mlp"


def _make_dummy_fastchem(dims: MLPDimensions | TransformerDimensions, batch: int, nz: int = 150):
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    seq = jax.random.normal(k1, (batch, nz, dims.sequence_dim))
    globs = jax.random.normal(k2, (batch, dims.global_dim))
    return seq, globs


def _make_dummy_vulcan(dims: MLPDimensions | TransformerDimensions, batch: int, nz: int = 150):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    seq = jax.random.normal(k1, (batch, nz, dims.sequence_dim))
    globs = jax.random.normal(k2, (batch, dims.global_dim))
    spec = jax.random.normal(k3, (batch, dims.spectrum_dim))
    return seq, globs, spec


def _time_fn(fn, n_warmup: int, n_bench: int) -> float:
    for _ in range(n_warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _print_header(model_label: str):
    print(f"\n{model_label}")
    print(f"{'Device':<8} {'Batch':>6} {'Median (ms)':>12} "
          f"{'Per-sample (ms)':>16} {'Throughput':>14}")
    print("-" * 62)


def _print_row(dev_name: str, bs: int, median_s: float):
    per_ms = (median_s / bs) * 1000
    tput = bs / median_s
    print(f"{dev_name:<8} {bs:>6} {median_s * 1000:>11.3f} "
          f"{per_ms:>15.4f} {tput:>11.0f} /s")


def benchmark_fastchem(payload: dict):
    dims = (
        MLPDimensions.from_dict(payload["model_dimensions"])
        if _uses_mlp(payload)
        else TransformerDimensions.from_dict(payload["model_dimensions"])
    )
    _print_header(
        f"FastChem {payload['config']['model_type']}  |  "
        f"sequence_dim={dims.sequence_dim}  global_dim={dims.global_dim}"
    )
    for dev_name in DEVICES:
        try:
            device = jax.devices(dev_name)[0]
        except RuntimeError:
            print(f"{dev_name:<8}  -- not available, skipping --")
            continue
        params = jax.device_put(payload["params"], device)
        for bs in BATCH_SIZES:
            seq, globs = _make_dummy_fastchem(dims, bs)
            seq, globs = jax.device_put((seq, globs), device)
            if _uses_mlp(payload):
                fn = lambda: apply_mlp(params, seq, globs, dims)
            else:
                fn = lambda: apply_transformer_model(params, seq, globs, None, dims)
            _print_row(dev_name, bs, _time_fn(fn, WARMUP_ITERS, BENCH_ITERS))


def benchmark_vulcan(payload: dict):
    dims = (
        MLPDimensions.from_dict(payload["model_dimensions"])
        if _uses_mlp(payload)
        else TransformerDimensions.from_dict(payload["model_dimensions"])
    )
    _print_header(
        f"VULCAN {payload['config']['model_type']}  |  "
        f"sequence_dim={dims.sequence_dim}  global_dim={dims.global_dim}"
    )
    for dev_name in DEVICES:
        try:
            device = jax.devices(dev_name)[0]
        except RuntimeError:
            print(f"{dev_name:<8}  -- not available, skipping --")
            continue
        params = jax.device_put(payload["params"], device)
        for bs in BATCH_SIZES:
            seq, globs, spec = _make_dummy_vulcan(dims, bs)
            seq, globs, spec = jax.device_put((seq, globs, spec), device)
            if _uses_mlp(payload):
                fn = lambda: apply_mlp(params, seq, globs, dims, spec)
            else:
                fn = lambda: apply_transformer_model(params, seq, globs, spec, dims)
            _print_row(dev_name, bs, _time_fn(fn, WARMUP_ITERS, BENCH_ITERS))


def main():
    with CHECKPOINT.open("rb") as f:
        payload = pickle.load(f)

    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")

    if str(payload.get("config", {}).get("chemistry_type", "")) == "fastchem":
        benchmark_fastchem(payload)
    else:
        benchmark_vulcan(payload)


if __name__ == "__main__":
    main()
