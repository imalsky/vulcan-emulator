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
CHECKPOINT = _ROOT / "models/equilibrium_only_silu/best.pt"
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
    EquilibriumMLPDimensions,
    ModelDimensions,
    apply_equilibrium_mlp,
    apply_model,
)


def _is_equilibrium(payload: dict) -> bool:
    return "d_hidden" in payload["model_dimensions"]


def _make_dummy_equilibrium(dims: EquilibriumMLPDimensions, batch: int, nz: int = 150):
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    seq = jax.random.normal(k1, (batch, nz, dims.sequence_dim))
    globs = jax.random.normal(k2, (batch, dims.global_dim))
    return seq, globs


def _make_dummy_transition(dims: ModelDimensions, batch: int, nz: int = 150):
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


def benchmark_equilibrium(payload: dict):
    dims = EquilibriumMLPDimensions.from_dict(payload["model_dimensions"])
    _print_header(
        f"Equilibrium MLP  |  d_hidden={dims.d_hidden}  "
        f"layers={dims.num_hidden_layers}"
    )
    for dev_name in DEVICES:
        try:
            device = jax.devices(dev_name)[0]
        except RuntimeError:
            print(f"{dev_name:<8}  -- not available, skipping --")
            continue
        params = jax.device_put(payload["params"], device)
        for bs in BATCH_SIZES:
            seq, globs = _make_dummy_equilibrium(dims, bs)
            seq, globs = jax.device_put((seq, globs), device)
            fn = lambda: apply_equilibrium_mlp(params, seq, globs, dims)
            _print_row(dev_name, bs, _time_fn(fn, WARMUP_ITERS, BENCH_ITERS))


def benchmark_transition(payload: dict):
    dims = ModelDimensions.from_dict(payload["model_dimensions"])
    _print_header(
        f"Transformer  |  d_model={dims.d_model}  "
        f"layers={dims.num_layers}  heads={dims.nhead}"
    )
    for dev_name in DEVICES:
        try:
            device = jax.devices(dev_name)[0]
        except RuntimeError:
            print(f"{dev_name:<8}  -- not available, skipping --")
            continue
        params = jax.device_put(payload["params"], device)
        for bs in BATCH_SIZES:
            seq, globs, spec = _make_dummy_transition(dims, bs)
            seq, globs, spec = jax.device_put((seq, globs, spec), device)
            fn = lambda: apply_model(params, seq, globs, spec, dims)
            _print_row(dev_name, bs, _time_fn(fn, WARMUP_ITERS, BENCH_ITERS))


def main():
    with CHECKPOINT.open("rb") as f:
        payload = pickle.load(f)

    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")

    if _is_equilibrium(payload):
        benchmark_equilibrium(payload)
    else:
        benchmark_transition(payload)


if __name__ == "__main__":
    main()
