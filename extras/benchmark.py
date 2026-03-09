#!/usr/bin/env python3
"""Benchmark one vectorized physical-space prediction call."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vulcan_emulator_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "vulcan_emulator_cache"))
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise RuntimeError("matplotlib is required for benchmarking plots. Install project dependencies.") from exc

from inference import load_physical_space_model, physical_inputs_from_processed_arrays
from script_utils import (
    count_fixed_split_samples,
    iter_fixed_split_batches,
    load_checkpoint,
    resolve_processed_root_from_checkpoint,
    resolve_run_dir,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
RUN_DIR_OVERRIDE: Path | None = None
TEST_SPLIT = "test"
CHECKPOINT_NAME = "best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
WARMUP_ITERS = 5
BENCHMARK_ITERS = 10
FIGURES_SUBDIR = "figures"
PLOT_FILENAME = "benchmark_batch_size_vs_time.png"
STYLE_PATH = Path(__file__).with_name("science.mplstyle")

plt.style.use(str(STYLE_PATH))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_vectorized_batch(
    *,
    processed_root: Path,
    config: dict,
    batch_size: int,
    normalization_metadata: dict,
    data_contract: dict,
) -> dict[str, np.ndarray]:
    """Load one contiguous batch and convert it back to physical inputs."""
    total_samples = count_fixed_split_samples(
        processed_root=processed_root,
        split=TEST_SPLIT,
        config=config,
        normalization_metadata=normalization_metadata,
    )
    if total_samples <= 0:
        raise RuntimeError(f"No samples are available in split '{TEST_SPLIT}'.")

    actual_batch_size = min(batch_size, total_samples)
    seq_parts: list[np.ndarray] = []
    glb_parts: list[np.ndarray] = []
    remaining = actual_batch_size

    for seq, glb, _tgt, _dt in iter_fixed_split_batches(
        processed_root=processed_root,
        split=TEST_SPLIT,
        config=config,
        normalization_metadata=normalization_metadata,
        batch_size=actual_batch_size,
    ):
        take = min(remaining, int(seq.shape[0]))
        seq_parts.append(np.asarray(seq[:take], dtype=np.float64))  # shape: (batch, nz, input_dim)
        glb_parts.append(np.asarray(glb[:take], dtype=np.float64))  # shape: (batch, global_dim)
        remaining -= take
        if remaining == 0:
            break

    if remaining != 0:
        raise RuntimeError(
            f"Unable to assemble a full batch of {actual_batch_size} samples from split '{TEST_SPLIT}'."
        )

    sequence_inputs = np.concatenate(seq_parts, axis=0)
    global_inputs = np.concatenate(glb_parts, axis=0)
    physical_inputs = physical_inputs_from_processed_arrays(
        sequence_inputs=sequence_inputs,
        global_inputs=global_inputs,
        normalization_metadata=normalization_metadata,
        data_contract=data_contract,
    )
    physical_inputs["batch_size"] = np.asarray([actual_batch_size], dtype=np.int64)
    return physical_inputs


def _to_device_tensors(
    *,
    physical_inputs: dict[str, np.ndarray],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Convert one vectorized physical-input batch to torch tensors."""
    return (
        torch.as_tensor(physical_inputs["pressure_bar"], dtype=dtype, device=device),
        torch.as_tensor(physical_inputs["temperature_k"], dtype=dtype, device=device),
        torch.as_tensor(physical_inputs["kzz_cm2_s"], dtype=dtype, device=device),
        torch.as_tensor(physical_inputs["anchor_ymix"], dtype=dtype, device=device),
        torch.as_tensor(physical_inputs["gravity_cm_s2"], dtype=dtype, device=device),
        torch.as_tensor(physical_inputs["metallicity_log10"], dtype=dtype, device=device),
        torch.as_tensor(physical_inputs["c_to_o"], dtype=dtype, device=device),
        torch.as_tensor(physical_inputs["dt_s"], dtype=dtype, device=device),
    )


def _slice_batch_tensors(
    tensors: tuple[torch.Tensor, ...],
    batch_size: int,
) -> tuple[torch.Tensor, ...]:
    """Return one view of the leading batch slice for all model inputs."""
    return tuple(tensor[:batch_size] for tensor in tensors)


def _benchmark_one_batch_size(
    *,
    model: torch.nn.Module,
    tensors: tuple[torch.Tensor, ...],
    device: torch.device,
) -> dict[str, float | tuple[int, ...]]:
    """Measure repeated forward latency for one vectorized batch size."""
    with torch.inference_mode():
        for _ in range(WARMUP_ITERS):
            _ = model(*tensors)
        _synchronize(device)

        latencies_s: list[float] = []
        output_shape: tuple[int, ...] | None = None
        for _ in range(BENCHMARK_ITERS):
            start_s = time.perf_counter()
            prediction = model(*tensors)
            _synchronize(device)
            latencies_s.append(time.perf_counter() - start_s)
            if output_shape is None:
                output_shape = tuple(int(dim) for dim in prediction.shape)

    latency_ms = 1.0e3 * np.asarray(latencies_s, dtype=np.float64)
    batch_size = int(tensors[0].shape[0])
    total_time_s = float(np.sum(latencies_s))
    amortized_latency_ms = latency_ms / float(batch_size)
    return {
        "batch_size": float(batch_size),
        "mean_latency_ms": float(np.mean(latency_ms)),
        "median_latency_ms": float(np.median(latency_ms)),
        "min_latency_ms": float(np.min(latency_ms)),
        "max_latency_ms": float(np.max(latency_ms)),
        "std_latency_ms": float(np.std(latency_ms)),
        "mean_ms_per_sample": float(np.mean(amortized_latency_ms)),
        "median_ms_per_sample": float(np.median(amortized_latency_ms)),
        "profiles_per_second": float((batch_size * BENCHMARK_ITERS) / total_time_s),
        "ms_per_profile": float(np.mean(latency_ms) / float(batch_size)),
        "output_shape": output_shape if output_shape is not None else (),
    }


def _plot_batch_size_vs_time(
    *,
    run_dir: Path,
    rows: list[dict[str, float | tuple[int, ...]]],
) -> Path:
    """Save one small latency-vs-batch-size plot under the run figures folder."""
    figures_dir = run_dir / FIGURES_SUBDIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = np.asarray([int(row["batch_size"]) for row in rows], dtype=np.int64)
    mean_ms_per_sample = np.asarray([float(row["mean_ms_per_sample"]) for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(batch_sizes, mean_ms_per_sample, marker="o", linewidth=2.0, color="#2753DB")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Mean Time Per Sample (ms)")
    ax.set_title("Batch Size vs Amortized Prediction Time")
    ax.set_box_aspect(1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path = figures_dir / PLOT_FILENAME
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    """Benchmark one vectorized forward pass through the physical-space wrapper."""
    if not BATCH_SIZES:
        raise ValueError("BATCH_SIZES must not be empty.")
    if min(BATCH_SIZES) <= 0:
        raise ValueError("All BATCH_SIZES must be > 0.")
    if WARMUP_ITERS < 0 or BENCHMARK_ITERS <= 0:
        raise ValueError("WARMUP_ITERS must be >= 0 and BENCHMARK_ITERS must be > 0.")

    resolved_device = torch.device(DEVICE)
    run_dir, config_path = resolve_run_dir(config_path=CONFIG_PATH, run_dir=RUN_DIR_OVERRIDE)
    processed_root = resolve_processed_root_from_checkpoint(run_dir=run_dir, checkpoint_name=CHECKPOINT_NAME)
    checkpoint = load_checkpoint(run_dir=run_dir, checkpoint_name=CHECKPOINT_NAME)
    model, normalization_metadata, data_contract = load_physical_space_model(
        run_dir,
        checkpoint_name=CHECKPOINT_NAME,
        device=resolved_device,
    )

    total_samples = count_fixed_split_samples(
        processed_root=processed_root,
        split=TEST_SPLIT,
        config=checkpoint["config"],
        normalization_metadata=normalization_metadata,
    )
    max_requested_batch_size = max(int(value) for value in BATCH_SIZES)
    max_loaded_batch_size = min(max_requested_batch_size, total_samples)
    if max_loaded_batch_size <= 0:
        raise RuntimeError(f"No samples are available in split '{TEST_SPLIT}'.")

    physical_inputs = _load_vectorized_batch(
        processed_root=processed_root,
        config=checkpoint["config"],
        batch_size=max_loaded_batch_size,
        normalization_metadata=normalization_metadata,
        data_contract=data_contract,
    )
    model_dtype = next(model.parameters()).dtype
    full_batch_tensors = _to_device_tensors(
        physical_inputs=physical_inputs,
        device=resolved_device,
        dtype=model_dtype,
    )
    available_batch_sizes = sorted({int(value) for value in BATCH_SIZES if int(value) <= max_loaded_batch_size})

    benchmark_rows: list[dict[str, float | tuple[int, ...]]] = []
    for batch_size in available_batch_sizes:
        batch_tensors = _slice_batch_tensors(full_batch_tensors, batch_size)
        benchmark_rows.append(
            _benchmark_one_batch_size(
                model=model,
                tensors=batch_tensors,
                device=resolved_device,
            )
        )

    if resolved_device.type == "cuda":
        device_name = torch.cuda.get_device_name(resolved_device)
    else:
        device_name = resolved_device.type

    plot_path = _plot_batch_size_vs_time(run_dir=run_dir, rows=benchmark_rows)
    largest_row = benchmark_rows[-1]

    print("Vectorized prediction benchmark")
    print(f"  Config path         : {config_path if config_path is not None else 'explicit run dir override'}")
    print(f"  Run dir             : {run_dir}")
    print(f"  Checkpoint          : {CHECKPOINT_NAME}")
    print(f"  Device              : {device_name}")
    print(f"  Split               : {TEST_SPLIT}")
    print(f"  Batch sizes         : {available_batch_sizes}")
    print(f"  Sequence length     : {int(data_contract['sequence_length'])}")
    print(f"  Target species      : {int(data_contract['target_dim'])}")
    print(f"  Model dtype         : {model_dtype}")
    print(f"  Warmup iterations   : {WARMUP_ITERS}")
    print(f"  Timed iterations    : {BENCHMARK_ITERS}")
    print(f"  Plot                : {plot_path}")
    print("  Timing summary:")
    for row in benchmark_rows:
        print(
            "    "
            f"batch={int(row['batch_size']):>4d} | "
            f"mean/sample={float(row['mean_ms_per_sample']):>10.6f} ms | "
            f"median/sample={float(row['median_ms_per_sample']):>10.6f} ms | "
            f"profiles/s={float(row['profiles_per_second']):>12.6f}"
        )
    print("  Largest batch details:")
    print(f"    Batch size        : {int(largest_row['batch_size'])}")
    print(f"    Output shape      : {largest_row['output_shape']}")
    print(f"    Mean batch (ms)   : {float(largest_row['mean_latency_ms']):.6f}")
    print(f"    Median batch (ms) : {float(largest_row['median_latency_ms']):.6f}")
    print(f"    Min latency       : {float(largest_row['min_latency_ms']):.6f}")
    print(f"    Max latency       : {float(largest_row['max_latency_ms']):.6f}")
    print(f"    Std latency       : {float(largest_row['std_latency_ms']):.6f}")
    print(f"    Mean/sample (ms)  : {float(largest_row['mean_ms_per_sample']):.6f}")
    print(f"    Median/sample(ms) : {float(largest_row['median_ms_per_sample']):.6f}")
    print(f"    Profiles / second : {float(largest_row['profiles_per_second']):.6f}")
    print(f"    ms / profile      : {float(largest_row['ms_per_profile']):.6f}")


if __name__ == "__main__":
    main()
