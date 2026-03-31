"""Normalization fitting and raw-to-processed tensor conversion.

This module bridges the gap between raw HDF5 runs (produced by
``generation.py``) and the ready-to-train NumPy arrays consumed by the
trainer.  The pipeline has four stages:

1. **Load** — read raw runs from HDF5, aligning stored species to the
   configured species contract.
2. **Split** — deterministically partition runs into train / val / test
   sets using a seeded shuffle.
3. **Fit normalization** — compute per-feature statistics (mean, std,
   floor) from the *training split only* so that validation and test
   data remain unseen during fitting.
4. **Apply & persist** — transform every split through the fitted
   normalization blocks and write the resulting ``.npy`` tensors,
   ``normalization.json``, ``data_contract.json``, and provenance
   manifests to ``processed_root``.

Two top-level entry points handle the two task kinds:

* ``preprocess_equilibrium_dataset`` — for equilibrium-only models
  (no trajectory, no spectrum).
* ``preprocess_raw_dataset`` — for full-VULCAN transition models
  (trajectory + spectrum + dt features).

``PROCESSED_DATA_VERSION`` is bumped whenever the on-disk tensor
layout changes in a backwards-incompatible way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..utils.config import (
    effective_transition_sampling,
    is_equilibrium,
    resolve_conditioning_inputs,
    task_kind,
)
from ..utils.helpers import ensure_dir, get_logger, resolve_path

LOGGER = get_logger(__name__)
from ..utils.provenance import fingerprint_payload, manifest_for_files
from .spectrum import SpectrumRecord, fixed_wavelength_grid, resample_spectrum
from .transition_sampling import fit_log10_dt_normalization

# Bump this integer whenever the processed tensor layout changes in a way
# that would silently break a model trained on a prior version.
PROCESSED_DATA_VERSION = 7


@dataclass(frozen=True)
class RawRun:
    """Raw transition run loaded from HDF5 before normalization.

    Fields mirror the HDF5 layout produced by ``generation.write_raw_run_hdf5``.
    Species columns are already reordered to match the config's
    ``state_species`` / ``output_species`` contract.
    """
    run_id: str
    pressure_bar: np.ndarray
    temperature_k: np.ndarray
    kzz_cm2_s: np.ndarray
    time_s: np.ndarray
    ymix_state: np.ndarray
    ymix_output: np.ndarray
    reference_ymix_state: np.ndarray
    target_mode: str
    globals: dict[str, float]
    spectrum_name: str
    spectrum_wavelength_nm: np.ndarray
    spectrum_flux_erg_cm2_s_nm: np.ndarray
    elemental_abundances_x_h: np.ndarray
    gravity_cm_s2: np.ndarray


@dataclass(frozen=True)
class RawEquilibriumRun:
    """Simplified raw run for equilibrium-only models (no trajectory/spectrum)."""
    run_id: str
    pressure_bar: np.ndarray
    temperature_k: np.ndarray
    equilibrium_ymix: np.ndarray
    globals: dict[str, float]
    elemental_abundances_x_h: np.ndarray
    gravity_cm_s2: np.ndarray


def _decode_species(values: np.ndarray) -> list[str]:
    """Decode stored species labels from HDF5 string arrays."""
    result: list[str] = []
    for item in values:
        if isinstance(item, bytes):
            result.append(item.decode("utf-8"))
        else:
            result.append(str(item))
    return result


def _require_column_constant(
    values: np.ndarray,
    *,
    name: str,
    run_label: str,
) -> None:
    """Reject profile inputs that vary with height in the current training contract."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        reference = arr[0]
        is_constant = np.allclose(arr, reference, rtol=1.0e-8, atol=0.0)
    elif arr.ndim == 2:
        reference = arr[0]
        is_constant = np.allclose(arr, reference[None, :], rtol=1.0e-8, atol=0.0)
    else:
        raise ValueError(f"{run_label}: {name} must be 1-D or 2-D, got shape {arr.shape}.")
    if not is_constant:
        raise ValueError(f"{run_label}: {name} must be vertically constant in the current contract.")


def _fit_standard(arr: np.ndarray) -> dict[str, Any]:
    """Fit mean and standard deviation for standard (z-score) scaling.

    Flattens all leading dimensions so statistics are computed per
    feature column.  Any column with std < 1e-8 is clamped to 1.0 to
    avoid division by zero at apply time.
    """
    arr2 = np.reshape(arr, (-1, arr.shape[-1]))
    mean = np.mean(arr2, axis=0)
    std = np.std(arr2, axis=0)
    std = np.where(std < 1.0e-8, 1.0, std)
    return {"method": "standard", "mean": mean.tolist(), "std": std.tolist()}


def _fit_none(arr: np.ndarray) -> dict[str, Any]:
    """Build a pass-through normalization block."""
    dim = int(arr.shape[-1]) if arr.ndim >= 1 else 1
    return {"method": "none", "mean": [0.0] * dim, "std": [1.0] * dim}


def _fit_log_standard(arr: np.ndarray, *, floor: float) -> dict[str, Any]:
    """Fit standard scaling in log10 space with a lower floor.

    Values below ``floor`` are clipped before the log10 transform,
    preventing -inf from corrupting statistics.  Used for mixing
    ratios and other strictly-positive quantities that span many
    orders of magnitude.
    """
    arr2 = np.reshape(np.clip(arr, floor, None), (-1, arr.shape[-1]))
    log10_arr = np.log10(arr2)
    mean = np.mean(log10_arr, axis=0)
    std = np.std(log10_arr, axis=0)
    std = np.where(std < 1.0e-8, 1.0, std)
    return {
        "method": "log-standard",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "floor": float(floor),
    }


def apply_block(x: np.ndarray, block: dict[str, Any]) -> np.ndarray:
    """Apply one normalization block (forward direction) to an array.

    Supports ``"standard"`` (z-score), ``"log-standard"`` (log10 then
    z-score), and ``"none"`` (identity passthrough).
    """
    method = block["method"]
    mean = np.asarray(block["mean"], dtype=np.float64)
    std = np.asarray(block["std"], dtype=np.float64)
    if method == "none":
        return np.asarray(x, dtype=np.float64)
    if method == "standard":
        return (np.asarray(x, dtype=np.float64) - mean) / std
    if method == "log-standard":
        floor = float(block["floor"])
        transformed = np.log10(np.clip(np.asarray(x, dtype=np.float64), floor, None))
        return (transformed - mean) / std
    raise ValueError(f"Unsupported normalization method: {method}")


def inverse_block(x: np.ndarray, block: dict[str, Any]) -> np.ndarray:
    """Invert one normalization block back to physical (original) space.

    This is the inverse of ``apply_block``: for ``"log-standard"``
    the z-score is undone then 10^x recovers the original magnitude.
    """
    method = block["method"]
    mean = np.asarray(block["mean"], dtype=np.float64)
    std = np.asarray(block["std"], dtype=np.float64)
    if method == "none":
        return np.asarray(x, dtype=np.float64)
    if method == "standard":
        return np.asarray(x, dtype=np.float64) * std + mean
    if method == "log-standard":
        log10_x = np.asarray(x, dtype=np.float64) * std + mean
        return np.power(10.0, log10_x)
    raise ValueError(f"Unsupported normalization method: {method}")


def _fit_block_by_method(
    arr: np.ndarray,
    *,
    method: str,
    floor: float | None = None,
) -> dict[str, Any]:
    """Fit one normalization block using an explicit config-selected method."""
    if method == "standard":
        return _fit_standard(arr)
    if method == "log-standard":
        if floor is None:
            raise ValueError("log-standard normalization requires a positive floor.")
        return _fit_log_standard(arr, floor=float(floor))
    if method == "none":
        return _fit_none(arr)
    raise ValueError(f"Unsupported normalization method: {method}")


def _fit_mixed_block(arr: np.ndarray, methods: list[str]) -> dict[str, Any]:
    """Fit a mixed normalization block with per-column methods.

    Each column of ``arr`` is normalized independently according to
    the corresponding entry in ``methods``.  This keeps boolean and
    one-hot features in physical space while continuous features are
    z-score normalized.
    """
    arr2 = np.reshape(arr, (-1, arr.shape[-1]))
    transformed_columns = []
    means = []
    stds = []
    floors = []
    for i, method in enumerate(methods):
        # Keep the per-feature transform explicit so boolean and one-hot features
        # stay in physical space while continuous features are normalized.
        column = arr2[:, i]
        if method == "none":
            means.append(0.0)
            stds.append(1.0)
            floors.append(None)
            transformed_columns.append(column)
        elif method == "standard":
            mean = float(np.mean(column))
            std = float(max(np.std(column), 1.0e-8))
            means.append(mean)
            stds.append(std)
            floors.append(None)
            transformed_columns.append((column - mean) / std)
        elif method == "log-standard":
            floor = 1.0e-30
            log_column = np.log10(np.clip(column, floor, None))
            mean = float(np.mean(log_column))
            std = float(max(np.std(log_column), 1.0e-8))
            means.append(mean)
            stds.append(std)
            floors.append(floor)
            transformed_columns.append((log_column - mean) / std)
        else:
            raise ValueError(f"Unsupported mixed normalization method: {method}")
    return {"method": "mixed", "methods": methods, "mean": means, "std": stds, "floor": floors}


def apply_mixed_block(x: np.ndarray, block: dict[str, Any]) -> np.ndarray:
    """Apply a mixed per-feature normalization block."""
    x = np.asarray(x, dtype=np.float64)
    outputs = []
    for i, method in enumerate(block["methods"]):
        column = x[..., i]
        mean = float(block["mean"][i])
        std = float(block["std"][i])
        floor = block["floor"][i]
        if method == "none":
            outputs.append(column)
        elif method == "standard":
            outputs.append((column - mean) / std)
        elif method == "log-standard":
            outputs.append((np.log10(np.clip(column, float(floor), None)) - mean) / std)
        else:
            raise ValueError(f"Unsupported mixed normalization method: {method}")
    return np.stack(outputs, axis=-1)


def inverse_mixed_block(x: np.ndarray, block: dict[str, Any]) -> np.ndarray:
    """Invert a mixed per-feature normalization block."""
    x = np.asarray(x, dtype=np.float64)
    outputs = []
    for i, method in enumerate(block["methods"]):
        column = x[..., i]
        mean = float(block["mean"][i])
        std = float(block["std"][i])
        if method == "none":
            outputs.append(column)
        elif method == "standard":
            outputs.append(column * std + mean)
        elif method == "log-standard":
            outputs.append(np.power(10.0, column * std + mean))
        else:
            raise ValueError(f"Unsupported mixed normalization method: {method}")
    return np.stack(outputs, axis=-1)


def _fit_log10_dt_block(stats: dict[str, float], *, method: str) -> dict[str, Any]:
    """Wrap fitted log-dt statistics using the configured normalization method."""
    if method == "standard":
        return {
            "method": "standard",
            "mean": [float(stats["mean"])],
            "std": [float(stats["std"])],
        }
    if method == "none":
        return {"method": "none", "mean": [0.0], "std": [1.0]}
    raise ValueError(f"Unsupported log10_dt_s normalization method: {method}")


def load_raw_run(
    source: str | Path | h5py.Group,
    *,
    config: dict[str, Any],
    spectrum_grid_nm: np.ndarray,
    run_id: str | None = None,
) -> RawRun:
    """Load one raw HDF5 transition run and align it to the configured species contract.

    *source* may be a file path (legacy per-file layout) or an already-opened
    ``h5py.Group`` from a consolidated ``runs.h5``.  When *source* is a group,
    *run_id* must be supplied explicitly.
    """
    requested_state_species = list(config["data_spec"]["state_species"])
    requested_output_species = list(config["data_spec"]["output_species"])

    def _extract(handle: h5py.Group) -> dict[str, Any]:
        return {
            "pressure_bar": np.asarray(handle["inputs/pressure_bar"], dtype=np.float64),
            "temperature_k": np.asarray(handle["inputs/temperature_k"], dtype=np.float64),
            "kzz_cm2_s": np.asarray(handle["inputs/kzz_cm2_s"], dtype=np.float64),
            "element_input_order": _decode_species(np.asarray(handle["inputs/element_input_order"])),
            "elemental_abundances_x_h": np.asarray(handle["inputs/elemental_abundances_x_h"], dtype=np.float64),
            "gravity_cm_s2": np.asarray(handle["inputs/gravity_cm_s2"], dtype=np.float64),
            "stored_state_species": _decode_species(np.asarray(handle["inputs/state_species"])),
            "stored_output_species": _decode_species(np.asarray(handle["inputs/output_species"])),
            "reference_ymix_state": np.asarray(handle["inputs/reference_ymix_state"], dtype=np.float64),
            "target_mode": handle["inputs/target_mode"][()].decode("utf-8"),
            "time_s": np.asarray(handle["trajectory/time_s"], dtype=np.float64),
            "ymix_state": np.asarray(handle["trajectory/ymix_state"], dtype=np.float64),
            "ymix_output": np.asarray(handle["trajectory/ymix_output"], dtype=np.float64),
            "globals_map": {
                key: float(np.asarray(handle[f"globals/{key}"]))
                for key in handle["globals"].keys()
            },
            "spectrum_name": str(np.asarray(handle["spectrum/name"]).astype(str)),
            "spectrum_wavelength_nm": np.asarray(handle["spectrum/wavelength_nm"], dtype=np.float64),
            "spectrum_flux": np.asarray(handle["spectrum/flux_erg_cm2_s_nm"], dtype=np.float64),
        }

    if isinstance(source, h5py.Group):
        d = _extract(source)
        label = run_id or "unknown"
    else:
        with h5py.File(source, "r") as handle:
            d = _extract(handle)
        label = Path(source).stem
        if run_id is None:
            run_id = label

    pressure_bar = d["pressure_bar"]
    temperature_k = d["temperature_k"]
    kzz_cm2_s = d["kzz_cm2_s"]
    time_s = d["time_s"]
    ymix_state = d["ymix_state"]
    ymix_output = d["ymix_output"]
    reference_ymix_state = d["reference_ymix_state"]

    if not np.all(np.isfinite(pressure_bar)):
        raise ValueError(f"{label}: non-finite pressure values detected.")
    if not np.all(np.isfinite(temperature_k)):
        raise ValueError(f"{label}: non-finite temperature values detected.")
    if not np.all(np.isfinite(kzz_cm2_s)):
        raise ValueError(f"{label}: non-finite Kzz values detected.")
    if not np.all(np.isfinite(d["elemental_abundances_x_h"])):
        raise ValueError(f"{label}: non-finite elemental abundance values detected.")
    if not np.all(np.isfinite(d["gravity_cm_s2"])):
        raise ValueError(f"{label}: non-finite gravity values detected.")
    if not np.all(np.diff(time_s) > 0.0):
        raise ValueError(f"{label}: time_s must be strictly increasing.")
    if ymix_state.shape[0] != time_s.size:
        raise ValueError(f"{label}: ymix_state time dimension does not match time_s.")
    if ymix_output.shape[0] != time_s.size:
        raise ValueError(f"{label}: ymix_output time dimension does not match time_s.")
    if reference_ymix_state.shape[0] != pressure_bar.size:
        raise ValueError(f"{label}: reference_ymix_state vertical dimension does not match pressure grid.")
    if d["elemental_abundances_x_h"].shape[0] != pressure_bar.size:
        raise ValueError(f"{label}: elemental abundance profile does not match pressure grid.")
    if d["gravity_cm_s2"].shape != pressure_bar.shape:
        raise ValueError(f"{label}: gravity profile does not match pressure grid.")

    state_indices = [d["stored_state_species"].index(name) for name in requested_state_species]
    output_indices = [d["stored_output_species"].index(name) for name in requested_output_species]
    element_order = list(config["data_spec"]["element_input_order"])
    element_indices = [d["element_input_order"].index(name) for name in element_order]
    elemental_profile = d["elemental_abundances_x_h"][:, element_indices]
    gravity_profile = np.asarray(d["gravity_cm_s2"], dtype=np.float64)
    _require_column_constant(elemental_profile, name="elemental_abundances_x_h", run_label=label)
    _require_column_constant(gravity_profile, name="gravity_cm_s2", run_label=label)
    reduced_globals = dict(d["globals_map"])
    reduced_globals.update(
        {
            name: float(elemental_profile[0, idx])
            for idx, name in enumerate(element_order)
        }
    )
    reduced_globals["gravity_cm_s2"] = float(gravity_profile[0])
    ymix_state = ymix_state[..., state_indices]
    ymix_output = ymix_output[..., output_indices]
    reference_ymix_state = reference_ymix_state[..., state_indices]
    resampled_spectrum = resample_spectrum(
        SpectrumRecord(
            name=d["spectrum_name"],
            wavelength_nm=d["spectrum_wavelength_nm"],
            flux_erg_cm2_s_nm=d["spectrum_flux"],
        ),
        spectrum_grid_nm,
    )
    return RawRun(
        run_id=run_id or label,
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        kzz_cm2_s=kzz_cm2_s,
        time_s=time_s,
        ymix_state=ymix_state,
        ymix_output=ymix_output,
        reference_ymix_state=reference_ymix_state,
        target_mode=d["target_mode"],
        globals=reduced_globals,
        spectrum_name=d["spectrum_name"],
        spectrum_wavelength_nm=np.asarray(spectrum_grid_nm, dtype=np.float64),
        spectrum_flux_erg_cm2_s_nm=np.asarray(resampled_spectrum, dtype=np.float64),
        elemental_abundances_x_h=elemental_profile,
        gravity_cm_s2=gravity_profile,
    )


def _split_indices(num_runs: int, *, config: dict[str, Any]) -> dict[str, list[int]]:
    """Split run indices into train/val/test partitions.

    Uses a seeded RNG shuffle to ensure reproducibility.  Edge cases
    (very small datasets) are handled by guaranteeing at least one
    sample per split.
    """
    split_cfg = config["normalization"]["split"]
    # The public config contract stores split settings under normalization.split.
    rng = np.random.default_rng(int(split_cfg["seed"]))
    perm = np.arange(num_runs, dtype=np.int32)
    rng.shuffle(perm)
    train_n = max(1, int(round(num_runs * float(split_cfg["train_fraction"]))))
    val_n = max(1, int(round(num_runs * float(split_cfg["val_fraction"]))))
    if train_n + val_n >= num_runs:
        val_n = max(1, num_runs - train_n - 1)
    test_n = num_runs - train_n - val_n
    if test_n < 1:
        test_n = 1
        if train_n > val_n:
            train_n -= 1
        else:
            val_n -= 1
    return {
        "train": perm[:train_n].tolist(),
        "val": perm[train_n : train_n + val_n].tolist(),
        "test": perm[train_n + val_n :].tolist(),
    }


def _normalization_payload(
    train_runs: list[RawRun],
    *,
    config: dict[str, Any],
    global_static_order: list[str],
    spectrum_grid_nm: np.ndarray,
) -> dict[str, Any]:
    """Fit all normalization blocks for the full-VULCAN transition pipeline.

    Statistics are computed from ``train_runs`` only.  The returned
    dict contains blocks for: ``sequence_static`` (P, T, Kzz),
    ``state`` (ymix input), ``target`` (ymix output),
    ``global_static`` (conditioning scalars), ``log10_dt_s`` (time
    step feature), and ``spectrum`` (stellar flux).
    """
    state_floor = float(config["normalization"]["state_floor"])
    spectrum_floor = float(config["normalization"]["spectrum_floor"])
    sequence_static = np.concatenate(
        [
            np.stack(
                [run.pressure_bar, run.temperature_k, run.kzz_cm2_s],
                axis=-1,
            )
            for run in train_runs
        ],
        axis=0,
    )
    state_values = np.concatenate([run.ymix_state.reshape(-1, run.ymix_state.shape[-1]) for run in train_runs], axis=0)
    target_values = np.concatenate(
        [run.ymix_output.reshape(-1, run.ymix_output.shape[-1]) for run in train_runs],
        axis=0,
    )
    spectrum_values = np.stack(
        [run.spectrum_flux_erg_cm2_s_nm for run in train_runs],
        axis=0,
    )
    global_static = np.stack(
        [
            np.array(
                [
                    resolve_conditioning_inputs(
                        raw_global_inputs=run.globals,
                        config=config,
                        required_global_inputs=list(config["data_spec"]["required_global_inputs"]),
                    )[name]
                    for name in global_static_order
                ],
                dtype=np.float64,
            )
            for run in train_runs
        ],
        axis=0,
    )
    transition_sampling = effective_transition_sampling(config)
    global_methods = [
        config["normalization"]["global_methods"][name]
        for name in global_static_order
    ]

    # Fit dt normalization from padded trajectory shells so each training split
    # uses a single consistent dt feature transform.
    dt_stats = fit_log10_dt_normalization(
        time_s=np.stack(
            [np.pad(run.time_s, (0, max(r.time_s.size for r in train_runs) - run.time_s.size), mode="constant") for run in train_runs],
            axis=0,
        ),
        valid_steps_mask=np.stack(
            [
                np.pad(np.ones(run.time_s.shape, dtype=bool), (0, max(r.time_s.size for r in train_runs) - run.time_s.size), mode="constant")
                for run in train_runs
            ],
            axis=0,
        ),
        dt_min_s=float(transition_sampling["dt_min_s"]),
        dt_max_s=float(transition_sampling["dt_max_s"]),
        min_future_saved_steps=int(transition_sampling["min_future_saved_steps"]),
    )

    sequence_methods = config["normalization"]["sequence_methods"]
    sequence_blocks = []
    for i, name in enumerate(("pressure_bar", "temperature_k", "kzz_cm2_s")):
        method = sequence_methods[name]
        feature = sequence_static[:, i : i + 1]
        if method == "standard":
            sequence_blocks.append(_fit_standard(feature))
        elif method == "log-standard":
            sequence_blocks.append(_fit_log_standard(feature, floor=1.0e-30))
        elif method == "none":
            sequence_blocks.append(_fit_none(feature))
        else:
            raise ValueError(f"Unsupported sequence normalization method: {method}")

    return {
        "sequence_static": {
            "feature_order": ["pressure_bar", "temperature_k", "kzz_cm2_s"],
            "blocks": sequence_blocks,
        },
        "state": _fit_block_by_method(
            state_values,
            method=config["normalization"]["state_method"],
            floor=state_floor,
        ),
        "target": _fit_block_by_method(
            target_values,
            method=config["normalization"]["target_method"],
            floor=state_floor,
        ),
        "global_static": _fit_mixed_block(global_static, global_methods),
        "log10_dt_s": _fit_log10_dt_block(
            dt_stats,
            method=config["normalization"]["log10_dt_method"],
        ),
        "spectrum": _fit_block_by_method(
            spectrum_values,
            method=config["normalization"]["spectrum_method"],
            floor=spectrum_floor,
        ),
        "spectrum_wavelength_nm": spectrum_grid_nm.tolist(),
    }


def _apply_sequence_static_normalization(x: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    """Apply the configured normalization blocks to sequence-static features."""
    blocks = payload["blocks"]
    parts = [apply_block(x[..., i : i + 1], block) for i, block in enumerate(blocks)]
    return np.concatenate(parts, axis=-1)


def load_raw_equilibrium_run(
    source: str | Path | h5py.Group,
    *,
    config: dict[str, Any],
    run_id: str | None = None,
) -> RawEquilibriumRun:
    """Load one raw equilibrium HDF5 run.

    *source* may be a file path (legacy per-file layout) or an already-opened
    ``h5py.Group`` from a consolidated ``runs.h5``.  When *source* is a group,
    *run_id* must be supplied explicitly.
    """
    requested_output_species = list(config["data_spec"]["output_species"])

    def _extract(handle: h5py.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, dict[str, float]]:
        pressure_bar = np.asarray(handle["inputs/pressure_bar"], dtype=np.float64)
        temperature_k = np.asarray(handle["inputs/temperature_k"], dtype=np.float64)
        element_input_order = _decode_species(np.asarray(handle["inputs/element_input_order"]))
        elemental_abundances_x_h = np.asarray(handle["inputs/elemental_abundances_x_h"], dtype=np.float64)
        gravity_cm_s2 = np.asarray(handle["inputs/gravity_cm_s2"], dtype=np.float64)
        stored_output_species = _decode_species(np.asarray(handle["inputs/output_species"]))
        equilibrium_ymix = np.asarray(handle["equilibrium/ymix"], dtype=np.float64)
        globals_map = {
            key: float(np.asarray(handle[f"globals/{key}"]))
            for key in handle["globals"].keys()
        }
        element_order = list(config["data_spec"]["element_input_order"])
        element_indices = [element_input_order.index(name) for name in element_order]
        elemental_profile = elemental_abundances_x_h[:, element_indices]
        return pressure_bar, temperature_k, elemental_profile, gravity_cm_s2, stored_output_species, equilibrium_ymix, globals_map

    if isinstance(source, h5py.Group):
        pressure_bar, temperature_k, elemental_profile, gravity_profile, stored_output_species, equilibrium_ymix, globals_map = _extract(source)
        label = run_id or "unknown"
    else:
        with h5py.File(source, "r") as handle:
            pressure_bar, temperature_k, elemental_profile, gravity_profile, stored_output_species, equilibrium_ymix, globals_map = _extract(handle)
        label = Path(source).stem
        if run_id is None:
            run_id = label

    if not np.all(np.isfinite(pressure_bar)):
        raise ValueError(f"{label}: non-finite pressure values detected.")
    if not np.all(np.isfinite(temperature_k)):
        raise ValueError(f"{label}: non-finite temperature values detected.")
    if not np.all(np.isfinite(elemental_profile)):
        raise ValueError(f"{label}: non-finite elemental abundance values detected.")
    if not np.all(np.isfinite(gravity_profile)):
        raise ValueError(f"{label}: non-finite gravity values detected.")
    if elemental_profile.shape[0] != pressure_bar.size:
        raise ValueError(f"{label}: elemental abundance profile does not match pressure grid.")
    if gravity_profile.shape != pressure_bar.shape:
        raise ValueError(f"{label}: gravity profile does not match pressure grid.")
    _require_column_constant(elemental_profile, name="elemental_abundances_x_h", run_label=label)
    _require_column_constant(gravity_profile, name="gravity_cm_s2", run_label=label)
    globals_map = dict(globals_map)
    globals_map.update(
        {
            name: float(elemental_profile[0, idx])
            for idx, name in enumerate(config["data_spec"]["element_input_order"])
        }
    )
    globals_map["gravity_cm_s2"] = float(gravity_profile[0])
    output_indices = [stored_output_species.index(name) for name in requested_output_species]
    equilibrium_ymix = equilibrium_ymix[:, output_indices]
    return RawEquilibriumRun(
        run_id=run_id or label,
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        equilibrium_ymix=equilibrium_ymix,
        globals=globals_map,
        elemental_abundances_x_h=elemental_profile,
        gravity_cm_s2=gravity_profile,
    )


def _equilibrium_normalization_payload(
    train_runs: list[RawEquilibriumRun],
    *,
    config: dict[str, Any],
    global_static_order: list[str],
) -> dict[str, Any]:
    """Fit normalization statistics for the equilibrium model.

    Simpler than the transition payload: only ``sequence_static``
    (P, T), ``target`` (equilibrium ymix), and ``global_static``
    blocks are needed — no spectrum, dt, or state trajectory blocks.
    """
    state_floor = float(config["normalization"]["state_floor"])
    sequence_methods = config["normalization"]["sequence_methods"]

    sequence_static = np.concatenate(
        [np.stack([run.pressure_bar, run.temperature_k], axis=-1) for run in train_runs],
        axis=0,
    )
    target_values = np.concatenate(
        [run.equilibrium_ymix for run in train_runs],
        axis=0,
    )
    global_static = np.stack(
        [
            np.array([run.globals[name] for name in global_static_order], dtype=np.float64)
            for run in train_runs
        ],
        axis=0,
    )
    global_methods = [
        config["normalization"]["global_methods"][name]
        for name in global_static_order
    ]

    feature_names = list(sequence_methods.keys())
    sequence_blocks = []
    for i, name in enumerate(feature_names):
        method = sequence_methods[name]
        feature = sequence_static[:, i : i + 1]
        if method == "standard":
            sequence_blocks.append(_fit_standard(feature))
        elif method == "log-standard":
            sequence_blocks.append(_fit_log_standard(feature, floor=1.0e-30))
        elif method == "none":
            sequence_blocks.append(_fit_none(feature))
        else:
            raise ValueError(f"Unsupported sequence normalization method: {method}")

    return {
        "sequence_static": {
            "feature_order": feature_names,
            "blocks": sequence_blocks,
        },
        "target": _fit_block_by_method(
            target_values,
            method=config["normalization"]["target_method"],
            floor=state_floor,
        ),
        "global_static": _fit_mixed_block(global_static, global_methods),
    }


def _discover_raw_runs(raw_root: Path) -> tuple[Path | None, list[str]]:
    """Auto-detect consolidated vs per-file raw run layout.

    Returns ``(consolidated_path, run_ids)`` when a ``runs.h5`` file
    exists, or ``(None, [])`` when only per-file runs are present
    (caller should fall back to the glob-based path).
    """
    consolidated = raw_root / "runs.h5"
    if consolidated.exists():
        with h5py.File(consolidated, "r") as f:
            return consolidated, sorted(f.keys())
    return None, []


def preprocess_equilibrium_dataset(
    config: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Convert raw equilibrium runs into training tensors (no trajectory/spectrum).

    Outputs per split: ``sequence_inputs.npy`` (P, T normalized),
    ``target_outputs.npy`` (equilibrium ymix, log-standard normalized),
    ``global_inputs.npy`` (conditioning scalars, mixed normalized).
    Also writes ``normalization.json``, ``data_contract.json``,
    ``splits.json``, and ``processed_manifest.json`` to ``processed_root``.
    """
    LOGGER.info("Preprocessing equilibrium dataset")
    raw_root = resolve_path(config["paths"]["raw_root"], project_root)
    processed_root = resolve_path(config["paths"]["processed_root"], project_root)
    ensure_dir(processed_root)

    consolidated_path, consolidated_ids = _discover_raw_runs(raw_root)
    if consolidated_path is not None:
        LOGGER.info("Reading %d runs from consolidated %s", len(consolidated_ids), consolidated_path)
        raw_source_files: list[Path] = [consolidated_path]
        with h5py.File(consolidated_path, "r") as f:
            raw_runs_unfiltered = [
                load_raw_equilibrium_run(f[rid], config=config, run_id=rid)
                for rid in consolidated_ids
            ]
    else:
        raw_run_files = sorted((raw_root / "runs").glob("run_*.h5"))
        if not raw_run_files:
            raise FileNotFoundError(
                f"No raw run files found under {raw_root / 'runs'} and no runs.h5 found."
            )
        LOGGER.info("Found %d raw run files in %s", len(raw_run_files), raw_root / "runs")
        raw_source_files = list(raw_run_files)
        raw_runs_unfiltered = [load_raw_equilibrium_run(path, config=config) for path in raw_run_files]

    # Filter out runs where VULCAN produced non-finite equilibrium mixing ratios.
    # This can happen for extreme parameter combinations (very low abundances).
    raw_runs = []
    n_dropped = 0
    for run in raw_runs_unfiltered:
        if np.all(np.isfinite(run.equilibrium_ymix)):
            raw_runs.append(run)
        else:
            n_dropped += 1
            LOGGER.warning(
                "Dropping run %s: non-finite equilibrium mixing ratios detected.", run.run_id,
            )
    if n_dropped > 0:
        LOGGER.info(
            "Dropped %d / %d runs with non-finite equilibrium data.",
            n_dropped, len(raw_runs_unfiltered),
        )
    if not raw_runs:
        raise RuntimeError("All raw runs contain non-finite data; nothing to preprocess.")

    split_indices = _split_indices(len(raw_runs), config=config)
    train_runs = [raw_runs[i] for i in split_indices["train"]]
    global_static_order = list(config["data_spec"]["global_static_feature_order"])
    normalization = _equilibrium_normalization_payload(
        train_runs, config=config, global_static_order=global_static_order,
    )
    sequence_feature_order = list(config["data_spec"]["sequence_static_feature_order"])

    for split_name, indices in split_indices.items():
        split_dir = ensure_dir(processed_root / split_name)
        runs = [raw_runs[i] for i in indices]
        nz = runs[0].pressure_bar.size
        target_dim = runs[0].equilibrium_ymix.shape[-1]
        n_seq_features = len(sequence_feature_order)

        sequence_inputs = np.zeros((len(runs), nz, n_seq_features), dtype=np.float32)
        target_outputs = np.zeros((len(runs), nz, target_dim), dtype=np.float32)
        global_inputs = np.zeros((len(runs), len(global_static_order)), dtype=np.float32)
        run_ids: list[str] = []

        for idx, run in enumerate(runs):
            static = np.stack([run.pressure_bar, run.temperature_k], axis=-1)
            sequence_inputs[idx] = _apply_sequence_static_normalization(
                static, normalization["sequence_static"]
            ).astype(np.float32)
            target_outputs[idx] = apply_block(
                run.equilibrium_ymix, normalization["target"]
            ).astype(np.float32)
            global_vector = np.array(
                [run.globals[name] for name in global_static_order], dtype=np.float64,
            )
            global_inputs[idx] = apply_mixed_block(
                global_vector[None, :], normalization["global_static"]
            )[0].astype(np.float32)
            run_ids.append(run.run_id)

        metadata = {
            "processed_data_version": PROCESSED_DATA_VERSION,
            "task_kind": task_kind(config),
            "model_type": "equilibrium",
            "split": split_name,
            "num_runs": len(runs),
            "num_levels": nz,
            "sequence_feature_order": sequence_feature_order,
            "output_species_order": list(config["data_spec"]["output_species"]),
            "global_static_feature_order": global_static_order,
            "element_input_order": list(config["data_spec"]["element_input_order"]),
        }
        np.save(split_dir / "sequence_inputs.npy", sequence_inputs)
        np.save(split_dir / "target_outputs.npy", target_outputs)
        np.save(split_dir / "global_inputs.npy", global_inputs)
        (split_dir / "run_ids.json").write_text(
            json.dumps(run_ids, indent=2) + "\n", encoding="utf-8",
        )
        (split_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
        )

    data_contract = {
        "processed_data_version": PROCESSED_DATA_VERSION,
        "task_kind": task_kind(config),
        "model_type": "equilibrium",
        "state_species_order": list(config["data_spec"]["state_species"]),
        "output_species_order": list(config["data_spec"]["output_species"]),
        "element_input_order": list(config["data_spec"]["element_input_order"]),
        "sequence_static_feature_order": sequence_feature_order,
        "global_static_feature_order": global_static_order,
        "sequence_dim": n_seq_features,
        "target_dim": len(config["data_spec"]["output_species"]),
        "global_dim": len(global_static_order),
    }
    (processed_root / "normalization.json").write_text(
        json.dumps(normalization, indent=2) + "\n", encoding="utf-8",
    )
    (processed_root / "data_contract.json").write_text(
        json.dumps(data_contract, indent=2) + "\n", encoding="utf-8",
    )
    (processed_root / "splits.json").write_text(
        json.dumps(split_indices, indent=2) + "\n", encoding="utf-8",
    )
    processed_manifest = {
        "raw_files": manifest_for_files(raw_source_files),
        "splits": split_indices,
        "task_kind": task_kind(config),
        "model_type": "equilibrium",
        "normalization_fingerprint": fingerprint_payload(normalization),
    }
    (processed_root / "processed_manifest.json").write_text(
        json.dumps(processed_manifest, indent=2) + "\n", encoding="utf-8",
    )
    LOGGER.info("Equilibrium preprocessing complete -> %s", processed_root)
    return {
        "processed_root": str(processed_root),
        "normalization": normalization,
        "data_contract": data_contract,
        "splits": split_indices,
    }


def preprocess_raw_dataset(
    config: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Convert raw runs into training tensors. Dispatches by model type."""
    if is_equilibrium(config):
        return preprocess_equilibrium_dataset(config, project_root=project_root)
    LOGGER.info("Preprocessing full_vulcan dataset")
    raw_root = resolve_path(config["paths"]["raw_root"], project_root)
    processed_root = resolve_path(config["paths"]["processed_root"], project_root)
    ensure_dir(processed_root)

    spectrum_grid_nm = fixed_wavelength_grid(
        float(config["stellar_spectrum"]["wavelength_min_nm"]),
        float(config["stellar_spectrum"]["wavelength_max_nm"]),
        int(config["stellar_spectrum"]["num_bins"]),
    )

    consolidated_path, consolidated_ids = _discover_raw_runs(raw_root)
    if consolidated_path is not None:
        LOGGER.info("Reading %d runs from consolidated %s", len(consolidated_ids), consolidated_path)
        raw_source_files: list[Path] = [consolidated_path]
        with h5py.File(consolidated_path, "r") as f:
            raw_runs = [
                load_raw_run(f[rid], config=config, spectrum_grid_nm=spectrum_grid_nm, run_id=rid)
                for rid in consolidated_ids
            ]
    else:
        raw_run_files = sorted((raw_root / "runs").glob("run_*.h5"))
        if not raw_run_files:
            raise FileNotFoundError(
                f"No raw run files found under {raw_root / 'runs'} and no runs.h5 found."
            )
        LOGGER.info("Found %d raw run files in %s", len(raw_run_files), raw_root / "runs")
        raw_source_files = list(raw_run_files)
        raw_runs = [load_raw_run(path, config=config, spectrum_grid_nm=spectrum_grid_nm) for path in raw_run_files]
    split_indices = _split_indices(len(raw_runs), config=config)
    train_runs = [raw_runs[i] for i in split_indices["train"]]
    target_modes = {run.target_mode for run in raw_runs}
    if len(target_modes) != 1:
        raise ValueError(f"Raw dataset mixes multiple target modes: {sorted(target_modes)}.")
    target_mode = target_modes.pop()
    global_static_order = list(config["data_spec"]["global_static_feature_order"])
    normalization = _normalization_payload(
        train_runs,
        config=config,
        global_static_order=global_static_order,
        spectrum_grid_nm=spectrum_grid_nm,
    )

    for split_name, indices in split_indices.items():
        split_dir = ensure_dir(processed_root / split_name)
        runs = [raw_runs[i] for i in indices]
        max_steps = max(run.time_s.size for run in runs)
        nz = runs[0].pressure_bar.size
        state_dim = runs[0].ymix_state.shape[-1]
        target_dim = runs[0].ymix_output.shape[-1]

        sequence_inputs = np.zeros((len(runs), nz, 3), dtype=np.float32)
        state_trajectories = np.zeros((len(runs), max_steps, nz, state_dim), dtype=np.float32)
        target_outputs = np.zeros((len(runs), max_steps, nz, target_dim), dtype=np.float32)
        global_inputs = np.zeros((len(runs), len(global_static_order)), dtype=np.float32)
        spectrum_inputs = np.zeros((len(runs), spectrum_grid_nm.size), dtype=np.float32)
        time_s = np.zeros((len(runs), max_steps), dtype=np.float64)
        valid_steps_mask = np.zeros((len(runs), max_steps), dtype=bool)
        run_ids: list[str] = []

        for idx, run in enumerate(runs):
            static = np.stack([run.pressure_bar, run.temperature_k, run.kzz_cm2_s], axis=-1)
            sequence_inputs[idx] = _apply_sequence_static_normalization(static, normalization["sequence_static"]).astype(np.float32)
            state_trajectories[idx, : run.time_s.size] = apply_block(run.ymix_state, normalization["state"]).astype(np.float32)
            target_outputs[idx, : run.time_s.size] = apply_block(run.ymix_output, normalization["target"]).astype(np.float32)
            static_inputs = resolve_conditioning_inputs(
                raw_global_inputs=run.globals,
                config=config,
                required_global_inputs=list(config["data_spec"]["required_global_inputs"]),
            )
            global_vector = np.array([static_inputs[name] for name in global_static_order], dtype=np.float64)
            global_inputs[idx] = apply_mixed_block(global_vector[None, :], normalization["global_static"])[0].astype(np.float32)
            spectrum_inputs[idx] = apply_block(run.spectrum_flux_erg_cm2_s_nm[None, :], normalization["spectrum"])[0].astype(np.float32)
            time_s[idx, : run.time_s.size] = run.time_s
            valid_steps_mask[idx, : run.time_s.size] = True
            run_ids.append(run.run_id)

        metadata = {
            "processed_data_version": PROCESSED_DATA_VERSION,
            "task_kind": task_kind(config),
            "split": split_name,
            "target_mode": target_mode,
            "num_runs": len(runs),
            "num_levels": nz,
            "max_steps": int(max_steps),
            "sequence_feature_order": ["pressure_bar", "temperature_k", "kzz_cm2_s"],
            "state_species_order": list(config["data_spec"]["state_species"]),
            "output_species_order": list(config["data_spec"]["output_species"]),
            "element_input_order": list(config["data_spec"]["element_input_order"]),
            "global_feature_order": list(config["data_spec"]["global_feature_order"]),
            "global_static_feature_order": list(config["data_spec"]["global_static_feature_order"]),
            "dt_feature_index": int(config["data_spec"]["dt_feature_index"]),
            "spectrum_num_bins": int(spectrum_grid_nm.size),
        }
        np.save(split_dir / "sequence_inputs.npy", sequence_inputs)
        np.save(split_dir / "state_trajectories.npy", state_trajectories)
        np.save(split_dir / "target_outputs.npy", target_outputs)
        np.save(split_dir / "global_inputs.npy", global_inputs)
        np.save(split_dir / "spectrum_inputs.npy", spectrum_inputs)
        np.save(split_dir / "time_s.npy", time_s)
        np.save(split_dir / "valid_steps_mask.npy", valid_steps_mask)
        (split_dir / "run_ids.json").write_text(json.dumps(run_ids, indent=2) + "\n", encoding="utf-8")
        (split_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    data_contract = {
        "processed_data_version": PROCESSED_DATA_VERSION,
        "task_kind": task_kind(config),
        "target_mode": target_mode,
        "state_species_order": list(config["data_spec"]["state_species"]),
        "output_species_order": list(config["data_spec"]["output_species"]),
        "element_input_order": list(config["data_spec"]["element_input_order"]),
        "sequence_static_feature_order": ["pressure_bar", "temperature_k", "kzz_cm2_s"],
        "global_feature_order": list(config["data_spec"]["global_feature_order"]),
        "global_static_feature_order": list(config["data_spec"]["global_static_feature_order"]),
        "dt_feature_index": int(config["data_spec"]["dt_feature_index"]),
        "sequence_dim": 3 + len(config["data_spec"]["state_species"]),
        "target_dim": len(config["data_spec"]["output_species"]),
        "spectrum_dim": int(spectrum_grid_nm.size),
        "spectrum_wavelength_nm": spectrum_grid_nm.tolist(),
    }
    (processed_root / "normalization.json").write_text(
        json.dumps(normalization, indent=2) + "\n",
        encoding="utf-8",
    )
    (processed_root / "data_contract.json").write_text(
        json.dumps(data_contract, indent=2) + "\n",
        encoding="utf-8",
    )
    (processed_root / "splits.json").write_text(
        json.dumps(split_indices, indent=2) + "\n",
        encoding="utf-8",
    )
    processed_manifest = {
        "raw_files": manifest_for_files(raw_source_files),
        "splits": split_indices,
        "task_kind": task_kind(config),
        "target_mode": target_mode,
        "normalization_fingerprint": fingerprint_payload(normalization),
    }
    (processed_root / "processed_manifest.json").write_text(
        json.dumps(processed_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Full VULCAN preprocessing complete -> %s", processed_root)
    return {
        "processed_root": str(processed_root),
        "normalization": normalization,
        "data_contract": data_contract,
        "splits": split_indices,
    }
