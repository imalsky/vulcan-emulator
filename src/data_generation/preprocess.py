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
   shared JSON metadata under the dataset-level ``info`` directory, and
   provenance manifests to disk.

Two top-level entry points handle the two chemistry types:

* ``preprocess_fastchem_dataset`` — for FastChem chemistry
  (no spectrum).
* ``preprocess_raw_dataset`` — for VULCAN chemistry
  (final-state + spectrum).

``PROCESSED_DATA_VERSION`` is bumped whenever the on-disk tensor
layout changes in a backwards-incompatible way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from ..constants import ELEMENT_INPUT_ORDER, PROCESSED_DATA_VERSION
from ..utils.config import (
    get_chemistry_type,
    get_model_type,
    resolve_conditioning_inputs,
    uses_fastchem,
)
from ..utils.helpers import ensure_dir, get_logger, resolve_path
from ..utils.provenance import fingerprint_payload, manifest_for_files
from .data_loader import processed_info_dir

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class RawRun:
    """Raw full-VULCAN run loaded from HDF5 before normalization.

    Fields mirror the HDF5 layout produced by ``generation.write_raw_run_hdf5``.
    Species columns are already reordered to match the config's
    ``output_species`` contract.
    """
    run_id: str
    pressure_bar: np.ndarray
    temperature_k: np.ndarray
    kzz_cm2_s: np.ndarray
    final_ymix_output: np.ndarray
    globals: dict[str, float]
    elemental_abundances_frac: np.ndarray
    gravity_cm_s2: np.ndarray


@dataclass(frozen=True)
class RawEquilibriumRun:
    """Simplified raw run for equilibrium-only models (no trajectory/spectrum)."""
    run_id: str
    pressure_bar: np.ndarray
    temperature_k: np.ndarray
    equilibrium_ymix: np.ndarray
    globals: dict[str, float]
    elemental_abundances_frac: np.ndarray
    gravity_cm_s2: np.ndarray



def _decode_species(values: np.ndarray) -> list[str]:
    """Decode species labels loaded from HDF5 string datasets.

    Parameters
    ----------
    values : np.ndarray
        One-dimensional object, bytes, or string array read from HDF5.

    Returns
    -------
    list[str]
        Python string labels in the original stored order.
    """
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
    """Reject vertical profiles that should remain column-constant.

    Parameters
    ----------
    values : np.ndarray
        One-dimensional profile ``(nz,)`` or stacked profile ``(nz, ncol)`` to
        check.
    name : str
        Field name used in validation errors.
    run_label : str
        Run identifier used in validation errors.

    Returns
    -------
    None
        The function returns silently when the supplied values are vertically
        constant within tolerance.
    """
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


def _elemental_conditioning_globals(
    *,
    elemental_profile: np.ndarray,
    run_label: str,
) -> dict[str, float]:
    """Reduce a per-level elemental profile to profile-global conditioning scalars.

    Parameters
    ----------
    elemental_profile : np.ndarray
        Elemental abundance profile with shape ``(nz, n_elements)`` ordered as
        ``ELEMENT_INPUT_ORDER``.
    run_label : str
        Run identifier used in validation errors.

    Returns
    -------
    dict[str, float]
        Mapping from elemental input name to one column-constant ``X/H`` value
        for the run.
    """
    profile = np.asarray(elemental_profile, dtype=np.float64)
    if profile.ndim != 2 or profile.shape[1] != len(ELEMENT_INPUT_ORDER):
        raise ValueError(
            f"{run_label}: elemental_abundances_frac must have shape (nz, {len(ELEMENT_INPUT_ORDER)})."
        )
    _require_column_constant(profile, name="elemental_abundances_frac", run_label=run_label)
    resolved = {
        name: float(profile[0, idx])
        for idx, name in enumerate(ELEMENT_INPUT_ORDER)
    }
    if not np.all(np.isfinite(list(resolved.values()))):
        raise ValueError(f"{run_label}: non-finite elemental conditioning globals detected.")
    return resolved


def _require_elemental_conditioning_globals(
    *,
    globals_map: dict[str, float],
    run_label: str,
) -> dict[str, float]:
    """Validate that the required elemental globals are present and finite.

    Parameters
    ----------
    globals_map : dict[str, float]
        Raw global-conditioning mapping for a run.
    run_label : str
        Run identifier used in validation errors.

    Returns
    -------
    dict[str, float]
        Subset of ``globals_map`` containing the required elemental inputs in
        ``ELEMENT_INPUT_ORDER``.
    """
    missing = [name for name in ELEMENT_INPUT_ORDER if name not in globals_map]
    if missing:
        raise ValueError(
            f"{run_label}: raw globals are missing required elemental-conditioning inputs "
            f"{missing}. Regenerate raw data with the current chemistry contract."
        )
    resolved = {
        name: float(globals_map[name])
        for name in ELEMENT_INPUT_ORDER
    }
    if not np.all(np.isfinite(list(resolved.values()))):
        raise ValueError(f"{run_label}: non-finite elemental-conditioning globals detected.")
    return resolved


def _fit_standard(arr: np.ndarray) -> dict[str, Any]:
    """Fit mean and standard deviation for standard (z-score) scaling.

    Parameters
    ----------
    arr : np.ndarray
        Input array whose last dimension enumerates feature columns.

    Returns
    -------
    dict[str, Any]
        Normalization payload containing method ``"standard"`` plus per-column
        means and standard deviations. Leading dimensions are flattened before
        computing statistics.
    """
    arr2 = np.reshape(arr, (-1, arr.shape[-1]))
    mean = np.mean(arr2, axis=0)
    std = np.std(arr2, axis=0)
    std = np.where(std < 1.0e-8, 1.0, std)
    return {"method": "standard", "mean": mean.tolist(), "std": std.tolist()}


def _fit_none(arr: np.ndarray) -> dict[str, Any]:
    """Build a no-op normalization block for features left in physical units.

    Parameters
    ----------
    arr : np.ndarray
        Feature array whose last dimension defines the number of feature
        columns that need identity normalization statistics.

    Returns
    -------
    dict[str, Any]
        Normalization payload with method ``"none"`` plus zero means and unit
        standard deviations for each feature column.
    """
    dim = int(arr.shape[-1]) if arr.ndim >= 1 else 1
    return {"method": "none", "mean": [0.0] * dim, "std": [1.0] * dim}


def _fit_log_standard(arr: np.ndarray, *, floor: float) -> dict[str, Any]:
    """Fit standard scaling in log10 space with a lower floor.

    Parameters
    ----------
    arr : np.ndarray
        Input array whose last dimension enumerates strictly positive feature
        columns.
    floor : float
        Lower bound applied before taking ``log10`` to avoid ``-inf`` values.

    Returns
    -------
    dict[str, Any]
        Normalization payload containing method ``"log-standard"``, per-column
        log-space means and standard deviations, and the applied floor.
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


def _fit_log_minmax(arr: np.ndarray, *, floor: float) -> dict[str, Any]:
    """Fit min-max scaling in log10 space.

    The forward transform is:

        normalized = (log10(clip(x, floor)) - log10_min) / (log10_max - log10_min)

    This maps the smallest values to 0 and the largest to 1.

    Parameters
    ----------
    arr : np.ndarray
        Input array whose last dimension enumerates feature columns.
    floor : float
        Lower bound applied before taking ``log10``.

    Returns
    -------
    dict[str, Any]
        Normalization payload with per-column log-space min and max values.
    """
    arr2 = np.reshape(np.clip(arr, floor, None), (-1, arr.shape[-1]))
    log10_arr = np.log10(arr2)
    log10_min = np.min(log10_arr, axis=0)
    log10_max = np.max(log10_arr, axis=0)
    span = log10_max - log10_min
    span = np.where(span < 1.0e-8, 1.0, span)
    return {
        "method": "log-minmax",
        "log10_min": log10_min.tolist(),
        "log10_max": log10_max.tolist(),
        "span": span.tolist(),
        "floor": float(floor),
    }


def apply_block(x: np.ndarray, block: dict[str, Any]) -> np.ndarray:
    """Apply one normalization block (forward direction) to an array.

    Parameters
    ----------
    x : np.ndarray
        Input array whose trailing dimension matches the statistics stored in
        ``block``.
    block : dict[str, Any]
        Normalization payload with method name and any required statistics.

    Returns
    -------
    np.ndarray
        Normalized array with the same shape as ``x``.
    """
    method = block["method"]
    if method == "none":
        return np.asarray(x, dtype=np.float64)
    if method == "log-minmax":
        floor = float(block["floor"])
        log10_min = np.asarray(block["log10_min"], dtype=np.float64)
        span = np.asarray(block["span"], dtype=np.float64)
        transformed = np.log10(np.clip(np.asarray(x, dtype=np.float64), floor, None))
        return (transformed - log10_min) / span
    mean = np.asarray(block["mean"], dtype=np.float64)
    std = np.asarray(block["std"], dtype=np.float64)
    if method == "standard":
        return (np.asarray(x, dtype=np.float64) - mean) / std
    if method == "log-standard":
        floor = float(block["floor"])
        transformed = np.log10(np.clip(np.asarray(x, dtype=np.float64), floor, None))
        return (transformed - mean) / std
    raise ValueError(f"Unsupported normalization method: {method}")


def inverse_block(x: np.ndarray, block: dict[str, Any]) -> np.ndarray:
    """Invert one normalization block back to physical (original) space.

    Parameters
    ----------
    x : np.ndarray
        Normalized array whose trailing dimension matches the statistics stored
        in ``block``.
    block : dict[str, Any]
        Normalization payload with method name and any required statistics.

    Returns
    -------
    np.ndarray
        Array with the same shape as ``x`` restored to physical units.
    """
    method = block["method"]
    if method == "none":
        return np.asarray(x, dtype=np.float64)
    if method == "log-minmax":
        log10_min = np.asarray(block["log10_min"], dtype=np.float64)
        span = np.asarray(block["span"], dtype=np.float64)
        log10_x = np.asarray(x, dtype=np.float64) * span + log10_min
        return np.power(10.0, log10_x)
    mean = np.asarray(block["mean"], dtype=np.float64)
    std = np.asarray(block["std"], dtype=np.float64)
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
    """Fit one normalization block selected by method name.

    Parameters
    ----------
    arr : np.ndarray
        Input array whose last dimension enumerates features.
    method : str
        Normalization method name: ``"standard"``, ``"log-standard"``, or
        ``"none"``.
    floor : float or None, optional
        Positive floor used by ``"log-standard"`` blocks before taking
        ``log10``.

    Returns
    -------
    dict[str, Any]
        Normalization payload compatible with ``apply_block``.
    """
    if method == "standard":
        return _fit_standard(arr)
    if method == "log-standard":
        if floor is None:
            raise ValueError("log-standard normalization requires a positive floor.")
        return _fit_log_standard(arr, floor=float(floor))
    if method == "log-minmax":
        if floor is None:
            raise ValueError("log-minmax normalization requires a positive floor.")
        return _fit_log_minmax(arr, floor=float(floor))
    if method == "none":
        return _fit_none(arr)
    raise ValueError(f"Unsupported normalization method: {method}")


def _fit_mixed_block(arr: np.ndarray, methods: list[str]) -> dict[str, Any]:
    """Fit a mixed normalization block with per-column methods.

    Parameters
    ----------
    arr : np.ndarray
        Input array whose last dimension enumerates feature columns.
    methods : list[str]
        Per-column normalization methods aligned with the trailing dimension of
        ``arr``.

    Returns
    -------
    dict[str, Any]
        Mixed normalization payload storing one method, mean, standard
        deviation, and optional floor per feature column.
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
            raw_std = float(np.std(column))
            # Clamp to 1.0 (not 1e-8) for near-constant features, matching
            # _fit_standard.  Using 1e-8 would amplify float-precision noise
            # into enormous normalized values at inference time.
            std = raw_std if raw_std >= 1.0e-8 else 1.0
            means.append(mean)
            stds.append(std)
            floors.append(None)
            transformed_columns.append((column - mean) / std)
        elif method == "log-standard":
            floor = 1.0e-30
            log_column = np.log10(np.clip(column, floor, None))
            mean = float(np.mean(log_column))
            raw_std = float(np.std(log_column))
            std = raw_std if raw_std >= 1.0e-8 else 1.0
            means.append(mean)
            stds.append(std)
            floors.append(floor)
            transformed_columns.append((log_column - mean) / std)
        elif method == "log-minmax":
            floor = 1.0e-30
            log_column = np.log10(np.clip(column, floor, None))
            log_min = float(np.min(log_column))
            log_max = float(np.max(log_column))
            raw_span = log_max - log_min
            span = raw_span if raw_span >= 1.0e-8 else 1.0
            means.append(log_min)
            stds.append(span)
            floors.append(floor)
            transformed_columns.append((log_column - log_min) / span)
        else:
            raise ValueError(f"Unsupported mixed normalization method: {method}")
    return {"method": "mixed", "methods": methods, "mean": means, "std": stds, "floor": floors}


def apply_mixed_block(x: np.ndarray, block: dict[str, Any]) -> np.ndarray:
    """Apply a mixed per-feature normalization block.

    Parameters
    ----------
    x : np.ndarray
        Input array whose last dimension enumerates feature columns.
    block : dict[str, Any]
        Mixed normalization payload with per-column methods, means, stds, and
        optional floors.

    Returns
    -------
    np.ndarray
        Normalized array with the same shape as ``x``.
    """
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
        elif method == "log-minmax":
            outputs.append((np.log10(np.clip(column, float(floor), None)) - mean) / std)
        else:
            raise ValueError(f"Unsupported mixed normalization method: {method}")
    return np.stack(outputs, axis=-1)


def inverse_mixed_block(x: np.ndarray, block: dict[str, Any]) -> np.ndarray:
    """Invert a mixed per-feature normalization block.

    Parameters
    ----------
    x : np.ndarray
        Normalized array in model space.
    block : dict[str, Any]
        Mixed normalization payload that was previously applied with
        ``apply_mixed_block``.

    Returns
    -------
    np.ndarray
        Array restored to physical feature space with the same shape as
        ``x``.
    """
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
        elif method == "log-minmax":
            outputs.append(np.power(10.0, column * std + mean))
        else:
            raise ValueError(f"Unsupported mixed normalization method: {method}")
    return np.stack(outputs, axis=-1)



def load_raw_run(
    source: str | Path | h5py.Group,
    *,
    config: dict[str, Any],
    run_id: str | None = None,
) -> RawRun:
    """Load one raw HDF5 full-VULCAN run and align it to the configured species contract."""
    requested_output_species = list(config["data_spec"]["output_species"])

    def _extract(handle: h5py.Group) -> dict[str, Any]:
        """Read the raw HDF5 datasets needed to build one aligned run record."""
        return {
            "pressure_bar": np.asarray(handle["inputs/pressure_bar"], dtype=np.float64),
            "temperature_k": np.asarray(handle["inputs/temperature_k"], dtype=np.float64),
            "kzz_cm2_s": np.asarray(handle["inputs/kzz_cm2_s"], dtype=np.float64),
            "element_input_order": _decode_species(np.asarray(handle["inputs/element_input_order"])),
            "elemental_abundances_frac": np.asarray(handle["inputs/elemental_abundances_frac"], dtype=np.float64),
            "gravity_cm_s2": np.asarray(handle["inputs/gravity_cm_s2"], dtype=np.float64),
            "stored_output_species": _decode_species(np.asarray(handle["inputs/output_species"])),
            "final_ymix_output": np.asarray(handle["final_state/ymix_output"], dtype=np.float64),
            "globals_map": {
                key: float(np.asarray(handle[f"globals/{key}"]))
                for key in handle["globals"].keys()
            },
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
    final_ymix_output = d["final_ymix_output"]

    if not np.all(np.isfinite(pressure_bar)):
        raise ValueError(f"{label}: non-finite pressure values detected.")
    if not np.all(np.isfinite(temperature_k)):
        raise ValueError(f"{label}: non-finite temperature values detected.")
    if not np.all(np.isfinite(kzz_cm2_s)):
        raise ValueError(f"{label}: non-finite Kzz values detected.")
    if not np.all(np.isfinite(d["elemental_abundances_frac"])):
        raise ValueError(f"{label}: non-finite elemental abundance values detected.")
    if not np.all(np.isfinite(d["gravity_cm_s2"])):
        raise ValueError(f"{label}: non-finite gravity values detected.")
    if final_ymix_output.shape[0] != pressure_bar.size:
        raise ValueError(f"{label}: final_ymix_output vertical dimension does not match pressure grid.")
    if d["elemental_abundances_frac"].shape[0] != pressure_bar.size:
        raise ValueError(f"{label}: elemental abundance profile does not match pressure grid.")
    if d["gravity_cm_s2"].shape != pressure_bar.shape:
        raise ValueError(f"{label}: gravity profile does not match pressure grid.")

    output_indices = [d["stored_output_species"].index(name) for name in requested_output_species]
    element_order = list(config["data_spec"]["element_input_order"])
    element_indices = [d["element_input_order"].index(name) for name in element_order]
    elemental_profile = d["elemental_abundances_frac"][:, element_indices]
    gravity_profile = np.asarray(d["gravity_cm_s2"], dtype=np.float64)
    element_globals = _elemental_conditioning_globals(
        elemental_profile=elemental_profile,
        run_label=label,
    )
    reduced_globals = dict(d["globals_map"])
    reduced_globals.update(element_globals)
    if "gravity_cm_s2" not in reduced_globals:
        reduced_globals["gravity_cm_s2"] = float(gravity_profile[0])
    final_ymix_output = final_ymix_output[:, output_indices]

    return RawRun(
        run_id=run_id or label,
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        kzz_cm2_s=kzz_cm2_s,
        final_ymix_output=final_ymix_output,
        globals=reduced_globals,
        elemental_abundances_frac=elemental_profile,
        gravity_cm_s2=gravity_profile,
    )


def _split_indices(num_runs: int, *, config: dict[str, Any]) -> dict[str, list[int]]:
    """Split run indices into train/val/test partitions.

    Parameters
    ----------
    num_runs : int
        Total number of available runs to partition.
    config : dict[str, Any]
        Validated config containing ``normalization.split`` settings.

    Returns
    -------
    dict[str, list[int]]
        Mapping from split names to lists of shuffled run indices.
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
) -> dict[str, Any]:
    """Fit all normalization blocks for the full-VULCAN pipeline."""
    state_floor = float(config["normalization"]["state_floor"])
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
    target_values = np.concatenate(
        [run.final_ymix_output for run in train_runs],
        axis=0,
    )
    required_global_inputs = list(config["data_spec"]["required_global_inputs"])
    resolved_global_static_rows: list[np.ndarray] = []
    for run in train_runs:
        resolved_inputs = resolve_conditioning_inputs(
            raw_global_inputs=run.globals,
            required_global_inputs=required_global_inputs,
        )
        resolved_global_static_rows.append(
            np.array(
                [resolved_inputs[name] for name in global_static_order],
                dtype=np.float64,
            )
        )
    global_static = np.stack(resolved_global_static_rows, axis=0)
    global_methods = [
        config["normalization"]["global_methods"][name]
        for name in global_static_order
    ]

    sequence_methods = config["normalization"]["sequence_methods"]
    sequence_blocks = []
    for i, name in enumerate(("pressure_bar", "temperature_k", "kzz_cm2_s")):
        method = sequence_methods[name]
        feature = sequence_static[:, i : i + 1]
        if method == "standard":
            sequence_blocks.append(_fit_standard(feature))
        elif method == "log-standard":
            sequence_blocks.append(_fit_log_standard(feature, floor=1.0e-30))
        elif method == "log-minmax":
            sequence_blocks.append(_fit_log_minmax(feature, floor=1.0e-30))
        elif method == "none":
            sequence_blocks.append(_fit_none(feature))
        else:
            raise ValueError(f"Unsupported sequence normalization method: {method}")

    return {
        "sequence_static": {
            "feature_order": ["pressure_bar", "temperature_k", "kzz_cm2_s"],
            "blocks": sequence_blocks,
        },
        "target": _fit_block_by_method(
            target_values,
            method=config["normalization"]["target_method"],
            floor=state_floor,
        ),
        "global_static": _fit_mixed_block(global_static, global_methods),
    }


def _log10_pressure_union_range(config: dict[str, Any]) -> tuple[float, float]:
    """Return the widest log10(P [bar]) range physically reachable by the config.

    The training distribution draws p_top ~ U(log) over
    pressure_top_bar_range and p_bottom ~ U(log) over
    pressure_bottom_bar_range, so any valid sampled level lies within
    [min(top_range), max(bottom_range)] on the pressure axis. The
    returned pair is in log10(bar).
    """
    top_lo = float(config["sampling"]["pressure_top_bar_range"][0])
    bot_hi = float(config["sampling"]["pressure_bottom_bar_range"][1])
    return (float(np.log10(top_lo)), float(np.log10(bot_hi)))


def _position_coord_from_pressure(
    pressure_bar: np.ndarray,
    *,
    log10_union_range: tuple[float, float],
) -> np.ndarray:
    """Map a 1-D pressure grid (bar) to normalized log10 position in [0, 1]."""
    lo, hi = log10_union_range
    if hi <= lo:
        raise ValueError("log10 pressure union range must be strictly increasing.")
    logp = np.log10(np.asarray(pressure_bar, dtype=np.float64))
    return ((logp - lo) / (hi - lo)).astype(np.float32)


def _apply_sequence_static_normalization(x: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    """Apply per-feature sequence-static normalization blocks.

    Parameters
    ----------
    x : np.ndarray
        Sequence-static feature array whose last dimension matches the order
        stored in ``payload["blocks"]``.
    payload : dict[str, Any]
        Sequence-static normalization payload containing one block per
        feature column.

    Returns
    -------
    np.ndarray
        Normalized sequence-static array with the same shape as ``x``.
    """
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

    Parameters
    ----------
    source : str, Path, or h5py.Group
        Raw equilibrium source expressed as a legacy per-file path or an
        already opened group from ``runs.h5``.
    config : dict[str, Any]
        Validated config defining the requested output-species and
        conditioning-input contract.
    run_id : str or None, optional
        Explicit run ID required when ``source`` is an ``h5py.Group``.

    Returns
    -------
    RawEquilibriumRun
        Loaded equilibrium run aligned to the configured output-species order
        and reduced global-input contract.
    """
    requested_output_species = list(config["data_spec"]["output_species"])

    def _extract(handle: h5py.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, dict[str, float]]:
        """Extract one raw FastChem-equilibrium run from an open HDF5 group.

        Parameters
        ----------
        handle : h5py.Group
            Group containing the equilibrium raw-run contract.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, dict[str, float]]
            Pressure profile, temperature profile, reordered elemental
            abundance profile, gravity profile, stored output-species labels,
            equilibrium mixing-ratio array, and raw globals mapping.
        """
        pressure_bar = np.asarray(handle["inputs/pressure_bar"], dtype=np.float64)
        temperature_k = np.asarray(handle["inputs/temperature_k"], dtype=np.float64)
        element_input_order = _decode_species(np.asarray(handle["inputs/element_input_order"]))
        elemental_abundances_frac = np.asarray(handle["inputs/elemental_abundances_frac"], dtype=np.float64)
        gravity_cm_s2 = np.asarray(handle["inputs/gravity_cm_s2"], dtype=np.float64)
        stored_output_species = _decode_species(np.asarray(handle["inputs/output_species"]))
        equilibrium_ymix = np.asarray(handle["equilibrium/ymix"], dtype=np.float64)
        globals_map = {
            key: float(np.asarray(handle[f"globals/{key}"]))
            for key in handle["globals"].keys()
        }
        element_order = list(config["data_spec"]["element_input_order"])
        element_indices = [element_input_order.index(name) for name in element_order]
        elemental_profile = elemental_abundances_frac[:, element_indices]
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
    _require_column_constant(gravity_profile, name="gravity_cm_s2", run_label=label)
    globals_map = dict(globals_map)
    globals_map.update(
        _elemental_conditioning_globals(
            elemental_profile=elemental_profile,
            run_label=label,
        )
    )
    globals_map = _require_elemental_conditioning_globals(
        globals_map=globals_map,
        run_label=label,
    )
    output_indices = [stored_output_species.index(name) for name in requested_output_species]
    equilibrium_ymix = equilibrium_ymix[:, output_indices]
    return RawEquilibriumRun(
        run_id=run_id or label,
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        equilibrium_ymix=equilibrium_ymix,
        globals=globals_map,
        elemental_abundances_frac=elemental_profile,
        gravity_cm_s2=gravity_profile,
    )


def _equilibrium_normalization_payload(
    train_runs: list[RawEquilibriumRun],
    *,
    config: dict[str, Any],
    global_static_order: list[str],
) -> dict[str, Any]:
    """Fit normalization statistics for the equilibrium model.

    Parameters
    ----------
    train_runs : list[RawEquilibriumRun]
        Training-only equilibrium runs used to fit normalization statistics.
    config : dict[str, Any]
        Validated config defining normalization methods and floors.
    global_static_order : list[str]
        Ordered global-input feature contract.

    Returns
    -------
    dict[str, Any]
        Normalization payload containing fitted blocks for sequence-static
        inputs, equilibrium targets, and global inputs.
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
        elif method == "log-minmax":
            sequence_blocks.append(_fit_log_minmax(feature, floor=1.0e-30))
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

    Parameters
    ----------
    raw_root : Path
        Raw dataset root directory.

    Returns
    -------
    tuple[Path | None, list[str]]
        Consolidated file path plus sorted run IDs when ``runs.h5`` exists, or
        ``(None, [])`` when callers should fall back to the legacy per-file
        layout.
    """
    consolidated = raw_root / "runs.h5"
    if consolidated.exists():
        with h5py.File(consolidated, "r") as f:
            return consolidated, sorted(f.keys())
    return None, []


def _write_processed_info_dir(
    info_dir: Path,
    *,
    normalization: dict[str, Any],
    data_contract: dict[str, Any],
    split_indices: dict[str, list[int]],
    raw_source_files: list[Path],
    chemistry_type: str,
    model_type: str,
) -> None:
    """Write the four shared info-dir artifacts for a processed dataset.

    Produces ``normalization.json``, ``data_contract.json``, ``splits.json``,
    and ``processed_manifest.json`` in ``info_dir`` using the same layout for
    both FastChem and VULCAN preprocessing.
    """
    (info_dir / "normalization.json").write_text(
        json.dumps(normalization, indent=2) + "\n", encoding="utf-8",
    )
    (info_dir / "data_contract.json").write_text(
        json.dumps(data_contract, indent=2) + "\n", encoding="utf-8",
    )
    (info_dir / "splits.json").write_text(
        json.dumps(split_indices, indent=2) + "\n", encoding="utf-8",
    )
    processed_manifest = {
        "raw_files": manifest_for_files(raw_source_files),
        "splits": split_indices,
        "chemistry_type": chemistry_type,
        "model_type": model_type,
        "normalization_fingerprint": fingerprint_payload(normalization),
    }
    (info_dir / "processed_manifest.json").write_text(
        json.dumps(processed_manifest, indent=2) + "\n", encoding="utf-8",
    )


def _load_raw_runs(
    raw_root: Path,
    *,
    config: dict[str, Any],
    load_run_fn: Callable[..., Any],
) -> tuple[list[Any], list[Path]]:
    """Load raw runs from either a consolidated HDF5 file or per-run files."""
    consolidated_path, consolidated_ids = _discover_raw_runs(raw_root)
    if consolidated_path is not None:
        LOGGER.info(
            "Reading %d runs from consolidated %s", len(consolidated_ids), consolidated_path,
        )
        with h5py.File(consolidated_path, "r") as f:
            raw_runs = [
                load_run_fn(f[rid], config=config, run_id=rid) for rid in consolidated_ids
            ]
        return raw_runs, [consolidated_path]

    raw_run_files = sorted((raw_root / "runs").glob("run_*.h5"))
    if not raw_run_files:
        raise FileNotFoundError(
            f"No raw run files found under {raw_root / 'runs'} and no runs.h5 found."
        )
    LOGGER.info("Found %d raw run files in %s", len(raw_run_files), raw_root / "runs")
    raw_runs = [load_run_fn(path, config=config) for path in raw_run_files]
    return raw_runs, list(raw_run_files)


def preprocess_equilibrium_dataset(
    config: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Convert raw equilibrium runs into training tensors (no trajectory/spectrum).

    Parameters
    ----------
    config : dict[str, Any]
        Validated pipeline config for FastChem equilibrium preprocessing.
    project_root : Path
        Repository root used to resolve configured raw and processed paths.

    Returns
    -------
    dict[str, Any]
        JSON-serializable artifact summary describing the processed dataset
        root, shared metadata files, and split directories written to disk.
    """
    LOGGER.info("Preprocessing FastChem dataset")
    chemistry_type = get_chemistry_type(config)
    model_type = get_model_type(config)
    raw_root = resolve_path(config["paths"]["raw_root"], project_root)
    processed_root = resolve_path(config["paths"]["processed_root"], project_root)
    ensure_dir(processed_root)
    info_dir = ensure_dir(processed_info_dir(processed_root))

    raw_runs_unfiltered, raw_source_files = _load_raw_runs(
        raw_root, config=config, load_run_fn=load_raw_equilibrium_run,
    )

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
    num_levels_range = [
        int(config["sampling"]["num_levels_range"][0]),
        int(config["sampling"]["num_levels_range"][1]),
    ]
    max_num_levels = num_levels_range[1]
    log10_p_union = _log10_pressure_union_range(config)

    for split_name, indices in split_indices.items():
        split_dir = ensure_dir(processed_root / split_name)
        runs = [raw_runs[i] for i in indices]
        target_dim = runs[0].equilibrium_ymix.shape[-1]
        n_seq_features = len(sequence_feature_order)

        sequence_inputs = np.zeros(
            (len(runs), max_num_levels, n_seq_features), dtype=np.float32
        )
        target_outputs = np.zeros(
            (len(runs), max_num_levels, target_dim), dtype=np.float32
        )
        valid_mask = np.zeros((len(runs), max_num_levels), dtype=bool)
        position_coord = np.zeros((len(runs), max_num_levels), dtype=np.float32)
        global_inputs = np.zeros((len(runs), len(global_static_order)), dtype=np.float32)
        run_ids: list[str] = []

        for idx, run in enumerate(runs):
            nz_i = int(run.pressure_bar.size)
            if nz_i < num_levels_range[0] or nz_i > num_levels_range[1]:
                raise ValueError(
                    f"Run {run.run_id} has num_levels={nz_i}, outside the configured "
                    f"range {num_levels_range}."
                )
            static = np.stack([run.pressure_bar, run.temperature_k], axis=-1)
            sequence_inputs[idx, :nz_i] = _apply_sequence_static_normalization(
                static, normalization["sequence_static"]
            ).astype(np.float32)
            target_outputs[idx, :nz_i] = apply_block(
                run.equilibrium_ymix, normalization["target"]
            ).astype(np.float32)
            position_coord[idx, :nz_i] = _position_coord_from_pressure(
                run.pressure_bar, log10_union_range=log10_p_union,
            )
            valid_mask[idx, :nz_i] = True
            global_vector = np.array(
                [run.globals[name] for name in global_static_order], dtype=np.float64,
            )
            global_inputs[idx] = apply_mixed_block(
                global_vector[None, :], normalization["global_static"]
            )[0].astype(np.float32)
            run_ids.append(run.run_id)

        metadata = {
            "processed_data_version": PROCESSED_DATA_VERSION,
            "chemistry_type": chemistry_type,
            "model_type": model_type,
            "split": split_name,
            "num_runs": len(runs),
            "max_num_levels": max_num_levels,
            "num_levels_range": num_levels_range,
            "sequence_feature_order": sequence_feature_order,
            "output_species_order": list(config["data_spec"]["output_species"]),
            "global_static_feature_order": global_static_order,
            "element_input_order": list(config["data_spec"]["element_input_order"]),
        }
        np.save(split_dir / "sequence_inputs.npy", sequence_inputs)
        np.save(split_dir / "target_outputs.npy", target_outputs)
        np.save(split_dir / "global_inputs.npy", global_inputs)
        np.save(split_dir / "valid_mask.npy", valid_mask)
        np.save(split_dir / "position_coord.npy", position_coord)
        (split_dir / "run_ids.json").write_text(
            json.dumps(run_ids, indent=2) + "\n", encoding="utf-8",
        )
        (split_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
        )

    data_contract = {
        "processed_data_version": PROCESSED_DATA_VERSION,
        "chemistry_type": chemistry_type,
        "model_type": model_type,
        "state_species_order": list(config["data_spec"]["state_species"]),
        "output_species_order": list(config["data_spec"]["output_species"]),
        "element_input_order": list(config["data_spec"]["element_input_order"]),
        "sequence_static_feature_order": sequence_feature_order,
        "global_static_feature_order": global_static_order,
        "sequence_dim": n_seq_features,
        "target_dim": len(config["data_spec"]["output_species"]),
        "global_dim": len(global_static_order),
        "max_num_levels": max_num_levels,
        "num_levels_range": num_levels_range,
        "pressure_top_bar_range": [
            float(config["sampling"]["pressure_top_bar_range"][0]),
            float(config["sampling"]["pressure_top_bar_range"][1]),
        ],
        "pressure_bottom_bar_range": [
            float(config["sampling"]["pressure_bottom_bar_range"][0]),
            float(config["sampling"]["pressure_bottom_bar_range"][1]),
        ],
        "log10_pressure_bar_union_range": [log10_p_union[0], log10_p_union[1]],
    }
    _write_processed_info_dir(
        info_dir,
        normalization=normalization,
        data_contract=data_contract,
        split_indices=split_indices,
        raw_source_files=raw_source_files,
        chemistry_type=chemistry_type,
        model_type=model_type,
    )
    LOGGER.info("FastChem preprocessing complete -> %s", processed_root)
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
    """Convert raw runs into processed tensors for the active chemistry type.

    Parameters
    ----------
    config : dict[str, Any]
        Validated pipeline config defining raw paths, feature orders,
        normalization methods, and spectrum settings.
    project_root : Path
        Repository root used to resolve config-relative paths.

    Returns
    -------
    dict[str, Any]
        Summary payload containing the processed-root path, normalization
        metadata, data contract, and train/val/test split indices.
    """
    if uses_fastchem(config):
        return preprocess_equilibrium_dataset(config, project_root=project_root)
    LOGGER.info("Preprocessing VULCAN dataset")
    chemistry_type = get_chemistry_type(config)
    model_type = get_model_type(config)
    raw_root = resolve_path(config["paths"]["raw_root"], project_root)
    processed_root = resolve_path(config["paths"]["processed_root"], project_root)
    ensure_dir(processed_root)
    info_dir = ensure_dir(processed_info_dir(processed_root))


    raw_runs, raw_source_files = _load_raw_runs(
        raw_root, config=config, load_run_fn=load_raw_run,
    )
    split_indices = _split_indices(len(raw_runs), config=config)
    train_runs = [raw_runs[i] for i in split_indices["train"]]
    global_static_order = list(config["data_spec"]["global_static_feature_order"])
    normalization = _normalization_payload(
        train_runs,
        config=config,
        global_static_order=global_static_order,
    )

    num_levels_range = [
        int(config["sampling"]["num_levels_range"][0]),
        int(config["sampling"]["num_levels_range"][1]),
    ]
    max_num_levels = num_levels_range[1]
    log10_p_union = _log10_pressure_union_range(config)

    for split_name, indices in split_indices.items():
        split_dir = ensure_dir(processed_root / split_name)
        runs = [raw_runs[i] for i in indices]
        target_dim = runs[0].final_ymix_output.shape[-1]

        sequence_inputs = np.zeros((len(runs), max_num_levels, 3), dtype=np.float32)
        target_outputs = np.zeros((len(runs), max_num_levels, target_dim), dtype=np.float32)
        valid_mask = np.zeros((len(runs), max_num_levels), dtype=bool)
        position_coord = np.zeros((len(runs), max_num_levels), dtype=np.float32)
        global_inputs = np.zeros((len(runs), len(global_static_order)), dtype=np.float32)
        run_ids: list[str] = []

        for idx, run in enumerate(runs):
            nz_i = int(run.pressure_bar.size)
            if nz_i < num_levels_range[0] or nz_i > num_levels_range[1]:
                raise ValueError(
                    f"Run {run.run_id} has num_levels={nz_i}, outside the configured "
                    f"range {num_levels_range}."
                )
            static = np.stack([run.pressure_bar, run.temperature_k, run.kzz_cm2_s], axis=-1)
            sequence_inputs[idx, :nz_i] = _apply_sequence_static_normalization(
                static,
                normalization["sequence_static"],
            ).astype(np.float32)
            target_outputs[idx, :nz_i] = apply_block(
                run.final_ymix_output, normalization["target"]
            ).astype(np.float32)
            position_coord[idx, :nz_i] = _position_coord_from_pressure(
                run.pressure_bar, log10_union_range=log10_p_union,
            )
            valid_mask[idx, :nz_i] = True
            static_inputs = resolve_conditioning_inputs(
                raw_global_inputs=run.globals,
                required_global_inputs=list(config["data_spec"]["required_global_inputs"]),
            )
            global_vector = np.array([static_inputs[name] for name in global_static_order], dtype=np.float64)
            global_inputs[idx] = apply_mixed_block(
                global_vector[None, :],
                normalization["global_static"],
            )[0].astype(np.float32)
            run_ids.append(run.run_id)

        metadata = {
            "processed_data_version": PROCESSED_DATA_VERSION,
            "chemistry_type": chemistry_type,
            "model_type": model_type,
            "split": split_name,
            "num_runs": len(runs),
            "max_num_levels": max_num_levels,
            "num_levels_range": num_levels_range,
            "sequence_feature_order": ["pressure_bar", "temperature_k", "kzz_cm2_s"],
            "output_species_order": list(config["data_spec"]["output_species"]),
            "element_input_order": list(config["data_spec"]["element_input_order"]),
            "global_feature_order": list(config["data_spec"]["global_feature_order"]),
            "global_static_feature_order": list(config["data_spec"]["global_static_feature_order"]),
        }
        np.save(split_dir / "sequence_inputs.npy", sequence_inputs)
        np.save(split_dir / "target_outputs.npy", target_outputs)
        np.save(split_dir / "global_inputs.npy", global_inputs)
        np.save(split_dir / "valid_mask.npy", valid_mask)
        np.save(split_dir / "position_coord.npy", position_coord)
        (split_dir / "run_ids.json").write_text(json.dumps(run_ids, indent=2) + "\n", encoding="utf-8")
        (split_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    data_contract = {
        "processed_data_version": PROCESSED_DATA_VERSION,
        "chemistry_type": chemistry_type,
        "model_type": model_type,
        "output_species_order": list(config["data_spec"]["output_species"]),
        "element_input_order": list(config["data_spec"]["element_input_order"]),
        "sequence_static_feature_order": ["pressure_bar", "temperature_k", "kzz_cm2_s"],
        "global_feature_order": list(config["data_spec"]["global_feature_order"]),
        "global_static_feature_order": list(config["data_spec"]["global_static_feature_order"]),
        "sequence_dim": 3,
        "target_dim": len(config["data_spec"]["output_species"]),
        "max_num_levels": max_num_levels,
        "num_levels_range": num_levels_range,
        "pressure_top_bar_range": [
            float(config["sampling"]["pressure_top_bar_range"][0]),
            float(config["sampling"]["pressure_top_bar_range"][1]),
        ],
        "pressure_bottom_bar_range": [
            float(config["sampling"]["pressure_bottom_bar_range"][0]),
            float(config["sampling"]["pressure_bottom_bar_range"][1]),
        ],
        "log10_pressure_bar_union_range": [log10_p_union[0], log10_p_union[1]],
    }
    _write_processed_info_dir(
        info_dir,
        normalization=normalization,
        data_contract=data_contract,
        split_indices=split_indices,
        raw_source_files=raw_source_files,
        chemistry_type=chemistry_type,
        model_type=model_type,
    )
    LOGGER.info("Full VULCAN preprocessing complete -> %s", processed_root)
    return {
        "processed_root": str(processed_root),
        "normalization": normalization,
        "data_contract": data_contract,
        "splits": split_indices,
    }
