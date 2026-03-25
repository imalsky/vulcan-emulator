from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .config_utils import (
    effective_transition_sampling,
    is_equilibrium,
    resolve_conditioning_inputs,
)
from .path_utils import ensure_dir, resolve_path
from .provenance import fingerprint_payload, manifest_for_files
from .spectrum import SpectrumRecord, fixed_wavelength_grid, resample_spectrum
from .transition_sampling import fit_log10_dt_normalization

PROCESSED_DATA_VERSION = 5


@dataclass(frozen=True)
class RawRun:
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


@dataclass(frozen=True)
class RawEquilibriumRun:
    """Simplified raw run for equilibrium-only models (no trajectory/spectrum)."""
    run_id: str
    pressure_bar: np.ndarray
    temperature_k: np.ndarray
    equilibrium_ymix: np.ndarray
    globals: dict[str, float]


def _decode_species(values: np.ndarray) -> list[str]:
    result: list[str] = []
    for item in values:
        if isinstance(item, bytes):
            result.append(item.decode("utf-8"))
        else:
            result.append(str(item))
    return result


def _fit_standard(arr: np.ndarray) -> dict[str, Any]:
    arr2 = np.reshape(arr, (-1, arr.shape[-1]))
    mean = np.mean(arr2, axis=0)
    std = np.std(arr2, axis=0)
    std = np.where(std < 1.0e-8, 1.0, std)
    return {"method": "standard", "mean": mean.tolist(), "std": std.tolist()}


def _fit_none(arr: np.ndarray) -> dict[str, Any]:
    dim = int(arr.shape[-1]) if arr.ndim >= 1 else 1
    return {"method": "none", "mean": [0.0] * dim, "std": [1.0] * dim}


def _fit_log_standard(arr: np.ndarray, *, floor: float) -> dict[str, Any]:
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


def _global_methods(feature_order: list[str]) -> list[str]:
    methods: list[str] = []
    for name in feature_order:
        if name.startswith("use_") or name.startswith("atm_base_"):
            methods.append("none")
        elif name == "metallicity_log10":
            methods.append("none")
        elif name == "gravity_cm_s2":
            methods.append("log-standard")
        elif name == "c_to_o":
            methods.append("standard")
        else:
            methods.append("standard")
    return methods


def _fit_mixed_block(arr: np.ndarray, methods: list[str]) -> dict[str, Any]:
    arr2 = np.reshape(arr, (-1, arr.shape[-1]))
    transformed_columns = []
    means = []
    stds = []
    floors = []
    for i, method in enumerate(methods):
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


def _fit_log10_dt_block(stats: dict[str, float]) -> dict[str, Any]:
    return {
        "method": "standard",
        "mean": [float(stats["mean"])],
        "std": [float(stats["std"])],
    }


def load_raw_run(
    raw_file: str | Path,
    *,
    config: dict[str, Any],
    spectrum_grid_nm: np.ndarray,
) -> RawRun:
    """Load one raw HDF5 run and align it to the configured species contract."""
    requested_state_species = list(config["data_spec"]["state_species"])
    requested_output_species = list(config["data_spec"]["output_species"])
    with h5py.File(raw_file, "r") as handle:
        pressure_bar = np.asarray(handle["inputs/pressure_bar"], dtype=np.float64)
        temperature_k = np.asarray(handle["inputs/temperature_k"], dtype=np.float64)
        kzz_cm2_s = np.asarray(handle["inputs/kzz_cm2_s"], dtype=np.float64)
        stored_state_species = _decode_species(np.asarray(handle["inputs/state_species"]))
        stored_output_species = _decode_species(np.asarray(handle["inputs/output_species"]))
        reference_ymix_state = np.asarray(handle["inputs/reference_ymix_state"], dtype=np.float64)
        target_mode = handle["inputs/target_mode"][()].decode("utf-8")
        time_s = np.asarray(handle["trajectory/time_s"], dtype=np.float64)
        ymix_state = np.asarray(handle["trajectory/ymix_state"], dtype=np.float64)
        ymix_output = np.asarray(handle["trajectory/ymix_output"], dtype=np.float64)
        globals_map = {
            key: float(np.asarray(handle[f"globals/{key}"]))
            for key in handle["globals"].keys()
        }
        spectrum_name = str(np.asarray(handle["spectrum/name"]).astype(str))
        spectrum_wavelength_nm = np.asarray(handle["spectrum/wavelength_nm"], dtype=np.float64)
        spectrum_flux = np.asarray(handle["spectrum/flux_erg_cm2_s_nm"], dtype=np.float64)

    if not np.all(np.isfinite(pressure_bar)):
        raise ValueError(f"{raw_file}: non-finite pressure values detected.")
    if not np.all(np.isfinite(temperature_k)):
        raise ValueError(f"{raw_file}: non-finite temperature values detected.")
    if not np.all(np.isfinite(kzz_cm2_s)):
        raise ValueError(f"{raw_file}: non-finite Kzz values detected.")
    if not np.all(np.diff(time_s) > 0.0):
        raise ValueError(f"{raw_file}: time_s must be strictly increasing.")
    if ymix_state.shape[0] != time_s.size:
        raise ValueError(f"{raw_file}: ymix_state time dimension does not match time_s.")
    if ymix_output.shape[0] != time_s.size:
        raise ValueError(f"{raw_file}: ymix_output time dimension does not match time_s.")
    if reference_ymix_state.shape[0] != pressure_bar.size:
        raise ValueError(f"{raw_file}: reference_ymix_state vertical dimension does not match pressure grid.")

    state_indices = [stored_state_species.index(name) for name in requested_state_species]
    output_indices = [stored_output_species.index(name) for name in requested_output_species]
    ymix_state = ymix_state[..., state_indices]
    ymix_output = ymix_output[..., output_indices]
    reference_ymix_state = reference_ymix_state[..., state_indices]
    resampled_spectrum = resample_spectrum(
        SpectrumRecord(
            name=spectrum_name,
            wavelength_nm=spectrum_wavelength_nm,
            flux_erg_cm2_s_nm=spectrum_flux,
        ),
        spectrum_grid_nm,
    )
    return RawRun(
        run_id=Path(raw_file).stem,
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        kzz_cm2_s=kzz_cm2_s,
        time_s=time_s,
        ymix_state=ymix_state,
        ymix_output=ymix_output,
        reference_ymix_state=reference_ymix_state,
        target_mode=target_mode,
        globals=globals_map,
        spectrum_name=spectrum_name,
        spectrum_wavelength_nm=np.asarray(spectrum_grid_nm, dtype=np.float64),
        spectrum_flux_erg_cm2_s_nm=np.asarray(resampled_spectrum, dtype=np.float64),
    )


def _split_indices(num_runs: int, *, config: dict[str, Any]) -> dict[str, list[int]]:
    rng = np.random.default_rng(int(config["preprocessing"]["seed"]))
    perm = np.arange(num_runs, dtype=np.int32)
    rng.shuffle(perm)
    train_n = max(1, int(round(num_runs * float(config["preprocessing"]["train_fraction"]))))
    val_n = max(1, int(round(num_runs * float(config["preprocessing"]["val_fraction"]))))
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
        "state": _fit_log_standard(state_values, floor=state_floor),
        "target": _fit_log_standard(target_values, floor=state_floor),
        "global_static": _fit_mixed_block(global_static, _global_methods(global_static_order)),
        "log10_dt_s": _fit_log10_dt_block(dt_stats),
        "spectrum": _fit_log_standard(spectrum_values, floor=spectrum_floor),
        "spectrum_wavelength_nm": spectrum_grid_nm.tolist(),
    }


def _apply_sequence_static_normalization(x: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    blocks = payload["blocks"]
    parts = [apply_block(x[..., i : i + 1], block) for i, block in enumerate(blocks)]
    return np.concatenate(parts, axis=-1)


def load_raw_equilibrium_run(
    raw_file: str | Path,
    *,
    config: dict[str, Any],
) -> RawEquilibriumRun:
    """Load one raw equilibrium HDF5 run."""
    requested_output_species = list(config["data_spec"]["output_species"])
    with h5py.File(raw_file, "r") as handle:
        pressure_bar = np.asarray(handle["inputs/pressure_bar"], dtype=np.float64)
        temperature_k = np.asarray(handle["inputs/temperature_k"], dtype=np.float64)
        stored_output_species = _decode_species(np.asarray(handle["inputs/output_species"]))
        equilibrium_ymix = np.asarray(handle["equilibrium/ymix"], dtype=np.float64)
        globals_map = {
            key: float(np.asarray(handle[f"globals/{key}"]))
            for key in handle["globals"].keys()
        }
    if not np.all(np.isfinite(pressure_bar)):
        raise ValueError(f"{raw_file}: non-finite pressure values detected.")
    if not np.all(np.isfinite(temperature_k)):
        raise ValueError(f"{raw_file}: non-finite temperature values detected.")
    output_indices = [stored_output_species.index(name) for name in requested_output_species]
    equilibrium_ymix = equilibrium_ymix[:, output_indices]
    return RawEquilibriumRun(
        run_id=Path(raw_file).stem,
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        equilibrium_ymix=equilibrium_ymix,
        globals=globals_map,
    )


def _equilibrium_normalization_payload(
    train_runs: list[RawEquilibriumRun],
    *,
    config: dict[str, Any],
    global_static_order: list[str],
) -> dict[str, Any]:
    """Fit normalization statistics for the equilibrium model."""
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
        "target": _fit_log_standard(target_values, floor=state_floor),
        "global_static": _fit_mixed_block(global_static, _global_methods(global_static_order)),
    }


def preprocess_equilibrium_dataset(
    config: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Convert raw equilibrium runs into training tensors (no trajectory/spectrum)."""
    raw_root = resolve_path(config["paths"]["raw_root"], project_root)
    processed_root = resolve_path(config["paths"]["processed_root"], project_root)
    ensure_dir(processed_root)
    raw_run_files = sorted((raw_root / "runs").glob("run_*.h5"))
    if not raw_run_files:
        raise FileNotFoundError(f"No raw run files found under {raw_root / 'runs'}.")

    raw_runs = [load_raw_equilibrium_run(path, config=config) for path in raw_run_files]
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
            "model_type": "equilibrium",
            "split": split_name,
            "num_runs": len(runs),
            "num_levels": nz,
            "sequence_feature_order": sequence_feature_order,
            "output_species_order": list(config["data_spec"]["output_species"]),
            "global_static_feature_order": global_static_order,
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
        "model_type": "equilibrium",
        "state_species_order": list(config["data_spec"]["state_species"]),
        "output_species_order": list(config["data_spec"]["output_species"]),
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
        "raw_files": manifest_for_files(raw_run_files),
        "splits": split_indices,
        "model_type": "equilibrium",
        "normalization_fingerprint": fingerprint_payload(normalization),
    }
    (processed_root / "processed_manifest.json").write_text(
        json.dumps(processed_manifest, indent=2) + "\n", encoding="utf-8",
    )
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
    raw_root = resolve_path(config["paths"]["raw_root"], project_root)
    processed_root = resolve_path(config["paths"]["processed_root"], project_root)
    ensure_dir(processed_root)
    raw_run_files = sorted((raw_root / "runs").glob("run_*.h5"))
    if not raw_run_files:
        raise FileNotFoundError(f"No raw run files found under {raw_root / 'runs'}.")

    spectrum_grid_nm = fixed_wavelength_grid(
        float(config["stellar_spectrum"]["wavelength_min_nm"]),
        float(config["stellar_spectrum"]["wavelength_max_nm"]),
        int(config["stellar_spectrum"]["num_bins"]),
    )
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
            "split": split_name,
            "target_mode": target_mode,
            "num_runs": len(runs),
            "num_levels": nz,
            "max_steps": int(max_steps),
            "sequence_feature_order": ["pressure_bar", "temperature_k", "kzz_cm2_s"],
            "state_species_order": list(config["data_spec"]["state_species"]),
            "output_species_order": list(config["data_spec"]["output_species"]),
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
        "target_mode": target_mode,
        "state_species_order": list(config["data_spec"]["state_species"]),
        "output_species_order": list(config["data_spec"]["output_species"]),
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
        "raw_files": manifest_for_files(raw_run_files),
        "splits": split_indices,
        "target_mode": target_mode,
        "normalization_fingerprint": fingerprint_payload(normalization),
    }
    (processed_root / "processed_manifest.json").write_text(
        json.dumps(processed_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "processed_root": str(processed_root),
        "normalization": normalization,
        "data_contract": data_contract,
        "splits": split_indices,
    }
