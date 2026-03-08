#!/usr/bin/env python3
"""Build a tiny synthetic processed dataset and run a full training smoke test."""

from __future__ import annotations

import json
import shutil
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_utils import load_and_validate_config, resolve_precision
from path_utils import ensure_runtime_dirs, resolve_paths
from provenance import PROCESSED_FINGERPRINT_FILENAME, build_processed_fingerprint
from trainer import run_training

TMP_ROOT = PROJECT_ROOT / "testing" / "_synthetic_smoke"
CONFIG_PATH = TMP_ROOT / "synthetic_config.json"
SYNTHETIC_FIXED_DT_S = 90.0
SYNTHETIC_POST_EQUILIBRIUM_TIME_S = 10.0
TOP20_SPECIES = [
    "H2", "He", "H", "H2O", "CH4", "CO", "CO2", "NH3", "HCN", "C2H2",
    "N2", "OH", "O", "NO", "NO2", "O2", "H2CO", "CH3OH", "C2H4", "C2H6",
]


def _normalization_stats(state_dim: int) -> dict[str, Any]:
    def one(size: int) -> dict[str, Any]:
        return {
            "method": "none",
            "mean": [0.0] * size,
            "std": [1.0] * size,
            "min": [0.0] * size,
            "max": [1.0] * size,
        }

    return {
        "epsilon": 1e-30,
        "sequence": {
            "pressure_bar": one(1),
            "temperature_k": one(1),
            "kzz_cm2_s": one(1),
            "anchor_ymix": one(state_dim),
        },
        "globals": {
            "gravity_cm_s2": one(1),
            "metallicity_log10": one(1),
            "c_to_o": one(1),
            "log10_dt_s": one(1),
        },
        "targets": {
            "ymix": one(state_dim),
        },
    }


def _config() -> dict[str, Any]:
    return {
        "paths": {
            "project_root": ".",
            "vulcan_source_path": "../VULCAN-master",
            "data_root": "testing/_synthetic_smoke/data",
            "models_root": "testing/_synthetic_smoke/models",
            "logs_root": "testing/_synthetic_smoke/logs",
        },
        "generation": {
            "num_runs": 4,
            "num_workers": 1,
            "split_ratios": {"train": 0.5, "val": 0.25, "test": 0.25},
            "random_seed": 7,
            "keep_vulcan_outputs_debug": False,
            "failure_policy": "fail_on_first_error",
            "run_timeout_seconds": 60,
            "manifest_filename": "dataset_manifest.json",
            "split_filename": "splits.json",
            "shard_size": 16,
            "worker_root": "testing/_synthetic_smoke/data/raw/_workers",
            "runs_root": "testing/_synthetic_smoke/data/raw/runs",
            "save_evo_frq": 1,
        },
        "trajectory_sampling": {
            "mode": "fixed_dt_post_equilibrium",
            "pairs_per_run": 1,
            "fixed_requested_dt_s": SYNTHETIC_FIXED_DT_S,
            "post_equilibrium_time_min_s": SYNTHETIC_POST_EQUILIBRIUM_TIME_S,
            "post_equilibrium_min_fraction_of_final_time": 0.0,
            "anchor_sampling": "uniform_valid_anchors",
            "target_selection": "nearest_saved_snapshot",
            "min_future_saved_steps": 1,
            "max_target_relative_dt_error": 0.0,
            "rollout_eval_points": 2,
        },
        "vulcan_runtime": {
            "runtime": 1e3,
            "dt_min": 1e-12,
            "dt_max": 1e2,
            "count_max": 1000,
            "trun_min": 0.0,
            "count_min": 0,
            "ini_mix": "EQ",
            "atm_base": "H2",
        },
        "tp_sampler": {
            "pressure_grid": {"nz": 4, "p_top_bar": 1e-3, "p_bottom_bar": 10.0},
            "temperature_limits_k": {"min": 100.0, "max": 4000.0},
            "max_sampling_attempts": 4,
            "adiabatic_gradient": 0.286,
            "convective_adjustment_probability": 0.0,
            "log10_kappa_ir": {"distribution": "uniform", "min": -1.0, "max": 1.0},
            "kappa_pressure_power_exponent": {"distribution": "uniform", "min": 0.0, "max": 1.0},
            "log10_gamma1": {"distribution": "uniform", "min": -1.0, "max": 1.0},
            "log10_gamma2": {"distribution": "uniform", "min": -1.0, "max": 1.0},
            "alpha_partition": {"distribution": "uniform", "min": 0.0, "max": 1.0},
            "t_int": {"distribution": "uniform", "min": 100.0, "max": 300.0},
            "t_irr": {"distribution": "uniform", "min": 500.0, "max": 1500.0},
            "temperature_shift": {"distribution": "uniform", "min": -50.0, "max": 50.0},
        },
        "gravity_sampler": {"distribution": "uniform", "min_cm_s2": 500.0, "max_cm_s2": 1500.0},
        "kzz_sampler": {
            "mode": "power_law_profile",
            "log10_kzz_at_1bar_min": 8.0,
            "log10_kzz_at_1bar_max": 9.0,
            "beta_min": 0.0,
            "beta_max": 0.5,
            "kzz_floor_cm2_s": 1e7,
            "kzz_cap_cm2_s": 1e10,
            "pressure_unit_for_profile": "bar",
        },
        "abundance_sampler": {
            "metallicity_mode": "log10_scale",
            "log10_metallicity_min": -0.5,
            "log10_metallicity_max": 0.5,
            "c_to_o_min": 0.3,
            "c_to_o_max": 1.0,
            "solar_abundances": {"O_H": 5.37e-4, "N_H": 7.08e-5, "He_H": 0.0838, "S_H": 1.41e-5},
        },
        "physics_toggles": {
            "use_photochemistry": False,
            "use_ion_chemistry": False,
            "use_eddy_diffusion": True,
            "use_molecular_diffusion": True,
            "use_upwind_molecular_diffusion": False,
            "use_boundary_conditions": False,
            "use_condensation": False,
            "use_settling": False,
            "use_initial_cold_trap": True,
            "use_sat_surface_h2o": False,
            "use_lowT_limit_rates": False,
            "use_adaptive_rtol": True,
        },
        "data_spec": {
            "state_species": TOP20_SPECIES,
            "output_species": TOP20_SPECIES,
            "required_input_profiles": ["pressure_bar", "temperature_k", "kzz_cm2_s"],
            "required_global_inputs": ["gravity_cm_s2", "metallicity_log10", "c_to_o", "log10_dt_s"],
            "required_state_inputs": ["anchor_ymix"],
            "time_input_transform": "log10_dt_seconds",
            "strict_non_finite": True,
        },
        "normalization": {
            "epsilon": 1e-30,
            "sequence_methods": {
                "pressure_bar": "none",
                "temperature_k": "none",
                "kzz_cm2_s": "none",
                "anchor_ymix": "none",
            },
            "global_methods": {
                "gravity_cm_s2": "none",
                "metallicity_log10": "none",
                "c_to_o": "none",
                "log10_dt_s": "none",
            },
            "target_method": "none",
        },
        "precision": {
            "input_dtype": "float32",
            "stats_accumulation_dtype": "float32",
            "model_dtype": "float32",
            "forward_dtype": "float32",
            "loss_dtype": "float32",
            "optimizer_state_dtype": "float32",
            "amp_autocast_dtype": "none",
        },
        "training": {
            "device": "cpu",
            "gpu_preload": False,
            "batch_size": 8,
            "epochs": 2,
            "learning_rate": 1e-3,
            "min_lr": 1e-4,
            "warmup_epochs": 0,
            "weight_decay": 1e-5,
            "gradient_clip": 1.0,
            "use_amp": False,
            "num_workers": 0,
            "seed": 11,
            "data_loading": {
                "mode": "ram",
                "max_cached_shards": 2,
                "large_shard_mmap_bytes": 1024,
                "ram_safety_fraction": 0.8,
                "copy_mmap_slices": False,
                "use_device_prefetch": False,
            },
            "model": {
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 96,
                "dropout": 0.0,
                "film_clamp": 10.0,
                "output_head_divisor": 2,
                "max_sequence_length": 8,
                "conditioning_hidden_dim": 32,
            },
            "output_folder": "synthetic_training_smoke",
        },
    }


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _make_raw_run(run_id: int, nz: int, state_dim: int, rng: np.random.Generator) -> dict[str, Any]:
    pressure_bar = np.logspace(1.0, -2.0, nz, dtype=np.float64)
    temperature_k = 900.0 - 50.0 * np.log10(pressure_bar + 1e-12) + 5.0 * run_id
    kzz_cm2_s = np.full((nz,), 1.0e8 + 1.0e7 * run_id, dtype=np.float64)
    gravity = 800.0 + 50.0 * run_id
    metallicity = -0.2 + 0.1 * run_id
    c_to_o = 0.5 + 0.05 * run_id
    time_s = np.array([0.0, 1.0, 10.0, 100.0], dtype=np.float64)

    base_logits = rng.normal(loc=0.0, scale=0.3, size=(nz, state_dim))
    profile_bias = np.linspace(-0.2, 0.2, nz, dtype=np.float64)[:, None]
    species_bias = np.linspace(-0.3, 0.3, state_dim, dtype=np.float64)[None, :]
    ymix_time = []
    for idx, t in enumerate(time_s):
        drift = 0.05 * np.log10(1.0 + t)
        logits = base_logits + drift * (profile_bias + species_bias) + 0.01 * idx
        ymix_time.append(_softmax_rows(logits))
    ymix_state = np.stack(ymix_time, axis=0)
    return {
        "run_id": run_id,
        "pressure_bar": pressure_bar,
        "temperature_k": temperature_k,
        "kzz_cm2_s": kzz_cm2_s,
        "gravity_cm_s2": gravity,
        "metallicity_log10": metallicity,
        "c_to_o": c_to_o,
        "time_s": time_s,
        "ymix_state": ymix_state,
    }


def _write_raw_run(path: Path, run: dict[str, Any], state_species: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs["run_id"] = int(run["run_id"])
        inputs = handle.create_group("inputs")
        inputs.create_dataset("pressure_bar", data=run["pressure_bar"])
        inputs.create_dataset("temperature_k", data=run["temperature_k"])
        inputs.create_dataset("kzz_cm2_s", data=run["kzz_cm2_s"])
        inputs.create_dataset("state_species", data=np.asarray(state_species, dtype=str_dtype))
        inputs.create_dataset("output_species", data=np.asarray(state_species, dtype=str_dtype))
        globals_group = handle.create_group("globals")
        globals_group.create_dataset("gravity_cm_s2", data=np.float64(run["gravity_cm_s2"]))
        globals_group.create_dataset("metallicity_log10", data=np.float64(run["metallicity_log10"]))
        globals_group.create_dataset("c_to_o", data=np.float64(run["c_to_o"]))
        trajectory = handle.create_group("trajectory")
        trajectory.create_dataset("time_s", data=run["time_s"])
        trajectory.create_dataset("ymix_state", data=run["ymix_state"])
        trajectory.create_dataset("ymix_output", data=run["ymix_state"])


def _build_processed_samples(run: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    static = np.stack([run["pressure_bar"], run["temperature_k"], run["kzz_cm2_s"]], axis=1)
    times_s = np.asarray(run["time_s"], dtype=np.float64)
    valid_anchor_indices = np.where(times_s >= SYNTHETIC_POST_EQUILIBRIUM_TIME_S)[0]
    if valid_anchor_indices.size == 0:
        raise RuntimeError("Synthetic trajectory did not contain any post-equilibrium anchor states.")

    anchor_idx = int(valid_anchor_indices[0])
    requested_target_time_s = float(times_s[anchor_idx] + SYNTHETIC_FIXED_DT_S)
    future_times = times_s[anchor_idx + 1 :]
    if future_times.size == 0:
        raise RuntimeError("Synthetic trajectory did not contain a future state after the anchor.")

    nearest_offset = int(np.argmin(np.abs(future_times - requested_target_time_s)))
    target_idx = anchor_idx + 1 + nearest_offset
    dt_s = float(times_s[target_idx] - times_s[anchor_idx])
    if not np.isclose(dt_s, SYNTHETIC_FIXED_DT_S, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(
            "Synthetic fixed-dt sample does not match the configured requested dt. "
            f"actual={dt_s} requested={SYNTHETIC_FIXED_DT_S}"
        )

    anchor = run["ymix_state"][anchor_idx]
    target = run["ymix_state"][target_idx]
    sequence = np.concatenate([static, anchor], axis=1).astype(np.float32, copy=False)
    globals_ = np.array(
        [
            run["gravity_cm_s2"],
            run["metallicity_log10"],
            run["c_to_o"],
            np.log10(dt_s),
        ],
        dtype=np.float32,
    )
    return [(sequence, globals_, target.astype(np.float32, copy=False), dt_s)]


def _write_processed_split(
    split_dir: Path,
    *,
    split_name: str,
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    state_species: list[str],
    normalization_fingerprint: str,
    shard_size: int,
) -> dict[str, Any]:
    seq_dir = split_dir / "sequence_inputs"
    glb_dir = split_dir / "globals"
    tgt_dir = split_dir / "targets"
    dt_dir = split_dir / "dt_s"
    for directory in (split_dir, seq_dir, glb_dir, tgt_dir, dt_dir):
        directory.mkdir(parents=True, exist_ok=True)

    num_shards = 0
    for start in range(0, len(samples), shard_size):
        chunk = samples[start : start + shard_size]
        seq = np.stack([item[0] for item in chunk], axis=0)
        glb = np.stack([item[1] for item in chunk], axis=0)
        tgt = np.stack([item[2] for item in chunk], axis=0)
        dt = np.asarray([item[3] for item in chunk], dtype=np.float32)
        np.save(seq_dir / f"shard_{num_shards:05d}.npy", seq, allow_pickle=False)
        np.save(glb_dir / f"shard_{num_shards:05d}.npy", glb, allow_pickle=False)
        np.save(tgt_dir / f"shard_{num_shards:05d}.npy", tgt, allow_pickle=False)
        np.save(dt_dir / f"shard_{num_shards:05d}.npy", dt, allow_pickle=False)
        num_shards += 1

    metadata = {
        "split": split_name,
        "total_samples": len(samples),
        "num_shards": num_shards,
        "sequence_length": int(samples[0][0].shape[0]),
        "input_dim": int(samples[0][0].shape[1]),
        "global_dim": int(samples[0][1].shape[0]),
        "target_dim": int(samples[0][2].shape[1]),
        "state_dim": int(samples[0][2].shape[1]),
        "sequence_feature_order": [
            "pressure_bar",
            "temperature_k",
            "kzz_cm2_s",
            *[f"anchor_ymix:{species}" for species in state_species],
        ],
        "global_feature_order": ["gravity_cm_s2", "metallicity_log10", "c_to_o", "log10_dt_s"],
        "state_species_order": list(state_species),
        "output_species_order": list(state_species),
        "output_from_state_indices": list(range(len(state_species))),
        "normalization_fingerprint": normalization_fingerprint,
        "dt_min_s": float(min(item[3] for item in samples)),
        "dt_max_s": float(max(item[3] for item in samples)),
    }
    with (split_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def _prepare_artifacts() -> tuple[dict[str, Any], Any]:
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    config = _config()
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    loaded = load_and_validate_config(CONFIG_PATH)
    paths = resolve_paths(loaded)
    ensure_runtime_dirs(paths)

    rng = np.random.default_rng(123)
    raw_runs: list[dict[str, Any]] = []
    raw_run_files: list[Path] = []
    for run_id in range(4):
        run = _make_raw_run(run_id, nz=4, state_dim=len(TOP20_SPECIES), rng=rng)
        raw_runs.append(run)
        run_file = paths.raw_root / "runs" / f"run_{run_id:06d}.h5"
        _write_raw_run(run_file, run, TOP20_SPECIES)
        raw_run_files.append(run_file)

    split_map = {"train": [0, 1], "val": [2], "test": [3]}
    normalization_metadata = _normalization_stats(len(TOP20_SPECIES))
    normalization_fingerprint = sha256(
        json.dumps(normalization_metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    summary: dict[str, Any] = {}
    for split_name, run_ids in split_map.items():
        samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        for run_id in run_ids:
            samples.extend(_build_processed_samples(raw_runs[run_id]))
        summary[split_name] = _write_processed_split(
            paths.processed_root / split_name,
            split_name=split_name,
            samples=samples,
            state_species=TOP20_SPECIES,
            normalization_fingerprint=normalization_fingerprint,
            shard_size=int(config["generation"]["shard_size"]),
        )

    manifest_path = paths.data_root / config["generation"]["manifest_filename"]
    manifest = {
        "num_runs": len(raw_run_files),
        "run_files": [str(path.relative_to(paths.root)) for path in raw_run_files],
        "split": split_map,
        "state_species": TOP20_SPECIES,
        "output_species": TOP20_SPECIES,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    split_path = paths.data_root / config["generation"]["split_filename"]
    split_path.write_text(json.dumps(split_map, indent=2), encoding="utf-8")

    norm_path = paths.processed_root / "normalization_metadata.json"
    norm_path.write_text(json.dumps(normalization_metadata, indent=2), encoding="utf-8")

    summary_path = paths.processed_root / "processed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fingerprint = build_processed_fingerprint(
        config=loaded,
        project_root=paths.root,
        raw_run_files=raw_run_files,
        manifest_path=manifest_path,
        split_path=split_path,
        normalization_path=norm_path,
        summary_path=summary_path,
        split_metadata_paths={
            split_name: paths.processed_root / split_name / "metadata.json"
            for split_name in ("train", "val", "test")
        },
    )
    fingerprint_path = paths.processed_root / PROCESSED_FINGERPRINT_FILENAME
    fingerprint_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    return loaded, paths


def main() -> None:
    config, paths = _prepare_artifacts()
    precision = resolve_precision(config)
    run_training(config, paths, precision)

    run_dir = paths.models_root / config["training"]["output_folder"]
    required = [
        run_dir / "best.pt",
        run_dir / "last.pt",
        run_dir / "metrics.json",
        run_dir / "data_contract.json",
        run_dir / "normalization_metadata.json",
        run_dir / PROCESSED_FINGERPRINT_FILENAME,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Synthetic training smoke missing expected artifacts: {missing}")

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
