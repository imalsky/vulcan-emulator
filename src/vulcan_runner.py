from __future__ import annotations

import concurrent.futures
import json
import pickle
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .path_utils import ensure_dir, resolve_path
from .provenance import manifest_for_files
from .sampling import RunSpecification, sample_run_specifications
from .spectrum import write_vulcan_spectrum_txt

_SOLAR_ELEMENT_ABUNDANCES = {
    "O_H": 5.37e-4,
    "C_H": 2.95e-4,
    "N_H": 7.08e-5,
    "S_H": 1.41e-5,
    "He_H": 8.38e-2,
}


@dataclass(frozen=True)
class GeneratedRawDataset:
    raw_root: Path
    run_files: list[Path]
    manifest_path: Path | None = None
    coverage_path: Path | None = None


def patch_python_assignments(text: str, assignments: dict[str, Any]) -> str:
    updated = text
    for name, value in assignments.items():
        replacement = f"{name} = {value!r}"
        pattern = re.compile(rf"^\s*{re.escape(name)}\s*=.*$", re.MULTILINE)
        if pattern.search(updated):
            updated = pattern.sub(replacement, updated, count=1)
        else:
            if not updated.endswith("\n"):
                updated += "\n"
            updated += replacement + "\n"
    return updated


def _requested_run_count(config: dict[str, Any], num_runs: int | None) -> int:
    return int(config["generation"]["num_runs"] if num_runs is None else num_runs)


def _generation_worker_count(config: dict[str, Any], total_runs: int) -> int:
    configured = int(config["generation"]["parallel_workers"])
    return max(1, min(configured, total_runs))


def _prepare_generation_directory(
    config: dict[str, Any],
    *,
    project_root: Path,
    num_runs: int | None,
) -> tuple[Path, Path, list[Path] | None]:
    """Prepare the raw-run directory and honour overwrite/reuse semantics."""
    raw_root = resolve_path(config["paths"]["raw_root"], project_root)
    runs_dir = ensure_dir(raw_root / "runs")
    requested_runs = _requested_run_count(config, num_runs)
    existing_files = sorted(runs_dir.glob("run_*.h5"))
    if bool(config["generation"]["overwrite"]):
        for path in existing_files:
            path.unlink()
        return raw_root, runs_dir, None
    if existing_files:
        if bool(config["generation"]["reuse_raw_if_present"]) and len(existing_files) == requested_runs:
            return raw_root, runs_dir, existing_files
        raise RuntimeError(
            f"Found {len(existing_files)} existing raw runs under {runs_dir}. "
            "Set generation.overwrite=true or align generation.num_runs with the existing dataset."
        )
    return raw_root, runs_dir, None


def _coverage_fraction(low: float, high: float, observed_low: float, observed_high: float) -> float:
    span = max(high - low, 1.0e-12)
    return float(np.clip((observed_high - observed_low) / span, 0.0, 1.0))


def _sampling_coverage_payload(
    *,
    config: dict[str, Any],
    specs: list[RunSpecification],
    run_files: list[Path],
    mode: str,
) -> dict[str, Any]:
    """Summarize the realized dataset coverage against the configured ranges."""
    gravity = np.asarray([spec.globals["gravity_cm_s2"] for spec in specs], dtype=np.float64)
    metallicity = np.asarray([spec.globals["metallicity_log10"] for spec in specs], dtype=np.float64)
    c_to_o = np.asarray([spec.globals["c_to_o"] for spec in specs], dtype=np.float64)
    temperature_rows: list[np.ndarray] = []
    log10_kzz_rows: list[np.ndarray] = []
    pressure_rows: list[np.ndarray] = []
    time_step_rows: list[np.ndarray] = []
    for path in run_files:
        with h5py.File(path, "r") as handle:
            temperature_rows.append(np.asarray(handle["inputs/temperature_k"], dtype=np.float64))
            pressure_rows.append(np.asarray(handle["inputs/pressure_bar"], dtype=np.float64))
            log10_kzz_rows.append(np.log10(np.asarray(handle["inputs/kzz_cm2_s"], dtype=np.float64)))
            time_s = np.asarray(handle["trajectory/time_s"], dtype=np.float64)
            if time_s.size > 1:
                time_step_rows.append(np.log10(np.diff(time_s)))
    temperature = np.concatenate(temperature_rows, axis=0)
    log10_kzz = np.concatenate(log10_kzz_rows, axis=0)
    pressure = np.concatenate(pressure_rows, axis=0)
    time_step_log10 = np.concatenate(time_step_rows, axis=0) if time_step_rows else np.zeros((0,), dtype=np.float64)
    gravity_range = [float(x) for x in config["sampling"]["gravity_range_cm_s2"]]
    metallicity_range = [float(x) for x in config["sampling"]["metallicity_log10_range"]]
    c_to_o_range = [float(x) for x in config["sampling"]["c_to_o_range"]]
    temperature_range = [float(x) for x in config["sampling"]["temperature_range_k"]]
    kzz_value = float(config["sampling"]["kzz_cm2_s"])
    log10_kzz_value = float(np.log10(max(kzz_value, 1.0e-30)))
    kzz_range = [log10_kzz_value, log10_kzz_value]
    time_range = [
        float(config["sampling"]["time_step_log10_min_s"]),
        float(config["sampling"]["time_step_log10_max_s"]),
    ]
    return {
        "mode": mode,
        "num_runs": len(specs),
        "configured_ranges": {
            "gravity_cm_s2": gravity_range,
            "metallicity_log10": metallicity_range,
            "c_to_o": c_to_o_range,
            "temperature_k": temperature_range,
            "log10_kzz_cm2_s": kzz_range,
            "log10_adjacent_dt_s": time_range,
            "pressure_bar": [
                float(config["sampling"]["pressure_top_bar"]),
                float(config["sampling"]["pressure_bottom_bar"]),
            ],
        },
        "realized_summary": {
            "gravity_cm_s2": {
                "min": float(np.min(gravity)),
                "max": float(np.max(gravity)),
                "coverage_fraction": _coverage_fraction(*gravity_range, float(np.min(gravity)), float(np.max(gravity))),
            },
            "metallicity_log10": {
                "min": float(np.min(metallicity)),
                "max": float(np.max(metallicity)),
                "coverage_fraction": _coverage_fraction(
                    *metallicity_range,
                    float(np.min(metallicity)),
                    float(np.max(metallicity)),
                ),
            },
            "c_to_o": {
                "min": float(np.min(c_to_o)),
                "max": float(np.max(c_to_o)),
                "coverage_fraction": _coverage_fraction(*c_to_o_range, float(np.min(c_to_o)), float(np.max(c_to_o))),
            },
            "temperature_k": {
                "min": float(np.min(temperature)),
                "max": float(np.max(temperature)),
                "coverage_fraction": _coverage_fraction(
                    *temperature_range,
                    float(np.min(temperature)),
                    float(np.max(temperature)),
                ),
            },
            "log10_kzz_cm2_s": {
                "min": float(np.min(log10_kzz)),
                "max": float(np.max(log10_kzz)),
                "coverage_fraction": _coverage_fraction(*kzz_range, float(np.min(log10_kzz)), float(np.max(log10_kzz))),
            },
            "log10_adjacent_dt_s": {
                "min": float(np.min(time_step_log10)) if time_step_log10.size else None,
                "max": float(np.max(time_step_log10)) if time_step_log10.size else None,
                "coverage_fraction": (
                    _coverage_fraction(*time_range, float(np.min(time_step_log10)), float(np.max(time_step_log10)))
                    if time_step_log10.size
                    else None
                ),
            },
            "pressure_bar": {
                "min": float(np.min(pressure)),
                "max": float(np.max(pressure)),
            },
        },
    }


def _write_generation_metadata(
    *,
    raw_root: Path,
    run_files: list[Path],
    specs: list[RunSpecification],
    config: dict[str, Any],
    mode: str,
) -> tuple[Path, Path]:
    """Persist the raw-run manifest and parameter-space coverage summary."""
    manifest_path = raw_root / "generation_manifest.json"
    coverage_path = raw_root / "sampling_coverage.json"
    manifest_payload = {
        "mode": mode,
        "run_files": manifest_for_files(run_files),
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    coverage_path.write_text(
        json.dumps(
            _sampling_coverage_payload(config=config, specs=specs, run_files=run_files, mode=mode),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path, coverage_path


def _sulfur_enabled(config: dict[str, Any]) -> bool:
    species = list(config["data_spec"]["state_species"]) + list(config["data_spec"]["output_species"])
    return any("S" in name for name in species)


def _vulcan_atom_list(config: dict[str, Any]) -> list[str]:
    atoms = ["H", "O", "C", "N", "He"]
    if _sulfur_enabled(config):
        atoms.append("S")
    return atoms


def _element_abundances_from_spec(spec: RunSpecification) -> dict[str, float]:
    metal_scale = 10.0 ** float(spec.globals["metallicity_log10"])
    oxygen_h = _SOLAR_ELEMENT_ABUNDANCES["O_H"] * metal_scale
    return {
        "O_H": float(oxygen_h),
        "C_H": float(oxygen_h * spec.globals["c_to_o"]),
        "N_H": float(_SOLAR_ELEMENT_ABUNDANCES["N_H"] * metal_scale),
        "S_H": float(_SOLAR_ELEMENT_ABUNDANCES["S_H"] * metal_scale),
        "He_H": float(_SOLAR_ELEMENT_ABUNDANCES["He_H"]),
        "fastchem_met_scale": float(metal_scale),
    }


def _write_tp_profile(path: Path, spec: RunSpecification) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Pressure(bar) Temperature(K) Kzz(cm2/s)\n")
        handle.write("Pressure Temp Kzz\n")
        for p, t, kzz in zip(spec.pressure_bar, spec.temperature_k, spec.kzz_cm2_s):
            handle.write(f"{p:.8e} {t:.8f} {kzz:.8e}\n")
    return path


def write_raw_run_hdf5(
    path: Path,
    *,
    spec: RunSpecification,
    ymix_state: np.ndarray,
    ymix_output: np.ndarray | None = None,
    output_species: list[str] | None = None,
) -> Path:
    ensure_dir(path.parent)
    output_species = list(output_species or spec.metadata.get("output_species", spec.metadata["state_species"]))
    ymix_output_array = np.asarray(ymix_output if ymix_output is not None else ymix_state, dtype=np.float64)
    with h5py.File(path, "w") as handle:
        inputs = handle.create_group("inputs")
        inputs.create_dataset("pressure_bar", data=np.asarray(spec.pressure_bar, dtype=np.float64))
        inputs.create_dataset("temperature_k", data=np.asarray(spec.temperature_k, dtype=np.float64))
        inputs.create_dataset("kzz_cm2_s", data=np.asarray(spec.kzz_cm2_s, dtype=np.float64))
        inputs.create_dataset(
            "state_species",
            data=np.asarray(spec.metadata["state_species"], dtype="S"),
        )
        inputs.create_dataset(
            "output_species",
            data=np.asarray(output_species, dtype="S"),
        )
        globals_group = handle.create_group("globals")
        for key, value in sorted(spec.globals.items()):
            globals_group.create_dataset(key, data=float(value))
        trajectory = handle.create_group("trajectory")
        trajectory.create_dataset("time_s", data=np.asarray(spec.time_s, dtype=np.float64))
        trajectory.create_dataset("ymix_state", data=np.asarray(ymix_state, dtype=np.float64))
        trajectory.create_dataset("ymix_output", data=ymix_output_array)
        spectrum = handle.create_group("spectrum")
        spectrum.create_dataset("name", data=np.bytes_(spec.spectrum.name))
        spectrum.create_dataset(
            "wavelength_nm",
            data=np.asarray(spec.spectrum.wavelength_nm, dtype=np.float64),
        )
        spectrum.create_dataset(
            "flux_erg_cm2_s_nm",
            data=np.asarray(spec.spectrum.flux_erg_cm2_s_nm, dtype=np.float64),
        )
    return path


def _laplacian_vertical(x: np.ndarray) -> np.ndarray:
    lap = np.zeros_like(x)
    lap[1:-1] = x[:-2] - 2.0 * x[1:-1] + x[2:]
    lap[0] = x[1] - x[0]
    lap[-1] = x[-2] - x[-1]
    return lap


def _uv_profile(
    state: np.ndarray,
    *,
    species_index: dict[str, int],
    spectrum: np.ndarray,
    wavelength_nm: np.ndarray,
) -> np.ndarray:
    uv_mask = wavelength_nm <= 300.0
    if not np.any(uv_mask):
        base_uv = 1.0
    else:
        uv_flux = float(np.trapezoid(spectrum[uv_mask], wavelength_nm[uv_mask]))
        total_flux = float(np.trapezoid(spectrum, wavelength_nm))
        base_uv = max(uv_flux / max(total_flux, 1.0e-30), 1.0e-4)
    h2o = state[:, species_index["H2O"]]
    h2s = state[:, species_index["H2S"]]
    so2 = state[:, species_index["SO2"]]
    absorber = h2o + 20.0 * h2s + 10.0 * so2
    tau = np.cumsum(absorber[::-1])[::-1] * 50.0
    return base_uv * np.exp(-np.clip(tau, 0.0, 30.0))


def _synthetic_tendencies(
    state: np.ndarray,
    *,
    spec: RunSpecification,
    species_index: dict[str, int],
) -> np.ndarray:
    p = np.asarray(spec.pressure_bar, dtype=np.float64)
    t = np.asarray(spec.temperature_k, dtype=np.float64)
    uv = _uv_profile(
        state,
        species_index=species_index,
        spectrum=spec.spectrum.flux_erg_cm2_s_nm,
        wavelength_nm=spec.spectrum.wavelength_nm,
    )
    oxygen = state[:, species_index["H2O"]] + state[:, species_index["CO2"]]
    oxidants = state[:, species_index["O"]] + state[:, species_index["OH"]] + 1.0e-4 * oxygen
    tend = np.zeros_like(state)
    temp_factor = np.exp((t - 1100.0) / 500.0)
    temp_factor = np.clip(temp_factor, 0.2, 10.0)
    pressure_factor = np.clip((p / np.median(p)) ** 0.1, 0.5, 2.0)

    # Radical chemistry needed to keep sulfur oxidation closer to Markovian.
    r_h2o = 2.5e-8 * (1.0 + 8.0 * uv) * state[:, species_index["H2O"]]
    tend[:, species_index["H2O"]] -= r_h2o
    tend[:, species_index["OH"]] += 0.7 * r_h2o
    tend[:, species_index["H"]] += 0.7 * r_h2o
    tend[:, species_index["O"]] += 0.3 * r_h2o

    r_o_to_oh = 1.0e-8 * pressure_factor * state[:, species_index["O"]] * state[:, species_index["H2"]]
    tend[:, species_index["O"]] -= r_o_to_oh
    tend[:, species_index["OH"]] += r_o_to_oh
    tend[:, species_index["H"]] += 0.5 * r_o_to_oh

    r_oh_to_h2o = 1.1e-8 * pressure_factor * state[:, species_index["OH"]] * state[:, species_index["H2"]]
    tend[:, species_index["OH"]] -= r_oh_to_h2o
    tend[:, species_index["H2O"]] += 0.8 * r_oh_to_h2o
    tend[:, species_index["H"]] += 0.2 * r_oh_to_h2o

    r_h_oh = 8.0e-9 * pressure_factor * state[:, species_index["H"]] * state[:, species_index["OH"]]
    tend[:, species_index["H"]] -= r_h_oh
    tend[:, species_index["OH"]] -= r_h_oh
    tend[:, species_index["H2O"]] += r_h_oh

    # Carbon chemistry.
    r_ch4 = 3.0e-8 * temp_factor * (1.0 + 6.0 * uv) * state[:, species_index["CH4"]]
    tend[:, species_index["CH4"]] -= r_ch4
    tend[:, species_index["CO"]] += 0.65 * r_ch4
    tend[:, species_index["CO2"]] += 0.20 * r_ch4 * oxygen / np.clip(oxygen + 1.0e-12, 1.0e-12, None)
    tend[:, species_index["H2O"]] -= 0.05 * r_ch4
    tend[:, species_index["H"]] += 0.15 * r_ch4

    r_co = 1.2e-8 * (1.0 + 0.5 * temp_factor) * oxidants * state[:, species_index["CO"]]
    tend[:, species_index["CO"]] -= r_co
    tend[:, species_index["CO2"]] += r_co
    tend[:, species_index["OH"]] -= 0.6 * r_co
    tend[:, species_index["O"]] -= 0.4 * r_co
    tend[:, species_index["H"]] += 0.2 * r_co

    # Nitrogen chemistry.
    r_nh3 = 1.5e-8 * temp_factor * (1.0 + 2.0 * uv) * state[:, species_index["NH3"]]
    tend[:, species_index["NH3"]] -= r_nh3
    tend[:, species_index["N2"]] += 0.5 * r_nh3
    tend[:, species_index["H"]] += 0.5 * r_nh3

    # Sulfur photochemistry.
    r_h2s = 4.0e-8 * pressure_factor * (1.0 + 10.0 * uv) * state[:, species_index["H2S"]]
    tend[:, species_index["H2S"]] -= r_h2s
    tend[:, species_index["SH"]] += 0.65 * r_h2s
    tend[:, species_index["S"]] += 0.25 * r_h2s
    tend[:, species_index["H"]] += 0.75 * r_h2s
    tend[:, species_index["SO"]] += 0.10 * r_h2s * oxidants / np.clip(oxidants + 1.0e-12, 1.0e-12, None)

    r_h2s_oh = 1.4e-8 * state[:, species_index["OH"]] * state[:, species_index["H2S"]]
    tend[:, species_index["H2S"]] -= r_h2s_oh
    tend[:, species_index["OH"]] -= r_h2s_oh
    tend[:, species_index["SH"]] += r_h2s_oh
    tend[:, species_index["H2O"]] += r_h2s_oh

    r_sh = 2.2e-8 * (1.0 + 6.0 * uv) * state[:, species_index["SH"]]
    tend[:, species_index["SH"]] -= r_sh
    tend[:, species_index["S"]] += 0.55 * r_sh
    tend[:, species_index["S2"]] += 0.15 * r_sh
    tend[:, species_index["SO"]] += 0.30 * r_sh * oxidants / np.clip(oxidants + 1.0e-12, 1.0e-12, None)
    tend[:, species_index["H"]] += 0.85 * r_sh

    r_s_to_so = 1.8e-8 * oxidants * (1.0 + 3.0 * uv) * state[:, species_index["S"]]
    tend[:, species_index["S"]] -= r_s_to_so
    tend[:, species_index["SO"]] += r_s_to_so
    tend[:, species_index["O"]] -= 0.7 * r_s_to_so
    tend[:, species_index["OH"]] -= 0.3 * r_s_to_so

    r_s2 = 8.0e-9 * np.sqrt(np.clip(state[:, species_index["S"]], 1.0e-30, None)) * state[:, species_index["S"]]
    tend[:, species_index["S"]] -= r_s2
    tend[:, species_index["S2"]] += r_s2

    r_so2 = 1.6e-8 * oxidants * state[:, species_index["SO"]]
    tend[:, species_index["SO"]] -= r_so2
    tend[:, species_index["SO2"]] += r_so2
    tend[:, species_index["O"]] -= 0.5 * r_so2
    tend[:, species_index["OH"]] -= 0.5 * r_so2
    tend[:, species_index["H"]] += 0.3 * r_so2

    # Mild oxidation of reduced sulfur back from SO2 at depth.
    r_back = 2.0e-9 * pressure_factor * state[:, species_index["SO2"]]
    tend[:, species_index["SO2"]] -= r_back
    tend[:, species_index["SO"]] += r_back

    # Vertical mixing.
    diffusion_species = [
        "H",
        "O",
        "OH",
        "H2O",
        "CO",
        "CO2",
        "CH4",
        "NH3",
        "H2S",
        "SH",
        "S",
        "SO",
        "SO2",
        "S2",
    ]
    mix_strength = np.clip(spec.kzz_cm2_s / max(np.max(spec.kzz_cm2_s), 1.0), 0.0, 1.0)
    for name in diffusion_species:
        i = species_index[name]
        tend[:, i] += 2.0e-7 * mix_strength * _laplacian_vertical(state[:, i])

    return tend


def _renormalize_state(
    state: np.ndarray,
    *,
    species_index: dict[str, int],
    reservoir_split: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(state, 1.0e-30, None)
    heavy_indices = [i for name, i in species_index.items() if name not in {"H2", "He"}]
    heavy_sum = np.sum(clipped[:, heavy_indices], axis=1)
    heavy_scale = np.where(heavy_sum > 0.995, 0.995 / heavy_sum, 1.0)
    clipped[:, heavy_indices] *= heavy_scale[:, None]
    heavy_sum = np.sum(clipped[:, heavy_indices], axis=1)
    reservoir = np.clip(1.0 - heavy_sum, 1.0e-5, 1.0)
    clipped[:, species_index["H2"]] = reservoir * reservoir_split
    clipped[:, species_index["He"]] = reservoir * (1.0 - reservoir_split)
    clipped /= np.sum(clipped, axis=1, keepdims=True)
    return clipped


def simulate_synthetic_trajectory(spec: RunSpecification) -> np.ndarray:
    species_order = list(spec.metadata["state_species"])
    idx = {name: i for i, name in enumerate(species_order)}
    state = np.asarray(spec.initial_ymix, dtype=np.float64).copy()
    reservoir_split = state[:, idx["H2"]] / np.clip(
        state[:, idx["H2"]] + state[:, idx["He"]],
        1.0e-12,
        None,
    )
    times = np.asarray(spec.time_s, dtype=np.float64)
    trajectory = np.zeros((times.size, *state.shape), dtype=np.float64)
    trajectory[0] = state
    for step in range(times.size - 1):
        dt_total = float(times[step + 1] - times[step])
        substeps = int(max(1, min(32, np.ceil(dt_total / 1.0e4))))
        dt = dt_total / substeps
        for _ in range(substeps):
            tend = _synthetic_tendencies(state, spec=spec, species_index=idx)
            state = _renormalize_state(
                state + dt * tend,
                species_index=idx,
                reservoir_split=reservoir_split,
            )
        trajectory[step + 1] = state
    return trajectory


def _slice_output_trajectory(
    trajectory: np.ndarray,
    *,
    state_species: list[str],
    output_species: list[str],
) -> np.ndarray:
    indices = [state_species.index(name) for name in output_species]
    return np.asarray(trajectory[..., indices], dtype=np.float64)


def _generate_single_synthetic_run(
    spec: RunSpecification,
    *,
    runs_dir: Path,
    output_species: list[str],
) -> Path:
    trajectory = simulate_synthetic_trajectory(spec)
    output_trajectory = _slice_output_trajectory(
        trajectory,
        state_species=list(spec.metadata["state_species"]),
        output_species=output_species,
    )
    h5_path = runs_dir / f"{spec.run_id}.h5"
    write_raw_run_hdf5(
        h5_path,
        spec=spec,
        ymix_state=trajectory,
        ymix_output=output_trajectory,
        output_species=output_species,
    )
    return h5_path


def generate_synthetic_raw_runs(
    config: dict[str, Any],
    *,
    project_root: Path,
    num_runs: int | None = None,
) -> GeneratedRawDataset:
    raw_root, runs_dir, reusable_files = _prepare_generation_directory(
        config,
        project_root=project_root,
        num_runs=num_runs,
    )
    if reusable_files is not None:
        manifest_path = raw_root / "generation_manifest.json"
        coverage_path = raw_root / "sampling_coverage.json"
        return GeneratedRawDataset(
            raw_root=raw_root,
            run_files=reusable_files,
            manifest_path=manifest_path if manifest_path.exists() else None,
            coverage_path=coverage_path if coverage_path.exists() else None,
        )
    specs = sample_run_specifications(
        config=config,
        project_root=project_root,
        num_runs=num_runs,
        seed=int(config["generation"]["seed"]),
    )
    state_species = list(config["data_spec"]["state_species"])
    output_species = list(config["data_spec"]["output_species"])
    prepared_specs: list[RunSpecification] = []
    for spec in specs:
        prepared_specs.append(
            RunSpecification(
            run_id=spec.run_id,
            pressure_bar=spec.pressure_bar,
            temperature_k=spec.temperature_k,
            kzz_cm2_s=spec.kzz_cm2_s,
            initial_ymix=spec.initial_ymix,
            time_s=spec.time_s,
            globals=spec.globals,
            spectrum=spec.spectrum,
            metadata={
                **spec.metadata,
                "state_species": state_species,
                "output_species": output_species,
            },
        )
        )
    worker_count = _generation_worker_count(config, len(prepared_specs))
    run_files: list[Path] = []
    if worker_count == 1:
        for spec in prepared_specs:
            run_files.append(
                _generate_single_synthetic_run(
                    spec,
                    runs_dir=runs_dir,
                    output_species=output_species,
                )
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _generate_single_synthetic_run,
                    spec,
                    runs_dir=runs_dir,
                    output_species=output_species,
                )
                for spec in prepared_specs
            ]
            for future in futures:
                run_files.append(future.result())
    run_files = sorted(run_files)
    manifest_path, coverage_path = _write_generation_metadata(
        raw_root=raw_root,
        run_files=run_files,
        specs=prepared_specs,
        config=config,
        mode="synthetic",
    )
    return GeneratedRawDataset(
        raw_root=raw_root,
        run_files=run_files,
        manifest_path=manifest_path,
        coverage_path=coverage_path,
    )


def _copy_vulcan_source(source_root: Path, worker_root: Path) -> None:
    if worker_root.exists():
        shutil.rmtree(worker_root)
    shutil.copytree(source_root, worker_root)


def _write_worker_inputs(worker_root: Path, spec: RunSpecification) -> tuple[Path, Path]:
    atm_dir = ensure_dir(worker_root / "atm")
    stellar_dir = ensure_dir(atm_dir / "stellar_flux")
    tp_file = _write_tp_profile(atm_dir / f"{spec.run_id}_tp.txt", spec)
    spectrum_file = write_vulcan_spectrum_txt(
        spec.spectrum,
        stellar_dir / f"{spec.spectrum.name}.txt",
    )
    return tp_file, spectrum_file


def _patch_vulcan_cfg(
    cfg_file: Path,
    *,
    spec: RunSpecification,
    config: dict[str, Any],
    tp_file: Path,
    spectrum_file: Path,
) -> None:
    text = cfg_file.read_text(encoding="utf-8")
    runtime = config["vulcan_runtime"]
    spectrum_cfg = config["stellar_spectrum"]
    element_abundances = _element_abundances_from_spec(spec)
    assignments = dict(runtime.get("cfg_assignments", {}))
    assignments.update(
        {
            "atom_list": _vulcan_atom_list(config),
            "use_photo": bool(config["physics_toggles"]["use_photochemistry"]),
            "use_ion": bool(config["physics_toggles"]["use_ion_chemistry"]),
            "use_Kzz": bool(config["physics_toggles"]["use_eddy_diffusion"]),
            "use_moldiff": bool(config["physics_toggles"]["use_molecular_diffusion"]),
            "use_vm_mol": bool(config["physics_toggles"]["use_upwind_molecular_diffusion"]),
            "use_topflux": bool(config["physics_toggles"]["use_boundary_conditions"]),
            "use_botflux": bool(config["physics_toggles"]["use_boundary_conditions"]),
            "use_condense": bool(config["physics_toggles"]["use_condensation"]),
            "use_settling": bool(config["physics_toggles"]["use_settling"]),
            "use_ini_cold_trap": bool(config["physics_toggles"]["use_initial_cold_trap"]),
            "use_sat_surfaceH2O": bool(config["physics_toggles"]["use_sat_surface_h2o"]),
            "use_lowT_limit_rates": bool(config["physics_toggles"]["use_lowT_limit_rates"]),
            "use_adapt_rtol": bool(config["physics_toggles"]["use_adaptive_rtol"]),
            "ini_mix": "EQ",
            "use_solar": False,
            "network": str(runtime["chemistry_file"]),
            "atm_file": str(tp_file.relative_to(cfg_file.parent)),
            "sflux_file": str(spectrum_file.relative_to(cfg_file.parent)),
            "atm_base": str(runtime["atm_base"]),
            "atm_type": "file",
            "Kzz_prof": "file",
            "T_cross_sp": list(runtime["t_cross_sp"]),
            "nz": int(spec.pressure_bar.size),
            "P_b": float(np.max(spec.pressure_bar) * 1.0e6),
            "P_t": float(np.min(spec.pressure_bar) * 1.0e6),
            "gs": float(spec.globals["gravity_cm_s2"]),
            "r_star": float(spectrum_cfg["radius_rsun"]),
            "orbit_radius": float(spectrum_cfg["semi_major_axis_au"]),
            "sl_angle": float(np.deg2rad(spectrum_cfg["zenith_angle_deg"])),
            "f_diurnal": float(spectrum_cfg["diurnal_factor"]),
            "save_evolution": True,
            "save_evo_frq": 1,
            "use_live_plot": False,
            "use_live_flux": False,
            "use_plot_end": False,
            "use_plot_evo": False,
            "use_save_movie": False,
            "use_flux_movie": False,
            "output_humanread": False,
            "plot_TP": False,
            "use_print_prog": False,
            "out_name": f"{spec.run_id}.vul",
            **element_abundances,
        }
    )
    cfg_file.write_text(patch_python_assignments(text, assignments), encoding="utf-8")


def _trusted_unpickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _fetch(container: Any, *keys: str) -> Any:
    current = container
    for key in keys:
        if isinstance(current, dict):
            current = current[key]
        else:
            current = getattr(current, key)
    return current


def _decode_species_list(values: Any) -> list[str]:
    result: list[str] = []
    for item in values:
        if isinstance(item, bytes):
            result.append(item.decode("utf-8"))
        else:
            result.append(str(item))
    return result


def convert_vulcan_output_to_hdf5(
    vulcan_output_path: Path,
    *,
    output_h5_path: Path,
    spec: RunSpecification,
    config: dict[str, Any],
) -> Path:
    data = _trusted_unpickle(vulcan_output_path)
    species = _decode_species_list(_fetch(data, "variable", "species"))
    pressure_bar = np.asarray(_fetch(data, "atm", "pco"), dtype=np.float64) / 1.0e6
    temperature_k = np.asarray(_fetch(data, "atm", "Tco"), dtype=np.float64)
    kzz_raw = np.asarray(_fetch(data, "atm", "Kzz"), dtype=np.float64)
    if kzz_raw.ndim == 1 and kzz_raw.size == pressure_bar.size - 1:
        kzz_cm2_s = np.concatenate([[kzz_raw[0]], 0.5 * (kzz_raw[:-1] + kzz_raw[1:]), [kzz_raw[-1]]])
    else:
        kzz_cm2_s = np.asarray(kzz_raw, dtype=np.float64)
    if "ymix_time" in _fetch(data, "variable"):
        ymix_time = np.asarray(_fetch(data, "variable", "ymix_time"), dtype=np.float64)
    elif "y_time" in _fetch(data, "variable") and "n_0" in _fetch(data, "atm"):
        y_time = np.asarray(_fetch(data, "variable", "y_time"), dtype=np.float64)
        n0 = np.asarray(_fetch(data, "atm", "n_0"), dtype=np.float64)
        ymix_time = y_time / n0[None, :, None]
    else:
        raise ValueError("VULCAN output is missing both ymix_time and the y_time/n_0 fallback.")
    time_s = np.asarray(_fetch(data, "variable", "t_time"), dtype=np.float64)
    if ymix_time.shape[0] != time_s.size:
        raise ValueError("VULCAN output does not contain a consistent time-history trajectory.")
    state_species = list(config["data_spec"]["state_species"])
    output_species = list(config["data_spec"]["output_species"])
    state_indices = [species.index(name) for name in state_species]
    output_indices = [species.index(name) for name in output_species]
    ymix_state = ymix_time[..., state_indices]
    ymix_output = ymix_time[..., output_indices]
    converted_spec = RunSpecification(
        run_id=spec.run_id,
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        kzz_cm2_s=kzz_cm2_s,
        initial_ymix=ymix_state[0],
        time_s=time_s,
        globals=spec.globals,
        spectrum=spec.spectrum,
        metadata={
            **spec.metadata,
            "state_species": state_species,
            "output_species": list(config["data_spec"]["output_species"]),
        },
    )
    return write_raw_run_hdf5(
        output_h5_path,
        spec=converted_spec,
        ymix_state=ymix_state,
        ymix_output=ymix_output,
        output_species=output_species,
    )


def _validated_vulcan_paths(config: dict[str, Any], *, project_root: Path) -> tuple[Path, Path]:
    source_root = resolve_path(config["paths"]["vulcan_source_root"], project_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Configured VULCAN source root does not exist: {source_root}")
    chemistry_file = source_root / str(config["vulcan_runtime"]["chemistry_file"])
    if not chemistry_file.exists():
        raise FileNotFoundError(f"Configured chemistry file does not exist: {chemistry_file}")
    cfg_file = source_root / str(config["vulcan_runtime"]["cfg_file"])
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configured VULCAN cfg file does not exist: {cfg_file}")
    return source_root, chemistry_file


def _run_single_vulcan_spec(
    spec: RunSpecification,
    *,
    source_root: Path,
    worker_base: Path,
    runs_dir: Path,
    config: dict[str, Any],
) -> Path:
    worker_root = worker_base / spec.run_id
    _copy_vulcan_source(source_root, worker_root)
    tp_file, spectrum_file = _write_worker_inputs(worker_root, spec)
    cfg_file = worker_root / config["vulcan_runtime"]["cfg_file"]
    _patch_vulcan_cfg(
        cfg_file,
        spec=spec,
        config=config,
        tp_file=tp_file,
        spectrum_file=spectrum_file,
    )
    python_executable = str(config["vulcan_runtime"]["python_executable"])
    if bool(config["vulcan_runtime"]["regenerate_chem_funs"]):
        subprocess.run(
            [python_executable, "make_chem_funs.py"],
            cwd=worker_root,
            check=True,
        )
        vulcan_cmd = [python_executable, "vulcan.py", "-n"]
    else:
        vulcan_cmd = [python_executable, "vulcan.py"]
    subprocess.run(vulcan_cmd, cwd=worker_root, check=True)

    output_candidates = sorted((worker_root / "output").glob("*.vul"))
    if not output_candidates:
        raise FileNotFoundError(f"No VULCAN output file found for run {spec.run_id} under {worker_root / 'output'}.")
    output_h5 = runs_dir / f"{spec.run_id}.h5"
    return convert_vulcan_output_to_hdf5(
        output_candidates[-1],
        output_h5_path=output_h5,
        spec=spec,
        config=config,
    )


def run_vulcan_generation(
    config: dict[str, Any],
    *,
    project_root: Path,
    num_runs: int | None = None,
) -> GeneratedRawDataset:
    raw_root, runs_dir, reusable_files = _prepare_generation_directory(
        config,
        project_root=project_root,
        num_runs=num_runs,
    )
    if reusable_files is not None:
        manifest_path = raw_root / "generation_manifest.json"
        coverage_path = raw_root / "sampling_coverage.json"
        return GeneratedRawDataset(
            raw_root=raw_root,
            run_files=reusable_files,
            manifest_path=manifest_path if manifest_path.exists() else None,
            coverage_path=coverage_path if coverage_path.exists() else None,
        )
    source_root, _ = _validated_vulcan_paths(config, project_root=project_root)
    specs = sample_run_specifications(
        config=config,
        project_root=project_root,
        num_runs=num_runs,
        seed=int(config["generation"]["seed"]),
    )
    worker_base = resolve_path(config["vulcan_runtime"]["worker_root"], project_root)
    prepared_specs: list[RunSpecification] = []
    for spec in specs:
        prepared_specs.append(
            RunSpecification(
            run_id=spec.run_id,
            pressure_bar=spec.pressure_bar,
            temperature_k=spec.temperature_k,
            kzz_cm2_s=spec.kzz_cm2_s,
            initial_ymix=spec.initial_ymix,
            time_s=spec.time_s,
            globals=spec.globals,
            spectrum=spec.spectrum,
            metadata={
                **spec.metadata,
                "state_species": list(config["data_spec"]["state_species"]),
                "output_species": list(config["data_spec"]["output_species"]),
            },
        )
        )
    worker_count = _generation_worker_count(config, len(prepared_specs))
    run_files: list[Path] = []
    if worker_count == 1:
        for spec in prepared_specs:
            run_files.append(
                _run_single_vulcan_spec(
                    spec,
                    source_root=source_root,
                    worker_base=worker_base,
                    runs_dir=runs_dir,
                    config=config,
                )
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _run_single_vulcan_spec,
                    spec,
                    source_root=source_root,
                    worker_base=worker_base,
                    runs_dir=runs_dir,
                    config=config,
                )
                for spec in prepared_specs
            ]
            for future in futures:
                run_files.append(future.result())
    run_files = sorted(run_files)
    manifest_path, coverage_path = _write_generation_metadata(
        raw_root=raw_root,
        run_files=run_files,
        specs=prepared_specs,
        config=config,
        mode="vulcan",
    )
    return GeneratedRawDataset(
        raw_root=raw_root,
        run_files=run_files,
        manifest_path=manifest_path,
        coverage_path=coverage_path,
    )


def generate_raw_dataset(
    config: dict[str, Any],
    *,
    project_root: Path,
    num_runs: int | None = None,
) -> GeneratedRawDataset:
    mode = str(config["generation"]["mode"]).lower()
    if mode == "synthetic":
        return generate_synthetic_raw_runs(config, project_root=project_root, num_runs=num_runs)
    if mode == "vulcan":
        return run_vulcan_generation(config, project_root=project_root, num_runs=num_runs)
    raise ValueError(f"Unsupported generation mode: {mode}")
