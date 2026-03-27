"""Raw dataset generation: VULCAN run orchestration and anchor state construction.

This module is the first stage of the data pipeline.  It samples
atmospheric parameters (via ``sampling.sample_run_specifications``),
then either:

* **equilibrium mode** — calls FastChem to compute chemical equilibrium
  at each pressure level and writes a simplified HDF5 per run, or
* **full_vulcan mode** — patches the VULCAN configuration, launches
  VULCAN as a subprocess, and writes the full trajectory HDF5, or
* **synthetic mode** — generates a heuristic smoke-test trajectory
  without calling VULCAN at all (useful for pipeline development).

All generated HDF5 files follow a shared layout (see ``spec.md``,
"Shared Data Contract") and are consumed downstream by
``preprocess.py``.  A generation manifest and sampling-coverage
summary are persisted alongside the runs for provenance.
"""

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

from ..utils.config import is_equilibrium, task_kind
from ..utils.helpers import ensure_dir, get_logger, resolve_path

LOGGER = get_logger(__name__)
from ..utils.provenance import manifest_for_files
from .sampling import RunSpecification, sample_run_specifications
from .spectrum import write_vulcan_spectrum_txt


def build_flat_h2_he_anchor(
    species_order: list[str],
    *,
    nz: int,
    h2_fraction: float = 0.85,
    he_fraction: float = 0.15,
    floor: float = 1.0e-12,
) -> np.ndarray:
    """Build a flat H2/He-dominated anchor state on the requested species grid.

    Returns a ``(nz, n_species)`` array where H2 and He carry the bulk
    of the composition and all other species are set to ``floor``.  The
    result is row-normalized so mixing ratios sum to 1.0 at each level.
    """
    state = np.full((int(nz), len(species_order)), float(floor), dtype=np.float64)
    idx = {name: i for i, name in enumerate(species_order)}
    if "H2" in idx:
        state[:, idx["H2"]] = float(h2_fraction)
    if "He" in idx:
        state[:, idx["He"]] = float(he_fraction)
    state /= np.sum(state, axis=1, keepdims=True)
    return state


# Solar photospheric abundances (Asplund et al. 2009) expressed as
# number ratios relative to hydrogen.  These are scaled by 10^[M/H]
# to produce the per-run elemental abundances fed to FastChem.
_SOLAR_ELEMENT_ABUNDANCES = {
    "O_H": 5.37e-4,
    "C_H": 2.95e-4,
    "N_H": 7.08e-5,
    "S_H": 1.41e-5,
    "He_H": 8.38e-2,
}
_FASTCHEM_METALLICITY_SCALED_ELEMENTS = {
    "C",
    "N",
    "O",
    "S",
    "P",
    "Si",
    "Ti",
    "V",
    "Cl",
    "K",
    "Na",
    "Mg",
    "F",
    "Ca",
    "Fe",
}


@dataclass(frozen=True)
class GeneratedRawDataset:
    """Paths describing one completed raw-data generation run."""
    raw_root: Path
    run_files: list[Path]
    consolidated_path: Path | None = None
    manifest_path: Path | None = None
    coverage_path: Path | None = None


def patch_python_assignments(text: str, assignments: dict[str, Any]) -> str:
    """Patch simple ``name = value`` assignments in a Python config file.

    Used to inject sampled parameters into VULCAN's ``vulcan_cfg.py``
    before launching each run.  Missing keys are appended at the end.
    """
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
    """Resolve the requested run count from the override or config."""
    return int(config["generation"]["num_runs"] if num_runs is None else num_runs)


def consolidate_runs_to_single_hdf5(
    run_files: list[Path],
    output_path: Path,
    *,
    delete_originals: bool = True,
) -> Path:
    """Merge per-run HDF5 files into a single file with one group per run.

    Each run becomes a top-level group named after the file stem
    (e.g. ``run_00000``).  Uses ``h5py.Group.copy`` for efficient
    internal copying without materializing data in Python.
    """
    ensure_dir(output_path.parent)
    with h5py.File(output_path, "w") as dest:
        for run_file in sorted(run_files):
            run_id = run_file.stem
            with h5py.File(run_file, "r") as src:
                dest_group = dest.create_group(run_id)
                for key in src:
                    src.copy(src[key], dest_group, name=key)
    LOGGER.info("Consolidated %d runs into %s", len(run_files), output_path)
    if delete_originals:
        for run_file in run_files:
            run_file.unlink()
        parent = run_files[0].parent if run_files else None
        if parent is not None and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
            LOGGER.info("Removed empty directory %s", parent)
    return output_path


def list_run_ids_from_consolidated(consolidated_path: Path) -> list[str]:
    """Return sorted run-group names from a consolidated HDF5 file."""
    with h5py.File(consolidated_path, "r") as f:
        return sorted(f.keys())


def _generation_worker_count(config: dict[str, Any], total_runs: int) -> int:
    """Cap the worker count by both config and the number of runs."""
    configured = int(config["generation"]["parallel_workers"])
    return max(1, min(configured, total_runs))


def _target_mode(config: dict[str, Any]) -> str:
    """Return the normalized generation target mode."""
    return str(config["generation"]["target_mode"]).lower()


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
    consolidated_path = raw_root / "runs.h5"
    existing_files = sorted(runs_dir.glob("run_*.h5"))
    # Also count runs inside a consolidated file, if present.
    consolidated_count = 0
    if consolidated_path.exists():
        consolidated_count = len(list_run_ids_from_consolidated(consolidated_path))
    if bool(config["generation"]["overwrite"]):
        for path in existing_files:
            path.unlink()
        if consolidated_path.exists():
            consolidated_path.unlink()
        return raw_root, runs_dir, None
    # Prefer consolidated file over per-file layout for reuse.
    total_existing = consolidated_count + len(existing_files)
    if total_existing > 0:
        if bool(config["generation"]["reuse_raw_if_present"]) and total_existing == requested_runs:
            manifest_path = raw_root / "generation_manifest.json"
            if not manifest_path.exists():
                raise RuntimeError(
                    "Existing raw runs cannot be safely reused because generation_manifest.json is missing. "
                    "Set generation.overwrite=true to regenerate them."
                )
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_mode = str(manifest_payload.get("mode", "")).lower()
            manifest_task_kind = str(manifest_payload.get("task_kind", "")).lower()
            current_mode = str(config["generation"]["mode"]).lower()
            current_task_kind = task_kind(config)
            if manifest_mode != current_mode or manifest_task_kind != current_task_kind:
                raise RuntimeError(
                    "Existing raw runs were generated with a different generation mode or task kind. "
                    "Set generation.overwrite=true to regenerate a compatible dataset."
                )
            # Return consolidated path as a single-element list when no per-file runs exist.
            reuse_files = existing_files if existing_files else [consolidated_path]
            return raw_root, runs_dir, reuse_files
        raise RuntimeError(
            f"Found {total_existing} existing raw runs (per-file: {len(existing_files)}, "
            f"consolidated: {consolidated_count}). "
            "Set generation.overwrite=true or align generation.num_runs with the existing dataset."
        )
    return raw_root, runs_dir, None


def _coverage_fraction(low: float, high: float, observed_low: float, observed_high: float) -> float:
    """Compute the covered fraction of a configured parameter interval."""
    span = max(high - low, 1.0e-12)
    return float(np.clip((observed_high - observed_low) / span, 0.0, 1.0))


def _sampling_coverage_payload(
    *,
    config: dict[str, Any],
    specs: list[RunSpecification],
    run_files: list[Path],
    mode: str,
) -> dict[str, Any]:
    """Summarize the realized dataset coverage against the configured ranges.

    Computes the fraction of each configured parameter interval that
    was actually covered by the sampled runs.  Written to
    ``sampling_coverage.json`` for diagnostic inspection.
    """
    equilibrium = is_equilibrium(config)
    metallicity = np.asarray([spec.globals["metallicity_log10"] for spec in specs], dtype=np.float64)
    c_to_o = np.asarray([spec.globals["c_to_o"] for spec in specs], dtype=np.float64)
    s_to_o = np.asarray([spec.globals.get("s_to_o", 0.026) for spec in specs], dtype=np.float64)
    temperature_rows: list[np.ndarray] = []
    pressure_rows: list[np.ndarray] = []
    for path in run_files:
        with h5py.File(path, "r") as handle:
            temperature_rows.append(np.asarray(handle["inputs/temperature_k"], dtype=np.float64))
            pressure_rows.append(np.asarray(handle["inputs/pressure_bar"], dtype=np.float64))
    temperature = np.concatenate(temperature_rows, axis=0)
    pressure = np.concatenate(pressure_rows, axis=0)
    metallicity_range = [float(x) for x in config["sampling"]["metallicity_log10_range"]]
    c_to_o_range = [float(x) for x in config["sampling"]["c_to_o_range"]]
    s_to_o_range = [float(x) for x in config["sampling"]["s_to_o_range"]]
    temperature_range = [float(x) for x in config["sampling"]["temperature_range_k"]]

    configured_ranges: dict[str, Any] = {
        "metallicity_log10": metallicity_range,
        "c_to_o": c_to_o_range,
        "s_to_o": s_to_o_range,
        "temperature_k": temperature_range,
        "pressure_bar": [
            float(config["sampling"]["pressure_top_bar"]),
            float(config["sampling"]["pressure_bottom_bar"]),
        ],
    }
    realized: dict[str, Any] = {
        "metallicity_log10": {
            "min": float(np.min(metallicity)),
            "max": float(np.max(metallicity)),
            "coverage_fraction": _coverage_fraction(
                *metallicity_range, float(np.min(metallicity)), float(np.max(metallicity)),
            ),
        },
        "c_to_o": {
            "min": float(np.min(c_to_o)),
            "max": float(np.max(c_to_o)),
            "coverage_fraction": _coverage_fraction(*c_to_o_range, float(np.min(c_to_o)), float(np.max(c_to_o))),
        },
        "s_to_o": {
            "min": float(np.min(s_to_o)),
            "max": float(np.max(s_to_o)),
            "coverage_fraction": _coverage_fraction(*s_to_o_range, float(np.min(s_to_o)), float(np.max(s_to_o))),
        },
        "temperature_k": {
            "min": float(np.min(temperature)),
            "max": float(np.max(temperature)),
            "coverage_fraction": _coverage_fraction(
                *temperature_range, float(np.min(temperature)), float(np.max(temperature)),
            ),
        },
        "pressure_bar": {
            "min": float(np.min(pressure)),
            "max": float(np.max(pressure)),
        },
    }

    if not equilibrium:
        gravity = np.asarray([spec.globals["gravity_cm_s2"] for spec in specs], dtype=np.float64)
        log10_kzz_rows: list[np.ndarray] = []
        time_step_rows: list[np.ndarray] = []
        for path in run_files:
            with h5py.File(path, "r") as handle:
                log10_kzz_rows.append(np.log10(np.asarray(handle["inputs/kzz_cm2_s"], dtype=np.float64)))
                time_s = np.asarray(handle["trajectory/time_s"], dtype=np.float64)
                if time_s.size > 1:
                    time_step_rows.append(np.log10(np.diff(time_s)))
        log10_kzz = np.concatenate(log10_kzz_rows, axis=0)
        time_step_log10 = np.concatenate(time_step_rows, axis=0) if time_step_rows else np.zeros((0,), dtype=np.float64)
        gravity_range = [float(x) for x in config["sampling"]["gravity_range_cm_s2"]]
        kzz_value = float(config["sampling"]["kzz_cm2_s"])
        log10_kzz_value = float(np.log10(max(kzz_value, 1.0e-30)))
        kzz_range = [log10_kzz_value, log10_kzz_value]
        time_range = [
            float(config["sampling"]["time_step_log10_min_s"]),
            float(config["sampling"]["time_step_log10_max_s"]),
        ]
        configured_ranges.update({
            "gravity_cm_s2": gravity_range,
            "log10_kzz_cm2_s": kzz_range,
            "log10_adjacent_dt_s": time_range,
        })
        realized.update({
            "gravity_cm_s2": {
                "min": float(np.min(gravity)),
                "max": float(np.max(gravity)),
                "coverage_fraction": _coverage_fraction(*gravity_range, float(np.min(gravity)), float(np.max(gravity))),
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
        })

    return {
        "mode": mode,
        "task_kind": task_kind(config),
        "target_mode": _target_mode(config),
        "model_type": "equilibrium" if equilibrium else "transition",
        "num_runs": len(specs),
        "configured_ranges": configured_ranges,
        "realized_summary": realized,
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
        "task_kind": task_kind(config),
        "target_mode": _target_mode(config),
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
    """Return whether the configured species basis includes sulfur chemistry."""
    species = list(config["data_spec"]["state_species"]) + list(config["data_spec"]["output_species"])
    return any("S" in name for name in species)


def _vulcan_atom_list(config: dict[str, Any]) -> list[str]:
    """Build the atom list expected by VULCAN/FastChem for this config."""
    atoms = ["H", "O", "C", "N", "He"]
    if _sulfur_enabled(config):
        atoms.append("S")
    return atoms


def _element_abundances_from_spec(spec: RunSpecification) -> dict[str, float]:
    """Map one sampled run specification to elemental abundance scalars.

    Metallicity scaling follows the convention [M/H] = log10(Z/Z_sun),
    so all metals are multiplied by 10^[M/H].  C/O and S/O ratios
    override the default solar proportions for carbon and sulfur.
    """
    metal_scale = 10.0 ** float(spec.globals["metallicity_log10"])
    oxygen_h = _SOLAR_ELEMENT_ABUNDANCES["O_H"] * metal_scale
    s_to_o = spec.globals.get("s_to_o")
    if s_to_o is not None:
        sulfur_h = float(oxygen_h * s_to_o)
    else:
        sulfur_h = float(_SOLAR_ELEMENT_ABUNDANCES["S_H"] * metal_scale)
    return {
        "O_H": float(oxygen_h),
        "C_H": float(oxygen_h * spec.globals["c_to_o"]),
        "N_H": float(_SOLAR_ELEMENT_ABUNDANCES["N_H"] * metal_scale),
        "S_H": sulfur_h,
        "He_H": float(_SOLAR_ELEMENT_ABUNDANCES["He_H"]),
        "fastchem_met_scale": float(metal_scale),
    }


def _write_tp_profile(path: Path, spec: RunSpecification) -> Path:
    """Write the temperature-pressure-Kzz profile consumed by VULCAN."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Pressure(bar) Temperature(K) Kzz(cm2/s)\n")
        handle.write("Pressure Temp Kzz\n")
        for p, t, kzz in zip(spec.pressure_bar, spec.temperature_k, spec.kzz_cm2_s):
            handle.write(f"{p:.8e} {t:.8f} {kzz:.8e}\n")
    return path


def _write_scalar_metadata(handle: h5py.File, metadata: dict[str, Any]) -> None:
    """Persist flat scalar metadata fields for provenance without duplicating array payloads."""
    scalar_items: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, bool):
            scalar_items[key] = np.bool_(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            scalar_items[key] = value
        elif isinstance(value, str):
            scalar_items[key] = np.bytes_(value)
    if not scalar_items:
        return
    metadata_group = handle.create_group("metadata")
    for key, value in sorted(scalar_items.items()):
        metadata_group.create_dataset(key, data=value)


def write_equilibrium_hdf5(
    path: Path,
    *,
    spec: RunSpecification,
    equilibrium_ymix: np.ndarray,
    state_species: list[str],
    output_species: list[str],
) -> Path:
    """Write a simplified HDF5 for equilibrium-only runs (no trajectory/spectrum).

    Layout::

        inputs/pressure_bar      (nz,)
        inputs/temperature_k     (nz,)
        inputs/output_species    (n_species,) string
        globals/{key}            scalar per global
        equilibrium/ymix         (nz, n_species)
    """
    ensure_dir(path.parent)
    with h5py.File(path, "w") as handle:
        inputs = handle.create_group("inputs")
        inputs.create_dataset("pressure_bar", data=np.asarray(spec.pressure_bar, dtype=np.float64))
        inputs.create_dataset("temperature_k", data=np.asarray(spec.temperature_k, dtype=np.float64))
        inputs.create_dataset("state_species", data=np.asarray(state_species, dtype="S"))
        inputs.create_dataset("output_species", data=np.asarray(output_species, dtype="S"))
        inputs.create_dataset("target_mode", data=np.bytes_("equilibrium"))
        globals_group = handle.create_group("globals")
        for key, value in sorted(spec.globals.items()):
            globals_group.create_dataset(key, data=float(value))
        _write_scalar_metadata(handle, spec.metadata)
        eq_group = handle.create_group("equilibrium")
        eq_group.create_dataset("ymix", data=np.asarray(equilibrium_ymix, dtype=np.float64))
    return path


def write_raw_run_hdf5(
    path: Path,
    *,
    spec: RunSpecification,
    ymix_state: np.ndarray,
    ymix_output: np.ndarray | None = None,
    reference_ymix_state: np.ndarray | None = None,
    output_species: list[str] | None = None,
    target_mode: str = "trajectory",
) -> Path:
    """Write one transition-style raw run in the repository HDF5 contract.

    Layout::

        inputs/pressure_bar          (nz,)
        inputs/temperature_k         (nz,)
        inputs/kzz_cm2_s             (nz,)
        inputs/state_species         (n_state,) string
        inputs/output_species        (n_output,) string
        inputs/reference_ymix_state  (nz, n_state)
        globals/{key}                scalar per global
        trajectory/time_s            (n_steps,)
        trajectory/ymix_state        (n_steps, nz, n_state)
        trajectory/ymix_output       (n_steps, nz, n_output)
        spectrum/name                string
        spectrum/wavelength_nm       (n_wav,)
        spectrum/flux_erg_cm2_s_nm   (n_wav,)
    """
    ensure_dir(path.parent)
    output_species = list(output_species or spec.metadata.get("output_species", spec.metadata["state_species"]))
    ymix_output_array = np.asarray(ymix_output if ymix_output is not None else ymix_state, dtype=np.float64)
    initial = spec.initial_ymix
    reference_state_array = np.asarray(
        reference_ymix_state if reference_ymix_state is not None else (initial if initial is not None else ymix_state[0]),
        dtype=np.float64,
    )
    with h5py.File(path, "w") as handle:
        inputs = handle.create_group("inputs")
        inputs.create_dataset("pressure_bar", data=np.asarray(spec.pressure_bar, dtype=np.float64))
        inputs.create_dataset("temperature_k", data=np.asarray(spec.temperature_k, dtype=np.float64))
        if spec.kzz_cm2_s is not None:
            inputs.create_dataset("kzz_cm2_s", data=np.asarray(spec.kzz_cm2_s, dtype=np.float64))
        inputs.create_dataset(
            "state_species",
            data=np.asarray(spec.metadata["state_species"], dtype="S"),
        )
        inputs.create_dataset(
            "output_species",
            data=np.asarray(output_species, dtype="S"),
        )
        inputs.create_dataset("reference_ymix_state", data=reference_state_array)
        inputs.create_dataset("target_mode", data=np.bytes_(str(target_mode)))
        globals_group = handle.create_group("globals")
        for key, value in sorted(spec.globals.items()):
            globals_group.create_dataset(key, data=float(value))
        _write_scalar_metadata(handle, spec.metadata)
        trajectory = handle.create_group("trajectory")
        time_s = spec.time_s if spec.time_s is not None else np.array([0.0], dtype=np.float64)
        trajectory.create_dataset("time_s", data=np.asarray(time_s, dtype=np.float64))
        trajectory.create_dataset("ymix_state", data=np.asarray(ymix_state, dtype=np.float64))
        trajectory.create_dataset("ymix_output", data=ymix_output_array)
        if spec.spectrum is not None:
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
    """Compute a simple second-difference operator along the vertical axis."""
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
    """Estimate a crude UV attenuation profile from the local composition."""
    uv_mask = wavelength_nm <= 300.0
    if not np.any(uv_mask):
        base_uv = 1.0
    else:
        uv_flux = float(np.trapz(spectrum[uv_mask], wavelength_nm[uv_mask]))
        total_flux = float(np.trapz(spectrum, wavelength_nm))
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
    """Compute heuristic chemistry tendencies for the synthetic smoke model.

    This is *not* real chemistry — the rates and branching ratios are
    hand-tuned to produce plausible-looking disequilibrium trajectories
    for pipeline testing without invoking VULCAN.  Includes simplified
    H/O radical, carbon, nitrogen, and sulfur photochemistry channels.
    """
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
    """Clamp and renormalize a state so mixing ratios remain physical."""
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
    """Integrate the synthetic chemistry tendencies over the sampled time grid."""
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
    """Slice the full state trajectory down to the configured output species."""
    indices = [state_species.index(name) for name in output_species]
    return np.asarray(trajectory[..., indices], dtype=np.float64)


def _build_reference_equilibrium_shell(
    *,
    reference_ymix_state: np.ndarray,
    state_species: list[str],
    output_species: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the two-step equilibrium shell used for equilibrium-only supervision."""
    flat_anchor = build_flat_h2_he_anchor(
        state_species,
        nz=int(reference_ymix_state.shape[0]),
    )
    reference_output = _slice_output_trajectory(
        reference_ymix_state[None, :, :],
        state_species=state_species,
        output_species=output_species,
    )[0]
    flat_output = _slice_output_trajectory(
        flat_anchor[None, :, :],
        state_species=state_species,
        output_species=output_species,
    )[0]
    time_s = np.asarray([0.0, 1.0], dtype=np.float64)
    ymix_state = np.stack([flat_anchor, reference_ymix_state], axis=0)
    ymix_output = np.stack([flat_output, reference_output], axis=0)
    return time_s, ymix_state, ymix_output


def _prepend_reference_state(
    *,
    time_s: np.ndarray,
    ymix_state: np.ndarray,
    ymix_output: np.ndarray,
    reference_ymix_state: np.ndarray,
    reference_ymix_output: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepend the exact reference state to a saved trajectory."""
    if time_s.size == 0:
        return (
            np.asarray([0.0], dtype=np.float64),
            reference_ymix_state[None, :, :],
            reference_ymix_output[None, :, :],
        )
    if np.isclose(float(time_s[0]), 0.0) and np.allclose(
        ymix_state[0],
        reference_ymix_state,
        rtol=0.0,
        atol=1.0e-12,
    ):
        return time_s, ymix_state, ymix_output
    if np.isclose(float(time_s[0]), 0.0):
        updated_state = np.asarray(ymix_state, dtype=np.float64).copy()
        updated_output = np.asarray(ymix_output, dtype=np.float64).copy()
        updated_state[0] = reference_ymix_state
        updated_output[0] = reference_ymix_output
        return np.asarray(time_s, dtype=np.float64), updated_state, updated_output
    return (
        np.concatenate([np.asarray([0.0], dtype=np.float64), np.asarray(time_s, dtype=np.float64)]),
        np.concatenate([reference_ymix_state[None, :, :], np.asarray(ymix_state, dtype=np.float64)], axis=0),
        np.concatenate([reference_ymix_output[None, :, :], np.asarray(ymix_output, dtype=np.float64)], axis=0),
    )


def _generate_single_synthetic_run(
    spec: RunSpecification,
    *,
    runs_dir: Path,
    output_species: list[str],
    target_mode: str,
) -> Path:
    """Generate and write one synthetic raw run."""
    state_species = list(spec.metadata["state_species"])
    reference_ymix_state = np.asarray(spec.initial_ymix, dtype=np.float64)
    if target_mode == "equilibrium_only":
        time_s, trajectory, output_trajectory = _build_reference_equilibrium_shell(
            reference_ymix_state=reference_ymix_state,
            state_species=state_species,
            output_species=output_species,
        )
        spec = RunSpecification(
            run_id=spec.run_id,
            pressure_bar=spec.pressure_bar,
            temperature_k=spec.temperature_k,
            kzz_cm2_s=spec.kzz_cm2_s,
            initial_ymix=spec.initial_ymix,
            time_s=time_s,
            globals=spec.globals,
            spectrum=spec.spectrum,
            metadata=spec.metadata,
        )
    else:
        trajectory = simulate_synthetic_trajectory(spec)
        output_trajectory = _slice_output_trajectory(
            trajectory,
            state_species=state_species,
            output_species=output_species,
        )
    h5_path = runs_dir / f"{spec.run_id}.h5"
    write_raw_run_hdf5(
        h5_path,
        spec=spec,
        ymix_state=trajectory,
        ymix_output=output_trajectory,
        reference_ymix_state=reference_ymix_state,
        output_species=output_species,
        target_mode=target_mode,
    )
    return h5_path


def generate_synthetic_raw_runs(
    config: dict[str, Any],
    *,
    project_root: Path,
    num_runs: int | None = None,
) -> GeneratedRawDataset:
    """Generate the synthetic smoke-test raw dataset."""
    LOGGER.info("Synthetic generation starting (num_runs=%s)", num_runs or "config default")
    raw_root, runs_dir, reusable_files = _prepare_generation_directory(
        config,
        project_root=project_root,
        num_runs=num_runs,
    )
    if reusable_files is not None:
        LOGGER.info("Reusing %d existing runs from %s", len(reusable_files), raw_root)
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
    target_mode = _target_mode(config)
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
                    target_mode=target_mode,
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
                    target_mode=target_mode,
                )
                for spec in prepared_specs
            ]
            for future in futures:
                run_files.append(future.result())
    run_files = sorted(run_files)
    LOGGER.info("Synthetic generation complete: %d runs written to %s", len(run_files), runs_dir)
    consolidated_path = consolidate_runs_to_single_hdf5(
        run_files, raw_root / "runs.h5",
    )
    manifest_path, coverage_path = _write_generation_metadata(
        raw_root=raw_root,
        run_files=[consolidated_path],
        specs=prepared_specs,
        config=config,
        mode="synthetic",
    )
    return GeneratedRawDataset(
        raw_root=raw_root,
        run_files=run_files,
        consolidated_path=consolidated_path,
        manifest_path=manifest_path,
        coverage_path=coverage_path,
    )


def _copy_vulcan_source(source_root: Path, worker_root: Path) -> None:
    """Create a fresh worker-local copy of the VULCAN source tree."""
    if worker_root.exists():
        shutil.rmtree(worker_root)
    shutil.copytree(source_root, worker_root)


def _copy_fastchem_runtime(source_root: Path, worker_root: Path) -> Path:
    """Copy just the FastChem runtime subset needed for equilibrium runs."""
    if worker_root.exists():
        shutil.rmtree(worker_root)
    worker_root.mkdir(parents=True, exist_ok=True)
    fastchem_root = worker_root / "fastchem_vulcan"
    ensure_dir(fastchem_root)
    shutil.copy2(source_root / "fastchem_vulcan" / "fastchem", fastchem_root / "fastchem")
    shutil.copytree(source_root / "fastchem_vulcan" / "input", fastchem_root / "input")
    shutil.copytree(
        source_root / "fastchem_vulcan" / "fastchem_src" / "chem_input",
        fastchem_root / "fastchem_src" / "chem_input",
    )
    ensure_dir(fastchem_root / "output")
    return fastchem_root


def _write_worker_inputs(worker_root: Path, spec: RunSpecification) -> tuple[Path, Path]:
    """Write the TP profile and stellar spectrum for one worker run."""
    atm_dir = ensure_dir(worker_root / "atm")
    stellar_dir = ensure_dir(atm_dir / "stellar_flux")
    tp_file = _write_tp_profile(atm_dir / f"{spec.run_id}_tp.txt", spec)
    spectrum_file = write_vulcan_spectrum_txt(
        spec.spectrum,
        stellar_dir / f"{spec.spectrum.name}.txt",
    )
    return tp_file, spectrum_file


def _write_fastchem_tp_profile(fastchem_root: Path, spec: RunSpecification) -> Path:
    """Write the FastChem TP profile input file."""
    tp_dir = ensure_dir(fastchem_root / "input" / "vulcan_TP")
    tp_path = tp_dir / "vulcan_TP.dat"
    with tp_path.open("w", encoding="utf-8") as handle:
        handle.write("#p (bar)    T (K)\n")
        for pressure_bar, temperature_k in zip(spec.pressure_bar, spec.temperature_k):
            handle.write(f"{pressure_bar:.8e}\t{temperature_k:.8f}\n")
    return tp_path


def _write_fastchem_element_abundances(
    fastchem_root: Path,
    *,
    spec: RunSpecification,
    config: dict[str, Any],
) -> Path:
    """Write the FastChem elemental abundance file for one sampled run."""
    input_dir = fastchem_root / "input"
    physics = config.get("physics_toggles", {})
    use_ion = bool(physics.get("use_ion_chemistry", False))
    parameters_name = "parameters_ion.dat" if use_ion else "parameters_wo_ion.dat"
    shutil.copyfile(input_dir / parameters_name, input_dir / "parameters.dat")

    element_abundances = _element_abundances_from_spec(spec)
    non_h_atoms = {atom for atom in _vulcan_atom_list(config) if atom != "H"}
    metallicity_offset = float(np.log10(element_abundances["fastchem_met_scale"]))
    solar_file = input_dir / "solar_element_abundances.dat"
    output_lines: list[str] = []
    with solar_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip() or raw_line.startswith("#"):
                output_lines.append(raw_line)
                continue
            parts = raw_line.split()
            species_name = parts[0].strip()
            if species_name in non_h_atoms:
                abundance_h = element_abundances.get(f"{species_name}_H")
                if abundance_h is None:
                    raise ValueError(f"Missing elemental abundance for {species_name} in FastChem setup.")
                output_lines.append(f"{species_name}\t{12.0 + np.log10(float(abundance_h)):.4f}\n")
            elif species_name in _FASTCHEM_METALLICITY_SCALED_ELEMENTS:
                output_lines.append(f"{species_name}\t{float(parts[1]) + metallicity_offset:.4f}\n")
            else:
                output_lines.append(raw_line)
    abundance_path = input_dir / "element_abundances_vulcan.dat"
    abundance_path.write_text("".join(output_lines), encoding="utf-8")
    return abundance_path


def _patch_vulcan_cfg(
    cfg_file: Path,
    *,
    spec: RunSpecification,
    config: dict[str, Any],
    tp_file: Path,
    spectrum_file: Path,
) -> None:
    """Patch the copied ``vulcan_cfg.py`` with run-specific inputs."""
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
    """Load a trusted pickle file produced by the local VULCAN runtime."""
    with path.open("rb") as handle:
        return pickle.load(handle)


def _fetch(container: Any, *keys: str) -> Any:
    """Traverse nested dict/object containers with a shared helper."""
    current = container
    for key in keys:
        if isinstance(current, dict):
            current = current[key]
        else:
            current = getattr(current, key)
    return current


def _decode_species_list(values: Any) -> list[str]:
    """Normalize stored species names to Python strings."""
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
    """Convert a VULCAN pickle output into the raw-run HDF5 contract."""
    data = _trusted_unpickle(vulcan_output_path)
    species = _decode_species_list(_fetch(data, "variable", "species"))

    # Extract atmospheric grid (VULCAN stores pressure in dyn/cm2, convert to bar).
    pressure_bar = np.asarray(_fetch(data, "atm", "pco"), dtype=np.float64) / 1.0e6
    temperature_k = np.asarray(_fetch(data, "atm", "Tco"), dtype=np.float64)

    # Reference (initial) state: convert number densities to mixing ratios.
    if "y_ini" not in _fetch(data, "variable") or "n_0" not in _fetch(data, "atm"):
        raise ValueError("VULCAN output is missing variable.y_ini or atm.n_0 required for exact FastChem extraction.")
    y_ini = np.asarray(_fetch(data, "variable", "y_ini"), dtype=np.float64)
    n0 = np.asarray(_fetch(data, "atm", "n_0"), dtype=np.float64)
    if y_ini.shape[0] != n0.size:
        raise ValueError("VULCAN output contains inconsistent y_ini and atm.n_0 shapes.")

    # Kzz may be on cell edges (nz-1); interpolate to cell centres if needed.
    kzz_raw = np.asarray(_fetch(data, "atm", "Kzz"), dtype=np.float64)
    if kzz_raw.ndim == 1 and kzz_raw.size == pressure_bar.size - 1:
        kzz_cm2_s = np.concatenate([[kzz_raw[0]], 0.5 * (kzz_raw[:-1] + kzz_raw[1:]), [kzz_raw[-1]]])
    else:
        kzz_cm2_s = np.asarray(kzz_raw, dtype=np.float64)

    # Extract time-resolved mixing-ratio trajectory (prefer ymix_time, fall back to y_time/n_0).
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
    reference_ymix_full = y_ini / np.clip(n0[:, None], 1.0e-30, None)
    reference_ymix_state = reference_ymix_full[:, state_indices]
    reference_ymix_output = reference_ymix_full[:, output_indices]
    ymix_state = ymix_time[..., state_indices]
    ymix_output = ymix_time[..., output_indices]
    target_mode = _target_mode(config)
    if target_mode == "equilibrium_only":
        time_s, ymix_state, ymix_output = _build_reference_equilibrium_shell(
            reference_ymix_state=reference_ymix_state,
            state_species=state_species,
            output_species=output_species,
        )
    else:
        time_s, ymix_state, ymix_output = _prepend_reference_state(
            time_s=time_s,
            ymix_state=ymix_state,
            ymix_output=ymix_output,
            reference_ymix_state=reference_ymix_state,
            reference_ymix_output=reference_ymix_output,
        )
    converted_spec = RunSpecification(
        run_id=spec.run_id,
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        kzz_cm2_s=kzz_cm2_s,
        initial_ymix=reference_ymix_state,
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
        reference_ymix_state=reference_ymix_state,
        output_species=output_species,
        target_mode=target_mode,
    )


def convert_fastchem_output_to_hdf5(
    fastchem_output_path: Path,
    *,
    output_h5_path: Path,
    spec: RunSpecification,
    config: dict[str, Any],
) -> Path:
    """Convert a FastChem equilibrium table into the raw-run HDF5 contract."""
    fc = np.genfromtxt(fastchem_output_path, names=True, dtype=None, encoding=None)
    if fc.dtype.names is None:
        raise ValueError(f"FastChem output at {fastchem_output_path} does not contain a named header.")
    state_species = list(config["data_spec"]["state_species"])
    output_species = list(config["data_spec"]["output_species"])
    requested_species = list(dict.fromkeys([*state_species, *output_species]))
    missing = [name for name in requested_species if name not in fc.dtype.names]
    if missing:
        raise ValueError(f"FastChem output is missing requested species: {missing}")
    reference_ymix_state = np.column_stack([np.asarray(fc[name], dtype=np.float64) for name in state_species])
    if reference_ymix_state.shape[0] != spec.pressure_bar.size:
        raise ValueError(
            "FastChem output row count does not match the configured pressure grid size."
        )

    if is_equilibrium(config):
        # Simplified equilibrium format: direct mapping, no trajectory shell.
        equilibrium_ymix = np.column_stack(
            [np.asarray(fc[name], dtype=np.float64) for name in output_species]
        )
        return write_equilibrium_hdf5(
            output_h5_path,
            spec=spec,
            equilibrium_ymix=equilibrium_ymix,
            state_species=state_species,
            output_species=output_species,
        )

    time_s, ymix_state, ymix_output = _build_reference_equilibrium_shell(
        reference_ymix_state=reference_ymix_state,
        state_species=state_species,
        output_species=output_species,
    )
    converted_spec = RunSpecification(
        run_id=spec.run_id,
        pressure_bar=np.asarray(spec.pressure_bar, dtype=np.float64),
        temperature_k=np.asarray(spec.temperature_k, dtype=np.float64),
        globals=spec.globals,
        metadata={
            **spec.metadata,
            "state_species": state_species,
            "output_species": output_species,
        },
        kzz_cm2_s=np.asarray(spec.kzz_cm2_s, dtype=np.float64) if spec.kzz_cm2_s is not None else None,
        initial_ymix=reference_ymix_state,
        time_s=time_s,
        spectrum=spec.spectrum,
    )
    return write_raw_run_hdf5(
        output_h5_path,
        spec=converted_spec,
        ymix_state=ymix_state,
        ymix_output=ymix_output,
        reference_ymix_state=reference_ymix_state,
        output_species=output_species,
        target_mode="equilibrium_only",
    )


def _validated_vulcan_paths(config: dict[str, Any], *, project_root: Path) -> tuple[Path, Path]:
    """Validate the configured VULCAN/FastChem source paths for this run mode."""
    source_root = resolve_path(config["paths"]["vulcan_source_root"], project_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Configured VULCAN source root does not exist: {source_root}")
    if is_equilibrium(config) or _target_mode(config) == "equilibrium_only":
        fastchem_root = source_root / "fastchem_vulcan"
        if not fastchem_root.exists():
            raise FileNotFoundError(f"Configured FastChem runtime does not exist: {fastchem_root}")
        fastchem_binary = fastchem_root / "fastchem"
        if not fastchem_binary.exists():
            raise FileNotFoundError(f"Configured FastChem binary does not exist: {fastchem_binary}")
        fastchem_input = fastchem_root / "input" / "config.input"
        if not fastchem_input.exists():
            raise FileNotFoundError(f"Configured FastChem input config does not exist: {fastchem_input}")
        chemical_elements = fastchem_root / "fastchem_src" / "chem_input" / "chemical_elements.dat"
        if not chemical_elements.exists():
            raise FileNotFoundError(
                "Configured FastChem chemical element table does not exist: "
                f"{chemical_elements}"
            )
        return source_root, fastchem_binary
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
    """Run one full VULCAN worker and convert its output to HDF5."""
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


def _run_single_fastchem_spec(
    spec: RunSpecification,
    *,
    source_root: Path,
    worker_base: Path,
    runs_dir: Path,
    config: dict[str, Any],
) -> Path:
    """Run one FastChem worker and convert its output to HDF5."""
    worker_root = worker_base / spec.run_id
    fastchem_root = _copy_fastchem_runtime(source_root, worker_root)
    _write_fastchem_element_abundances(
        fastchem_root,
        spec=spec,
        config=config,
    )
    _write_fastchem_tp_profile(fastchem_root, spec)
    subprocess.run(["./fastchem", "input/config.input"], cwd=fastchem_root, check=True)
    fastchem_output = fastchem_root / "output" / "vulcan_EQ.dat"
    if not fastchem_output.exists():
        raise FileNotFoundError(f"No FastChem equilibrium output found for run {spec.run_id} under {fastchem_output}.")
    output_h5 = runs_dir / f"{spec.run_id}.h5"
    return convert_fastchem_output_to_hdf5(
        fastchem_output,
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
    """Generate raw data from external VULCAN/FastChem runtimes."""
    LOGGER.info("VULCAN generation starting (num_runs=%s)", num_runs or "config default")
    raw_root, runs_dir, reusable_files = _prepare_generation_directory(
        config,
        project_root=project_root,
        num_runs=num_runs,
    )
    if reusable_files is not None:
        LOGGER.info("Reusing %d existing runs from %s", len(reusable_files), raw_root)
        manifest_path = raw_root / "generation_manifest.json"
        coverage_path = raw_root / "sampling_coverage.json"
        return GeneratedRawDataset(
            raw_root=raw_root,
            run_files=reusable_files,
            manifest_path=manifest_path if manifest_path.exists() else None,
            coverage_path=coverage_path if coverage_path.exists() else None,
        )
    source_root, _ = _validated_vulcan_paths(config, project_root=project_root)
    target_mode = _target_mode(config)
    equilibrium = is_equilibrium(config)
    specs = sample_run_specifications(
        config=config,
        project_root=project_root,
        num_runs=num_runs,
        seed=int(config["generation"]["seed"]),
        include_initial_ymix=not equilibrium and target_mode != "equilibrium_only",
        include_time_grid=not equilibrium and target_mode != "equilibrium_only",
    )
    worker_root_key = "vulcan_runtime" if "vulcan_runtime" in config else None
    worker_base = resolve_path(
        config["vulcan_runtime"]["worker_root"] if worker_root_key else "data/vulcan_workers",
        project_root,
    )
    prepared_specs: list[RunSpecification] = []
    for spec in specs:
        prepared_specs.append(
            RunSpecification(
                run_id=spec.run_id,
                pressure_bar=spec.pressure_bar,
                temperature_k=spec.temperature_k,
                globals=spec.globals,
                metadata={
                    **spec.metadata,
                    "state_species": list(config["data_spec"]["state_species"]),
                    "output_species": list(config["data_spec"]["output_species"]),
                },
                kzz_cm2_s=spec.kzz_cm2_s,
                initial_ymix=spec.initial_ymix,
                time_s=spec.time_s,
                spectrum=spec.spectrum,
            )
        )
    worker_count = _generation_worker_count(config, len(prepared_specs))
    run_files: list[Path] = []
    run_single = _run_single_fastchem_spec if (equilibrium or target_mode == "equilibrium_only") else _run_single_vulcan_spec
    if worker_count == 1:
        for spec in prepared_specs:
            run_files.append(
                run_single(
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
                    run_single,
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
    LOGGER.info("VULCAN generation complete: %d runs written to %s", len(run_files), runs_dir)
    consolidated_path = consolidate_runs_to_single_hdf5(
        run_files, raw_root / "runs.h5",
    )
    manifest_path, coverage_path = _write_generation_metadata(
        raw_root=raw_root,
        run_files=[consolidated_path],
        specs=prepared_specs,
        config=config,
        mode="vulcan",
    )
    return GeneratedRawDataset(
        raw_root=raw_root,
        run_files=run_files,
        consolidated_path=consolidated_path,
        manifest_path=manifest_path,
        coverage_path=coverage_path,
    )


def generate_raw_dataset(
    config: dict[str, Any],
    *,
    project_root: Path,
    num_runs: int | None = None,
) -> GeneratedRawDataset:
    """Dispatch raw-data generation to the configured backend."""
    mode = str(config["generation"]["mode"]).lower()
    if mode == "synthetic":
        return generate_synthetic_raw_runs(config, project_root=project_root, num_runs=num_runs)
    if mode == "vulcan":
        return run_vulcan_generation(config, project_root=project_root, num_runs=num_runs)
    raise ValueError(f"Unsupported generation mode: {mode}")
