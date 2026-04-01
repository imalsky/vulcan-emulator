"""Raw dataset generation: FastChem/VULCAN orchestration and final-state writing.

This module is the first stage of the data pipeline.  It samples
atmospheric parameters (via ``sampling.sample_run_specifications``),
then either:

* **fastchem chemistry** — calls FastChem to compute chemical equilibrium
  at each pressure level and writes a simplified HDF5 per run, or
* **vulcan chemistry** — patches the VULCAN configuration, launches
  VULCAN as a subprocess, and writes the final converged output state, or
* **synthetic mode** — generates a heuristic smoke-test final state
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

from ..utils.config import (
    ELEMENT_INPUT_ORDER,
    PUBLIC_PHYSICS_TOGGLES,
    SUPPORTED_ATM_BASES,
    get_chemistry_type,
    get_model_type,
    uses_fastchem,
)
from ..utils.helpers import ensure_dir, get_logger, resolve_path

LOGGER = get_logger(__name__)
from ..utils.provenance import manifest_for_files
from .sampling import RunSpecification, sample_run_specifications
from .spectrum import write_vulcan_spectrum_txt

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
            manifest_chemistry_type = str(manifest_payload.get("chemistry_type", "")).lower()
            current_mode = str(config["generation"]["mode"]).lower()
            current_chemistry_type = get_chemistry_type(config)
            if manifest_mode != current_mode or manifest_chemistry_type != current_chemistry_type:
                raise RuntimeError(
                    "Existing raw runs were generated with a different generation mode or chemistry_type. "
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
    fastchem = uses_fastchem(config)
    metallicity = np.asarray([spec.globals["metallicity_log10"] for spec in specs], dtype=np.float64)
    c_to_o = np.asarray([spec.globals["c_to_o"] for spec in specs], dtype=np.float64)
    s_to_o = np.asarray([spec.globals.get("s_to_o", 0.026) for spec in specs], dtype=np.float64)
    temperature_rows: list[np.ndarray] = []
    pressure_rows: list[np.ndarray] = []
    for path in run_files:
        with h5py.File(path, "r") as handle:
            sources = [handle] if "inputs" in handle else [handle[run_id] for run_id in sorted(handle.keys())]
            for source in sources:
                temperature_rows.append(np.asarray(source["inputs/temperature_k"], dtype=np.float64))
                pressure_rows.append(np.asarray(source["inputs/pressure_bar"], dtype=np.float64))
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

    if not fastchem:
        gravity = np.asarray([spec.globals["gravity_cm_s2"] for spec in specs], dtype=np.float64)
        log10_kzz_rows: list[np.ndarray] = []
        for path in run_files:
            with h5py.File(path, "r") as handle:
                sources = [handle] if "inputs" in handle else [handle[run_id] for run_id in sorted(handle.keys())]
                for source in sources:
                    log10_kzz_rows.append(np.log10(np.asarray(source["inputs/kzz_cm2_s"], dtype=np.float64)))
        log10_kzz = np.concatenate(log10_kzz_rows, axis=0)
        gravity_range = [float(x) for x in config["sampling"]["gravity_range_cm_s2"]]
        kzz_value = float(config["sampling"]["kzz_cm2_s"])
        log10_kzz_value = float(np.log10(max(kzz_value, 1.0e-30)))
        kzz_range = [log10_kzz_value, log10_kzz_value]
        configured_ranges.update({
            "gravity_cm_s2": gravity_range,
            "log10_kzz_cm2_s": kzz_range,
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
        })

    return {
        "mode": mode,
        "chemistry_type": get_chemistry_type(config),
        "model_type": get_model_type(config),
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
        "chemistry_type": get_chemistry_type(config),
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
    if spec.elemental_abundances_x_h is not None:
        element_profile = np.asarray(spec.elemental_abundances_x_h, dtype=np.float64)
        if element_profile.ndim != 2 or element_profile.shape[1] != len(ELEMENT_INPUT_ORDER):
            raise ValueError(
                "RunSpecification.elemental_abundances_x_h must have shape "
                f"(nz, {len(ELEMENT_INPUT_ORDER)})."
            )
        element_vector = element_profile[0]
        element_scalars = {
            name: float(element_vector[idx])
            for idx, name in enumerate(ELEMENT_INPUT_ORDER)
        }
        oxygen_h = float(element_scalars["O_H"])
        return {
            **element_scalars,
            "fastchem_met_scale": float(oxygen_h / _SOLAR_ELEMENT_ABUNDANCES["O_H"]),
        }
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


def _element_profile_from_spec(spec: RunSpecification) -> np.ndarray:
    """Return the per-level elemental-abundance input profile for one run."""
    if spec.elemental_abundances_x_h is not None:
        profile = np.asarray(spec.elemental_abundances_x_h, dtype=np.float64)
        if profile.ndim != 2 or profile.shape[1] != len(ELEMENT_INPUT_ORDER):
            raise ValueError(
                "RunSpecification.elemental_abundances_x_h must have shape "
                f"(nz, {len(ELEMENT_INPUT_ORDER)})."
            )
        return profile
    scalars = _element_abundances_from_spec(spec)
    vector = np.array([float(scalars[name]) for name in ELEMENT_INPUT_ORDER], dtype=np.float64)
    return np.repeat(vector[None, :], int(spec.pressure_bar.size), axis=0)


def _gravity_profile_from_spec(spec: RunSpecification) -> np.ndarray:
    """Return the per-level gravity input profile for one run."""
    if spec.gravity_cm_s2 is not None:
        profile = np.asarray(spec.gravity_cm_s2, dtype=np.float64)
        if profile.ndim != 1:
            raise ValueError("RunSpecification.gravity_cm_s2 must be a 1-D array.")
        return profile
    gravity_value = float(spec.globals.get("gravity_cm_s2", 0.0))
    return np.full(int(spec.pressure_bar.size), gravity_value, dtype=np.float64)


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
        inputs/element_input_order (n_elements,) string
        inputs/elemental_abundances_x_h (nz, n_elements)
        inputs/gravity_cm_s2     (nz,)
        inputs/output_species    (n_species,) string
        globals/{key}            scalar per global
        equilibrium/ymix         (nz, n_species)
    """
    ensure_dir(path.parent)
    element_profile = _element_profile_from_spec(spec)
    gravity_profile = _gravity_profile_from_spec(spec)
    with h5py.File(path, "w") as handle:
        inputs = handle.create_group("inputs")
        inputs.create_dataset("pressure_bar", data=np.asarray(spec.pressure_bar, dtype=np.float64))
        inputs.create_dataset("temperature_k", data=np.asarray(spec.temperature_k, dtype=np.float64))
        inputs.create_dataset("element_input_order", data=np.asarray(ELEMENT_INPUT_ORDER, dtype="S"))
        inputs.create_dataset("elemental_abundances_x_h", data=element_profile)
        inputs.create_dataset("gravity_cm_s2", data=gravity_profile)
        inputs.create_dataset("state_species", data=np.asarray(state_species, dtype="S"))
        inputs.create_dataset("output_species", data=np.asarray(output_species, dtype="S"))
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
    final_ymix_output: np.ndarray,
    output_species: list[str] | None = None,
) -> Path:
    """Write one full-VULCAN raw run in the repository HDF5 contract.

    Layout::

        inputs/pressure_bar          (nz,)
        inputs/temperature_k         (nz,)
        inputs/kzz_cm2_s             (nz,)
        inputs/element_input_order   (n_elements,) string
        inputs/elemental_abundances_x_h (nz, n_elements)
        inputs/gravity_cm_s2         (nz,)
        inputs/state_species         (n_state,) string
        inputs/output_species        (n_output,) string
        globals/{key}                scalar per global
        final_state/ymix_output      (nz, n_output)
        spectrum/name                string
        spectrum/wavelength_nm       (n_wav,)
        spectrum/flux_erg_cm2_s_nm   (n_wav,)
    """
    ensure_dir(path.parent)
    output_species = list(output_species or spec.metadata.get("output_species", spec.metadata["state_species"]))
    element_profile = _element_profile_from_spec(spec)
    gravity_profile = _gravity_profile_from_spec(spec)
    with h5py.File(path, "w") as handle:
        inputs = handle.create_group("inputs")
        inputs.create_dataset("pressure_bar", data=np.asarray(spec.pressure_bar, dtype=np.float64))
        inputs.create_dataset("temperature_k", data=np.asarray(spec.temperature_k, dtype=np.float64))
        if spec.kzz_cm2_s is not None:
            inputs.create_dataset("kzz_cm2_s", data=np.asarray(spec.kzz_cm2_s, dtype=np.float64))
        inputs.create_dataset("element_input_order", data=np.asarray(ELEMENT_INPUT_ORDER, dtype="S"))
        inputs.create_dataset("elemental_abundances_x_h", data=element_profile)
        inputs.create_dataset("gravity_cm_s2", data=gravity_profile)
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
        _write_scalar_metadata(handle, spec.metadata)
        final_state = handle.create_group("final_state")
        final_state.create_dataset(
            "ymix_output",
            data=np.asarray(final_ymix_output, dtype=np.float64),
        )
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


def _synthetic_vertical_coordinate(pressure_bar: np.ndarray) -> np.ndarray:
    """Return a depth coordinate in [0, 1] with 0 at the top and 1 at the bottom."""
    logp = np.log10(np.clip(np.asarray(pressure_bar, dtype=np.float64), 1.0e-30, None))
    return np.clip((logp - np.min(logp)) / max(np.max(logp) - np.min(logp), 1.0e-8), 0.0, 1.0)


def _synthetic_uv_strength(spec: RunSpecification) -> float:
    """Estimate one scalar UV forcing strength from the selected stellar spectrum."""
    if spec.spectrum is None:
        return 0.0
    wavelength_nm = np.asarray(spec.spectrum.wavelength_nm, dtype=np.float64)
    flux = np.asarray(spec.spectrum.flux_erg_cm2_s_nm, dtype=np.float64)
    uv_mask = wavelength_nm <= 300.0
    if not np.any(uv_mask):
        return 0.0
    total_flux = float(np.trapz(flux, wavelength_nm))
    uv_flux = float(np.trapz(flux[uv_mask], wavelength_nm[uv_mask]))
    return float(np.clip(uv_flux / max(total_flux, 1.0e-30), 0.0, 1.0))


def _assign_species(state: np.ndarray, species_index: dict[str, int], name: str, values: np.ndarray) -> None:
    """Add one profile into the requested species column when it exists."""
    if name in species_index:
        state[:, species_index[name]] += np.asarray(values, dtype=np.float64)


def _synthetic_final_state(
    spec: RunSpecification,
    *,
    output_species: list[str],
) -> np.ndarray:
    """Build one heuristic final-state composition for smoke-test generation.

    The synthetic path is intentionally lightweight. It maps sampled pressure,
    temperature, elemental abundances, Kzz, spectrum, and public physics knobs
    directly to one plausible final-state profile without any timestep or
    anchor-state machinery.
    """
    state_species = list(spec.metadata["state_species"])
    species_index = {name: idx for idx, name in enumerate(state_species)}
    state = np.full((int(spec.pressure_bar.size), len(state_species)), 1.0e-30, dtype=np.float64)

    elements = _element_abundances_from_spec(spec)
    he_h = float(elements["He_H"])
    c_h = float(elements["C_H"])
    o_h = float(elements["O_H"])
    n_h = float(elements["N_H"])
    s_h = float(elements["S_H"])

    pressure_bar = np.asarray(spec.pressure_bar, dtype=np.float64)
    temperature_k = np.asarray(spec.temperature_k, dtype=np.float64)
    depth = _synthetic_vertical_coordinate(pressure_bar)
    top = 1.0 - depth
    hot = np.clip((temperature_k - 1000.0) / 1200.0, 0.0, 1.0)
    cool = np.clip((1200.0 - temperature_k) / 900.0, 0.0, 1.0)
    photo_strength = _synthetic_uv_strength(spec) * float(spec.globals.get("use_photochemistry", 0.0))
    photo_profile = photo_strength * np.exp(-3.0 * depth)
    kzz_profile = np.asarray(spec.kzz_cm2_s if spec.kzz_cm2_s is not None else np.full_like(pressure_bar, 1.0e6), dtype=np.float64)
    mix_strength = (
        np.clip((np.log10(np.clip(kzz_profile, 1.0, None)) - 6.0) / 4.0, 0.0, 1.0)
        * float(spec.globals.get("use_eddy_diffusion", 0.0))
    )
    condense = float(spec.globals.get("use_condensation", 0.0))
    settling = float(spec.globals.get("use_settling", 0.0))
    cold_trap = float(spec.globals.get("use_initial_cold_trap", 0.0))
    sat_surface_h2o = float(spec.globals.get("use_sat_surface_h2o", 0.0))
    molecular_diffusion = float(spec.globals.get("use_molecular_diffusion", 0.0))
    upwind_diffusion = float(spec.globals.get("use_upwind_molecular_diffusion", 0.0))
    boundary_conditions = float(spec.globals.get("use_boundary_conditions", 0.0))
    ion_chemistry = float(spec.globals.get("use_ion_chemistry", 0.0))

    cold_depletion = np.clip(cool * (0.35 + 0.45 * top), 0.0, 1.0)
    sulfur_depletion = np.clip((0.25 * condense + 0.20 * settling + 0.25 * cold_trap) * cold_depletion, 0.0, 0.8)
    water_depletion = np.clip((0.20 * condense + 0.15 * settling + 0.20 * cold_trap) * cold_depletion, 0.0, 0.7)

    _assign_species(
        state,
        species_index,
        "H2O",
        o_h * (0.45 + 0.20 * top + 0.20 * cool + 0.10 * sat_surface_h2o * depth) * (1.0 - 0.35 * photo_profile) * (1.0 - water_depletion),
    )
    _assign_species(
        state,
        species_index,
        "CO",
        c_h * (0.35 + 0.35 * hot + 0.20 * depth + 0.10 * mix_strength),
    )
    _assign_species(
        state,
        species_index,
        "CO2",
        np.minimum(c_h, o_h) * (0.05 + 0.15 * photo_profile + 0.10 * top + 0.10 * molecular_diffusion * top),
    )
    _assign_species(
        state,
        species_index,
        "CH4",
        c_h * (0.18 + 0.42 * depth + 0.08 * mix_strength) * (1.0 - 0.55 * hot),
    )
    _assign_species(
        state,
        species_index,
        "N2",
        n_h * (0.45 + 0.25 * hot + 0.15 * depth + 0.10 * mix_strength),
    )
    _assign_species(
        state,
        species_index,
        "NH3",
        n_h * (0.16 + 0.30 * depth + 0.10 * cool) * (1.0 - 0.40 * photo_profile),
    )
    _assign_species(
        state,
        species_index,
        "H2S",
        s_h * (0.50 + 0.22 * depth + 0.10 * cool) * (1.0 - 0.45 * photo_profile) * (1.0 - sulfur_depletion),
    )
    _assign_species(
        state,
        species_index,
        "SH",
        s_h * (0.004 + 0.03 * photo_profile + 0.01 * upwind_diffusion * top),
    )
    _assign_species(
        state,
        species_index,
        "S",
        s_h * (0.002 + 0.02 * photo_profile + 0.006 * ion_chemistry * top),
    )
    _assign_species(
        state,
        species_index,
        "SO",
        s_h * (0.001 + 0.03 * photo_profile + 0.01 * hot) * (1.0 - 0.20 * sulfur_depletion),
    )
    _assign_species(
        state,
        species_index,
        "SO2",
        s_h * (0.001 + 0.04 * photo_profile + 0.01 * mix_strength) * (1.0 - 0.15 * sulfur_depletion),
    )
    _assign_species(
        state,
        species_index,
        "S2",
        s_h * (0.0005 + 0.01 * depth + 0.005 * upwind_diffusion * top),
    )
    _assign_species(
        state,
        species_index,
        "H",
        1.0e-7 * (1.0 + 30.0 * photo_profile + 5.0 * hot + 3.0 * ion_chemistry + 2.0 * boundary_conditions * top),
    )
    _assign_species(
        state,
        species_index,
        "O",
        o_h * 1.0e-3 * (1.0 + 15.0 * photo_profile + 3.0 * ion_chemistry),
    )
    _assign_species(
        state,
        species_index,
        "OH",
        o_h * 2.0e-3 * (1.0 + 8.0 * photo_profile + 2.0 * boundary_conditions * top),
    )

    base_name = next(
        (
            name
            for name in ("H2", "N2", "CO2", "H2O", "O2")
            if float(spec.globals.get(f"atm_base_{name}", 0.0)) > 0.5
        ),
        "H2",
    )
    heavy_sum = np.sum(state, axis=1)
    heavy_scale = np.where(heavy_sum > 0.98, 0.98 / heavy_sum, 1.0)
    state *= heavy_scale[:, None]
    reservoir = np.clip(1.0 - np.sum(state, axis=1), 1.0e-6, 1.0)

    if base_name == "N2" and "N2" in species_index:
        _assign_species(state, species_index, "N2", 0.82 * reservoir)
        _assign_species(state, species_index, "H2", 0.08 * reservoir)
        _assign_species(state, species_index, "He", 0.10 * reservoir)
    elif base_name == "CO2" and "CO2" in species_index:
        _assign_species(state, species_index, "CO2", 0.72 * reservoir)
        _assign_species(state, species_index, "H2", 0.12 * reservoir)
        _assign_species(state, species_index, "He", 0.16 * reservoir)
    elif base_name == "H2O" and "H2O" in species_index:
        _assign_species(state, species_index, "H2O", 0.65 * reservoir)
        _assign_species(state, species_index, "H2", 0.15 * reservoir)
        _assign_species(state, species_index, "He", 0.20 * reservoir)
    elif base_name == "O2":
        _assign_species(state, species_index, "O", 0.35 * reservoir)
        _assign_species(state, species_index, "OH", 0.20 * reservoir)
        _assign_species(state, species_index, "CO2", 0.15 * reservoir)
        _assign_species(state, species_index, "H2O", 0.10 * reservoir)
        _assign_species(state, species_index, "H2", 0.05 * reservoir)
        _assign_species(state, species_index, "He", 0.15 * reservoir)
    else:
        he_reservoir_fraction = np.clip(he_h / max(1.0 + he_h, 1.0e-6), 0.08, 0.18)
        _assign_species(state, species_index, "H2", (1.0 - he_reservoir_fraction) * reservoir)
        _assign_species(state, species_index, "He", he_reservoir_fraction * reservoir)

    state = np.clip(state, 1.0e-30, None)
    state /= np.sum(state, axis=1, keepdims=True)
    output_indices = [state_species.index(name) for name in output_species]
    return np.asarray(state[:, output_indices], dtype=np.float64)


def _generate_single_synthetic_run(
    spec: RunSpecification,
    *,
    runs_dir: Path,
    output_species: list[str],
) -> Path:
    """Generate and write one synthetic raw run."""
    final_ymix_output = _synthetic_final_state(spec, output_species=output_species)
    h5_path = runs_dir / f"{spec.run_id}.h5"
    if spec.kzz_cm2_s is None:
        write_equilibrium_hdf5(
            h5_path,
            spec=spec,
            equilibrium_ymix=final_ymix_output,
            state_species=list(spec.metadata["state_species"]),
            output_species=output_species,
        )
    else:
        write_raw_run_hdf5(
            h5_path,
            spec=spec,
            final_ymix_output=final_ymix_output,
            output_species=output_species,
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
    prepared_specs: list[RunSpecification] = []
    for spec in specs:
        prepared_specs.append(
            RunSpecification(
                run_id=spec.run_id,
                pressure_bar=spec.pressure_bar,
                temperature_k=spec.temperature_k,
                kzz_cm2_s=spec.kzz_cm2_s,
                globals=spec.globals,
                spectrum=spec.spectrum,
                metadata={
                    **spec.metadata,
                    "state_species": state_species,
                    "output_species": output_species,
                },
                elemental_abundances_x_h=spec.elemental_abundances_x_h,
                gravity_cm_s2=spec.gravity_cm_s2,
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
    default_preset = dict(config.get("default_science_preset", {}))
    default_physics = dict(default_preset.get("physics_toggles", {}))
    physics = {
        name: bool(spec.globals.get(name, default_physics.get(name, False)))
        for name in PUBLIC_PHYSICS_TOGGLES
    }
    atm_base = next(
        (
            name
            for name in SUPPORTED_ATM_BASES
            if float(spec.globals.get(f"atm_base_{name}", 0.0)) > 0.5
        ),
        str(default_preset.get("atm_base", runtime.get("atm_base", "H2"))),
    )
    assignments = dict(runtime.get("cfg_assignments", {}))
    assignments.update(
        {
            "atom_list": _vulcan_atom_list(config),
            "use_photo": physics["use_photochemistry"],
            "use_ion": physics["use_ion_chemistry"],
            "use_Kzz": physics["use_eddy_diffusion"],
            "use_moldiff": physics["use_molecular_diffusion"],
            "use_vm_mol": physics["use_upwind_molecular_diffusion"],
            "use_topflux": physics["use_boundary_conditions"],
            "use_botflux": physics["use_boundary_conditions"],
            "use_condense": physics["use_condensation"],
            "use_settling": physics["use_settling"],
            "use_ini_cold_trap": physics["use_initial_cold_trap"],
            "use_sat_surfaceH2O": physics["use_sat_surface_h2o"],
            "use_lowT_limit_rates": bool(runtime["use_lowT_limit_rates"]),
            "use_adapt_rtol": bool(runtime["use_adaptive_rtol"]),
            "ini_mix": "EQ",
            "use_solar": False,
            "network": str(runtime["chemistry_file"]),
            "atm_file": str(tp_file.relative_to(cfg_file.parent)),
            "sflux_file": str(spectrum_file.relative_to(cfg_file.parent)),
            "atm_base": atm_base,
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

    # Extract the final converged mixing ratios from VULCAN output.
    if "ymix_time" in _fetch(data, "variable"):
        ymix_time = np.asarray(_fetch(data, "variable", "ymix_time"), dtype=np.float64)
    elif "y_time" in _fetch(data, "variable") and "n_0" in _fetch(data, "atm"):
        y_time = np.asarray(_fetch(data, "variable", "y_time"), dtype=np.float64)
        n0 = np.asarray(_fetch(data, "atm", "n_0"), dtype=np.float64)
        ymix_time = y_time / n0[None, :, None]
    else:
        raise ValueError("VULCAN output is missing both ymix_time and the y_time/n_0 fallback.")
    output_species = list(config["data_spec"]["output_species"])
    output_indices = [species.index(name) for name in output_species]
    # Take the last timestep as the final converged state.
    final_ymix_output = ymix_time[-1, :, :][:, output_indices]
    converted_spec = RunSpecification(
        run_id=spec.run_id,
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        kzz_cm2_s=kzz_cm2_s,
        globals=spec.globals,
        spectrum=spec.spectrum,
        metadata={
            **spec.metadata,
            "state_species": list(config["data_spec"]["state_species"]),
            "output_species": output_species,
        },
        elemental_abundances_x_h=(
            np.asarray(spec.elemental_abundances_x_h, dtype=np.float64)
            if spec.elemental_abundances_x_h is not None
            else None
        ),
        gravity_cm_s2=(
            np.asarray(spec.gravity_cm_s2, dtype=np.float64)
            if spec.gravity_cm_s2 is not None
            else None
        ),
    )
    return write_raw_run_hdf5(
        output_h5_path,
        spec=converted_spec,
        final_ymix_output=final_ymix_output,
        output_species=output_species,
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


def _validated_vulcan_paths(config: dict[str, Any], *, project_root: Path) -> tuple[Path, Path]:
    """Validate the configured VULCAN/FastChem source paths for this run mode."""
    source_root = resolve_path(config["paths"]["vulcan_source_root"], project_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Configured VULCAN source root does not exist: {source_root}")
    if uses_fastchem(config):
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
    fastchem = uses_fastchem(config)
    worker_root_key = "vulcan_runtime" if "vulcan_runtime" in config else None
    worker_base = resolve_path(
        config["vulcan_runtime"]["worker_root"] if worker_root_key else "data/vulcan_workers",
        project_root,
    )
    run_single = _run_single_fastchem_spec if fastchem else _run_single_vulcan_spec
    backfill = config["generation"].get("backfill", {"enabled": True, "max_retries": 3})
    target_count = num_runs or int(config["generation"]["num_runs"])

    def _sample_and_prepare(n: int, seed: int) -> list[RunSpecification]:
        specs = sample_run_specifications(
            config=config,
            project_root=project_root,
            num_runs=n,
            seed=seed,
        )
        prepared: list[RunSpecification] = []
        for spec in specs:
            prepared.append(
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
                    spectrum=spec.spectrum,
                    elemental_abundances_x_h=spec.elemental_abundances_x_h,
                    gravity_cm_s2=spec.gravity_cm_s2,
                )
            )
        return prepared

    def _run_batch(specs: list[RunSpecification]) -> tuple[list[Path], list[str]]:
        """Run a batch of specs, returning successes and failed run IDs."""
        worker_count = _generation_worker_count(config, len(specs))
        successes: list[Path] = []
        failures: list[str] = []
        if worker_count == 1:
            for spec in specs:
                try:
                    successes.append(
                        run_single(
                            spec,
                            source_root=source_root,
                            worker_base=worker_base,
                            runs_dir=runs_dir,
                            config=config,
                        )
                    )
                except Exception:
                    LOGGER.warning("Run %s failed, will backfill", spec.run_id, exc_info=True)
                    failures.append(spec.run_id)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_id = {
                    executor.submit(
                        run_single,
                        spec,
                        source_root=source_root,
                        worker_base=worker_base,
                        runs_dir=runs_dir,
                        config=config,
                    ): spec.run_id
                    for spec in specs
                }
                for future in concurrent.futures.as_completed(future_to_id):
                    run_id = future_to_id[future]
                    try:
                        successes.append(future.result())
                    except Exception:
                        LOGGER.warning("Run %s failed, will backfill", run_id, exc_info=True)
                        failures.append(run_id)
        return successes, failures

    # Initial batch.
    base_seed = int(config["generation"]["seed"])
    prepared_specs = _sample_and_prepare(target_count, base_seed)
    run_files, all_failures = _run_batch(prepared_specs)
    all_specs = list(prepared_specs)

    # Backfill rounds.
    if backfill.get("enabled", True) and all_failures:
        max_retries = int(backfill.get("max_retries", 3))
        for attempt in range(1, max_retries + 1):
            shortfall = target_count - len(run_files)
            if shortfall <= 0:
                break
            LOGGER.info(
                "Backfill attempt %d/%d: %d runs needed",
                attempt, max_retries, shortfall,
            )
            backfill_seed = base_seed + 1000 * attempt
            backfill_specs = _sample_and_prepare(shortfall, backfill_seed)
            new_successes, new_failures = _run_batch(backfill_specs)
            run_files.extend(new_successes)
            all_failures.extend(new_failures)
            all_specs.extend(backfill_specs)
            if not new_failures:
                break

    if all_failures:
        failed_log = raw_root / "failed_runs.json"
        failed_log.write_text(json.dumps(all_failures, indent=2), encoding="utf-8")
        shortfall = target_count - len(run_files)
        if shortfall > 0:
            raise RuntimeError(
                f"Generation finished {shortfall} successful runs short of the requested total. "
                f"See {failed_log} for failed run IDs."
            )

    run_files = sorted(run_files)
    LOGGER.info("VULCAN generation complete: %d runs written to %s", len(run_files), runs_dir)
    consolidated_path = consolidate_runs_to_single_hdf5(
        run_files, raw_root / "runs.h5",
    )
    successful_specs = [s for s in all_specs if s.run_id not in set(all_failures)]
    manifest_path, coverage_path = _write_generation_metadata(
        raw_root=raw_root,
        run_files=[consolidated_path],
        specs=successful_specs,
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
