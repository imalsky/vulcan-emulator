"""Raw dataset generation: FastChem/VULCAN orchestration and final-state writing.

This module is the first stage of the data pipeline.  It samples
atmospheric parameters (via ``sampling.sample_run_specifications``),
then either:

* **fastchem chemistry** — calls FastChem to compute chemical equilibrium
  at each pressure level and writes a simplified HDF5 per run, or
* **vulcan chemistry** — patches the VULCAN configuration, launches
  VULCAN as a subprocess, and writes the final converged output state.

All generated HDF5 files follow a shared layout (see ``spec.md``,
"Shared Data Contract") and are consumed downstream by
``preprocess.py``.  A generation manifest and sampling-coverage
summary are persisted alongside the runs for provenance.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
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
    dataset_info_root,
    dataset_raw_root,
    dataset_run_root,
    get_chemistry_type,
    get_model_type,
    uses_fastchem,
)
from ..constants import ELEMENT_INPUT_ORDER, PUBLIC_PHYSICS_TOGGLES, SOLAR_ABUNDANCES, SUPPORTED_ATM_BASES
from ..utils.helpers import ensure_dir, get_logger, resolve_path
from ..utils.provenance import manifest_for_files
from .sampling import RunSpecification, sample_run_specifications
from .spectrum import write_vulcan_spectrum_txt

LOGGER = get_logger(__name__)
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
    run_root: Path
    raw_root: Path
    run_ids: list[str]
    consolidated_path: Path
    manifest_path: Path | None = None
    coverage_path: Path | None = None


def patch_python_assignments(text: str, assignments: dict[str, Any]) -> str:
    """Patch simple ``name = value`` assignments in a Python config file.

    Parameters
    ----------
    text : str
        Original Python source text to patch.
    assignments : dict[str, Any]
        Mapping from variable names to replacement Python values. Each value is
        serialized with ``repr`` and applied to the first matching top-level
        ``name = ...`` assignment.

    Returns
    -------
    str
        Updated Python source text with all requested assignments replaced in
        place, or appended at the end when a name is not already present.
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
    """Return the number of raw runs the generation stage should produce.

    Parameters
    ----------
    config : dict[str, Any]
        Validated pipeline config containing ``generation.num_runs``.
    num_runs : int or None
        Optional command-line override for the configured run count.

    Returns
    -------
    int
        Requested number of generated runs after applying the override when
        provided.
    """
    return int(config["generation"]["num_runs"] if num_runs is None else num_runs)


def consolidate_runs_to_single_hdf5(
    run_files: list[Path],
    output_path: Path,
    *,
    delete_originals: bool = True,
) -> Path:
    """Merge per-run HDF5 files into a single file with one group per run.

    Parameters
    ----------
    run_files : list[Path]
        Per-run HDF5 files to merge from a temporary staging directory.
    output_path : Path
        Destination consolidated HDF5 file, usually ``runs.h5``.
    delete_originals : bool, default=True
        When ``True``, remove the original per-run files after the consolidated
        file has been written successfully.

    Returns
    -------
    Path
        Path to the consolidated HDF5 file where each original run is stored
        as a top-level group named after the source file stem.
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
    """List run identifiers stored inside a consolidated raw-data HDF5 file.

    Parameters
    ----------
    consolidated_path : Path
        Path to ``runs.h5`` where each top-level group corresponds to one run.

    Returns
    -------
    list[str]
        Sorted run IDs such as ``["run_00000", "run_00001", ...]``.
    """
    with h5py.File(consolidated_path, "r") as f:
        return sorted(f.keys())


def _detect_available_cpus() -> int:
    """Return the number of CPUs available to this process.

    Checks PBS (``NCPUS``), SLURM (``SLURM_CPUS_ON_NODE``), and
    ``os.cpu_count()`` in that order.
    """
    for env_var in ("NCPUS", "SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK"):
        value = os.environ.get(env_var)
        if value is not None:
            try:
                n = int(value)
                if n >= 1:
                    return n
            except ValueError:
                pass
    return os.cpu_count() or 1


def _generation_worker_count(config: dict[str, Any], total_runs: int) -> int:
    """Return the effective number of parallel generation workers to launch.

    Parameters
    ----------
    config : dict[str, Any]
        Validated config containing ``generation.parallel_workers``.
        A value of ``0`` means *auto-detect* from the runtime environment
        (PBS ``NCPUS``, SLURM, or ``os.cpu_count()``).
    total_runs : int
        Number of runs that still need to be generated.

    Returns
    -------
    int
        Worker count capped to at least one worker and at most ``total_runs``.
    """
    configured = int(config["generation"]["parallel_workers"])
    if configured <= 0:
        configured = _detect_available_cpus()
    return max(1, min(configured, total_runs))


def _prepare_generation_directory(
    config: dict[str, Any],
    *,
    project_root: Path,
    num_runs: int | None,
) -> tuple[Path, Path, Path, Path | None, Path | None]:
    """Prepare the raw-data output directory and resolve reuse semantics.

    Parameters
    ----------
    config : dict[str, Any]
        Validated generation config containing raw-data paths and overwrite /
        reuse settings.
    project_root : Path
        Repository root used to resolve config-relative paths.
    num_runs : int or None
        Optional override for the requested run count.

    Returns
    -------
    tuple[Path, Path, Path, Path | None, Path | None]
        Run root, raw-data root, metadata root, per-run staging directory
        (``raw/runs/``) for new generation, and either ``None`` when new
        generation should proceed or the reusable consolidated ``runs.h5``
        path when reuse is allowed.  Per-run files are written under
        ``raw/runs/`` so they survive a crash and can be resumed.
    """
    run_root = resolve_path(dataset_run_root(config), project_root)
    raw_root = resolve_path(dataset_raw_root(config), project_root)
    info_root = resolve_path(dataset_info_root(config), project_root)
    ensure_dir(run_root)
    ensure_dir(raw_root)
    ensure_dir(info_root)
    requested_runs = _requested_run_count(config, num_runs)
    consolidated_path = raw_root / "runs.h5"

    # Per-run HDF5 staging lives under raw_root so completed runs survive
    # a crash and can be resumed on the next invocation.
    runs_dir = raw_root / "runs"

    # Count runs inside a consolidated file, if present.
    consolidated_count = 0
    if consolidated_path.exists():
        consolidated_count = len(list_run_ids_from_consolidated(consolidated_path))
    if bool(config["generation"]["overwrite"]):
        if consolidated_path.exists():
            consolidated_path.unlink()
        if runs_dir.exists():
            shutil.rmtree(runs_dir)
        for metadata_path in (
            info_root / "generation_manifest.json",
            info_root / "sampling_coverage.json",
            info_root / "failed_runs.json",
        ):
            if metadata_path.exists():
                metadata_path.unlink()
        ensure_dir(runs_dir)
        return run_root, raw_root, info_root, runs_dir, None
    # Reuse only the canonical consolidated raw-data layout.
    if consolidated_count > 0:
        if bool(config["generation"]["reuse_raw_if_present"]) and consolidated_count == requested_runs:
            manifest_path = info_root / "generation_manifest.json"
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
            return run_root, raw_root, info_root, None, consolidated_path
        raise RuntimeError(
            f"Found {consolidated_count} existing raw runs. "
            "Set generation.overwrite=true or align generation.num_runs with the existing dataset."
        )
    ensure_dir(runs_dir)
    return run_root, raw_root, info_root, runs_dir, None


def _coverage_fraction(low: float, high: float, observed_low: float, observed_high: float) -> float:
    """Compute how much of a configured interval is covered by sampled values.

    Parameters
    ----------
    low : float
        Configured lower bound of the allowed parameter range.
    high : float
        Configured upper bound of the allowed parameter range.
    observed_low : float
        Minimum sampled value observed in the generated dataset.
    observed_high : float
        Maximum sampled value observed in the generated dataset.

    Returns
    -------
    float
        Fraction of the configured interval covered by the observed samples,
        clipped to ``[0, 1]``.
    """
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

    Parameters
    ----------
    config : dict[str, Any]
        Validated generation config containing the configured sampling ranges.
    specs : list[RunSpecification]
        Sampled run specifications used to generate the raw dataset.
    run_files : list[Path]
        Raw run files or consolidated files whose stored pressure and
        temperature profiles define the realized coverage.
    mode : str
        Generation mode label recorded in the output payload.

    Returns
    -------
    dict[str, Any]
        JSON-serializable diagnostic payload describing configured ranges,
        realized min/max values, and coverage fractions for key sampled
        parameters.
    """
    fastchem = uses_fastchem(config)
    frac_keys = ("He_H", "C_H", "O_H", "N_H", "S_H")
    frac_arrays = {
        key: np.asarray([spec.globals[key] for spec in specs], dtype=np.float64)
        for key in frac_keys
    }
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
    _element_to_config_key = {
        "He_H": "he_frac_range",
        "C_H": "c_frac_range",
        "O_H": "o_frac_range",
        "N_H": "n_frac_range",
        "S_H": "s_frac_range",
    }
    frac_ranges = {
        key: [float(x) for x in config["sampling"][_element_to_config_key[key]]]
        for key in frac_keys
    }
    temperature_range = [float(x) for x in config["sampling"]["temperature_range_k"]]

    configured_ranges: dict[str, Any] = {
        **{key: frac_ranges[key] for key in frac_keys},
        "temperature_k": temperature_range,
        "pressure_bar": [
            float(config["sampling"]["pressure_top_bar"]),
            float(config["sampling"]["pressure_bottom_bar"]),
        ],
    }
    realized: dict[str, Any] = {
        key: {
            "min": float(np.min(frac_arrays[key])),
            "max": float(np.max(frac_arrays[key])),
            "coverage_fraction": _coverage_fraction(
                *frac_ranges[key],
                float(np.min(frac_arrays[key])),
                float(np.max(frac_arrays[key])),
            ),
        }
        for key in frac_keys
    }
    realized.update({
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
    })

    if not fastchem:
        gravity = np.asarray([spec.globals["gravity_cm_s2"] for spec in specs], dtype=np.float64)
        planet_radius = np.asarray(
            [spec.globals["planet_radius_cm"] for spec in specs],
            dtype=np.float64,
        )
        log10_kzz_rows: list[np.ndarray] = []
        for path in run_files:
            with h5py.File(path, "r") as handle:
                sources = [handle] if "inputs" in handle else [handle[run_id] for run_id in sorted(handle.keys())]
                for source in sources:
                    log10_kzz_rows.append(np.log10(np.asarray(source["inputs/kzz_cm2_s"], dtype=np.float64)))
        log10_kzz = np.concatenate(log10_kzz_rows, axis=0)
        gravity_range = [float(x) for x in config["sampling"]["gravity_range_cm_s2"]]
        planet_radius_range = [float(x) for x in config["sampling"]["planet_radius_range_cm"]]
        kzz_value = float(config["sampling"]["kzz_cm2_s"])
        log10_kzz_value = float(np.log10(max(kzz_value, 1.0e-30)))
        kzz_range = [log10_kzz_value, log10_kzz_value]
        configured_ranges.update({
            "gravity_cm_s2": gravity_range,
            "planet_radius_cm": planet_radius_range,
            "log10_kzz_cm2_s": kzz_range,
        })
        realized.update({
            "gravity_cm_s2": {
                "min": float(np.min(gravity)),
                "max": float(np.max(gravity)),
                "coverage_fraction": _coverage_fraction(*gravity_range, float(np.min(gravity)), float(np.max(gravity))),
            },
            "planet_radius_cm": {
                "min": float(np.min(planet_radius)),
                "max": float(np.max(planet_radius)),
                "coverage_fraction": _coverage_fraction(
                    *planet_radius_range,
                    float(np.min(planet_radius)),
                    float(np.max(planet_radius)),
                ),
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
    info_root: Path,
    run_files: list[Path],
    specs: list[RunSpecification],
    config: dict[str, Any],
    mode: str,
) -> tuple[Path, Path]:
    """Persist raw-run provenance and sampling-coverage metadata.

    Parameters
    ----------
    info_root : Path
        Shared metadata directory that receives the generation metadata files.
    run_files : list[Path]
        Raw HDF5 artefacts included in the manifest.
    specs : list[RunSpecification]
        Successfully generated run specifications used to summarize coverage.
    config : dict[str, Any]
        Validated pipeline config.
    mode : str
        Generation mode label such as ``"synthetic"`` or ``"vulcan"``.

    Returns
    -------
    tuple[Path, Path]
        Paths to ``generation_manifest.json`` and
        ``sampling_coverage.json``.
    """
    manifest_path = info_root / "generation_manifest.json"
    coverage_path = info_root / "sampling_coverage.json"
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
    """Return whether the configured species lists require sulfur chemistry.

    Parameters
    ----------
    config : dict[str, Any]
        Validated config whose ``data_spec`` section defines the state and
        output species names.

    Returns
    -------
    bool
        ``True`` when any requested species name contains ``"S"`` and the
        downstream chemistry runtime should include sulfur-bearing elements.
    """
    species = list(config["data_spec"]["state_species"]) + list(config["data_spec"]["output_species"])
    return any("S" in name for name in species)


def _vulcan_atom_list(config: dict[str, Any]) -> list[str]:
    """Build the elemental basis expected by the VULCAN and FastChem runtimes.

    Parameters
    ----------
    config : dict[str, Any]
        Validated config describing the requested species basis.

    Returns
    -------
    list[str]
        Ordered atom symbols passed into the chemistry runtime, including
        sulfur when the configured species set requires it.
    """
    atoms = ["H", "O", "C", "N", "He"]
    if _sulfur_enabled(config):
        atoms.append("S")
    return atoms


def _element_abundances_from_spec(spec: RunSpecification) -> dict[str, float]:
    """Extract elemental number fractions from a run specification.

    The run specification carries elemental *number fractions* (summing with
    hydrogen to 1).  This function validates the fractions and computes the
    auxiliary ``fastchem_met_scale`` scalar.

    Parameters
    ----------
    spec : RunSpecification
        Sampled run specification containing either an explicit per-level
        elemental fraction profile or scalar fraction globals.

    Returns
    -------
    dict[str, float]
        Mapping containing number fractions ``He_H``, ``C_H``,
        ``O_H``, ``N_H``, ``S_H``, plus the auxiliary
        ``fastchem_met_scale`` scalar used by the FastChem runtime.
    """
    if spec.elemental_abundances_frac is not None:
        frac_profile = np.asarray(spec.elemental_abundances_frac, dtype=np.float64)
        if frac_profile.ndim != 2 or frac_profile.shape[1] != len(ELEMENT_INPUT_ORDER):
            raise ValueError(
                "RunSpecification.elemental_abundances_frac must have shape "
                f"(nz, {len(ELEMENT_INPUT_ORDER)})."
            )
        frac_vector = frac_profile[0]
        fractions = {
            name: float(frac_vector[idx])
            for idx, name in enumerate(ELEMENT_INPUT_ORDER)
        }
    else:
        fractions = {name: float(spec.globals[name]) for name in ELEMENT_INPUT_ORDER}

    h_frac = 1.0 - sum(fractions.values())
    if h_frac <= 0.0:
        raise ValueError(
            f"Elemental fractions sum to >= 1.0 (H_frac={h_frac:.6f}); "
            "hydrogen remainder is non-positive."
        )
    return {
        **fractions,
        "fastchem_met_scale": float(
            fractions["O_H"] / SOLAR_ABUNDANCES["O_H"]
        ),
    }


def _element_profile_from_spec(spec: RunSpecification) -> np.ndarray:
    """Return the per-level elemental number-fraction profile for one run.

    Parameters
    ----------
    spec : RunSpecification
        Sampled run specification containing either a precomputed elemental
        fraction profile or scalar fraction globals.

    Returns
    -------
    np.ndarray
        Elemental fraction profile with shape
        ``(nz, len(ELEMENT_INPUT_ORDER))`` ordered by ``ELEMENT_INPUT_ORDER``.
    """
    if spec.elemental_abundances_frac is not None:
        profile = np.asarray(spec.elemental_abundances_frac, dtype=np.float64)
        if profile.ndim != 2 or profile.shape[1] != len(ELEMENT_INPUT_ORDER):
            raise ValueError(
                "RunSpecification.elemental_abundances_frac must have shape "
                f"(nz, {len(ELEMENT_INPUT_ORDER)})."
            )
        return profile
    fractions = {name: float(spec.globals[name]) for name in ELEMENT_INPUT_ORDER}
    vector = np.array([fractions[name] for name in ELEMENT_INPUT_ORDER], dtype=np.float64)
    return np.repeat(vector[None, :], int(spec.pressure_bar.size), axis=0)


def _gravity_profile_from_spec(spec: RunSpecification) -> np.ndarray:
    """Return the per-level gravity profile for one run.

    Parameters
    ----------
    spec : RunSpecification
        Sampled run specification containing either a gravity profile or a
        profile-global gravity scalar.

    Returns
    -------
    np.ndarray
        Gravity profile with shape ``(nz,)`` in ``cm s^-2``.
    """
    if spec.gravity_cm_s2 is not None:
        profile = np.asarray(spec.gravity_cm_s2, dtype=np.float64)
        if profile.ndim != 1:
            raise ValueError("RunSpecification.gravity_cm_s2 must be a 1-D array.")
        return profile
    gravity_value = float(spec.globals.get("gravity_cm_s2", 0.0))
    return np.full(int(spec.pressure_bar.size), gravity_value, dtype=np.float64)


def _write_tp_profile(path: Path, spec: RunSpecification) -> Path:
    """Write the VULCAN TPK profile text file for one run.

    Parameters
    ----------
    path : Path
        Destination text file.
    spec : RunSpecification
        Sampled run specification providing pressure, temperature, and ``Kzz``
        profiles.

    Returns
    -------
    Path
        Path to the written profile file.
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Pressure(bar) Temperature(K) Kzz(cm2/s)\n")
        handle.write("Pressure Temp Kzz\n")
        for p, t, kzz in zip(spec.pressure_bar, spec.temperature_k, spec.kzz_cm2_s):
            handle.write(f"{p:.8e} {t:.8f} {kzz:.8e}\n")
    return path


def _write_scalar_metadata(handle: h5py.File, metadata: dict[str, Any]) -> None:
    """Persist scalar provenance fields under the HDF5 ``metadata`` group.

    Parameters
    ----------
    handle : h5py.File
        Open output HDF5 file that will receive a ``metadata`` group.
    metadata : dict[str, Any]
        Flat metadata mapping. Only scalar booleans, numbers, and strings are
        serialized.

    Returns
    -------
    None
        Scalar metadata values are written into ``handle`` under a
        ``metadata`` group when any serializable entries are present.
    """
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

    Parameters
    ----------
    path : Path
        Destination raw-run HDF5 path.
    spec : RunSpecification
        Sampled run specification providing pressure, temperature, elemental
        abundances, gravity, globals, and metadata.
    equilibrium_ymix : np.ndarray
        Equilibrium mixing-ratio tensor with shape ``(nz, n_output_species)``.
    state_species : list[str]
        Species labels recorded for the state basis.
    output_species : list[str]
        Species labels corresponding to the columns of ``equilibrium_ymix``.

    Returns
    -------
    Path
        Path to the written equilibrium raw-run file.

    Layout::

        inputs/pressure_bar      (nz,)
        inputs/temperature_k     (nz,)
        inputs/element_input_order (n_elements,) string
        inputs/elemental_abundances_frac (nz, n_elements)
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
        inputs.create_dataset("elemental_abundances_frac", data=element_profile)
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
    """Write one VULCAN raw run in the repository HDF5 contract.

    Parameters
    ----------
    path : Path
        Destination raw-run HDF5 path.
    spec : RunSpecification
        Sampled run specification providing per-level inputs, globals, and
        optional spectrum data.
    final_ymix_output : np.ndarray
        Final converged composition tensor with shape ``(nz, n_output_species)``.
    output_species : list[str] or None, optional
        Output species order for ``final_ymix_output``. When omitted, the value
        is inferred from ``spec.metadata``.

    Returns
    -------
    Path
        Path to the written raw-run HDF5 file.

    Layout::

        inputs/pressure_bar          (nz,)
        inputs/temperature_k         (nz,)
        inputs/kzz_cm2_s             (nz,)
        inputs/element_input_order   (n_elements,) string
        inputs/elemental_abundances_frac (nz, n_elements)
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
        inputs.create_dataset("elemental_abundances_frac", data=element_profile)
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


def _copy_vulcan_source(source_root: Path, worker_root: Path) -> None:
    """Refresh one worker-local VULCAN checkout from the shared source tree.

    Parameters
    ----------
    source_root : Path
        Root of the canonical VULCAN source tree in the project workspace.
    worker_root : Path
        Worker-local destination directory that should receive a clean copy of
        the source tree.

    Returns
    -------
    None
        The function replaces any existing ``worker_root`` tree and copies the
        full source directory into place.
    """
    if worker_root.exists():
        shutil.rmtree(worker_root)
    shutil.copytree(source_root, worker_root)


def _cleanup_worker_root(worker_root: Path) -> None:
    """Remove one worker-local runtime tree after a run completes.

    Parameters
    ----------
    worker_root : Path
        Worker-local directory to delete.

    Returns
    -------
    None
        The worker directory is removed when it exists.
    """
    shutil.rmtree(worker_root, ignore_errors=True)


def _copy_fastchem_runtime(source_root: Path, worker_root: Path) -> Path:
    """Copy the minimal FastChem runtime needed for one worker run.

    Parameters
    ----------
    source_root : Path
        Root of the checked-out VULCAN/FastChem source tree.
    worker_root : Path
        Worker-local directory that will receive the copied runtime.

    Returns
    -------
    Path
        Path to the worker-local ``fastchem_vulcan`` directory.
    """
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
    """Write worker-local TP and stellar-spectrum inputs for one run.

    Parameters
    ----------
    worker_root : Path
        Worker-local VULCAN checkout directory.
    spec : RunSpecification
        Run specification providing the atmospheric profile and stellar
        spectrum.

    Returns
    -------
    tuple[Path, Path]
        Paths to the written TP profile file and stellar-spectrum file.
    """
    atm_dir = ensure_dir(worker_root / "atm")
    stellar_dir = ensure_dir(atm_dir / "stellar_flux")
    tp_file = _write_tp_profile(atm_dir / f"{spec.run_id}_tp.txt", spec)
    spectrum_file = write_vulcan_spectrum_txt(
        spec.spectrum,
        stellar_dir / f"{spec.spectrum.name}.txt",
    )
    return tp_file, spectrum_file


def _write_fastchem_tp_profile(fastchem_root: Path, spec: RunSpecification) -> Path:
    """Write the FastChem pressure-temperature input profile.

    Parameters
    ----------
    fastchem_root : Path
        Worker-local FastChem runtime directory.
    spec : RunSpecification
        Run specification providing pressure and temperature profiles.

    Returns
    -------
    Path
        Path to the written ``vulcan_TP.dat`` file.
    """
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
    """Write the FastChem elemental-abundance input file for one run.

    Parameters
    ----------
    fastchem_root : Path
        Worker-local FastChem runtime directory.
    spec : RunSpecification
        Run specification supplying metallicity and elemental-abundance
        conditioning values.
    config : dict[str, Any]
        Validated config used to choose the ion / non-ion parameter template
        and atom list.

    Returns
    -------
    Path
        Path to the written ``element_abundances_vulcan.dat`` file.
    """
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
                key = "He_H" if species_name == "He" else f"{species_name}_H"
                number_frac = element_abundances.get(key)
                if number_frac is None:
                    raise ValueError(f"Missing elemental abundance for {species_name} in FastChem setup.")
                output_lines.append(f"{species_name}\t{12.0 + np.log10(float(number_frac)):.4f}\n")
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
    """Patch a worker-local ``vulcan_cfg.py`` with run-specific inputs.

    Parameters
    ----------
    cfg_file : Path
        Worker-local config file to update in place.
    spec : RunSpecification
        Run specification supplying the sampled atmospheric inputs and global
        conditioning values.
    config : dict[str, Any]
        Validated pipeline config containing runtime defaults and spectrum
        settings.
    tp_file : Path
        Worker-local TP profile consumed by VULCAN.
    spectrum_file : Path
        Worker-local stellar-spectrum file consumed by VULCAN.

    Returns
    -------
    None
        ``cfg_file`` is updated in place with run-specific profile, spectrum,
        elemental abundance, and runtime toggle assignments.
    """
    text = cfg_file.read_text(encoding="utf-8")
    runtime = config["vulcan_runtime"]
    element_abundances = _element_abundances_from_spec(spec)
    default_preset = dict(config.get("default_science_preset", {}))
    default_physics = dict(default_preset.get("physics_toggles", {}))
    runtime_top_bc_flux_file = runtime.get("top_bc_flux_file")
    runtime_bot_bc_flux_file = runtime.get("bot_bc_flux_file")
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
            "Rp": float(spec.globals["planet_radius_cm"]),
            "rocky": bool(runtime["rocky"]),
            "r_star": float(spec.globals["r_star_rsun"]),
            "orbit_radius": float(spec.globals["semi_major_axis_au"]),
            "sl_angle": float(np.deg2rad(spec.globals["zenith_angle_deg"])),
            "f_diurnal": float(spec.globals["diurnal_factor"]),
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
    if runtime_top_bc_flux_file is not None:
        assignments["top_BC_flux_file"] = str(runtime_top_bc_flux_file)
    if runtime_bot_bc_flux_file is not None:
        assignments["bot_BC_flux_file"] = str(runtime_bot_bc_flux_file)
    cfg_file.write_text(patch_python_assignments(text, assignments), encoding="utf-8")


def _trusted_unpickle(path: Path) -> Any:
    """Load a trusted pickle payload emitted by the local chemistry runtime.

    Parameters
    ----------
    path : Path
        Filesystem path to a pickle file produced by the local VULCAN runtime.

    Returns
    -------
    Any
        Deserialized Python object stored in the pickle payload.
    """
    with path.open("rb") as handle:
        return pickle.load(handle)


def _fetch(container: Any, *keys: str) -> Any:
    """Traverse nested dict/object containers using a shared key path.

    Parameters
    ----------
    container : Any
        Root object or mapping.
    *keys : str
        Attribute or dict keys to follow in sequence.

    Returns
    -------
    Any
        Nested value located at the supplied path.
    """
    current = container
    for key in keys:
        if isinstance(current, dict):
            current = current[key]
        else:
            current = getattr(current, key)
    return current


def _decode_species_list(values: Any) -> list[str]:
    """Normalize stored species labels to plain Python strings.

    Parameters
    ----------
    values : Any
        Iterable of bytes or string-like species labels.

    Returns
    -------
    list[str]
        Species labels converted to Python ``str`` objects.
    """
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
    """Convert one VULCAN pickle output into the shared raw HDF5 contract.

    Parameters
    ----------
    vulcan_output_path : Path
        Pickled VULCAN output file produced by one worker run.
    output_h5_path : Path
        Destination raw-run HDF5 file.
    spec : RunSpecification
        Original sampled run specification used to restore external metadata.
    config : dict[str, Any]
        Validated config providing state and output species orderings.

    Returns
    -------
    Path
        Path to the written raw-run HDF5 file.
    """
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
    gravity_profile_runtime = None
    try:
        gravity_profile_runtime = np.asarray(_fetch(data, "atm", "g"), dtype=np.float64)
    except (AttributeError, KeyError):
        gravity_profile_runtime = None

    # Extract the final converged mixing ratios from VULCAN output.
    variable = _fetch(data, "variable")
    if "ymix_time" not in variable:
        raise ValueError(
            "VULCAN output is missing variable.ymix_time "
            "required to derive the final raw-data contract."
        )
    ymix_time = np.asarray(_fetch(data, "variable", "ymix_time"), dtype=np.float64)
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
        elemental_abundances_frac=(
            np.asarray(spec.elemental_abundances_frac, dtype=np.float64)
            if spec.elemental_abundances_frac is not None
            else None
        ),
        gravity_cm_s2=(
            gravity_profile_runtime
            if gravity_profile_runtime is not None
            and gravity_profile_runtime.ndim == 1
            and gravity_profile_runtime.shape == pressure_bar.shape
            and np.all(np.isfinite(gravity_profile_runtime))
            else (
                np.asarray(spec.gravity_cm_s2, dtype=np.float64)
                if spec.gravity_cm_s2 is not None
                else None
            )
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
    """Convert one FastChem equilibrium table into the raw HDF5 contract.

    Parameters
    ----------
    fastchem_output_path : Path
        FastChem text output containing per-level equilibrium abundances.
    output_h5_path : Path
        Destination raw-run HDF5 file.
    spec : RunSpecification
        Original sampled run specification.
    config : dict[str, Any]
        Validated config defining the requested state and output species.

    Returns
    -------
    Path
        Path to the written equilibrium raw-run HDF5 file.
    """
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
    """Validate external runtime paths for the active chemistry backend.

    Parameters
    ----------
    config : dict[str, Any]
        Validated config containing ``paths.vulcan_source_root`` and runtime
        settings.
    project_root : Path
        Repository root used to resolve relative paths.

    Returns
    -------
    tuple[Path, Path]
        Source-root path plus the backend-specific key path: either the
        FastChem binary or the configured VULCAN chemistry file.
    """
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
    """Execute one worker-local VULCAN run and convert its output to HDF5.

    Parameters
    ----------
    spec : RunSpecification
        Sampled run specification to execute.
    source_root : Path
        Root of the shared VULCAN source tree.
    worker_base : Path
        Parent directory for worker-local runtime copies.
    runs_dir : Path
        Directory receiving the converted raw HDF5 file.
    config : dict[str, Any]
        Validated config containing runtime settings.

    Returns
    -------
    Path
        Path to the written raw-run HDF5 file.
    """
    worker_root = worker_base / spec.run_id
    try:
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
            raise FileNotFoundError(
                f"No VULCAN output file found for run {spec.run_id} under {worker_root / 'output'}."
            )
        output_h5 = runs_dir / f"{spec.run_id}.h5"
        return convert_vulcan_output_to_hdf5(
            output_candidates[-1],
            output_h5_path=output_h5,
            spec=spec,
            config=config,
        )
    finally:
        _cleanup_worker_root(worker_root)


def _run_single_fastchem_spec(
    spec: RunSpecification,
    *,
    source_root: Path,
    worker_base: Path,
    runs_dir: Path,
    config: dict[str, Any],
) -> Path:
    """Execute one worker-local FastChem run and convert its output to HDF5.

    Parameters
    ----------
    spec : RunSpecification
        Sampled run specification to execute.
    source_root : Path
        Root of the shared VULCAN/FastChem source tree.
    worker_base : Path
        Parent directory for worker-local runtime copies.
    runs_dir : Path
        Directory receiving the converted raw HDF5 file.
    config : dict[str, Any]
        Validated config containing runtime settings.

    Returns
    -------
    Path
        Path to the written equilibrium raw-run HDF5 file.
    """
    worker_root = worker_base / spec.run_id
    try:
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
            raise FileNotFoundError(
                f"No FastChem equilibrium output found for run {spec.run_id} under {fastchem_output}."
            )
        output_h5 = runs_dir / f"{spec.run_id}.h5"
        return convert_fastchem_output_to_hdf5(
            fastchem_output,
            output_h5_path=output_h5,
            spec=spec,
            config=config,
        )
    finally:
        _cleanup_worker_root(worker_root)


def run_vulcan_generation(
    config: dict[str, Any],
    *,
    project_root: Path,
    num_runs: int | None = None,
) -> GeneratedRawDataset:
    """Generate raw data by running the external VULCAN or FastChem backend.

    Parameters
    ----------
    config : dict[str, Any]
        Validated generation config.
    project_root : Path
        Repository root used to resolve runtime and data paths.
    num_runs : int or None, optional
        Optional override for ``generation.num_runs``.

    Returns
    -------
    GeneratedRawDataset
        Paths describing the generated raw dataset and its provenance files.
    """
    LOGGER.info("VULCAN generation starting (num_runs=%s)", num_runs or "config default")
    run_root, raw_root, info_root, runs_dir, reusable_path = _prepare_generation_directory(
        config,
        project_root=project_root,
        num_runs=num_runs,
    )
    if reusable_path is not None:
        run_ids = list_run_ids_from_consolidated(reusable_path)
        LOGGER.info("Reusing %d existing runs from %s", len(run_ids), raw_root)
        manifest_path = info_root / "generation_manifest.json"
        coverage_path = info_root / "sampling_coverage.json"
        return GeneratedRawDataset(
            run_root=run_root,
            raw_root=raw_root,
            run_ids=run_ids,
            consolidated_path=reusable_path,
            manifest_path=manifest_path if manifest_path.exists() else None,
            coverage_path=coverage_path if coverage_path.exists() else None,
        )
    if runs_dir is None:
        raise RuntimeError("VULCAN generation requires a writable staging directory.")
    source_root, _ = _validated_vulcan_paths(config, project_root=project_root)
    fastchem = uses_fastchem(config)
    worker_root_key = "vulcan_runtime" if "vulcan_runtime" in config else None
    worker_base = resolve_path(
        config["vulcan_runtime"]["worker_root"]
        if worker_root_key
        else "data/vulcan_workers",
        project_root,
    )
    run_single = _run_single_fastchem_spec if fastchem else _run_single_vulcan_spec
    backfill = config["generation"].get("backfill", {"enabled": True, "max_retries": 3})
    target_count = num_runs or int(config["generation"]["num_runs"])

    def _sample_and_prepare(n: int, seed: int, *, start_index: int) -> list[RunSpecification]:
        """Sample run specs and attach the current processed species contract.

        Parameters
        ----------
        n : int
            Number of specifications to sample.
        seed : int
            Sampling seed for the batch.
        start_index : int
            Run-index offset used to assign deterministic ``run_XXXXX`` IDs.

        Returns
        -------
        list[RunSpecification]
            Prepared run specifications with output-species metadata attached.
        """
        specs = sample_run_specifications(
            config=config,
            project_root=project_root,
            num_runs=n,
            seed=seed,
        )
        prepared: list[RunSpecification] = []
        for index_offset, spec in enumerate(specs):
            run_id = f"run_{start_index + index_offset:05d}"
            prepared.append(
                RunSpecification(
                    run_id=run_id,
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
                    elemental_abundances_frac=spec.elemental_abundances_frac,
                    gravity_cm_s2=spec.gravity_cm_s2,
                )
            )
        return prepared

    def _run_batch(specs: list[RunSpecification]) -> tuple[list[Path], list[str]]:
        """Execute one batch of run specifications through the active backend.

        Parameters
        ----------
        specs : list[RunSpecification]
            Run specifications to execute.

        Returns
        -------
        tuple[list[Path], list[str]]
            Successfully written raw-run files and failed run IDs that may be
            backfilled later.
        """
        worker_count = _generation_worker_count(config, len(specs))
        LOGGER.info("Running %d specs with %d parallel workers", len(specs), worker_count)
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

    # Resume: detect per-run HDF5 files that survived from a previous
    # interrupted run so we don't redo work.
    existing_run_files = sorted(runs_dir.glob("run_*.h5"))
    if existing_run_files:
        LOGGER.info(
            "Resuming: found %d completed per-run files from a previous run in %s",
            len(existing_run_files),
            runs_dir,
        )

    # Initial batch.
    base_seed = int(config["generation"]["seed"])
    next_run_index = 0
    prepared_specs = _sample_and_prepare(target_count, base_seed, start_index=next_run_index)
    next_run_index += len(prepared_specs)

    # Skip specs whose per-run HDF5 already exists.
    existing_stems = {p.stem for p in existing_run_files}
    remaining_specs = [s for s in prepared_specs if s.run_id not in existing_stems]
    skipped = len(prepared_specs) - len(remaining_specs)
    if skipped:
        LOGGER.info("Skipping %d already-completed runs", skipped)

    run_files: list[Path] = list(existing_run_files)
    all_failures: list[str] = []
    all_specs = list(prepared_specs)

    if remaining_specs:
        new_successes, new_failures = _run_batch(remaining_specs)
        run_files.extend(new_successes)
        all_failures.extend(new_failures)

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
            backfill_specs = _sample_and_prepare(
                shortfall,
                backfill_seed,
                start_index=next_run_index,
            )
            next_run_index += len(backfill_specs)
            new_successes, new_failures = _run_batch(backfill_specs)
            run_files.extend(new_successes)
            all_failures.extend(new_failures)
            all_specs.extend(backfill_specs)
            if not new_failures:
                break

    if all_failures:
        failed_log = info_root / "failed_runs.json"
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
    # Per-run files are cleaned up by consolidate_runs_to_single_hdf5
    # (delete_originals=True by default).  Remove the now-empty staging
    # directory only after consolidation succeeds.
    if runs_dir.exists() and not any(runs_dir.iterdir()):
        runs_dir.rmdir()
    failed_ids = set(all_failures)
    successful_specs = [s for s in all_specs if s.run_id not in failed_ids]
    manifest_path, coverage_path = _write_generation_metadata(
        info_root=info_root,
        run_files=[consolidated_path],
        specs=successful_specs,
        config=config,
        mode="vulcan",
    )
    return GeneratedRawDataset(
        run_root=run_root,
        raw_root=raw_root,
        run_ids=[spec.run_id for spec in successful_specs],
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
    """Dispatch raw-data generation to the configured backend.

    Parameters
    ----------
    config : dict[str, Any]
        Validated generation config.
    project_root : Path
        Repository root used to resolve backend paths.
    num_runs : int or None, optional
        Optional override for the configured run count.

    Returns
    -------
    GeneratedRawDataset
        Description of the generated or reused raw dataset.
    """
    mode = str(config["generation"]["mode"]).lower()
    if mode == "vulcan":
        return run_vulcan_generation(config, project_root=project_root, num_runs=num_runs)
    raise ValueError(f"Unsupported generation mode: {mode}")
