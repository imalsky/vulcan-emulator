"""Test-only synthetic data generation.

Generates heuristic mock final-state compositions without calling VULCAN or
FastChem.  Useful for pipeline smoke tests and integration tests that need
structurally valid HDF5 datasets but do not require physically meaningful
chemistry.

Extracted from ``src.data_generation.generation`` to keep synthetic-mode
code out of the production data pipeline.
"""

from __future__ import annotations

import concurrent.futures
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from src.data_generation.generation import (
    GeneratedRawDataset,
    _element_abundances_from_spec,
    _generation_worker_count,
    _prepare_generation_directory,
    _write_generation_metadata,
    list_run_ids_from_consolidated,
    merge_run_files_to_chunk,
    write_equilibrium_hdf5,
    write_raw_run_hdf5,
)
from src.data_generation.sampling import RunSpecification, sample_run_specifications
from src.utils.helpers import get_logger

LOGGER = get_logger(__name__)


def _synthetic_vertical_coordinate(pressure_bar: np.ndarray) -> np.ndarray:
    """Map a pressure profile onto a normalized vertical coordinate.

    Parameters
    ----------
    pressure_bar : np.ndarray
        One-dimensional pressure profile in bar ordered from top to bottom or
        bottom to top.

    Returns
    -------
    np.ndarray
        Array with the same shape as ``pressure_bar`` containing values in
        ``[0, 1]``, where smaller pressures map near 0 and larger pressures
        map near 1.
    """
    logp = np.log10(np.clip(np.asarray(pressure_bar, dtype=np.float64), 1.0e-30, None))
    return np.clip((logp - np.min(logp)) / max(np.max(logp) - np.min(logp), 1.0e-8), 0.0, 1.0)


def _assign_species(state: np.ndarray, species_index: dict[str, int], name: str, values: np.ndarray) -> None:
    """Accumulate one species profile into a state tensor when the column exists.

    Parameters
    ----------
    state : np.ndarray
        State tensor with shape ``(nz, n_species)`` that is updated in place.
    species_index : dict[str, int]
        Mapping from species names to columns in ``state``.
    name : str
        Species name to update.
    values : np.ndarray
        Vertical profile with shape ``(nz,)`` or broadcast-compatible values to
        add into the selected species column.

    Returns
    -------
    None
        ``state`` is mutated in place when ``name`` appears in
        ``species_index``; otherwise the function is a no-op.
    """
    if name in species_index:
        state[:, species_index[name]] += np.asarray(values, dtype=np.float64)


def _synthetic_final_state(
    spec: RunSpecification,
    *,
    output_species: list[str],
) -> np.ndarray:
    """Build one heuristic final-state composition for smoke-test generation.

    Parameters
    ----------
    spec : RunSpecification
        Sampled run specification containing the vertical profile, elemental
        abundances, global toggles, and optional spectrum data.
    output_species : list[str]
        Species order required for the returned final-state tensor.

    Returns
    -------
    np.ndarray
        Heuristic final-state composition with shape
        ``(nz, len(output_species))`` in physical mixing-ratio space.
    """
    state_species = list(spec.metadata["state_species"])
    species_index = {name: idx for idx, name in enumerate(state_species)}
    state = np.full((int(spec.pressure_bar.size), len(state_species)), 1.0e-30, dtype=np.float64)

    elements = _element_abundances_from_spec(spec)
    he_frac = float(elements["He_H"])
    c_frac = float(elements["C_H"])
    o_frac = float(elements["O_H"])
    n_frac = float(elements["N_H"])
    s_frac = float(elements["S_H"])

    pressure_bar = np.asarray(spec.pressure_bar, dtype=np.float64)
    temperature_k = np.asarray(spec.temperature_k, dtype=np.float64)
    depth = _synthetic_vertical_coordinate(pressure_bar)
    top = 1.0 - depth
    hot = np.clip((temperature_k - 1000.0) / 1200.0, 0.0, 1.0)
    cool = np.clip((1200.0 - temperature_k) / 900.0, 0.0, 1.0)
    # Photochemistry is disabled — UV forcing is always zero.
    photo_profile = np.zeros_like(depth)
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
        o_frac * (0.45 + 0.20 * top + 0.20 * cool + 0.10 * sat_surface_h2o * depth) * (1.0 - 0.35 * photo_profile) * (1.0 - water_depletion),
    )
    _assign_species(
        state,
        species_index,
        "CO",
        c_frac * (0.35 + 0.35 * hot + 0.20 * depth + 0.10 * mix_strength),
    )
    _assign_species(
        state,
        species_index,
        "CO2",
        np.minimum(c_frac, o_frac) * (0.05 + 0.15 * photo_profile + 0.10 * top + 0.10 * molecular_diffusion * top),
    )
    _assign_species(
        state,
        species_index,
        "CH4",
        c_frac * (0.18 + 0.42 * depth + 0.08 * mix_strength) * (1.0 - 0.55 * hot),
    )
    _assign_species(
        state,
        species_index,
        "N2",
        n_frac * (0.45 + 0.25 * hot + 0.15 * depth + 0.10 * mix_strength),
    )
    _assign_species(
        state,
        species_index,
        "NH3",
        n_frac * (0.16 + 0.30 * depth + 0.10 * cool) * (1.0 - 0.40 * photo_profile),
    )
    _assign_species(
        state,
        species_index,
        "H2S",
        s_frac * (0.50 + 0.22 * depth + 0.10 * cool) * (1.0 - 0.45 * photo_profile) * (1.0 - sulfur_depletion),
    )
    _assign_species(
        state,
        species_index,
        "SH",
        s_frac * (0.004 + 0.03 * photo_profile + 0.01 * upwind_diffusion * top),
    )
    _assign_species(
        state,
        species_index,
        "S",
        s_frac * (0.002 + 0.02 * photo_profile + 0.006 * ion_chemistry * top),
    )
    _assign_species(
        state,
        species_index,
        "SO",
        s_frac * (0.001 + 0.03 * photo_profile + 0.01 * hot) * (1.0 - 0.20 * sulfur_depletion),
    )
    _assign_species(
        state,
        species_index,
        "SO2",
        s_frac * (0.001 + 0.04 * photo_profile + 0.01 * mix_strength) * (1.0 - 0.15 * sulfur_depletion),
    )
    _assign_species(
        state,
        species_index,
        "S2",
        s_frac * (0.0005 + 0.01 * depth + 0.005 * upwind_diffusion * top),
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
        o_frac * 1.0e-3 * (1.0 + 15.0 * photo_profile + 3.0 * ion_chemistry),
    )
    _assign_species(
        state,
        species_index,
        "OH",
        o_frac * 2.0e-3 * (1.0 + 8.0 * photo_profile + 2.0 * boundary_conditions * top),
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
        he_reservoir_fraction = np.clip(he_frac / max(1.0 + he_frac, 1.0e-6), 0.08, 0.18)
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
    """Generate and serialize one synthetic raw run.

    Parameters
    ----------
    spec : RunSpecification
        Sampled atmospheric specification for the run.
    runs_dir : Path
        Directory receiving the generated HDF5 file.
    output_species : list[str]
        Output species ordering to write into the run contract.

    Returns
    -------
    Path
        Path to the written synthetic raw-run HDF5 file.
    """
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
    """Generate a full synthetic raw dataset without external chemistry runtimes.

    Parameters
    ----------
    config : dict[str, Any]
        Validated generation config.
    project_root : Path
        Repository root used to resolve raw-data paths.
    num_runs : int or None, optional
        Optional override for ``generation.num_runs``.

    Returns
    -------
    GeneratedRawDataset
        Paths describing the generated raw dataset, manifest, and coverage
        summary.
    """
    LOGGER.info("Synthetic generation starting (num_runs=%s)", num_runs or "config default")
    run_root, raw_root, info_root, runs_dir, chunks_dir, reusable_path = _prepare_generation_directory(
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
    if runs_dir is None or chunks_dir is None:
        raise RuntimeError("Synthetic generation requires writable staging directories.")
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
                elemental_abundances_frac=spec.elemental_abundances_frac,
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
    consolidated_path = merge_run_files_to_chunk(
        run_files, raw_root / "runs.h5",
    )
    for staging in (runs_dir, chunks_dir):
        if staging.exists() and not any(staging.iterdir()):
            staging.rmdir()
    manifest_path, coverage_path = _write_generation_metadata(
        info_root=info_root,
        run_files=[consolidated_path],
        specs=prepared_specs,
        config=config,
    )
    return GeneratedRawDataset(
        run_root=run_root,
        raw_root=raw_root,
        run_ids=[spec.run_id for spec in prepared_specs],
        consolidated_path=consolidated_path,
        manifest_path=manifest_path,
        coverage_path=coverage_path,
    )
