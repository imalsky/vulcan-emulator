"""ExoGibbs equilibrium chemistry generation backend.

Replaces the FastChem subprocess path with direct in-process calls to
ExoGibbs (a JAX-native Gibbs free energy minimizer). Uses AAG21 solar
abundances as background, overriding He/C/O/N/S with the sampled globals.

The output contract is identical to FastChem: ``write_equilibrium_hdf5``
produces the same HDF5 layout consumed by normalization and training.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..utils.helpers import get_logger
from .generation import (
    GeneratedRawDataset,
    _attach_species_metadata,
    _prepare_generation_directory,
    _requested_run_count,
    _write_generation_metadata,
    merge_chunks_to_runs_h5,
    merge_run_files_to_chunk,
    write_equilibrium_hdf5,
)
from .sampling import (
    RunSpecification,
    build_sampling_plan,
    sample_run_specifications_slice,
)

LOGGER = get_logger(__name__)


@dataclass
class _ExoGibbsRuntime:
    """Cached ExoGibbs runtime objects (expensive to initialize, reusable)."""

    chem: Any
    initializer: Any
    options: Any
    species_indices: np.ndarray
    output_species: list[str]


_RUNTIME: _ExoGibbsRuntime | None = None


def _init_runtime(config: dict[str, Any]) -> _ExoGibbsRuntime:
    """Lazily initialize ExoGibbs runtime objects."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME

    from exogibbs.api import (
        get_default_equilibrium_grid_path,
        load_equilibrium_grid_netcdf,
    )
    from exogibbs.api.equilibrium import (
        EquilibriumOptions,
        GridEquilibriumInitializer,
    )
    from exogibbs.presets.fastchem import chemsetup

    from ..models.classical_reference import FASTCHEM_TO_EXOGIBBS_SPECIES_ALIASES

    LOGGER.info("Initializing ExoGibbs runtime...")
    chem = chemsetup()
    grid_path = get_default_equilibrium_grid_path("fastchem")
    grid = load_equilibrium_grid_netcdf(str(grid_path))
    initializer = GridEquilibriumInitializer(grid=grid, preset_name="fastchem")
    opts = EquilibriumOptions(
        epsilon_crit=1e-11, max_iter=1000, method="vmap_cold"
    )

    output_species = list(config["data_spec"]["output_species"])
    indices: list[int] = []
    for label in output_species:
        aliases = FASTCHEM_TO_EXOGIBBS_SPECIES_ALIASES.get(str(label))
        if aliases is None:
            raise KeyError(f"No ExoGibbs alias mapping defined for {label!r}.")
        index = next(
            (chem.species.index(name) for name in aliases if name in chem.species),
            None,
        )
        if index is None:
            raise KeyError(
                f"Could not find any ExoGibbs alias for {label!r} in {aliases!r}."
            )
        indices.append(int(index))

    LOGGER.info("ExoGibbs runtime ready: %d species mapped.", len(indices))
    _RUNTIME = _ExoGibbsRuntime(
        chem=chem,
        initializer=initializer,
        options=opts,
        species_indices=np.array(indices, dtype=np.int32),
        output_species=output_species,
    )
    return _RUNTIME


def _build_element_vector(
    runtime: _ExoGibbsRuntime,
    globals_map: dict[str, float],
) -> jnp.ndarray:
    """Build the 28-element + electron ExoGibbs vector from sampled globals.

    Uses AAG21 solar for all elements as background, overrides He/C/O/N/S
    with the sampled n_X/n_H values, then converts to mole fractions.
    """
    from exojax.utils.zsol import nsol

    solar = nsol()
    chem = runtime.chem

    x_over_h: dict[str, float] = {}
    for element in chem.elements[:-1]:  # exclude 'e-'
        x_over_h[str(element)] = float(solar[str(element)]) / float(solar["H"])

    # Override with sampled globals
    _ELEMENT_TO_SYMBOL = {"He_H": "He", "C_H": "C", "O_H": "O", "N_H": "N", "S_H": "S"}
    for key, symbol in _ELEMENT_TO_SYMBOL.items():
        if key in globals_map and symbol in x_over_h:
            x_over_h[symbol] = float(globals_map[key])

    # H is the reference
    x_over_h["H"] = 1.0

    # Convert n_X/n_H to n_X/n_total (mole fractions)
    total_ratio = sum(x_over_h.values())
    mole_fracs = [x_over_h[str(el)] / total_ratio for el in chem.elements[:-1]]
    mole_fracs.append(0.0)  # electrons

    return jnp.array(mole_fracs, dtype=jnp.float64)


_VMR_FLOOR = 1.0e-30
_MAX_NAN_FRACTION = 0.20


def _run_single_profile(
    runtime: _ExoGibbsRuntime,
    spec: RunSpecification,
) -> np.ndarray | None:
    """Run ExoGibbs equilibrium for one atmospheric profile.

    Returns the VMR array of shape (nz, n_output_species), or None on failure.
    Levels where ExoGibbs fails to converge (NaN) are replaced with the
    training floor value. Profiles with >20% NaN levels are rejected.
    """
    from exogibbs.api.equilibrium import equilibrium_profile

    element_vector = _build_element_vector(runtime, spec.globals)
    temperature_k = jnp.array(spec.temperature_k, dtype=jnp.float64)
    pressure_bar = jnp.array(spec.pressure_bar, dtype=jnp.float64)

    res = equilibrium_profile(
        runtime.chem,
        temperature_k,
        pressure_bar,
        element_vector,
        Pref=1.0,
        initializer=runtime.initializer,
        options=runtime.options,
    )

    vmr_all = np.asarray(res.x, dtype=np.float64)
    vmr_output = vmr_all[:, runtime.species_indices]

    nan_mask = np.any(~np.isfinite(vmr_output), axis=1)
    nan_fraction = float(nan_mask.sum()) / max(len(nan_mask), 1)
    if nan_fraction > _MAX_NAN_FRACTION:
        return None

    vmr_output = np.where(np.isfinite(vmr_output), vmr_output, _VMR_FLOOR)
    vmr_output = np.clip(vmr_output, _VMR_FLOOR, None)

    return vmr_output


def run_exogibbs_generation(
    config: dict[str, Any],
    *,
    project_root: Path,
    num_runs: int | None = None,
) -> GeneratedRawDataset:
    """Generate equilibrium chemistry training data using ExoGibbs.

    Mirrors the FastChem generation path but calls ExoGibbs directly in
    Python (no subprocess). The output HDF5 contract is identical.
    """
    requested = _requested_run_count(config, num_runs)
    (
        run_root,
        raw_root,
        info_root,
        runs_dir,
        chunks_dir,
        reuse_path,
    ) = _prepare_generation_directory(
        config, project_root=project_root, num_runs=num_runs
    )

    if reuse_path is not None:
        from .generation import list_run_ids_from_consolidated

        LOGGER.info("Reusing existing ExoGibbs raw data at %s", reuse_path)
        run_ids = list_run_ids_from_consolidated(reuse_path)
        return GeneratedRawDataset(
            run_root=run_root,
            raw_root=raw_root,
            run_ids=run_ids,
            consolidated_path=reuse_path,
        )

    assert runs_dir is not None and chunks_dir is not None

    runtime = _init_runtime(config)
    plan = build_sampling_plan(config=config, project_root=project_root)
    chunk_size = int(config["generation"]["sample_chunk_size"])
    state_species = list(config["data_spec"]["state_species"])
    output_species = list(config["data_spec"]["output_species"])

    all_specs: list[RunSpecification] = []
    chunk_files: list[Path] = []
    failed_run_ids: list[str] = []

    # JIT warmup
    LOGGER.info("Warming up ExoGibbs JIT compilation...")
    warmup_specs = sample_run_specifications_slice(plan, start=0, end=1)
    _run_single_profile(runtime, warmup_specs[0])
    LOGGER.info("JIT warmup complete.")

    total_chunks = (requested + chunk_size - 1) // chunk_size
    LOGGER.info(
        "Generating %d ExoGibbs equilibrium runs in %d chunks of %d...",
        requested,
        total_chunks,
        chunk_size,
    )

    t0 = time.time()
    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, requested)
        specs = sample_run_specifications_slice(plan, start=start, end=end)
        specs = _attach_species_metadata(specs, config)

        run_files: list[Path] = []
        for spec in specs:
            vmr = _run_single_profile(runtime, spec)
            if vmr is None:
                failed_run_ids.append(spec.run_id)
                LOGGER.debug("ExoGibbs failed for %s", spec.run_id)
                continue

            run_path = runs_dir / f"{spec.run_id}.h5"
            write_equilibrium_hdf5(
                run_path,
                spec=spec,
                equilibrium_ymix=vmr,
                state_species=state_species,
                output_species=output_species,
            )
            run_files.append(run_path)
            all_specs.append(spec)

        if run_files:
            chunk_path = chunks_dir / f"chunk_{chunk_idx:04d}.h5"
            merge_run_files_to_chunk(run_files, chunk_path)
            chunk_files.append(chunk_path)

        elapsed = time.time() - t0
        rate = end / max(elapsed, 1.0)
        LOGGER.info(
            "Chunk %d/%d done (%d runs, %.0f runs/s, %d failed total)",
            chunk_idx + 1,
            total_chunks,
            len(run_files),
            rate,
            len(failed_run_ids),
        )

    # Backfill failed runs using a separate plan with a different seed
    backfill_cfg = config["generation"].get("backfill", {})
    if failed_run_ids and backfill_cfg.get("enabled", False):
        max_retries = int(backfill_cfg.get("max_retries", 10))
        LOGGER.info(
            "Backfilling %d failed runs (max %d retries)...",
            len(failed_run_ids),
            max_retries,
        )
        backfill_id_counter = requested
        for retry in range(max_retries):
            if not failed_run_ids:
                break
            n_needed = len(failed_run_ids)
            failed_run_ids.clear()
            backfill_plan = build_sampling_plan(
                config=config,
                project_root=project_root,
                num_runs=n_needed,
                seed=int(config["generation"]["seed"]) + 1000 + retry,
            )
            backfill_specs = sample_run_specifications_slice(
                backfill_plan, start=0, end=n_needed
            )
            backfill_specs = _attach_species_metadata(
                backfill_specs, config,
                start_index=backfill_id_counter,
            )
            backfill_id_counter += n_needed
            for spec in backfill_specs:
                vmr = _run_single_profile(runtime, spec)
                if vmr is None:
                    failed_run_ids.append(spec.run_id)
                    continue
                run_path = runs_dir / f"{spec.run_id}.h5"
                write_equilibrium_hdf5(
                    run_path,
                    spec=spec,
                    equilibrium_ymix=vmr,
                    state_species=state_species,
                    output_species=output_species,
                )
                chunk_path = chunks_dir / f"backfill_{retry:02d}_{spec.run_id}.h5"
                merge_run_files_to_chunk([run_path], chunk_path)
                chunk_files.append(chunk_path)
                all_specs.append(spec)

    if failed_run_ids:
        LOGGER.warning(
            "%d runs failed after backfill and will be missing from the dataset.",
            len(failed_run_ids),
        )

    # Merge all chunks into runs.h5
    consolidated_path = raw_root / "runs.h5"
    merge_chunks_to_runs_h5(chunk_files, consolidated_path)

    # Write metadata
    manifest_path, coverage_path = _write_generation_metadata(
        info_root=info_root,
        run_files=[consolidated_path],
        specs=all_specs,
        config=config,
    )

    from .generation import list_run_ids_from_consolidated

    run_ids = list_run_ids_from_consolidated(consolidated_path)
    elapsed = time.time() - t0
    LOGGER.info(
        "ExoGibbs generation complete: %d runs in %.1f minutes (%.0f runs/s).",
        len(run_ids),
        elapsed / 60.0,
        len(run_ids) / max(elapsed, 1.0),
    )

    return GeneratedRawDataset(
        run_root=run_root,
        raw_root=raw_root,
        run_ids=run_ids,
        consolidated_path=consolidated_path,
        manifest_path=manifest_path,
        coverage_path=coverage_path,
    )
