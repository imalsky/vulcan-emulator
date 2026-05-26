"""ExoGibbs equilibrium chemistry generation backend.

Replaces the FastChem subprocess path with direct in-process calls to
ExoGibbs (a JAX-native Gibbs free energy minimizer). Uses AAG21 solar
abundances as background, overriding He/C/O/N/S with the sampled globals.

The output contract is identical to FastChem: ``write_equilibrium_hdf5``
produces the same HDF5 layout consumed by normalization and training.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..utils.helpers import get_logger
from .generation import (
    _SHARD_BACKFILL_SLOT_SIZE,
    _SHARD_RUN_ID_MAX_EXCLUSIVE,
    GeneratedRawDataset,
    _attach_species_metadata,
    _generation_worker_count,
    _prepare_generation_directory,
    _requested_run_count,
    _run_specification_to_payload,
    _write_generation_metadata,
    list_run_ids_from_consolidated,
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

    from exogibbs.api.equilibrium import (
        DefaultEquilibriumInitializer,
        EquilibriumOptions,
    )
    from exogibbs.presets.fastchem import chemsetup

    from ..models.classical_reference import FASTCHEM_TO_EXOGIBBS_SPECIES_ALIASES

    LOGGER.info("Initializing ExoGibbs runtime...")
    chem = chemsetup()
    initializer = DefaultEquilibriumInitializer()
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

    Uses the ExoGibbs reference vector for all elements as background,
    overrides He/C/O/N/S with sampled n_X/n_H values, then converts to mole
    fractions.
    """
    chem = runtime.chem
    reference = np.asarray(chem.element_vector_reference, dtype=np.float64)
    if reference.shape[0] != len(chem.elements):
        raise ValueError(
            "ExoGibbs element_vector_reference does not match chem.elements."
        )

    reference_by_element = {
        str(element): float(value)
        for element, value in zip(chem.elements, reference)
    }
    h_reference = reference_by_element["H"]

    x_over_h: dict[str, float] = {}
    for element in chem.elements[:-1]:  # exclude 'e-'
        x_over_h[str(element)] = reference_by_element[str(element)] / h_reference

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


def _process_run(
    runtime: _ExoGibbsRuntime,
    spec: RunSpecification,
    runs_dir: Path,
    state_species: list[str],
    output_species: list[str],
) -> tuple[RunSpecification, Path | None]:
    """Compute equilibrium + write HDF5 for one run. Thread-safe (unique paths)."""
    vmr = _run_single_profile(runtime, spec)
    if vmr is None:
        return spec, None
    run_path = runs_dir / f"{spec.run_id}.h5"
    write_equilibrium_hdf5(
        run_path,
        spec=spec,
        equilibrium_ymix=vmr,
        state_species=state_species,
        output_species=output_species,
    )
    return spec, run_path


def run_exogibbs_generation(
    config: dict[str, Any],
    *,
    project_root: Path,
    num_runs: int | None = None,
    shard_id: int | None = None,
    num_shards: int | None = None,
    staging_root: Path | None = None,
) -> GeneratedRawDataset:
    """Generate equilibrium chemistry training data using ExoGibbs.

    Mirrors the FastChem generation path but calls ExoGibbs directly in
    Python (no subprocess). The output HDF5 contract is identical.
    """
    is_sharded = shard_id is not None
    if is_sharded != (num_shards is not None):
        raise ValueError("shard_id and num_shards must be passed together or not at all.")
    if is_sharded:
        if not (0 <= shard_id < num_shards):
            raise ValueError(
                f"shard_id={shard_id} out of range for num_shards={num_shards}."
            )
        LOGGER.info(
            "ExoGibbs generation starting (shard %d/%d, num_runs=%s, staging_root=%s)",
            shard_id,
            num_shards,
            num_runs or "config default",
            staging_root,
        )
    else:
        LOGGER.info("ExoGibbs generation starting (num_runs=%s)", num_runs or "config default")

    target_count = _requested_run_count(config, num_runs)
    (
        run_root,
        raw_root,
        info_root,
        runs_dir,
        chunks_dir,
        reuse_path,
    ) = _prepare_generation_directory(
        config,
        project_root=project_root,
        num_runs=num_runs,
        shard_id=shard_id,
        staging_root=staging_root,
    )

    if reuse_path is not None:
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
    configured_chunk_size = int(config["generation"]["sample_chunk_size"])
    state_species = list(config["data_spec"]["state_species"])
    output_species = list(config["data_spec"]["output_species"])

    shard_start = (shard_id * target_count) // num_shards if is_sharded else 0
    shard_end = (
        ((shard_id + 1) * target_count) // num_shards
        if is_sharded
        else target_count
    )
    shard_count = shard_end - shard_start
    shard_tag = f"_s{shard_id:02d}" if is_sharded else ""
    chunk_size = max(1, min(configured_chunk_size, max(shard_count, 1)))

    max_workers = _generation_worker_count(config, max(shard_count, 1))
    LOGGER.info("Using %d parallel workers for ExoGibbs generation.", max_workers)

    LOGGER.info("Warming up ExoGibbs JIT compilation...")
    warmup_specs = sample_run_specifications_slice(plan, start=0, end=1)
    _run_single_profile(runtime, warmup_specs[0])
    LOGGER.info("JIT warmup complete.")

    chunk_files: list[Path] = sorted(chunks_dir.glob("chunk_*.h5"))
    completed_run_ids: set[str] = set()
    for chunk_path in chunk_files:
        completed_run_ids.update(list_run_ids_from_consolidated(chunk_path))
    if chunk_files:
        LOGGER.info(
            "Resuming: found %d existing chunk files covering %d runs in %s",
            len(chunk_files),
            len(completed_run_ids),
            chunks_dir,
        )

    deterministic_specs: list[RunSpecification] = []
    successful_backfill_specs: list[RunSpecification] = []
    all_failures: list[str] = []

    total_chunks = (shard_count + chunk_size - 1) // chunk_size
    LOGGER.info(
        "Generating %d ExoGibbs equilibrium runs in %d chunks of %d...",
        shard_count,
        total_chunks,
        chunk_size,
    )

    pool = ThreadPoolExecutor(max_workers=max_workers) if max_workers > 1 else None

    def _run_batch(
        specs: list[RunSpecification],
    ) -> tuple[list[Path], list[RunSpecification], list[str]]:
        """Run one ExoGibbs batch while isolating per-profile failures."""
        run_files: list[Path] = []
        successes: list[RunSpecification] = []
        failures: list[str] = []
        if pool is None:
            for spec in specs:
                try:
                    completed_spec, run_path = _process_run(
                        runtime,
                        spec,
                        runs_dir,
                        state_species,
                        output_species,
                    )
                except Exception:
                    LOGGER.warning("Run %s failed, will backfill", spec.run_id, exc_info=True)
                    failures.append(spec.run_id)
                    continue
                if run_path is None:
                    failures.append(completed_spec.run_id)
                else:
                    run_files.append(run_path)
                    successes.append(completed_spec)
            return run_files, successes, failures

        future_to_spec = {
            pool.submit(
                _process_run,
                runtime,
                spec,
                runs_dir,
                state_species,
                output_species,
            ): spec
            for spec in specs
        }
        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                completed_spec, run_path = future.result()
            except Exception:
                LOGGER.warning("Run %s failed, will backfill", spec.run_id, exc_info=True)
                failures.append(spec.run_id)
                continue
            if run_path is None:
                failures.append(completed_spec.run_id)
            else:
                run_files.append(run_path)
                successes.append(completed_spec)
        return run_files, successes, failures

    t0 = time.time()
    try:
        for chunk_idx, start in enumerate(range(shard_start, shard_end, chunk_size)):
            end = min(start + chunk_size, shard_end)
            specs = sample_run_specifications_slice(plan, start=start, end=end)
            specs = _attach_species_metadata(specs, config)
            deterministic_specs.extend(specs)
            pending_specs = [spec for spec in specs if spec.run_id not in completed_run_ids]
            skipped = len(specs) - len(pending_specs)
            if skipped:
                LOGGER.info(
                    "Chunk [%d, %d): skipping %d already-completed runs",
                    start,
                    end,
                    skipped,
                )
            if not pending_specs:
                continue

            run_files, _, failures = _run_batch(pending_specs)
            all_failures.extend(failures)

            if run_files:
                chunk_path = chunks_dir / f"chunk{shard_tag}_{start:06d}_{end:06d}.h5"
                suffix = 0
                while chunk_path.exists():
                    suffix += 1
                    chunk_path = chunks_dir / (
                        f"chunk{shard_tag}_{start:06d}_{end:06d}_{suffix:03d}.h5"
                    )
                merge_run_files_to_chunk(run_files, chunk_path)
                chunk_files.append(chunk_path)
                completed_run_ids.update(path.stem for path in run_files)

            elapsed = time.time() - t0
            rate = len(completed_run_ids) / max(elapsed, 1.0)
            LOGGER.info(
                "Chunk %d/%d done (%d new runs, %.0f runs/s, %d failed total)",
                chunk_idx + 1,
                total_chunks,
                len(run_files),
                rate,
                len(all_failures),
            )

        backfill_cfg = config["generation"].get("backfill", {})
        target_for_shortfall = shard_count if is_sharded else target_count
        if is_sharded:
            next_run_index = target_count + shard_id * _SHARD_BACKFILL_SLOT_SIZE
            backfill_id_slot_end = target_count + (shard_id + 1) * _SHARD_BACKFILL_SLOT_SIZE
        else:
            next_run_index = target_count
            backfill_id_slot_end = _SHARD_RUN_ID_MAX_EXCLUSIVE

        if bool(backfill_cfg.get("enabled", False)) and all_failures:
            max_retries = int(backfill_cfg.get("max_retries", 10))
            LOGGER.info(
                "Backfilling failed ExoGibbs runs (max %d retries)...",
                max_retries,
            )
            base_seed = int(config["generation"]["seed"])
            for attempt in range(1, max_retries + 1):
                shortfall = target_for_shortfall - len(completed_run_ids)
                if shortfall <= 0:
                    break
                if next_run_index + shortfall > backfill_id_slot_end:
                    raise RuntimeError(
                        f"Backfill would overflow this shard's ID slot: "
                        f"next_run_index={next_run_index}, shortfall={shortfall}, "
                        f"slot_end={backfill_id_slot_end}."
                    )
                backfill_seed = base_seed + 1000 * attempt + (
                    1_000_000 * shard_id if is_sharded else 0
                )
                backfill_plan = build_sampling_plan(
                    config=config,
                    project_root=project_root,
                    num_runs=shortfall,
                    seed=backfill_seed,
                )
                raw_backfill_specs = sample_run_specifications_slice(
                    backfill_plan,
                    start=0,
                    end=shortfall,
                )
                backfill_specs = _attach_species_metadata(
                    raw_backfill_specs,
                    config,
                    start_index=next_run_index,
                )
                next_run_index += len(backfill_specs)
                run_files, success_specs, failures = _run_batch(backfill_specs)
                all_failures.extend(failures)
                successful_backfill_specs.extend(success_specs)
                if run_files:
                    chunk_path = chunks_dir / f"chunk_backfill{shard_tag}_{attempt:02d}.h5"
                    suffix = 0
                    while chunk_path.exists():
                        suffix += 1
                        chunk_path = chunks_dir / (
                            f"chunk_backfill{shard_tag}_{attempt:02d}_{suffix:03d}.h5"
                        )
                    merge_run_files_to_chunk(run_files, chunk_path)
                    chunk_files.append(chunk_path)
                    completed_run_ids.update(path.stem for path in run_files)
                if not failures:
                    break
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    straggler_run_files = sorted(runs_dir.glob("run_*.h5"))
    if straggler_run_files:
        final_chunk = chunks_dir / f"chunk_stragglers{shard_tag}.h5"
        merge_run_files_to_chunk(straggler_run_files, final_chunk)
        chunk_files.append(final_chunk)
        completed_run_ids.update(list_run_ids_from_consolidated(final_chunk))

    final_target = shard_count if is_sharded else target_count
    shortfall = final_target - len(completed_run_ids)
    remaining_failures = sorted(set(all_failures) - completed_run_ids)
    failed_log = (
        info_root / "shards" / f"failed_runs_s{shard_id:02d}.json"
        if is_sharded
        else info_root / "failed_runs.json"
    )
    if all_failures or shortfall > 0:
        failed_log.parent.mkdir(parents=True, exist_ok=True)
        failed_log.write_text(
            json.dumps(remaining_failures or all_failures, indent=2) + "\n",
            encoding="utf-8",
        )
    if shortfall > 0:
        raise RuntimeError(
            f"ExoGibbs generation finished {shortfall} successful runs "
            f"short of the requested total. See {failed_log} for failed run IDs."
        )

    successful_deterministic_specs = [
        spec for spec in deterministic_specs if spec.run_id in completed_run_ids
    ]
    successful_specs = successful_deterministic_specs + successful_backfill_specs

    if is_sharded:
        deterministic_ids = [f"run_{i:06d}" for i in range(shard_start, shard_end)]
        deterministic_set = set(deterministic_ids)
        deterministic_ok = sorted(rid for rid in completed_run_ids if rid in deterministic_set)
        deterministic_failed = sorted(rid for rid in deterministic_ids if rid not in completed_run_ids)
        backfill_ok = sorted(
            rid for rid in completed_run_ids
            if rid.startswith("run_") and int(rid.split("_", 1)[1]) >= target_count
        )
        fragment = {
            "schema_version": 1,
            "shard_id": shard_id,
            "num_shards": num_shards,
            "shard_start": shard_start,
            "shard_end": shard_end,
            "config_chemistry_type": "exogibbs",
            "config_seed": int(config["generation"]["seed"]),
            "config_num_runs": target_count,
            "deterministic_run_ids": deterministic_ids,
            "successful_deterministic_run_ids": deterministic_ok,
            "failed_deterministic_run_ids": deterministic_failed,
            "successful_backfill_run_ids": backfill_ok,
            "successful_backfill_specs": [
                _run_specification_to_payload(spec)
                for spec in successful_backfill_specs
            ],
            "remaining_failures": remaining_failures,
            "chunk_files_relpath": [
                str(path.relative_to(raw_root)) for path in sorted(chunk_files)
            ],
            "backfill_seed_formula": "base_seed + 1000*attempt + 1_000_000*shard_id",
            "shard_backfill_slot_size": _SHARD_BACKFILL_SLOT_SIZE,
            "backfill_id_slot_start": target_count + shard_id * _SHARD_BACKFILL_SLOT_SIZE,
            "backfill_id_slot_end": target_count + (shard_id + 1) * _SHARD_BACKFILL_SLOT_SIZE,
            "completed_at_iso8601": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        }
        fragment_path = info_root / "shards" / f"shard_s{shard_id:02d}.json"
        fragment_path.parent.mkdir(parents=True, exist_ok=True)
        fragment_path.write_text(json.dumps(fragment, indent=2) + "\n", encoding="utf-8")
        if runs_dir.exists() and not any(runs_dir.iterdir()):
            runs_dir.rmdir()
        LOGGER.info(
            "ExoGibbs shard %d/%d complete: %d runs across %d chunk files",
            shard_id,
            num_shards,
            len(completed_run_ids),
            len(chunk_files),
        )
        return GeneratedRawDataset(
            run_root=run_root,
            raw_root=raw_root,
            run_ids=sorted(completed_run_ids),
            consolidated_path=raw_root / "runs.h5",
            manifest_path=fragment_path,
            coverage_path=None,
        )

    # Merge all chunks into runs.h5
    consolidated_path = raw_root / "runs.h5"
    merge_chunks_to_runs_h5(chunk_files, consolidated_path)
    if chunks_dir.exists() and not any(chunks_dir.iterdir()):
        chunks_dir.rmdir()
    if runs_dir.exists() and not any(runs_dir.iterdir()):
        runs_dir.rmdir()

    # Write metadata
    manifest_path, coverage_path = _write_generation_metadata(
        info_root=info_root,
        run_files=[consolidated_path],
        specs=successful_specs,
        config=config,
    )

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
