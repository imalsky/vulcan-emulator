"""Shared notebook helpers for apples-to-apples chemistry comparisons.

This module centralizes the three reference paths used by the demo notebooks:

1. exported emulator bundle inference,
2. live FastChem subprocess reruns using the same input-writing logic as
   dataset generation, and
3. ExoGibbs element-vector construction that can mimic the FastChem training
   contract as closely as possible.

The helpers intentionally avoid any notebook-specific plotting code so the
notebooks can stay focused on analysis and commentary.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from ..constants import (
    FASTCHEM_LODDERS_SOLAR_ABUNDANCES,
    FASTCHEM_TRACKED_ELEMENTS,
    SOLAR_ABUNDANCES,
)
from ..utils.helpers import resolve_path

EPSILON = 1.0e-30

# Mirror src.data_generation.generation so notebook-side "live FastChem"
# comparisons use the exact same hidden-metal scaling assumption as training.
FASTCHEM_METALLICITY_SCALED_ELEMENTS = frozenset(
    {"P", "Si", "Ti", "V", "Cl", "K", "Na", "Mg", "F", "Ca", "Fe"}
)
FEMH_VOLATILE_WEIGHTS: dict[str, float] = {"C_H": 0.48, "O_H": 0.31, "S_H": 0.21}
ALPHA_FE_SLOPE = 0.15

# These are the same solar X/H anchors used by the data-generation code.
SOLAR_ELEMENT_ABUNDANCES = dict(SOLAR_ABUNDANCES)

FASTCHEM_TO_EXOGIBBS_SPECIES_ALIASES: dict[str, tuple[str, ...]] = {
    "H2": ("H2",),
    "He": ("He1", "He"),
    "H": ("H1", "H"),
    "O": ("O1", "O"),
    "OH": ("H1O1", "OH"),
    "H2O": ("H2O1", "O1H2", "H2O"),
    "CO": ("C1O1", "CO"),
    "CO2": ("C1O2", "CO2"),
    "CH4": ("C1H4", "H4C1", "CH4"),
    "N2": ("N2",),
    "NH3": ("H3N1", "N1H3", "NH3"),
    "H2S": ("H2S1", "S1H2", "H2S"),
    "SH": ("H1S1", "SH"),
    "S": ("S1", "S"),
    "SO": ("O1S1", "SO"),
    "SO2": ("O2S1", "SO2"),
    "S2": ("S2",),
}


@dataclass(frozen=True)
class FastChemTestContext:
    """Resolved processed/raw dataset context for one exported FastChem bundle."""

    bundle_path: Path
    processed_root: Path
    raw_root: Path | None
    split: Any
    normalization: dict[str, Any]
    contract: dict[str, Any]
    run_id_to_index: dict[str, int]
    raw_run_ids: frozenset[str]


@dataclass(frozen=True)
class FastChemTestCase:
    """One processed test example restored to physical units."""

    run_id: str
    pressure_bar: np.ndarray
    temperature_k: np.ndarray
    global_inputs: dict[str, float]
    stored_target_ymix: np.ndarray
    output_species: list[str]
    raw_globals: dict[str, float] | None
    raw_metadata: dict[str, Any] | None


@dataclass(frozen=True)
class RawEquilibriumProfile:
    """One raw FastChem equilibrium profile loaded from ``runs.h5``."""

    run_id: str
    pressure_bar: np.ndarray
    temperature_k: np.ndarray
    equilibrium_ymix: np.ndarray
    output_species: list[str]
    globals: dict[str, float]


def fastchem_globals_from_log_abund(
    log_abund: Sequence[float],
    *,
    element_names: Sequence[str] = ("He", "C", "O", "N", "S"),
) -> dict[str, float]:
    """Convert ``log10(X/H)`` values into the bundle's global-input mapping."""

    if len(log_abund) != len(element_names):
        raise ValueError(
            f"log_abund must have length {len(element_names)}, got {len(log_abund)}."
        )
    return {
        f"{element}_H": float(10.0 ** float(value))
        for element, value in zip(element_names, log_abund)
    }


def compute_fastchem_metallicity_offset_dex(
    globals_map: Mapping[str, float],
    *,
    solar_abundances: Mapping[str, float] | None = None,
) -> float:
    """Return the FastChem hidden-metal proxy used during data generation.

    The proxy is the inverse-variance-weighted mean of the volatile [X/H]
    offsets for C, O, and S, divided by ``1 - ALPHA_FE_SLOPE`` to convert the
    thin-disk [alpha/H] trend back to an approximate [Fe/H].
    """

    reference = SOLAR_ELEMENT_ABUNDANCES if solar_abundances is None else solar_abundances
    volatile_dex = sum(
        weight * np.log10(float(globals_map[name]) / float(reference[name]))
        for name, weight in FEMH_VOLATILE_WEIGHTS.items()
    )
    return float(volatile_dex / (1.0 - ALPHA_FE_SLOPE))


def build_exogibbs_element_vector(
    chem: Any,
    globals_map: Mapping[str, float],
    *,
    solar_abundances: Mapping[str, float] | None = None,
    mode: str = "fastchem_proxy",
) -> jnp.ndarray:
    """Build an ExoGibbs element vector from the emulator's 5-D globals.

    Parameters
    ----------
    chem : Any
        ExoGibbs chemistry object returned by ``chemsetup()``.
    globals_map : Mapping[str, float]
        FastChem-style elemental globals keyed as ``He_H``, ``C_H``, etc.
    solar_abundances : mapping or None, optional
        Per-element solar abundances keyed by element symbol. Only consulted
        in ``"aas_fixed"`` mode; ignored in ``"fastchem_proxy"`` (which is
        pinned to the FastChem training baseline). When omitted for
        ``"aas_fixed"`` the helper falls back to ``exojax.utils.zsol.nsol()``.
    mode : {"fastchem_proxy", "aas_fixed"}
        ``"fastchem_proxy"`` mirrors the exact element vector FastChem saw
        during data generation: Lodders (2009) as the background for the 16
        elements FastChem's solar file lists, He/C/O/N/S overwritten with the
        provided globals, the 11 refractory metals (P/Si/Ti/V/Cl/K/Na/Mg/F/
        Ca/Fe) additionally scaled by the same hidden-metallicity proxy used
        during data generation, and all elements outside FastChem's tracked
        set (e.g. Al/Ar/Co/Cr/Cu/Ge/Mn/Ne/Ni/Zn in ExoGibbs' 28-element
        setup) zeroed because FastChem never tracked them.
        ``"aas_fixed"`` leaves every non-free element at the supplied solar
        (AAG21 by default) and only overwrites He/C/O/N/S.
    """

    if mode not in {"fastchem_proxy", "aas_fixed"}:
        raise ValueError(
            "mode must be 'fastchem_proxy' or 'aas_fixed', "
            f"got {mode!r}."
        )

    if mode == "fastchem_proxy":
        # FastChem's solar_element_abundances.dat carries values in n_X/n_H
        # form (the astronomical log_eps = log10(n_X/n_H)+12 convention).
        # Reproduce that vector exactly, then normalize to the n_X/n_total
        # mole-fraction convention ExoGibbs expects in `element_vector`.
        x_over_h = {
            str(element): (
                float(FASTCHEM_LODDERS_SOLAR_ABUNDANCES[str(element)])
                if str(element) in FASTCHEM_TRACKED_ELEMENTS
                else 0.0
            )
            for element in chem.elements[:-1]
        }
        # Refractories carry the same hidden-metallicity proxy scaling
        # applied at training time by _write_abundances.
        proxy_scale = 10.0 ** compute_fastchem_metallicity_offset_dex(
            globals_map=globals_map
        )
        for element in FASTCHEM_METALLICITY_SCALED_ELEMENTS:
            if element in x_over_h:
                x_over_h[element] *= proxy_scale
        # He/C/O/N/S were overwritten with the sampled globals in the
        # FastChem input file, so do the same here before normalization.
        for element in ("He", "C", "O", "N", "S"):
            key = f"{element}_H"
            if key in globals_map and element in x_over_h:
                x_over_h[element] = float(globals_map[key])
        # Hydrogen is the reference; its ratio to itself is 1.
        if "H" in x_over_h:
            x_over_h["H"] = 1.0
        # Convert n_X/n_H -> n_X/n_total by dividing by the sum.
        total_ratio = sum(x_over_h.values())
        element_values = {key: value / total_ratio for key, value in x_over_h.items()}
    else:
        if solar_abundances is None:
            from exojax.utils.zsol import nsol

            solar_abundances = nsol()
        element_values = {
            str(element): float(solar_abundances[str(element)])
            for element in chem.elements[:-1]
        }
        for element in ("He", "C", "O", "N", "S"):
            key = f"{element}_H"
            if key in globals_map and element in element_values:
                element_values[element] = float(globals_map[key])

    vector = np.array(
        [element_values[str(element)] for element in chem.elements[:-1]],
        dtype=np.float64,
    )
    return jnp.append(jnp.asarray(vector), jnp.asarray(0.0, dtype=jnp.float64))


def build_exogibbs_species_indices(
    chem: Any,
    species_labels: Sequence[str],
) -> jnp.ndarray:
    """Map the emulator's 17 species onto ExoGibbs indices."""

    indices: list[int] = []
    for label in species_labels:
        aliases = FASTCHEM_TO_EXOGIBBS_SPECIES_ALIASES.get(str(label))
        if aliases is None:
            raise KeyError(f"No ExoGibbs alias mapping defined for {label!r}.")
        index = next((chem.species.index(name) for name in aliases if name in chem.species), None)
        if index is None:
            raise KeyError(
                f"Could not find any ExoGibbs alias for {label!r} in {aliases!r}."
            )
        indices.append(int(index))
    return jnp.asarray(indices, dtype=jnp.int32)


def mean_abs_log10_error(
    left: np.ndarray,
    right: np.ndarray,
    *,
    epsilon: float = EPSILON,
) -> float:
    """Return mean absolute log10 error between two VMR tables."""

    left_c = np.clip(np.asarray(left, dtype=np.float64), epsilon, None)
    right_c = np.clip(np.asarray(right, dtype=np.float64), epsilon, None)
    return float(np.mean(np.abs(np.log10(left_c) - np.log10(right_c))))


def _decode_labels(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _decode_scalar(value: Any) -> Any:
    scalar = np.asarray(value)[()]
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    if isinstance(scalar, np.bool_):
        return bool(scalar)
    if isinstance(scalar, np.integer):
        return int(scalar)
    if isinstance(scalar, np.floating):
        return float(scalar)
    return scalar


def _resolve_processed_root(bundle_path: Path, config: Mapping[str, Any], project_root: Path) -> Path:
    from ..utils.helpers import resolve_path as _resolve_path

    paths = config.get("paths", {}) if isinstance(config.get("paths"), Mapping) else {}
    candidates: list[Path] = []
    processed_value = paths.get("processed_root") if isinstance(paths, Mapping) else None
    if isinstance(processed_value, str) and processed_value:
        candidates.append(_resolve_path(processed_value, project_root))
    candidates.append(project_root / "data" / bundle_path.parent.name / "processed")
    for candidate in candidates:
        if (candidate / "test" / "metadata.json").exists():
            return candidate
    raise FileNotFoundError(f"No processed test split under: {candidates}")


def _resolve_raw_root(
    bundle_path: Path,
    config: Mapping[str, Any],
    processed_root: Path,
    project_root: Path,
) -> tuple[Path | None, frozenset[str]]:
    from ..data_generation.generation import list_run_ids_from_consolidated
    from ..utils.helpers import resolve_path as _resolve_path

    paths = config.get("paths", {}) if isinstance(config.get("paths"), Mapping) else {}
    candidates: list[Path] = []
    raw_value = paths.get("raw_root") if isinstance(paths, Mapping) else None
    if isinstance(raw_value, str) and raw_value:
        candidates.append(_resolve_path(raw_value, project_root))
    candidates.append(processed_root.parent / "raw")
    candidates.append(project_root / "data" / bundle_path.parent.name / "raw")
    for candidate in candidates:
        consolidated = candidate / "runs.h5"
        if consolidated.exists():
            run_ids = frozenset(list_run_ids_from_consolidated(consolidated))
            if run_ids:
                return candidate, run_ids
    return None, frozenset()


def load_fastchem_test_context(
    bundle_path: Path,
    config: Mapping[str, Any],
    *,
    project_root: Path,
    require_raw: bool = False,
) -> FastChemTestContext:
    """Load processed/raw dataset context for the selected FastChem bundle."""

    from ..data_generation.data_loader import load_processed_dataset

    processed_root = _resolve_processed_root(bundle_path, config, project_root)
    splits, normalization, contract = load_processed_dataset(processed_root)
    split = splits["test"]
    raw_root, raw_run_ids = _resolve_raw_root(bundle_path, config, processed_root, project_root)
    if require_raw and raw_root is None:
        raise FileNotFoundError("Raw dataset required but not found.")
    return FastChemTestContext(
        bundle_path=bundle_path,
        processed_root=processed_root,
        raw_root=raw_root,
        split=split,
        normalization=normalization,
        contract=contract,
        run_id_to_index={rid: i for i, rid in enumerate(split.run_ids)},
        raw_run_ids=raw_run_ids,
    )


def _load_raw_sidecars(raw_root: Path, run_id: str) -> tuple[dict[str, float], dict[str, Any]]:
    import h5py

    with h5py.File(raw_root / "runs.h5", "r") as handle:
        group = handle[run_id]
        globals_map = {
            key: float(np.asarray(group[f"globals/{key}"]))
            for key in group["globals"].keys()
        }
        if "inputs/element_input_order" in group and "inputs/elemental_abundances_frac" in group:
            labels = _decode_labels(np.asarray(group["inputs/element_input_order"]))
            profile = np.asarray(group["inputs/elemental_abundances_frac"], dtype=np.float64)
            for idx, label in enumerate(labels):
                globals_map[label] = float(profile[0, idx])
        metadata: dict[str, Any] = {}
        if "metadata" in group:
            metadata = {
                key: _decode_scalar(group[f"metadata/{key}"][()])
                for key in group["metadata"].keys()
            }
    return globals_map, metadata


def load_fastchem_raw_metadata_map(
    raw_root: Path,
    run_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Load the raw temperature-profile metadata for a list of run ids."""

    import h5py

    output: dict[str, dict[str, Any]] = {}
    with h5py.File(raw_root / "runs.h5", "r") as handle:
        for run_id in run_ids:
            group = handle[run_id]
            if "metadata" not in group:
                output[str(run_id)] = {}
                continue
            output[str(run_id)] = {
                key: _decode_scalar(group[f"metadata/{key}"][()])
                for key in group["metadata"].keys()
            }
    return output


def classify_temperature_profile_bucket(metadata: Mapping[str, Any]) -> str:
    """Return the plotting bucket used by the extras notebook."""

    source = metadata.get("temperature_profile_source")
    if source == "pt_library":
        return "pt_library"
    if source == "analytic":
        key = "temperature_profile_analytic_convective_adjustment_applied"
        return "analytic_convective" if bool(metadata[key]) else "analytic_radiative"
    raise KeyError("Unsupported temperature_profile_source.")


def load_fastchem_test_case(ctx: FastChemTestContext, run_id: str) -> FastChemTestCase:
    """Restore one processed test example to physical units."""

    from ..data_generation.preprocess import inverse_block, inverse_mixed_block

    idx = ctx.run_id_to_index[run_id]
    nz = int(np.asarray(ctx.split.valid_mask[idx]).sum())
    seq_in = np.asarray(ctx.split.sequence_inputs[idx, :nz], dtype=np.float64)
    glob_in = np.asarray(ctx.split.global_inputs[idx : idx + 1], dtype=np.float64)
    target = np.asarray(ctx.split.target_outputs[idx, :nz], dtype=np.float64)

    pressure_bar = inverse_block(seq_in[:, 0:1], ctx.normalization["sequence_static"]["blocks"][0])[:, 0]
    temperature_k = inverse_block(seq_in[:, 1:2], ctx.normalization["sequence_static"]["blocks"][1])[:, 0]
    global_vec = inverse_mixed_block(glob_in, ctx.normalization["global_static"])[0]
    feature_order = list(ctx.contract["global_static_feature_order"])
    global_map = {name: float(global_vec[i]) for i, name in enumerate(feature_order)}
    stored = inverse_block(target, ctx.normalization["target"])

    raw_globals = raw_metadata = None
    if ctx.raw_root is not None and run_id in ctx.raw_run_ids:
        raw_globals, raw_metadata = _load_raw_sidecars(ctx.raw_root, run_id)

    return FastChemTestCase(
        run_id=run_id,
        pressure_bar=np.asarray(pressure_bar, dtype=np.float64),
        temperature_k=np.asarray(temperature_k, dtype=np.float64),
        global_inputs=global_map,
        stored_target_ymix=np.asarray(stored, dtype=np.float64),
        output_species=list(ctx.contract["output_species_order"]),
        raw_globals=raw_globals,
        raw_metadata=raw_metadata,
    )


def load_raw_equilibrium_profile(raw_root: Path, run_id: str) -> RawEquilibriumProfile:
    """Load one raw-equilibrium FastChem profile from ``runs.h5``."""

    import h5py

    with h5py.File(raw_root / "runs.h5", "r") as handle:
        group = handle[run_id]
        pressure_bar = np.asarray(group["inputs/pressure_bar"], dtype=np.float64)
        temperature_k = np.asarray(group["inputs/temperature_k"], dtype=np.float64)
        output_species = _decode_labels(np.asarray(group["inputs/output_species"]))
        equilibrium_ymix = np.asarray(group["equilibrium/ymix"], dtype=np.float64)
        globals_map = {
            key: float(np.asarray(group[f"globals/{key}"]))
            for key in group["globals"].keys()
        }
        if "inputs/element_input_order" in group and "inputs/elemental_abundances_frac" in group:
            labels = _decode_labels(np.asarray(group["inputs/element_input_order"]))
            profile = np.asarray(group["inputs/elemental_abundances_frac"], dtype=np.float64)
            for idx, label in enumerate(labels):
                globals_map[label] = float(profile[0, idx])
    return RawEquilibriumProfile(
        run_id=run_id,
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        equilibrium_ymix=equilibrium_ymix,
        output_species=output_species,
        globals=globals_map,
    )


def resolve_vulcan_source_root(config: Mapping[str, Any], *, project_root: Path) -> Path:
    """Resolve ``paths.vulcan_source_root`` relative to the notebook project root."""

    paths = config.get("paths", {})
    root = paths.get("vulcan_source_root") if isinstance(paths, Mapping) else None
    if not isinstance(root, str) or not root:
        raise KeyError("Bundle config missing paths.vulcan_source_root.")
    return resolve_path(root, project_root)


def chemsetup_matched_to_fastchem(fastchem_source_root: Path, *, silent: bool = True) -> Any:
    """Return an ExoGibbs ``ChemicalSetup`` pinned to the same ``logK`` file VULCAN-FastChem ships.

    Notebook comparisons use this helper instead of the bare
    ``exogibbs.presets.fastchem.chemsetup()`` so the classical-reference
    thermochemistry is read from the same file tree FastChem itself uses at
    training time — eliminating the ion-species difference (FastChem runs
    ``parameters_wo_ion.dat``; ExoGibbs' default ``logK.dat`` carries ions)
    and pinning both codes to the same set of species-level 5-term fits
    for the ~330 gas species they share.

    A residual thermodynamic-fit difference remains because VULCAN-FastChem
    is built against the NASA-9 polynomial form in
    ``nasa9_logK_SNCHOPTi.dat``; switching FastChem to the 5-term file
    would require rebuilding the trained emulator, which is out of scope
    for notebook comparisons.
    """

    logk_path = fastchem_source_root / "fastchem_vulcan" / "input" / "logK_wo_ions.dat"
    if not logk_path.is_file():
        raise FileNotFoundError(
            f"VULCAN FastChem thermo file not found: {logk_path}."
        )

    from exogibbs.io import load_data as _load_data
    from exogibbs.presets import fastchem as _fc
    from exogibbs.presets.fastchem import chemsetup

    abs_str = str(logk_path)
    original_io = _load_data.get_data_filepath
    original_fc = _fc.get_data_filepath

    def _patched(filename):
        # Pass absolute paths through untouched; route package-relative
        # paths (e.g. element files consulted by ``chemsetup``) to the
        # installed ExoGibbs data directory as usual.
        if str(filename) == abs_str:
            return logk_path
        return original_io(filename)

    _load_data.get_data_filepath = _patched
    _fc.get_data_filepath = _patched
    try:
        return chemsetup(path=abs_str, silent=silent)
    finally:
        _load_data.get_data_filepath = original_io
        _fc.get_data_filepath = original_fc


def _sulfur_enabled(config: Mapping[str, Any]) -> bool:
    spec = config.get("data_spec", {})
    names = list(spec.get("state_species", [])) + list(spec.get("output_species", []))
    return any("S" in name for name in names)


def _explicit_elements(config: Mapping[str, Any]) -> set[str]:
    atoms = {"O", "C", "N", "He"}
    if _sulfur_enabled(config):
        atoms.add("S")
    return atoms


def _element_abundances(globals_map: Mapping[str, float]) -> dict[str, float]:
    return {
        key: float(globals_map[key])
        for key in ("He_H", "C_H", "O_H", "N_H", "S_H")
    }


def _write_tp(
    fastchem_root: Path,
    pressure_bar: np.ndarray,
    temperature_k: np.ndarray,
) -> None:
    tp_dir = fastchem_root / "input" / "vulcan_TP"
    tp_dir.mkdir(parents=True, exist_ok=True)
    with (tp_dir / "vulcan_TP.dat").open("w", encoding="utf-8") as handle:
        handle.write("#p (bar)    T (K)\n")
        for pressure, temperature in zip(pressure_bar, temperature_k):
            handle.write(f"{pressure:.8e}\t{temperature:.8f}\n")


def _write_abundances(
    fastchem_root: Path,
    globals_map: Mapping[str, float],
    config: Mapping[str, Any],
) -> None:
    input_dir = fastchem_root / "input"
    params_name = (
        "parameters_ion.dat"
        if config.get("physics_toggles", {}).get("use_ion_chemistry")
        else "parameters_wo_ion.dat"
    )
    (input_dir / "parameters.dat").write_text(
        (input_dir / params_name).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    abundances = _element_abundances(globals_map)
    explicit = _explicit_elements(config)
    met_offset = compute_fastchem_metallicity_offset_dex(abundances)
    solar_file = input_dir / "solar_element_abundances.dat"

    output_lines: list[str] = []
    for line in solar_file.read_text(encoding="utf-8").splitlines(keepends=True):
        if not line.strip() or line.startswith("#"):
            output_lines.append(line)
            continue
        parts = line.split()
        species_name = parts[0]
        if species_name in explicit:
            key = "He_H" if species_name == "He" else f"{species_name}_H"
            output_lines.append(
                f"{species_name}\t{np.log10(abundances[key]) + 12.0:.4f}\n"
            )
        elif species_name in FASTCHEM_METALLICITY_SCALED_ELEMENTS:
            output_lines.append(f"{species_name}\t{float(parts[1]) + met_offset:.4f}\n")
        else:
            output_lines.append(line)
    (input_dir / "element_abundances_vulcan.dat").write_text(
        "".join(output_lines),
        encoding="utf-8",
    )


def _load_fastchem_output(output_path: Path, species: Sequence[str]) -> np.ndarray:
    payload = np.genfromtxt(output_path, names=True, dtype=None, encoding=None)
    rows = np.atleast_1d(payload)
    available = set(rows.dtype.names or ())
    columns = []
    for name in species:
        aliases = FASTCHEM_TO_EXOGIBBS_SPECIES_ALIASES.get(str(name), (str(name),))
        resolved = next((alias for alias in (name, *aliases) if alias in available), None)
        if resolved is None:
            raise KeyError(
                f"FastChem output {output_path} has no column for species {name!r} "
                f"(tried aliases {aliases!r})."
            )
        columns.append(np.asarray(rows[resolved], dtype=np.float64))
    return np.column_stack(columns)


def _save_fastchem_debug_artifacts(
    fastchem_root: Path,
    debug_dir: Path,
    *,
    stdout_text: str,
) -> None:
    """Persist the minimal FastChem runtime files needed for postmortem debugging."""

    debug_dir.mkdir(parents=True, exist_ok=True)
    artifact_map = {
        fastchem_root / "input" / "config.input": debug_dir / "config.input",
        fastchem_root / "input" / "parameters.dat": debug_dir / "parameters.dat",
        fastchem_root / "input" / "element_abundances_vulcan.dat": debug_dir / "element_abundances_vulcan.dat",
        fastchem_root / "input" / "vulcan_TP" / "vulcan_TP.dat": debug_dir / "vulcan_TP.dat",
        fastchem_root / "output" / "vulcan_EQ.dat": debug_dir / "vulcan_EQ.dat",
        fastchem_root / "output" / "monitor_output.dat": debug_dir / "monitor_output.dat",
    }
    for source, destination in artifact_map.items():
        if source.exists():
            shutil.copy2(source, destination)
    (debug_dir / "stdout.txt").write_text(stdout_text, encoding="utf-8")


def read_fastchem_monitor_fail_mask(monitor_path: Path) -> np.ndarray | None:
    """Parse ``monitor_output.dat`` and return a boolean fail-mask of shape (nz,).

    A level is flagged ``True`` when *any* element's per-row status in the
    FastChem monitor log is ``"fail"`` — this is how FastChem signals that
    its inner Newton step did not satisfy that element's mass-balance
    constraint, even when the outer iteration reports ``c_convergence=ok``.
    Notebooks use this mask to exclude non-converged levels from
    FastChem↔ExoGibbs comparison metrics.

    Returns ``None`` when ``monitor_path`` does not exist (e.g. when the
    monitor file was not emitted).
    """

    if not monitor_path.exists():
        return None
    text = monitor_path.read_text(encoding="utf-8").splitlines()
    if not text:
        return None
    # Per-element status fields start at column index 8 in the monitor
    # schema (grid_point, c_iter, c_conv, P, T, n_tot, n_g, m, <elements…>).
    flags: list[bool] = []
    for line in text[1:]:
        parts = line.split()
        if not parts:
            continue
        flags.append(any(status == "fail" for status in parts[8:]))
    if not flags:
        return None
    return np.asarray(flags, dtype=bool)


def run_fastchem_online(
    source_root: Path,
    pressure_bar: np.ndarray,
    temperature_k: np.ndarray,
    globals_map: Mapping[str, float],
    output_species: Sequence[str],
    config: Mapping[str, Any],
    *,
    debug_dir: Path | None = None,
    return_fail_mask: bool = False,
    raise_on_fail: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray | None]:
    """Rerun FastChem with the same abundance-file logic as data generation.

    Parameters
    ----------
    debug_dir : Path | None, optional
        When provided, save the exact FastChem input and output text files for
        this rerun under ``debug_dir`` before the temporary runtime is removed.
    return_fail_mask : bool, optional
        When ``True`` the return value becomes ``(vmr, fail_mask)`` where
        ``fail_mask`` is the boolean array produced by
        :func:`read_fastchem_monitor_fail_mask` (or ``None`` if the monitor
        log is not available). Notebooks use this to drop non-converged
        levels from comparison metrics. Default ``False`` preserves the
        previous VMR-only return type.
    raise_on_fail : bool, optional
        When ``True`` this call raises ``RuntimeError`` if any level is
        flagged by the monitor log. Default ``False``. Orthogonal to
        ``return_fail_mask``.
    """

    from ..data_generation.generation import _copy_fastchem_runtime

    with tempfile.TemporaryDirectory(prefix="fastchem_compare_") as tmp:
        fastchem_root = _copy_fastchem_runtime(source_root, Path(tmp))
        _write_abundances(fastchem_root, globals_map, config)
        _write_tp(fastchem_root, pressure_bar, temperature_k)
        result = subprocess.run(
            ["./fastchem", "input/config.input"],
            cwd=fastchem_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if debug_dir is not None:
            _save_fastchem_debug_artifacts(
                fastchem_root,
                debug_dir,
                stdout_text=result.stdout,
            )
        if result.returncode != 0:
            raise RuntimeError(f"FastChem failed:\n{result.stdout}")
        vmr = _load_fastchem_output(
            fastchem_root / "output" / "vulcan_EQ.dat",
            output_species,
        )
        if raise_on_fail or return_fail_mask:
            fail_mask = read_fastchem_monitor_fail_mask(
                fastchem_root / "output" / "monitor_output.dat"
            )
            if raise_on_fail and fail_mask is not None and bool(fail_mask.any()):
                n_fail = int(fail_mask.sum())
                raise RuntimeError(
                    f"FastChem monitor reports non-converged levels on {n_fail} "
                    f"of {fail_mask.size} rows. Pass raise_on_fail=False and "
                    "inspect the fail_mask (return_fail_mask=True) to see which "
                    "levels were flagged."
                )
            if return_fail_mask:
                return vmr, fail_mask
        return vmr
