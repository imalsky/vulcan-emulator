"""Compare one saved FastChem test profile against a fresh FastChem rerun.

Picks a random (or user-specified) run from the saved processed test split,
loads the matching stored raw profile, reruns FastChem with the same inputs,
and saves a three-panel comparison figure: P-T profile, mixing ratios, and
log-space residuals.

Usage
-----
    python extras/compare_saved_test_profile_fastchem.py
    python extras/compare_saved_test_profile_fastchem.py --run-id run_00042
    python extras/compare_saved_test_profile_fastchem.py --bundle models/fastchem_transformer/best_exported.npz
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from _common import (
    EPSILON,
    FASTCHEM_METALLICITY_SCALED_ELEMENTS,
    SOLAR_ELEMENT_ABUNDANCES,
    apply_style,
    load_fastchem_test_context,
    plots_dir_for_bundle,
    resolve_bundle_path,
    resolve_vulcan_source_root,
    select_fastchem_test_run_id,
)
from matplotlib.lines import Line2D
from src.data_generation.generation import (
    _copy_fastchem_runtime,
)
from src.models.export_bundle import load_exported_model
from src.utils.helpers import resolve_path, resolve_project_root

# ======================================================================
# Data structures
# ======================================================================

@dataclass(frozen=True)
class RawEquilibriumProfile:
    """One raw equilibrium run stored in physical units."""

    run_id: str
    pressure_bar: np.ndarray
    temperature_k: np.ndarray
    equilibrium_ymix: np.ndarray
    output_species: list[str]
    globals: dict[str, float]


# ======================================================================
# CLI
# ======================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--bundle", default=None,
        help="Path to an exported FastChem transformer bundle.",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Explicit run ID from the saved processed test split.",
    )
    parser.add_argument(
        "--vulcan-source-root", default=None,
        help="VULCAN-master checkout used to rerun FastChem.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for the saved comparison figure.",
    )
    return parser.parse_args(argv)


# ======================================================================
# HDF5 helpers
# ======================================================================

def _decode_labels(values: np.ndarray) -> list[str]:
    """Decode an HDF5 string array into Python strings."""
    return [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in values
    ]


def _extract_raw_profile(handle: h5py.Group, run_id: str) -> RawEquilibriumProfile:
    """Extract one raw equilibrium profile from an open HDF5 group."""
    pressure_bar = np.asarray(handle["inputs/pressure_bar"], dtype=np.float64)
    temperature_k = np.asarray(handle["inputs/temperature_k"], dtype=np.float64)
    output_species = _decode_labels(np.asarray(handle["inputs/output_species"]))
    equilibrium_ymix = np.asarray(handle["equilibrium/ymix"], dtype=np.float64)

    # Collect scalar global parameters.
    globals_map = {
        key: float(np.asarray(handle[f"globals/{key}"]))
        for key in handle["globals"].keys()
    }

    # Override with per-element abundances when stored explicitly.
    if "inputs/element_input_order" in handle and "inputs/elemental_abundances_x_h" in handle:
        element_labels = _decode_labels(np.asarray(handle["inputs/element_input_order"]))
        element_profile = np.asarray(
            handle["inputs/elemental_abundances_x_h"], dtype=np.float64,
        )
        for index, label in enumerate(element_labels):
            globals_map[label] = float(element_profile[0, index])

    return RawEquilibriumProfile(
        run_id=run_id,
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        equilibrium_ymix=equilibrium_ymix,
        output_species=output_species,
        globals=globals_map,
    )

def _load_raw_equilibrium_profile(raw_root: Path, run_id: str) -> RawEquilibriumProfile:
    """Load one raw equilibrium profile from the consolidated HDF5 dataset."""
    consolidated_path = raw_root / "runs.h5"
    if not consolidated_path.exists():
        raise FileNotFoundError(f"Consolidated raw dataset not found: {consolidated_path}")
    with h5py.File(consolidated_path, "r") as handle:
        if run_id not in handle:
            raise KeyError(f"Run ID {run_id!r} not found in {consolidated_path}")
        return _extract_raw_profile(handle[run_id], run_id)


# ======================================================================
# FastChem execution
# ======================================================================

def _element_abundances_from_globals(globals_map: dict[str, float]) -> dict[str, float]:
    """Map profile globals to elemental abundances expected by FastChem.

    When the stored globals include ``metallicity_log10`` (the original
    sampled value), metallicity is recovered directly so that the
    ``fastchem_met_scale`` is bit-identical to the value used during data
    generation.  Falling back to ``O_H / solar_O_H`` would introduce a
    float-precision round-trip error that can shift background-element
    abundances by ~0.0001 dex — enough to perturb equilibrium near
    chemical transitions.
    """
    required = ("He_H", "C_H", "O_H", "N_H", "S_H")
    missing = [k for k in required if k not in globals_map]
    if missing:
        raise ValueError(
            f"FastChem comparison requires elemental abundances {required}, missing {missing}."
        )
    # Prefer the original sampled metallicity to avoid O_H round-trip error.
    if "metallicity_log10" in globals_map:
        met_scale = 10.0 ** float(globals_map["metallicity_log10"])
    else:
        met_scale = float(globals_map["O_H"]) / SOLAR_ELEMENT_ABUNDANCES["O_H"]
    return {
        "He_H": float(globals_map["He_H"]),
        "C_H": float(globals_map["C_H"]),
        "O_H": float(globals_map["O_H"]),
        "N_H": float(globals_map["N_H"]),
        "S_H": float(globals_map["S_H"]),
        "fastchem_met_scale": met_scale,
    }


def _write_fastchem_tp_profile(
    fastchem_root: Path,
    *,
    pressure_bar: np.ndarray,
    temperature_k: np.ndarray,
) -> Path:
    """Write a FastChem-format T-P profile for the selected run."""
    tp_dir = fastchem_root / "input" / "vulcan_TP"
    tp_dir.mkdir(parents=True, exist_ok=True)
    tp_path = tp_dir / "vulcan_TP.dat"
    with tp_path.open("w", encoding="utf-8") as fh:
        fh.write("#p (bar)    T (K)\n")
        for p, t in zip(pressure_bar, temperature_k):
            fh.write(f"{p:.3e}\t{t:.1f}\n")
    return tp_path


def _sulfur_enabled(config: dict) -> bool:
    """Return whether the configured species lists require sulfur chemistry.

    Mirrors the identical check in ``src.data_generation.generation`` so that
    the comparison rerun partitions elements (explicit X/H vs metallicity-
    scaled) exactly as the original data-generation run did.
    """
    data_spec = config.get("data_spec", {})
    species = list(data_spec.get("state_species", [])) + list(data_spec.get("output_species", []))
    return any("S" in name for name in species)


def _explicit_element_set(config: dict) -> set[str]:
    """Build the set of non-H atoms written with explicit X/H abundances.

    This must match ``_vulcan_atom_list`` in the generation module; elements
    NOT in this set are instead written via the metallicity-offset path.
    If the set differs between generation and comparison, FastChem receives
    slightly different input files and the equilibrium output diverges.
    """
    atoms = {"O", "C", "N", "He"}
    if _sulfur_enabled(config):
        atoms.add("S")
    return atoms


def _write_fastchem_element_abundances(
    fastchem_root: Path,
    *,
    globals_map: dict[str, float],
    config: dict,
) -> Path:
    """Write FastChem elemental abundances derived from the X/H globals.

    Parameters
    ----------
    fastchem_root : Path
        Temporary FastChem working directory.
    globals_map : dict
        Stored run globals including elemental abundances and, when
        available, ``metallicity_log10``.
    config : dict
        Full pipeline config (from the exported bundle) used to decide
        which elements get explicit X/H values vs metallicity scaling.
    """
    input_dir = fastchem_root / "input"

    # Copy the parameters file matching the original generation config.
    physics = config.get("physics_toggles", {})
    use_ion = bool(physics.get("use_ion_chemistry", False))
    parameters_name = "parameters_ion.dat" if use_ion else "parameters_wo_ion.dat"
    parameters_src = input_dir / parameters_name
    if not parameters_src.exists():
        raise FileNotFoundError(f"FastChem parameters file not found: {parameters_src}")
    (input_dir / "parameters.dat").write_text(
        parameters_src.read_text(encoding="utf-8"), encoding="utf-8",
    )

    element_abundances = _element_abundances_from_globals(globals_map)
    metallicity_offset = float(np.log10(element_abundances["fastchem_met_scale"]))

    # Determine which elements get explicit X/H values (the rest are
    # metallicity-scaled) — must mirror the generation's _vulcan_atom_list.
    explicit_atoms = _explicit_element_set(config)

    solar_file = input_dir / "solar_element_abundances.dat"
    if not solar_file.exists():
        raise FileNotFoundError(f"FastChem solar abundance table not found: {solar_file}")

    # Patch solar abundances with run-specific values.
    output_lines: list[str] = []
    for raw_line in solar_file.read_text(encoding="utf-8").splitlines(keepends=True):
        if not raw_line.strip() or raw_line.startswith("#"):
            output_lines.append(raw_line)
            continue
        parts = raw_line.split()
        species_name = parts[0]
        if species_name in explicit_atoms:
            # Write with the exact stored X/H value, matching generation.
            key = "He_H" if species_name == "He" else f"{species_name}_H"
            abundance_h = element_abundances.get(key)
            if abundance_h is None:
                raise ValueError(f"Missing elemental abundance for {species_name}.")
            new_value = np.log10(float(abundance_h)) + 12.0
            output_lines.append(f"{species_name}\t{new_value:.4f}\n")
        elif species_name in FASTCHEM_METALLICITY_SCALED_ELEMENTS:
            # Scale from solar using the metallicity offset.
            new_value = float(parts[1]) + metallicity_offset
            output_lines.append(f"{species_name}\t{new_value:.4f}\n")
        else:
            output_lines.append(raw_line)

    output_path = input_dir / "element_abundances_vulcan.dat"
    output_path.write_text("".join(output_lines), encoding="utf-8")
    return output_path


def _load_fastchem_output(output_path: Path, output_species: list[str]) -> np.ndarray:
    """Load the FastChem equilibrium table, ordered by repository species list."""
    if not output_path.exists():
        raise FileNotFoundError(f"FastChem output file not found: {output_path}")
    payload = np.genfromtxt(output_path, names=True, dtype=None, encoding=None)
    if payload.dtype.names is None:
        raise ValueError(f"FastChem output at {output_path} has no named header.")
    rows = np.atleast_1d(payload)
    missing = [name for name in output_species if name not in rows.dtype.names]
    if missing:
        raise ValueError(f"FastChem output is missing species columns: {missing}")
    return np.column_stack(
        [np.asarray(rows[name], dtype=np.float64) for name in output_species]
    )


def _run_fastchem_online(
    *,
    source_root: Path,
    pressure_bar: np.ndarray,
    temperature_k: np.ndarray,
    globals_map: dict[str, float],
    output_species: list[str],
    config: dict,
) -> np.ndarray:
    """Rerun the bundled FastChem executable for one raw test profile.

    Parameters
    ----------
    source_root : Path
        VULCAN-master source tree containing the FastChem binary.
    pressure_bar, temperature_k : np.ndarray
        1-D physical arrays defining the atmospheric column.
    globals_map : dict
        Elemental abundance globals (X/H keys), plus ``metallicity_log10``
        when available for exact metallicity recovery.
    output_species : list[str]
        Species names whose mixing ratios are returned.
    config : dict
        Full pipeline config from the exported bundle, used to select
        ion chemistry and determine the explicit-element set.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(n_levels, n_species)`` with linear mixing ratios.
    """
    with tempfile.TemporaryDirectory(prefix="fastchem_compare_") as tmpdir:
        worker_root = Path(tmpdir)
        fastchem_root = _copy_fastchem_runtime(source_root, worker_root)
        _write_fastchem_element_abundances(
            fastchem_root, globals_map=globals_map, config=config,
        )
        _write_fastchem_tp_profile(
            fastchem_root,
            pressure_bar=pressure_bar,
            temperature_k=temperature_k,
        )
        result = subprocess.run(
            ["./fastchem", "input/config.input"],
            cwd=fastchem_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"FastChem execution failed.\nCommand output:\n{result.stdout}"
            )
        return _load_fastchem_output(
            fastchem_root / "output" / "vulcan_EQ.dat", output_species,
        )


# ======================================================================
# Plotting
# ======================================================================

def _mixing_ratio_xlim(values: list[np.ndarray]) -> tuple[float, float]:
    """Choose a stable log-scale x-axis range for mixing-ratio panels."""
    clipped = [np.clip(arr, EPSILON, None) for arr in values]
    minimum = min(float(np.min(arr)) for arr in clipped)
    lower = 10.0 ** np.floor(np.log10(max(minimum, EPSILON)))
    return lower, 3.0


def _plot_profile_comparison(
    *,
    profile: RawEquilibriumProfile,
    fastchem_ymix: np.ndarray,
    output_path: Path,
) -> None:
    """Render and save a three-panel comparison figure.

    Panels: P-T profile | mixing ratios (stored vs rerun) | log residuals.
    """
    apply_style()

    fig, (ax_pt, ax_mix, ax_delta) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    colors = plt.cm.tab20(np.linspace(0, 1, len(profile.output_species)))

    # Clip to EPSILON so log10 never sees zero.
    clipped_truth = np.clip(profile.equilibrium_ymix, EPSILON, None)
    clipped_fastchem = np.clip(fastchem_ymix, EPSILON, None)

    # --- Panel 1: P-T profile ---
    ax_pt.plot(profile.temperature_k, profile.pressure_bar, color="black", lw=2.0)
    ax_pt.set_xlabel("Temperature [K]")
    ax_pt.set_ylabel("Pressure [bar]")
    ax_pt.set_yscale("log")
    ax_pt.invert_yaxis()
    ax_pt.set_xlim(0.0, 3000.0)
    ax_pt.set_title("Test P-T Profile")

    # --- Panels 2 & 3: mixing ratios and residuals ---
    for i, name in enumerate(profile.output_species):
        c = colors[i]
        ax_mix.plot(clipped_truth[:, i], profile.pressure_bar, color=c, lw=1.6, label=name)
        ax_mix.plot(clipped_fastchem[:, i], profile.pressure_bar, color=c, lw=1.2, ls="--")

        # Residual in dex with epsilon floor to suppress numerical noise.
        residual = np.log10(clipped_fastchem[:, i] + EPSILON) - np.log10(clipped_truth[:, i] + EPSILON)
        ax_delta.plot(residual, profile.pressure_bar, color=c, lw=1.3)

    x_min, x_max = _mixing_ratio_xlim([clipped_truth, clipped_fastchem])
    ax_mix.set_xscale("log")
    ax_mix.set_xlim(x_min, x_max)
    ax_mix.set_xlabel("Mixing Ratio")
    ax_mix.set_title("Mixing Ratios")

    # Two-legend layout: species labels + solid/dashed key.
    species_legend = ax_mix.legend(fontsize=7, ncol=3, loc="lower left")
    ax_mix.add_artist(species_legend)
    ax_mix.legend(
        handles=[
            Line2D([0], [0], color="black", lw=1.6, label="Test"),
            Line2D([0], [0], color="black", lw=1.2, ls="--", label="FastChem"),
        ],
        fontsize=8,
        loc="upper left",
    )

    # Symmetric residual axis.
    max_abs_delta = float(
        np.max(np.abs(
            np.log10(clipped_fastchem + EPSILON) - np.log10(clipped_truth + EPSILON)
        ))
    )
    delta_limit = max(0.5, np.ceil(max_abs_delta))
    ax_delta.axvline(0.0, color="black", lw=1.0, alpha=0.6)
    ax_delta.set_xlim(-delta_limit, delta_limit)
    ax_delta.set_xlabel(r"$\log_{10}(\mathrm{FastChem}) - \log_{10}(\mathrm{Test})$")
    ax_delta.set_title("FastChem Residual")

    fig.suptitle(
        (
            f"{profile.run_id}   He/H={profile.globals['He_H']:.3e}, "
            f"C/H={profile.globals['C_H']:.3e}, O/H={profile.globals['O_H']:.3e}, "
            f"N/H={profile.globals['N_H']:.3e}, S/H={profile.globals['S_H']:.3e}"
        ),
        fontsize=12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================

def main(argv: list[str] | None = None) -> int:
    """Load one raw profile, rerun FastChem, and save a comparison figure."""
    args = _parse_args(argv)
    project_root = resolve_project_root(Path(__file__).resolve())
    bundle_path = resolve_bundle_path(project_root, args.bundle)
    model = load_exported_model(bundle_path)
    if not model.uses_fastchem or not model.uses_transformer:
        raise RuntimeError(f"Expected a FastChem transformer bundle, got {bundle_path}.")

    context = load_fastchem_test_context(
        project_root,
        bundle_path=bundle_path,
        config=model.config,
        require_raw=True,
    )
    assert context.raw_root is not None

    selected_run_id = select_fastchem_test_run_id(
        context,
        run_id=args.run_id,
        require_raw=True,
    )
    profile = _load_raw_equilibrium_profile(context.raw_root, selected_run_id)

    # Rerun FastChem online.
    source_root = resolve_vulcan_source_root(
        project_root, config=model.config, explicit_root=args.vulcan_source_root,
    )
    fastchem_ymix = _run_fastchem_online(
        source_root=source_root,
        pressure_bar=profile.pressure_bar,
        temperature_k=profile.temperature_k,
        globals_map=profile.globals,
        output_species=profile.output_species,
        config=model.config,
    )
    if fastchem_ymix.shape != profile.equilibrium_ymix.shape:
        raise ValueError(
            "FastChem output shape mismatch: "
            f"{fastchem_ymix.shape} vs {profile.equilibrium_ymix.shape}"
        )

    # Save comparison figure to the model's plots directory.
    output_path = (
        resolve_path(args.output, project_root)
        if args.output is not None
        else plots_dir_for_bundle(bundle_path) / f"{profile.run_id}_fastchem_compare.png"
    )
    _plot_profile_comparison(
        profile=profile, fastchem_ymix=fastchem_ymix, output_path=output_path,
    )

    print(f"Bundle path      : {bundle_path}")
    print(f"Processed root   : {context.processed_root}")
    print(f"Raw dataset root : {context.raw_root}")
    print(f"Selected run     : {profile.run_id}")
    print(f"Saved figure     : {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
