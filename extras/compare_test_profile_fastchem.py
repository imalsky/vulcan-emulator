"""Compare one processed test profile against a fresh FastChem rerun.

Usage:
    python extras/compare_test_profile_fastchem.py
    python extras/compare_test_profile_fastchem.py --run-id run_00042
    python extras/compare_test_profile_fastchem.py --config config/fastchem_mlp_config.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from src.data_generation.generation import _copy_fastchem_runtime  # noqa: E402
from src.utils.config import dataset_run_root, load_and_validate_config  # noqa: E402
from src.utils.helpers import resolve_path, resolve_project_root  # noqa: E402

_STYLE = _ROOT / "extras" / "science.mplstyle"
_DEFAULT_CONFIG = _ROOT / "config" / "fastchem_mlp_config.json"

_SOLAR_ELEMENT_ABUNDANCES: dict[str, float] = {
    "O_H": 5.37e-4,
    "C_H": 2.95e-4,
    "N_H": 7.08e-5,
    "S_H": 1.41e-5,
    "He_H": 8.38e-2,
}

_FASTCHEM_METALLICITY_SCALED_ELEMENTS: set[str] = {
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
class RawEquilibriumProfile:
    """One raw equilibrium run in physical units."""

    run_id: str
    pressure_bar: np.ndarray
    temperature_k: np.ndarray
    equilibrium_ymix: np.ndarray
    output_species: list[str]
    globals: dict[str, float]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the FastChem comparison utility."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="Path to a FastChem config file.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run ID from the processed test split.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for the saved comparison figure.",
    )
    return parser.parse_args(argv)


def _decode_labels(values: np.ndarray) -> list[str]:
    """Decode an HDF5 string array into Python strings."""
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values]


def _load_test_run_ids(run_root: Path) -> list[str]:
    """Load processed test-split run IDs from the canonical dataset layout."""
    run_ids_path = run_root / "test" / "run_ids.json"
    if not run_ids_path.exists():
        raise FileNotFoundError(f"Processed test run IDs not found: {run_ids_path}")
    return list(json.loads(run_ids_path.read_text(encoding="utf-8")))


def _extract_raw_profile(handle: h5py.Group, run_id: str) -> RawEquilibriumProfile:
    """Extract one raw equilibrium profile from an open HDF5 group."""
    pressure_bar = np.asarray(handle["inputs/pressure_bar"], dtype=np.float64)
    temperature_k = np.asarray(handle["inputs/temperature_k"], dtype=np.float64)
    output_species = _decode_labels(np.asarray(handle["inputs/output_species"]))
    equilibrium_ymix = np.asarray(handle["equilibrium/ymix"], dtype=np.float64)

    globals_map = {
        key: float(np.asarray(handle[f"globals/{key}"]))
        for key in handle["globals"].keys()
    }
    if "inputs/element_input_order" in handle and "inputs/elemental_abundances_x_h" in handle:
        element_labels = _decode_labels(np.asarray(handle["inputs/element_input_order"]))
        element_profile = np.asarray(handle["inputs/elemental_abundances_x_h"], dtype=np.float64)
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
    """Load one raw equilibrium profile from the consolidated ``runs.h5`` file."""
    consolidated_path = raw_root / "runs.h5"
    if not consolidated_path.exists():
        raise FileNotFoundError(f"Consolidated raw runs file not found: {consolidated_path}")
    with h5py.File(consolidated_path, "r") as handle:
        if run_id not in handle:
            raise KeyError(f"Run ID {run_id!r} not found in {consolidated_path}")
        return _extract_raw_profile(handle[run_id], run_id)


def _select_run_id(run_ids: list[str], *, run_id: str | None) -> str:
    """Select an explicit or random run ID from the processed test split."""
    if run_id is not None:
        if run_id not in run_ids:
            raise KeyError(f"Requested run ID {run_id!r} is not present in the processed test split.")
        return run_id
    rng = np.random.default_rng()
    return str(run_ids[int(rng.integers(0, len(run_ids)))])


def _element_abundances_from_globals(globals_map: dict[str, float]) -> dict[str, float]:
    """Map profile globals to the elemental abundances expected by FastChem."""
    explicit_keys = ("He_H", "C_H", "O_H", "N_H", "S_H")
    missing = [key for key in explicit_keys if key not in globals_map]
    if missing:
        raise ValueError(
            "FastChem comparison requires explicit elemental abundances "
            f"{explicit_keys}, missing {missing}."
        )
    oxygen_h = float(globals_map["O_H"])
    return {
        "He_H": float(globals_map["He_H"]),
        "C_H": float(globals_map["C_H"]),
        "O_H": oxygen_h,
        "N_H": float(globals_map["N_H"]),
        "S_H": float(globals_map["S_H"]),
        "fastchem_met_scale": float(oxygen_h / _SOLAR_ELEMENT_ABUNDANCES["O_H"]),
    }


def _write_fastchem_tp_profile(
    fastchem_root: Path,
    *,
    pressure_bar: np.ndarray,
    temperature_k: np.ndarray,
) -> Path:
    """Write a FastChem TP profile file for the selected test run."""
    tp_dir = fastchem_root / "input" / "vulcan_TP"
    tp_dir.mkdir(parents=True, exist_ok=True)
    tp_path = tp_dir / "vulcan_TP.dat"
    with tp_path.open("w", encoding="utf-8") as handle:
        handle.write("#p (bar)    T (K)\n")
        for pressure_value, temperature_value in zip(pressure_bar, temperature_k):
            handle.write(f"{pressure_value:.3e}\t{temperature_value:.1f}\n")
    return tp_path


def _write_fastchem_element_abundances(
    fastchem_root: Path,
    *,
    globals_map: dict[str, float],
) -> Path:
    """Write FastChem elemental abundances using the repository's ``X/H`` globals."""
    input_dir = fastchem_root / "input"
    parameters_src = input_dir / "parameters_wo_ion.dat"
    if not parameters_src.exists():
        raise FileNotFoundError(f"FastChem parameters file not found: {parameters_src}")
    (input_dir / "parameters.dat").write_text(parameters_src.read_text(encoding="utf-8"), encoding="utf-8")

    element_abundances = _element_abundances_from_globals(globals_map)
    metallicity_offset = float(np.log10(element_abundances["fastchem_met_scale"]))

    solar_file = input_dir / "solar_element_abundances.dat"
    if not solar_file.exists():
        raise FileNotFoundError(f"FastChem solar abundance table not found: {solar_file}")

    output_lines: list[str] = []
    for raw_line in solar_file.read_text(encoding="utf-8").splitlines(keepends=True):
        if not raw_line.strip() or raw_line.startswith("#"):
            output_lines.append(raw_line)
            continue
        parts = raw_line.split()
        species_name = parts[0]
        if species_name in {"O", "C", "N", "S"}:
            key = f"{species_name}_H"
            new_value = np.log10(element_abundances[key]) + 12.0
            output_lines.append(f"{species_name}\t{new_value:.4f}\n")
        elif species_name == "He":
            new_value = np.log10(element_abundances["He_H"]) + 12.0
            output_lines.append(f"{species_name}\t{new_value:.4f}\n")
        elif species_name in _FASTCHEM_METALLICITY_SCALED_ELEMENTS:
            new_value = float(parts[1]) + metallicity_offset
            output_lines.append(f"{species_name}\t{new_value:.4f}\n")
        else:
            output_lines.append(raw_line)

    output_path = input_dir / "element_abundances_vulcan.dat"
    output_path.write_text("".join(output_lines), encoding="utf-8")
    return output_path


def _load_fastchem_output(output_path: Path, output_species: list[str]) -> np.ndarray:
    """Load the FastChem equilibrium table in the repository species order."""
    if not output_path.exists():
        raise FileNotFoundError(f"FastChem output file not found: {output_path}")
    payload = np.genfromtxt(output_path, names=True, dtype=None, encoding=None)
    if payload.dtype.names is None:
        raise ValueError(f"FastChem output at {output_path} does not contain a named header.")
    rows = np.atleast_1d(payload)
    missing = [name for name in output_species if name not in rows.dtype.names]
    if missing:
        raise ValueError(f"FastChem output is missing requested species columns: {missing}")
    return np.column_stack([np.asarray(rows[name], dtype=np.float64) for name in output_species])


def _run_fastchem_online(
    *,
    source_root: Path,
    pressure_bar: np.ndarray,
    temperature_k: np.ndarray,
    globals_map: dict[str, float],
    output_species: list[str],
) -> np.ndarray:
    """Rerun the bundled FastChem executable for one selected raw test profile."""
    with tempfile.TemporaryDirectory(prefix="fastchem_compare_") as tmpdir:
        worker_root = Path(tmpdir)
        fastchem_root = _copy_fastchem_runtime(source_root, worker_root)
        _write_fastchem_element_abundances(fastchem_root, globals_map=globals_map)
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
            raise RuntimeError(f"FastChem execution failed.\nCommand output:\n{result.stdout}")
        return _load_fastchem_output(fastchem_root / "output" / "vulcan_EQ.dat", output_species)


def _mixing_ratio_xlim(values: list[np.ndarray]) -> tuple[float, float]:
    """Choose a stable log-scale x-axis range for mixing-ratio profile plots."""
    clipped = [np.clip(array, 1.0e-30, None) for array in values]
    minimum = min(float(np.min(array)) for array in clipped)
    lower = 10.0 ** np.floor(np.log10(max(minimum, 1.0e-30)))
    return lower, 3.0


def _plot_profile_comparison(
    *,
    profile: RawEquilibriumProfile,
    fastchem_ymix: np.ndarray,
    output_path: Path,
) -> None:
    """Render and save a comparison plot between stored and rerun FastChem profiles."""
    if _STYLE.exists():
        plt.style.use(str(_STYLE))

    fig, (ax_pt, ax_mix, ax_delta) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    colors = plt.cm.tab20(np.linspace(0, 1, len(profile.output_species)))
    clipped_truth = np.clip(profile.equilibrium_ymix, 1.0e-30, None)
    clipped_fastchem = np.clip(fastchem_ymix, 1.0e-30, None)

    ax_pt.plot(profile.temperature_k, profile.pressure_bar, color="black", lw=2.0)
    ax_pt.set_xlabel("Temperature [K]")
    ax_pt.set_ylabel("Pressure [bar]")
    ax_pt.set_yscale("log")
    ax_pt.invert_yaxis()
    ax_pt.set_xlim(0.0, 3000.0)
    ax_pt.set_title("Test P-T Profile")

    for species_index, species_name in enumerate(profile.output_species):
        color = colors[species_index]
        ax_mix.plot(clipped_truth[:, species_index], profile.pressure_bar, color=color, lw=1.6, label=species_name)
        ax_mix.plot(clipped_fastchem[:, species_index], profile.pressure_bar, color=color, lw=1.2, ls="--")
        residual = np.log10(clipped_fastchem[:, species_index]) - np.log10(clipped_truth[:, species_index])
        ax_delta.plot(residual, profile.pressure_bar, color=color, lw=1.3)

    x_min, x_max = _mixing_ratio_xlim([clipped_truth, clipped_fastchem])
    ax_mix.set_xscale("log")
    ax_mix.set_xlim(x_min, x_max)
    ax_mix.set_xlabel("Mixing Ratio")
    ax_mix.set_title("Mixing Ratios")

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

    max_abs_delta = float(np.max(np.abs(np.log10(clipped_fastchem) - np.log10(clipped_truth))))
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


def main(argv: list[str] | None = None) -> int:
    """Load one test profile, rerun FastChem, and save a comparison figure."""
    args = _parse_args(argv)
    project_root = resolve_project_root(Path(__file__).resolve())
    config = load_and_validate_config(resolve_path(args.config, project_root))
    if config["chemistry_type"] != "fastchem":
        raise ValueError("compare_test_profile_fastchem.py requires a fastchem config.")

    run_root = resolve_path(dataset_run_root(config), project_root)
    raw_root = run_root / "raw"
    source_root = resolve_path(config["paths"]["vulcan_source_root"], project_root)

    test_run_ids = _load_test_run_ids(run_root)
    selected_run_id = _select_run_id(test_run_ids, run_id=args.run_id)
    profile = _load_raw_equilibrium_profile(raw_root, selected_run_id)
    fastchem_ymix = _run_fastchem_online(
        source_root=source_root,
        pressure_bar=profile.pressure_bar,
        temperature_k=profile.temperature_k,
        globals_map=profile.globals,
        output_species=profile.output_species,
    )
    if fastchem_ymix.shape != profile.equilibrium_ymix.shape:
        raise ValueError(
            "FastChem output shape does not match the test profile shape: "
            f"{fastchem_ymix.shape} vs {profile.equilibrium_ymix.shape}"
        )

    output_path = (
        resolve_path(args.output, project_root)
        if args.output is not None
        else _ROOT / "extras" / "plots" / f"{profile.run_id}_fastchem_compare.png"
    )
    _plot_profile_comparison(profile=profile, fastchem_ymix=fastchem_ymix, output_path=output_path)

    print(f"Selected test run: {profile.run_id}")
    print(f"Saved comparison figure to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
