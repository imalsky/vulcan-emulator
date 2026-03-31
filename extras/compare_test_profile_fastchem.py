from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


_ROOT = Path(__file__).resolve().parents[1]
_STYLE = _ROOT / "extras" / "plots" / "paper.mplstyle"

CONFIG_PATH = _ROOT / "config" / "equilibrium_only_config.json"
PROCESSED_ROOT_OVERRIDE: Path | None = None
RAW_ROOT_OVERRIDE: Path | None = None
RUN_ID: str | None = None
OUTPUT_PATH: Path | None = None

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


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (_ROOT / path).resolve()


def _element_abundances_from_globals(globals_map: dict[str, float]) -> dict[str, float]:
    """Map one profile's global inputs to the elemental abundances used by FastChem."""

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
    """Write the TP profile using VULCAN's embedded FastChem formatting."""

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
    """Write abundances to mirror VULCAN's embedded FastChem setup."""

    input_dir = fastchem_root / "input"
    parameters_src = input_dir / "parameters_wo_ion.dat"
    if not parameters_src.exists():
        raise FileNotFoundError(f"FastChem parameters file not found: {parameters_src}")

    shutil.copyfile(parameters_src, input_dir / "parameters.dat")

    element_abundances = _element_abundances_from_globals(globals_map)
    metallicity_offset = float(np.log10(element_abundances["fastchem_met_scale"]))

    solar_file = input_dir / "solar_element_abundances.dat"
    if not solar_file.exists():
        raise FileNotFoundError(f"FastChem solar abundance table not found: {solar_file}")

    output_lines: list[str] = []
    with solar_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
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
                old_value = float(parts[1])
                new_value = old_value + metallicity_offset
                output_lines.append(f"{species_name}\t{new_value:.4f}\n")
            else:
                output_lines.append(raw_line)

    output_path = input_dir / "element_abundances_vulcan.dat"
    output_path.write_text("".join(output_lines), encoding="utf-8")
    return output_path


def _copy_fastchem_runtime(source_root: Path, worker_root: Path) -> Path:
    """Copy the bundled FastChem runtime into a temporary worker directory."""

    fastchem_src_root = source_root / "fastchem_vulcan"
    if not fastchem_src_root.exists():
        raise FileNotFoundError(f"Bundled FastChem source not found: {fastchem_src_root}")

    fastchem_root = worker_root / "fastchem_vulcan"
    fastchem_root.mkdir(parents=True, exist_ok=True)

    shutil.copy2(fastchem_src_root / "fastchem", fastchem_root / "fastchem")
    shutil.copytree(fastchem_src_root / "input", fastchem_root / "input")
    shutil.copytree(
        fastchem_src_root / "fastchem_src" / "chem_input",
        fastchem_root / "fastchem_src" / "chem_input",
    )
    (fastchem_root / "output").mkdir(parents=True, exist_ok=True)

    return fastchem_root


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
) -> np.ndarray:
    """Rerun the bundled FastChem executable inside the current Python environment."""

    with tempfile.TemporaryDirectory(prefix="fastchem_compare_") as tmpdir:
        worker_root = Path(tmpdir)
        fastchem_root = _copy_fastchem_runtime(source_root, worker_root)

        _write_fastchem_element_abundances(
            fastchem_root,
            globals_map=globals_map,
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
                "FastChem execution failed.\n"
                f"Command output:\n{result.stdout}"
            )

        return _load_fastchem_output(
            fastchem_root / "output" / "vulcan_EQ.dat",
            output_species,
        )


def _load_test_run_ids(processed_root: Path) -> list[str]:
    run_ids_path = processed_root / "test" / "run_ids.json"
    if not run_ids_path.exists():
        raise FileNotFoundError(f"Processed test run IDs not found: {run_ids_path}")
    return list(json.loads(run_ids_path.read_text(encoding="utf-8")))


def _load_raw_equilibrium_profile(raw_root: Path, run_id: str) -> RawEquilibriumProfile:
    """Load one raw equilibrium profile from either consolidated or per-file layout."""

    consolidated_path = raw_root / "runs.h5"
    if consolidated_path.exists():
        with h5py.File(consolidated_path, "r") as handle:
            if run_id not in handle:
                raise KeyError(f"Run ID {run_id!r} not found in {consolidated_path}")
            source = handle[run_id]
            return _extract_raw_profile(source, run_id)

    legacy_path = raw_root / "runs" / f"{run_id}.h5"
    if not legacy_path.exists():
        raise FileNotFoundError(
            f"Run ID {run_id!r} not found in {consolidated_path} or {legacy_path}"
        )

    with h5py.File(legacy_path, "r") as handle:
        return _extract_raw_profile(handle, run_id)


def _extract_raw_profile(handle: h5py.Group, run_id: str) -> RawEquilibriumProfile:
    pressure_bar = np.asarray(handle["inputs/pressure_bar"], dtype=np.float64)
    temperature_k = np.asarray(handle["inputs/temperature_k"], dtype=np.float64)
    output_species = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in np.asarray(handle["inputs/output_species"])
    ]
    equilibrium_ymix = np.asarray(handle["equilibrium/ymix"], dtype=np.float64)
    globals_map = {
        key: float(np.asarray(handle[f"globals/{key}"]))
        for key in handle["globals"].keys()
    }
    if "inputs/elemental_abundances_x_h" in handle and "inputs/element_input_order" in handle:
        element_labels = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in np.asarray(handle["inputs/element_input_order"])
        ]
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


def _select_run_id(run_ids: list[str], *, run_id: str | None) -> str:
    if run_id is not None:
        if run_id not in run_ids:
            raise KeyError(
                f"Requested run ID {run_id!r} is not present in the processed test split."
            )
        return run_id

    rng = np.random.default_rng()
    return str(run_ids[int(rng.integers(0, len(run_ids)))])


def _mixing_ratio_xlim(values: list[np.ndarray]) -> tuple[float, float]:
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

        ax_mix.plot(
            clipped_truth[:, species_index],
            profile.pressure_bar,
            color=color,
            lw=1.6,
            label=species_name,
        )
        ax_mix.plot(
            clipped_fastchem[:, species_index],
            profile.pressure_bar,
            color=color,
            lw=1.2,
            ls="--",
        )

        residual = np.log10(clipped_fastchem[:, species_index]) - np.log10(
            clipped_truth[:, species_index]
        )
        ax_delta.plot(
            residual,
            profile.pressure_bar,
            color=color,
            lw=1.3,
        )

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

    max_abs_delta = float(
        np.max(
            np.abs(
                np.log10(clipped_fastchem) - np.log10(clipped_truth)
            )
        )
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


def main() -> int:
    config_path = CONFIG_PATH.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = _load_json(config_path)

    processed_root = _resolve_path(
        PROCESSED_ROOT_OVERRIDE or config["paths"]["processed_root"]
    )
    raw_root = _resolve_path(
        RAW_ROOT_OVERRIDE or config["paths"]["raw_root"]
    )
    source_root = _resolve_path(config["paths"]["vulcan_source_root"])

    test_run_ids = _load_test_run_ids(processed_root)
    selected_run_id = _select_run_id(test_run_ids, run_id=RUN_ID)
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
        OUTPUT_PATH.resolve()
        if OUTPUT_PATH is not None
        else (_ROOT / "extras" / "plots" / f"{profile.run_id}_fastchem_compare.png")
    )

    _plot_profile_comparison(
        profile=profile,
        fastchem_ymix=fastchem_ymix,
        output_path=output_path,
    )

    print(f"Selected test run: {profile.run_id}")
    print(f"Saved comparison figure to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
