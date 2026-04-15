"""Debug the notebook chemistry mismatch against live FastChem and ExoGibbs.

This script is designed to be run in the ``vulcan`` conda environment so the
runtime matches the working notebook environment.

Usage
-----
    conda run -n vulcan python extras/debug_notebook_chemistry_mismatch.py
    conda run -n vulcan python extras/debug_notebook_chemistry_mismatch.py --run-id run_00042
"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import sys
from typing import Any

import numpy as np
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # noqa: E402
    from extras._common import (
        EPSILON,
        FastChemTestCase,
        SOLAR_ELEMENT_ABUNDANCES,
        load_fastchem_test_case,
        load_fastchem_test_context,
        resolve_bundle_path,
        resolve_vulcan_source_root,
        select_fastchem_test_run_id,
    )
    from extras.compare_saved_test_profile_fastchem import _run_fastchem_online
except ImportError:  # noqa: E402
    from _common import (
        EPSILON,
        FastChemTestCase,
        SOLAR_ELEMENT_ABUNDANCES,
        load_fastchem_test_case,
        load_fastchem_test_context,
        resolve_bundle_path,
        resolve_vulcan_source_root,
        select_fastchem_test_run_id,
    )
    from compare_saved_test_profile_fastchem import _run_fastchem_online
from exogibbs.api import (  # noqa: E402
    get_default_equilibrium_grid_path,
    load_equilibrium_grid_netcdf,
)
from exogibbs.api.equilibrium import (  # noqa: E402
    EquilibriumOptions,
    GridEquilibriumInitializer,
    equilibrium_profile,
)
from exogibbs.presets.fastchem import chemsetup  # noqa: E402
from src.models.export_bundle import (  # noqa: E402
    _predict_fastchem_profile_impl,
    load_exported_model,
)
from src.utils.helpers import resolve_project_root  # noqa: E402

POWERLAW_T0 = 1200.0
POWERLAW_ALPHA = 0.1
PHOTOSPHERE_PRESSURE_BAR = 0.1
NOTEBOOK_PRESSURE_BOTTOM_BAR = 1.0e1
NOTEBOOK_PRESSURE_TOP_BAR = 1.0e-5
NOTEBOOK_NUM_LEVELS = 100
THRESHOLDS = (0.0, 1.0e-12)
EXOGIBBS_SPECIES_MAP = {
    "H2": "H2",
    "He": "He1",
    "H": "H1",
    "O": "O1",
    "OH": "H1O1",
    "H2O": "H2O1",
    "CO": "C1O1",
    "CO2": "C1O2",
    "CH4": "C1H4",
    "N2": "N2",
    "NH3": "H3N1",
    "H2S": "H2S1",
    "SH": "H1S1",
    "S": "S1",
    "SO": "O1S1",
    "SO2": "O2S1",
    "S2": "S2",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        default=None,
        help="Path to an exported FastChem transformer bundle.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit saved processed test run ID.",
    )
    parser.add_argument(
        "--vulcan-source-root",
        default=None,
        help="Optional explicit VULCAN-master checkout.",
    )
    return parser.parse_args(argv)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _count_fastchem_species_entries(path: Path) -> int:
    count = 0
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue
        if ":" in line and index + 1 < len(lines):
            count += 1
            index += 2
            continue
        index += 1
    return count


def _powerlaw_temperature_from_pressure(pressure_bar: np.ndarray) -> np.ndarray:
    pressure = np.asarray(pressure_bar, dtype=np.float64)
    return POWERLAW_T0 * pressure**POWERLAW_ALPHA


def _build_profile(
    *,
    num_levels: int,
    pressure_bottom_bar: float,
    pressure_top_bar: float,
) -> tuple[np.ndarray, np.ndarray]:
    pressure_bar = np.logspace(
        np.log10(pressure_bottom_bar),
        np.log10(pressure_top_bar),
        int(num_levels),
        dtype=np.float64,
    )
    return pressure_bar, _powerlaw_temperature_from_pressure(pressure_bar)


def _aag21_number_fraction_map(chem: Any) -> dict[str, float]:
    reference = np.asarray(chem.element_vector_reference, dtype=np.float64)
    elements = list(chem.elements)
    h_index = elements.index("H")
    return {
        element: float(reference[index] / reference[h_index])
        for index, element in enumerate(elements)
        if element != "e-"
    }


def _aag21_five_inputs(chem: Any) -> dict[str, float]:
    aag21_fracs = _aag21_number_fraction_map(chem)
    return {
        "He_H": aag21_fracs["He"],
        "C_H": aag21_fracs["C"],
        "O_H": aag21_fracs["O"],
        "N_H": aag21_fracs["N"],
        "S_H": aag21_fracs["S"],
    }


def _he_fixed_aag21_inputs(aag21_inputs: dict[str, float]) -> dict[str, float]:
    updated = dict(aag21_inputs)
    updated["He_H"] = SOLAR_ELEMENT_ABUNDANCES["He_H"]
    return updated


def _nearest_training_manifold_inputs(aag21_inputs: dict[str, float]) -> dict[str, float]:
    oxygen_frac = float(aag21_inputs["O_H"])
    return {
        "He_H": SOLAR_ELEMENT_ABUNDANCES["He_H"],
        "O_H": oxygen_frac,
        "C_H": oxygen_frac * SOLAR_ELEMENT_ABUNDANCES["C_H"] / SOLAR_ELEMENT_ABUNDANCES["O_H"],
        "N_H": SOLAR_ELEMENT_ABUNDANCES["N_H"] * oxygen_frac / SOLAR_ELEMENT_ABUNDANCES["O_H"],
        "S_H": oxygen_frac * SOLAR_ELEMENT_ABUNDANCES["S_H"] / SOLAR_ELEMENT_ABUNDANCES["O_H"],
    }


def _global_zscores(model: Any, inputs: dict[str, float]) -> dict[str, float]:
    payload = model.normalization["global_static"]
    feature_order = list(model.data_contract["global_static_feature_order"])
    zscores: dict[str, float] = {}
    for index, name in enumerate(feature_order):
        method = str(payload["methods"][index]).lower()
        mean = float(payload["mean"][index])
        std = float(payload["std"][index])
        value = float(inputs[name])
        if method == "standard":
            zscores[name] = (value - mean) / std
            continue
        if method in {"log-standard", "log-minmax"}:
            zscores[name] = (np.log10(max(value, 1.0e-30)) - mean) / std
            continue
        zscores[name] = np.nan
    return zscores


def _mean_abs_log10_error(
    left: np.ndarray,
    right: np.ndarray,
    *,
    threshold: float = 0.0,
) -> float:
    left_safe = np.asarray(left, dtype=np.float64)
    right_safe = np.asarray(right, dtype=np.float64)
    mask = (left_safe > threshold) | (right_safe > threshold)
    if not np.any(mask):
        return 0.0
    left_clipped = np.clip(left_safe[mask], EPSILON, None)
    right_clipped = np.clip(right_safe[mask], EPSILON, None)
    return float(np.mean(np.abs(np.log10(left_clipped) - np.log10(right_clipped))))


def _photosphere_species_summary(
    vmr: np.ndarray,
    pressure_bar: np.ndarray,
    species: list[str],
) -> dict[str, float]:
    index = int(np.argmin(np.abs(np.asarray(pressure_bar) - PHOTOSPHERE_PRESSURE_BAR)))
    return {
        name: float(vmr[index, species.index(name)])
        for name in ("H2", "He", "CO", "H2O")
    }


def _print_vmr_summary(
    *,
    label: str,
    vmr: np.ndarray,
    pressure_bar: np.ndarray,
    species: list[str],
) -> None:
    vmr_sum = np.sum(vmr, axis=1)
    photosphere = _photosphere_species_summary(vmr, pressure_bar, species)
    print(label)
    print(f"  species-sum min/max: {vmr_sum.min():.6f}, {vmr_sum.max():.6f}")
    for name, value in photosphere.items():
        print(f"  {name}@0.1bar = {value:.6e}")
    print(f"  any VMR > 1: {bool(np.any(vmr > 1.0))}")


def _print_comparison_summary(
    *,
    label: str,
    candidate_vmr: np.ndarray,
    reference_vmr: np.ndarray,
    pressure_bar: np.ndarray,
    species: list[str],
) -> None:
    print(label)
    for threshold in THRESHOLDS:
        print(
            f"  mean |Δlog10 VMR| (>{threshold:.0e}) = "
            f"{_mean_abs_log10_error(candidate_vmr, reference_vmr, threshold=threshold):.6f}"
        )
    candidate_photosphere = _photosphere_species_summary(candidate_vmr, pressure_bar, species)
    reference_photosphere = _photosphere_species_summary(reference_vmr, pressure_bar, species)
    for name in ("H2", "He", "CO", "H2O"):
        print(
            f"  {name}@0.1bar: candidate={candidate_photosphere[name]:.6e} "
            f"reference={reference_photosphere[name]:.6e}"
        )


def _run_transformer_public(
    model: Any,
    *,
    pressure_bar: np.ndarray,
    temperature_k: np.ndarray,
    inputs: dict[str, float],
) -> tuple[np.ndarray | None, str | None]:
    try:
        vmr = np.asarray(
            model.predict_fastchem_profile(
                pressure_bar=pressure_bar,
                temperature_k=temperature_k,
                global_inputs=inputs,
                return_log10=False,
            ),
            dtype=np.float64,
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, f"{type(exc).__name__}: {exc}"
    return vmr, None


def _run_transformer_diagnostic(
    model: Any,
    *,
    pressure_bar: np.ndarray,
    temperature_k: np.ndarray,
    inputs: dict[str, float],
) -> np.ndarray:
    return np.asarray(
        _predict_fastchem_profile_impl(
            params=model.params,
            dims=model.dims,
            normalization=model.normalization,
            data_contract=model.data_contract,
            model_type=model.model_type,
            pressure_bar=pressure_bar,
            temperature_k=temperature_k,
            global_inputs=inputs,
            return_log10=False,
        ),
        dtype=np.float64,
    )


def _run_exogibbs_standard(
    chem: Any,
    *,
    pressure_bar: np.ndarray,
    temperature_k: np.ndarray,
) -> np.ndarray:
    opts = EquilibriumOptions(epsilon_crit=1.0e-11, max_iter=1000, method="vmap_cold")
    result = equilibrium_profile(
        chem,
        temperature_k,
        pressure_bar,
        np.asarray(chem.element_vector_reference, dtype=np.float64),
        Pref=1.0,
        initializer=None,
        options=opts,
    )
    mixing_ratio = np.asarray(result.x, dtype=np.float64)
    return np.column_stack(
        [mixing_ratio[:, chem.species.index(EXOGIBBS_SPECIES_MAP[name])] for name in EXOGIBBS_SPECIES_MAP]
    )


def _run_exogibbs_legacy(
    chem: Any,
    grid: Any,
    *,
    pressure_bar: np.ndarray,
    temperature_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    initializer = GridEquilibriumInitializer(grid=grid, preset_name="fastchem")
    opts = EquilibriumOptions(epsilon_crit=1.0e-11, max_iter=1000, method="vmap_cold")
    result = equilibrium_profile(
        chem,
        temperature_k,
        pressure_bar,
        np.asarray(chem.element_vector_reference, dtype=np.float64),
        Pref=1.0,
        initializer=initializer,
        options=opts,
    )
    mixing_ratio = np.asarray(result.x, dtype=np.float64)
    reduced = np.column_stack(
        [mixing_ratio[:, chem.species.index(EXOGIBBS_SPECIES_MAP[name])] for name in EXOGIBBS_SPECIES_MAP]
    )
    finite_rows = np.isfinite(reduced).all(axis=1)
    return reduced, finite_rows


def _print_input_set_summary(
    *,
    label: str,
    model: Any,
    inputs: dict[str, float],
) -> None:
    print(label)
    print(f"  raw globals: {inputs}")
    zscores = _global_zscores(model, inputs)
    print(f"  normalized z-scores: {zscores}")


def _report_saved_test_case(
    *,
    model: Any,
    test_case: FastChemTestCase,
    vulcan_source_root: Path,
) -> None:
    print("\n=== Saved Processed Test Profile ===")
    print(f"run_id: {test_case.run_id}")
    if test_case.raw_metadata is not None:
        print(f"profile_source: {test_case.raw_metadata.get('temperature_profile_source', 'unknown')}")

    public_vmr, public_error = _run_transformer_public(
        model,
        pressure_bar=test_case.pressure_bar,
        temperature_k=test_case.temperature_k,
        inputs=test_case.global_inputs,
    )
    if public_error is not None:
        raise RuntimeError(f"Saved test case unexpectedly failed the public transformer path: {public_error}")

    live_fastchem = _run_fastchem_online(
        source_root=vulcan_source_root,
        pressure_bar=test_case.pressure_bar,
        temperature_k=test_case.temperature_k,
        globals_map=test_case.raw_globals or test_case.global_inputs,
        output_species=test_case.output_species,
        config=model.config,
    )
    _print_vmr_summary(
        label="saved transformer output",
        vmr=public_vmr,
        pressure_bar=test_case.pressure_bar,
        species=test_case.output_species,
    )
    _print_comparison_summary(
        label="saved transformer vs stored processed target",
        candidate_vmr=public_vmr,
        reference_vmr=test_case.stored_target_ymix,
        pressure_bar=test_case.pressure_bar,
        species=test_case.output_species,
    )
    _print_comparison_summary(
        label="saved transformer vs live FastChem",
        candidate_vmr=public_vmr,
        reference_vmr=live_fastchem,
        pressure_bar=test_case.pressure_bar,
        species=test_case.output_species,
    )


def _report_notebook_profile(
    *,
    label: str,
    model: Any,
    chem: Any,
    grid: Any,
    pressure_bar: np.ndarray,
    temperature_k: np.ndarray,
    vulcan_source_root: Path,
    include_nearest_manifold: bool = True,
) -> None:
    print(f"\n=== {label} ===")
    print(
        f"grid: {pressure_bar.size} levels, {pressure_bar[0]:.1e} -> {pressure_bar[-1]:.1e} bar | "
        f"Tmin={temperature_k.min():.2f} K"
    )
    bundle_inputs = dict(SOLAR_ELEMENT_ABUNDANCES)
    aag21_inputs = _aag21_five_inputs(chem)
    he_fixed_inputs = _he_fixed_aag21_inputs(aag21_inputs)

    input_sets: list[tuple[str, dict[str, float]]] = [
        ("bundle reference five-element inputs", bundle_inputs),
        ("raw AAG21 five-element inputs", aag21_inputs),
        ("AAG21 with He_H fixed to bundle reference", he_fixed_inputs),
    ]
    if include_nearest_manifold:
        input_sets.append(
            ("nearest training-manifold inputs", _nearest_training_manifold_inputs(aag21_inputs))
        )

    for input_label, inputs in input_sets:
        print("")
        _print_input_set_summary(label=input_label, model=model, inputs=inputs)
        public_vmr, public_error = _run_transformer_public(
            model,
            pressure_bar=pressure_bar,
            temperature_k=temperature_k,
            inputs=inputs,
        )
        if public_error is not None:
            print(f"  public transformer path: {public_error}")
            diagnostic_vmr = _run_transformer_diagnostic(
                model,
                pressure_bar=pressure_bar,
                temperature_k=temperature_k,
                inputs=inputs,
            )
            print("  diagnostic transformer path: ran private core to quantify the rejected input")
        else:
            diagnostic_vmr = public_vmr
            print("  public transformer path: success")

        live_fastchem = _run_fastchem_online(
            source_root=vulcan_source_root,
            pressure_bar=pressure_bar,
            temperature_k=temperature_k,
            globals_map=inputs,
            output_species=list(EXOGIBBS_SPECIES_MAP),
            config=model.config,
        )
        _print_vmr_summary(
            label="  transformer output",
            vmr=diagnostic_vmr,
            pressure_bar=pressure_bar,
            species=list(EXOGIBBS_SPECIES_MAP),
        )
        _print_vmr_summary(
            label="  live FastChem output",
            vmr=live_fastchem,
            pressure_bar=pressure_bar,
            species=list(EXOGIBBS_SPECIES_MAP),
        )
        _print_comparison_summary(
            label="  transformer vs live FastChem",
            candidate_vmr=diagnostic_vmr,
            reference_vmr=live_fastchem,
            pressure_bar=pressure_bar,
            species=list(EXOGIBBS_SPECIES_MAP),
        )

    live_fastchem_aag21 = _run_fastchem_online(
        source_root=vulcan_source_root,
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        globals_map=_aag21_five_inputs(chem),
        output_species=list(EXOGIBBS_SPECIES_MAP),
        config=model.config,
    )
    exogibbs_standard = _run_exogibbs_standard(
        chem,
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
    )
    _print_vmr_summary(
        label="standard ExoGibbs output (initializer=None)",
        vmr=exogibbs_standard,
        pressure_bar=pressure_bar,
        species=list(EXOGIBBS_SPECIES_MAP),
    )
    _print_comparison_summary(
        label="standard ExoGibbs vs live FastChem (AAG21 abundances)",
        candidate_vmr=exogibbs_standard,
        reference_vmr=live_fastchem_aag21,
        pressure_bar=pressure_bar,
        species=list(EXOGIBBS_SPECIES_MAP),
    )

    legacy_vmr, legacy_finite_rows = _run_exogibbs_legacy(
        chem,
        grid,
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
    )
    bad_rows = np.where(~legacy_finite_rows)[0]
    print("legacy ExoGibbs notebook setup (GridEquilibriumInitializer)")
    print(f"  packaged grid path: {get_default_equilibrium_grid_path('fastchem')}")
    print(
        f"  packaged grid temperature range: "
        f"{float(grid.temperature_axis[0]):.1f} -> {float(grid.temperature_axis[-1]):.1f} K"
    )
    print(f"  finite rows: {int(np.sum(legacy_finite_rows))}/{legacy_finite_rows.size}")
    print(f"  bad row indices: {bad_rows.tolist()}")
    if bad_rows.size:
        print(
            "  first bad row: "
            f"index={int(bad_rows[0])} pressure={pressure_bar[int(bad_rows[0])]:.6e} bar "
            f"temperature={temperature_k[int(bad_rows[0])]:.2f} K"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    project_root = resolve_project_root(Path(__file__).resolve())
    bundle_path = resolve_bundle_path(project_root, args.bundle)
    model = load_exported_model(bundle_path)
    if not model.uses_fastchem or not model.uses_transformer:
        raise RuntimeError(f"Expected a FastChem transformer bundle, got {bundle_path}.")

    vulcan_source_root = resolve_vulcan_source_root(
        project_root,
        config=model.config,
        explicit_root=args.vulcan_source_root,
    )
    context = load_fastchem_test_context(
        project_root,
        bundle_path=bundle_path,
        config=model.config,
        require_raw=True,
    )
    selected_run_id = select_fastchem_test_run_id(
        context,
        run_id=args.run_id,
        require_raw=True,
        rng=np.random.default_rng(0),
    )
    test_case = load_fastchem_test_case(context, run_id=selected_run_id)

    chem = chemsetup(silent=True)
    grid = load_equilibrium_grid_netcdf(str(get_default_equilibrium_grid_path("fastchem")))
    training_num_levels = int(model.config["sampling"]["num_levels"])
    training_pressure_top_bar = float(model.config["sampling"]["pressure_top_bar"])
    training_pressure_bottom_bar = float(model.config["sampling"]["pressure_bottom_bar"])

    print("=== Runtime Summary ===")
    print(f"python_executable: {sys.executable}")
    print(f"bundle_path: {bundle_path}")
    print(f"vulcan_source_root: {vulcan_source_root}")
    print(
        "package_versions: "
        f"jax={_package_version('jax')} "
        f"jaxlib={_package_version('jaxlib')} "
        f"numpy={_package_version('numpy')} "
        f"exojax={_package_version('exojax')} "
        f"exogibbs={_package_version('exogibbs')}"
    )
    print(
        f"transformer_training_contract: {training_num_levels} levels, "
        f"{training_pressure_bottom_bar:.1e} -> {training_pressure_top_bar:.1e} bar"
    )
    print(f"transformer_global_static_normalization: {model.normalization['global_static']}")
    print(
        f"exogibbs_fastchem_preset: {len(chem.species)} species, "
        f"{len(chem.elements)} elements, source={chem.metadata['source']}"
    )
    print(
        "fastchem_tables: "
        f"standard_logK_entries={_count_fastchem_species_entries(vulcan_source_root / 'fastchem_vulcan' / 'input' / 'logK.dat')} "
        f"vulcan_nasa9_entries={_count_fastchem_species_entries(vulcan_source_root / 'fastchem_vulcan' / 'input' / 'nasa9_logK_SNCHOPTi.dat')}"
    )

    _report_saved_test_case(
        model=model,
        test_case=test_case,
        vulcan_source_root=vulcan_source_root,
    )

    validated_pressure, validated_temperature = _build_profile(
        num_levels=training_num_levels,
        pressure_bottom_bar=training_pressure_bottom_bar,
        pressure_top_bar=training_pressure_top_bar,
    )
    _report_notebook_profile(
        label="Notebook Synthetic Profile (Validated 50-Level Grid)",
        model=model,
        chem=chem,
        grid=grid,
        pressure_bar=validated_pressure,
        temperature_k=validated_temperature,
        vulcan_source_root=vulcan_source_root,
    )

    notebook_pressure, notebook_temperature = _build_profile(
        num_levels=NOTEBOOK_NUM_LEVELS,
        pressure_bottom_bar=NOTEBOOK_PRESSURE_BOTTOM_BAR,
        pressure_top_bar=NOTEBOOK_PRESSURE_TOP_BAR,
    )
    _report_notebook_profile(
        label="Notebook Synthetic Profile (Off-Contract 100-Level Grid)",
        model=model,
        chem=chem,
        grid=grid,
        pressure_bar=notebook_pressure,
        temperature_k=notebook_temperature,
        vulcan_source_root=vulcan_source_root,
        include_nearest_manifold=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
