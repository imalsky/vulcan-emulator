"""FastChem emulator demo on one saved processed test profile.

Runs four sections:
  1. Load the exported JAX model bundle.
  2. Recover one stored processed test profile in physical units.
  3. Run equilibrium inference (JAX forward pass).
  4. Optionally compare against a live FastChem rerun and save one figure.

Usage
-----
    python extras/fastchem_saved_test_profile_demo.py
    python extras/fastchem_saved_test_profile_demo.py --bundle models/fastchem_transformer/best_exported.npz
    python extras/fastchem_saved_test_profile_demo.py --run-id run_00042
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _common import (
    EPSILON,
    FastChemTestCase,
    apply_style,
    load_fastchem_test_case,
    load_fastchem_test_context,
    plots_dir_for_bundle,
    resolve_bundle_path,
    resolve_vulcan_source_root,
    select_fastchem_test_run_id,
)
from compare_saved_test_profile_fastchem import _run_fastchem_online
from src.models.export_bundle import load_exported_model
from src.utils.helpers import resolve_path, resolve_project_root

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
        help="Optional explicit run ID from the saved processed test split.",
    )
    parser.add_argument(
        "--vulcan-source-root", default=None,
        help="VULCAN-master checkout for the FastChem comparison.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for generated figures (default: model plots dir).",
    )
    return parser.parse_args(argv)


def _fastchem_rerun_globals(test_case: FastChemTestCase) -> dict[str, float]:
    """Prefer raw globals for the FastChem rerun when they are available."""
    if test_case.raw_globals is not None:
        return test_case.raw_globals
    return test_case.global_inputs


# ======================================================================
# Main
# ======================================================================

def main(argv: list[str] | None = None) -> int:
    """Run the standalone FastChem emulator on one saved processed test case."""
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
    )
    selected_run_id = select_fastchem_test_run_id(context, run_id=args.run_id)
    test_case = load_fastchem_test_case(context, run_id=selected_run_id)
    vulcan_source_root = resolve_vulcan_source_root(
        project_root, config=model.config, explicit_root=args.vulcan_source_root,
    )
    output_dir = (
        resolve_path(args.output_dir, project_root)
        if args.output_dir is not None
        else plots_dir_for_bundle(bundle_path)
    )

    species = list(test_case.output_species)
    stored_log10 = np.log10(np.clip(test_case.stored_target_ymix, EPSILON, None))

    print("=" * 72)
    print("SECTION 1 -- Loading exported JAX model bundle")
    print("=" * 72)
    print(f"  Bundle path      : {bundle_path}")
    print(f"  Processed root   : {context.processed_root}")
    print(f"  Model type       : FastChem {model.model_type}")
    print(f"  Bundle version   : {model.export_version}")
    print(f"  Output species   : {', '.join(species)}")

    print("\n" + "=" * 72)
    print("SECTION 2 -- Recovering a saved processed test profile")
    print("=" * 72)
    print(f"  Selected run     : {test_case.run_id}")
    print(
        f"  Pressure grid    : {test_case.pressure_bar[-1]:.1e} - "
        f"{test_case.pressure_bar[0]:.1f} bar  ({test_case.pressure_bar.size} levels)"
    )
    print(
        f"  Temperature      : {test_case.temperature_k[-1]:.0f} K (top) - "
        f"{test_case.temperature_k[0]:.0f} K (bottom)"
    )
    if test_case.raw_metadata is not None:
        source = test_case.raw_metadata.get("temperature_profile_source", "unknown")
        print(f"  Profile source   : {source}")
    print(f"  Global inputs    : {test_case.global_inputs}")

    print("\n" + "=" * 72)
    print("SECTION 3 -- Running equilibrium inference (JAX forward pass)")
    print("=" * 72)
    emulator_log10 = np.asarray(
        model.predict_fastchem_profile(
            pressure_bar=test_case.pressure_bar,
            temperature_k=test_case.temperature_k,
            global_inputs=test_case.global_inputs,
            return_log10=True,
        )
    )
    emulator_ymix = np.power(10.0, emulator_log10)
    phot_idx = int(np.argmin(np.abs(test_case.pressure_bar - 0.1)))

    print(f"  Output shape     : {emulator_log10.shape}  (levels x species)")
    print("\n  Key species at photosphere (P ~ 0.1 bar):")
    for name in ["H2", "H2O", "CO", "CO2", "CH4", "NH3", "H2S"]:
        if name in species:
            species_index = species.index(name)
            print(
                f"    {name:>5s}  stored = {stored_log10[phot_idx, species_index]:+.3f} dex, "
                f"emulator = {emulator_log10[phot_idx, species_index]:+.3f} dex"
            )

    print("\n" + "=" * 72)
    print("SECTION 4 -- FastChem comparison  (requires VULCAN-master)")
    print("=" * 72)
    fastchem_ymix: np.ndarray | None = None
    try:
        if not vulcan_source_root.exists():
            print(f"  Skipped -- VULCAN source not found at: {vulcan_source_root}")
        else:
            print(f"  Running FastChem from: {vulcan_source_root}")
            fastchem_ymix = _run_fastchem_online(
                source_root=vulcan_source_root,
                pressure_bar=test_case.pressure_bar,
                temperature_k=test_case.temperature_k,
                globals_map=_fastchem_rerun_globals(test_case),
                output_species=species,
                config=model.config,
            )
            h2o_index = species.index("H2O")
            stored_h2o = float(stored_log10[phot_idx, h2o_index])
            emulator_h2o = float(emulator_log10[phot_idx, h2o_index])
            fastchem_h2o = float(np.log10(max(fastchem_ymix[phot_idx, h2o_index], EPSILON)))
            print(f"  FastChem output shape: {fastchem_ymix.shape}")
            print(
                f"  H2O @ 0.1 bar -- stored: {stored_h2o:.3f} dex, "
                f"emulator: {emulator_h2o:.3f} dex, "
                f"FastChem: {fastchem_h2o:.3f} dex"
            )
    except Exception as exc:
        print(f"  Skipped -- {type(exc).__name__}: {exc}")

    print("\n" + "=" * 72)
    print("SECTION 5 -- Generating plots")
    print("=" * 72)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib.lines import Line2D

        apply_style()

        fig, (ax_pt, ax_mix) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
        colors = plt.cm.tab20(np.linspace(0, 1, len(species)))

        ax_pt.plot(test_case.temperature_k, test_case.pressure_bar, color="black", lw=2.0)
        ax_pt.set_xlabel("Temperature [K]")
        ax_pt.set_ylabel("Pressure [bar]")
        ax_pt.set_yscale("log")
        ax_pt.invert_yaxis()
        ax_pt.set_xlim(0.0, 3000.0)
        ax_pt.set_title("Stored Test P-T Profile")
        ax_pt.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=8))
        ax_pt.yaxis.set_minor_locator(mticker.NullLocator())

        for species_index, species_name in enumerate(species):
            color = colors[species_index]
            ax_mix.plot(
                np.clip(test_case.stored_target_ymix[:, species_index], EPSILON, None),
                test_case.pressure_bar,
                color=color,
                lw=1.6,
                label=species_name,
            )
            ax_mix.plot(
                np.clip(emulator_ymix[:, species_index], EPSILON, None),
                test_case.pressure_bar,
                color=color,
                lw=1.2,
                ls="--",
            )
            if fastchem_ymix is not None:
                ax_mix.plot(
                    np.clip(fastchem_ymix[:, species_index], EPSILON, None),
                    test_case.pressure_bar,
                    color=color,
                    lw=1.0,
                    ls=":",
                )

        ax_mix.set_xscale("log")
        ax_mix.set_xlim(1.0e-20, 3.0)
        ax_mix.set_xlabel("Mixing Ratio")
        ax_mix.set_title("Stored Test vs Emulator")

        species_legend = ax_mix.legend(fontsize=7, ncol=3, loc="lower left")
        ax_mix.add_artist(species_legend)
        handles = [
            Line2D([0], [0], color="black", lw=1.6, ls="-", label="Stored test"),
            Line2D([0], [0], color="black", lw=1.2, ls="--", label="Emulator"),
        ]
        if fastchem_ymix is not None:
            handles.append(Line2D([0], [0], color="black", lw=1.0, ls=":", label="FastChem"))
        ax_mix.legend(handles=handles, fontsize=8, loc="upper right")

        fig.suptitle(
            f"{test_case.run_id}  "
            f"(He/H={test_case.global_inputs['He_H']:.3e}, "
            f"C/H={test_case.global_inputs['C_H']:.3e}, "
            f"O/H={test_case.global_inputs['O_H']:.3e})",
            fontsize=11,
        )
        fig.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "01_mixing_ratios.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        print(f"  Saved: {output_path}")
    except ImportError as exc:
        print(f"  Matplotlib not available -- skipping plots ({exc})")

    print("\n" + "=" * 72)
    print("Done. All outputs are in:", output_dir)
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
