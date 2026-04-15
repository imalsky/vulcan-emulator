"""Compare one saved FastChem test profile against a fresh FastChem rerun.

Picks a random (or user-specified) run from the saved processed test split,
loads the matching stored raw profile, reruns FastChem with the same inputs,
and saves a three-panel comparison figure: P-T profile, mixing ratios, and
log-space residuals.

Usage
-----
    python extras/fastchem_comparison.py
    python extras/fastchem_comparison.py --run-id run_00042
    python extras/fastchem_comparison.py --bundle models/fastchem_transformer/best_exported.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _common import (
    EPSILON,
    apply_style,
    load_fastchem_test_context,
    load_raw_equilibrium_profile,
    plots_dir_for_bundle,
    resolve_bundle_path,
    resolve_vulcan_source_root,
    run_fastchem_online,
    select_fastchem_test_run_id,
)
from matplotlib.lines import Line2D
from src.models.export_bundle import load_exported_model
from src.utils.helpers import resolve_path, resolve_project_root


# ======================================================================
# CLI
# ======================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
# Plotting
# ======================================================================

def _mixing_ratio_xlim(values: list[np.ndarray]) -> tuple[float, float]:
    clipped = [np.clip(arr, EPSILON, None) for arr in values]
    minimum = min(float(np.min(arr)) for arr in clipped)
    lower = 10.0 ** np.floor(np.log10(max(minimum, EPSILON)))
    return lower, 3.0


def _plot_profile_comparison(
    *,
    profile,
    fastchem_ymix: np.ndarray,
    output_path: Path,
) -> None:
    """Render and save a three-panel comparison figure."""
    apply_style()

    fig, (ax_pt, ax_mix, ax_delta) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    colors = plt.cm.tab20(np.linspace(0, 1, len(profile.output_species)))

    clipped_truth = np.clip(profile.equilibrium_ymix, EPSILON, None)
    clipped_fastchem = np.clip(fastchem_ymix, EPSILON, None)

    ax_pt.plot(profile.temperature_k, profile.pressure_bar, color="black", lw=2.0)
    ax_pt.set_xlabel("Temperature [K]")
    ax_pt.set_ylabel("Pressure [bar]")
    ax_pt.set_yscale("log")
    ax_pt.invert_yaxis()
    ax_pt.set_xlim(0.0, 3000.0)
    ax_pt.set_title("Test P-T Profile")

    for i, name in enumerate(profile.output_species):
        c = colors[i]
        ax_mix.plot(clipped_truth[:, i], profile.pressure_bar, color=c, lw=1.6, label=name)
        ax_mix.plot(clipped_fastchem[:, i], profile.pressure_bar, color=c, lw=1.2, ls="--")
        residual = np.log10(clipped_fastchem[:, i] + EPSILON) - np.log10(clipped_truth[:, i] + EPSILON)
        ax_delta.plot(residual, profile.pressure_bar, color=c, lw=1.3)

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
        context, run_id=args.run_id, require_raw=True,
    )
    profile = load_raw_equilibrium_profile(context.raw_root, selected_run_id)

    source_root = resolve_vulcan_source_root(
        project_root, config=model.config, explicit_root=args.vulcan_source_root,
    )
    fastchem_ymix = run_fastchem_online(
        source_root=source_root,
        pressure_bar=profile.pressure_bar,
        temperature_k=profile.temperature_k,
        globals_map=profile.globals,
        output_species=profile.output_species,
        config=model.config,
    )
    if fastchem_ymix.shape != profile.equilibrium_ymix.shape:
        raise ValueError(
            f"FastChem output shape mismatch: "
            f"{fastchem_ymix.shape} vs {profile.equilibrium_ymix.shape}"
        )

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
