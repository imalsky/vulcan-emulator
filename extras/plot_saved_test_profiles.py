"""Plot stored FastChem test-set P-T profiles by profile-source category.

Overlays saved test-set profiles on a single panel with distinct line styles:
  - Solid:     PT-library
  - Dashed:    analytic (radiative only)
  - Dash-dot:  analytic (convective adjustment applied)

Usage
-----
    python extras/plot_saved_test_profiles.py
    python extras/plot_saved_test_profiles.py --bundle models/fastchem_transformer/best_exported.npz --seed 42 -n 20
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from _common import (
    FastChemTestContext,
    apply_style,
    classify_temperature_profile_bucket,
    load_fastchem_raw_metadata_map,
    load_fastchem_test_case,
    load_fastchem_test_context,
    plots_dir_for_bundle,
    resolve_bundle_path,
)
from src.models.export_bundle import load_exported_model
from src.utils.helpers import resolve_path, resolve_project_root

_BUCKET_FEATURE_KEYS: dict[str, list[str]] = {
    "pt_library": [
        "temperature_profile_Teq",
        "temperature_profile_LogMet",
        "temperature_profile_LogDrag",
        "temperature_profile_Mstar",
        "temperature_profile_Rp",
        "temperature_profile_logG",
        "temperature_profile_lon",
        "temperature_profile_lat",
    ],
    "analytic_radiative": [
        "temperature_profile_analytic_t_int_k",
        "temperature_profile_analytic_t_irr_k",
        "temperature_profile_analytic_log10_kappa_ir_m2_kg",
        "temperature_profile_analytic_log10_gamma_1",
        "temperature_profile_analytic_log10_gamma_2",
        "temperature_profile_analytic_alpha",
        "temperature_profile_analytic_temperature_shift_k",
    ],
    "analytic_convective": [
        "temperature_profile_analytic_t_int_k",
        "temperature_profile_analytic_t_irr_k",
        "temperature_profile_analytic_log10_kappa_ir_m2_kg",
        "temperature_profile_analytic_log10_gamma_1",
        "temperature_profile_analytic_log10_gamma_2",
        "temperature_profile_analytic_alpha",
        "temperature_profile_analytic_temperature_shift_k",
        "temperature_profile_analytic_adiabatic_gradient",
    ],
}


def _metadata_feature(metadata: dict[str, Any], key: str) -> float:
    """Convert stored scalar metadata into a numeric selection feature."""
    value = metadata.get(key, 0.0)
    if isinstance(value, bool):
        return float(value)
    return float(value)


def _select_diverse_run_ids(
    run_ids: list[str],
    *,
    context: FastChemTestContext,
    metadata_map: dict[str, dict[str, Any]],
    feature_keys: list[str],
    num_profiles: int,
    rng: np.random.Generator,
) -> list[str]:
    """Choose a metadata-diverse subset of saved test runs."""
    if num_profiles <= 0 or not run_ids:
        return []
    if num_profiles >= len(run_ids):
        return list(run_ids)

    feature_rows = []
    for run_id in run_ids:
        metadata = metadata_map[run_id]
        temp_norm = np.asarray(
            context.split.sequence_inputs[context.run_id_to_index[run_id], :, 1],
            dtype=np.float64,
        )
        feature_rows.append(
            [_metadata_feature(metadata, key) for key in feature_keys]
            + [
                float(np.min(temp_norm)),
                float(np.max(temp_norm)),
                float(np.mean(temp_norm)),
            ]
        )

    features = np.asarray(feature_rows, dtype=np.float64)
    scale = np.ptp(features, axis=0)
    scale[scale == 0.0] = 1.0
    normalized = (features - np.mean(features, axis=0, keepdims=True)) / scale

    centroid = np.mean(normalized, axis=0)
    selected = [int(np.argmax(np.sum((normalized - centroid) ** 2, axis=1)))]
    remaining = np.ones(len(run_ids), dtype=bool)
    remaining[selected[0]] = False

    while len(selected) < num_profiles:
        candidate_idx = np.flatnonzero(remaining)
        distances = np.sum(
            (normalized[candidate_idx, None, :] - normalized[np.asarray(selected), :]) ** 2,
            axis=2,
        )
        min_dist = np.min(distances, axis=1)
        chosen = int(candidate_idx[np.argmax(min_dist + 1.0e-12 * rng.random(candidate_idx.size))])
        selected.append(chosen)
        remaining[chosen] = False

    return [run_ids[index] for index in selected]


def _bucket_test_run_ids(
    run_ids: list[str],
    metadata_map: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    """Group saved test runs by raw temperature-profile provenance."""
    buckets = {
        "pt_library": [],
        "analytic_radiative": [],
        "analytic_convective": [],
    }
    for run_id in run_ids:
        bucket = classify_temperature_profile_bucket(metadata_map[run_id])
        buckets[bucket].append(run_id)
    return buckets


def _color_palette(cmap: Any, n: int) -> list[Any]:
    """Sample *n* evenly spaced colors from the interior of a colormap."""
    return [cmap(0.35 + 0.55 * i / max(n - 1, 1)) for i in range(n)]


def main(argv: list[str] | None = None) -> None:
    """Plot saved processed FastChem test profiles by category."""
    parser = argparse.ArgumentParser(description="Plot saved test-set PT profiles.")
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--bundle", default=None,
        help="Path to an exported FastChem transformer bundle.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "-n", "--num-profiles", type=int, default=5,
        help="Number of profiles per analytic category (radiative, convective).",
    )
    parser.add_argument(
        "--num-roth", type=int, default=5,
        help="Number of PT-library profiles (overrides -n for PT-library).",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

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
    metadata_map = load_fastchem_raw_metadata_map(context.raw_root, context.split.run_ids)
    buckets = _bucket_test_run_ids(context.split.run_ids, metadata_map)
    rng = np.random.default_rng(args.seed)

    selected_pt_library = _select_diverse_run_ids(
        buckets["pt_library"],
        context=context,
        metadata_map=metadata_map,
        feature_keys=_BUCKET_FEATURE_KEYS["pt_library"],
        num_profiles=min(args.num_roth, len(buckets["pt_library"])),
        rng=rng,
    )
    selected_analytic_radiative = _select_diverse_run_ids(
        buckets["analytic_radiative"],
        context=context,
        metadata_map=metadata_map,
        feature_keys=_BUCKET_FEATURE_KEYS["analytic_radiative"],
        num_profiles=min(args.num_profiles, len(buckets["analytic_radiative"])),
        rng=rng,
    )
    selected_analytic_convective = _select_diverse_run_ids(
        buckets["analytic_convective"],
        context=context,
        metadata_map=metadata_map,
        feature_keys=_BUCKET_FEATURE_KEYS["analytic_convective"],
        num_profiles=min(args.num_profiles, len(buckets["analytic_convective"])),
        rng=rng,
    )

    pt_library_cases = [
        load_fastchem_test_case(context, run_id=run_id)
        for run_id in selected_pt_library
    ]
    analytic_radiative_cases = [
        load_fastchem_test_case(context, run_id=run_id)
        for run_id in selected_analytic_radiative
    ]
    analytic_convective_cases = [
        load_fastchem_test_case(context, run_id=run_id)
        for run_id in selected_analytic_convective
    ]

    apply_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    red_cmap = plt.cm.Reds
    blue_cmap = plt.cm.Blues
    purple_cmap = plt.cm.Purples

    for index, test_case in enumerate(pt_library_cases):
        color = _color_palette(red_cmap, len(pt_library_cases))[index]
        ax.plot(
            test_case.temperature_k,
            test_case.pressure_bar,
            ls="-",
            lw=2.2,
            alpha=0.8,
            color=color,
            label="PT-library" if index == 0 else None,
        )

    for index, test_case in enumerate(analytic_radiative_cases):
        color = _color_palette(blue_cmap, len(analytic_radiative_cases))[index]
        ax.plot(
            test_case.temperature_k,
            test_case.pressure_bar,
            ls="--",
            lw=2.2,
            alpha=0.8,
            color=color,
            label="Analytic (radiative)" if index == 0 else None,
        )

    for index, test_case in enumerate(analytic_convective_cases):
        color = _color_palette(purple_cmap, len(analytic_convective_cases))[index]
        ax.plot(
            test_case.temperature_k,
            test_case.pressure_bar,
            ls=(0, (5, 2, 1, 2)),
            lw=2.5,
            alpha=0.8,
            color=color,
            label="Analytic (conv. adj.)" if index == 0 else None,
        )

    ax.set_yscale("log")
    ax.set_ylim(1.0e2, 1.0e-7)
    ax.set_xlim(0.0, 3000.0)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_title("Saved Test P-T Profiles")
    ax.legend(loc="best")

    if args.output:
        output_path = resolve_path(args.output, project_root)
    else:
        output_path = plots_dir_for_bundle(bundle_path) / "saved_test_profiles.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
