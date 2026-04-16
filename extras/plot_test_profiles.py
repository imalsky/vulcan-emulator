"""Plot stored FastChem test-set P-T profiles by profile-source category.

Overlays saved test-set profiles on a single panel with distinct line styles:
  - Solid:     PT-library
  - Dashed:    analytic (radiative only)
  - Dash-dot:  analytic (convective adjustment applied)

Usage
-----
    python extras/plot_test_profiles.py
    python extras/plot_test_profiles.py --bundle models/fastchem_transformer/best_exported.npz --seed 42 -n 20
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

# ── Metadata keys used for diversity-based profile selection ─────────────

_SHARED_ANALYTIC_KEYS = [
    "temperature_profile_analytic_t_int_k",
    "temperature_profile_analytic_t_eq_k",
    "temperature_profile_analytic_log10_delta",
    "temperature_profile_analytic_log10_gamma",
    "temperature_profile_analytic_alpha",
    "temperature_profile_analytic_log10_p_trans_bar",
]

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
    "analytic_radiative": _SHARED_ANALYTIC_KEYS,
    "analytic_convective": [
        *_SHARED_ANALYTIC_KEYS,
        "temperature_profile_analytic_adiabatic_gradient",
    ],
}

# ── Plot styling ─────────────────────────────────────────────────────────

_BUCKET_STYLE: dict[str, dict[str, Any]] = {
    "pt_library": {
        "cmap": "Reds",
        "ls": "-",
        "lw": 2.2,
        "label": "PT-library",
    },
    "analytic_radiative": {
        "cmap": "Blues",
        "ls": "--",
        "lw": 2.2,
        "label": "Analytic (radiative)",
    },
    "analytic_convective": {
        "cmap": "Purples",
        "ls": (0, (5, 2, 1, 2)),
        "lw": 2.5,
        "label": "Analytic (conv. adj.)",
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────

def _color_palette(cmap_name: str, n: int) -> list[Any]:
    """Sample *n* evenly spaced colors from the interior of a colormap."""
    cmap = plt.get_cmap(cmap_name)
    return [cmap(0.35 + 0.55 * i / max(n - 1, 1)) for i in range(n)]


def _select_diverse_run_ids(
    run_ids: list[str],
    *,
    context: FastChemTestContext,
    metadata_map: dict[str, dict[str, Any]],
    feature_keys: list[str],
    num_profiles: int,
    rng: np.random.Generator,
) -> list[str]:
    """Choose a metadata-diverse subset via greedy farthest-point sampling."""
    if num_profiles <= 0 or not run_ids:
        return []
    if num_profiles >= len(run_ids):
        return list(run_ids)

    # Build a feature matrix from metadata + normalized temperature stats.
    feature_rows = []
    for run_id in run_ids:
        metadata = metadata_map[run_id]
        temp_norm = np.asarray(
            context.split.sequence_inputs[context.run_id_to_index[run_id], :, 1],
            dtype=np.float64,
        )
        row = [float(metadata.get(k, 0.0)) for k in feature_keys]
        row += [float(np.min(temp_norm)), float(np.max(temp_norm)), float(np.mean(temp_norm))]
        feature_rows.append(row)

    features = np.asarray(feature_rows, dtype=np.float64)
    scale = np.ptp(features, axis=0)
    scale[scale == 0.0] = 1.0
    normalized = (features - np.mean(features, axis=0, keepdims=True)) / scale

    # Greedy farthest-point: start with the point farthest from the centroid.
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

    return [run_ids[i] for i in selected]


def _bucket_test_run_ids(
    run_ids: list[str],
    metadata_map: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    """Group test runs by temperature-profile source category."""
    buckets: dict[str, list[str]] = {name: [] for name in _BUCKET_STYLE}
    for run_id in run_ids:
        bucket = classify_temperature_profile_bucket(metadata_map[run_id])
        buckets[bucket].append(run_id)
    return buckets


# ── Plotting ─────────────────────────────────────────────────────────────

def _plot_bucket(
    ax: plt.Axes,
    test_cases: list[Any],
    style: dict[str, Any],
) -> None:
    """Plot one category of P-T profiles on the given axes."""
    colors = _color_palette(style["cmap"], len(test_cases))
    for i, case in enumerate(test_cases):
        ax.plot(
            case.temperature_k,
            case.pressure_bar,
            ls=style["ls"],
            lw=style["lw"],
            alpha=0.8,
            color=colors[i],
            label=style["label"] if i == 0 else None,
        )


# ── CLI ──────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot saved test-set PT profiles.")
    parser.add_argument("--bundle", default=None, help="Path to an exported model bundle.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-n", "--num-profiles", type=int, default=5,
                        help="Profiles per analytic category.")
    parser.add_argument("--num-roth", type=int, default=5,
                        help="PT-library profiles (overrides -n for that category).")
    parser.add_argument("-o", "--output", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    project_root = resolve_project_root(Path(__file__).resolve())

    # Load model and test data.
    bundle_path = resolve_bundle_path(project_root, args.bundle)
    model = load_exported_model(bundle_path)
    if not model.uses_fastchem or not model.uses_transformer:
        raise RuntimeError(f"Expected a FastChem transformer bundle, got {bundle_path}.")

    context = load_fastchem_test_context(
        project_root, bundle_path=bundle_path, config=model.config, require_raw=True,
    )
    assert context.raw_root is not None
    metadata_map = load_fastchem_raw_metadata_map(context.raw_root, context.split.run_ids)
    rng = np.random.default_rng(args.seed)

    # Bucket runs by profile source, then select a diverse subset from each.
    buckets = _bucket_test_run_ids(context.split.run_ids, metadata_map)
    count_for = {
        "pt_library": args.num_roth,
        "analytic_radiative": args.num_profiles,
        "analytic_convective": args.num_profiles,
    }
    selected: dict[str, list[Any]] = {}
    for bucket_name, run_ids in buckets.items():
        chosen_ids = _select_diverse_run_ids(
            run_ids,
            context=context,
            metadata_map=metadata_map,
            feature_keys=_BUCKET_FEATURE_KEYS[bucket_name],
            num_profiles=min(count_for[bucket_name], len(run_ids)),
            rng=rng,
        )
        selected[bucket_name] = [
            load_fastchem_test_case(context, run_id=rid) for rid in chosen_ids
        ]

    # Plot.
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    for bucket_name, cases in selected.items():
        _plot_bucket(ax, cases, _BUCKET_STYLE[bucket_name])

    ax.set_yscale("log")
    ax.set_ylim(1.0e2, 1.0e-7)
    ax.set_xlim(0.0, 3000.0)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_title("Saved Test P-T Profiles")
    ax.legend(loc="best")

    output_path = (
        resolve_path(args.output, project_root)
        if args.output
        else plots_dir_for_bundle(bundle_path) / "saved_test_profiles.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
