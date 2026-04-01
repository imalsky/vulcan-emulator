"""Plot randomly sampled PT profiles: analytic, Roth (PT-library), and convective.

Overlays all three types on a single panel with distinct line styles:
  - Solid:  Roth PT-library
  - Dashed: analytic (radiative only)
  - Dash-dot: convective adjustment applied

Usage:
    python extras/plot_profiles.py
    python extras/plot_profiles.py --config config/fastchem_mlp_config.json --seed 42 -n 20
"""

from __future__ import annotations

import argparse
import glob as globmod
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.utils.numpy_compat import patch_numpy_asarray_copy

patch_numpy_asarray_copy()

import matplotlib.pyplot as plt
import numpy as np

from src.utils.config import load_and_validate_config
from src.data_generation.roth_sampling import (
    RothProfile,
    _matches_filters,
    _parse_pt_profile_filename,
    load_roth_profiles,
)
from src.data_generation.sampling import (
    sample_pressure_grid,
    _sample_analytic_temperature_profile_record,
    _validate_temperature_profile,
)

_STYLE = _ROOT / "extras" / "science.mplstyle"


def _resolve_output_dir(config: dict) -> Path:
    """Return the plots directory under the configured checkpoint root.

    Parameters
    ----------
    config : dict
        Validated pipeline config containing ``paths.checkpoints_root``.

    Returns
    -------
    Path
        Existing plots directory for saving the profile figure.
    """
    ckpt_root = _ROOT / config["paths"]["checkpoints_root"]
    plots_dir = ckpt_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def _select_diverse_roth_profiles(
    profiles: list[RothProfile],
    *,
    num_profiles: int,
    rng: np.random.Generator,
) -> list[RothProfile]:
    """Choose a broad subset of PT-library profiles without replacement.

    Profiles are embedded in a small feature space using both filename
    metadata and simple temperature summary statistics, then selected with
    greedy farthest-point sampling so the plotted set spans as much of the
    available library range as possible.
    """
    if num_profiles <= 0 or not profiles:
        return []
    if num_profiles >= len(profiles):
        return list(profiles)

    feature_rows = []
    for profile in profiles:
        meta = profile.metadata
        temperature_k = np.asarray(profile.temperature_k, dtype=np.float64)
        feature_rows.append(
            [
                float(meta.get("Teq", np.mean(temperature_k))),
                float(meta.get("LogMet", 0.0)),
                float(meta.get("LogDrag", 0.0)),
                float(meta.get("Mstar", 0.0)),
                float(meta.get("Rp", 0.0)),
                float(meta.get("logG", 0.0)),
                float(meta.get("lon", 0.0)),
                float(meta.get("lat", 0.0)),
                float(np.min(temperature_k)),
                float(np.max(temperature_k)),
                float(np.mean(temperature_k)),
            ]
        )

    features = np.asarray(feature_rows, dtype=np.float64)
    scale = np.ptp(features, axis=0)
    scale[scale == 0.0] = 1.0
    normalized = (features - np.mean(features, axis=0, keepdims=True)) / scale

    centroid = np.mean(normalized, axis=0)
    selected = [int(np.argmax(np.sum((normalized - centroid) ** 2, axis=1)))]
    remaining = np.ones(len(profiles), dtype=bool)
    remaining[selected[0]] = False

    while len(selected) < num_profiles:
        candidate_idx = np.flatnonzero(remaining)
        distances = np.sum(
            (normalized[candidate_idx, None, :] - normalized[np.asarray(selected), :]) ** 2,
            axis=2,
        )
        min_distance = np.min(distances, axis=1)
        # Small jitter keeps tie-breaking stable but non-degenerate.
        chosen = int(candidate_idx[np.argmax(min_distance + 1.0e-12 * rng.random(candidate_idx.size))])
        selected.append(chosen)
        remaining[chosen] = False

    return [profiles[idx] for idx in selected]


def _select_diverse_roth_files(
    file_paths: list[Path],
    *,
    filters: dict,
    num_files: int,
    rng: np.random.Generator,
) -> list[Path]:
    """Choose a metadata-diverse subset of PT-library files.

    Parameters
    ----------
    file_paths : list[Path]
        Candidate PT-library files.
    filters : dict
        Metadata filters that candidate files must satisfy.
    num_files : int
        Requested number of files to keep.
    rng : np.random.Generator
        Random number generator used to break distance ties.

    Returns
    -------
    list[Path]
        Selected subset of eligible PT-library files.
    """
    eligible_paths: list[Path] = []
    feature_rows = []
    for path in file_paths:
        metadata = _parse_pt_profile_filename(path)
        if not _matches_filters(metadata, filters):
            continue
        eligible_paths.append(path)
        feature_rows.append(
            [
                float(metadata.get("Teq", 0.0)),
                float(metadata.get("LogMet", 0.0)),
                float(metadata.get("LogDrag", 0.0)),
                float(metadata.get("Mstar", 0.0)),
                float(metadata.get("Rp", 0.0)),
                float(metadata.get("logG", 0.0)),
                float(bool(metadata.get("TiOVO", False))),
            ]
        )

    if num_files <= 0 or not eligible_paths:
        return []
    if num_files >= len(eligible_paths):
        return eligible_paths

    features = np.asarray(feature_rows, dtype=np.float64)
    scale = np.ptp(features, axis=0)
    scale[scale == 0.0] = 1.0
    normalized = (features - np.mean(features, axis=0, keepdims=True)) / scale

    centroid = np.mean(normalized, axis=0)
    selected = [int(np.argmax(np.sum((normalized - centroid) ** 2, axis=1)))]
    remaining = np.ones(len(eligible_paths), dtype=bool)
    remaining[selected[0]] = False

    while len(selected) < num_files:
        candidate_idx = np.flatnonzero(remaining)
        distances = np.sum(
            (normalized[candidate_idx, None, :] - normalized[np.asarray(selected), :]) ** 2,
            axis=2,
        )
        min_distance = np.min(distances, axis=1)
        chosen = int(candidate_idx[np.argmax(min_distance + 1.0e-12 * rng.random(candidate_idx.size))])
        selected.append(chosen)
        remaining[chosen] = False

    return [eligible_paths[idx] for idx in selected]


def main(argv: list[str] | None = None) -> None:
    """Sample analytic and PT-library profiles and save a comparison plot.

    Parameters
    ----------
    argv : list[str] or None, optional
        Optional CLI argument vector. When ``None``, arguments are read from
        ``sys.argv``.

    Returns
    -------
    None
        A profile comparison figure is written to disk.
    """
    parser = argparse.ArgumentParser(description="Plot sampled PT profiles.")
    parser.add_argument("--config", default="config/fastchem_mlp_config.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-n", "--num-profiles", type=int, default=5,
                        help="Number of profiles per type (radiative, convective, Roth)")
    parser.add_argument("--num-roth", type=int, default=5,
                        help="Number of Roth PT-library profiles to draw (overrides -n for Roth)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    config = load_and_validate_config(_ROOT / args.config)
    config["_project_root"] = str(_ROOT.resolve())

    pressure_bar = sample_pressure_grid(
        num_levels=int(config["sampling"]["num_levels"]),
        pressure_top_bar=float(config["sampling"]["pressure_top_bar"]),
        pressure_bottom_bar=float(config["sampling"]["pressure_bottom_bar"]),
    )
    rng = np.random.default_rng(args.seed)

    plt.style.use(str(_STYLE))
    fig, ax = plt.subplots(figsize=(8, 8))

    # Color maps: blues for analytic, reds for Roth, purples for convective adjustment.
    blue_cmap = plt.cm.Blues
    red_cmap = plt.cm.Reds
    purple_cmap = plt.cm.Purples

    # --- Collect analytic profiles, 5 radiative and 5 convective ---
    analytic_rad, analytic_conv = [], []
    target_per_type = args.num_profiles
    max_draws = target_per_type * 30  # safety limit
    draws = 0
    while (len(analytic_rad) < target_per_type or len(analytic_conv) < target_per_type) and draws < max_draws:
        temperature_k, meta = _sample_analytic_temperature_profile_record(
            pressure_bar, config=config, rng=rng,
        )
        conv = meta.get("analytic_convective_adjustment_applied", False)
        if conv and len(analytic_conv) < target_per_type:
            analytic_conv.append(temperature_k)
        elif not conv and len(analytic_rad) < target_per_type:
            analytic_rad.append(temperature_k)
        draws += 1

    # --- Collect Roth profiles ---
    roth_temps = []
    roth_cfg = config.get("roth_sampler", {"enabled": False})
    if roth_cfg.get("enabled", False):
        data_glob = Path(str(roth_cfg["data_glob"]))
        if not data_glob.is_absolute():
            data_glob = _ROOT / data_glob
        all_files = [Path(path) for path in sorted(globmod.glob(str(data_glob)))]
        file_budget = min(max(args.num_roth, 3), len(all_files))
        selected_files = _select_diverse_roth_files(
            all_files,
            filters=roth_cfg.get("filters", {}),
            num_files=file_budget,
            rng=rng,
        )
        validation = config["temperature_profiles"]["validation"]
        roth_profiles: list[RothProfile] = []
        for path in selected_files:
            file_profiles: list[RothProfile] = []
            for profile in load_roth_profiles(
                str(path),
                pressure_grid_bar=pressure_bar,
                filters=roth_cfg.get("filters", {}),
            ):
                is_valid, _ = _validate_temperature_profile(
                    np.asarray(profile.temperature_k, dtype=np.float64),
                    validation=validation,
                )
                if is_valid:
                    file_profiles.append(profile)
            if file_profiles:
                roth_profiles.extend(
                    _select_diverse_roth_profiles(
                        file_profiles,
                        num_profiles=1,
                        rng=rng,
                    )
                )
        selected_roth = _select_diverse_roth_profiles(
            roth_profiles,
            num_profiles=min(args.num_roth, len(roth_profiles)),
            rng=rng,
        )
        roth_temps = [
            np.asarray(profile.temperature_k, dtype=np.float64)
            for profile in selected_roth
        ]

    # --- Plot: each profile gets a unique color within its category ---
    def _colors(cmap, n):
        """Return a small evenly spaced color palette sampled from a colormap.

        Parameters
        ----------
        cmap : Any
            Matplotlib colormap callable.
        n : int
            Number of colors to sample.

        Returns
        -------
        list[Any]
            Color values sampled from the interior of ``cmap``.
        """
        return [cmap(0.35 + 0.55 * i / max(n - 1, 1)) for i in range(n)]

    for i, t in enumerate(roth_temps):
        c = _colors(red_cmap, len(roth_temps))[i]
        ax.plot(t, pressure_bar, ls="-", lw=2.2, alpha=0.8, color=c,
                label="Roth PT-library" if i == 0 else None)

    for i, t in enumerate(analytic_rad):
        c = _colors(blue_cmap, len(analytic_rad))[i]
        ax.plot(t, pressure_bar, ls="--", lw=2.2, alpha=0.8, color=c,
                label="Analytic (radiative)" if i == 0 else None)

    for i, t in enumerate(analytic_conv):
        c = _colors(purple_cmap, len(analytic_conv))[i]
        ax.plot(t, pressure_bar, ls=(0, (5, 2, 1, 2)), lw=2.5, alpha=0.8, color=c,
                label="Analytic (conv. adj.)" if i == 0 else None)

    ax.set_yscale("log")
    ax.set_ylim(1e2, 1e-7)
    ax.set_xlim(0, 3000)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_title("Sampled PT Profiles")
    ax.legend(loc="best")

    if args.output:
        out = Path(args.output)
    else:
        out = _resolve_output_dir(config) / "analytic_profiles.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
