#!/usr/bin/env python3
"""Plot one Roth native PT column against its interpolated target-grid profile."""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vulcan_emulator_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "vulcan_emulator_cache"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise RuntimeError("matplotlib is required for plotting. Install project dependencies.") from exc

from config_utils import load_and_validate_config
from roth_sampling import sample_roth_profiles
from sampling import build_pressure_grid

STYLE_PATH = Path(__file__).with_name("science.mplstyle")
plt.style.use(str(STYLE_PATH))


def _parse_args() -> argparse.Namespace:
    """Parse the small plotting CLI."""
    parser = argparse.ArgumentParser(
        description="Plot one Roth native PT profile against the interpolated shared-grid profile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.json"),
        help="Relative or absolute path to the JSON config file.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index within the sampled Roth profile set to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/roth_profile_test.png"),
        help="Relative or absolute output image path.",
    )
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    """Resolve relative paths from the project root."""
    runtime_root = Path(os.path.normpath(os.environ.get("VULCAN_EMULATOR_PROJECT_ROOT", str(PROJECT_ROOT))))
    return path if path.is_absolute() else runtime_root / path


def main() -> None:
    """Render one Roth profile comparison figure."""
    args = _parse_args()
    if args.sample_index < 0:
        raise ValueError("--sample-index must be >= 0.")

    config_path = _resolve_path(args.config)
    config = load_and_validate_config(config_path)
    plot_config = copy.deepcopy(config)
    plot_config["roth_sampler"]["enabled"] = True
    plot_config["roth_sampler"]["num_profiles"] = max(
        int(plot_config["roth_sampler"]["num_profiles"]),
        args.sample_index + 1,
        1,
    )

    pressure_bar = build_pressure_grid(plot_config["tp_sampler"])
    rng = np.random.default_rng(int(plot_config["generation"]["random_seed"]) + 1)
    profiles = sample_roth_profiles(
        plot_config,
        target_pressure_bar=pressure_bar,
        rng=rng,
    )
    if args.sample_index >= len(profiles):
        raise RuntimeError(
            f"Requested sample index {args.sample_index}, but only {len(profiles)} Roth profiles were sampled."
        )
    profile = profiles[args.sample_index]

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(
        profile.native_temperature_k,
        profile.native_pressure_bar,
        color="#2A6F97",
        linewidth=2.2,
        label="Native Roth column",
    )
    ax.plot(
        profile.interpolated_temperature_k,
        pressure_bar,
        color="#C84C09",
        linewidth=2.2,
        linestyle="--",
        label="Interpolated target-grid profile",
    )
    ax.axhline(
        profile.native_pressure_min_bar,
        color="#6C757D",
        linestyle=":",
        linewidth=1.5,
        label="Native pressure bounds",
    )
    ax.axhline(
        profile.native_pressure_max_bar,
        color="#6C757D",
        linestyle=":",
        linewidth=1.5,
    )
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure (bar)")
    ax.set_title(
        "Roth PT Interpolation | "
        f"{profile.source_file} | lon={profile.lon_deg:.3f}, lat={profile.lat_deg:.3f}"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Roth PT profile plot")
    print(f"  Config path          : {config_path}")
    print(f"  Output path          : {output_path}")
    print(f"  Source file          : {profile.source_file}")
    print(f"  Source column        : lon={profile.lon_deg:.6f}, lat={profile.lat_deg:.6f}")
    print(f"  Native pressure min  : {profile.native_pressure_min_bar:.6e} bar")
    print(f"  Native pressure max  : {profile.native_pressure_max_bar:.6e} bar")
    print(f"  Extrapolated top     : {profile.extrapolated_top}")
    print(f"  Extrapolated bottom  : {profile.extrapolated_bottom}")


if __name__ == "__main__":
    main()
