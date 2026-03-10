#!/usr/bin/env python3
"""Plot several extrapolated Roth PT profiles from the configured filter space."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

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
from roth_sampling import (
    RothSamplingError,
    RothColumnProfile,
    _passes_column_filters,
    _validate_native_column,
    discover_filtered_roth_files,
    interpolate_roth_temperature,
    iter_roth_columns,
    validate_roth_filters_against_grid,
)
from sampling import build_pressure_grid

STYLE_PATH = PROJECT_ROOT / "extras" / "science.mplstyle"
plt.style.use(str(STYLE_PATH))

_DEEP_ADIABATIC_GRADIENT = 2.0 / 7.0
_DEEP_ADIABAT_PRESSURE_SPAN = 20.0


@dataclass(frozen=True)
class _SelectionStats:
    """Small debug summary for one Roth testing selection pass."""

    configured_num_profiles: int
    requested_count: int
    filtered_files: int
    scanned_files: int
    scanned_columns: int
    valid_columns: int
    extrapolated_columns: int
    selected_unique_files: int
    elapsed_seconds: float


def _parse_args() -> argparse.Namespace:
    """Parse the tiny Roth testing CLI."""
    parser = argparse.ArgumentParser(
        description="Plot 5 extrapolated Roth PT profiles using the configured Roth filters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.json"),
        help="Relative or absolute path to the JSON config file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("roth/figures/roth_testing.png"),
        help="Relative or absolute output image path. Defaults under roth/figures/ beside roth-grid/.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of extrapolated profiles to overlay.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print lightweight progress while scanning Roth files.",
    )
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    """Resolve relative paths from the project root."""
    runtime_root = Path(os.path.normpath(os.environ.get("VULCAN_EMULATOR_PROJECT_ROOT", str(PROJECT_ROOT))))
    return path if path.is_absolute() else runtime_root / path


def _debug(enabled: bool, message: str) -> None:
    """Print one debug line when requested."""
    if enabled:
        print(f"[roth_testing] {message}")


def _allowed_value_mask(values: np.ndarray, allowed: list[float]) -> np.ndarray:
    """Return an allow-list mask with the same numeric tolerance as Roth sampling."""
    if not allowed:
        return np.ones(values.shape, dtype=bool)
    mask = np.zeros(values.shape, dtype=bool)
    for candidate in allowed:
        mask |= np.isclose(values, float(candidate), atol=1.0e-9, rtol=0.0)
    return mask


def _load_roth_file_matrix(filepath: Path) -> np.ndarray | None:
    """Load one Roth file into a dense numeric matrix when the file structure is regular."""
    lines = filepath.read_text(encoding="utf-8").splitlines()
    if len(lines) <= 1:
        return None

    block_len = 0
    for line in lines[1:]:
        if line.strip():
            block_len += 1
        elif block_len > 0:
            break
    if block_len <= 0:
        return None

    cleaned = "\n".join(line for line in lines[1:] if line.strip())
    raw = np.fromstring(cleaned.replace("\n", ","), sep=",", dtype=np.float64)
    if raw.size == 0 or raw.size % 11 != 0:
        return None

    rows = raw.reshape(-1, 11)
    if rows.shape[0] % block_len != 0:
        return None
    return rows.reshape(-1, block_len, 11)


def _collect_file_candidates_fast(
    *,
    file_info: Any,
    runtime_root: Path,
    target_pressure_bar: np.ndarray,
    column_filters: dict[str, Any],
    min_source_levels: int,
) -> tuple[list[RothColumnProfile], int, int, int] | None:
    """Vectorized candidate scan for regular Roth files."""
    matrix = _load_roth_file_matrix(file_info.path)
    if matrix is None:
        return None

    scanned_columns = int(matrix.shape[0])
    if matrix.shape[1] < min_source_levels:
        return [], scanned_columns, 0, 0

    lon_deg = np.asarray(matrix[:, 0, 1], dtype=np.float64)
    lat_deg = np.asarray(matrix[:, 0, 2], dtype=np.float64)
    pressure_bar = np.asarray(matrix[:, :, 3], dtype=np.float64)
    temperature_k = np.asarray(matrix[:, :, 4], dtype=np.float64)

    lon_allowed = [float(value) for value in list(column_filters.get("lon", []))]
    lat_allowed = [float(value) for value in list(column_filters.get("lat", []))]
    column_mask = _allowed_value_mask(lon_deg, lon_allowed) & _allowed_value_mask(lat_deg, lat_allowed)
    if not np.any(column_mask):
        return [], scanned_columns, 0, 0

    lon_deg = lon_deg[column_mask]
    lat_deg = lat_deg[column_mask]
    pressure_bar = pressure_bar[column_mask]
    temperature_k = temperature_k[column_mask]

    order = np.argsort(pressure_bar, axis=1)
    pressure_sorted = np.take_along_axis(pressure_bar, order, axis=1)
    temperature_sorted = np.take_along_axis(temperature_k, order, axis=1)

    valid_mask = (
        np.all(np.isfinite(pressure_sorted), axis=1)
        & np.all(np.isfinite(temperature_sorted), axis=1)
        & np.all(pressure_sorted > 0.0, axis=1)
        & np.all(temperature_sorted > 0.0, axis=1)
        & np.all(temperature_sorted <= 2500.0, axis=1)
        & np.all(np.diff(pressure_sorted, axis=1) > 0.0, axis=1)
    )
    valid_columns = int(np.sum(valid_mask))
    if valid_columns <= 0:
        return [], scanned_columns, 0, 0

    lon_deg = lon_deg[valid_mask]
    lat_deg = lat_deg[valid_mask]
    pressure_sorted = pressure_sorted[valid_mask]
    temperature_sorted = temperature_sorted[valid_mask]

    if not np.all(np.isclose(pressure_sorted, pressure_sorted[:1], atol=1.0e-12, rtol=0.0)):
        return None

    source_log_p = np.log10(pressure_sorted[0])
    target_log_p = np.log10(np.asarray(target_pressure_bar, dtype=np.float64))
    extrapolated_top = bool(np.any(target_log_p < source_log_p[0]))
    extrapolated_bottom = bool(np.any(target_log_p > source_log_p[-1]))
    if not (extrapolated_top or extrapolated_bottom):
        return [], scanned_columns, valid_columns, 0

    interpolator = PchipInterpolator(source_log_p, temperature_sorted, axis=1, extrapolate=False)
    derivative = interpolator.derivative()
    interpolated_temperature_k = np.empty(
        (temperature_sorted.shape[0], target_log_p.size),
        dtype=np.float64,
    )
    inside = (target_log_p >= source_log_p[0]) & (target_log_p <= source_log_p[-1])
    interpolated_temperature_k[:, inside] = interpolator(target_log_p[inside])

    if extrapolated_top:
        top_offsets = target_log_p[target_log_p < source_log_p[0]] - source_log_p[0]
        top_slopes = np.asarray(derivative(source_log_p[0]), dtype=np.float64)
        interpolated_temperature_k[:, target_log_p < source_log_p[0]] = (
            temperature_sorted[:, [0]] + top_slopes[:, None] * top_offsets[None, :]
        )
    if extrapolated_bottom:
        bottom_offsets = target_log_p[target_log_p > source_log_p[-1]] - source_log_p[-1]
        bottom_slopes = np.asarray(derivative(source_log_p[-1]), dtype=np.float64)
        interpolated_temperature_k[:, target_log_p > source_log_p[-1]] = (
            temperature_sorted[:, [-1]] + bottom_slopes[:, None] * bottom_offsets[None, :]
        )

    candidate_mask = np.all(
        np.isfinite(interpolated_temperature_k)
        & (interpolated_temperature_k > 0.0)
        & (interpolated_temperature_k <= 2500.0),
        axis=1,
    )
    extrapolated_columns = int(np.sum(candidate_mask))
    if extrapolated_columns <= 0:
        return [], scanned_columns, valid_columns, 0

    candidates: list[RothColumnProfile] = []
    for row_index in np.flatnonzero(candidate_mask):
        candidates.append(
            RothColumnProfile(
                source_file=file_info.path.name,
                source_relpath=str(file_info.path.relative_to(runtime_root)),
                metadata=dict(file_info.metadata),
                planet_mass_jup=float(file_info.planet_mass_jup),
                lon_deg=float(lon_deg[row_index]),
                lat_deg=float(lat_deg[row_index]),
                native_pressure_bar=pressure_sorted[row_index].copy(),
                native_temperature_k=temperature_sorted[row_index].copy(),
                interpolated_temperature_k=interpolated_temperature_k[row_index].copy(),
                native_pressure_min_bar=float(np.min(pressure_sorted[row_index])),
                native_pressure_max_bar=float(np.max(pressure_sorted[row_index])),
                extrapolated_top=extrapolated_top,
                extrapolated_bottom=extrapolated_bottom,
            )
        )
    return candidates, scanned_columns, valid_columns, extrapolated_columns


def _collect_file_candidates_slow(
    *,
    file_info: Any,
    runtime_root: Path,
    target_pressure_bar: np.ndarray,
    column_filters: dict[str, Any],
    min_source_levels: int,
) -> tuple[list[RothColumnProfile], int, int, int]:
    """Fallback per-column scan for irregular Roth files."""
    candidates: list[RothColumnProfile] = []
    scanned_columns = 0
    valid_columns = 0
    extrapolated_columns = 0

    for lon_deg, lat_deg, native_pressure_bar, native_temperature_k in iter_roth_columns(file_info.path):
        scanned_columns += 1
        if not _passes_column_filters(
            lon_deg=lon_deg,
            lat_deg=lat_deg,
            column_filters=column_filters,
        ):
            continue
        validated = _validate_native_column(
            pressure_bar=native_pressure_bar,
            temperature_k=native_temperature_k,
            min_source_levels=min_source_levels,
        )
        if validated is None:
            continue
        valid_columns += 1
        try:
            interpolated_temperature_k, extrapolated_top, extrapolated_bottom = interpolate_roth_temperature(
                pressure_bar=validated[0],
                temperature_k=validated[1],
                target_pressure_bar=target_pressure_bar,
            )
        except RothSamplingError:
            continue
        if not (bool(extrapolated_top) or bool(extrapolated_bottom)):
            continue

        extrapolated_columns += 1
        candidates.append(
            RothColumnProfile(
                source_file=file_info.path.name,
                source_relpath=str(file_info.path.relative_to(runtime_root)),
                metadata=dict(file_info.metadata),
                planet_mass_jup=float(file_info.planet_mass_jup),
                lon_deg=float(lon_deg),
                lat_deg=float(lat_deg),
                native_pressure_bar=validated[0].copy(),
                native_temperature_k=validated[1].copy(),
                interpolated_temperature_k=interpolated_temperature_k,
                native_pressure_min_bar=float(np.min(validated[0])),
                native_pressure_max_bar=float(np.max(validated[0])),
                extrapolated_top=bool(extrapolated_top),
                extrapolated_bottom=bool(extrapolated_bottom),
            )
        )

    return candidates, scanned_columns, valid_columns, extrapolated_columns


def _select_extrapolated_profiles(
    *,
    config: dict,
    count: int,
    debug: bool,
) -> tuple[np.ndarray, list[RothColumnProfile], _SelectionStats]:
    """Scan Roth columns and stop once enough extrapolated profiles are found."""
    if count <= 0:
        raise ValueError("--count must be > 0.")

    roth_cfg = dict(config["roth_sampler"])
    if not bool(roth_cfg["enabled"]):
        raise RuntimeError("roth_sampler.enabled must be true for roth_testing.py.")

    started_at = time.perf_counter()
    runtime_root = Path(os.path.normpath(os.environ.get("VULCAN_EMULATOR_PROJECT_ROOT", str(PROJECT_ROOT))))
    pressure_bar = build_pressure_grid(config["tp_sampler"])
    validate_roth_filters_against_grid(roth_cfg, project_root=runtime_root)
    files = discover_filtered_roth_files(roth_cfg, project_root=runtime_root)
    filtered_files = len(files)
    rng = np.random.default_rng(int(config["generation"]["random_seed"]) + 17)
    column_filters = dict(roth_cfg["column_filters"])
    min_source_levels = int(roth_cfg["interpolation"]["min_source_levels"])

    _debug(
        debug,
        (
            f"configured roth_sampler.num_profiles={int(roth_cfg['num_profiles'])}; "
            f"plot only needs {count} extrapolated profiles, so this script scans files in deterministic order and stops once enough candidates exist."
        ),
    )
    _debug(
        debug,
        "selection favors variety by picking one extrapolated column per file first, then filling from extra columns if needed.",
    )
    _debug(debug, f"filtered Roth files={filtered_files}")

    selected: list[RothColumnProfile] = []
    overflow_candidates: list[RothColumnProfile] = []
    overflow_seen = 0
    scanned_files = 0
    scanned_columns = 0
    valid_columns = 0
    extrapolated_columns = 0

    for file_info in files:
        scanned_files += 1
        fast_result = _collect_file_candidates_fast(
            file_info=file_info,
            runtime_root=runtime_root,
            target_pressure_bar=pressure_bar,
            column_filters=column_filters,
            min_source_levels=min_source_levels,
        )
        if fast_result is None:
            file_candidates, file_scanned_columns, file_valid_columns, file_extrapolated_columns = (
                _collect_file_candidates_slow(
                    file_info=file_info,
                    runtime_root=runtime_root,
                    target_pressure_bar=pressure_bar,
                    column_filters=column_filters,
                    min_source_levels=min_source_levels,
                )
            )
        else:
            file_candidates, file_scanned_columns, file_valid_columns, file_extrapolated_columns = fast_result
        scanned_columns += file_scanned_columns
        valid_columns += file_valid_columns
        extrapolated_columns += file_extrapolated_columns

        if file_candidates:
            chosen_index = int(rng.integers(0, len(file_candidates)))
            selected.append(file_candidates[chosen_index])
            if len(selected) < count:
                for candidate_index, candidate in enumerate(file_candidates):
                    if candidate_index == chosen_index:
                        continue
                    overflow_seen += 1
                    if len(overflow_candidates) < count:
                        overflow_candidates.append(candidate)
                    else:
                        replace_index = int(rng.integers(0, overflow_seen))
                        if replace_index < count:
                            overflow_candidates[replace_index] = candidate

        if debug and (
            scanned_files == 1
            or scanned_files % 25 == 0
            or len(selected) >= count
            or len(selected) + len(overflow_candidates) >= count
        ):
            elapsed = time.perf_counter() - started_at
            _debug(
                True,
                (
                    f"scanned_files={scanned_files}/{filtered_files} "
                    f"scanned_columns={scanned_columns} valid_columns={valid_columns} "
                    f"extrapolated_columns={extrapolated_columns} selected={len(selected)} "
                    f"overflow_candidates={len(overflow_candidates)} "
                    f"elapsed={elapsed:.2f}s"
                ),
            )
        if len(selected) >= count or len(selected) + len(overflow_candidates) >= count:
            break

    if len(selected) < count and overflow_candidates:
        for candidate_index in rng.permutation(len(overflow_candidates)):
            selected.append(overflow_candidates[int(candidate_index)])
            if len(selected) >= count:
                break

    stats = _SelectionStats(
        configured_num_profiles=int(roth_cfg["num_profiles"]),
        requested_count=count,
        filtered_files=filtered_files,
        scanned_files=scanned_files,
        scanned_columns=scanned_columns,
        valid_columns=valid_columns,
        extrapolated_columns=extrapolated_columns,
        selected_unique_files=len({profile.source_file for profile in selected}),
        elapsed_seconds=time.perf_counter() - started_at,
    )

    if len(selected) < count:
        raise RuntimeError(
            f"Only found {len(selected)} extrapolated Roth profiles after scanning "
            f"{scanned_files}/{filtered_files} files and {scanned_columns} columns; need {count}."
        )
    return pressure_bar, selected, stats


def _plot_deep_adiabat_reference(
    ax: Any,
    *,
    pressure_bar: np.ndarray,
    profiles: list[RothColumnProfile],
) -> None:
    """Plot a short deep-adiabat reference segment on the PT figure."""
    if not profiles:
        return

    deepest_index = int(np.argmax(pressure_bar))
    anchor_temperature_k = float(
        np.median([profile.interpolated_temperature_k[deepest_index] for profile in profiles])
    )
    bottom_pressure_bar = float(np.max(pressure_bar))
    top_pressure_bar = max(float(np.min(pressure_bar)), bottom_pressure_bar / _DEEP_ADIABAT_PRESSURE_SPAN)
    if not top_pressure_bar < bottom_pressure_bar:
        return

    reference_pressure_bar = np.geomspace(bottom_pressure_bar, top_pressure_bar, 32)
    reference_temperature_k = anchor_temperature_k * (
        reference_pressure_bar / bottom_pressure_bar
    ) ** _DEEP_ADIABATIC_GRADIENT
    ax.plot(
        reference_temperature_k,
        reference_pressure_bar,
        color="#303030",
        linestyle=(0, (5, 3)),
        linewidth=1.8,
        alpha=0.9,
        label="Deep adiabat (~2/7)",
    )


def main() -> None:
    """Render one multi-profile Roth testing figure."""
    args = _parse_args()
    config_path = _resolve_path(args.config)
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_and_validate_config(config_path)
    pressure_bar, profiles, stats = _select_extrapolated_profiles(
        config=config,
        count=int(args.count),
        debug=bool(args.debug),
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap("tab10")
    interpolation_line_labeled = False
    for idx, profile in enumerate(profiles):
        ax.plot(
            np.clip(profile.interpolated_temperature_k, 1.0, None),
            np.clip(pressure_bar, 1.0e-30, None),
            linewidth=2.0,
            color=cmap(idx % 10),
            label=(
                f"{Path(profile.source_file).stem} | "
                f"lon={profile.lon_deg:.1f}, lat={profile.lat_deg:.1f}"
            ),
        )
        for interpolation_start_bar in (
            profile.native_pressure_min_bar if profile.extrapolated_top else None,
            profile.native_pressure_max_bar if profile.extrapolated_bottom else None,
        ):
            if interpolation_start_bar is None:
                continue
            ax.axhline(
                float(interpolation_start_bar),
                color="#7A7A7A",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
                label="Interpolation start" if not interpolation_line_labeled else None,
            )
            interpolation_line_labeled = True

    _plot_deep_adiabat_reference(ax, pressure_bar=pressure_bar, profiles=profiles)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure (bar)")
    ax.set_title("Extrapolated Roth PT Profiles")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Roth testing plot")
    print(f"  Config path  : {config_path}")
    print(f"  Output path  : {output_path}")
    print(f"  Profiles     : {len(profiles)}")
    print("  Debug")
    print(f"    configured_num_profiles : {stats.configured_num_profiles}")
    print(f"    requested_count         : {stats.requested_count}")
    print(f"    filtered_files          : {stats.filtered_files}")
    print(f"    scanned_files           : {stats.scanned_files}")
    print(f"    scanned_columns         : {stats.scanned_columns}")
    print(f"    valid_columns           : {stats.valid_columns}")
    print(f"    extrapolated_columns    : {stats.extrapolated_columns}")
    print(f"    selected_unique_files   : {stats.selected_unique_files}")
    print(f"    elapsed_seconds         : {stats.elapsed_seconds:.2f}")
    for idx, profile in enumerate(profiles, start=1):
        print(
            f"  {idx}: {profile.source_file} | lon={profile.lon_deg:.3f}, lat={profile.lat_deg:.3f} | "
            f"top={profile.extrapolated_top} bottom={profile.extrapolated_bottom}"
        )


if __name__ == "__main__":
    main()
