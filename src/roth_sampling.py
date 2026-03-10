"""Roth GCM profile discovery, filtering, sampling, and interpolation."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.interpolate import PchipInterpolator

ROTH_FILENAME_PATTERN = re.compile(
    r"Teq_([\d.]+)-"
    r"LogMet_([-]?[\d.]+)-"
    r"LogDrag_([-]?[\d.]+)-"
    r"Mstar_([\d.]+)-"
    r"Rp_([\d.]+)-"
    r"logG_([-]?[\d.]+)-"
    r"TiOVO_(true|false)"
)
ROTH_FILTER_KEYS = ("Teq", "LogMet", "LogDrag", "Mstar", "Rp", "logG", "TiOVO")
ROTH_OPTIONAL_FILTER_KEYS = ("planet_mass_jup",)
ROTH_COLUMN_FILTER_KEYS = ("lon", "lat")

_GRAVITATIONAL_CONSTANT = 6.67430e-11
_JUPITER_RADIUS_M = 7.1492e7
_JUPITER_MASS_KG = 1.89813e27


class RothSamplingError(ValueError):
    """Raised when Roth grid data or controls violate the expected contract."""


@dataclass(frozen=True)
class RothFileInfo:
    """One Roth grid file plus its parsed filename metadata."""

    path: Path
    metadata: dict[str, float | bool]
    planet_mass_jup: float


@dataclass(frozen=True)
class RothColumnProfile:
    """One selected Roth lon/lat column on both native and target grids."""

    source_file: str
    source_relpath: str
    metadata: dict[str, float | bool]
    planet_mass_jup: float
    lon_deg: float
    lat_deg: float
    native_pressure_bar: np.ndarray
    native_temperature_k: np.ndarray
    interpolated_temperature_k: np.ndarray
    native_pressure_min_bar: float
    native_pressure_max_bar: float
    extrapolated_top: bool
    extrapolated_bottom: bool


def _project_root() -> Path:
    """Resolve the project root consistently with the rest of the repo."""
    env_root = os.environ.get("VULCAN_EMULATOR_PROJECT_ROOT")
    if env_root:
        return Path(os.path.normpath(env_root))
    return Path(__file__).resolve().parent.parent


def derive_planet_mass_jup(*, rp_rjup: float, logg_cgs: float) -> float:
    """Derive planet mass in Jupiter masses from radius and log10(g) in cgs."""
    gravity_m_s2 = (10.0 ** float(logg_cgs)) / 100.0
    radius_m = float(rp_rjup) * _JUPITER_RADIUS_M
    mass_kg = gravity_m_s2 * (radius_m * radius_m) / _GRAVITATIONAL_CONSTANT
    return float(mass_kg / _JUPITER_MASS_KG)


def parse_roth_filename(filepath: str | Path) -> dict[str, float | bool] | None:
    """Parse Roth filename metadata from one PT grid file path."""
    path = Path(filepath)
    match = ROTH_FILENAME_PATTERN.search(path.name)
    if match is None:
        return None
    return {
        "Teq": float(match.group(1)),
        "LogMet": float(match.group(2)),
        "LogDrag": float(match.group(3)),
        "Mstar": float(match.group(4)),
        "Rp": float(match.group(5)),
        "logG": float(match.group(6)),
        "TiOVO": match.group(7).lower() == "true",
    }


def _resolve_roth_paths(data_glob: str, *, project_root: Path | None = None) -> list[Path]:
    """Resolve Roth grid files from the configured relative glob."""
    root = _project_root() if project_root is None else project_root
    return sorted((root).glob(data_glob))


def discover_roth_filter_values(
    data_glob: str,
    *,
    project_root: Path | None = None,
) -> dict[str, list[float | bool]]:
    """Discover the metadata values currently present in the Roth grid filenames."""
    values: dict[str, set[float | bool]] = {
        key: set() for key in (*ROTH_FILTER_KEYS, *ROTH_OPTIONAL_FILTER_KEYS)
    }
    found_any = False
    for path in _resolve_roth_paths(data_glob, project_root=project_root):
        metadata = parse_roth_filename(path)
        if metadata is None:
            continue
        found_any = True
        for key in ROTH_FILTER_KEYS:
            values[key].add(metadata[key])
        values["planet_mass_jup"].add(
            derive_planet_mass_jup(
                rp_rjup=float(metadata["Rp"]),
                logg_cgs=float(metadata["logG"]),
            )
        )
    if not found_any:
        raise RothSamplingError(f"No parseable Roth grid files matched '{data_glob}'.")
    return {
        key: sorted(
            value_set,
            key=lambda item: (int(isinstance(item, bool)), item),
        )
        for key, value_set in values.items()
    }


def _numeric_allowed(value: float, allowed: list[float], *, atol: float = 1.0e-9) -> bool:
    """Return True when *value* matches one of the allowed numeric values."""
    return any(abs(float(value) - float(candidate)) <= atol for candidate in allowed)


def _validate_subset_allowlist(
    *,
    values: list[float | bool],
    allowed: list[float | bool],
    field: str,
) -> None:
    """Require configured allow-list values to be a subset of discovered Roth values."""
    allowed_numeric = [float(item) for item in allowed if not isinstance(item, bool)]
    allowed_bools = [bool(item) for item in allowed if isinstance(item, bool)]
    for value in values:
        if isinstance(value, bool):
            if bool(value) not in allowed_bools:
                raise RothSamplingError(
                    f"{field} contains {value}, which is not present in the Roth grid."
                )
            continue
        if not _numeric_allowed(float(value), allowed_numeric):
            raise RothSamplingError(
                f"{field} contains {value}, which is not present in the Roth grid."
            )


def validate_roth_filters_against_grid(
    roth_cfg: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> None:
    """Validate configured Roth allow-lists against the discovered grid values."""
    discovered = discover_roth_filter_values(
        str(roth_cfg["data_glob"]),
        project_root=project_root,
    )
    filters = dict(roth_cfg["filters"])
    for key in ROTH_FILTER_KEYS:
        configured = list(filters.get(key, []))
        if key == "TiOVO":
            _validate_subset_allowlist(
                values=[bool(item) for item in configured],
                allowed=list(discovered[key]),
                field=f"roth_sampler.filters.{key}",
            )
        else:
            _validate_subset_allowlist(
                values=[float(item) for item in configured],
                allowed=list(discovered[key]),
                field=f"roth_sampler.filters.{key}",
            )
    _validate_subset_allowlist(
        values=[float(item) for item in list(filters.get("planet_mass_jup", []))],
        allowed=list(discovered["planet_mass_jup"]),
        field="roth_sampler.filters.planet_mass_jup",
    )


def _matches_file_filters(file_info: RothFileInfo, filters: dict[str, Any]) -> bool:
    """Return True when a Roth file passes the configured metadata allow-lists."""
    for key in ROTH_FILTER_KEYS:
        allowed = list(filters.get(key, []))
        if not allowed:
            continue
        value = file_info.metadata[key]
        if isinstance(value, bool):
            if bool(value) not in [bool(item) for item in allowed]:
                return False
            continue
        if not _numeric_allowed(float(value), [float(item) for item in allowed]):
            return False
    allowed_masses = list(filters.get("planet_mass_jup", []))
    if allowed_masses and not _numeric_allowed(
        file_info.planet_mass_jup,
        [float(item) for item in allowed_masses],
    ):
        return False
    return True


def discover_filtered_roth_files(
    roth_cfg: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> list[RothFileInfo]:
    """Resolve Roth grid files and apply filename-level allow-list filters."""
    data_glob = str(roth_cfg["data_glob"])
    filters = dict(roth_cfg["filters"])
    filtered: list[RothFileInfo] = []
    for path in _resolve_roth_paths(data_glob, project_root=project_root):
        metadata = parse_roth_filename(path)
        if metadata is None:
            continue
        file_info = RothFileInfo(
            path=path,
            metadata=metadata,
            planet_mass_jup=derive_planet_mass_jup(
                rp_rjup=float(metadata["Rp"]),
                logg_cgs=float(metadata["logG"]),
            ),
        )
        if _matches_file_filters(file_info, filters):
            filtered.append(file_info)
    if not filtered:
        raise RothSamplingError(
            "No Roth grid files remained after applying the configured filename filters."
        )
    return filtered


def _passes_column_filters(
    *,
    lon_deg: float,
    lat_deg: float,
    column_filters: dict[str, Any],
) -> bool:
    """Apply optional lon/lat allow-lists to one Roth column."""
    lon_allowed = [float(value) for value in list(column_filters.get("lon", []))]
    lat_allowed = [float(value) for value in list(column_filters.get("lat", []))]
    if lon_allowed and not _numeric_allowed(lon_deg, lon_allowed):
        return False
    if lat_allowed and not _numeric_allowed(lat_deg, lat_allowed):
        return False
    return True


def iter_roth_columns(filepath: Path) -> Iterator[tuple[float, float, np.ndarray, np.ndarray]]:
    """Yield one native Roth PT column at a time from a grid file."""
    current_lon: float | None = None
    current_lat: float | None = None
    pressure_bar: list[float] = []
    temperature_k: list[float] = []

    with filepath.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            tokens = [token for token in re.split(r"[,\s]+", stripped) if token]
            if len(tokens) < 5:
                continue
            try:
                lon_deg = float(tokens[1])
                lat_deg = float(tokens[2])
                pressure_value = float(tokens[3])
                temperature_value = float(tokens[4])
            except ValueError:
                continue

            if current_lon is None:
                current_lon = lon_deg
                current_lat = lat_deg
            elif lon_deg != current_lon or lat_deg != current_lat:
                yield (
                    float(current_lon),
                    float(current_lat),
                    np.asarray(pressure_bar, dtype=np.float64),
                    np.asarray(temperature_k, dtype=np.float64),
                )
                current_lon = lon_deg
                current_lat = lat_deg
                pressure_bar = []
                temperature_k = []

            pressure_bar.append(pressure_value)
            temperature_k.append(temperature_value)

    if current_lon is not None:
        yield (
            float(current_lon),
            float(current_lat),
            np.asarray(pressure_bar, dtype=np.float64),
            np.asarray(temperature_k, dtype=np.float64),
        )


def _validate_native_column(
    *,
    pressure_bar: np.ndarray,
    temperature_k: np.ndarray,
    min_source_levels: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Validate and sort one native Roth PT column."""
    if pressure_bar.size < min_source_levels or temperature_k.size != pressure_bar.size:
        return None
    if np.any(~np.isfinite(pressure_bar)) or np.any(~np.isfinite(temperature_k)):
        return None
    if np.any(pressure_bar <= 0.0) or np.any(temperature_k <= 0.0):
        return None

    order = np.argsort(pressure_bar)
    pressure_sorted = pressure_bar[order]
    temperature_sorted = temperature_k[order]
    if np.any(np.diff(pressure_sorted) <= 0.0):
        return None
    if pressure_sorted.size < min_source_levels:
        return None
    return pressure_sorted, temperature_sorted


def interpolate_roth_temperature(
    *,
    pressure_bar: np.ndarray,
    temperature_k: np.ndarray,
    target_pressure_bar: np.ndarray,
) -> tuple[np.ndarray, bool, bool]:
    """Interpolate one Roth PT column onto the target grid with log-pressure extrapolation."""
    if pressure_bar.ndim != 1 or temperature_k.ndim != 1:
        raise RothSamplingError("Roth interpolation expects one-dimensional source arrays.")
    if pressure_bar.size != temperature_k.size:
        raise RothSamplingError("Roth source pressure and temperature lengths must match.")
    if pressure_bar.size < 2:
        raise RothSamplingError("Roth interpolation requires at least two source levels.")

    source_log_p = np.log10(pressure_bar)
    target_log_p = np.log10(np.asarray(target_pressure_bar, dtype=np.float64))
    interpolator = PchipInterpolator(source_log_p, temperature_k, extrapolate=False)
    derivative = interpolator.derivative()

    result = np.empty_like(target_log_p, dtype=np.float64)
    inside = (target_log_p >= source_log_p[0]) & (target_log_p <= source_log_p[-1])
    result[inside] = interpolator(target_log_p[inside])

    extrapolated_top = bool(np.any(target_log_p < source_log_p[0]))
    extrapolated_bottom = bool(np.any(target_log_p > source_log_p[-1]))

    if extrapolated_top:
        slope_top = float(derivative(source_log_p[0]))
        result[target_log_p < source_log_p[0]] = temperature_k[0] + slope_top * (
            target_log_p[target_log_p < source_log_p[0]] - source_log_p[0]
        )
    if extrapolated_bottom:
        slope_bottom = float(derivative(source_log_p[-1]))
        result[target_log_p > source_log_p[-1]] = temperature_k[-1] + slope_bottom * (
            target_log_p[target_log_p > source_log_p[-1]] - source_log_p[-1]
        )

    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise RothSamplingError("Interpolated Roth temperature profile is invalid.")
    return result.astype(np.float64), extrapolated_top, extrapolated_bottom


def sample_roth_profiles(
    config: dict[str, Any],
    *,
    target_pressure_bar: np.ndarray,
    rng: np.random.Generator,
    project_root: Path | None = None,
) -> list[RothColumnProfile]:
    """Sample Roth PT columns uniformly from the valid filtered column pool."""
    roth_cfg = config["roth_sampler"]
    target_count = int(roth_cfg["num_profiles"])
    if not bool(roth_cfg["enabled"]) or target_count <= 0:
        return []

    validate_roth_filters_against_grid(roth_cfg, project_root=project_root)
    files = discover_filtered_roth_files(roth_cfg, project_root=project_root)
    column_filters = dict(roth_cfg["column_filters"])
    min_source_levels = int(roth_cfg["interpolation"]["min_source_levels"])

    reservoir: list[
        tuple[
            RothFileInfo,
            float,
            float,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            bool,
            bool,
        ]
    ] = []
    valid_columns = 0

    for file_info in files:
        for lon_deg, lat_deg, native_pressure_bar, native_temperature_k in iter_roth_columns(
            file_info.path
        ):
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
            try:
                interpolated_temperature_k, extrapolated_top, extrapolated_bottom = (
                    interpolate_roth_temperature(
                        pressure_bar=validated[0],
                        temperature_k=validated[1],
                        target_pressure_bar=target_pressure_bar,
                    )
                )
            except RothSamplingError:
                continue
            valid_columns += 1
            candidate = (
                file_info,
                float(lon_deg),
                float(lat_deg),
                validated[0],
                validated[1],
                interpolated_temperature_k,
                bool(extrapolated_top),
                bool(extrapolated_bottom),
            )
            if len(reservoir) < target_count:
                reservoir.append(candidate)
            else:
                replace_index = int(rng.integers(0, valid_columns))
                if replace_index < target_count:
                    reservoir[replace_index] = candidate

    if valid_columns < target_count:
        raise RothSamplingError(
            "Requested "
            f"{target_count} Roth profiles, but only {valid_columns} valid filtered columns exist."
        )

    root = _project_root() if project_root is None else project_root
    selected_profiles: list[RothColumnProfile] = []
    for (
        file_info,
        lon_deg,
        lat_deg,
        native_pressure_bar,
        native_temperature_k,
        interpolated_temperature_k,
        extrapolated_top,
        extrapolated_bottom,
    ) in reservoir:
        selected_profiles.append(
            RothColumnProfile(
                source_file=file_info.path.name,
                source_relpath=str(file_info.path.relative_to(root)),
                metadata=dict(file_info.metadata),
                planet_mass_jup=float(file_info.planet_mass_jup),
                lon_deg=float(lon_deg),
                lat_deg=float(lat_deg),
                native_pressure_bar=native_pressure_bar.copy(),
                native_temperature_k=native_temperature_k.copy(),
                interpolated_temperature_k=interpolated_temperature_k,
                native_pressure_min_bar=float(np.min(native_pressure_bar)),
                native_pressure_max_bar=float(np.max(native_pressure_bar)),
                extrapolated_top=bool(extrapolated_top),
                extrapolated_bottom=bool(extrapolated_bottom),
            )
        )
    return selected_profiles
