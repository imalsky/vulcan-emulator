"""PT-library profile loading, column extraction, and metadata filtering.

This module loads pre-computed temperature-pressure profiles from the
Roth et al. PT library (or any compatible tabular source) and
interpolates them onto the emulator's fixed log-pressure grid.

Supported source formats:

* ``.dat`` — multi-column CSV with ``(index, lon, lat, pressure, temperature)``
  rows.  One ``.dat`` file may contain multiple (lon, lat) columns, each
  expanded into a separate ``RothProfile``.  Atmospheric parameters are
  encoded in the filename (e.g., ``Teq_1500-LogMet_0.0-...``).
* ``.npz`` / ``.json`` / ``.csv`` — single-profile tabular formats with
  ``pressure_bar`` and ``temperature_k`` arrays (and optional metadata).

All profiles are interpolated onto the target grid using a monotone
PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) in
log-pressure space, with edge clamping to avoid extrapolation artefacts.
"""

from __future__ import annotations

import csv
import glob
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..constants import ROTH_NUMERIC_FILTER_KEYS

RothFilterValue = Union[float, Tuple[float, float], bool]
RothFilterConfig = Dict[str, RothFilterValue]
_PT_PROFILE_FILENAME_PATTERN = re.compile(
    r"Teq_(?P<Teq>[\d.]+)-"
    r"LogMet_(?P<LogMet>[-]?[\d.]+)-"
    r"LogDrag_(?P<LogDrag>[-]?[\d.]+)-"
    r"Mstar_(?P<Mstar>[\d.]+)-"
    r"Rp_(?P<Rp>[\d.]+)-"
    r"logG_(?P<logG>[-]?[\d.]+)-"
    r"TiOVO_(?P<TiOVO>true|false)"
)
# Matches a line that begins with a signed decimal (integer index column of a
# data row). Used to skip header/metadata lines in Roth .dat files.
_ROW_LEADING_NUMERIC_RE = re.compile(r"^-?\d")


@dataclass(frozen=True)
class RothProfile:
    """One temperature-pressure profile loaded from the configured PT-library source.

    Attributes
    ----------
    pressure_bar : 1-D array
        Pressure levels in bar.
    temperature_k : 1-D array
        Temperature at each pressure level in Kelvin.
    metadata : dict
        Source parameters parsed from the filename (Teq, LogMet, etc.)
        plus ``source_file`` for provenance.
    """
    pressure_bar: np.ndarray
    temperature_k: np.ndarray
    metadata: dict[str, Any]


def _prepare_profile_coordinates(
    source_pressure_bar: np.ndarray,
    source_temperature_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate, sort, and deduplicate one source profile in log-pressure space.

    Parameters
    ----------
    source_pressure_bar : np.ndarray
        Source pressure samples in bar.
    source_temperature_k : np.ndarray
        Source temperature samples in Kelvin aligned with
        ``source_pressure_bar``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Deduplicated ``(log10_pressure, temperature)`` arrays sorted in
        ascending pressure order and ready for interpolation.
    """
    pressure = np.asarray(source_pressure_bar, dtype=np.float64)
    temperature = np.asarray(source_temperature_k, dtype=np.float64)
    finite_mask = np.isfinite(pressure) & np.isfinite(temperature) & (pressure > 0.0)
    if np.count_nonzero(finite_mask) < 2:
        raise ValueError("Roth profile must contain at least two finite pressure-temperature samples.")

    log_pressure = np.log10(pressure[finite_mask])
    temperature = temperature[finite_mask]
    sort_idx = np.argsort(log_pressure)
    log_pressure = log_pressure[sort_idx]
    temperature = temperature[sort_idx]

    unique_log_pressure, inverse = np.unique(log_pressure, return_inverse=True)
    if unique_log_pressure.size != log_pressure.size:
        summed_temperature = np.zeros(unique_log_pressure.shape, dtype=np.float64)
        counts = np.zeros(unique_log_pressure.shape, dtype=np.float64)
        np.add.at(summed_temperature, inverse, temperature)
        np.add.at(counts, inverse, 1.0)
        temperature = summed_temperature / counts
        log_pressure = unique_log_pressure

    if log_pressure.size < 2:
        raise ValueError("Roth profile collapses to fewer than two unique pressure levels.")
    return log_pressure, temperature


def _pchip_interpolate(
    x_target: np.ndarray,
    x_source: np.ndarray,
    y_source: np.ndarray,
) -> np.ndarray:
    """Evaluate a shape-preserving cubic Hermite interpolant with edge clamping.

    Thin wrapper over :class:`scipy.interpolate.PchipInterpolator` (same
    Fritsch-Carlson algorithm). Evaluation outside ``[x_source[0],
    x_source[-1]]`` is replaced with the nearest endpoint value to avoid
    cubic extrapolation artefacts.
    """
    pchip = PchipInterpolator(x_source, y_source, extrapolate=False)
    result = pchip(x_target)
    result = np.where(x_target <= x_source[0], y_source[0], result)
    result = np.where(x_target >= x_source[-1], y_source[-1], result)
    return result


def _interpolate_profile(
    pressure_bar: np.ndarray,
    source_pressure_bar: np.ndarray,
    source_temperature_k: np.ndarray,
) -> np.ndarray:
    """Interpolate a source profile onto the emulator's fixed pressure grid.

    Parameters
    ----------
    pressure_bar : np.ndarray
        Target emulator pressure grid in bar.
    source_pressure_bar : np.ndarray
        Source profile pressure samples in bar.
    source_temperature_k : np.ndarray
        Source profile temperature samples in Kelvin.

    Returns
    -------
    np.ndarray
        Temperature profile interpolated onto ``pressure_bar``.
    """
    log_target = np.log10(pressure_bar)
    log_source, temperature = _prepare_profile_coordinates(
        source_pressure_bar,
        source_temperature_k,
    )
    return _pchip_interpolate(log_target, log_source, temperature)


def _parse_pt_profile_filename(path: Path) -> dict[str, float | bool | str]:
    """Parse PT-library source parameters from one ``.dat`` filename.

    Parameters
    ----------
    path : Path
        PT-library source file path whose filename encodes the metadata.

    Returns
    -------
    dict[str, float | bool | str]
        Parsed metadata mapping containing the numeric filter keys, ``TiOVO``,
        and ``source_file`` for provenance.
    """
    match = _PT_PROFILE_FILENAME_PATTERN.search(path.name)
    if match is None:
        raise ValueError(
            f"PT profile filename {path.name!r} does not match the expected naming convention."
        )
    metadata: dict[str, float | bool | str] = {"source_file": str(path)}
    for key in ROTH_NUMERIC_FILTER_KEYS:
        metadata[key] = float(match.group(key))
    metadata["TiOVO"] = match.group("TiOVO").lower() == "true"
    return metadata


def _matches_filters(metadata: dict[str, Any], filters: RothFilterConfig | None) -> bool:
    """Return whether one profile metadata record satisfies the filter config.

    Parameters
    ----------
    metadata : dict[str, Any]
        Parsed PT-profile metadata, typically from the filename or serialized
        sidecar fields.
    filters : RothFilterConfig or None
        Optional numeric or boolean constraints keyed by PT-library metadata
        name.

    Returns
    -------
    bool
        ``True`` when all configured constraints are satisfied.
    """
    for key, requirement in (filters or {}).items():
        if key not in metadata:
            return False
        value = metadata[key]
        if isinstance(requirement, bool):
            if bool(value) is not requirement:
                return False
            continue
        numeric_value = float(value)
        if isinstance(requirement, tuple):
            lower, upper = requirement
            if numeric_value < lower or numeric_value > upper:
                return False
            continue
        if not np.isclose(numeric_value, float(requirement)):
            return False
    return True


def _load_pt_profile_rows(path: Path) -> np.ndarray:
    """Load the numeric rows from one Roth PT-library ``.dat`` file.

    Parameters
    ----------
    path : Path
        Source CSV-like file containing profile rows with at least five
        numeric columns.

    Returns
    -------
    np.ndarray
        Two-dimensional float array containing the numeric block extracted
        from the file.
    """
    numeric_lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.count(",") < 4:
                # Roth PT ``.dat`` files may begin with a short (e.g. 3-column)
                # header row; data rows always have at least five comma-separated
                # columns (index, lon, lat, pressure, temperature).
                continue
            if not _ROW_LEADING_NUMERIC_RE.match(stripped):
                continue
            numeric_lines.append(stripped)
    if not numeric_lines:
        raise ValueError(f"PT profile file {path} does not contain any numeric rows.")
    rows = np.genfromtxt(
        io.StringIO("\n".join(numeric_lines)),
        delimiter=",",
        dtype=np.float64,
    )
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] < 5:
        raise ValueError(f"PT profile file {path} does not contain the expected columns.")
    return np.asarray(rows, dtype=np.float64)


def _load_pt_dat_profiles(path: Path) -> list[RothProfile]:
    """Expand one PT-library ``.dat`` file into one candidate profile per ``(lon, lat)`` column.

    Parameters
    ----------
    path : Path
        PT-library source file containing one or more ``(lon, lat)`` profile
        columns.

    Returns
    -------
    list[RothProfile]
        Extracted profile records, one per unique ``(lon, lat)`` pair in the
        file.
    """
    rows = _load_pt_profile_rows(path)
    base_metadata = _parse_pt_profile_filename(path)
    lon_lat = np.asarray(rows[:, 1:3], dtype=np.float64)
    unique_columns = np.unique(lon_lat, axis=0)
    profiles: list[RothProfile] = []
    for lon, lat in unique_columns:
        column_mask = (lon_lat[:, 0] == lon) & (lon_lat[:, 1] == lat)
        column_rows = rows[column_mask]
        sort_indices = np.argsort(column_rows[:, 3])
        source_pressure_bar = np.asarray(column_rows[sort_indices, 3], dtype=np.float64)
        source_temperature_k = np.asarray(column_rows[sort_indices, 4], dtype=np.float64)
        profiles.append(
            RothProfile(
                pressure_bar=source_pressure_bar,
                temperature_k=source_temperature_k,
                metadata={
                    **base_metadata,
                    "lon": float(lon),
                    "lat": float(lat),
                },
            )
        )
    return profiles


def _load_tabular_roth_profile(path: Path) -> RothProfile:
    """Load one tabular Roth profile from NPZ, JSON, or CSV storage.

    Parameters
    ----------
    path : Path
        Source file containing ``pressure_bar`` and ``temperature_k`` arrays,
        plus optional metadata.

    Returns
    -------
    RothProfile
        One in-memory pressure-temperature profile with any available
        provenance metadata attached.
    """
    if path.suffix.lower() == ".npz":
        arrays = np.load(path)
        metadata = json.loads(str(arrays["metadata"].item())) if "metadata" in arrays else {}
        return RothProfile(
            pressure_bar=np.asarray(arrays["pressure_bar"], dtype=np.float64),
            temperature_k=np.asarray(arrays["temperature_k"], dtype=np.float64),
            metadata=metadata,
        )
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return RothProfile(
            pressure_bar=np.asarray(payload["pressure_bar"], dtype=np.float64),
            temperature_k=np.asarray(payload["temperature_k"], dtype=np.float64),
            metadata=dict(payload.get("metadata", {})),
        )
    rows: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((float(row["pressure_bar"]), float(row["temperature_k"])))
    if not rows:
        raise ValueError(f"Roth profile file {path} is empty.")
    pressure_bar = np.array([row[0] for row in rows], dtype=np.float64)
    temperature_k = np.array([row[1] for row in rows], dtype=np.float64)
    return RothProfile(pressure_bar=pressure_bar, temperature_k=temperature_k, metadata={"source_file": str(path)})


def roth_library_pressure_bounds(
    profiles: list[RothProfile],
) -> tuple[float, float]:
    """Return ``(min_bar, max_bar)`` across the union of native pressure grids.

    Used by the sampler to clip per-run target pressure grids to the range
    covered by the loaded PT library so PCHIP interpolation never falls into
    the constant-extrapolation branch.

    Parameters
    ----------
    profiles : list[RothProfile]
        Non-empty list of PT-library profiles loaded at native resolution.

    Returns
    -------
    tuple[float, float]
        ``(min_pressure_bar, max_pressure_bar)`` across all supplied profiles.
    """
    if not profiles:
        raise ValueError("Cannot compute native pressure bounds from an empty profile list.")
    mins = [float(np.min(profile.pressure_bar)) for profile in profiles]
    maxs = [float(np.max(profile.pressure_bar)) for profile in profiles]
    return min(mins), max(maxs)


def _load_pt_bundle(path: Path) -> list[RothProfile]:
    """Load a prebaked single-file PT-library bundle.

    The bundle layout is the one produced by ``scripts/prebake_pt_profiles.py``:
    concatenated ``pressure_bar`` and ``temperature_k`` arrays, an integer
    ``offsets`` index of length ``N_profiles + 1``, and an object array of
    JSON-encoded per-profile metadata. Reading the bundle is a single
    ``np.load`` plus N slices, in contrast to per-file globs.
    """
    with np.load(path, allow_pickle=True) as archive:
        pressure_concat = np.asarray(archive["pressure_bar"], dtype=np.float64)
        temperature_concat = np.asarray(archive["temperature_k"], dtype=np.float64)
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
        metadata_array = np.asarray(archive["metadata"])
    profiles: list[RothProfile] = []
    for i in range(metadata_array.size):
        start = int(offsets[i])
        stop = int(offsets[i + 1])
        metadata_payload = metadata_array[i]
        if isinstance(metadata_payload, bytes):
            metadata_payload = metadata_payload.decode("utf-8")
        metadata = json.loads(str(metadata_payload))
        profiles.append(
            RothProfile(
                pressure_bar=pressure_concat[start:stop].copy(),
                temperature_k=temperature_concat[start:stop].copy(),
                metadata=metadata,
            )
        )
    return profiles


def load_roth_profiles_native(
    data_glob: str,
    *,
    filters: RothFilterConfig | None = None,
) -> list[RothProfile]:
    """Load and filter PT-library profiles at their native pressure grid.

    Accepts either a glob (``*.dat`` / ``*.npz`` / ``*.json`` / ``*.csv``)
    or a single prebaked bundle path (``*.bundle.npz``). The bundle path
    short-circuits the per-file dispatch to a single ``np.load``.

    No interpolation is done here — interpolation onto a per-run pressure
    grid is the caller's responsibility. This split exists so the library
    can be loaded from disk once and reused across many runs with
    different pressure grids (see ``_load_configured_roth_profiles``).
    """
    candidate = Path(data_glob)
    if candidate.is_file() and candidate.name.endswith(".bundle.npz"):
        bundle_profiles = _load_pt_bundle(candidate)
        return [profile for profile in bundle_profiles if _matches_filters(profile.metadata, filters)]
    result: list[RothProfile] = []
    for path_str in sorted(glob.glob(data_glob)):
        path = Path(path_str)
        if path.suffix.lower() == ".dat":
            source_metadata = _parse_pt_profile_filename(path)
            if not _matches_filters(source_metadata, filters):
                continue
            loaded_profiles = _load_pt_dat_profiles(path)
        else:
            loaded_profile = _load_tabular_roth_profile(path)
            if not _matches_filters(loaded_profile.metadata, filters):
                continue
            loaded_profiles = [loaded_profile]
        result.extend(loaded_profiles)
    return result


def load_roth_profiles(
    data_glob: str,
    *,
    pressure_grid_bar: np.ndarray,
    filters: RothFilterConfig | None = None,
) -> list[RothProfile]:
    """Load, filter, and interpolate PT-library profiles onto the requested grid.

    Parameters
    ----------
    data_glob : str
        Shell glob matching PT-library source files (``.dat``, ``.npz``,
        ``.json``, or ``.csv``).
    pressure_grid_bar : 1-D array
        Target pressure grid in bar.
    filters : dict, optional
        Per-parameter constraints.  Numeric keys accept an exact value
        or a ``(low, high)`` range tuple; boolean keys accept ``True``
        / ``False``.

    Returns
    -------
    list[RothProfile]
        Profiles interpolated onto ``pressure_grid_bar``, filtered by
        the supplied constraints.
    """
    result: list[RothProfile] = []
    for profile in load_roth_profiles_native(data_glob, filters=filters):
        interpolated = _interpolate_profile(
            pressure_grid_bar,
            profile.pressure_bar,
            profile.temperature_k,
        )
        result.append(
            RothProfile(
                pressure_bar=np.asarray(pressure_grid_bar, dtype=np.float64),
                temperature_k=interpolated,
                metadata=dict(profile.metadata),
            )
        )
    return result
