from __future__ import annotations

import csv
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROTH_FILTER_KEYS = ("phase", "limb", "column")
ROTH_OPTIONAL_FILTER_KEYS = ("label",)


@dataclass(frozen=True)
class RothProfile:
    pressure_bar: np.ndarray
    temperature_k: np.ndarray
    metadata: dict[str, Any]


def _interpolate_profile(
    pressure_bar: np.ndarray,
    source_pressure_bar: np.ndarray,
    source_temperature_k: np.ndarray,
) -> np.ndarray:
    log_target = np.log10(pressure_bar)
    log_source = np.log10(source_pressure_bar)
    return np.interp(log_target, log_source[::-1], source_temperature_k[::-1])


def _load_one_profile(path: Path) -> RothProfile:
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


def load_roth_profiles(
    data_glob: str,
    *,
    pressure_grid_bar: np.ndarray,
    filters: dict[str, list[str]] | None = None,
) -> list[RothProfile]:
    result: list[RothProfile] = []
    for path_str in sorted(glob.glob(data_glob)):
        profile = _load_one_profile(Path(path_str))
        include = True
        for key, allowed in (filters or {}).items():
            value = str(profile.metadata.get(key, ""))
            if allowed and value not in allowed:
                include = False
                break
        if include:
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
