"""Stellar spectrum handling: Planck function and template IO.

Provides utilities for generating, reading, writing, and managing
stellar spectra used as conditioning inputs for the full-VULCAN model.

All flux quantities are stored as surface flux density in CGS units
(erg cm-2 s-1 nm-1).  The planet-star distance scaling is handled
externally by VULCAN's ``r_star / orbit_radius`` configuration.
"""

from __future__ import annotations

import glob as glob_module
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import C_LIGHT, H_PLANCK, K_BOLTZMANN


@dataclass(frozen=True)
class SpectrumRecord:
    """Named stellar spectrum sampled on a wavelength grid.

    Attributes
    ----------
    name : str
        Human-readable identifier (e.g., ``"blackbody_template"``).
    wavelength_nm : 1-D array
        Strictly increasing wavelength grid in nanometres.
    flux_erg_cm2_s_nm : 1-D array
        Non-negative surface flux density at each wavelength bin.
    metadata : dict
        Free-form provenance (source file, Teff, etc.).
    """

    name: str
    wavelength_nm: np.ndarray
    flux_erg_cm2_s_nm: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self: "SpectrumRecord") -> None:
        """Validate the in-memory spectrum contract before use downstream."""
        if self.wavelength_nm.ndim != 1 or self.flux_erg_cm2_s_nm.ndim != 1:
            raise ValueError("Spectrum arrays must be one-dimensional.")
        if self.wavelength_nm.size != self.flux_erg_cm2_s_nm.size:
            raise ValueError("Wavelength and flux arrays must have matching lengths.")
        if self.wavelength_nm.size < 2:
            raise ValueError("Spectrum must contain at least two wavelength samples.")
        if not np.all(np.isfinite(self.wavelength_nm)) or not np.all(
            np.isfinite(self.flux_erg_cm2_s_nm)
        ):
            raise ValueError("Spectrum contains non-finite values.")
        if not np.all(np.diff(self.wavelength_nm) > 0.0):
            raise ValueError("Spectrum wavelength grid must be strictly increasing.")
        if np.any(self.flux_erg_cm2_s_nm < 0.0):
            raise ValueError("Spectrum flux must be non-negative.")


def _planck_lambda_erg_cm2_s_sr_cm(
    wavelength_cm: np.ndarray,
    temperature_k: float,
) -> np.ndarray:
    """Evaluate the Planck function B_lambda(T) in cgs per wavelength interval."""
    numerator = 2.0 * H_PLANCK * C_LIGHT**2
    exponent = (H_PLANCK * C_LIGHT) / (wavelength_cm * K_BOLTZMANN * temperature_k)
    denom = np.expm1(np.clip(exponent, 1.0e-10, 700.0))
    return numerator / (wavelength_cm**5 * denom)


def blackbody_surface_flux(
    wavelength_nm: np.ndarray,
    *,
    temperature_k: float,
) -> np.ndarray:
    """Return stellar-surface flux density for a blackbody spectrum."""
    wavelength_cm = np.asarray(wavelength_nm, dtype=np.float64) * 1.0e-7
    radiance = _planck_lambda_erg_cm2_s_sr_cm(wavelength_cm, temperature_k)
    surface_exitance = np.pi * radiance  # erg / cm^2 / s / cm
    return surface_exitance * 1.0e-7


def generate_blackbody_template(
    *,
    num_points: int = 2401,
    wavelength_min_nm: float = 100.0,
    wavelength_max_nm: float = 700.0,
    teff_k: float = 5485.0,
    radius_rsun: float = 0.939,
    semi_major_axis_au: float = 0.04858,
    name: str = "blackbody_template",
) -> SpectrumRecord:
    """Generate a blackbody stellar surface-flux template."""
    wavelength_nm = np.linspace(
        wavelength_min_nm,
        wavelength_max_nm,
        int(num_points),
        dtype=np.float64,
    )
    flux = blackbody_surface_flux(wavelength_nm, temperature_k=teff_k)
    record = SpectrumRecord(
        name=name,
        wavelength_nm=wavelength_nm,
        flux_erg_cm2_s_nm=flux,
        metadata={
            "kind": "analytic_blackbody_surface_flux",
            "teff_k": float(teff_k),
            "radius_rsun": float(radius_rsun),
            "semi_major_axis_au": float(semi_major_axis_au),
        },
    )
    record.validate()
    return record


def write_vulcan_spectrum_txt(record: SpectrumRecord, path: str | Path) -> Path:
    """Write a spectrum record in VULCAN's two-column text format."""
    record.validate()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("# wavelength_nm flux_erg_cm2_s_nm\n")
        for wavelength, flux in zip(record.wavelength_nm, record.flux_erg_cm2_s_nm):
            handle.write(f"{wavelength:.8f} {flux:.16e}\n")
    return out_path


def read_vulcan_spectrum_txt(path: str | Path, *, name: str | None = None) -> SpectrumRecord:
    """Read a VULCAN text spectrum into a validated ``SpectrumRecord``."""
    loaded = np.loadtxt(Path(path), comments="#")
    if loaded.ndim != 2 or loaded.shape[1] < 2:
        raise ValueError("Spectrum text file must contain at least two columns.")
    record = SpectrumRecord(
        name=name or Path(path).stem,
        wavelength_nm=loaded[:, 0].astype(np.float64),
        flux_erg_cm2_s_nm=loaded[:, 1].astype(np.float64),
        metadata={"source_file": str(path)},
    )
    record.validate()
    return record


def load_spectrum_records_from_glob(pattern: str) -> dict[str, SpectrumRecord]:
    """Load a spectrum library from a filesystem glob of text spectra."""
    result: dict[str, SpectrumRecord] = {}
    matched_paths = [Path(path) for path in sorted(glob_module.glob(pattern, recursive=True))]
    if not matched_paths:
        raise FileNotFoundError(f"No stellar spectrum files matched {pattern!r}.")
    for path in matched_paths:
        record = read_vulcan_spectrum_txt(path, name=path.stem)
        if record.name in result:
            raise ValueError(f"Duplicate stellar spectrum record name {record.name!r} from {path}.")
        result[record.name] = record
    return result
