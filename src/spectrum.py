from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

C_LIGHT = 2.99792458e10  # cm / s
H_PLANCK = 6.62607015e-27  # erg s
K_BOLTZMANN = 1.380649e-16  # erg / K
R_SUN_CM = 6.957e10
AU_CM = 1.495978707e13


@dataclass(frozen=True)
class SpectrumRecord:
    name: str
    wavelength_nm: np.ndarray
    flux_erg_cm2_s_nm: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.wavelength_nm.ndim != 1 or self.flux_erg_cm2_s_nm.ndim != 1:
            raise ValueError("Spectrum arrays must be one-dimensional.")
        if self.wavelength_nm.size != self.flux_erg_cm2_s_nm.size:
            raise ValueError("Wavelength and flux arrays must have matching lengths.")
        if self.wavelength_nm.size < 2:
            raise ValueError("Spectrum must contain at least two wavelength samples.")
        if not np.all(np.isfinite(self.wavelength_nm)) or not np.all(np.isfinite(self.flux_erg_cm2_s_nm)):
            raise ValueError("Spectrum contains non-finite values.")
        if not np.all(np.diff(self.wavelength_nm) > 0.0):
            raise ValueError("Spectrum wavelength grid must be strictly increasing.")
        if np.any(self.flux_erg_cm2_s_nm < 0.0):
            raise ValueError("Spectrum flux must be non-negative.")


def _planck_lambda_erg_cm2_s_sr_cm(wavelength_cm: np.ndarray, temperature_k: float) -> np.ndarray:
    numerator = 2.0 * H_PLANCK * C_LIGHT**2
    exponent = (H_PLANCK * C_LIGHT) / (wavelength_cm * K_BOLTZMANN * temperature_k)
    denom = np.expm1(np.clip(exponent, 1.0e-10, 700.0))
    return numerator / (wavelength_cm**5 * denom)


def blackbody_surface_flux(
    wavelength_nm: np.ndarray,
    *,
    temperature_k: float,
) -> np.ndarray:
    """Return stellar-surface flux density for a blackbody spectrum.

    VULCAN expects the stellar spectrum file to be a surface flux; the
    planet-star distance scaling is applied internally through
    ``r_star`` / ``orbit_radius`` in ``vulcan_cfg.py``.
    """
    wavelength_cm = np.asarray(wavelength_nm, dtype=np.float64) * 1.0e-7
    radiance = _planck_lambda_erg_cm2_s_sr_cm(wavelength_cm, temperature_k)
    surface_exitance = np.pi * radiance  # erg / cm^2 / s / cm
    return surface_exitance * 1.0e-7


def generate_wasp39b_template(
    *,
    num_points: int = 2401,
    wavelength_min_nm: float = 100.0,
    wavelength_max_nm: float = 700.0,
    teff_k: float = 5485.0,
    radius_rsun: float = 0.939,
    semi_major_axis_au: float = 0.04858,
    name: str = "wasp39b_template",
) -> SpectrumRecord:
    wavelength_nm = np.linspace(wavelength_min_nm, wavelength_max_nm, int(num_points), dtype=np.float64)
    flux = blackbody_surface_flux(
        wavelength_nm,
        temperature_k=teff_k,
    )
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
    record.validate()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("# wavelength_nm flux_erg_cm2_s_nm\n")
        for wavelength, flux in zip(record.wavelength_nm, record.flux_erg_cm2_s_nm):
            handle.write(f"{wavelength:.8f} {flux:.16e}\n")
    return out_path


def read_vulcan_spectrum_txt(path: str | Path, *, name: str | None = None) -> SpectrumRecord:
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


def resample_spectrum(record: SpectrumRecord, wavelength_grid_nm: np.ndarray) -> np.ndarray:
    record.validate()
    grid = np.asarray(wavelength_grid_nm, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("wavelength_grid_nm must be a one-dimensional array with at least two elements.")
    left = float(record.flux_erg_cm2_s_nm[0])
    right = float(record.flux_erg_cm2_s_nm[-1])
    return np.interp(grid, record.wavelength_nm, record.flux_erg_cm2_s_nm, left=left, right=right)


def fixed_wavelength_grid(
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    num_bins: int,
) -> np.ndarray:
    return np.linspace(float(wavelength_min_nm), float(wavelength_max_nm), int(num_bins), dtype=np.float64)


def save_spectrum_manifest(records: Iterable[SpectrumRecord], output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"records": []}
    for record in records:
        record.validate()
        npz_name = f"{record.name}.npz"
        npz_path = output_path / npz_name
        np.savez(
            npz_path,
            wavelength_nm=record.wavelength_nm.astype(np.float64),
            flux_erg_cm2_s_nm=record.flux_erg_cm2_s_nm.astype(np.float64),
            metadata=json.dumps(record.metadata, sort_keys=True),
        )
        manifest["records"].append(
            {
                "name": record.name,
                "file": npz_name,
                "metadata": record.metadata,
            }
        )
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def load_spectrum_manifest(manifest_path: str | Path) -> dict[str, SpectrumRecord]:
    manifest_file = Path(manifest_path)
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    result: dict[str, SpectrumRecord] = {}
    for entry in payload.get("records", []):
        arrays = np.load(manifest_file.parent / entry["file"])
        metadata_text = arrays["metadata"].item() if arrays["metadata"].shape == () else str(arrays["metadata"])
        record = SpectrumRecord(
            name=str(entry["name"]),
            wavelength_nm=np.asarray(arrays["wavelength_nm"], dtype=np.float64),
            flux_erg_cm2_s_nm=np.asarray(arrays["flux_erg_cm2_s_nm"], dtype=np.float64),
            metadata=json.loads(metadata_text),
        )
        record.validate()
        result[record.name] = record
    return result
