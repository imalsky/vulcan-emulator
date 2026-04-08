"""Stellar spectrum handling: Planck function, template IO, and model packing.

Provides utilities for generating, reading, writing, sanitizing, and packing
stellar spectra used as conditioning inputs for the full-VULCAN model.

All flux quantities are stored as surface flux density in CGS units
(erg cm-2 s-1 nm-1).  The planet-star distance scaling is handled
externally by VULCAN's ``r_star / orbit_radius`` configuration.
"""

from __future__ import annotations

import glob as glob_module
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Fundamental physical constants in CGS units (CODATA 2018).
C_LIGHT = 2.99792458e10  # speed of light [cm s-1]
H_PLANCK = 6.62607015e-27  # Planck constant [erg s]
K_BOLTZMANN = 1.380649e-16  # Boltzmann constant [erg K-1]
R_SUN_CM = 6.957e10  # solar radius [cm] (IAU 2015 nominal)
AU_CM = 1.495978707e13  # astronomical unit [cm] (IAU 2012)


@dataclass(frozen=True)
class SpectrumRecord:
    """Named stellar spectrum sampled on a wavelength grid.

    Attributes
    ----------
    name : str
        Human-readable identifier (e.g., ``"wasp39b_template"``).
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

    def validate(self) -> None:
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
    """Generate the default blackbody-based WASP-39b stellar template."""
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


def resample_spectrum(record: SpectrumRecord, wavelength_grid_nm: np.ndarray) -> np.ndarray:
    """Resample a spectrum onto a fixed wavelength grid via linear interpolation."""
    record.validate()
    grid = np.asarray(wavelength_grid_nm, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError(
            "wavelength_grid_nm must be a one-dimensional array with at least two elements."
        )
    left = float(record.flux_erg_cm2_s_nm[0])
    right = float(record.flux_erg_cm2_s_nm[-1])
    return np.interp(
        grid,
        record.wavelength_nm,
        record.flux_erg_cm2_s_nm,
        left=left,
        right=right,
    )


def fixed_wavelength_grid(
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    num_bins: int,
) -> np.ndarray:
    """Build a fixed, linearly-spaced wavelength grid."""
    return np.linspace(
        float(wavelength_min_nm),
        float(wavelength_max_nm),
        int(num_bins),
        dtype=np.float64,
    )


def _coalesce_duplicate_wavelengths(
    wavelength_nm: np.ndarray,
    flux_erg_cm2_s_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average duplicate wavelength samples onto a unique strictly increasing grid."""
    unique_wavelengths, inverse = np.unique(wavelength_nm, return_inverse=True)
    summed_flux = np.zeros_like(unique_wavelengths, dtype=np.float64)
    counts = np.zeros_like(unique_wavelengths, dtype=np.float64)
    np.add.at(summed_flux, inverse, flux_erg_cm2_s_nm)
    np.add.at(counts, inverse, 1.0)
    return unique_wavelengths, summed_flux / np.maximum(counts, 1.0)


def sanitize_spectrum_arrays(
    wavelength_nm: np.ndarray,
    flux_erg_cm2_s_nm: np.ndarray,
    *,
    wavelength_min_nm: float | None = None,
    wavelength_max_nm: float | None = None,
    min_points: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort, de-duplicate, clip, and validate raw spectrum arrays.

    This intentionally preserves the *native* sampling when possible. The only
    destructive operations are:
    1. dropping non-finite rows,
    2. clipping negative fluxes to zero,
    3. sorting by wavelength,
    4. averaging duplicate wavelength samples, and
    5. clipping to the requested wavelength interval.
    """
    wavelength = np.asarray(wavelength_nm, dtype=np.float64).reshape(-1)
    flux = np.asarray(flux_erg_cm2_s_nm, dtype=np.float64).reshape(-1)
    if wavelength.size != flux.size:
        raise ValueError("Wavelength and flux arrays must have the same length.")
    mask = np.isfinite(wavelength) & np.isfinite(flux)
    wavelength = wavelength[mask]
    flux = np.maximum(flux[mask], 0.0)
    if wavelength_min_nm is not None:
        mask = wavelength >= float(wavelength_min_nm)
        wavelength = wavelength[mask]
        flux = flux[mask]
    if wavelength_max_nm is not None:
        mask = wavelength <= float(wavelength_max_nm)
        wavelength = wavelength[mask]
        flux = flux[mask]
    if wavelength.size < int(min_points):
        raise ValueError(
            "Spectrum does not contain enough valid samples after clipping to the configured "
            "wavelength interval."
        )
    order = np.argsort(wavelength, kind="mergesort")
    wavelength = wavelength[order]
    flux = flux[order]
    wavelength, flux = _coalesce_duplicate_wavelengths(wavelength, flux)
    if wavelength.size < int(min_points):
        raise ValueError("Spectrum must contain at least two unique wavelength samples.")
    if not np.all(np.diff(wavelength) > 0.0):
        raise ValueError("Sanitized wavelength grid must be strictly increasing.")
    return wavelength.astype(np.float64), flux.astype(np.float64)


def vulcan_wavelength_bins(
    wavelength_min_nm: float = 2.0,
    wavelength_max_nm: float = 700.0,
    dbin1_nm: float = 0.1,
    dbin2_nm: float = 2.0,
    dbin_12trans_nm: float = 240.0,
) -> np.ndarray:
    """Construct VULCAN's non-uniform wavelength bin-centre array.

    Mirrors the bin construction in VULCAN's ``op.py``: fine spacing
    (``dbin1_nm``) below the transition wavelength, coarse spacing
    (``dbin2_nm``) above.
    """
    wmin = float(wavelength_min_nm)
    wmax = float(wavelength_max_nm)
    trans = float(dbin_12trans_nm)
    d1 = float(dbin1_nm)
    d2 = float(dbin2_nm)
    if trans >= wmin and trans <= wmax:
        bins = np.concatenate((
            np.arange(wmin, trans, d1),
            np.arange(trans, wmax, d2),
        ))
    elif trans < wmin:
        bins = np.arange(wmin, wmax, d2)
    else:
        bins = np.arange(wmin, wmax, d1)
    return bins.astype(np.float64)


def _flux_conserving_rebin(
    wavelength_nm: np.ndarray,
    flux_erg_cm2_s_nm: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Rebin a spectrum onto arbitrary bin edges with flux conservation.

    Returns the average flux density in each bin such that the trapezoidal
    integral over each bin matches the original spectrum.

    Parameters
    ----------
    wavelength_nm, flux_erg_cm2_s_nm : (N,) float64
        Sorted input spectrum.
    edges : (M+1,) float64
        Bin edges defining M output bins.

    Returns
    -------
    rebinned_flux : (M,) float64
    """
    w = wavelength_nm.astype(np.float64)
    f = flux_erg_cm2_s_nm.astype(np.float64)
    edges = np.asarray(edges, dtype=np.float64)

    # Merge bin edges into the sample grid so that the trapezoidal rule
    # integrates exactly to/from every edge.  Flux at inserted edges is
    # linearly interpolated.
    edge_flux = np.interp(edges, w, f)
    merged_w = np.concatenate([w, edges])
    merged_f = np.concatenate([f, edge_flux])
    order = np.argsort(merged_w, kind="stable")
    merged_w = merged_w[order]
    merged_f = merged_f[order]
    # Remove exact duplicates (keep first).
    keep = np.concatenate(([True], np.diff(merged_w) > 0.0))
    merged_w = merged_w[keep]
    merged_f = merged_f[keep]

    # Cumulative trapezoidal integral on the merged grid.
    cum = np.empty(len(merged_w), dtype=np.float64)
    cum[0] = 0.0
    cum[1:] = np.cumsum(0.5 * (merged_f[:-1] + merged_f[1:]) * np.diff(merged_w))

    # Look up cumulative integral at each bin edge (edges are exact members
    # of merged_w, so searchsorted finds them without interpolation error).
    edge_idx = np.searchsorted(merged_w, edges)
    edge_idx = np.clip(edge_idx, 0, len(cum) - 1)
    bin_integrals = cum[edge_idx[1:]] - cum[edge_idx[:-1]]

    bin_widths = np.maximum(edges[1:] - edges[:-1], 1.0e-12)
    return bin_integrals / bin_widths


def _vulcan_bin_edges(
    centers: np.ndarray,
    dbin1_nm: float,
    dbin2_nm: float,
    dbin_12trans_nm: float,
) -> np.ndarray:
    """Compute contiguous bin edges from VULCAN bin centres.

    Within each resolution segment the edges are simply center ± half-width.
    At the UV/visible transition the boundary is placed at the midpoint
    between the last UV centre and the first visible centre so that bins
    tile without gaps or overlaps.
    """
    n = len(centers)
    edges = np.empty(n + 1, dtype=np.float64)
    # Interior edges: midpoints between adjacent centres.
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    # First and last edges: half a bin-width beyond the first/last centre.
    uv_mask = centers < dbin_12trans_nm
    first_width = dbin1_nm if uv_mask[0] else dbin2_nm
    last_width = dbin1_nm if uv_mask[-1] else dbin2_nm
    edges[0] = centers[0] - 0.5 * first_width
    edges[-1] = centers[-1] + 0.5 * last_width
    return edges


def pack_spectrum_tokens(
    wavelength_nm: np.ndarray,
    flux_erg_cm2_s_nm: np.ndarray,
    *,
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    max_tokens: int,
    dbin1_nm: float = 0.1,
    dbin2_nm: float = 2.0,
    dbin_12trans_nm: float = 240.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack an arbitrary spectrum into VULCAN's native wavelength grid.

    The input spectrum (any native grid) is sanitized, then rebinned onto
    VULCAN's non-uniform wavelength grid (fine UV bins, coarse visible bins).
    The result is zero-padded to ``max_tokens`` with a boolean validity mask.
    """
    clean_wavelengths, clean_flux = sanitize_spectrum_arrays(
        wavelength_nm,
        flux_erg_cm2_s_nm,
        wavelength_min_nm=wavelength_min_nm,
        wavelength_max_nm=wavelength_max_nm,
        min_points=2,
    )
    centers = vulcan_wavelength_bins(
        wavelength_min_nm=wavelength_min_nm,
        wavelength_max_nm=wavelength_max_nm,
        dbin1_nm=dbin1_nm,
        dbin2_nm=dbin2_nm,
        dbin_12trans_nm=dbin_12trans_nm,
    )
    edges = _vulcan_bin_edges(centers, dbin1_nm, dbin2_nm, dbin_12trans_nm)
    rebinned_flux = _flux_conserving_rebin(clean_wavelengths, clean_flux, edges)
    valid_count = len(centers)
    if valid_count > int(max_tokens):
        raise ValueError(
            f"VULCAN bin grid produces {valid_count} bins but max_tokens is "
            f"{max_tokens}. Increase max_tokens to at least {valid_count}."
        )
    padded_wavelengths = np.zeros((int(max_tokens),), dtype=np.float32)
    padded_flux = np.zeros((int(max_tokens),), dtype=np.float32)
    padded_mask = np.zeros((int(max_tokens),), dtype=bool)
    padded_wavelengths[:valid_count] = centers.astype(np.float32)
    padded_flux[:valid_count] = rebinned_flux.astype(np.float32)
    padded_mask[:valid_count] = True
    return padded_wavelengths, padded_flux, padded_mask


def save_spectrum_manifest(records: Iterable[SpectrumRecord], output_dir: str | Path) -> Path:
    """Persist a spectrum library and its manifest under one directory."""
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
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def load_spectrum_manifest(manifest_path: str | Path) -> dict[str, SpectrumRecord]:
    """Load a saved spectrum manifest into validated in-memory records."""
    manifest_file = Path(manifest_path)
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    result: dict[str, SpectrumRecord] = {}
    for entry in payload.get("records", []):
        arrays = np.load(manifest_file.parent / entry["file"])
        metadata_text = (
            arrays["metadata"].item() if arrays["metadata"].shape == () else str(arrays["metadata"])
        )
        record = SpectrumRecord(
            name=str(entry["name"]),
            wavelength_nm=np.asarray(arrays["wavelength_nm"], dtype=np.float64),
            flux_erg_cm2_s_nm=np.asarray(arrays["flux_erg_cm2_s_nm"], dtype=np.float64),
            metadata=json.loads(metadata_text),
        )
        record.validate()
        result[record.name] = record
    return result


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
