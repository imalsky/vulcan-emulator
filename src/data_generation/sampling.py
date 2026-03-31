"""Atmospheric run specification sampling: temperature profiles, pressure grids.

This module is the primary entry point for constructing the atmospheric
configurations used to generate raw training runs.  It supports three
temperature-profile sources, selected by ``temperature_profiles.source_mode``:

- **analytic**: Profiles are generated from the Line et al. (2013) radiative-
  equilibrium parameterization with Robinson & Catling (2012) thermal opacity
  modifications and an optional convective adjustment.  Ten random parameters
  are drawn per profile (see ``_sample_analytic_temperature_profile_record``).

- **pt_library**: Profiles are loaded from externally computed GCM output files
  (Roth .dat format).  Each ``(lon, lat)`` column in a file is expanded into a
  separate 1D profile and interpolated onto the configured pressure grid using
  shape-preserving PCHIP interpolation in log-pressure space.

- **mixed**: Each run independently selects analytic or PT-library with a
  configurable probability (``temperature_profiles.analytic_probability``),
  giving training sets that cover both parameterized shapes and realistic
  GCM-derived profiles.

The module also handles Latin-hypercube sampling of the global conditioning
scalars (metallicity, C/O, S/O, and optionally gravity) plus stellar-spectrum
selection for the final-state full-VULCAN task.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expn

from ..utils.config import ELEMENT_INPUT_ORDER, is_equilibrium, static_conditioning_defaults
from .roth_sampling import RothFilterValue, RothProfile, load_roth_profiles
from .spectrum import SpectrumRecord, load_spectrum_manifest, save_spectrum_manifest

logger = logging.getLogger(__name__)

# Maximum number of rejection-resampling attempts for analytic profiles.
_MAX_PROFILE_ATTEMPTS = 100

# Bar-to-Pascal conversion factor.
_BAR_TO_PA = 1.0e5

_SOLAR_ELEMENT_ABUNDANCES = {
    "O_H": 5.37e-4,
    "C_H": 2.95e-4,
    "N_H": 7.08e-5,
    "S_H": 1.41e-5,
    "He_H": 8.38e-2,
}


@dataclass(frozen=True)
class RunSpecification:
    """All inputs needed to generate one raw run."""
    run_id: str
    pressure_bar: np.ndarray
    temperature_k: np.ndarray
    globals: dict[str, float]
    metadata: dict[str, Any]
    kzz_cm2_s: np.ndarray | None = None
    spectrum: SpectrumRecord | None = None
    elemental_abundances_x_h: np.ndarray | None = None
    gravity_cm_s2: np.ndarray | None = None


def _element_scalars_from_sampled_globals(globals_map: dict[str, float]) -> dict[str, float]:
    """Derive FastChem-native hydrogen-normalized elemental abundances from sampled globals."""
    metal_scale = 10.0 ** float(globals_map["metallicity_log10"])
    oxygen_h = _SOLAR_ELEMENT_ABUNDANCES["O_H"] * metal_scale
    sulfur_h = oxygen_h * float(globals_map["s_to_o"])
    return {
        "He_H": float(_SOLAR_ELEMENT_ABUNDANCES["He_H"]),
        "C_H": float(oxygen_h * float(globals_map["c_to_o"])),
        "O_H": float(oxygen_h),
        "N_H": float(_SOLAR_ELEMENT_ABUNDANCES["N_H"] * metal_scale),
        "S_H": float(sulfur_h),
    }


def _element_profile_from_scalars(
    element_scalars: dict[str, float],
    *,
    num_levels: int,
) -> np.ndarray:
    """Broadcast a column-constant elemental composition to the required API shape."""
    element_vector = np.array(
        [float(element_scalars[name]) for name in ELEMENT_INPUT_ORDER],
        dtype=np.float64,
    )
    return np.repeat(element_vector[None, :], int(num_levels), axis=0)


def _equilibrium_gravity_profile(
    *,
    pressure_bar: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Build the required per-level gravity array for equilibrium runs."""
    analytic_sampler = config["temperature_profiles"].get("analytic_sampler", {})
    gravity_cm_s2 = 100.0 * float(analytic_sampler.get("reference_gravity_m_s2", 25.0))
    return np.full(np.asarray(pressure_bar).shape, gravity_cm_s2, dtype=np.float64)


def sample_pressure_grid(
    *,
    num_levels: int,
    pressure_top_bar: float,
    pressure_bottom_bar: float,
) -> np.ndarray:
    """Build the fixed log-spaced pressure grid used by the surrogate.

    Parameters
    ----------
    num_levels : int
        Number of vertical levels in the atmospheric column.
    pressure_top_bar : float
        Lowest pressure (top of atmosphere) in bar.
    pressure_bottom_bar : float
        Highest pressure (bottom of atmosphere) in bar.

    Returns
    -------
    np.ndarray
        1-D array of shape ``(num_levels,)`` with pressures in bar, ordered
        from high pressure (bottom) to low pressure (top) — the same
        direction as increasing altitude.
    """
    return np.logspace(
        math.log10(pressure_bottom_bar),
        math.log10(pressure_top_bar),
        int(num_levels),
        dtype=np.float64,
    )


def _sample_range_value(values: list[float] | tuple[float, float], *, rng: np.random.Generator) -> float:
    """Draw one scalar from an inclusive config range."""
    lower = float(values[0])
    upper = float(values[1])
    return float(rng.uniform(lower, upper))


def _sample_normal_value(spec: dict[str, float], *, rng: np.random.Generator) -> float:
    """Draw one scalar from a normal config specification."""
    return float(rng.normal(float(spec["mean"]), float(spec["std"])))


def _compute_xi(gamma: float, tau: np.ndarray) -> np.ndarray:
    """Compute the xi penetration function for a visible-channel opacity ratio.

    Implements the two-stream approximation term from Line et al. (2013):

        xi(gamma, tau) = 2/3
            + (2 / (3*gamma)) * (1 + (gamma*tau/2 - 1) * exp(-gamma*tau))
            + (2*gamma / 3) * (1 - tau^2/2) * E_2(gamma*tau)

    Parameters
    ----------
    gamma : float
        Ratio of the visible-channel Planck mean opacity to the thermal opacity.
        Must be > 0.
    tau : np.ndarray
        Infrared optical depth at each pressure level.

    Returns
    -------
    np.ndarray
        The xi contribution at each pressure level, same shape as *tau*.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    tau = np.asarray(tau, dtype=np.float64)
    gt = gamma * tau
    term1 = 2.0 / 3.0
    term2 = (2.0 / (3.0 * gamma)) * (1.0 + (gt / 2.0 - 1.0) * np.exp(-gt))
    term3 = (2.0 * gamma / 3.0) * (1.0 - 0.5 * tau ** 2) * expn(2, gt)
    return term1 + term2 + term3


def _compute_optical_depth(
    pressure_bar: np.ndarray,
    *,
    kappa_ir_m2_kg: float,
    gravity_m_s2: float,
    power_law_n: float,
) -> np.ndarray:
    """Compute gray infrared optical depth following the reference formulation.

    tau(P) = (kappa_ref * P_ref_Pa) / (g * n) * (P / P_ref)^n

    where P_ref = 1 bar.  This is the integrated form assuming
    kappa(P) = kappa_ref * (P/P_ref)^(n-1).

    Parameters
    ----------
    pressure_bar : np.ndarray
        Pressure grid in bar.
    kappa_ir_m2_kg : float
        Reference infrared opacity in m^2/kg.
    gravity_m_s2 : float
        Surface gravity in m/s^2.
    power_law_n : float
        Pressure power-law exponent.  Must be > 0.

    Returns
    -------
    np.ndarray
        Gray infrared optical depth at each pressure level.
    """
    if power_law_n <= 0:
        raise ValueError(f"power_law_n must be > 0, got {power_law_n}")
    tau_scale = (kappa_ir_m2_kg * _BAR_TO_PA) / gravity_m_s2
    return (tau_scale / power_law_n) * pressure_bar ** power_law_n


def _apply_convective_adjustment(
    pressure_bar: np.ndarray,
    temperature_k: np.ndarray,
    *,
    adiabatic_gradient: float,
) -> np.ndarray:
    """Apply convective adjustment following the reference implementation.

    Walks from the top of the atmosphere downward.  When the local
    d(ln T)/d(ln P) gradient exceeds the adiabatic gradient, the profile
    is replaced by an adiabat from that level to the bottom of the column.

    This matches the reference ``apply_convection`` in create_profiles.py.
    ``alpha * (gamma - 1) / gamma`` where gamma is the adiabatic index.

    Parameters
    ----------
    pressure_bar : np.ndarray
        Pressure grid in bar, shape ``(nz,)``.  Ordered from high pressure
        (bottom) to low pressure (top).
    temperature_k : np.ndarray
        Temperature profile in Kelvin, shape ``(nz,)``.
    adiabatic_gradient : float
        The adiabatic temperature gradient d(ln T)/d(ln P).

    Returns
    -------
    np.ndarray
        Adjusted temperature profile, shape ``(nz,)``.
    """
    adjusted = np.copy(temperature_k)
    log_p = np.log10(pressure_bar)
    log_t = np.log10(temperature_k)
    for i in range(1, len(adjusted)):
        gradient = (log_t[i] - log_t[i - 1]) / (log_p[i] - log_p[i - 1])
        if gradient > adiabatic_gradient:
            p_top = pressure_bar[i - 1]
            t_top = adjusted[i - 1]
            adjusted[i:] = t_top * (pressure_bar[i:] / p_top) ** adiabatic_gradient
            break
    return adjusted


def _validate_temperature_profile(
    temperature_k: np.ndarray,
    *,
    validation: dict[str, Any],
) -> tuple[bool, str]:
    """Check whether a temperature profile meets the shared validity criteria.

    Validation bounds live at ``temperature_profiles.validation`` and apply
    uniformly to analytic, Roth, and any other profile source.

    Parameters
    ----------
    temperature_k : np.ndarray
        Temperature profile in Kelvin.
    validation : dict
        Shared temperature validation config.

    Returns
    -------
    tuple[bool, str]
        ``(is_valid, reason)`` — True if valid, else a description of the failure.
    """
    if not np.all(np.isfinite(temperature_k)):
        return False, "Profile contains non-finite values"
    t_min = float(validation["min_temperature_k"])
    t_max = float(validation["max_temperature_k"])
    if np.any(temperature_k < t_min):
        return False, f"Temperature below {t_min:.0f} K (min: {np.min(temperature_k):.1f} K)"
    if np.any(temperature_k > t_max):
        return False, f"Temperature above {t_max:.0f} K (max: {np.max(temperature_k):.1f} K)"
    return True, ""


def _sample_analytic_temperature_profile_record(
    pressure_bar: np.ndarray,
    *,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample a valid analytic PT profile via rejection sampling.

    Draws parameters from the configured distributions, computes the
    three-channel Line et al. (2013) radiative equilibrium profile,
    optionally applies convective adjustment and a temperature shift,
    then validates.  If the profile fails validation it is discarded
    and a fresh draw is attempted (up to ``_MAX_PROFILE_ATTEMPTS``).

    The profile equation is:
        T^4(tau) = (3*T_int^4/4) * (2/3 + tau)
                 + (3*T_irr^4/4) * ((1-alpha)*xi_1 + alpha*xi_2)

    Parameters
    ----------
    pressure_bar : np.ndarray
        Pressure grid in bar, shape ``(nz,)``.
    config : dict
        Validated pipeline config (must contain ``temperature_profiles.analytic_sampler``).
    rng : np.random.Generator
        Random number generator for reproducible sampling.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        ``(temperature_k, metadata)`` — the temperature profile in Kelvin and a
        flat dictionary of all sampled parameter values for provenance tracking.
    """
    sampler = config["temperature_profiles"]["analytic_sampler"]
    validation = config["temperature_profiles"]["validation"]
    gravity_m_s2 = float(sampler["reference_gravity_m_s2"])

    for attempt in range(_MAX_PROFILE_ATTEMPTS):
        # --- Draw parameters ---
        t_int_k = abs(_sample_normal_value(sampler["t_int_k_normal"], rng=rng))
        t_irr_k = abs(_sample_normal_value(sampler["t_irr_k_normal"], rng=rng))
        log10_kappa_ir = _sample_normal_value(sampler["log10_kappa_ir_m2_kg_normal"], rng=rng)
        power_law_n = _sample_range_value(sampler["power_law_n_range"], rng=rng)
        log10_gamma_1 = _sample_range_value(sampler["log10_gamma_1_range"], rng=rng)
        log10_gamma_2 = _sample_range_value(sampler["log10_gamma_2_range"], rng=rng)
        alpha = _sample_range_value(sampler["alpha_range"], rng=rng)
        temperature_shift_k = _sample_range_value(sampler["temperature_shift_k_range"], rng=rng)

        kappa_ir = 10.0 ** log10_kappa_ir
        gamma_1 = 10.0 ** log10_gamma_1
        gamma_2 = 10.0 ** log10_gamma_2

        # --- Compute optical depth and profile ---
        try:
            optical_depth = _compute_optical_depth(
                pressure_bar,
                kappa_ir_m2_kg=kappa_ir,
                gravity_m_s2=gravity_m_s2,
                power_law_n=power_law_n,
            )
        except ValueError:
            continue

        xi_1 = _compute_xi(gamma_1, optical_depth)
        xi_2 = _compute_xi(gamma_2, optical_depth)

        t4_deep = (3.0 * t_int_k ** 4 / 4.0) * (2.0 / 3.0 + optical_depth)
        t4_channel1 = (3.0 * t_irr_k ** 4 / 4.0) * (1.0 - alpha) * xi_1
        t4_channel2 = (3.0 * t_irr_k ** 4 / 4.0) * alpha * xi_2
        t4_total = t4_deep + t4_channel1 + t4_channel2

        if np.any(t4_total < 0):
            continue

        profile_k = t4_total ** 0.25

        # --- Convective adjustment (probability-gated) ---
        convective_adjustment_applied = bool(
            rng.random() < float(sampler["convection_probability"])
        )
        adiabatic_gradient = None
        if convective_adjustment_applied:
            adiabatic_gradient = _sample_range_value(
                sampler["adiabatic_gradient_range"], rng=rng,
            )
            profile_k = _apply_convective_adjustment(
                pressure_bar, profile_k,
                adiabatic_gradient=adiabatic_gradient,
            )

        # --- Temperature shift ---
        profile_k = profile_k + temperature_shift_k

        # --- Validate (reject if out of bounds) ---
        is_valid, reason = _validate_temperature_profile(
            profile_k, validation=validation,
        )
        if not is_valid:
            logger.debug("Analytic profile attempt %d rejected: %s", attempt + 1, reason)
            continue

        metadata: dict[str, Any] = {
            "source": "analytic",
            "analytic_profile_type": "line_2013",
            "analytic_reference_gravity_m_s2": gravity_m_s2,
            "analytic_t_int_k": t_int_k,
            "analytic_t_irr_k": t_irr_k,
            "analytic_log10_kappa_ir_m2_kg": log10_kappa_ir,
            "analytic_kappa_ir_m2_kg": kappa_ir,
            "analytic_power_law_n": power_law_n,
            "analytic_log10_gamma_1": log10_gamma_1,
            "analytic_log10_gamma_2": log10_gamma_2,
            "analytic_gamma_1": gamma_1,
            "analytic_gamma_2": gamma_2,
            "analytic_alpha": alpha,
            "analytic_temperature_shift_k": temperature_shift_k,
            "analytic_convective_adjustment_applied": convective_adjustment_applied,
        }
        if adiabatic_gradient is not None:
            metadata["analytic_adiabatic_gradient"] = adiabatic_gradient
        return np.asarray(profile_k, dtype=np.float64), metadata

    raise RuntimeError(
        f"Failed to generate a valid analytic profile after {_MAX_PROFILE_ATTEMPTS} attempts. "
        "Consider widening the validation bounds or narrowing parameter distributions."
    )


def _resolve_roth_data_glob(config: dict[str, Any], roth_cfg: dict[str, Any]) -> str:
    """Resolve a Roth profile glob against the project root when needed."""
    data_glob = str(roth_cfg["data_glob"])
    project_root = config.get("_project_root")
    if project_root is None:
        return data_glob
    glob_path = Path(data_glob)
    if glob_path.is_absolute():
        return data_glob
    return str((Path(project_root) / glob_path).resolve())


def _load_configured_roth_profiles(
    pressure_bar: np.ndarray,
    *,
    config: dict[str, Any],
    roth_cfg: dict[str, Any],
) -> list[RothProfile]:
    """Load the configured Roth profiles onto the requested pressure grid."""
    data_glob = _resolve_roth_data_glob(config, roth_cfg)
    filter_items: tuple[tuple[str, RothFilterValue], ...] = tuple(
        sorted(roth_cfg.get("filters", {}).items())
    )
    validation_items = tuple(
        sorted(config["temperature_profiles"]["validation"].items())
    )
    pressure_key = tuple(np.asarray(pressure_bar, dtype=np.float64).tolist())
    cache_key = (data_glob, pressure_key, filter_items, validation_items)
    cache = config.setdefault("_roth_profile_cache", {})
    profiles = cache.get(cache_key)
    if profiles is None:
        # Cache the interpolated PT-library profiles so mixed sampling does not reload them per run.
        loaded_profiles = load_roth_profiles(
            data_glob,
            pressure_grid_bar=pressure_bar,
            filters=roth_cfg.get("filters", {}),
        )
        validation = config["temperature_profiles"]["validation"]
        profiles = []
        rejected_profiles = 0
        for profile in loaded_profiles:
            temperature_k = np.asarray(profile.temperature_k, dtype=np.float64)
            is_valid, reason = _validate_temperature_profile(
                temperature_k,
                validation=validation,
            )
            if is_valid:
                profiles.append(profile)
                continue
            rejected_profiles += 1
            logger.debug(
                "Rejected PT-library profile %s: %s",
                profile.metadata.get("source_file", "<unknown>"),
                reason,
            )
        if rejected_profiles:
            logger.debug(
                "Rejected %d PT-library profiles outside shared temperature bounds.",
                rejected_profiles,
            )
        cache[cache_key] = profiles
    if not profiles:
        raise FileNotFoundError(
            "roth_sampler.enabled=true but no temperature profiles matched "
            f"{data_glob!r} after applying filters and shared temperature validation."
        )
    return profiles


def _choose_roth_profile(
    profiles: list[RothProfile],
    *,
    rng: np.random.Generator,
) -> RothProfile:
    """Choose one Roth profile from a preloaded profile list."""
    return profiles[int(rng.integers(0, len(profiles)))]


def _sample_roth_profile(
    pressure_bar: np.ndarray,
    *,
    config: dict[str, Any],
    roth_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> RothProfile:
    """Sample one interpolated Roth profile from the configured library."""
    profiles = _load_configured_roth_profiles(
        pressure_bar,
        config=config,
        roth_cfg=roth_cfg,
    )
    return _choose_roth_profile(profiles, rng=rng)


def _temperature_profile_metadata(
    profile: RothProfile | None,
    *,
    analytic_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build flat provenance metadata for the selected temperature-profile source."""
    if profile is None:
        metadata: dict[str, Any] = {"temperature_profile_source": "analytic"}
        if analytic_metadata:
            for key, value in analytic_metadata.items():
                if key == "source":
                    continue
                metadata[f"temperature_profile_{key}"] = value
        return metadata
    metadata = {"temperature_profile_source": "pt_library"}
    for key, value in profile.metadata.items():
        metadata[f"temperature_profile_{key}"] = value
    return metadata


def _sample_temperature_profile_record(
    pressure_bar: np.ndarray,
    *,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample one temperature profile together with flat provenance metadata."""
    roth_cfg = config.get("roth_sampler", {"enabled": False})
    if roth_cfg.get("enabled", False):
        if roth_cfg.get("source_mode", "roth") == "mixed":
            profiles = _load_configured_roth_profiles(
                pressure_bar,
                config=config,
                roth_cfg=roth_cfg,
            )
            analytic_probability = float(roth_cfg["analytic_probability"])
            # Mixed mode lets one dataset cover both analytic shapes and PT-library profiles.
            if float(rng.random()) >= analytic_probability:
                chosen = _choose_roth_profile(profiles, rng=rng)
                return (
                    np.asarray(chosen.temperature_k, dtype=np.float64),
                    _temperature_profile_metadata(chosen),
                )
        else:
            chosen = _sample_roth_profile(
                pressure_bar,
                config=config,
                roth_cfg=roth_cfg,
                rng=rng,
            )
            return (
                np.asarray(chosen.temperature_k, dtype=np.float64),
                _temperature_profile_metadata(chosen),
            )

    analytic_profile, analytic_metadata = _sample_analytic_temperature_profile_record(
        pressure_bar,
        config=config,
        rng=rng,
    )
    return (
        analytic_profile,
        _temperature_profile_metadata(None, analytic_metadata=analytic_metadata),
    )


def sample_temperature_profile(
    pressure_bar: np.ndarray,
    *,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a temperature profile from the analytic, Roth, or mixed-source path."""
    temperature_k, _ = _sample_temperature_profile_record(
        pressure_bar,
        config=config,
        rng=rng,
    )
    return temperature_k


def sample_kzz_profile(
    pressure_bar: np.ndarray,
    *,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a depth-constant eddy diffusion profile."""
    del rng
    kzz_value = float(np.clip(config["sampling"]["kzz_cm2_s"], 1.0, None))
    return np.full(np.asarray(pressure_bar).shape, kzz_value, dtype=np.float64)


def _latin_hypercube_unit_samples(
    *,
    num_samples: int,
    num_dimensions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a Latin-hypercube design matrix in the unit hypercube [0, 1]^d.

    Latin-hypercube sampling (LHS) ensures that each dimension is stratified
    into ``num_samples`` equal-probability bins, with exactly one sample per
    bin.  This provides better coverage of the parameter space than pure
    random sampling, especially at moderate sample counts.

    Parameters
    ----------
    num_samples : int
        Number of samples (rows) in the design matrix.  Must be >= 1.
    num_dimensions : int
        Number of dimensions (columns).
    rng : np.random.Generator
        Random number generator for reproducible designs.

    Returns
    -------
    np.ndarray
        Design matrix of shape ``(num_samples, num_dimensions)`` with values
        in [0, 1].
    """
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1.")
    cutpoints = np.linspace(0.0, 1.0, num_samples + 1, dtype=np.float64)
    samples = np.empty((num_samples, num_dimensions), dtype=np.float64)
    for dim in range(num_dimensions):
        offsets = rng.uniform(0.0, 1.0, size=num_samples)
        coords = cutpoints[:-1] + offsets * (cutpoints[1:] - cutpoints[:-1])
        samples[:, dim] = coords[rng.permutation(num_samples)]
    return samples


def _scale_unit_interval(value: float, lower: float, upper: float) -> float:
    """Linearly map a unit-interval sample into a physical range."""
    return float(lower + value * (upper - lower))


def _ensure_wasp39_template(
    *,
    project_root: Path,
    config: dict[str, Any],
) -> SpectrumRecord:
    """Load the configured stellar template from disk."""
    spectrum_cfg = config["stellar_spectrum"]
    template_path = (project_root / spectrum_cfg["template_file"]).resolve()
    if not template_path.exists():
        raise FileNotFoundError(
            f"Configured stellar_spectrum.template_file does not exist: {template_path}"
        )
    from .spectrum import read_vulcan_spectrum_txt

    return read_vulcan_spectrum_txt(template_path, name=spectrum_cfg["template_name"])


def ensure_default_spectrum_library(
    *,
    project_root: Path,
    config: dict[str, Any],
) -> Path:
    """Write the default fixed-grid spectrum library and return its manifest path."""
    output_dir = project_root / "data" / "spectra_library"
    output_dir.mkdir(parents=True, exist_ok=True)
    template = _ensure_wasp39_template(project_root=project_root, config=config)
    manifest_path = output_dir / "manifest.json"
    save_spectrum_manifest([template], output_dir)
    return manifest_path


def load_default_spectra(
    *,
    project_root: Path,
    config: dict[str, Any],
) -> dict[str, SpectrumRecord]:
    """Load the shipped spectrum library records keyed by spectrum name."""
    manifest = ensure_default_spectrum_library(project_root=project_root, config=config)
    return load_spectrum_manifest(manifest)


def sample_run_specifications(
    *,
    config: dict[str, Any],
    project_root: Path,
    num_runs: int | None = None,
    seed: int | None = None,
) -> list[RunSpecification]:
    """Sample the full set of atmospheric configurations used to generate raw runs.

    Uses Latin-hypercube sampling (LHC) to stratify the global conditioning
    scalars (metallicity, C/O, S/O, and optionally gravity) over the configured
    ranges.  For each run, a temperature profile is independently drawn from
    the configured source (analytic, PT-library, or mixed).

    Parameters
    ----------
    config : dict
        Validated pipeline config.
    project_root : Path
        Filesystem root of the project (for resolving relative paths).
    num_runs : int or None
        Override for ``generation.num_runs``.
    seed : int or None
        Override for ``generation.seed``.

    Returns
    -------
    list[RunSpecification]
        One specification per run, ready for raw data generation.
    """
    rng = np.random.default_rng(
        int(config["generation"]["seed"] if seed is None else seed)
    )
    total_runs = int(config["generation"]["num_runs"] if num_runs is None else num_runs)
    equilibrium = is_equilibrium(config)

    if equilibrium:
        # LHC over (metallicity, C/O, S/O) — 3 dimensions, no gravity.
        design = _latin_hypercube_unit_samples(
            num_samples=total_runs, num_dimensions=3, rng=rng,
        )
    else:
        # LHC over (gravity, metallicity, C/O, S/O) — 4 dimensions.
        design = _latin_hypercube_unit_samples(
            num_samples=total_runs, num_dimensions=4, rng=rng,
        )

    # Spectrum loading only needed for full-VULCAN models.
    spectra: dict[str, SpectrumRecord] | None = None
    spectrum_names: list[str] | None = None
    if not equilibrium:
        spectra = load_default_spectra(project_root=project_root, config=config)
        if not spectra:
            raise RuntimeError("No stellar spectra available for sampling.")
        spectrum_names = sorted(spectra.keys())

    pressure_bar = sample_pressure_grid(
        num_levels=int(config["sampling"]["num_levels"]),
        pressure_top_bar=float(config["sampling"]["pressure_top_bar"]),
        pressure_bottom_bar=float(config["sampling"]["pressure_bottom_bar"]),
    )
    result: list[RunSpecification] = []
    physics_defaults = static_conditioning_defaults(config)
    for run_idx in range(total_runs):
        if equilibrium:
            metallicity = _scale_unit_interval(
                design[run_idx, 0],
                *[float(x) for x in config["sampling"]["metallicity_log10_range"]],
            )
            c_to_o = _scale_unit_interval(
                design[run_idx, 1],
                *[float(x) for x in config["sampling"]["c_to_o_range"]],
            )
            s_to_o = _scale_unit_interval(
                design[run_idx, 2],
                *[float(x) for x in config["sampling"]["s_to_o_range"]],
            )
        else:
            gravity = _scale_unit_interval(
                design[run_idx, 0],
                *[float(x) for x in config["sampling"]["gravity_range_cm_s2"]],
            )
            metallicity = _scale_unit_interval(
                design[run_idx, 1],
                *[float(x) for x in config["sampling"]["metallicity_log10_range"]],
            )
            c_to_o = _scale_unit_interval(
                design[run_idx, 2],
                *[float(x) for x in config["sampling"]["c_to_o_range"]],
            )
            s_to_o = _scale_unit_interval(
                design[run_idx, 3],
                *[float(x) for x in config["sampling"]["s_to_o_range"]],
            )

        temperature_k, temperature_metadata = _sample_temperature_profile_record(
            pressure_bar,
            config=config,
            rng=rng,
        )
        if equilibrium:
            base_globals: dict[str, float] = {
                "metallicity_log10": float(metallicity),
                "c_to_o": float(c_to_o),
                "s_to_o": float(s_to_o),
            }
            element_scalars = _element_scalars_from_sampled_globals(base_globals)
            gravity_profile = _equilibrium_gravity_profile(
                pressure_bar=np.asarray(pressure_bar, dtype=np.float64),
                config=config,
            )
        else:
            base_globals = {
                "gravity_cm_s2": float(gravity),
                "metallicity_log10": float(metallicity),
                "c_to_o": float(c_to_o),
                "s_to_o": float(s_to_o),
            }
            element_scalars = _element_scalars_from_sampled_globals(base_globals)
            gravity_profile = np.full(
                np.asarray(pressure_bar).shape,
                float(gravity),
                dtype=np.float64,
            )
        elemental_profile = _element_profile_from_scalars(
            element_scalars,
            num_levels=np.asarray(pressure_bar).size,
        )

        if equilibrium:
            globals_map = {**base_globals, **element_scalars}
            result.append(
                RunSpecification(
                    run_id=f"run_{run_idx:05d}",
                    pressure_bar=np.asarray(pressure_bar, dtype=np.float64),
                    temperature_k=np.asarray(temperature_k, dtype=np.float64),
                    globals=globals_map,
                    metadata=dict(temperature_metadata),
                    elemental_abundances_x_h=elemental_profile,
                    gravity_cm_s2=gravity_profile,
                )
            )
        else:
            kzz = sample_kzz_profile(pressure_bar, config=config, rng=rng)
            assert spectra is not None and spectrum_names is not None
            spectrum_name = spectrum_names[int(rng.integers(0, len(spectrum_names)))]
            template = spectra[spectrum_name]
            spectrum = SpectrumRecord(
                name=f"{template.name}_run{run_idx:05d}",
                wavelength_nm=np.asarray(template.wavelength_nm, dtype=np.float64),
                flux_erg_cm2_s_nm=np.asarray(template.flux_erg_cm2_s_nm, dtype=np.float64),
                metadata={**template.metadata, "template_name": template.name},
            )
            globals_map = {
                **base_globals,
                **element_scalars,
                **{key: float(value) for key, value in physics_defaults.items()},
            }
            result.append(
                RunSpecification(
                    run_id=f"run_{run_idx:05d}",
                    pressure_bar=np.asarray(pressure_bar, dtype=np.float64),
                    temperature_k=np.asarray(temperature_k, dtype=np.float64),
                    globals=globals_map,
                    metadata={
                        **temperature_metadata,
                        "spectrum_name": spectrum.name,
                    },
                    kzz_cm2_s=np.asarray(kzz, dtype=np.float64),
                    spectrum=spectrum,
                    elemental_abundances_x_h=elemental_profile,
                    gravity_cm_s2=gravity_profile,
                )
            )
    return result
