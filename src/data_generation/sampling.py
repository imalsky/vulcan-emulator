"""Atmospheric run specification sampling: temperature profiles, pressure grids.

This module is the primary entry point for constructing the atmospheric
configurations used to generate raw training runs.  It supports three
temperature-profile sources, selected by ``temperature_profiles.source_mode``:

- **analytic**: Profiles are generated from the Piette & Madhusudhan (2019)
  modified Guillot (2010) radiative-equilibrium parameterization with an
  optional convective adjustment.  Eight random parameters are drawn per
  profile (six physics plus two convection; see
  ``_sample_analytic_temperature_profile_record``).

- **pt_library**: Profiles are loaded from externally computed GCM output files
  (Roth .dat format).  Each ``(lon, lat)`` column in a file is expanded into a
  separate 1D profile and interpolated onto the configured pressure grid using
  shape-preserving PCHIP interpolation in log-pressure space.

- **mixed**: Each run independently selects analytic or PT-library with a
  configurable probability (``temperature_profiles.analytic_probability``),
  giving training sets that cover both parameterized shapes and realistic
  GCM-derived profiles.

The module also handles Latin-hypercube sampling of the global conditioning
scalars (metallicity, C/O, S/O, and for VULCAN also surface gravity, planet
radius, and irradiation geometry) plus stellar-spectrum selection for VULCAN
chemistry runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import ELEMENT_INPUT_ORDER, PUBLIC_PHYSICS_TOGGLES, SUPPORTED_ATM_BASES
from ..utils.config import uses_fastchem
from ..utils.helpers import get_logger
from .roth_sampling import RothFilterValue, RothProfile, load_roth_profiles
from .spectrum import (
    SpectrumRecord,
    load_spectrum_records_from_glob,
)

LOGGER = get_logger(__name__)

# Maximum number of rejection-resampling attempts for analytic profiles.
_MAX_PROFILE_ATTEMPTS = 100


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
    elemental_abundances_frac: np.ndarray | None = None
    gravity_cm_s2: np.ndarray | None = None


def _element_fractions_from_sampled_globals(globals_map: dict[str, float]) -> dict[str, float]:
    """Extract elemental number fractions from sampled globals.

    Parameters
    ----------
    globals_map : dict[str, float]
        Sampled global scalars containing ``He_H``, ``C_H``,
        ``O_H``, ``N_H``, and ``S_H``.

    Returns
    -------
    dict[str, float]
        Elemental number fractions keyed by ``ELEMENT_INPUT_ORDER`` names.
        The implicit hydrogen fraction is ``1 - sum(fractions)``.
    """

    fractions = {name: float(globals_map[name]) for name in ELEMENT_INPUT_ORDER}
    total = sum(fractions.values())
    if total >= 1.0:
        raise ValueError(
            f"Elemental number fractions sum to {total:.6f} >= 1.0; "
            "hydrogen remainder would be non-positive."
        )
    return fractions


def _element_profile_from_fractions(
    element_fractions: dict[str, float],
    *,
    num_levels: int,
) -> np.ndarray:
    """Broadcast elemental fractions to a per-level profile tensor.

    Parameters
    ----------
    element_fractions : dict[str, float]
        Elemental number fractions keyed by ``ELEMENT_INPUT_ORDER``.
    num_levels : int
        Number of vertical levels in the atmospheric column.

    Returns
    -------
    np.ndarray
        Profile tensor of shape ``(num_levels, len(ELEMENT_INPUT_ORDER))``.
    """

    vector = np.array(
        [float(element_fractions[name]) for name in ELEMENT_INPUT_ORDER],
        dtype=np.float64,
    )
    return np.repeat(vector[None, :], int(num_levels), axis=0)


def _equilibrium_gravity_profile(
    *,
    pressure_bar: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Build the per-level gravity array required by equilibrium datasets.

    Parameters
    ----------
    pressure_bar : np.ndarray
        Pressure grid whose shape defines the output profile length.
    config : dict[str, Any]
        Validated config containing
        ``temperature_profiles.analytic_sampler.reference_gravity_m_s2``.

    Returns
    -------
    np.ndarray
        Column-constant gravity profile with shape matching ``pressure_bar``.
    """
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
    """Sample one scalar uniformly from a two-endpoint numeric range.

    Parameters
    ----------
    values : list[float] or tuple[float, float]
        Two-element ``[lower, upper]`` range specification from the config.
    rng : np.random.Generator
        Random number generator driving reproducible sampling.

    Returns
    -------
    float
        Uniform sample between the lower and upper endpoints.
    """
    lower = float(values[0])
    upper = float(values[1])
    return float(rng.uniform(lower, upper))


def _guillot_temperature(
    pressure_bar: np.ndarray,
    *,
    delta: float,
    gamma: float,
    t_int_k: float,
    t_eq_k: float,
) -> np.ndarray:
    """Compute the Guillot (2010) radiative-equilibrium temperature profile.

    Implements Eq. 16 from Piette & Madhusudhan (2019):

        T^4(P) = (3/4)*T_int^4*(2/3 + delta*P)
               + (3/4)*T_eq^4*[2/3 + 1/(gamma*sqrt(3))
               + (gamma/sqrt(3) - 1/(gamma*sqrt(3)))
                 * exp(-gamma*delta*sqrt(3)*P)]

    Parameters
    ----------
    pressure_bar : np.ndarray
        Pressure grid in bar, shape ``(nz,)``.
    delta : float
        Ratio of infrared opacity to gravity (kappa_IR / g).  Must be > 0.
    gamma : float
        Ratio of visible to infrared mean opacity.  Must be > 0.
    t_int_k : float
        Internal (intrinsic) temperature in Kelvin.
    t_eq_k : float
        Equilibrium temperature in Kelvin.

    Returns
    -------
    np.ndarray
        Temperature profile in Kelvin, same shape as *pressure_bar*.
    """
    pressure_bar = np.asarray(pressure_bar, dtype=np.float64)
    sqrt3 = math.sqrt(3.0)
    tau = delta * pressure_bar

    t4_internal = 0.75 * t_int_k ** 4 * (2.0 / 3.0 + tau)
    inv_gamma_sqrt3 = 1.0 / (gamma * sqrt3)
    gamma_sqrt3 = gamma * sqrt3
    t4_stellar = 0.75 * t_eq_k ** 4 * (
        2.0 / 3.0
        + inv_gamma_sqrt3
        + (gamma / sqrt3 - inv_gamma_sqrt3) * np.exp(-gamma_sqrt3 * tau)
    )
    t4_total = t4_internal + t4_stellar
    return np.clip(t4_total, 0.0, None) ** 0.25


def _apply_upper_atmosphere_modification(
    pressure_bar: np.ndarray,
    temperature_k: np.ndarray,
    *,
    alpha: float,
    p_trans_bar: float,
    smoothing_width_dex: float = 1.25,
) -> np.ndarray:
    """Apply the Piette & Madhusudhan (2019) upper-atmosphere modification.

    Implements Eq. 15:

        T(P) = <T_Guillot(P) * (1 - alpha / (1 + P / P_trans))>_P

    where ``<...>_P`` denotes boxcar smoothing over *smoothing_width_dex*
    decades in log10(P).

    Parameters
    ----------
    pressure_bar : np.ndarray
        Pressure grid in bar, shape ``(nz,)``.  Must be uniformly spaced
        in log10(P).
    temperature_k : np.ndarray
        Guillot base temperature profile in Kelvin, shape ``(nz,)``.
    alpha : float
        Strength of the upper-atmosphere modification.  Must lie in [0, 1).
    p_trans_bar : float
        Transition pressure in bar.
    smoothing_width_dex : float, optional
        Width of the boxcar smoothing window in decades of log10(P).
        Default is 1.25 per the original paper.

    Returns
    -------
    np.ndarray
        Modified and smoothed temperature profile, shape ``(nz,)``.
    """
    modified = temperature_k * (1.0 - alpha / (1.0 + pressure_bar / p_trans_bar))

    nz = len(pressure_bar)
    if nz < 2:
        return modified

    dp = abs(np.log10(pressure_bar[1]) - np.log10(pressure_bar[0]))
    if dp <= 0.0:
        return modified

    window = max(1, int(round(smoothing_width_dex / dp)))
    if window <= 1:
        return modified

    pad = window // 2
    padded = np.pad(modified, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[:nz]


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

    Draws parameters from uniform distributions, computes the Piette &
    Madhusudhan (2019) modified Guillot (2010) radiative-equilibrium
    profile, optionally applies convective adjustment, then validates.
    If the profile fails validation it is discarded and a fresh draw is
    attempted (up to ``_MAX_PROFILE_ATTEMPTS``).

    The profile equations are (Piette & Madhusudhan 2019, Eqs. 15-16):

        T_G^4(P) = (3/4)*T_int^4*(2/3 + delta*P)
                 + (3/4)*T_eq^4*[2/3 + 1/(gamma*sqrt(3))
                 + (gamma/sqrt(3) - 1/(gamma*sqrt(3)))
                   * exp(-gamma*delta*sqrt(3)*P)]

        T(P) = <T_G(P) * (1 - alpha/(1 + P/P_trans))>_P

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

    for attempt in range(_MAX_PROFILE_ATTEMPTS):
        # --- Draw parameters ---
        t_int_k = _sample_range_value(sampler["t_int_k_range"], rng=rng)
        t_eq_k = _sample_range_value(sampler["t_eq_k_range"], rng=rng)
        log10_delta = _sample_range_value(sampler["log10_delta_range"], rng=rng)
        log10_gamma = _sample_range_value(sampler["log10_gamma_range"], rng=rng)
        alpha = _sample_range_value(sampler["alpha_range"], rng=rng)
        log10_p_trans = _sample_range_value(sampler["log10_p_trans_bar_range"], rng=rng)

        delta = 10.0 ** log10_delta
        gamma = 10.0 ** log10_gamma
        p_trans_bar = 10.0 ** log10_p_trans

        # --- Compute Guillot base profile ---
        profile_k = _guillot_temperature(
            pressure_bar,
            delta=delta,
            gamma=gamma,
            t_int_k=t_int_k,
            t_eq_k=t_eq_k,
        )

        if not np.all(np.isfinite(profile_k)):
            continue

        # --- Upper atmosphere modification + smoothing ---
        profile_k = _apply_upper_atmosphere_modification(
            pressure_bar, profile_k,
            alpha=alpha,
            p_trans_bar=p_trans_bar,
        )

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

        # --- Validate (reject if out of bounds) ---
        is_valid, reason = _validate_temperature_profile(
            profile_k, validation=validation,
        )
        if not is_valid:
            LOGGER.debug("Analytic profile attempt %d rejected: %s", attempt + 1, reason)
            continue

        metadata: dict[str, Any] = {
            "source": "analytic",
            "analytic_profile_type": "piette_2019",
            "analytic_t_int_k": t_int_k,
            "analytic_t_eq_k": t_eq_k,
            "analytic_log10_delta": log10_delta,
            "analytic_delta": delta,
            "analytic_log10_gamma": log10_gamma,
            "analytic_gamma": gamma,
            "analytic_alpha": alpha,
            "analytic_log10_p_trans_bar": log10_p_trans,
            "analytic_p_trans_bar": p_trans_bar,
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
    """Resolve a PT-library glob against the project root when needed.

    Parameters
    ----------
    config : dict[str, Any]
        Runtime config that may contain ``_project_root``.
    roth_cfg : dict[str, Any]
        Temperature-profile config block containing ``data_glob``.

    Returns
    -------
    str
        Absolute or unchanged glob string ready for filesystem expansion.
    """
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
    """Load, validate, cache, and interpolate configured PT-library profiles.

    Parameters
    ----------
    pressure_bar : np.ndarray
        Target pressure grid in bar with shape ``(nz,)``.
    config : dict[str, Any]
        Runtime config containing shared temperature validation bounds and an
        internal cache.
    roth_cfg : dict[str, Any]
        Temperature-profile configuration containing the PT-library glob and
        optional filters.

    Returns
    -------
    list[RothProfile]
        Interpolated Roth profiles that satisfy both metadata filters and the
        shared temperature validity bounds.
    """
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
            LOGGER.debug(
                "Rejected PT-library profile %s: %s",
                profile.metadata.get("source_file", "<unknown>"),
                reason,
            )
        if rejected_profiles:
            LOGGER.debug(
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
    """Select one preloaded Roth pressure-temperature profile at random.

    Parameters
    ----------
    profiles : list[RothProfile]
        In-memory catalog of Roth profiles already validated and loaded from
        disk.
    rng : np.random.Generator
        Random number generator used to choose the profile index.

    Returns
    -------
    RothProfile
        One randomly selected profile record from ``profiles``.
    """
    return profiles[int(rng.integers(0, len(profiles)))]


def _sample_roth_profile(
    pressure_bar: np.ndarray,
    *,
    config: dict[str, Any],
    roth_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> RothProfile:
    """Sample one interpolated PT-library profile from the configured pool.

    Parameters
    ----------
    pressure_bar : np.ndarray
        Target pressure grid in bar with shape ``(nz,)``.
    config : dict[str, Any]
        Runtime config used to load and cache available profiles.
    roth_cfg : dict[str, Any]
        PT-library configuration block.
    rng : np.random.Generator
        Random generator used to choose one candidate profile.

    Returns
    -------
    RothProfile
        One interpolated profile sampled from the configured library.
    """
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
    """Build flat provenance metadata for the selected temperature profile.

    Parameters
    ----------
    profile : RothProfile or None
        Chosen PT-library profile, or ``None`` when the analytic sampler was
        used.
    analytic_metadata : dict[str, Any] or None, optional
        Parameter metadata returned by the analytic sampler.

    Returns
    -------
    dict[str, Any]
        Flat metadata dictionary suitable for storage in run provenance.
    """
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
    """Sample one temperature profile together with provenance metadata.

    Parameters
    ----------
    pressure_bar : np.ndarray
        Target pressure grid in bar with shape ``(nz,)``.
    config : dict[str, Any]
        Validated config describing the temperature-profile source mode and
        associated sampler settings.
    rng : np.random.Generator
        Random generator used for mixed-mode branching and parameter draws.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        Temperature profile in Kelvin with shape ``(nz,)`` plus a flat
        metadata dictionary describing the sampled source and parameters.
    """
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
    """Sample a temperature profile on the supplied pressure grid.

    Parameters
    ----------
    pressure_bar : np.ndarray
        Target pressure grid in bar with shape ``(nz,)``.
    config : dict[str, Any]
        Validated config selecting the analytic, PT-library, or mixed source.
    rng : np.random.Generator
        Random generator used by the selected sampler.

    Returns
    -------
    np.ndarray
        Temperature profile in Kelvin with shape ``(nz,)``.
    """
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
    """Construct the configured depth-constant eddy-diffusion profile.

    Parameters
    ----------
    pressure_bar : np.ndarray
        Pressure grid whose shape defines the output profile length.
    config : dict[str, Any]
        Validated config containing ``sampling.kzz_cm2_s``.
    rng : np.random.Generator
        Unused random generator kept for a uniform sampler interface.

    Returns
    -------
    np.ndarray
        Constant ``Kzz`` profile with shape matching ``pressure_bar``.
    """
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
    """Map a unit-interval coordinate onto a physical parameter range.

    Parameters
    ----------
    value : float
        Sample in the unit interval, typically from Latin-hypercube sampling.
    lower : float
        Physical lower bound for the parameter.
    upper : float
        Physical upper bound for the parameter.

    Returns
    -------
    float
        Linearly rescaled value in ``[lower, upper]``.
    """
    return float(lower + value * (upper - lower))


def _ensure_stellar_template(
    *,
    project_root: Path,
    config: dict[str, Any],
) -> SpectrumRecord:
    """Load or synthesize the configured default stellar template.

    Parameters
    ----------
    project_root : Path
        Repository root used to resolve the template path.
    config : dict[str, Any]
        Validated config containing ``stellar_spectrum.template_file`` and
        ``template_name``.

    Returns
    -------
    SpectrumRecord
        Parsed or synthesized stellar template record.
    """
    spectrum_cfg = config["stellar_spectrum"]
    template_file = spectrum_cfg.get("template_file")
    if template_file in {None, ""}:
        from .spectrum import generate_blackbody_template

        return generate_blackbody_template(
            wavelength_min_nm=float(spectrum_cfg["wavelength_min_nm"]),
            wavelength_max_nm=float(spectrum_cfg["wavelength_max_nm"]),
            teff_k=float(spectrum_cfg.get("teff_k") or 5485.0),
            radius_rsun=float(spectrum_cfg.get("radius_rsun") or 0.939),
            semi_major_axis_au=float(spectrum_cfg.get("semi_major_axis_au") or 0.04858),
            name=str(spectrum_cfg["template_name"]),
        )
    template_path = (project_root / str(template_file)).resolve()
    if not template_path.exists():
        raise FileNotFoundError(
            f"Configured stellar_spectrum.template_file does not exist: {template_path}"
        )
    from .spectrum import read_vulcan_spectrum_txt

    return read_vulcan_spectrum_txt(template_path, name=spectrum_cfg["template_name"])


def _resolve_spectrum_library_glob(
    *,
    project_root: Path,
    config: dict[str, Any],
) -> str | None:
    """Resolve the optional stellar-spectrum library glob.

    Parameters
    ----------
    project_root : Path
        Repository root used to resolve relative glob paths.
    config : dict[str, Any]
        Validated config containing the optional
        ``stellar_spectrum.library_glob`` field.

    Returns
    -------
    str or None
        Absolute glob string when configured, otherwise ``None``.
    """
    library_glob = config["stellar_spectrum"].get("library_glob")
    if not library_glob:
        return None
    glob_path = Path(str(library_glob))
    if glob_path.is_absolute():
        return str(glob_path)
    return str((project_root / glob_path).resolve())


def _load_configured_spectrum_records(
    *,
    project_root: Path,
    config: dict[str, Any],
) -> dict[str, SpectrumRecord]:
    """Load the configured stellar spectrum library for VULCAN sampling.

    Parameters
    ----------
    project_root : Path
        Repository root used to resolve template and glob paths.
    config : dict[str, Any]
        Validated config containing stellar-spectrum settings.

    Returns
    -------
    dict[str, SpectrumRecord]
        Mapping from spectrum name to in-memory spectrum record.
    """
    library_glob = _resolve_spectrum_library_glob(project_root=project_root, config=config)
    if library_glob:
        return load_spectrum_records_from_glob(library_glob)
    template = _ensure_stellar_template(project_root=project_root, config=config)
    return {template.name: template}


def _science_preset_conditioning_inputs(preset: dict[str, Any]) -> dict[str, float]:
    """Flatten one curated science preset into model-conditioning scalars.

    Parameters
    ----------
    preset : dict[str, Any]
        Science preset containing ``physics_toggles`` and ``atm_base``.

    Returns
    -------
    dict[str, float]
        Flat conditioning vector components including public physics toggles
        and one-hot atmosphere-base flags.
    """
    physics = {
        name: float(bool(preset["physics_toggles"][name]))
        for name in PUBLIC_PHYSICS_TOGGLES
    }
    atm_base = str(preset["atm_base"])
    one_hot = {
        f"atm_base_{name}": 1.0 if name == atm_base else 0.0
        for name in SUPPORTED_ATM_BASES
    }
    return {**physics, **one_hot}


def load_default_spectra(
    *,
    project_root: Path,
    config: dict[str, Any],
) -> dict[str, SpectrumRecord]:
    """Load the configured stellar spectrum library into memory.

    Parameters
    ----------
    project_root : Path
        Repository root used to resolve template and glob paths.
    config : dict[str, Any]
        Validated config describing the library source.

    Returns
    -------
    dict[str, SpectrumRecord]
        Spectrum records keyed by spectrum name.
    """
    return _load_configured_spectrum_records(project_root=project_root, config=config)


def sample_run_specifications(
    *,
    config: dict[str, Any],
    project_root: Path,
    num_runs: int | None = None,
    seed: int | None = None,
) -> list[RunSpecification]:
    """Sample the full set of atmospheric configurations used to generate raw runs.

    Uses Latin-hypercube sampling (LHC) to stratify the global conditioning
    scalars (metallicity, C/O, S/O, and for VULCAN also surface gravity,
    planet radius, stellar radius, orbital separation, zenith angle, and
    diurnal factor) over the configured ranges. For each run, a temperature
    profile is independently drawn from the configured source (analytic,
    PT-library, or mixed).

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
    fastchem = uses_fastchem(config)

    if fastchem:
        # LHC over (He_H, C_H, O_H, N_H, S_H) — 5 dimensions.
        design = _latin_hypercube_unit_samples(
            num_samples=total_runs, num_dimensions=5, rng=rng,
        )
    else:
        # LHC over (surface gravity, planet radius, stellar radius, orbit,
        # zenith angle, diurnal factor, He_H, C_H, O_H, N_H, S_H).
        design = _latin_hypercube_unit_samples(
            num_samples=total_runs, num_dimensions=11, rng=rng,
        )

    # Spectrum loading only needed for VULCAN chemistry.
    spectra: dict[str, SpectrumRecord] | None = None
    spectrum_names: list[str] | None = None
    science_presets: list[dict[str, Any]] | None = None
    if not fastchem:
        spectra = load_default_spectra(project_root=project_root, config=config)
        if not spectra:
            raise RuntimeError("No stellar spectra available for sampling.")
        spectrum_names = sorted(spectra.keys())
        science_presets = list(config["science_presets"])

    pressure_bar = sample_pressure_grid(
        num_levels=int(config["sampling"]["num_levels"]),
        pressure_top_bar=float(config["sampling"]["pressure_top_bar"]),
        pressure_bottom_bar=float(config["sampling"]["pressure_bottom_bar"]),
    )
    result: list[RunSpecification] = []
    for run_idx in range(total_runs):
        if fastchem:
            he_frac = _scale_unit_interval(
                design[run_idx, 0],
                *[float(x) for x in config["sampling"]["he_frac_range"]],
            )
            c_frac = _scale_unit_interval(
                design[run_idx, 1],
                *[float(x) for x in config["sampling"]["c_frac_range"]],
            )
            o_frac = _scale_unit_interval(
                design[run_idx, 2],
                *[float(x) for x in config["sampling"]["o_frac_range"]],
            )
            n_frac = _scale_unit_interval(
                design[run_idx, 3],
                *[float(x) for x in config["sampling"]["n_frac_range"]],
            )
            s_frac = _scale_unit_interval(
                design[run_idx, 4],
                *[float(x) for x in config["sampling"]["s_frac_range"]],
            )
        else:
            gravity = _scale_unit_interval(
                design[run_idx, 0],
                *[float(x) for x in config["sampling"]["gravity_range_cm_s2"]],
            )
            planet_radius_cm = _scale_unit_interval(
                design[run_idx, 1],
                *[float(x) for x in config["sampling"]["planet_radius_range_cm"]],
            )
            stellar_radius_rsun = _scale_unit_interval(
                design[run_idx, 2],
                *[float(x) for x in config["sampling"]["stellar_radius_range_rsun"]],
            )
            semi_major_axis_au = _scale_unit_interval(
                design[run_idx, 3],
                *[float(x) for x in config["sampling"]["semi_major_axis_range_au"]],
            )
            zenith_angle_deg = _scale_unit_interval(
                design[run_idx, 4],
                *[float(x) for x in config["sampling"]["zenith_angle_range_deg"]],
            )
            diurnal_factor = _scale_unit_interval(
                design[run_idx, 5],
                *[float(x) for x in config["sampling"]["diurnal_factor_range"]],
            )
            he_frac = _scale_unit_interval(
                design[run_idx, 6],
                *[float(x) for x in config["sampling"]["he_frac_range"]],
            )
            c_frac = _scale_unit_interval(
                design[run_idx, 7],
                *[float(x) for x in config["sampling"]["c_frac_range"]],
            )
            o_frac = _scale_unit_interval(
                design[run_idx, 8],
                *[float(x) for x in config["sampling"]["o_frac_range"]],
            )
            n_frac = _scale_unit_interval(
                design[run_idx, 9],
                *[float(x) for x in config["sampling"]["n_frac_range"]],
            )
            s_frac = _scale_unit_interval(
                design[run_idx, 10],
                *[float(x) for x in config["sampling"]["s_frac_range"]],
            )

        temperature_k, temperature_metadata = _sample_temperature_profile_record(
            pressure_bar,
            config=config,
            rng=rng,
        )
        if fastchem:
            base_globals: dict[str, float] = {
                "He_H": float(he_frac),
                "C_H": float(c_frac),
                "O_H": float(o_frac),
                "N_H": float(n_frac),
                "S_H": float(s_frac),
            }
            element_fractions = _element_fractions_from_sampled_globals(base_globals)
            gravity_profile = _equilibrium_gravity_profile(
                pressure_bar=np.asarray(pressure_bar, dtype=np.float64),
                config=config,
            )
        else:
            base_globals = {
                "gravity_cm_s2": float(gravity),
                "planet_radius_cm": float(planet_radius_cm),
                "r_star_rsun": float(stellar_radius_rsun),
                "semi_major_axis_au": float(semi_major_axis_au),
                "zenith_angle_deg": float(zenith_angle_deg),
                "diurnal_factor": float(diurnal_factor),
                "He_H": float(he_frac),
                "C_H": float(c_frac),
                "O_H": float(o_frac),
                "N_H": float(n_frac),
                "S_H": float(s_frac),
            }
            element_fractions = _element_fractions_from_sampled_globals(base_globals)
            gravity_profile = np.full(
                np.asarray(pressure_bar).shape,
                float(gravity),
                dtype=np.float64,
            )
        elemental_profile = _element_profile_from_fractions(
            element_fractions,
            num_levels=np.asarray(pressure_bar).size,
        )

        if fastchem:
            globals_map = {**base_globals, **element_fractions}
            result.append(
                RunSpecification(
                    run_id=f"run_{run_idx:05d}",
                    pressure_bar=np.asarray(pressure_bar, dtype=np.float64),
                    temperature_k=np.asarray(temperature_k, dtype=np.float64),
                    globals=globals_map,
                    metadata=dict(temperature_metadata),
                    elemental_abundances_frac=elemental_profile,
                    gravity_cm_s2=gravity_profile,
                )
            )
        else:
            kzz = sample_kzz_profile(pressure_bar, config=config, rng=rng)
            assert spectra is not None and spectrum_names is not None and science_presets is not None
            spectrum_name = spectrum_names[int(rng.integers(0, len(spectrum_names)))]
            template = spectra[spectrum_name]
            preset = science_presets[int(rng.integers(0, len(science_presets)))]
            spectrum = SpectrumRecord(
                name=f"{template.name}_run{run_idx:05d}",
                wavelength_nm=np.asarray(template.wavelength_nm, dtype=np.float64),
                flux_erg_cm2_s_nm=np.asarray(template.flux_erg_cm2_s_nm, dtype=np.float64),
                metadata={**template.metadata, "template_name": template.name},
            )
            globals_map = {
                **base_globals,
                **element_fractions,
                **_science_preset_conditioning_inputs(preset),
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
                        "science_preset_name": str(preset["name"]),
                    },
                    kzz_cm2_s=np.asarray(kzz, dtype=np.float64),
                    spectrum=spectrum,
                    elemental_abundances_frac=elemental_profile,
                    gravity_cm_s2=gravity_profile,
                )
            )
    return result
