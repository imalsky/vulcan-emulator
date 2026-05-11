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
scalars (the hydrogen-normalized elemental fractions ``He_H, C_H, O_H, N_H,
S_H``, and for VULCAN also surface gravity, planet radius, and irradiation
geometry) plus stellar-spectrum selection for VULCAN chemistry runs.
"""

from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import qmc

from ..constants import (
    ANALYTIC_SAMPLER_GRAVITY_CM_S2,
    ELEMENT_INPUT_ORDER,
    PUBLIC_PHYSICS_TOGGLES,
    SUPPORTED_ATM_BASES,
)
from ..utils.config import uses_fastchem
from ..utils.helpers import get_logger
from .roth_sampling import (
    RothFilterValue,
    RothProfile,
    _interpolate_profile,
    load_roth_profiles_native,
    roth_library_pressure_bounds,
)
from .spectrum import (
    SpectrumRecord,
    generate_blackbody_template,
    load_spectrum_records_from_glob,
    read_vulcan_spectrum_txt,
)

LOGGER = get_logger(__name__)

# Maximum number of rejection-resampling attempts for analytic profiles.
_MAX_PROFILE_ATTEMPTS = 100

# Module-level cache for Roth PT-library profiles at their NATIVE pressure
# grid. Keyed by (data_glob, filter_items, validation_items). Interpolation
# onto each run's pressure grid is done just-in-time in _sample_roth_profile,
# so variable per-run grids do not invalidate this cache — previously the key
# included a pressure fingerprint, causing a full library re-read on every
# run when grids were randomized.
_ROTH_PROFILE_CACHE: dict[tuple, list[RothProfile]] = {}
# Native pressure bounds (min_bar, max_bar) paired with each cached profile
# list. Computed once per cache entry so per-run grid clipping is O(1).
_ROTH_BOUNDS_CACHE: dict[tuple, tuple[float, float]] = {}
_ROTH_CACHE_LOCK = threading.Lock()


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


def _detect_available_cpus() -> int:
    """Return the number of CPUs available to this process."""
    for env_var in ("NCPUS", "SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK"):
        value = os.environ.get(env_var)
        if value is not None:
            try:
                n_cpus = int(value)
                if n_cpus >= 1:
                    return n_cpus
            except ValueError:
                pass
    return os.cpu_count() or 1


# Sampling work is dominated by GIL-bound Python on tiny per-run arrays
# (PCHIP interpolation, validation, dict construction, RNG draws). Past
# ~32 threads the GIL contention and context-switch overhead overwhelm any
# further parallelism, so we cap independent of `parallel_workers` (which
# is sized for the GIL-releasing fastchem subprocess phase).
_SAMPLING_WORKER_CAP = 32
_CORNER_COVERAGE_LABELS = (
    "carbon_rich_hot",
    "oxygen_rich_hot",
    "high_metallicity_hot",
    "near_unity_c_o_hot",
)
_NEAR_UNITY_C_TO_O_RANGE = (0.5, 2.0)


def _sampling_worker_count(config: dict[str, Any], total_specs: int) -> int:
    """Return the effective number of parallel sampling workers to launch."""
    configured = int(config["generation"]["parallel_workers"])
    if configured <= 0:
        configured = _detect_available_cpus()
    return max(1, min(configured, total_specs, _SAMPLING_WORKER_CAP))


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
) -> np.ndarray:
    """Build the per-level gravity array required by equilibrium datasets.

    Uses the fixed ``ANALYTIC_SAMPLER_GRAVITY_CM_S2`` constant; see the
    constants module for why this is not a sampling range (degenerate with
    ``delta`` in the Guillot form).

    Parameters
    ----------
    pressure_bar : np.ndarray
        Pressure grid whose shape defines the output profile length.

    Returns
    -------
    np.ndarray
        Column-constant gravity profile with shape matching ``pressure_bar``.
    """
    return np.full(
        np.asarray(pressure_bar).shape,
        ANALYTIC_SAMPLER_GRAVITY_CM_S2,
        dtype=np.float64,
    )


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


def _sample_column_pressure_grid(
    sampling_cfg: dict[str, Any],
    *,
    rng: np.random.Generator,
    native_p_bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    """Draw one per-run pressure grid from the configured (num_levels, p_top, p_bottom) ranges.

    num_levels is drawn uniformly (integer); p_top and p_bottom are each
    drawn uniformly in log10(bar) so the distribution is flat on the
    physically meaningful axis.  Returns a log-spaced 1-D pressure grid
    (bar), ordered from high pressure (bottom) to low pressure (top).

    When ``native_p_bounds`` is supplied the configured ``p_top`` and
    ``p_bottom`` ranges are intersected with the native PT-library
    ``[min_bar, max_bar]`` bounds so the resulting grid never extends
    outside the library's coverage. This prevents the PCHIP interpolator
    from falling into its constant-extrapolation branch for Roth profiles.
    """
    nl_lo, nl_hi = sampling_cfg["num_levels_range"]
    num_levels = int(rng.integers(int(nl_lo), int(nl_hi) + 1))

    top_lo, top_hi = (float(x) for x in sampling_cfg["pressure_top_bar_range"])
    bot_lo, bot_hi = (float(x) for x in sampling_cfg["pressure_bottom_bar_range"])
    if native_p_bounds is not None:
        native_min, native_max = native_p_bounds
        top_lo = max(top_lo, native_min)
        top_hi = max(top_hi, native_min)
        bot_lo = min(bot_lo, native_max)
        bot_hi = min(bot_hi, native_max)
        if top_lo > top_hi or bot_lo > bot_hi or top_hi >= bot_lo:
            raise ValueError(
                "Configured pressure_top/pressure_bottom ranges do not "
                f"intersect the native PT-library bounds {native_p_bounds!r}."
            )

    log_top = rng.uniform(math.log10(top_lo), math.log10(top_hi))
    log_bottom = rng.uniform(math.log10(bot_lo), math.log10(bot_hi))
    p_top = 10.0**log_top
    p_bottom = 10.0**log_bottom
    return sample_pressure_grid(
        num_levels=num_levels,
        pressure_top_bar=p_top,
        pressure_bottom_bar=p_bottom,
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

    return uniform_filter1d(modified, size=window, mode="nearest")


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
    if adjusted.size < 2:
        return adjusted
    log_p = np.log10(pressure_bar)
    log_t = np.log10(temperature_k)
    gradients = np.diff(log_t) / np.diff(log_p)
    crossings = np.flatnonzero(gradients > adiabatic_gradient)
    if crossings.size:
        first = int(crossings[0]) + 1
        p_top = pressure_bar[first - 1]
        t_top = adjusted[first - 1]
        adjusted[first:] = t_top * (pressure_bar[first:] / p_top) ** adiabatic_gradient
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


def _power_law_temperature(
    pressure_bar: np.ndarray,
    *,
    t0_k: float,
    alpha: float,
    p_ref_bar: float,
) -> np.ndarray:
    """Compute the pure power-law temperature profile ``T = T0 * (P/P_ref)**alpha``.

    Mirrors ExoJAX's ``art.powerlaw_temperature(T0, alpha)`` family — the
    parameterization driving NUTS in
    ``exojax_demo/06_classical_vs_emulator_retrieval.ipynb``. Adding this
    shape to training keeps the emulator in-distribution under that
    retrieval prior. See spec.md "Analytic profile shapes".
    """
    pressure_bar = np.asarray(pressure_bar, dtype=np.float64)
    return t0_k * (pressure_bar / p_ref_bar) ** alpha


def _sample_power_law_temperature_profile_record(
    pressure_bar: np.ndarray,
    *,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample a valid power-law PT profile via rejection sampling.

    Mirrors :func:`_sample_analytic_temperature_profile_record` for the
    Guillot path: draw ``(T0, alpha)`` uniformly from the configured ranges,
    evaluate the power-law form, and accept iff the profile is finite and
    inside the shared ``temperature_profiles.validation`` bounds. The
    Piette upper-atmosphere modification and the convective adjustment do
    not apply to power-law draws — the whole point is to expose the pure
    constant-log-slope shape ExoJAX retrieval consumers sample over.
    """
    sampler = config["temperature_profiles"]["analytic_sampler"]
    validation = config["temperature_profiles"]["validation"]
    t0_range = sampler["power_law_t0_range_k"]
    alpha_range = sampler["power_law_alpha_range"]
    p_ref_bar = float(sampler["power_law_p_ref_bar"])

    for attempt in range(_MAX_PROFILE_ATTEMPTS):
        t0_k = _sample_range_value(t0_range, rng=rng)
        alpha = _sample_range_value(alpha_range, rng=rng)

        profile_k = _power_law_temperature(
            pressure_bar,
            t0_k=t0_k,
            alpha=alpha,
            p_ref_bar=p_ref_bar,
        )

        if not np.all(np.isfinite(profile_k)):
            continue

        is_valid, reason = _validate_temperature_profile(
            profile_k, validation=validation,
        )
        if not is_valid:
            LOGGER.debug(
                "Power-law profile attempt %d rejected: %s", attempt + 1, reason,
            )
            continue

        metadata: dict[str, Any] = {
            "source": "analytic",
            "analytic_profile_type": "power_law",
            "analytic_t0_k": t0_k,
            "analytic_alpha": alpha,
            "analytic_p_ref_bar": p_ref_bar,
            "analytic_convective_adjustment_applied": False,
        }
        return np.asarray(profile_k, dtype=np.float64), metadata

    raise RuntimeError(
        f"Failed to generate a valid power-law profile after {_MAX_PROFILE_ATTEMPTS} attempts. "
        "Consider widening the validation bounds or tightening "
        "power_law_t0_range_k / power_law_alpha_range to keep T(P_bottom) "
        "below max_temperature_k."
    )


def _sample_analytic_temperature_profile_record(
    pressure_bar: np.ndarray,
    *,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample a valid analytic PT profile via rejection sampling.

    Each call coin-flips on ``power_law_probability`` (default 0.0). Tails
    delegate to :func:`_sample_power_law_temperature_profile_record`; the
    rest follow the Piette & Madhusudhan (2019) modified Guillot (2010)
    radiative-equilibrium path with optional convective adjustment, then
    validate. If a profile fails validation it is discarded and a fresh
    draw is attempted (up to ``_MAX_PROFILE_ATTEMPTS``).

    The Guillot profile equations are (Piette & Madhusudhan 2019, Eqs. 15-16):

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

    if float(rng.random()) < float(sampler.get("power_law_probability", 0.0)):
        return _sample_power_law_temperature_profile_record(
            pressure_bar, config=config, rng=rng,
        )

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

    Notes
    -----
    Prefers a prebaked sibling ``<dir>.bundle.npz`` file when present, since
    parsing the original ``.dat`` library is the dominant startup cost on a
    fresh process. The original glob is the fallback when no bundle exists.
    Bake one with ``python assets/prebake_pt_profiles.py``.
    """
    data_glob = str(roth_cfg["data_glob"])
    project_root = config.get("_project_root")
    glob_path = Path(data_glob)
    if not glob_path.is_absolute() and project_root is not None:
        glob_path = (project_root / glob_path).resolve()
    bundle_path = glob_path.parent.with_suffix(".bundle.npz")
    if bundle_path.is_file():
        return str(bundle_path)
    return str(glob_path)


def _load_configured_roth_profiles(
    *,
    config: dict[str, Any],
    roth_cfg: dict[str, Any],
) -> list[RothProfile]:
    """Load, validate, and cache configured PT-library profiles at native resolution.

    Profiles are cached at their native pressure grid, keyed on
    ``(data_glob, filter_items, validation_items)``. Interpolation onto a
    run's pressure grid is done by the caller on the single chosen profile
    (see ``_sample_roth_profile``), so variable per-run pressure grids do
    not cause cache misses.

    Temperature validation runs on native temperatures. PCHIP interpolation
    is shape-preserving, so temperature bounds satisfied at the native
    resolution remain satisfied on any in-range interpolated grid.
    """
    data_glob = _resolve_roth_data_glob(config, roth_cfg)
    filter_items: tuple[tuple[str, RothFilterValue], ...] = tuple(
        sorted(roth_cfg.get("filters", {}).items())
    )
    validation_items = tuple(
        sorted(config["temperature_profiles"]["validation"].items())
    )
    cache_key = (data_glob, filter_items, validation_items)
    profiles = _ROTH_PROFILE_CACHE.get(cache_key)
    if profiles is None:
        with _ROTH_CACHE_LOCK:
            profiles = _ROTH_PROFILE_CACHE.get(cache_key)
            if profiles is None:
                LOGGER.info("Loading PT-library profiles from %s", data_glob)
                loaded_profiles = load_roth_profiles_native(
                    data_glob,
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
                _ROTH_PROFILE_CACHE[cache_key] = profiles
                if profiles:
                    _ROTH_BOUNDS_CACHE[cache_key] = roth_library_pressure_bounds(profiles)
                LOGGER.info("Loaded %d PT-library profiles from %s", len(profiles), data_glob)
    if not profiles:
        raise FileNotFoundError(
            "roth_sampler.enabled=true but no temperature profiles matched "
            f"{data_glob!r} after applying filters and shared temperature validation."
        )
    return profiles


def _prime_sampling_caches(plan: SamplingPlan) -> None:
    """Warm expensive shared sampling caches before threaded sampling begins."""
    if plan.roth_cfg.get("enabled", False):
        _load_configured_roth_profiles(config=plan.config, roth_cfg=plan.roth_cfg)


def _roth_library_native_bounds(
    *,
    config: dict[str, Any],
    roth_cfg: dict[str, Any],
) -> tuple[float, float]:
    """Return cached ``(min_bar, max_bar)`` bounds for the configured PT library.

    Lazily loads and caches the Roth profile library on first call (via
    :func:`_load_configured_roth_profiles`) so the cache key matches the
    profile cache exactly.
    """
    _load_configured_roth_profiles(config=config, roth_cfg=roth_cfg)
    data_glob = _resolve_roth_data_glob(config, roth_cfg)
    filter_items: tuple[tuple[str, RothFilterValue], ...] = tuple(
        sorted(roth_cfg.get("filters", {}).items())
    )
    validation_items = tuple(
        sorted(config["temperature_profiles"]["validation"].items())
    )
    return _ROTH_BOUNDS_CACHE[(data_glob, filter_items, validation_items)]


def _decide_temperature_profile_source(
    roth_cfg: dict[str, Any],
    *,
    rng: np.random.Generator,
) -> str:
    """Return ``"roth"`` or ``"analytic"`` for this run, per config + RNG.

    Consumes one ``rng.random()`` draw only when the sampler is in ``mixed``
    mode; otherwise deterministic given ``roth_cfg``. Deciding the source
    before the pressure grid is drawn lets the caller clip the grid to the
    PT library's native range when Roth is chosen.
    """
    if not roth_cfg.get("enabled", False):
        return "analytic"
    if roth_cfg.get("source_mode", "roth") == "mixed":
        analytic_probability = float(roth_cfg["analytic_probability"])
        return "analytic" if float(rng.random()) < analytic_probability else "roth"
    return "roth"


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
    profiles = _load_configured_roth_profiles(config=config, roth_cfg=roth_cfg)
    native = _choose_roth_profile(profiles, rng=rng)
    interpolated_temperature_k = _interpolate_profile(
        pressure_bar,
        native.pressure_bar,
        native.temperature_k,
    )
    return RothProfile(
        pressure_bar=np.asarray(pressure_bar, dtype=np.float64),
        temperature_k=interpolated_temperature_k,
        metadata=dict(native.metadata),
    )


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
    source: str | None = None,
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
    source : str, optional
        Pre-decided source (``"roth"`` or ``"analytic"``). If ``None``, the
        source is decided here using one ``rng`` draw (mixed mode). Callers
        that also need to clip the pressure grid to native bounds should
        decide upfront via :func:`_decide_temperature_profile_source` and
        pass the choice in.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        Temperature profile in Kelvin with shape ``(nz,)`` plus a flat
        metadata dictionary describing the sampled source and parameters.
    """
    roth_cfg = config["roth_sampler"]
    chosen_source = source if source is not None else _decide_temperature_profile_source(roth_cfg, rng=rng)

    if chosen_source == "roth":
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
    """Sample a depth-constant eddy-diffusion profile from the configured range.

    Parameters
    ----------
    pressure_bar : np.ndarray
        Pressure grid whose shape defines the output profile length.
    config : dict[str, Any]
        Validated config containing ``sampling.kzz_range_cm2_s`` (a pair)
        and an optional ``sampling.scales.kzz_cm2_s`` entry.
    rng : np.random.Generator
        Random generator used to draw the per-run constant Kzz value.

    Returns
    -------
    np.ndarray
        Constant ``Kzz`` profile with shape matching ``pressure_bar``.
    """
    lo, hi = (float(x) for x in config["sampling"]["kzz_range_cm2_s"])
    scale = _sampling_scale(config, "kzz_cm2_s", default="log")
    kzz_value = float(np.clip(
        _scale_unit_interval(float(rng.random()), lo, hi, scale),
        1.0,
        None,
    ))
    return np.full(pressure_bar.shape, kzz_value, dtype=np.float64)


def _latin_hypercube_unit_samples(
    *,
    num_samples: int,
    num_dimensions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a Latin-hypercube design matrix in the unit hypercube [0, 1]^d.

    Thin wrapper over :class:`scipy.stats.qmc.LatinHypercube`: each dimension
    is stratified into ``num_samples`` equal-probability bins with exactly one
    sample per bin. The supplied ``rng`` threads determinism through scipy's
    QMC engine.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1.")
    sampler = qmc.LatinHypercube(d=num_dimensions, seed=rng)
    return sampler.random(num_samples)


def _scale_unit_interval(
    value: float,
    lower: float,
    upper: float,
    scale: str = "linear",
) -> float:
    """Map a unit-interval coordinate onto a physical parameter range.

    Parameters
    ----------
    value : float
        Sample in the unit interval, typically from Latin-hypercube sampling.
    lower : float
        Physical lower bound for the parameter.
    upper : float
        Physical upper bound for the parameter.
    scale : {"linear", "log"}
        If ``"log"``, interpolate log10-uniformly between ``lower`` and
        ``upper`` (both must be strictly positive). Otherwise interpolate
        linearly.

    Returns
    -------
    float
        Rescaled value in ``[lower, upper]``.
    """
    if scale == "log":
        if lower <= 0.0 or upper <= 0.0:
            raise ValueError(
                f"log-scale sampling requires positive bounds; got [{lower}, {upper}]"
            )
        log_lower = math.log10(lower)
        log_upper = math.log10(upper)
        return float(10.0 ** (log_lower + value * (log_upper - log_lower)))
    return float(lower + value * (upper - lower))


def _unit_interval_from_scaled_value(
    value: float,
    lower: float,
    upper: float,
    scale: str = "linear",
) -> float:
    """Map a physical value back to the unit interval for one sampled range."""
    clipped = float(np.clip(value, lower, upper))
    if scale == "log":
        if lower <= 0.0 or upper <= 0.0:
            raise ValueError(
                f"log-scale sampling requires positive bounds; got [{lower}, {upper}]"
            )
        log_lower = math.log10(lower)
        log_upper = math.log10(upper)
        return float((math.log10(clipped) - log_lower) / (log_upper - log_lower))
    return float((clipped - lower) / (upper - lower))


def _unit_band(value: float, lower: float, upper: float) -> float:
    """Remap a unit sample into a closed unit-interval sub-band."""
    return float(lower + value * (upper - lower))


def _sampling_scale(config: dict[str, Any], key: str, default: str = "linear") -> str:
    """Return the sampling scale (``"linear"`` or ``"log"``) for a given key.

    Sampling scales live under ``sampling.scales`` in the config. Missing
    entries fall back to linear so existing configs keep their old behavior.
    """
    scales = config["sampling"].get("scales") or {}
    mode = str(scales.get(key, default)).lower()
    if mode not in ("linear", "log"):
        raise ValueError(
            f"sampling.scales.{key} must be 'linear' or 'log', got {mode!r}"
        )
    return mode


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
        return generate_blackbody_template(
            wavelength_min_nm=float(spectrum_cfg["wavelength_min_nm"]),
            wavelength_max_nm=float(spectrum_cfg["wavelength_max_nm"]),
            teff_k=float(spectrum_cfg["teff_k"]),
            radius_rsun=float(spectrum_cfg["radius_rsun"]),
            semi_major_axis_au=float(spectrum_cfg["semi_major_axis_au"]),
            name=str(spectrum_cfg["template_name"]),
        )
    template_path = (project_root / str(template_file)).resolve()
    if not template_path.exists():
        raise FileNotFoundError(
            f"Configured stellar_spectrum.template_file does not exist: {template_path}"
        )
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


@dataclass(frozen=True)
class SamplingPlan:
    """Pre-built sampling state reused across slices of a generation run.

    The LHS design matrix and per-run seed sequence are built once up front so
    sampling can stream in chunks (``sample_run_specifications_slice``) without
    reshuffling stratification or losing seed determinism.
    """

    config: dict[str, Any]
    fastchem: bool
    total_runs: int
    design: np.ndarray
    child_seeds: list[np.random.SeedSequence]
    corner_coverage_cfg: dict[str, Any] | None
    corner_coverage_labels: list[str | None]
    sampling_cfg: dict[str, Any]
    roth_cfg: dict[str, Any]
    # Pre-resolved fraction ranges/scales, shared across all runs.
    he_frac_range: tuple[float, float]
    he_frac_scale: str
    c_frac_range: tuple[float, float]
    c_frac_scale: str
    o_frac_range: tuple[float, float]
    o_frac_scale: str
    n_frac_range: tuple[float, float]
    n_frac_scale: str
    s_frac_range: tuple[float, float]
    s_frac_scale: str
    # VULCAN-only fields; None for FastChem plans.
    gravity_range: tuple[float, float] | None
    gravity_scale: str | None
    planet_radius_range: tuple[float, float] | None
    planet_radius_scale: str | None
    stellar_radius_range: tuple[float, float] | None
    stellar_radius_scale: str | None
    semi_major_axis_range: tuple[float, float] | None
    semi_major_axis_scale: str | None
    zenith_range: tuple[float, float] | None
    zenith_scale: str | None
    diurnal_range: tuple[float, float] | None
    diurnal_scale: str | None
    spectra: dict[str, SpectrumRecord] | None
    spectrum_names: list[str] | None
    science_presets: list[dict[str, Any]] | None


def _build_corner_coverage_labels(
    *,
    total_runs: int,
    corner_cfg: dict[str, Any] | None,
    rng: np.random.Generator,
) -> list[str | None]:
    """Assign deterministic corner-coverage labels to selected run indices."""
    labels: list[str | None] = [None] * int(total_runs)
    if not corner_cfg or not bool(corner_cfg.get("enabled", False)):
        return labels
    target_count = int(round(float(corner_cfg["fraction"]) * int(total_runs)))
    target_count = max(0, min(int(total_runs), target_count))
    if target_count == 0:
        return labels

    order = rng.permutation(int(total_runs))
    base_count, remainder = divmod(target_count, len(_CORNER_COVERAGE_LABELS))
    counts = [
        base_count + (1 if i < remainder else 0)
        for i in range(len(_CORNER_COVERAGE_LABELS))
    ]
    offset = 0
    for label, count in zip(_CORNER_COVERAGE_LABELS, counts):
        for run_idx in order[offset: offset + count]:
            labels[int(run_idx)] = label
        offset += count
    return labels


def _apply_corner_abundance_units(
    plan: SamplingPlan,
    *,
    label: str | None,
    c_unit: float,
    o_unit: float,
) -> tuple[float, float]:
    """Remap C/O coordinates into the configured corner-coverage stratum."""
    if label is None or plan.corner_coverage_cfg is None:
        return c_unit, o_unit

    q = float(plan.corner_coverage_cfg["abundance_quantile_width"])
    if label == "carbon_rich_hot":
        return _unit_band(c_unit, 1.0 - q, 1.0), _unit_band(o_unit, 0.0, q)
    if label == "oxygen_rich_hot":
        return _unit_band(c_unit, 0.0, q), _unit_band(o_unit, 1.0 - q, 1.0)
    if label == "high_metallicity_hot":
        return _unit_band(c_unit, 1.0 - q, 1.0), _unit_band(o_unit, 1.0 - q, 1.0)
    if label != "near_unity_c_o_hot":
        raise ValueError(f"Unknown corner coverage label {label!r}.")

    o_unit = _unit_band(o_unit, q, 1.0 - q)
    o_frac = _scale_unit_interval(o_unit, *plan.o_frac_range, plan.o_frac_scale)
    c_min, c_max = plan.c_frac_range
    ratio_lo = max(_NEAR_UNITY_C_TO_O_RANGE[0], c_min / o_frac)
    ratio_hi = min(_NEAR_UNITY_C_TO_O_RANGE[1], c_max / o_frac)
    if ratio_lo > ratio_hi:
        ratio_lo = c_min / o_frac
        ratio_hi = c_max / o_frac
    ratio = _scale_unit_interval(c_unit, ratio_lo, ratio_hi, "log")
    c_frac = o_frac * ratio
    c_unit = _unit_interval_from_scaled_value(
        c_frac, *plan.c_frac_range, plan.c_frac_scale,
    )
    return c_unit, o_unit


def _corner_temperature_satisfied(
    profile_k: np.ndarray,
    corner_cfg: dict[str, Any],
) -> bool:
    """Return whether one PT profile reaches the configured corner threshold."""
    profile = np.asarray(profile_k, dtype=np.float64)
    return (
        float(np.max(profile)) >= float(corner_cfg["hot_tmax_k"])
        or float(np.ptp(profile)) >= float(corner_cfg["large_trange_k"])
    )


def _sample_corner_temperature_profile_record(
    pressure_bar: np.ndarray,
    *,
    config: dict[str, Any],
    rng: np.random.Generator,
    source: str,
    corner_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample a valid PT profile, preferring hot or large-range corner profiles."""
    max_attempts = int(corner_cfg["max_profile_resample_attempts"])
    last_profile: np.ndarray | None = None
    last_metadata: dict[str, Any] | None = None
    for attempt in range(1, max_attempts + 1):
        profile, metadata = _sample_temperature_profile_record(
            pressure_bar, config=config, rng=rng, source=source,
        )
        if _corner_temperature_satisfied(profile, corner_cfg):
            return profile, {
                **metadata,
                "corner_coverage_profile_attempts": attempt,
                "corner_coverage_profile_satisfied": True,
            }
        last_profile = profile
        last_metadata = metadata

    assert last_profile is not None and last_metadata is not None
    return last_profile, {
        **last_metadata,
        "corner_coverage_profile_attempts": max_attempts,
        "corner_coverage_profile_satisfied": False,
    }


def build_sampling_plan(
    *,
    config: dict[str, Any],
    project_root: Path,
    num_runs: int | None = None,
    seed: int | None = None,
) -> SamplingPlan:
    """Construct a ``SamplingPlan`` covering every run for this generation.

    Builds the Latin-hypercube design matrix and per-run seed sequence once
    so downstream callers can pull specifications out slice-by-slice (see
    :func:`sample_run_specifications_slice`) while preserving seed-level
    determinism against the single-call :func:`sample_run_specifications`.
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

    spectra: dict[str, SpectrumRecord] | None = None
    spectrum_names: list[str] | None = None
    science_presets: list[dict[str, Any]] | None = None
    if not fastchem:
        spectra = load_default_spectra(project_root=project_root, config=config)
        if not spectra:
            raise RuntimeError("No stellar spectra available for sampling.")
        spectrum_names = sorted(spectra.keys())
        science_presets = list(config["science_presets"])

    sampling_cfg = config["sampling"]
    he_frac_range = (float(sampling_cfg["he_frac_range"][0]), float(sampling_cfg["he_frac_range"][1]))
    c_frac_range = (float(sampling_cfg["c_frac_range"][0]), float(sampling_cfg["c_frac_range"][1]))
    o_frac_range = (float(sampling_cfg["o_frac_range"][0]), float(sampling_cfg["o_frac_range"][1]))
    n_frac_range = (float(sampling_cfg["n_frac_range"][0]), float(sampling_cfg["n_frac_range"][1]))
    s_frac_range = (float(sampling_cfg["s_frac_range"][0]), float(sampling_cfg["s_frac_range"][1]))

    gravity_range = gravity_scale = None
    planet_radius_range = planet_radius_scale = None
    stellar_radius_range = stellar_radius_scale = None
    semi_major_axis_range = semi_major_axis_scale = None
    zenith_range = zenith_scale = None
    diurnal_range = diurnal_scale = None
    if not fastchem:
        gravity_range = (
            float(sampling_cfg["gravity_range_cm_s2"][0]),
            float(sampling_cfg["gravity_range_cm_s2"][1]),
        )
        gravity_scale = _sampling_scale(config, "gravity_cm_s2")
        planet_radius_range = (
            float(sampling_cfg["planet_radius_range_cm"][0]),
            float(sampling_cfg["planet_radius_range_cm"][1]),
        )
        planet_radius_scale = _sampling_scale(config, "planet_radius_cm")
        stellar_radius_range = (
            float(sampling_cfg["stellar_radius_range_rsun"][0]),
            float(sampling_cfg["stellar_radius_range_rsun"][1]),
        )
        stellar_radius_scale = _sampling_scale(config, "r_star_rsun")
        semi_major_axis_range = (
            float(sampling_cfg["semi_major_axis_range_au"][0]),
            float(sampling_cfg["semi_major_axis_range_au"][1]),
        )
        semi_major_axis_scale = _sampling_scale(config, "semi_major_axis_au")
        zenith_range = (
            float(sampling_cfg["zenith_angle_range_deg"][0]),
            float(sampling_cfg["zenith_angle_range_deg"][1]),
        )
        zenith_scale = _sampling_scale(config, "zenith_angle_deg")
        diurnal_range = (
            float(sampling_cfg["diurnal_factor_range"][0]),
            float(sampling_cfg["diurnal_factor_range"][1]),
        )
        diurnal_scale = _sampling_scale(config, "diurnal_factor")

    # Per-run independent RNG streams so the loop parallelizes without races.
    child_seeds = np.random.SeedSequence(
        int(config["generation"]["seed"] if seed is None else seed)
    ).spawn(total_runs)

    corner_coverage_cfg = sampling_cfg.get("corner_coverage")
    corner_coverage_labels = (
        _build_corner_coverage_labels(
            total_runs=total_runs, corner_cfg=corner_coverage_cfg, rng=rng,
        )
        if fastchem
        else [None] * total_runs
    )

    roth_cfg = config["roth_sampler"]

    return SamplingPlan(
        config=config,
        fastchem=fastchem,
        total_runs=total_runs,
        design=design,
        child_seeds=list(child_seeds),
        corner_coverage_cfg=corner_coverage_cfg if fastchem else None,
        corner_coverage_labels=corner_coverage_labels,
        sampling_cfg=sampling_cfg,
        roth_cfg=roth_cfg,
        he_frac_range=he_frac_range,
        he_frac_scale=_sampling_scale(config, "he_frac"),
        c_frac_range=c_frac_range,
        c_frac_scale=_sampling_scale(config, "c_frac"),
        o_frac_range=o_frac_range,
        o_frac_scale=_sampling_scale(config, "o_frac"),
        n_frac_range=n_frac_range,
        n_frac_scale=_sampling_scale(config, "n_frac"),
        s_frac_range=s_frac_range,
        s_frac_scale=_sampling_scale(config, "s_frac"),
        gravity_range=gravity_range,
        gravity_scale=gravity_scale,
        planet_radius_range=planet_radius_range,
        planet_radius_scale=planet_radius_scale,
        stellar_radius_range=stellar_radius_range,
        stellar_radius_scale=stellar_radius_scale,
        semi_major_axis_range=semi_major_axis_range,
        semi_major_axis_scale=semi_major_axis_scale,
        zenith_range=zenith_range,
        zenith_scale=zenith_scale,
        diurnal_range=diurnal_range,
        diurnal_scale=diurnal_scale,
        spectra=spectra,
        spectrum_names=spectrum_names,
        science_presets=science_presets,
    )


def _sample_one_from_plan(plan: SamplingPlan, run_idx: int) -> RunSpecification:
    """Sample a single ``RunSpecification`` for ``run_idx`` against ``plan``.

    Kept deliberately symmetric with the historical inline ``_sample_one`` so
    seed-determinism is byte-identical to the pre-refactor code path.
    """
    design = plan.design
    config = plan.config
    sampling_cfg = plan.sampling_cfg
    roth_cfg = plan.roth_cfg
    fastchem = plan.fastchem
    corner_label = plan.corner_coverage_labels[run_idx]

    per_rng = np.random.default_rng(plan.child_seeds[run_idx])
    # Decide profile source first so we can clip the pressure grid to the
    # PT-library's native bounds when Roth is chosen — otherwise PCHIP
    # would fall into its constant-extrapolation branch for target grids
    # that extend outside the native range, producing flat plateaus and
    # abrupt gradient kinks at the boundary.
    profile_source = _decide_temperature_profile_source(roth_cfg, rng=per_rng)
    native_p_bounds: tuple[float, float] | None = None
    if profile_source == "roth":
        native_p_bounds = _roth_library_native_bounds(
            config=config, roth_cfg=roth_cfg,
        )
    pressure_bar = _sample_column_pressure_grid(
        sampling_cfg, rng=per_rng, native_p_bounds=native_p_bounds,
    )
    if fastchem:
        c_unit, o_unit = _apply_corner_abundance_units(
            plan,
            label=corner_label,
            c_unit=float(design[run_idx, 1]),
            o_unit=float(design[run_idx, 2]),
        )
        he_frac = _scale_unit_interval(
            design[run_idx, 0], *plan.he_frac_range, plan.he_frac_scale,
        )
        c_frac = _scale_unit_interval(
            c_unit, *plan.c_frac_range, plan.c_frac_scale,
        )
        o_frac = _scale_unit_interval(
            o_unit, *plan.o_frac_range, plan.o_frac_scale,
        )
        n_frac = _scale_unit_interval(
            design[run_idx, 3], *plan.n_frac_range, plan.n_frac_scale,
        )
        s_frac = _scale_unit_interval(
            design[run_idx, 4], *plan.s_frac_range, plan.s_frac_scale,
        )
    else:
        assert plan.gravity_range is not None and plan.gravity_scale is not None
        assert plan.planet_radius_range is not None and plan.planet_radius_scale is not None
        assert plan.stellar_radius_range is not None and plan.stellar_radius_scale is not None
        assert plan.semi_major_axis_range is not None and plan.semi_major_axis_scale is not None
        assert plan.zenith_range is not None and plan.zenith_scale is not None
        assert plan.diurnal_range is not None and plan.diurnal_scale is not None
        gravity = _scale_unit_interval(
            design[run_idx, 0], *plan.gravity_range, plan.gravity_scale,
        )
        planet_radius_cm = _scale_unit_interval(
            design[run_idx, 1], *plan.planet_radius_range, plan.planet_radius_scale,
        )
        stellar_radius_rsun = _scale_unit_interval(
            design[run_idx, 2], *plan.stellar_radius_range, plan.stellar_radius_scale,
        )
        semi_major_axis_au = _scale_unit_interval(
            design[run_idx, 3], *plan.semi_major_axis_range, plan.semi_major_axis_scale,
        )
        zenith_angle_deg = _scale_unit_interval(
            design[run_idx, 4], *plan.zenith_range, plan.zenith_scale,
        )
        diurnal_factor = _scale_unit_interval(
            design[run_idx, 5], *plan.diurnal_range, plan.diurnal_scale,
        )
        he_frac = _scale_unit_interval(
            design[run_idx, 6], *plan.he_frac_range, plan.he_frac_scale,
        )
        c_frac = _scale_unit_interval(
            design[run_idx, 7], *plan.c_frac_range, plan.c_frac_scale,
        )
        o_frac = _scale_unit_interval(
            design[run_idx, 8], *plan.o_frac_range, plan.o_frac_scale,
        )
        n_frac = _scale_unit_interval(
            design[run_idx, 9], *plan.n_frac_range, plan.n_frac_scale,
        )
        s_frac = _scale_unit_interval(
            design[run_idx, 10], *plan.s_frac_range, plan.s_frac_scale,
        )

    if fastchem and corner_label is not None and plan.corner_coverage_cfg is not None:
        temperature_k, temperature_metadata = _sample_corner_temperature_profile_record(
            pressure_bar,
            config=config,
            rng=per_rng,
            source=profile_source,
            corner_cfg=plan.corner_coverage_cfg,
        )
        temperature_metadata = {
            **temperature_metadata,
            "corner_coverage_label": corner_label,
        }
    else:
        temperature_k, temperature_metadata = _sample_temperature_profile_record(
            pressure_bar,
            config=config,
            rng=per_rng,
            source=profile_source,
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
            pressure_bar=pressure_bar,
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
            pressure_bar.shape,
            float(gravity),
            dtype=np.float64,
        )
    # SeedSequence.generate_state derives a reproducible uint32 from the
    # per-run seed material (parent seed + spawn path), giving every run a
    # stable numeric seed that survives retries with different chunk seeding.
    rng_seed = int(plan.child_seeds[run_idx].generate_state(1, dtype=np.uint32)[0])
    if fastchem:
        globals_map = {**base_globals, **element_fractions}
        return RunSpecification(
            run_id=f"run_{run_idx:05d}",
            pressure_bar=pressure_bar,
            temperature_k=temperature_k,
            globals=globals_map,
            metadata={**temperature_metadata, "rng_seed": rng_seed},
            elemental_abundances_frac=None,
            gravity_cm_s2=gravity_profile,
        )
    kzz = sample_kzz_profile(pressure_bar, config=config, rng=per_rng)
    assert (
        plan.spectra is not None
        and plan.spectrum_names is not None
        and plan.science_presets is not None
    )
    spectrum_names = plan.spectrum_names
    spectrum_name = spectrum_names[int(per_rng.integers(0, len(spectrum_names)))]
    template = plan.spectra[spectrum_name]
    preset = plan.science_presets[int(per_rng.integers(0, len(plan.science_presets)))]
    spectrum = SpectrumRecord(
        name=f"{template.name}_run{run_idx:05d}",
        wavelength_nm=template.wavelength_nm,
        flux_erg_cm2_s_nm=template.flux_erg_cm2_s_nm,
        metadata={**template.metadata, "template_name": template.name},
    )
    globals_map = {
        **base_globals,
        **element_fractions,
        **_science_preset_conditioning_inputs(preset),
    }
    return RunSpecification(
        run_id=f"run_{run_idx:05d}",
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        globals=globals_map,
        metadata={
            **temperature_metadata,
            "spectrum_name": spectrum.name,
            "science_preset_name": str(preset["name"]),
            "rng_seed": rng_seed,
        },
        kzz_cm2_s=kzz,
        spectrum=spectrum,
        elemental_abundances_frac=None,
        gravity_cm_s2=gravity_profile,
    )


def sample_run_specifications_slice(
    plan: SamplingPlan,
    *,
    start: int,
    end: int,
) -> list[RunSpecification]:
    """Sample the half-open ``[start, end)`` slice of ``plan`` in parallel.

    Output is seed-identical to ``sample_run_specifications`` restricted to
    the same index range; streaming a generation run chunk-by-chunk therefore
    produces byte-identical specs to the historical one-shot call.
    """
    if start < 0 or end > plan.total_runs or start > end:
        raise ValueError(
            f"Invalid slice [{start}, {end}) for plan with {plan.total_runs} runs."
        )
    count = end - start
    if count == 0:
        return []
    _prime_sampling_caches(plan)
    indices = range(start, end)
    max_workers = _sampling_worker_count(plan.config, count)
    if count == 1 or max_workers == 1:
        specs = [_sample_one_from_plan(plan, i) for i in indices]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            specs = list(pool.map(lambda i: _sample_one_from_plan(plan, i), indices))
    LOGGER.info(
        "Sampled run specifications %d..%d / %d",
        start,
        end,
        plan.total_runs,
    )
    return specs


def sample_run_specifications(
    *,
    config: dict[str, Any],
    project_root: Path,
    num_runs: int | None = None,
    seed: int | None = None,
) -> list[RunSpecification]:
    """Sample the full set of atmospheric configurations used to generate raw runs.

    Uses Latin-hypercube sampling (LHC) to stratify the global conditioning
    scalars (the hydrogen-normalized elemental fractions ``He_H, C_H, O_H,
    N_H, S_H``, and for VULCAN also surface gravity, planet radius, stellar
    radius, orbital separation, zenith angle, and diurnal factor) over the
    configured ranges. For each run, a temperature profile is independently
    drawn from the configured source (analytic, PT-library, or mixed).

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
    plan = build_sampling_plan(
        config=config,
        project_root=project_root,
        num_runs=num_runs,
        seed=seed,
    )
    return sample_run_specifications_slice(plan, start=0, end=plan.total_runs)
