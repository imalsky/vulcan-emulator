"""Sampling utilities for TP, Kzz, abundance, and run specifications.

Generates the physical parameter space for VULCAN training data:

- **TP profiles**: Modified Line-2013 parameterization with configurable
  opacity, gamma factors, internal/irradiation temperatures, and optional
  convective adjustment.
- **Kzz profiles**: Power-law family ``Kzz(p) = Kzz_1bar * p^(-beta)``
  with configurable floor and cap.
- **Abundances**: Metallicity (log-uniform) + C/O ratio sampling, with
  elemental abundances derived from scaled solar values.
- **Gravity**: Uniform or fixed surface gravity.

All sampling distributions and ranges are config-driven.  Rejection sampling
handles TP profiles that violate physical temperature bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import expn


class SamplingError(ValueError):
    """Raised when sampled physics inputs violate constraints."""


@dataclass(frozen=True)
class RunSpec:
    """Single VULCAN run specification and sampled parameters."""

    run_id: int
    pressure_bar: np.ndarray
    temperature_k: np.ndarray
    kzz_cm2_s: np.ndarray
    gravity_cm_s2: float
    metallicity_log10: float
    c_to_o: float
    abundances: dict[str, float]
    tp_params: dict[str, float]
    kzz_params: dict[str, float]


def _sample_distribution(spec: dict[str, Any], rng: np.random.Generator, name: str) -> float:
    """Sample one scalar from an explicit uniform or normal specification."""
    dist = str(spec["distribution"]).lower()
    if dist == "uniform":
        low = float(spec["min"])
        high = float(spec["max"])
        if high <= low:
            raise SamplingError(f"Invalid uniform range for {name}: min={low}, max={high}")
        return float(rng.uniform(low, high))

    if dist == "normal":
        mean = float(spec["mean"])
        std = float(spec["std"])
        if std <= 0:
            raise SamplingError(f"Invalid normal std for {name}: {std}")
        value = float(rng.normal(mean, std))
        if "min" in spec:
            value = max(value, float(spec["min"]))
        if "max" in spec:
            value = min(value, float(spec["max"]))
        return value

    raise SamplingError(f"Unsupported distribution for {name}: {dist}")


def build_pressure_grid(tp_cfg: dict[str, Any]) -> np.ndarray:
    """Construct log-spaced pressure grid in bar, descending bottom->top."""
    grid = tp_cfg["pressure_grid"]
    nz = int(grid["nz"])
    p_top = float(grid["p_top_bar"])
    p_bottom = float(grid["p_bottom_bar"])
    if nz <= 1 or p_top <= 0 or p_bottom <= p_top:
        raise SamplingError("Invalid pressure grid settings.")
    return np.logspace(np.log10(p_bottom), np.log10(p_top), nz, dtype=np.float64)


def _xi_gamma(gamma: float, tau: np.ndarray) -> np.ndarray:
    """Evaluate the Line-2013 irradiation integral for one gamma channel.

    Computes xi(gamma, tau) = 2/3 + 2/(3*gamma) * [1 + (gamma*tau/2 - 1)*exp(-gamma*tau)]
                              + 2*gamma/3 * (1 - tau^2/2) * E_2(gamma*tau)

    where E_2 is the second-order exponential integral.  This describes how
    stellar irradiation penetrates the atmosphere as a function of optical
    depth, and is used in the two-stream temperature profile calculation.

    Args:
        gamma: Ratio of visible to thermal opacity (dimensionless).
        tau: Thermal optical depth profile (array over pressure levels).

    Returns:
        The irradiation function xi evaluated at each optical depth.
    """
    x = gamma * tau
    x = np.maximum(x, 1e-12)
    term1 = 2.0 / 3.0
    term2 = (2.0 / (3.0 * gamma)) * (1.0 + ((x / 2.0) - 1.0) * np.exp(-x))
    term3 = (2.0 * gamma / 3.0) * (1.0 - 0.5 * tau * tau) * expn(2, x)
    return term1 + term2 + term3


def generate_tp_profile(
    pressure_bar: np.ndarray,
    tp_cfg: dict[str, Any],
    gravity_cm_s2: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """Generate one temperature profile from modified Line-2013 parameterization.

    The temperature is computed from the two-stream radiative transfer solution:

        T^4 = (3/4)*T_int^4*(2/3 + tau)
              + (3/4)*T_irr^4*[(1-alpha)*xi(gamma1, tau) + alpha*xi(gamma2, tau)]

    where tau is the thermal optical depth profile derived from
    ``kappa_IR * (p/p0)^beta``, and ``xi`` is the irradiation integral.

    An optional convective adjustment enforces an adiabatic lapse rate in
    the deep atmosphere (applied probabilistically based on config).

    Args:
        pressure_bar: Log-spaced pressure grid in bar (descending, bottom-to-top).
        tp_cfg: TP sampler configuration with distribution specs.
        gravity_cm_s2: Surface gravity in cm/s^2.
        rng: NumPy random generator for reproducible sampling.

    Returns:
        Tuple of (temperature_k array, parameter dict for provenance).

    Raises:
        SamplingError: If the sampled profile violates temperature bounds or
            contains non-finite values.
    """
    params = {
        "log10_kappa_ir": _sample_distribution(tp_cfg["log10_kappa_ir"], rng, "log10_kappa_ir"),
        "kappa_pressure_power_exponent": _sample_distribution(
            tp_cfg["kappa_pressure_power_exponent"], rng, "kappa_pressure_power_exponent"
        ),
        "log10_gamma1": _sample_distribution(tp_cfg["log10_gamma1"], rng, "log10_gamma1"),
        "log10_gamma2": _sample_distribution(tp_cfg["log10_gamma2"], rng, "log10_gamma2"),
        "alpha_partition": _sample_distribution(tp_cfg["alpha_partition"], rng, "alpha_partition"),
        "t_int": _sample_distribution(tp_cfg["t_int"], rng, "t_int"),
        "t_irr": _sample_distribution(tp_cfg["t_irr"], rng, "t_irr"),
        "temperature_shift": _sample_distribution(
            tp_cfg["temperature_shift"], rng, "temperature_shift"
        ),
    }

    kappa_ir = 10.0 ** params["log10_kappa_ir"]
    gamma1 = 10.0 ** params["log10_gamma1"]
    gamma2 = 10.0 ** params["log10_gamma2"]
    alpha = params["alpha_partition"]

    p0 = 1.0
    tau = kappa_ir * np.power(
        np.maximum(pressure_bar / p0, 1e-30), params["kappa_pressure_power_exponent"]
    )
    xi1 = _xi_gamma(gamma1, tau)
    xi2 = _xi_gamma(gamma2, tau)

    tint4 = params["t_int"] ** 4
    tirr4 = params["t_irr"] ** 4

    t4 = (
        (3.0 * tint4 / 4.0) * (2.0 / 3.0 + tau)
        + (3.0 * tirr4 / 4.0) * (1.0 - alpha) * xi1
        + (3.0 * tirr4 / 4.0) * alpha * xi2
    )
    t4 = np.maximum(t4, 1e-12)
    temperature = np.power(t4, 0.25) + params["temperature_shift"]

    # Optional simple convective adjustment in pressure-increasing direction.
    prob = float(tp_cfg["convective_adjustment_probability"])
    if rng.uniform(0.0, 1.0) < prob:
        nabla_ad = float(tp_cfg["adiabatic_gradient"])
        p_inc = pressure_bar[::-1]
        t_inc = temperature[::-1].copy()
        for idx in range(1, t_inc.size):
            p_ratio = p_inc[idx] / p_inc[idx - 1]
            t_max = t_inc[idx - 1] * np.power(p_ratio, nabla_ad)
            if t_inc[idx] > t_max:
                t_inc[idx] = t_max
        temperature = t_inc[::-1]
        params["convective_adjustment_applied"] = 1.0
    else:
        params["convective_adjustment_applied"] = 0.0

    if not np.all(np.isfinite(temperature)):
        raise SamplingError("Non-finite temperature profile sampled.")

    t_min = float(tp_cfg["temperature_limits_k"]["min"])
    t_max = float(tp_cfg["temperature_limits_k"]["max"])
    if np.any(temperature < t_min) or np.any(temperature > t_max):
        raise SamplingError(f"Sampled temperature out of bounds [{t_min}, {t_max}] K.")

    params["gravity_cm_s2"] = float(gravity_cm_s2)
    return temperature.astype(np.float64), params


def sample_kzz_profile(
    pressure_bar: np.ndarray,
    kzz_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """Sample Kzz power-law profile with floor/cap clipping."""
    log10_kzz_1bar = float(
        rng.uniform(kzz_cfg["log10_kzz_at_1bar_min"], kzz_cfg["log10_kzz_at_1bar_max"])
    )
    beta = float(rng.uniform(kzz_cfg["beta_min"], kzz_cfg["beta_max"]))
    kzz_1bar = 10.0**log10_kzz_1bar

    profile = kzz_1bar * np.power(np.maximum(pressure_bar, 1e-30), -beta)
    floor = float(kzz_cfg["kzz_floor_cm2_s"])
    cap = float(kzz_cfg["kzz_cap_cm2_s"])
    profile = np.clip(profile, floor, cap)

    if not np.all(np.isfinite(profile)):
        raise SamplingError("Non-finite Kzz profile sampled.")

    params = {
        "log10_kzz_at_1bar": log10_kzz_1bar,
        "beta": beta,
        "kzz_floor_cm2_s": floor,
        "kzz_cap_cm2_s": cap,
    }
    return profile.astype(np.float64), params


def sample_abundances(
    abundance_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[dict[str, float], float, float]:
    """Sample metallicity + C/O and derive elemental abundance inputs."""
    log10_metallicity = float(
        rng.uniform(abundance_cfg["log10_metallicity_min"], abundance_cfg["log10_metallicity_max"])
    )
    c_to_o = float(rng.uniform(abundance_cfg["c_to_o_min"], abundance_cfg["c_to_o_max"]))
    metallicity_scale = 10.0**log10_metallicity

    solar = abundance_cfg["solar_abundances"]
    o_h = float(solar["O_H"]) * metallicity_scale
    # C/H is intentionally derived from sampled O/H and C/O.
    c_h = o_h * c_to_o

    abundances = {
        "O_H": o_h,
        "C_H": c_h,
        "N_H": float(solar["N_H"]) * metallicity_scale,
        "S_H": float(solar["S_H"]) * metallicity_scale,
        "He_H": float(solar["He_H"]),
        "fastchem_met_scale": metallicity_scale,
    }

    if any(val <= 0.0 or not np.isfinite(val) for val in abundances.values()):
        raise SamplingError("Invalid sampled abundance values.")

    return abundances, log10_metallicity, c_to_o


def sample_gravity(gravity_cfg: dict[str, Any], rng: np.random.Generator) -> float:
    """Sample or resolve gravity from explicit configuration."""
    distribution = str(gravity_cfg["distribution"]).lower()
    if distribution == "uniform":
        min_cm_s2 = float(gravity_cfg["min_cm_s2"])
        max_cm_s2 = float(gravity_cfg["max_cm_s2"])
        if min_cm_s2 <= 0.0 or max_cm_s2 <= min_cm_s2:
            raise SamplingError("gravity_sampler uniform bounds must satisfy 0 < min < max.")
        return float(rng.uniform(min_cm_s2, max_cm_s2))

    if distribution == "fixed":
        value_cm_s2 = float(gravity_cfg["value_cm_s2"])
        if value_cm_s2 <= 0.0:
            raise SamplingError("gravity_sampler.value_cm_s2 must be > 0.")
        return value_cm_s2

    raise SamplingError(f"Unsupported gravity_sampler.distribution: {distribution}")


def build_run_specs(config: dict[str, Any]) -> list[RunSpec]:
    """Create deterministic run specifications from configured samplers."""
    generation = config["generation"]
    rng = np.random.default_rng(int(generation["random_seed"]))
    pressure_bar = build_pressure_grid(config["tp_sampler"])
    tp_cfg = config["tp_sampler"]
    gravity_cfg = config["gravity_sampler"]
    max_attempts = int(tp_cfg["max_sampling_attempts"])
    if max_attempts <= 0:
        raise SamplingError("tp_sampler.max_sampling_attempts must be > 0.")

    run_specs: list[RunSpec] = []
    for run_id in range(int(generation["num_runs"])):
        last_tp_error: SamplingError | None = None
        for _attempt in range(max_attempts):
            abundances, metallicity_log10, c_to_o = sample_abundances(
                config["abundance_sampler"],
                rng,
            )
            gravity_cm_s2 = sample_gravity(gravity_cfg, rng)
            try:
                temperature_k, tp_params = generate_tp_profile(
                    pressure_bar=pressure_bar,
                    tp_cfg=tp_cfg,
                    gravity_cm_s2=gravity_cm_s2,
                    rng=rng,
                )
            except SamplingError as exc:
                last_tp_error = exc
                continue

            kzz_cm2_s, kzz_params = sample_kzz_profile(pressure_bar, config["kzz_sampler"], rng)
            run_specs.append(
                RunSpec(
                    run_id=run_id,
                    pressure_bar=pressure_bar.copy(),
                    temperature_k=temperature_k,
                    kzz_cm2_s=kzz_cm2_s,
                    gravity_cm_s2=gravity_cm_s2,
                    metallicity_log10=metallicity_log10,
                    c_to_o=c_to_o,
                    abundances=abundances,
                    tp_params=tp_params,
                    kzz_params=kzz_params,
                )
            )
            break
        else:
            reason = (
                str(last_tp_error)
                if last_tp_error is not None
                else "unknown temperature sampling failure"
            )
            raise SamplingError(
                f"Failed to sample a valid TP profile for run_id={run_id} "
                f"after {max_attempts} attempts: {reason}"
            )

    return run_specs
