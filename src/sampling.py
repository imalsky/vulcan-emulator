from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config_utils import static_conditioning_defaults
from .roth_sampling import load_roth_profiles
from .spectrum import (
    SpectrumRecord,
    load_spectrum_manifest,
    save_spectrum_manifest,
)

# Temperature profile sampling ranges around a smooth hot-Jupiter profile.
_TP_TRANSITION_LOG10_PRESSURE_RANGE = (-2.0, 0.7)
_TP_TRANSITION_WIDTH_RANGE = (0.4, 1.0)
_TP_PROFILE_NOISE_STD_K = 15.0

# Mixing profile constants used by the synthetic smoke path.
_HELIUM_FRACTION_RANGE = (0.11, 0.17)


@dataclass(frozen=True)
class RunSpecification:
    run_id: str
    pressure_bar: np.ndarray
    temperature_k: np.ndarray
    kzz_cm2_s: np.ndarray
    initial_ymix: np.ndarray
    time_s: np.ndarray
    globals: dict[str, float]
    spectrum: SpectrumRecord
    metadata: dict[str, Any]


def sample_pressure_grid(
    *,
    num_levels: int,
    pressure_top_bar: float,
    pressure_bottom_bar: float,
) -> np.ndarray:
    """Build the fixed pressure grid used by the surrogate."""
    return np.logspace(
        math.log10(pressure_bottom_bar),
        math.log10(pressure_top_bar),
        int(num_levels),
        dtype=np.float64,
    )


def _analytic_temperature_profile(
    pressure_bar: np.ndarray,
    *,
    rng: np.random.Generator,
    t_low: float,
    t_high: float,
) -> np.ndarray:
    """Sample a smooth hot-Jupiter temperature profile on the fixed pressure grid."""
    logp = np.log10(np.asarray(pressure_bar, dtype=np.float64))
    t_deep = rng.uniform(t_high - 120.0, t_high)
    t_upper = rng.uniform(t_low, t_low + 120.0)
    logp_transition = rng.uniform(*_TP_TRANSITION_LOG10_PRESSURE_RANGE)
    width = rng.uniform(*_TP_TRANSITION_WIDTH_RANGE)
    logistic = 1.0 / (1.0 + np.exp(-(logp - logp_transition) / width))
    profile = t_upper + (t_deep - t_upper) * logistic
    profile += rng.normal(0.0, _TP_PROFILE_NOISE_STD_K, size=profile.shape)
    return np.clip(profile, 200.0, None)


def sample_temperature_profile(
    pressure_bar: np.ndarray,
    *,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a temperature profile, optionally from an external profile library."""
    roth_cfg = config.get("roth_sampler", {"enabled": False})
    if roth_cfg.get("enabled", False):
        profiles = load_roth_profiles(
            roth_cfg["data_glob"],
            pressure_grid_bar=pressure_bar,
            filters=roth_cfg.get("filters", {}),
        )
        if not profiles:
            raise FileNotFoundError(
                f"roth_sampler.enabled=true but no temperature profiles matched {roth_cfg['data_glob']!r}."
            )
        chosen = profiles[int(rng.integers(0, len(profiles)))]
        return np.asarray(chosen.temperature_k, dtype=np.float64)

    t_low, t_high = [float(x) for x in config["sampling"]["temperature_range_k"]]
    return _analytic_temperature_profile(
        pressure_bar,
        rng=rng,
        t_low=t_low,
        t_high=t_high,
    )


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


def _heavy_species_budget(metallicity_log10: float) -> float:
    heavy = 0.01 * (10.0 ** metallicity_log10)
    return float(np.clip(heavy, 1.0e-4, 0.15))


def sample_initial_ymix(
    pressure_bar: np.ndarray,
    *,
    config: dict[str, Any],
    rng: np.random.Generator,
    metallicity_log10: float,
    c_to_o: float,
) -> np.ndarray:
    """Sample a sulfur-aware initial composition for the synthetic smoke mode."""
    species = list(config["data_spec"]["state_species"])
    nz = pressure_bar.size
    state_dim = len(species)
    y = np.full((nz, state_dim), 1.0e-30, dtype=np.float64)
    idx = {name: i for i, name in enumerate(species)}

    heavy_budget = _heavy_species_budget(metallicity_log10)
    he_fraction = rng.uniform(*_HELIUM_FRACTION_RANGE)
    h2_fraction = max(1.0 - heavy_budget - he_fraction, 0.7)

    logp = np.log10(pressure_bar)
    deep_weight = (logp - logp.min()) / max(logp.max() - logp.min(), 1.0e-6)
    deep_weight = np.clip(deep_weight, 0.0, 1.0)

    carbon_budget = heavy_budget * (0.35 + 0.35 * (c_to_o / max(c_to_o + 1.0, 1.0e-6)))
    oxygen_budget = heavy_budget * 0.35
    nitrogen_budget = heavy_budget * 0.1
    sulfur_budget = heavy_budget * 0.03 * (10.0 ** (0.5 * metallicity_log10))
    sulfur_budget = float(np.clip(sulfur_budget, 1.0e-8, 0.02 * heavy_budget))

    y[:, idx["He"]] = he_fraction
    y[:, idx["H2"]] = h2_fraction
    y[:, idx["H"]] = 5.0e-7 * (1.0 + 6.0 * (1.0 - deep_weight))
    y[:, idx["O"]] = oxygen_budget * 3.0e-4 * (1.0 + 3.0 * (1.0 - deep_weight))
    y[:, idx["OH"]] = oxygen_budget * 8.0e-4 * (1.0 + 4.0 * (1.0 - deep_weight))

    y[:, idx["H2O"]] = oxygen_budget * (0.7 + 0.3 * (1.0 - deep_weight))
    y[:, idx["CO"]] = carbon_budget * (0.3 + 0.5 * deep_weight)
    y[:, idx["CO2"]] = carbon_budget * (0.08 + 0.18 * (1.0 - deep_weight))
    y[:, idx["CH4"]] = carbon_budget * (0.2 + 0.25 * (1.0 - deep_weight))
    y[:, idx["N2"]] = nitrogen_budget * (0.6 + 0.3 * deep_weight)
    y[:, idx["NH3"]] = nitrogen_budget * (0.2 + 0.2 * (1.0 - deep_weight))
    y[:, idx["H2S"]] = sulfur_budget * (0.85 + 0.1 * deep_weight)
    y[:, idx["SH"]] = sulfur_budget * 1.0e-3 * (1.0 + 2.0 * (1.0 - deep_weight))
    y[:, idx["S"]] = sulfur_budget * 5.0e-4 * (1.0 + 2.0 * (1.0 - deep_weight))
    y[:, idx["SO"]] = sulfur_budget * 5.0e-5
    y[:, idx["SO2"]] = sulfur_budget * 1.0e-5
    y[:, idx["S2"]] = sulfur_budget * 1.0e-5

    perturb = np.exp(rng.normal(0.0, 0.15, size=y.shape))
    y *= perturb
    y = np.clip(y, 1.0e-30, None)
    other_sum = np.sum(y[:, [i for name, i in idx.items() if name not in {"H2", "He"}]], axis=1)
    reservoir = np.clip(1.0 - other_sum, 1.0e-4, 1.0)
    y[:, idx["H2"]] = reservoir * (h2_fraction / max(h2_fraction + he_fraction, 1.0e-12))
    y[:, idx["He"]] = reservoir * (he_fraction / max(h2_fraction + he_fraction, 1.0e-12))
    y /= np.sum(y, axis=1, keepdims=True)
    return y


def sample_time_grid(*, config: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """Sample a monotonically increasing saved-time grid for the synthetic path."""
    sampling = config["sampling"]
    steps = int(sampling["num_time_steps"])
    log_dt = rng.uniform(
        float(sampling["time_step_log10_min_s"]),
        float(sampling["time_step_log10_max_s"]),
        size=steps - 1,
    )
    dt_s = np.power(10.0, np.asarray(log_dt, dtype=np.float64))
    time_s = np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(dt_s)])
    return time_s


def _latin_hypercube_unit_samples(
    *,
    num_samples: int,
    num_dimensions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a deterministic Latin-hypercube design in [0, 1]."""
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
    return float(lower + value * (upper - lower))


def _ensure_wasp39_template(
    *,
    project_root: Path,
    config: dict[str, Any],
) -> SpectrumRecord:
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
    manifest = ensure_default_spectrum_library(project_root=project_root, config=config)
    return load_spectrum_manifest(manifest)


def sample_run_specifications(
    *,
    config: dict[str, Any],
    project_root: Path,
    num_runs: int | None = None,
    seed: int | None = None,
) -> list[RunSpecification]:
    """Sample the atmospheric configurations used to generate raw runs."""
    rng = np.random.default_rng(
        int(config["generation"]["seed"] if seed is None else seed)
    )
    total_runs = int(config["generation"]["num_runs"] if num_runs is None else num_runs)
    design = _latin_hypercube_unit_samples(
        num_samples=total_runs,
        num_dimensions=3,
        rng=rng,
    )
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
        temperature_k = sample_temperature_profile(pressure_bar, config=config, rng=rng)
        kzz = sample_kzz_profile(pressure_bar, config=config, rng=rng)
        spectrum_name = spectrum_names[int(rng.integers(0, len(spectrum_names)))]
        template = spectra[spectrum_name]
        spectrum = SpectrumRecord(
            name=f"{template.name}_run{run_idx:05d}",
            wavelength_nm=np.asarray(template.wavelength_nm, dtype=np.float64),
            flux_erg_cm2_s_nm=np.asarray(template.flux_erg_cm2_s_nm, dtype=np.float64),
            metadata={**template.metadata, "template_name": template.name},
        )
        initial_ymix = sample_initial_ymix(
            pressure_bar,
            config=config,
            rng=rng,
            metallicity_log10=metallicity,
            c_to_o=c_to_o,
        )
        globals_map = {
            "gravity_cm_s2": float(gravity),
            "metallicity_log10": float(metallicity),
            "c_to_o": float(c_to_o),
            **{key: float(value) for key, value in physics_defaults.items()},
        }
        result.append(
            RunSpecification(
                run_id=f"run_{run_idx:05d}",
                pressure_bar=np.asarray(pressure_bar, dtype=np.float64),
                temperature_k=np.asarray(temperature_k, dtype=np.float64),
                kzz_cm2_s=np.asarray(kzz, dtype=np.float64),
                initial_ymix=np.asarray(initial_ymix, dtype=np.float64),
                time_s=sample_time_grid(config=config, rng=rng),
                globals=globals_map,
                spectrum=spectrum,
                metadata={"spectrum_name": spectrum.name},
            )
        )
    return result
