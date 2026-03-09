"""Configuration loading and strict validation.

Implements fail-fast validation for all 13+ config sections.  Every numeric
field is type-checked (rejecting booleans as integers), every range is
validated, and cross-section consistency is enforced (e.g., target
normalization must match anchor normalization, d_model must be divisible
by nhead, AMP requires CUDA).

The base-10 log convention is explicitly enforced: all log-scale parameters
must use ``log10_`` prefixes, and normalization methods that would double-log
already-transformed values are rejected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

SUPPORTED_DTYPE_NAMES = {"float16", "bfloat16", "float32", "float64", "none"}
TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}

SUPPORTED_ATM_BASES = ("H2", "N2", "O2", "CO2", "H2O")
SUPPORTED_CONDITIONING_TOGGLES = (
    "use_eddy_diffusion",
    "use_molecular_diffusion",
    "use_upwind_molecular_diffusion",
    "use_boundary_conditions",
    "use_condensation",
    "use_settling",
    "use_initial_cold_trap",
    "use_sat_surface_h2o",
    "use_lowT_limit_rates",
    "use_adaptive_rtol",
)
CORE_GLOBAL_INPUTS = (
    "gravity_cm_s2",
    "metallicity_log10",
    "c_to_o",
    "log10_dt_s",
)
OPTIONAL_GLOBAL_INPUTS = (
    *SUPPORTED_CONDITIONING_TOGGLES,
    *tuple(f"atm_base_{name}" for name in SUPPORTED_ATM_BASES),
)
SUPPORTED_GLOBAL_INPUTS = (*CORE_GLOBAL_INPUTS, *OPTIONAL_GLOBAL_INPUTS)


@dataclass(frozen=True)
class PrecisionConfig:
    """Resolved precision policy with strict compatibility checks."""

    input_dtype: torch.dtype
    stats_dtype: torch.dtype
    model_dtype: torch.dtype
    forward_dtype: torch.dtype
    loss_dtype: torch.dtype
    optimizer_state_dtype: torch.dtype
    amp_dtype: torch.dtype | None
    use_amp: bool


class ConfigValidationError(ValueError):
    """Raised when configuration violates required contract."""


def static_conditioning_defaults(config: dict[str, Any]) -> dict[str, float]:
    """Return dataset-wide conditioning values derived from config settings."""
    physics = config["physics_toggles"]
    runtime = config["vulcan_runtime"]
    defaults = {
        name: float(bool(physics[name]))
        for name in SUPPORTED_CONDITIONING_TOGGLES
    }
    atm_base = str(runtime["atm_base"])
    defaults.update(
        {
            f"atm_base_{name}": 1.0 if atm_base == name else 0.0
            for name in SUPPORTED_ATM_BASES
        }
    )
    return defaults


def resolve_conditioning_inputs(
    *,
    raw_global_inputs: dict[str, float],
    config: dict[str, Any],
    required_global_inputs: list[str],
) -> dict[str, float]:
    """Resolve all non-time conditioning inputs for one run/sample."""
    resolved: dict[str, float] = {}
    defaults = static_conditioning_defaults(config)
    for name in required_global_inputs:
        if name == "log10_dt_s":
            continue
        if name in raw_global_inputs:
            resolved[name] = float(raw_global_inputs[name])
            continue
        if name in defaults:
            resolved[name] = float(defaults[name])
            continue
        raise ConfigValidationError(
            f"Missing required conditioning input '{name}' in raw run globals."
        )
    return resolved


def _require_keys(container: dict[str, Any], required: set[str], scope: str) -> None:
    """Require a fixed key set in one config mapping."""
    missing = sorted(required - set(container.keys()))
    if missing:
        raise ConfigValidationError(f"Missing required keys in {scope}: {missing}")


def _reject_extra_keys(container: dict[str, Any], allowed: set[str], scope: str) -> None:
    """Reject unexpected keys in one config mapping."""
    extras = sorted(set(container.keys()) - allowed)
    if extras:
        raise ConfigValidationError(f"Unexpected keys in {scope}: {extras}")


def _as_float(value: Any, field: str) -> float:
    """Parse one scalar as float while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(f"'{field}' must be numeric.")
    return float(value)


def _as_int(value: Any, field: str) -> int:
    """Parse one scalar as int while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(f"'{field}' must be an integer.")
    return int(value)


def _as_bool(value: Any, field: str) -> bool:
    """Parse one scalar as bool with no implicit coercion."""
    if not isinstance(value, bool):
        raise ConfigValidationError(f"'{field}' must be a boolean.")
    return bool(value)


def _parse_dtype_name(value: Any, field: str, *, allow_none: bool = False) -> torch.dtype | None:
    """Resolve one configured dtype string to a Torch dtype."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"'{field}' must be a non-empty string.")
    lowered = value.strip().lower()
    if lowered not in SUPPORTED_DTYPE_NAMES:
        raise ConfigValidationError(
            f"Unsupported dtype for '{field}': {value}. Supported: {sorted(SUPPORTED_DTYPE_NAMES)}"
        )
    if lowered == "none":
        if not allow_none:
            raise ConfigValidationError(f"'{field}' cannot be 'none'.")
        return None
    return TORCH_DTYPE_MAP[lowered]


def resolve_precision(config: dict[str, Any]) -> PrecisionConfig:
    """Resolve and validate configured precision policy."""
    precision = config["precision"]
    training = config["training"]
    _require_keys(
        precision,
        {
            "input_dtype",
            "stats_accumulation_dtype",
            "model_dtype",
            "forward_dtype",
            "loss_dtype",
            "optimizer_state_dtype",
            "amp_autocast_dtype",
        },
        "precision",
    )

    input_dtype = _parse_dtype_name(precision["input_dtype"], "precision.input_dtype")
    stats_dtype = _parse_dtype_name(
        precision["stats_accumulation_dtype"],
        "precision.stats_accumulation_dtype",
    )
    model_dtype = _parse_dtype_name(precision["model_dtype"], "precision.model_dtype")
    forward_dtype = _parse_dtype_name(precision["forward_dtype"], "precision.forward_dtype")
    loss_dtype = _parse_dtype_name(precision["loss_dtype"], "precision.loss_dtype")
    optimizer_dtype = _parse_dtype_name(
        precision["optimizer_state_dtype"],
        "precision.optimizer_state_dtype",
    )
    amp_dtype = _parse_dtype_name(
        precision["amp_autocast_dtype"],
        "precision.amp_autocast_dtype",
        allow_none=True,
    )

    use_amp = _as_bool(training["use_amp"], "training.use_amp")
    device = str(training["device"]).lower()

    if stats_dtype not in (torch.float32, torch.float64):
        raise ConfigValidationError(
            "precision.stats_accumulation_dtype must be float32 or float64."
        )
    if forward_dtype != model_dtype:
        raise ConfigValidationError("precision.forward_dtype must match precision.model_dtype.")
    if optimizer_dtype != model_dtype:
        raise ConfigValidationError(
            "precision.optimizer_state_dtype must match precision.model_dtype."
        )

    if use_amp:
        if device != "cuda":
            raise ConfigValidationError("training.use_amp=true requires training.device='cuda'.")
        if amp_dtype not in (torch.float16, torch.bfloat16):
            raise ConfigValidationError(
                "precision.amp_autocast_dtype must be float16 or bfloat16 when AMP is enabled."
            )
        if model_dtype != torch.float32:
            raise ConfigValidationError("AMP requires precision.model_dtype='float32'.")
    else:
        if amp_dtype is not None:
            raise ConfigValidationError(
                "precision.amp_autocast_dtype must be 'none' when AMP is disabled."
            )

    return PrecisionConfig(
        input_dtype=input_dtype,
        stats_dtype=stats_dtype,
        model_dtype=model_dtype,
        forward_dtype=forward_dtype,
        loss_dtype=loss_dtype,
        optimizer_state_dtype=optimizer_dtype,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
    )


def _validate_relative_path_values(paths_cfg: dict[str, Any]) -> None:
    """Require all configured path values to be relative non-empty strings."""
    for key, value in paths_cfg.items():
        if not isinstance(value, str) or not value:
            raise ConfigValidationError(f"paths.{key} must be a non-empty string.")
        if Path(value).is_absolute():
            raise ConfigValidationError(f"paths.{key} must be relative, got: {value}")


def _validate_paths(paths_cfg: dict[str, Any]) -> None:
    """Validate the top-level `paths` section against the project contract."""
    allowed = {
        "project_root",
        "vulcan_source_path",
        "data_root",
        "raw_root",
        "processed_root",
        "models_root",
        "logs_root",
    }
    _require_keys(paths_cfg, allowed, "paths")
    _reject_extra_keys(paths_cfg, allowed, "paths")
    _validate_relative_path_values(paths_cfg)
    if str(paths_cfg["project_root"]) != ".":
        raise ConfigValidationError("paths.project_root must be '.'.")


def _validate_distribution_spec(spec: dict[str, Any], scope: str) -> None:
    """Validate one scalar sampling distribution specification."""
    if not isinstance(spec, dict):
        raise ConfigValidationError(f"{scope} must be a mapping.")

    _require_keys(spec, {"distribution"}, scope)
    distribution = str(spec["distribution"]).lower()
    if distribution == "uniform":
        _require_keys(spec, {"distribution", "min", "max"}, scope)
        lower = _as_float(spec["min"], f"{scope}.min")
        upper = _as_float(spec["max"], f"{scope}.max")
        if upper <= lower:
            raise ConfigValidationError(f"{scope} uniform bounds must satisfy min < max.")
        return

    if distribution == "normal":
        _require_keys(spec, {"distribution", "mean", "std"}, scope)
        _ = _as_float(spec["mean"], f"{scope}.mean")
        if _as_float(spec["std"], f"{scope}.std") <= 0.0:
            raise ConfigValidationError(f"{scope}.std must be > 0.")
        if "min" in spec and "max" in spec:
            lower = _as_float(spec["min"], f"{scope}.min")
            upper = _as_float(spec["max"], f"{scope}.max")
            if upper <= lower:
                raise ConfigValidationError(f"{scope} clamp bounds must satisfy min < max.")
        elif "min" in spec:
            _ = _as_float(spec["min"], f"{scope}.min")
        elif "max" in spec:
            _ = _as_float(spec["max"], f"{scope}.max")
        return

    raise ConfigValidationError(f"Unsupported distribution for {scope}: {distribution}")


def _validate_generation(cfg: dict[str, Any]) -> None:
    """Validate generation-stage controls and output-path settings."""
    allowed = {
        "num_runs",
        "num_workers",
        "split_ratios",
        "random_seed",
        "keep_vulcan_outputs_debug",
        "failure_policy",
        "run_timeout_seconds",
        "manifest_filename",
        "split_filename",
        "shard_size",
        "worker_root",
        "save_evo_frq",
        "max_trajectory_snapshots",
    }
    _require_keys(cfg, allowed, "generation")
    _reject_extra_keys(cfg, allowed, "generation")

    if _as_int(cfg["num_runs"], "generation.num_runs") <= 0:
        raise ConfigValidationError("generation.num_runs must be > 0.")
    if _as_int(cfg["num_workers"], "generation.num_workers") <= 0:
        raise ConfigValidationError("generation.num_workers must be > 0.")
    _ = _as_int(cfg["random_seed"], "generation.random_seed")
    _ = _as_bool(cfg["keep_vulcan_outputs_debug"], "generation.keep_vulcan_outputs_debug")
    if str(cfg["failure_policy"]) not in (
        "fail_on_first_error",
        "collect_all_errors",
        "continue_on_error",
    ):
        raise ConfigValidationError(
            "generation.failure_policy must be 'fail_on_first_error', "
            "'collect_all_errors', or 'continue_on_error'."
        )
    if _as_int(cfg["run_timeout_seconds"], "generation.run_timeout_seconds") <= 0:
        raise ConfigValidationError("generation.run_timeout_seconds must be > 0.")
    if _as_int(cfg["shard_size"], "generation.shard_size") <= 0:
        raise ConfigValidationError("generation.shard_size must be > 0.")
    if _as_int(cfg["save_evo_frq"], "generation.save_evo_frq") <= 0:
        raise ConfigValidationError("generation.save_evo_frq must be > 0.")
    max_snap = _as_int(cfg["max_trajectory_snapshots"], "generation.max_trajectory_snapshots")
    if max_snap < 0:
        raise ConfigValidationError("generation.max_trajectory_snapshots must be >= 0 (0 = no limit).")

    for path_key in ("worker_root", "manifest_filename", "split_filename"):
        value = cfg[path_key]
        if not isinstance(value, str) or not value:
            raise ConfigValidationError(f"generation.{path_key} must be a non-empty string path.")
        if Path(value).is_absolute():
            raise ConfigValidationError(f"generation.{path_key} must be relative, got: {value}")

    ratios = cfg["split_ratios"]
    _require_keys(ratios, {"train", "val", "test"}, "generation.split_ratios")
    train = _as_float(ratios["train"], "generation.split_ratios.train")
    val = _as_float(ratios["val"], "generation.split_ratios.val")
    test = _as_float(ratios["test"], "generation.split_ratios.test")
    if min(train, val, test) <= 0.0:
        raise ConfigValidationError("Split ratios must all be > 0.")
    if abs((train + val + test) - 1.0) > 1e-9:
        raise ConfigValidationError("Split ratios must sum to 1.0.")


def _validate_trajectory_sampling(cfg: dict[str, Any]) -> None:
    """Validate log-uniform all-pairs transition sampling controls."""
    _require_keys(
        cfg,
        {
            "mode",
            "pairs_per_run",
            "dt_min_s",
            "dt_max_s",
            "min_future_saved_steps",
            "rollout_eval_points",
        },
        "trajectory_sampling",
    )
    if str(cfg["mode"]) != "log_uniform_all_pairs":
        raise ConfigValidationError(
            "trajectory_sampling.mode must be 'log_uniform_all_pairs'."
        )
    if _as_int(cfg["pairs_per_run"], "trajectory_sampling.pairs_per_run") <= 0:
        raise ConfigValidationError("trajectory_sampling.pairs_per_run must be > 0.")

    dt_min_s = _as_float(cfg["dt_min_s"], "trajectory_sampling.dt_min_s")
    dt_max_s = _as_float(cfg["dt_max_s"], "trajectory_sampling.dt_max_s")
    if dt_min_s <= 0.0:
        raise ConfigValidationError("trajectory_sampling.dt_min_s must be > 0.")
    if dt_max_s <= dt_min_s:
        raise ConfigValidationError("trajectory_sampling.dt_max_s must be > dt_min_s.")

    if _as_int(
        cfg["min_future_saved_steps"],
        "trajectory_sampling.min_future_saved_steps",
    ) < 1:
        raise ConfigValidationError(
            "trajectory_sampling.min_future_saved_steps must be >= 1."
        )

    if _as_int(cfg["rollout_eval_points"], "trajectory_sampling.rollout_eval_points") < 2:
        raise ConfigValidationError(
            "trajectory_sampling.rollout_eval_points must be >= 2."
        )


def _validate_vulcan_runtime(cfg: dict[str, Any]) -> None:
    """Validate the explicit VULCAN runtime override section."""
    _require_keys(
        cfg,
        {
            "runtime",
            "dt_min",
            "dt_max",
            "count_max",
            "trun_min",
            "count_min",
            "ini_mix",
            "atm_base",
        },
        "vulcan_runtime",
    )
    runtime = _as_float(cfg["runtime"], "vulcan_runtime.runtime")
    dt_min = _as_float(cfg["dt_min"], "vulcan_runtime.dt_min")
    dt_max = _as_float(cfg["dt_max"], "vulcan_runtime.dt_max")
    if runtime <= 0.0:
        raise ConfigValidationError("vulcan_runtime.runtime must be > 0.")
    if dt_min <= 0.0 or dt_max <= dt_min:
        raise ConfigValidationError(
            "vulcan_runtime.dt_* must satisfy 0 < dt_min < dt_max."
        )
    if _as_int(cfg["count_max"], "vulcan_runtime.count_max") <= 0:
        raise ConfigValidationError("vulcan_runtime.count_max must be > 0.")
    if _as_float(cfg["trun_min"], "vulcan_runtime.trun_min") < 0.0:
        raise ConfigValidationError("vulcan_runtime.trun_min must be >= 0.")
    if _as_int(cfg["count_min"], "vulcan_runtime.count_min") < 0:
        raise ConfigValidationError("vulcan_runtime.count_min must be >= 0.")

    ini_mix = str(cfg["ini_mix"])
    if ini_mix != "EQ":
        raise ConfigValidationError(
            "vulcan_runtime.ini_mix must be 'EQ' in this codebase. Other initialization modes "
            "need additional input artifacts that are not wired into the generator."
        )

    atm_base = str(cfg["atm_base"])
    if atm_base not in set(SUPPORTED_ATM_BASES):
        raise ConfigValidationError(
            "vulcan_runtime.atm_base must be one of {'H2','N2','O2','CO2','H2O'}."
        )


def _validate_tp_sampler(tp_cfg: dict[str, Any]) -> None:
    """Validate TP-profile sampling controls and physical bounds."""
    _require_keys(
        tp_cfg,
        {
            "pressure_grid",
            "temperature_limits_k",
            "max_sampling_attempts",
            "adiabatic_gradient",
            "convective_adjustment_probability",
            "log10_kappa_ir",
            "kappa_pressure_power_exponent",
            "log10_gamma1",
            "log10_gamma2",
            "alpha_partition",
            "t_int",
            "t_irr",
            "temperature_shift",
        },
        "tp_sampler",
    )
    grid = tp_cfg["pressure_grid"]
    _require_keys(grid, {"nz", "p_top_bar", "p_bottom_bar"}, "tp_sampler.pressure_grid")
    if _as_int(grid["nz"], "tp_sampler.pressure_grid.nz") <= 1:
        raise ConfigValidationError("tp_sampler.pressure_grid.nz must be > 1.")
    p_top = _as_float(grid["p_top_bar"], "tp_sampler.pressure_grid.p_top_bar")
    p_bottom = _as_float(grid["p_bottom_bar"], "tp_sampler.pressure_grid.p_bottom_bar")
    if p_top <= 0.0 or p_bottom <= 0.0 or p_bottom <= p_top:
        raise ConfigValidationError("Pressure bounds must satisfy 0 < p_top < p_bottom.")

    temp_limits = tp_cfg["temperature_limits_k"]
    _require_keys(temp_limits, {"min", "max"}, "tp_sampler.temperature_limits_k")
    t_min = _as_float(temp_limits["min"], "tp_sampler.temperature_limits_k.min")
    t_max = _as_float(temp_limits["max"], "tp_sampler.temperature_limits_k.max")
    if t_min <= 0.0 or t_max <= t_min:
        raise ConfigValidationError("Invalid temperature limits.")

    if _as_int(tp_cfg["max_sampling_attempts"], "tp_sampler.max_sampling_attempts") <= 0:
        raise ConfigValidationError("tp_sampler.max_sampling_attempts must be > 0.")
    if _as_float(tp_cfg["adiabatic_gradient"], "tp_sampler.adiabatic_gradient") <= 0.0:
        raise ConfigValidationError("tp_sampler.adiabatic_gradient must be > 0.")

    prob = _as_float(
        tp_cfg["convective_adjustment_probability"],
        "tp_sampler.convective_adjustment_probability",
    )
    if not (0.0 <= prob <= 1.0):
        raise ConfigValidationError("convective_adjustment_probability must be in [0,1].")

    for key in (
        "log10_kappa_ir",
        "kappa_pressure_power_exponent",
        "log10_gamma1",
        "log10_gamma2",
        "alpha_partition",
        "t_int",
        "t_irr",
        "temperature_shift",
    ):
        _validate_distribution_spec(tp_cfg[key], f"tp_sampler.{key}")

    alpha_cfg = tp_cfg["alpha_partition"]
    if "min" not in alpha_cfg or "max" not in alpha_cfg:
        raise ConfigValidationError(
            "tp_sampler.alpha_partition must define explicit min/max bounds within [0,1]."
        )
    alpha_min = _as_float(alpha_cfg["min"], "tp_sampler.alpha_partition.min")
    alpha_max = _as_float(alpha_cfg["max"], "tp_sampler.alpha_partition.max")
    if alpha_min < 0.0 or alpha_max > 1.0:
        raise ConfigValidationError("tp_sampler.alpha_partition bounds must stay within [0,1].")


def _validate_gravity_sampler(gravity_cfg: dict[str, Any]) -> None:
    """Validate the gravity sampling strategy for one run family."""
    _require_keys(gravity_cfg, {"distribution"}, "gravity_sampler")
    distribution = str(gravity_cfg["distribution"]).lower()
    if distribution == "uniform":
        _require_keys(gravity_cfg, {"distribution", "min_cm_s2", "max_cm_s2"}, "gravity_sampler")
        g_min = _as_float(gravity_cfg["min_cm_s2"], "gravity_sampler.min_cm_s2")
        g_max = _as_float(gravity_cfg["max_cm_s2"], "gravity_sampler.max_cm_s2")
        if g_min <= 0.0 or g_max <= g_min:
            raise ConfigValidationError(
                "gravity_sampler uniform bounds must satisfy 0 < min < max."
            )
        return
    if distribution == "fixed":
        _require_keys(gravity_cfg, {"distribution", "value_cm_s2"}, "gravity_sampler")
        value = _as_float(gravity_cfg["value_cm_s2"], "gravity_sampler.value_cm_s2")
        if value <= 0.0:
            raise ConfigValidationError("gravity_sampler.value_cm_s2 must be > 0.")
        return
    raise ConfigValidationError(
        "gravity_sampler.distribution must be one of {'uniform', 'fixed'}."
    )


def _validate_kzz_sampler(kzz_cfg: dict[str, Any]) -> None:
    """Validate the configured Kzz profile family and clipping limits."""
    _require_keys(
        kzz_cfg,
        {
            "mode",
            "log10_kzz_at_1bar_min",
            "log10_kzz_at_1bar_max",
            "beta_min",
            "beta_max",
            "kzz_floor_cm2_s",
            "kzz_cap_cm2_s",
            "pressure_unit_for_profile",
        },
        "kzz_sampler",
    )
    if str(kzz_cfg["mode"]) != "power_law_profile":
        raise ConfigValidationError("kzz_sampler.mode must be 'power_law_profile'.")
    if str(kzz_cfg["pressure_unit_for_profile"]) != "bar":
        raise ConfigValidationError("kzz_sampler.pressure_unit_for_profile must be 'bar'.")

    lo = _as_float(kzz_cfg["log10_kzz_at_1bar_min"], "kzz_sampler.log10_kzz_at_1bar_min")
    hi = _as_float(kzz_cfg["log10_kzz_at_1bar_max"], "kzz_sampler.log10_kzz_at_1bar_max")
    if hi <= lo:
        raise ConfigValidationError("Invalid Kzz@1bar log10 range.")
    beta_lo = _as_float(kzz_cfg["beta_min"], "kzz_sampler.beta_min")
    beta_hi = _as_float(kzz_cfg["beta_max"], "kzz_sampler.beta_max")
    if beta_hi <= beta_lo:
        raise ConfigValidationError("Invalid kzz beta range.")
    floor = _as_float(kzz_cfg["kzz_floor_cm2_s"], "kzz_sampler.kzz_floor_cm2_s")
    cap = _as_float(kzz_cfg["kzz_cap_cm2_s"], "kzz_sampler.kzz_cap_cm2_s")
    if floor <= 0.0 or cap <= floor:
        raise ConfigValidationError("Kzz floor/cap must satisfy 0 < floor < cap.")


def _validate_abundance_sampler(abund_cfg: dict[str, Any]) -> None:
    """Validate elemental abundance sampling and solar reference values."""
    _require_keys(
        abund_cfg,
        {
            "metallicity_mode",
            "log10_metallicity_min",
            "log10_metallicity_max",
            "c_to_o_min",
            "c_to_o_max",
            "solar_abundances",
        },
        "abundance_sampler",
    )
    if str(abund_cfg["metallicity_mode"]) != "log10_scale":
        raise ConfigValidationError("abundance_sampler.metallicity_mode must be 'log10_scale'.")
    lo = _as_float(abund_cfg["log10_metallicity_min"], "abundance_sampler.log10_metallicity_min")
    hi = _as_float(abund_cfg["log10_metallicity_max"], "abundance_sampler.log10_metallicity_max")
    if hi <= lo:
        raise ConfigValidationError("Invalid metallicity range.")
    co_lo = _as_float(abund_cfg["c_to_o_min"], "abundance_sampler.c_to_o_min")
    co_hi = _as_float(abund_cfg["c_to_o_max"], "abundance_sampler.c_to_o_max")
    if co_hi <= co_lo or co_lo <= 0.0:
        raise ConfigValidationError("Invalid C/O range.")

    solar = abund_cfg["solar_abundances"]
    _require_keys(solar, {"O_H", "N_H", "He_H", "S_H"}, "abundance_sampler.solar_abundances")
    for key, value in solar.items():
        if _as_float(value, f"abundance_sampler.solar_abundances.{key}") <= 0.0:
            raise ConfigValidationError(f"Solar abundance {key} must be > 0.")


def _validate_species_list(species: Any, field: str) -> list[str]:
    """Validate one ordered species-name list."""
    if not isinstance(species, list) or not species:
        raise ConfigValidationError(f"{field} must be a non-empty list of species names.")
    if any((not isinstance(item, str) or not item.strip()) for item in species):
        raise ConfigValidationError(f"{field} contains an invalid species name.")
    if len(set(species)) != len(species):
        raise ConfigValidationError(f"{field} contains duplicate species names.")
    return [str(item) for item in species]


def _validate_data_spec(data_spec: dict[str, Any]) -> None:
    """Validate the explicit transition-model input/output feature contract."""
    _require_keys(
        data_spec,
        {
            "state_species",
            "output_species",
            "required_input_profiles",
            "required_global_inputs",
            "required_state_inputs",
            "time_input_transform",
            "strict_non_finite",
        },
        "data_spec",
    )

    state_species = _validate_species_list(data_spec["state_species"], "data_spec.state_species")
    output_species = _validate_species_list(data_spec["output_species"], "data_spec.output_species")
    missing_output = [species for species in output_species if species not in state_species]
    if missing_output:
        raise ConfigValidationError(
            "data_spec.output_species must be a subset of data_spec.state_species; missing "
            f"{missing_output}."
        )

    if str(data_spec["time_input_transform"]) != "log10_dt_seconds":
        raise ConfigValidationError(
            "data_spec.time_input_transform must be 'log10_dt_seconds'."
        )
    _ = _as_bool(data_spec["strict_non_finite"], "data_spec.strict_non_finite")

    expected_inputs = ["pressure_bar", "temperature_k", "kzz_cm2_s"]
    if list(data_spec["required_input_profiles"]) != expected_inputs:
        raise ConfigValidationError(
            f"data_spec.required_input_profiles must equal {expected_inputs}."
        )
    required_globals = list(data_spec["required_global_inputs"])
    expected_prefix = list(CORE_GLOBAL_INPUTS)
    if required_globals[: len(expected_prefix)] != expected_prefix:
        raise ConfigValidationError(
            "data_spec.required_global_inputs must begin with "
            f"{expected_prefix}."
        )
    if any(name not in SUPPORTED_GLOBAL_INPUTS for name in required_globals):
        invalid = sorted(set(required_globals) - set(SUPPORTED_GLOBAL_INPUTS))
        raise ConfigValidationError(
            "data_spec.required_global_inputs contains unsupported entries: "
            f"{invalid}."
        )
    canonical = [name for name in SUPPORTED_GLOBAL_INPUTS if name in required_globals]
    if required_globals != canonical:
        raise ConfigValidationError(
            "data_spec.required_global_inputs must follow the canonical ordering "
            f"{canonical}."
        )
    if list(data_spec["required_state_inputs"]) != ["anchor_ymix"]:
        raise ConfigValidationError(
            "data_spec.required_state_inputs must equal ['anchor_ymix']."
        )


def _validate_normalization(norm_cfg: dict[str, Any], data_spec: dict[str, Any]) -> None:
    """Validate explicit normalization methods for all feature groups."""
    _require_keys(
        norm_cfg,
        {"epsilon", "sequence_methods", "global_methods", "target_method"},
        "normalization",
    )
    if _as_float(norm_cfg["epsilon"], "normalization.epsilon") <= 0.0:
        raise ConfigValidationError("normalization.epsilon must be > 0.")

    allowed = {"standard", "log-standard", "log-min-max", "none"}

    sequence_methods = norm_cfg["sequence_methods"]
    if not isinstance(sequence_methods, dict) or not sequence_methods:
        raise ConfigValidationError("normalization.sequence_methods must be a non-empty mapping.")
    required_sequence = {"pressure_bar", "temperature_k", "kzz_cm2_s", "anchor_ymix"}
    if set(sequence_methods) != required_sequence:
        raise ConfigValidationError(
            "normalization.sequence_methods must define exactly keys "
            f"{sorted(required_sequence)}, got {sorted(sequence_methods)}."
        )
    for key, method in sequence_methods.items():
        if method not in allowed:
            raise ConfigValidationError(
                f"Unsupported normalization.sequence_methods.{key}: {method}."
            )

    global_methods = norm_cfg["global_methods"]
    if not isinstance(global_methods, dict) or not global_methods:
        raise ConfigValidationError("normalization.global_methods must be a non-empty mapping.")
    required_globals = list(data_spec["required_global_inputs"])
    if list(global_methods.keys()) != required_globals:
        raise ConfigValidationError(
            "normalization.global_methods must define exactly keys "
            f"{required_globals}, got {list(global_methods.keys())}."
        )
    for key, method in global_methods.items():
        if method not in allowed:
            raise ConfigValidationError(
                f"Unsupported normalization.global_methods.{key}: {method}."
            )
    if global_methods["metallicity_log10"] not in {"standard", "none"}:
        raise ConfigValidationError(
            "normalization.global_methods.metallicity_log10 cannot use log-based methods "
            "because metallicity_log10 is already base-10 transformed."
        )
    if global_methods["log10_dt_s"] not in {"standard", "none"}:
        raise ConfigValidationError(
            "normalization.global_methods.log10_dt_s cannot use log-based methods because "
            "time_input_transform already produces log10_dt_s."
        )

    target_method = str(norm_cfg["target_method"])
    if target_method not in allowed:
        raise ConfigValidationError("Unsupported normalization.target_method.")
    if target_method != str(sequence_methods["anchor_ymix"]):
        raise ConfigValidationError(
            "normalization.target_method must match normalization.sequence_methods.anchor_ymix "
            "so residual training occurs in one shared normalized state space."
        )


def _validate_training(training: dict[str, Any]) -> None:
    """Validate training hyperparameters, model shape, and loading policy."""
    _require_keys(
        training,
        {
            "device",
            "gpu_preload",
            "batch_size",
            "epochs",
            "learning_rate",
            "min_lr",
            "warmup_epochs",
            "weight_decay",
            "gradient_clip",
            "use_amp",
            "num_workers",
            "seed",
            "data_loading",
            "model",
            "output_folder",
        },
        "training",
    )

    device = str(training["device"]).lower()
    if device not in {"cpu", "cuda", "mps"}:
        raise ConfigValidationError("training.device must be one of {'cpu','cuda','mps'}.")
    if _as_int(training["batch_size"], "training.batch_size") <= 0:
        raise ConfigValidationError("training.batch_size must be > 0.")
    gpu_preload = _as_bool(training["gpu_preload"], "training.gpu_preload")
    epochs = _as_int(training["epochs"], "training.epochs")
    if epochs <= 0:
        raise ConfigValidationError("training.epochs must be > 0.")
    warmup = _as_int(training["warmup_epochs"], "training.warmup_epochs")
    if warmup < 0 or warmup > epochs:
        raise ConfigValidationError("training.warmup_epochs must be in [0, epochs].")

    lr = _as_float(training["learning_rate"], "training.learning_rate")
    min_lr = _as_float(training["min_lr"], "training.min_lr")
    if lr <= 0.0 or min_lr <= 0.0 or min_lr > lr:
        raise ConfigValidationError("training.min_lr must be >0 and <= training.learning_rate.")
    if _as_float(training["weight_decay"], "training.weight_decay") < 0.0:
        raise ConfigValidationError("training.weight_decay must be >= 0.")
    if _as_float(training["gradient_clip"], "training.gradient_clip") <= 0.0:
        raise ConfigValidationError("training.gradient_clip must be > 0.")
    if _as_int(training["num_workers"], "training.num_workers") < 0:
        raise ConfigValidationError("training.num_workers must be >= 0.")
    _ = _as_int(training["seed"], "training.seed")
    _ = _as_bool(training["use_amp"], "training.use_amp")

    data_loading = training["data_loading"]
    _require_keys(
        data_loading,
        {
            "mode",
            "max_cached_shards",
            "large_shard_mmap_bytes",
            "ram_safety_fraction",
            "copy_mmap_slices",
            "use_device_prefetch",
        },
        "training.data_loading",
    )
    mode = str(data_loading["mode"]).lower()
    if mode not in {"auto", "ram", "disk"}:
        raise ConfigValidationError(
            "training.data_loading.mode must be one of {'auto','ram','disk'}."
        )
    if _as_int(data_loading["max_cached_shards"], "training.data_loading.max_cached_shards") <= 0:
        raise ConfigValidationError("training.data_loading.max_cached_shards must be > 0.")
    if _as_int(
        data_loading["large_shard_mmap_bytes"],
        "training.data_loading.large_shard_mmap_bytes",
    ) <= 0:
        raise ConfigValidationError(
            "training.data_loading.large_shard_mmap_bytes must be > 0."
        )
    safety = _as_float(
        data_loading["ram_safety_fraction"],
        "training.data_loading.ram_safety_fraction",
    )
    if not (0.0 < safety <= 1.0):
        raise ConfigValidationError("training.data_loading.ram_safety_fraction must be in (0,1].")
    _ = _as_bool(data_loading["copy_mmap_slices"], "training.data_loading.copy_mmap_slices")
    use_prefetch = _as_bool(
        data_loading["use_device_prefetch"],
        "training.data_loading.use_device_prefetch",
    )
    if gpu_preload and device != "cuda":
        raise ConfigValidationError("training.gpu_preload=true requires training.device='cuda'.")
    if use_prefetch and device != "cuda":
        raise ConfigValidationError(
            "training.data_loading.use_device_prefetch=true requires training.device='cuda'."
        )

    model_cfg = training["model"]
    _require_keys(
        model_cfg,
        {
            "d_model",
            "nhead",
            "num_layers",
            "dim_feedforward",
            "dropout",
            "film_clamp",
            "output_head_divisor",
            "max_sequence_length",
            "conditioning_hidden_dim",
        },
        "training.model",
    )
    d_model = _as_int(model_cfg["d_model"], "training.model.d_model")
    if d_model <= 0:
        raise ConfigValidationError("training.model.d_model must be > 0.")
    nhead = _as_int(model_cfg["nhead"], "training.model.nhead")
    if nhead <= 0:
        raise ConfigValidationError("training.model.nhead must be > 0.")
    if d_model % 2 != 0:
        raise ConfigValidationError("training.model.d_model must be even.")
    if d_model % nhead != 0:
        raise ConfigValidationError(
            "training.model.d_model must be divisible by training.model.nhead."
        )
    if _as_int(model_cfg["num_layers"], "training.model.num_layers") <= 0:
        raise ConfigValidationError("training.model.num_layers must be > 0.")
    if _as_int(model_cfg["dim_feedforward"], "training.model.dim_feedforward") <= 0:
        raise ConfigValidationError("training.model.dim_feedforward must be > 0.")
    if _as_float(model_cfg["dropout"], "training.model.dropout") < 0.0:
        raise ConfigValidationError("training.model.dropout must be >= 0.")
    if _as_float(model_cfg["film_clamp"], "training.model.film_clamp") <= 0.0:
        raise ConfigValidationError("training.model.film_clamp must be > 0.")
    if _as_int(model_cfg["output_head_divisor"], "training.model.output_head_divisor") <= 0:
        raise ConfigValidationError("training.model.output_head_divisor must be > 0.")
    if _as_int(model_cfg["max_sequence_length"], "training.model.max_sequence_length") <= 0:
        raise ConfigValidationError("training.model.max_sequence_length must be > 0.")
    if _as_int(model_cfg["conditioning_hidden_dim"], "training.model.conditioning_hidden_dim") <= 0:
        raise ConfigValidationError(
            "training.model.conditioning_hidden_dim must be > 0."
        )
    output_folder = training["output_folder"]
    if not isinstance(output_folder, str) or not output_folder:
        raise ConfigValidationError("training.output_folder must be a non-empty relative path.")
    if Path(output_folder).is_absolute():
        raise ConfigValidationError("training.output_folder must be relative.")


def _validate_physics_toggles(physics: dict[str, Any]) -> None:
    """Validate supported VULCAN on/off physics and solver toggles."""
    _require_keys(
        physics,
        {
            "use_photochemistry",
            "use_ion_chemistry",
            "use_eddy_diffusion",
            "use_molecular_diffusion",
            "use_upwind_molecular_diffusion",
            "use_boundary_conditions",
            "use_condensation",
            "use_settling",
            "use_initial_cold_trap",
            "use_sat_surface_h2o",
            "use_lowT_limit_rates",
            "use_adaptive_rtol",
        },
        "physics_toggles",
    )
    for key in (
        "use_photochemistry",
        "use_ion_chemistry",
        "use_eddy_diffusion",
        "use_molecular_diffusion",
        "use_upwind_molecular_diffusion",
        "use_boundary_conditions",
        "use_condensation",
        "use_settling",
        "use_initial_cold_trap",
        "use_sat_surface_h2o",
        "use_lowT_limit_rates",
        "use_adaptive_rtol",
    ):
        _ = _as_bool(physics[key], f"physics_toggles.{key}")

    if physics["use_photochemistry"]:
        raise ConfigValidationError(
            "Photochemistry is intentionally not wired in this emulator. Set use_photochemistry=false."
        )
    if physics["use_ion_chemistry"]:
        raise ConfigValidationError(
            "Ion chemistry is intentionally not wired in this emulator. Set use_ion_chemistry=false."
        )
    if physics["use_settling"] and not physics["use_condensation"]:
        raise ConfigValidationError(
            "physics_toggles.use_settling=true requires physics_toggles.use_condensation=true."
        )
    if physics["use_upwind_molecular_diffusion"] and not physics["use_molecular_diffusion"]:
        raise ConfigValidationError(
            "physics_toggles.use_upwind_molecular_diffusion=true requires "
            "physics_toggles.use_molecular_diffusion=true."
        )


def _validate_boundary_conditions(
    boundary_cfg: dict[str, Any] | None,
    *,
    required: bool,
) -> None:
    """Validate boundary-condition settings when that subsystem is enabled."""
    if boundary_cfg is None:
        if required:
            raise ConfigValidationError(
                "physics_toggles.use_boundary_conditions=true requires a top-level "
                "'boundary_conditions' section."
            )
        return

    _require_keys(
        boundary_cfg,
        {
            "use_topflux",
            "use_botflux",
            "top_BC_flux_file",
            "bot_BC_flux_file",
            "use_fix_sp_bot",
        },
        "boundary_conditions",
    )

    use_topflux = _as_bool(boundary_cfg["use_topflux"], "boundary_conditions.use_topflux")
    use_botflux = _as_bool(boundary_cfg["use_botflux"], "boundary_conditions.use_botflux")
    for field_name in ("top_BC_flux_file", "bot_BC_flux_file"):
        value = boundary_cfg[field_name]
        if not isinstance(value, str) or not value:
            raise ConfigValidationError(
                f"boundary_conditions.{field_name} must be a non-empty string."
            )
        if Path(value).is_absolute():
            raise ConfigValidationError(f"boundary_conditions.{field_name} must be relative.")

    fixed_bottom = boundary_cfg["use_fix_sp_bot"]
    if not isinstance(fixed_bottom, dict):
        raise ConfigValidationError("boundary_conditions.use_fix_sp_bot must be a mapping.")
    for species_name, value in fixed_bottom.items():
        if not isinstance(species_name, str) or not species_name.strip():
            raise ConfigValidationError(
                "boundary_conditions.use_fix_sp_bot keys must be non-empty species names."
            )
        ratio = _as_float(value, f"boundary_conditions.use_fix_sp_bot.{species_name}")
        if not (0.0 <= ratio <= 1.0):
            raise ConfigValidationError(
                f"boundary_conditions.use_fix_sp_bot.{species_name} must be in [0, 1]."
            )

    if required and not use_topflux and not use_botflux and not fixed_bottom:
        raise ConfigValidationError(
            "boundary_conditions are enabled, but no top flux, bottom flux, or fixed bottom "
            "mixing ratios were configured."
        )


def _validate_log10_convention(config: dict[str, Any]) -> None:
    """Require the project-wide base-10 naming convention for log controls."""
    required_log10_keys = {
        "tp_sampler.log10_kappa_ir",
        "tp_sampler.log10_gamma1",
        "tp_sampler.log10_gamma2",
        "kzz_sampler.log10_kzz_at_1bar_min",
        "kzz_sampler.log10_kzz_at_1bar_max",
        "abundance_sampler.log10_metallicity_min",
        "abundance_sampler.log10_metallicity_max",
    }
    missing: list[str] = []
    for dotted in required_log10_keys:
        section, key = dotted.split(".", 1)
        if key not in config[section]:
            missing.append(dotted)
    if missing:
        raise ConfigValidationError(
            f"Base-10 log convention violated; missing required keys: {sorted(missing)}"
        )



def _validate_cross_section_contracts(config: dict[str, Any]) -> None:
    """Validate relationships that span multiple config sections."""
    trajectory_sampling = config["trajectory_sampling"]
    vulcan_runtime = config["vulcan_runtime"]

    dt_min_s = float(trajectory_sampling["dt_min_s"])
    dt_max_s = float(trajectory_sampling["dt_max_s"])
    runtime = float(vulcan_runtime["runtime"])

    if dt_min_s >= runtime:
        raise ConfigValidationError(
            "trajectory_sampling.dt_min_s must be < vulcan_runtime.runtime."
        )
    if dt_max_s > runtime:
        raise ConfigValidationError(
            "trajectory_sampling.dt_max_s must be <= vulcan_runtime.runtime."
        )


def load_and_validate_config(path: Path) -> dict[str, Any]:
    """Load JSON config and validate all required contract rules."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        try:
            config = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ConfigValidationError(f"Invalid JSON config: {exc}") from exc

    required_top = {
        "paths",
        "generation",
        "trajectory_sampling",
        "vulcan_runtime",
        "tp_sampler",
        "gravity_sampler",
        "kzz_sampler",
        "abundance_sampler",
        "physics_toggles",
        "data_spec",
        "normalization",
        "precision",
        "training",
    }
    _require_keys(config, required_top, "root")

    _validate_paths(config["paths"])
    _validate_generation(config["generation"])
    _validate_trajectory_sampling(config["trajectory_sampling"])
    _validate_vulcan_runtime(config["vulcan_runtime"])
    _validate_tp_sampler(config["tp_sampler"])
    _validate_gravity_sampler(config["gravity_sampler"])
    _validate_kzz_sampler(config["kzz_sampler"])
    _validate_abundance_sampler(config["abundance_sampler"])
    _validate_data_spec(config["data_spec"])
    _validate_normalization(config["normalization"], config["data_spec"])
    _validate_training(config["training"])
    _validate_physics_toggles(config["physics_toggles"])
    _validate_boundary_conditions(
        config.get("boundary_conditions"),
        required=bool(config["physics_toggles"]["use_boundary_conditions"]),
    )
    _validate_log10_convention(config)
    _validate_cross_section_contracts(config)
    _ = resolve_precision(config)
    return config
