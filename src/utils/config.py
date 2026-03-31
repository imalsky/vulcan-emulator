"""Configuration loading, validation, and normalization for both task types.

This module is the single source of truth for config schema validation.  It:

1. Loads a JSON config file from disk.
2. Validates every field against its expected type, range, and task-specific
   constraints (equilibrium_only vs full_vulcan).
3. Derives internal aliases (``model_type``, ``training.model``, ``roth_sampler``)
   so downstream code can rely on a normalized, validated structure.
4. Returns a dict that is safe to pass to any pipeline stage.

Validation is strict and fail-fast: any schema violation raises
``ConfigValidationError`` with a descriptive message identifying the
offending field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUPPORTED_ATM_BASES = ("H2", "N2", "O2", "CO2", "H2O")
PUBLIC_PHYSICS_TOGGLES = (
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
)
TASK_KINDS = ("equilibrium_only", "full_vulcan")
TASK_KIND_TO_MODEL_TYPE = {
    "equilibrium_only": "equilibrium",
    "full_vulcan": "full_vulcan",
}
ELEMENT_INPUT_ORDER = ("He_H", "C_H", "O_H", "N_H", "S_H")
FULL_VULCAN_CORE_GLOBAL_INPUTS = (
    "gravity_cm_s2",
    *ELEMENT_INPUT_ORDER,
)
EQUILIBRIUM_CORE_GLOBAL_INPUTS = ELEMENT_INPUT_ORDER
FULL_VULCAN_OPTIONAL_GLOBAL_INPUTS = (
    *PUBLIC_PHYSICS_TOGGLES,
    *tuple(f"atm_base_{name}" for name in SUPPORTED_ATM_BASES),
)
DEFAULT_REQUIRED_GLOBAL_INPUTS = (
    *FULL_VULCAN_CORE_GLOBAL_INPUTS,
    *FULL_VULCAN_OPTIONAL_GLOBAL_INPUTS,
)
DEFAULT_STATE_SPECIES = (
    "H2",
    "He",
    "H",
    "O",
    "OH",
    "H2O",
    "CO",
    "CO2",
    "CH4",
    "N2",
    "NH3",
    "H2S",
    "SH",
    "S",
    "SO",
    "SO2",
    "S2",
)
ALLOWED_MODEL_TYPES = tuple(sorted(set(TASK_KIND_TO_MODEL_TYPE.values())))
_ALLOWED_SPECTRUM_ENCODERS = {"autoencoder", "linear", "none"}
_ALLOWED_ACTIVATIONS = {
    "elu",
    "gelu",
    "leaky_relu",
    "relu",
    "selu",
    "silu",
    "softplus",
    "tanh",
}
_ALLOWED_LR_SCHEDULERS = {"cosine", "reduce_on_plateau"}
_ALLOWED_TEMPERATURE_PROFILE_SOURCE_MODES = {"analytic", "pt_library", "mixed"}
_ALLOWED_NORMALIZATION_METHODS = {"standard", "log-standard", "none"}
_ALLOWED_TEMPERATURE_PROFILE_NUMERIC_FILTER_KEYS = {
    "Teq",
    "LogMet",
    "LogDrag",
    "Mstar",
    "Rp",
    "logG",
}
_ALLOWED_TEMPERATURE_PROFILE_BOOLEAN_FILTER_KEYS = {"TiOVO"}
_ALLOWED_TEMPERATURE_PROFILE_FILTER_KEYS = (
    _ALLOWED_TEMPERATURE_PROFILE_NUMERIC_FILTER_KEYS
    | _ALLOWED_TEMPERATURE_PROFILE_BOOLEAN_FILTER_KEYS
)
_REMOVED_LEGACY_KEYS = {
    "training.live_sampling": (
        "training.live_sampling has been removed. The full-VULCAN path now trains "
        "only on final converged outputs."
    ),
    "sampling.num_time_steps": (
        "sampling.num_time_steps has been removed. Full-VULCAN raw generation no "
        "longer stores sampled trajectories."
    ),
    "sampling.time_step_log10_min_s": (
        "sampling.time_step_log10_min_s has been removed. Full-VULCAN raw generation "
        "no longer stores sampled trajectories."
    ),
    "sampling.time_step_log10_max_s": (
        "sampling.time_step_log10_max_s has been removed. Full-VULCAN raw generation "
        "no longer stores sampled trajectories."
    ),
    "generation.target_mode": (
        "generation.target_mode has been removed. task.kind now fully determines the "
        "supported generation contract."
    ),
    "normalization.state_method": (
        "normalization.state_method has been removed. Full-VULCAN training now uses "
        "final-state targets only."
    ),
    "normalization.log10_dt_method": (
        "normalization.log10_dt_method has been removed. Full-VULCAN training no "
        "longer includes timestep control."
    ),
    "full_vulcan.trajectory_sampling": (
        "full_vulcan.trajectory_sampling has been removed. The supported full-VULCAN "
        "task is final-state-only."
    ),
}
_INTERNAL_VULCAN_RUNTIME_DEFAULTS = {
    "python_executable": "python",
    "cfg_file": "vulcan_cfg.py",
    "worker_root": "data/vulcan_workers",
    "regenerate_chem_funs": False,
    "cfg_assignments": {},
    "use_lowT_limit_rates": True,
    "use_adaptive_rtol": True,
}


class ConfigValidationError(ValueError):
    """Raised when a configuration file violates the required contract."""


def _require_keys(mapping: dict[str, Any], keys: tuple[str, ...] | list[str], scope: str) -> None:
    """Raise when a mapping is missing required keys for a config scope."""
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigValidationError(f"Missing required keys in {scope}: {missing}")


def _nested_key_present(mapping: dict[str, Any], path: str) -> bool:
    """Return True when a dotted config path exists in the user payload."""
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def _reject_removed_legacy_keys(config: dict[str, Any]) -> None:
    """Fail fast on removed config keys with targeted migration messages."""
    for path, message in _REMOVED_LEGACY_KEYS.items():
        if _nested_key_present(config, path):
            raise ConfigValidationError(message)


def _as_bool(value: Any, field: str) -> bool:
    """Validate and return a boolean config value."""
    if not isinstance(value, bool):
        raise ConfigValidationError(f"{field} must be a boolean.")
    return value


def _as_int(value: Any, field: str) -> int:
    """Validate and return an integer config value."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(f"{field} must be an integer.")
    return int(value)


def _as_float(value: Any, field: str) -> float:
    """Validate and return a numeric config value as a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(f"{field} must be numeric.")
    return float(value)


def _as_nonempty_str(value: Any, field: str) -> str:
    """Validate and return a non-empty string config value."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"{field} must be a non-empty string.")
    return value.strip()


def _as_string_list(value: Any, field: str) -> list[str]:
    """Validate and return a deduplicated list of non-empty strings."""
    if not isinstance(value, list) or not value:
        raise ConfigValidationError(f"{field} must be a non-empty list.")
    result = [_as_nonempty_str(item, field) for item in value]
    if len(set(result)) != len(result):
        raise ConfigValidationError(f"{field} contains duplicate entries.")
    return result


def _normalized_method_name(
    value: Any,
    field: str,
    *,
    allowed: set[str] | None = None,
) -> str:
    """Validate and return one normalization method name."""
    method_name = _as_nonempty_str(value, field).lower()
    allowed_methods = _ALLOWED_NORMALIZATION_METHODS if allowed is None else allowed
    if method_name not in allowed_methods:
        raise ConfigValidationError(
            f"{field} must be one of {sorted(allowed_methods)}, got {method_name!r}."
        )
    return method_name


def _task_kind(config: dict[str, Any]) -> str:
    """Return the canonical task kind from the new top-level task block."""
    task = config.get("task")
    if not isinstance(task, dict):
        raise ConfigValidationError("root.task must be a mapping.")
    kind = _as_nonempty_str(task.get("kind"), "task.kind").lower()
    if kind not in TASK_KINDS:
        raise ConfigValidationError(f"task.kind must be one of {TASK_KINDS}, got {kind!r}.")
    return kind


def get_model_type(config: dict[str, Any]) -> str:
    """Return the normalized internal model type derived from task.kind."""
    if "task" in config:
        return TASK_KIND_TO_MODEL_TYPE[_task_kind(config)]
    model_type = config.get("model_type")
    if model_type is not None:
        normalized = _as_nonempty_str(model_type, "model_type").lower()
        if normalized in ALLOWED_MODEL_TYPES:
            return normalized
    raise ConfigValidationError("Config must define task.kind.")


def is_equilibrium(config: dict[str, Any]) -> bool:
    """True when the config describes an equilibrium-only model."""
    return get_model_type(config) == "equilibrium"


def task_kind(config: dict[str, Any]) -> str:
    """Return the canonical task kind for the active config."""
    if "task" in config:
        return _task_kind(config)
    return "equilibrium_only" if is_equilibrium(config) else "full_vulcan"


def static_conditioning_defaults(config: dict[str, Any]) -> dict[str, float]:
    """Build static conditioning defaults for full-VULCAN models."""
    if is_equilibrium(config):
        return {}
    physics = config["physics_toggles"]
    runtime = config["vulcan_runtime"]
    defaults = {name: float(bool(physics[name])) for name in PUBLIC_PHYSICS_TOGGLES}
    atm_base = _as_nonempty_str(runtime["atm_base"], "full_vulcan.vulcan_runtime.atm_base")
    if atm_base not in SUPPORTED_ATM_BASES:
        raise ConfigValidationError(
            f"full_vulcan.vulcan_runtime.atm_base must be one of {SUPPORTED_ATM_BASES}, got {atm_base!r}."
        )
    defaults.update(
        {
            f"atm_base_{name}": 1.0 if name == atm_base else 0.0
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
    """Resolve required global inputs from raw values plus static defaults."""
    defaults = static_conditioning_defaults(config)
    resolved: dict[str, float] = {}
    for name in required_global_inputs:
        if name in raw_global_inputs:
            resolved[name] = float(raw_global_inputs[name])
        elif name in defaults:
            resolved[name] = float(defaults[name])
        else:
            raise ConfigValidationError(
                f"Missing required conditioning input {name!r} in raw globals."
            )
    return resolved


def global_static_feature_order(config: dict[str, Any]) -> list[str]:
    """Return the global feature order (static conditioning inputs)."""
    return list(config["data_spec"]["required_global_inputs"])


def global_feature_order(config: dict[str, Any]) -> list[str]:
    """Full global feature vector order (identical to static order)."""
    return list(config["data_spec"]["required_global_inputs"])


def _validate_equilibrium_model_config(model: dict[str, Any], scope: str) -> dict[str, Any]:
    """Validate the equilibrium-MLP config section."""
    _require_keys(
        model,
        ["d_hidden", "num_hidden_layers", "conditioning_hidden_dim", "film_clamp"],
        scope,
    )
    normalized = dict(model)
    for key in ("d_hidden", "num_hidden_layers", "conditioning_hidden_dim"):
        normalized[key] = _as_int(normalized[key], f"{scope}.{key}")
    normalized["film_clamp"] = _as_float(normalized["film_clamp"], f"{scope}.film_clamp")
    normalized["activation"] = _as_nonempty_str(
        normalized.get("activation", "leaky_relu"),
        f"{scope}.activation",
    ).lower()
    if normalized["activation"] not in _ALLOWED_ACTIVATIONS:
        raise ConfigValidationError(
            f"{scope}.activation must be one of {_ALLOWED_ACTIVATIONS}."
        )
    if normalized["d_hidden"] < 8 or normalized["num_hidden_layers"] < 1:
        raise ConfigValidationError(f"{scope} dimensions are too small.")
    if normalized["conditioning_hidden_dim"] < 1:
        raise ConfigValidationError(f"{scope}.conditioning_hidden_dim must be >= 1.")
    if normalized["film_clamp"] <= 0.0:
        raise ConfigValidationError(f"{scope}.film_clamp must be positive.")
    return normalized


def _validate_full_vulcan_model_config(model: dict[str, Any], scope: str) -> dict[str, Any]:
    """Validate the full-VULCAN transformer config section."""
    _require_keys(
        model,
        [
            "d_model",
            "nhead",
            "num_layers",
            "dim_feedforward",
            "conditioning_hidden_dim",
            "film_clamp",
            "output_head_divisor",
        ],
        scope,
    )
    normalized = dict(model)
    for key in (
        "d_model",
        "nhead",
        "num_layers",
        "dim_feedforward",
        "conditioning_hidden_dim",
        "output_head_divisor",
    ):
        normalized[key] = _as_int(normalized[key], f"{scope}.{key}")
    normalized["film_clamp"] = _as_float(normalized["film_clamp"], f"{scope}.film_clamp")
    normalized["activation"] = _as_nonempty_str(
        normalized.get("activation", "leaky_relu"),
        f"{scope}.activation",
    ).lower()
    if normalized["activation"] not in _ALLOWED_ACTIVATIONS:
        raise ConfigValidationError(
            f"{scope}.activation must be one of {_ALLOWED_ACTIVATIONS}."
        )
    if normalized["d_model"] < 8 or normalized["nhead"] < 1 or normalized["num_layers"] < 1:
        raise ConfigValidationError(f"{scope} dimensions are too small.")
    if normalized["conditioning_hidden_dim"] < 1:
        raise ConfigValidationError(f"{scope}.conditioning_hidden_dim must be >= 1.")
    if normalized["d_model"] % normalized["nhead"] != 0:
        raise ConfigValidationError(f"{scope}.d_model must be divisible by nhead.")
    if normalized["dim_feedforward"] < normalized["d_model"]:
        raise ConfigValidationError(f"{scope}.dim_feedforward must be >= d_model.")
    if normalized["output_head_divisor"] < 1:
        raise ConfigValidationError(f"{scope}.output_head_divisor must be >= 1.")
    if normalized["film_clamp"] <= 0.0:
        raise ConfigValidationError(f"{scope}.film_clamp must be positive.")
    return normalized


def _validate_training_scheduler(scheduler: Any, scope: str) -> dict[str, Any]:
    """Validate the optional training scheduler block."""
    if scheduler is None:
        scheduler = {}
    if not isinstance(scheduler, dict):
        raise ConfigValidationError(f"{scope} must be a mapping.")
    normalized = dict(scheduler)
    normalized["name"] = _as_nonempty_str(
        normalized.get("name", "reduce_on_plateau"),
        f"{scope}.name",
    ).lower()
    if normalized["name"] not in _ALLOWED_LR_SCHEDULERS:
        raise ConfigValidationError(
            f"{scope}.name must be one of {_ALLOWED_LR_SCHEDULERS}."
        )
    if normalized["name"] == "cosine":
        return {"name": "cosine"}

    normalized["factor"] = _as_float(
        normalized.get("factor", 0.5),
        f"{scope}.factor",
    )
    normalized["patience"] = _as_int(
        normalized.get("patience", 10),
        f"{scope}.patience",
    )
    normalized["threshold"] = _as_float(
        normalized.get("threshold", 1.0e-4),
        f"{scope}.threshold",
    )
    if not 0.0 < normalized["factor"] < 1.0:
        raise ConfigValidationError(f"{scope}.factor must lie strictly between 0 and 1.")
    if normalized["patience"] < 0:
        raise ConfigValidationError(f"{scope}.patience must be >= 0.")
    if normalized["threshold"] < 0.0:
        raise ConfigValidationError(f"{scope}.threshold must be >= 0.")
    return {
        "name": "reduce_on_plateau",
        "factor": normalized["factor"],
        "patience": normalized["patience"],
        "threshold": normalized["threshold"],
    }


def _validate_numeric_range(
    value: Any,
    field: str,
    *,
    allow_equal: bool = False,
) -> list[float]:
    """Validate a two-number range field and return normalized bounds."""
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigValidationError(f"{field} must be a length-2 list.")
    lower = _as_float(value[0], f"{field}[0]")
    upper = _as_float(value[1], f"{field}[1]")
    if allow_equal:
        if upper < lower:
            raise ConfigValidationError(f"{field} must satisfy lower <= upper.")
    elif upper <= lower:
        raise ConfigValidationError(f"{field} must satisfy lower < upper.")
    return [lower, upper]


def _validate_normal_distribution(spec: Any, field: str) -> dict[str, float]:
    """Validate a normal-distribution config block."""
    if not isinstance(spec, dict):
        raise ConfigValidationError(f"{field} must be a mapping.")
    _require_keys(spec, ["mean", "std"], field)
    mean = _as_float(spec["mean"], f"{field}.mean")
    std = _as_float(spec["std"], f"{field}.std")
    if std < 0.0:
        raise ConfigValidationError(f"{field}.std must be non-negative.")
    return {"mean": mean, "std": std}


def _validate_temperature_profile_validation(
    validation_config: Any,
    scope: str,
) -> dict[str, float]:
    """Validate shared temperature-profile bounds."""
    field = f"{scope}.validation"
    if not isinstance(validation_config, dict):
        raise ConfigValidationError(f"{field} must be a mapping.")
    normalized = dict(validation_config)
    _require_keys(normalized, ["min_temperature_k", "max_temperature_k"], field)
    normalized["min_temperature_k"] = _as_float(
        normalized["min_temperature_k"],
        f"{field}.min_temperature_k",
    )
    normalized["max_temperature_k"] = _as_float(
        normalized["max_temperature_k"],
        f"{field}.max_temperature_k",
    )
    if normalized["min_temperature_k"] <= 0.0:
        raise ConfigValidationError(f"{field}.min_temperature_k must be > 0.")
    if normalized["max_temperature_k"] <= normalized["min_temperature_k"]:
        raise ConfigValidationError(
            f"{field}.max_temperature_k must be greater than min_temperature_k."
        )
    return normalized


def _validate_analytic_temperature_sampler(sampler_config: Any) -> dict[str, Any]:
    """Validate the Line/Robinson analytic PT sampler configuration.

    Checks the analytic-profile parameter distributions and ranges used by
    the Line et al. (2013) sampler. Shared temperature validity bounds are
    validated separately under ``temperature_profiles.validation``.
    """
    scope = "temperature_profiles.analytic_sampler"
    if not isinstance(sampler_config, dict):
        raise ConfigValidationError(f"{scope} must be a mapping.")
    normalized = dict(sampler_config)
    _require_keys(
        normalized,
        [
            "reference_gravity_m_s2",
            "t_int_k_normal",
            "t_irr_k_normal",
            "log10_kappa_ir_m2_kg_normal",
            "power_law_n_range",
            "log10_gamma_1_range",
            "log10_gamma_2_range",
            "alpha_range",
            "temperature_shift_k_range",
            "convection_probability",
            "adiabatic_gradient_range",
        ],
        scope,
    )
    normalized["reference_gravity_m_s2"] = _as_float(
        normalized["reference_gravity_m_s2"],
        f"{scope}.reference_gravity_m_s2",
    )
    if normalized["reference_gravity_m_s2"] <= 0.0:
        raise ConfigValidationError(f"{scope}.reference_gravity_m_s2 must be positive.")

    for key in (
        "t_int_k_normal",
        "t_irr_k_normal",
        "log10_kappa_ir_m2_kg_normal",
    ):
        normalized[key] = _validate_normal_distribution(normalized[key], f"{scope}.{key}")

    for key in (
        "power_law_n_range",
        "log10_gamma_1_range",
        "log10_gamma_2_range",
        "temperature_shift_k_range",
        "adiabatic_gradient_range",
    ):
        normalized[key] = _validate_numeric_range(normalized[key], f"{scope}.{key}")

    normalized["alpha_range"] = _validate_numeric_range(
        normalized["alpha_range"],
        f"{scope}.alpha_range",
    )
    if normalized["alpha_range"][0] < 0.0 or normalized["alpha_range"][1] > 1.0:
        raise ConfigValidationError(f"{scope}.alpha_range must lie within [0, 1].")
    if normalized["power_law_n_range"][0] <= 0.0:
        raise ConfigValidationError(f"{scope}.power_law_n_range[0] must be > 0.")
    if normalized["adiabatic_gradient_range"][0] <= 0.0:
        raise ConfigValidationError(f"{scope}.adiabatic_gradient_range[0] must be > 0.")

    normalized["convection_probability"] = _as_float(
        normalized["convection_probability"],
        f"{scope}.convection_probability",
    )
    if not 0.0 <= normalized["convection_probability"] <= 1.0:
        raise ConfigValidationError(
            f"{scope}.convection_probability must lie in [0, 1]."
        )

    return normalized


def _validate_temperature_profiles(profile_config: Any) -> dict[str, Any]:
    """Validate shared temperature-profile sampling settings.

    Handles source_mode selection (analytic / pt_library / mixed), filter
    key validation (Teq, LogMet, etc.), analytic_sampler sub-validation,
    data_glob requirement for library modes, and analytic_probability bounds.
    """
    if not isinstance(profile_config, dict):
        raise ConfigValidationError("temperature_profiles must be a mapping.")
    normalized = dict(profile_config)
    _require_keys(normalized, ["source_mode", "validation"], "temperature_profiles")
    normalized["source_mode"] = _as_nonempty_str(
        normalized["source_mode"],
        "temperature_profiles.source_mode",
    ).lower()
    if normalized["source_mode"] not in _ALLOWED_TEMPERATURE_PROFILE_SOURCE_MODES:
        raise ConfigValidationError(
            "temperature_profiles.source_mode must be one of "
            f"{_ALLOWED_TEMPERATURE_PROFILE_SOURCE_MODES}."
        )
    normalized["validation"] = _validate_temperature_profile_validation(
        normalized["validation"],
        "temperature_profiles",
    )

    raw_filters = normalized.get("filters", {})
    if not isinstance(raw_filters, dict):
        raise ConfigValidationError("temperature_profiles.filters must be a mapping.")
    filters: dict[str, float | tuple[float, float] | bool] = {}
    for key, raw_value in raw_filters.items():
        filter_key = _as_nonempty_str(key, "temperature_profiles.filters")
        if filter_key not in _ALLOWED_TEMPERATURE_PROFILE_FILTER_KEYS:
            raise ConfigValidationError(
                f"temperature_profiles.filters.{filter_key} is not supported. "
                f"Allowed keys are {sorted(_ALLOWED_TEMPERATURE_PROFILE_FILTER_KEYS)}."
            )
        field = f"temperature_profiles.filters.{filter_key}"
        if filter_key in _ALLOWED_TEMPERATURE_PROFILE_BOOLEAN_FILTER_KEYS:
            filters[filter_key] = _as_bool(raw_value, field)
            continue
        if isinstance(raw_value, list):
            if len(raw_value) != 2:
                raise ConfigValidationError(
                    f"{field} must be a numeric scalar or a two-number inclusive range."
                )
            lower = _as_float(raw_value[0], field)
            upper = _as_float(raw_value[1], field)
            if upper < lower:
                raise ConfigValidationError(f"{field} range upper bound must be >= lower bound.")
            filters[filter_key] = (lower, upper)
            continue
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ConfigValidationError(
                f"{field} must be a numeric scalar or a two-number inclusive range."
            )
        filters[filter_key] = _as_float(raw_value, field)
    normalized["filters"] = filters

    source_mode = normalized["source_mode"]
    analytic_sampler = normalized.get("analytic_sampler")
    if source_mode in {"analytic", "mixed"}:
        if analytic_sampler is None:
            raise ConfigValidationError(
                "temperature_profiles.analytic_sampler is required when "
                "temperature_profiles.source_mode is 'analytic' or 'mixed'."
            )
        normalized["analytic_sampler"] = _validate_analytic_temperature_sampler(analytic_sampler)
    elif analytic_sampler is not None:
        normalized["analytic_sampler"] = _validate_analytic_temperature_sampler(analytic_sampler)
    else:
        normalized.pop("analytic_sampler", None)

    if source_mode in {"pt_library", "mixed"}:
        normalized["data_glob"] = _as_nonempty_str(
            normalized.get("data_glob"),
            "temperature_profiles.data_glob",
        )
    else:
        normalized.pop("data_glob", None)

    if source_mode == "mixed":
        normalized["analytic_probability"] = _as_float(
            normalized.get("analytic_probability", 0.5),
            "temperature_profiles.analytic_probability",
        )
        if not 0.0 < normalized["analytic_probability"] < 1.0:
            raise ConfigValidationError(
                "temperature_profiles.analytic_probability must lie strictly between 0 and 1 "
                "when temperature_profiles.source_mode='mixed'."
            )
    else:
        normalized.pop("analytic_probability", None)
    return normalized


def _validate_split(split: Any) -> dict[str, Any]:
    """Validate the train/val/test split policy stored under normalization."""
    if not isinstance(split, dict):
        raise ConfigValidationError("normalization.split must be a mapping.")
    _require_keys(
        split,
        ["train_fraction", "val_fraction", "test_fraction", "seed"],
        "normalization.split",
    )
    normalized = dict(split)
    for key in ("train_fraction", "val_fraction", "test_fraction"):
        normalized[key] = _as_float(normalized[key], f"normalization.split.{key}")
        if normalized[key] <= 0.0:
            raise ConfigValidationError(f"normalization.split.{key} must be positive.")
    total_fraction = (
        normalized["train_fraction"]
        + normalized["val_fraction"]
        + normalized["test_fraction"]
    )
    if abs(total_fraction - 1.0) > 1.0e-6:
        raise ConfigValidationError("normalization.split fractions must sum to 1.")
    normalized["seed"] = _as_int(normalized["seed"], "normalization.split.seed")
    return normalized


def load_and_validate_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config file, apply defaults, and validate its full contract.

    This is the primary entry point for config loading.  It reads the JSON
    file, determines the task kind, validates every section (paths, data_spec,
    sampling, temperature_profiles, generation, normalization, training, and
    the task-specific block), derives internal aliases, and returns a fully
    normalized dict.

    Parameters
    ----------
    path : str or Path
        Filesystem path to the JSON config file.

    Returns
    -------
    dict[str, Any]
        Validated and normalized config.  Safe to pass to any pipeline stage.

    Raises
    ------
    ConfigValidationError
        If any field violates its type, range, or task-specific constraint.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    _reject_removed_legacy_keys(config)

    kind = _task_kind(config)
    equilibrium = kind == "equilibrium_only"
    config["task"] = {"kind": kind}
    config["model_type"] = TASK_KIND_TO_MODEL_TYPE[kind]

    required_root = [
        "task",
        "paths",
        "data_spec",
        "sampling",
        "temperature_profiles",
        "generation",
        "normalization",
        "training",
    ]
    task_section = "equilibrium_only" if equilibrium else "full_vulcan"
    forbidden_section = "full_vulcan" if equilibrium else "equilibrium_only"
    required_root.append(task_section)
    _require_keys(config, required_root, "root")
    if forbidden_section in config:
        raise ConfigValidationError(
            f"{forbidden_section} must not be defined when task.kind={kind!r}."
        )

    paths = config["paths"]
    _require_keys(
        paths,
        [
            "raw_root",
            "processed_root",
            "checkpoints_root",
            "vulcan_source_root",
        ],
        "paths",
    )
    for key in paths:
        paths[key] = _as_nonempty_str(paths[key], f"paths.{key}")

    data_spec = config["data_spec"]
    if not isinstance(data_spec, dict):
        raise ConfigValidationError("data_spec must be a mapping.")
    state_species = _as_string_list(
        data_spec.get("state_species", list(DEFAULT_STATE_SPECIES)),
        "data_spec.state_species",
    )
    output_species = _as_string_list(
        data_spec.get("output_species", list(state_species)),
        "data_spec.output_species",
    )
    if "required_global_inputs" in data_spec:
        raise ConfigValidationError(
            "data_spec.required_global_inputs is derived internally and must not be set in the user config."
        )
    if "element_input_order" in data_spec:
        raise ConfigValidationError(
            "data_spec.element_input_order is derived internally and must not be set in the user config."
        )
    derived_globals = (
        list(EQUILIBRIUM_CORE_GLOBAL_INPUTS)
        if equilibrium
        else list(DEFAULT_REQUIRED_GLOBAL_INPUTS)
    )
    data_spec["state_species"] = state_species
    data_spec["output_species"] = output_species
    data_spec["element_input_order"] = list(ELEMENT_INPUT_ORDER)
    data_spec["required_global_inputs"] = derived_globals

    sampling = config["sampling"]
    required_sampling = [
        "num_levels",
        "pressure_top_bar",
        "pressure_bottom_bar",
        "temperature_range_k",
        "metallicity_log10_range",
        "c_to_o_range",
        "s_to_o_range",
    ]
    if not equilibrium:
        required_sampling += [
            "gravity_range_cm_s2",
            "kzz_cm2_s",
        ]
    _require_keys(sampling, required_sampling, "sampling")
    sampling["num_levels"] = _as_int(sampling["num_levels"], "sampling.num_levels")
    if sampling["num_levels"] < 4:
        raise ConfigValidationError("sampling.num_levels must be >= 4.")
    sampling["pressure_top_bar"] = _as_float(
        sampling["pressure_top_bar"],
        "sampling.pressure_top_bar",
    )
    sampling["pressure_bottom_bar"] = _as_float(
        sampling["pressure_bottom_bar"],
        "sampling.pressure_bottom_bar",
    )
    if sampling["pressure_top_bar"] <= 0.0 or sampling["pressure_bottom_bar"] <= 0.0:
        raise ConfigValidationError("Pressure bounds must be positive.")
    if sampling["pressure_top_bar"] >= sampling["pressure_bottom_bar"]:
        raise ConfigValidationError("pressure_top_bar must be smaller than pressure_bottom_bar.")
    range_keys = [
        "temperature_range_k",
        "metallicity_log10_range",
        "c_to_o_range",
        "s_to_o_range",
    ]
    if not equilibrium:
        range_keys.append("gravity_range_cm_s2")
    for key in range_keys:
        values = sampling[key]
        if not isinstance(values, list) or len(values) != 2:
            raise ConfigValidationError(f"sampling.{key} must be a length-2 list.")
        low = _as_float(values[0], f"sampling.{key}[0]")
        high = _as_float(values[1], f"sampling.{key}[1]")
        if not low < high:
            raise ConfigValidationError(f"sampling.{key} must be strictly increasing.")
        sampling[key] = [low, high]
    if not equilibrium:
        sampling["kzz_cm2_s"] = _as_float(sampling["kzz_cm2_s"], "sampling.kzz_cm2_s")
        if sampling["kzz_cm2_s"] <= 0.0:
            raise ConfigValidationError("sampling.kzz_cm2_s must be positive.")

    config["temperature_profiles"] = _validate_temperature_profiles(config["temperature_profiles"])

    generation = config["generation"]
    _require_keys(
        generation,
        ["mode", "num_runs", "seed", "overwrite", "reuse_raw_if_present", "parallel_workers"],
        "generation",
    )
    generation["mode"] = _as_nonempty_str(generation["mode"], "generation.mode").lower()
    if generation["mode"] not in {"synthetic", "vulcan"}:
        raise ConfigValidationError("generation.mode must be 'synthetic' or 'vulcan'.")
    generation["num_runs"] = _as_int(generation["num_runs"], "generation.num_runs")
    generation["seed"] = _as_int(generation["seed"], "generation.seed")
    generation["overwrite"] = _as_bool(generation["overwrite"], "generation.overwrite")
    generation["reuse_raw_if_present"] = _as_bool(
        generation["reuse_raw_if_present"],
        "generation.reuse_raw_if_present",
    )
    generation["parallel_workers"] = _as_int(
        generation["parallel_workers"],
        "generation.parallel_workers",
    )
    if generation["num_runs"] < 1:
        raise ConfigValidationError("generation.num_runs must be >= 1.")
    if generation["parallel_workers"] < 1:
        raise ConfigValidationError("generation.parallel_workers must be >= 1.")
    backfill = generation.get("backfill", {"enabled": True, "max_retries": 3})
    if not isinstance(backfill, dict):
        raise ConfigValidationError("generation.backfill must be a mapping.")
    backfill["enabled"] = _as_bool(backfill.get("enabled", True), "generation.backfill.enabled")
    backfill["max_retries"] = _as_int(
        backfill.get("max_retries", 3), "generation.backfill.max_retries"
    )
    if backfill["max_retries"] < 0:
        raise ConfigValidationError("generation.backfill.max_retries must be >= 0.")
    generation["backfill"] = backfill

    normalization = config["normalization"]
    required_normalization_keys = [
        "split",
        "state_floor",
        "sequence_methods",
        "global_methods",
        "target_method",
    ]
    if not equilibrium:
        required_normalization_keys.extend(
            ["spectrum_floor", "spectrum_method"]
        )
    _require_keys(normalization, required_normalization_keys, "normalization")
    normalization["split"] = _validate_split(normalization["split"])
    normalization["state_floor"] = _as_float(
        normalization["state_floor"],
        "normalization.state_floor",
    )
    if normalization["state_floor"] <= 0.0:
        raise ConfigValidationError("normalization.state_floor must be positive.")
    if not isinstance(normalization["sequence_methods"], dict):
        raise ConfigValidationError("normalization.sequence_methods must be a mapping.")
    if not isinstance(normalization["global_methods"], dict):
        raise ConfigValidationError("normalization.global_methods must be a mapping.")
    if equilibrium:
        expected_sequence_methods = {"pressure_bar", "temperature_k"}
    else:
        expected_sequence_methods = {"pressure_bar", "temperature_k", "kzz_cm2_s"}
        normalization["spectrum_floor"] = _as_float(
            normalization.get("spectrum_floor"),
            "normalization.spectrum_floor",
        )
        if normalization["spectrum_floor"] <= 0.0:
            raise ConfigValidationError("normalization.spectrum_floor must be positive.")
    if set(normalization["sequence_methods"].keys()) != expected_sequence_methods:
        raise ConfigValidationError(
            f"normalization.sequence_methods must define exactly {expected_sequence_methods}."
        )
    for key, method in normalization["sequence_methods"].items():
        normalization["sequence_methods"][key] = _normalized_method_name(
            method,
            f"normalization.sequence_methods.{key}",
        )
    normalization["target_method"] = _normalized_method_name(
        normalization["target_method"],
        "normalization.target_method",
    )
    expected_global_methods = set(global_static_feature_order(config))
    if set(normalization["global_methods"].keys()) != expected_global_methods:
        raise ConfigValidationError(
            f"normalization.global_methods must define exactly {expected_global_methods}."
        )
    for key, method in normalization["global_methods"].items():
        normalization["global_methods"][key] = _normalized_method_name(
            method,
            f"normalization.global_methods.{key}",
        )
    if not equilibrium:
        normalization["spectrum_method"] = _normalized_method_name(
            normalization["spectrum_method"],
            "normalization.spectrum_method",
        )

    training = config["training"]
    _require_keys(
        training,
        [
            "seed",
            "batch_size",
            "epochs",
            "learning_rate",
            "min_lr",
            "warmup_epochs",
            "weight_decay",
            "gradient_clip",
            "loss",
        ],
        "training",
    )
    for key in ("seed", "batch_size", "epochs", "warmup_epochs"):
        training[key] = _as_int(training[key], f"training.{key}")
    for key in ("learning_rate", "min_lr", "weight_decay", "gradient_clip"):
        training[key] = _as_float(training[key], f"training.{key}")
    training["scheduler"] = _validate_training_scheduler(
        training.get("scheduler"),
        "training.scheduler",
    )
    if training["batch_size"] < 1 or training["epochs"] < 1:
        raise ConfigValidationError("training.batch_size and training.epochs must be >= 1.")
    if training["learning_rate"] <= 0.0 or training["min_lr"] <= 0.0:
        raise ConfigValidationError("Learning rates must be positive.")
    if training["min_lr"] > training["learning_rate"]:
        raise ConfigValidationError("training.min_lr cannot exceed training.learning_rate.")
    if training["gradient_clip"] <= 0.0:
        raise ConfigValidationError("training.gradient_clip must be positive.")
    loss = training["loss"]
    loss_required = ["lambda_z", "lambda_phys"]
    if not equilibrium:
        loss_required.append("lambda_spectrum")
    _require_keys(loss, loss_required, "training.loss")
    for key in loss_required:
        loss[key] = _as_float(loss[key], f"training.loss.{key}")
        if loss[key] < 0.0:
            raise ConfigValidationError(f"training.loss.{key} must be non-negative.")

    if equilibrium:
        section = config["equilibrium_only"]
        if not isinstance(section, dict):
            raise ConfigValidationError("equilibrium_only must be a mapping.")
        _require_keys(section, ["model"], "equilibrium_only")
        section["model"] = _validate_equilibrium_model_config(
            section["model"],
            "equilibrium_only.model",
        )
        config["equilibrium_only"] = section
        config["training"]["model"] = dict(section["model"])
        config["roth_sampler"] = {
            "enabled": config["temperature_profiles"]["source_mode"] in {"pt_library", "mixed"},
            "source_mode": (
                "mixed"
                if config["temperature_profiles"]["source_mode"] == "mixed"
                else "roth"
            ),
            "analytic_probability": config["temperature_profiles"].get("analytic_probability"),
            "data_glob": config["temperature_profiles"].get("data_glob", ""),
            "filters": dict(config["temperature_profiles"]["filters"]),
        }
    else:
        section = config["full_vulcan"]
        if not isinstance(section, dict):
            raise ConfigValidationError("full_vulcan must be a mapping.")
        _require_keys(
            section,
            [
                "model",
                "physics_toggles",
                "vulcan_runtime",
                "stellar_spectrum",
            ],
            "full_vulcan",
        )
        physics = section["physics_toggles"]
        for name in PUBLIC_PHYSICS_TOGGLES:
            physics[name] = _as_bool(
                physics.get(name, False),
                f"full_vulcan.physics_toggles.{name}",
            )
        runtime = section["vulcan_runtime"]
        _require_keys(
            runtime,
            [
                "chemistry_file",
                "atm_base",
                "t_cross_sp",
            ],
            "full_vulcan.vulcan_runtime",
        )
        runtime["chemistry_file"] = _as_nonempty_str(
            runtime["chemistry_file"],
            "full_vulcan.vulcan_runtime.chemistry_file",
        )
        runtime["atm_base"] = _as_nonempty_str(
            runtime["atm_base"],
            "full_vulcan.vulcan_runtime.atm_base",
        )
        if runtime["atm_base"] not in SUPPORTED_ATM_BASES:
            raise ConfigValidationError(
                "full_vulcan.vulcan_runtime.atm_base must be one of "
                f"{SUPPORTED_ATM_BASES}, got {runtime['atm_base']!r}."
            )
        runtime["t_cross_sp"] = _as_string_list(
            runtime["t_cross_sp"],
            "full_vulcan.vulcan_runtime.t_cross_sp",
        )
        runtime["python_executable"] = _as_nonempty_str(
            runtime.get("python_executable", _INTERNAL_VULCAN_RUNTIME_DEFAULTS["python_executable"]),
            "full_vulcan.vulcan_runtime.python_executable",
        )
        runtime["cfg_file"] = _as_nonempty_str(
            runtime.get("cfg_file", _INTERNAL_VULCAN_RUNTIME_DEFAULTS["cfg_file"]),
            "full_vulcan.vulcan_runtime.cfg_file",
        )
        runtime["worker_root"] = _as_nonempty_str(
            runtime.get("worker_root", _INTERNAL_VULCAN_RUNTIME_DEFAULTS["worker_root"]),
            "full_vulcan.vulcan_runtime.worker_root",
        )
        runtime["regenerate_chem_funs"] = _as_bool(
            runtime.get(
                "regenerate_chem_funs",
                _INTERNAL_VULCAN_RUNTIME_DEFAULTS["regenerate_chem_funs"],
            ),
            "full_vulcan.vulcan_runtime.regenerate_chem_funs",
        )
        cfg_assignments = runtime.get(
            "cfg_assignments",
            _INTERNAL_VULCAN_RUNTIME_DEFAULTS["cfg_assignments"],
        )
        if not isinstance(cfg_assignments, dict):
            raise ConfigValidationError(
                "full_vulcan.vulcan_runtime.cfg_assignments must be a mapping."
            )
        runtime["cfg_assignments"] = dict(cfg_assignments)
        runtime["use_lowT_limit_rates"] = _as_bool(
            runtime.get(
                "use_lowT_limit_rates",
                _INTERNAL_VULCAN_RUNTIME_DEFAULTS["use_lowT_limit_rates"],
            ),
            "full_vulcan.vulcan_runtime.use_lowT_limit_rates",
        )
        runtime["use_adaptive_rtol"] = _as_bool(
            runtime.get(
                "use_adaptive_rtol",
                _INTERNAL_VULCAN_RUNTIME_DEFAULTS["use_adaptive_rtol"],
            ),
            "full_vulcan.vulcan_runtime.use_adaptive_rtol",
        )
        spectrum = section["stellar_spectrum"]
        _require_keys(
            spectrum,
            [
                "enabled",
                "template_name",
                "template_file",
                "num_bins",
                "wavelength_min_nm",
                "wavelength_max_nm",
                "encoder_mode",
                "latent_dim",
                "hidden_dim",
                "teff_k",
                "radius_rsun",
                "semi_major_axis_au",
                "zenith_angle_deg",
                "diurnal_factor",
            ],
            "full_vulcan.stellar_spectrum",
        )
        spectrum["enabled"] = _as_bool(
            spectrum["enabled"],
            "full_vulcan.stellar_spectrum.enabled",
        )
        spectrum["template_name"] = _as_nonempty_str(
            spectrum["template_name"],
            "full_vulcan.stellar_spectrum.template_name",
        )
        spectrum["template_file"] = _as_nonempty_str(
            spectrum["template_file"],
            "full_vulcan.stellar_spectrum.template_file",
        )
        spectrum["num_bins"] = _as_int(
            spectrum["num_bins"],
            "full_vulcan.stellar_spectrum.num_bins",
        )
        spectrum["latent_dim"] = _as_int(
            spectrum["latent_dim"],
            "full_vulcan.stellar_spectrum.latent_dim",
        )
        spectrum["hidden_dim"] = _as_int(
            spectrum["hidden_dim"],
            "full_vulcan.stellar_spectrum.hidden_dim",
        )
        if spectrum["num_bins"] < 8:
            raise ConfigValidationError("full_vulcan.stellar_spectrum.num_bins must be >= 8.")
        if spectrum["latent_dim"] < 1 or spectrum["latent_dim"] > spectrum["num_bins"]:
            raise ConfigValidationError(
                "full_vulcan.stellar_spectrum.latent_dim must lie in [1, num_bins]."
            )
        spectrum["wavelength_min_nm"] = _as_float(
            spectrum["wavelength_min_nm"],
            "full_vulcan.stellar_spectrum.wavelength_min_nm",
        )
        spectrum["wavelength_max_nm"] = _as_float(
            spectrum["wavelength_max_nm"],
            "full_vulcan.stellar_spectrum.wavelength_max_nm",
        )
        if spectrum["wavelength_min_nm"] >= spectrum["wavelength_max_nm"]:
            raise ConfigValidationError(
                "full_vulcan.stellar_spectrum.wavelength_min_nm must be smaller than wavelength_max_nm."
            )
        spectrum["encoder_mode"] = _as_nonempty_str(
            spectrum["encoder_mode"],
            "full_vulcan.stellar_spectrum.encoder_mode",
        ).lower()
        if spectrum["encoder_mode"] not in _ALLOWED_SPECTRUM_ENCODERS:
            raise ConfigValidationError(
                "full_vulcan.stellar_spectrum.encoder_mode must be one of "
                f"{_ALLOWED_SPECTRUM_ENCODERS}."
            )
        for key in ("teff_k", "radius_rsun", "semi_major_axis_au", "diurnal_factor"):
            spectrum[key] = _as_float(
                spectrum[key],
                f"full_vulcan.stellar_spectrum.{key}",
            )
            if spectrum[key] <= 0.0:
                raise ConfigValidationError(
                    f"full_vulcan.stellar_spectrum.{key} must be positive."
                )
        spectrum["zenith_angle_deg"] = _as_float(
            spectrum["zenith_angle_deg"],
            "full_vulcan.stellar_spectrum.zenith_angle_deg",
        )
        if spectrum["zenith_angle_deg"] < 0.0 or spectrum["zenith_angle_deg"] >= 90.0:
            raise ConfigValidationError(
                "full_vulcan.stellar_spectrum.zenith_angle_deg must lie in [0, 90)."
            )
        section["model"] = _validate_full_vulcan_model_config(
            section["model"],
            "full_vulcan.model",
        )
        section["physics_toggles"] = physics
        section["vulcan_runtime"] = runtime
        section["stellar_spectrum"] = spectrum
        config["full_vulcan"] = section
        config["training"]["model"] = dict(section["model"])
        config["physics_toggles"] = dict(physics)
        config["vulcan_runtime"] = dict(runtime)
        config["stellar_spectrum"] = dict(spectrum)
        config["roth_sampler"] = {
            "enabled": config["temperature_profiles"]["source_mode"] in {"pt_library", "mixed"},
            "source_mode": (
                "mixed"
                if config["temperature_profiles"]["source_mode"] == "mixed"
                else "roth"
            ),
            "analytic_probability": config["temperature_profiles"].get("analytic_probability"),
            "data_glob": config["temperature_profiles"].get("data_glob", ""),
            "filters": dict(config["temperature_profiles"]["filters"]),
        }

    config["training"] = {
        "seed": training["seed"],
        "batch_size": training["batch_size"],
        "epochs": training["epochs"],
        "learning_rate": training["learning_rate"],
        "min_lr": training["min_lr"],
        "warmup_epochs": training["warmup_epochs"],
        "scheduler": training["scheduler"],
        "weight_decay": training["weight_decay"],
        "gradient_clip": training["gradient_clip"],
        "model": training["model"],
        "loss": training["loss"],
    }

    config["data_spec"]["state_dim"] = len(state_species)
    config["data_spec"]["target_dim"] = len(output_species)
    if equilibrium:
        config["data_spec"]["sequence_static_feature_order"] = [
            "pressure_bar",
            "temperature_k",
        ]
    else:
        config["data_spec"]["sequence_static_feature_order"] = [
            "pressure_bar",
            "temperature_k",
            "kzz_cm2_s",
        ]
    config["data_spec"]["global_static_feature_order"] = global_static_feature_order(config)
    config["data_spec"]["global_feature_order"] = global_feature_order(config)
    return config
