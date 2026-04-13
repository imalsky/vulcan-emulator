"""Configuration loading, validation, and normalization for emulator configs.

This module is the single source of truth for config schema validation.  It:

1. Loads a JSON config file from disk.
2. Validates every field against its expected type, range, and chemistry/model
   constraints.
3. Derives internal aliases (``training.model``, ``roth_sampler``)
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
CHEMISTRY_TYPES = ("fastchem", "vulcan")
MODEL_TYPES = ("mlp", "transformer")
ELEMENT_INPUT_ORDER = ("He_H", "C_H", "O_H", "N_H", "S_H")
FASTCHEM_CONDITIONING_INPUT_ORDER = ELEMENT_INPUT_ORDER
VULCAN_CONDITIONING_INPUT_ORDER = (
    "gravity_cm_s2",
    "planet_radius_cm",
    *FASTCHEM_CONDITIONING_INPUT_ORDER,
)
VULCAN_CORE_GLOBAL_INPUTS = VULCAN_CONDITIONING_INPUT_ORDER
FASTCHEM_CORE_GLOBAL_INPUTS = FASTCHEM_CONDITIONING_INPUT_ORDER
VULCAN_STELLAR_GLOBAL_INPUTS = (
    "r_star_rsun",
    "semi_major_axis_au",
    "zenith_angle_deg",
    "diurnal_factor",
)
VULCAN_OPTIONAL_GLOBAL_INPUTS = (
    *PUBLIC_PHYSICS_TOGGLES,
    *tuple(f"atm_base_{name}" for name in SUPPORTED_ATM_BASES),
)
DEFAULT_REQUIRED_GLOBAL_INPUTS = (
    *VULCAN_CORE_GLOBAL_INPUTS,
    *VULCAN_STELLAR_GLOBAL_INPUTS,
    *VULCAN_OPTIONAL_GLOBAL_INPUTS,
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
_ALLOWED_SPECTRUM_ENCODERS = {"perceiver", "none"}
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
_ALLOWED_NORMALIZATION_METHODS = {"standard", "log-standard", "log-minmax", "none"}
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
    "task.kind": (
        "task.kind has been removed. Use top-level chemistry_type and model_type instead."
    ),
    "equilibrium_only": (
        "equilibrium_only has been removed. Use top-level model plus chemistry_type='fastchem'."
    ),
    "full_vulcan": (
        "full_vulcan has been removed. Use top-level model plus a top-level vulcan block."
    ),
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
        "generation.target_mode has been removed. chemistry_type now determines the "
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
        "full_vulcan.trajectory_sampling has been removed. The supported VULCAN "
        "task is final-state-only."
    ),
    "vulcan.trajectory_sampling": (
        "vulcan.trajectory_sampling has been removed. The supported VULCAN "
        "task is final-state-only."
    ),
    "vulcan.stellar_spectrum.enabled": (
        "vulcan.stellar_spectrum.enabled has been removed. The stellar spectrum "
        "encoder is always active when a vulcan block is present."
    ),
    "vulcan.stellar_spectrum.num_bins": (
        "vulcan.stellar_spectrum.num_bins has been removed. Use max_tokens instead."
    ),
    "normalization.spectrum_method": (
        "normalization.spectrum_method has been removed. Spectra are normalized "
        "on-the-fly inside the encoder."
    ),
    "training.loss.lambda_spectrum": (
        "training.loss.lambda_spectrum has been removed. Spectrum loss is no "
        "longer a separate training objective."
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
    "rocky": False,
    "top_bc_flux_file": None,
    "bot_bc_flux_file": None,
}
_DEFAULT_SCIENCE_PRESET_NAME = "default"
_LEGACY_VULCAN_STELLAR_GEOMETRY_ALIASES = {
    "radius_rsun": "stellar_radius_range_rsun",
    "semi_major_axis_au": "semi_major_axis_range_au",
    "zenith_angle_deg": "zenith_angle_range_deg",
    "diurnal_factor": "diurnal_factor_range",
}


class ConfigValidationError(ValueError):
    """Raised when a configuration file violates the required contract."""


def _require_keys(mapping: dict[str, Any], keys: tuple[str, ...] | list[str], scope: str) -> None:
    """Validate that a config subsection contains every required key.

    Parameters
    ----------
    mapping : dict[str, Any]
        Config subsection to validate.
    keys : tuple[str, ...] or list[str]
        Required keys that must appear in ``mapping``.
    scope : str
        Human-readable config scope used in validation errors.

    Returns
    -------
    None
        The function returns silently when all keys are present.

    Raises
    ------
    ConfigValidationError
        If one or more required keys are missing.
    """
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigValidationError(f"Missing required keys in {scope}: {missing}")


def _nested_key_present(mapping: dict[str, Any], path: str) -> bool:
    """Return whether a dotted config path exists in a nested mapping.

    Parameters
    ----------
    mapping : dict[str, Any]
        Root user-config payload.
    path : str
        Dotted key path such as ``"training.scheduler.name"``.

    Returns
    -------
    bool
        ``True`` when every component of the path exists in the nested
        mapping structure.
    """
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def _reject_removed_legacy_keys(config: dict[str, Any]) -> None:
    """Reject deprecated config keys that now require explicit migration.

    Parameters
    ----------
    config : dict[str, Any]
        Full user configuration payload before normalization.

    Returns
    -------
    None
        The function returns silently when no removed legacy keys are present.

    Raises
    ------
    ConfigValidationError
        If a removed dotted config path still exists in ``config``.
    """
    for path, message in _REMOVED_LEGACY_KEYS.items():
        if _nested_key_present(config, path):
            raise ConfigValidationError(message)


def _as_bool(value: Any, field: str) -> bool:
    """Validate and return one boolean config field.

    Parameters
    ----------
    value : Any
        Raw config value to validate.
    field : str
        Field name used in validation errors.

    Returns
    -------
    bool
        Validated boolean value.
    """
    if not isinstance(value, bool):
        raise ConfigValidationError(f"{field} must be a boolean.")
    return value


def _as_int(value: Any, field: str) -> int:
    """Validate and return one integer config field.

    Parameters
    ----------
    value : Any
        Raw config value to validate.
    field : str
        Field name used in validation errors.

    Returns
    -------
    int
        Validated integer value.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(f"{field} must be an integer.")
    return int(value)


def _as_float(value: Any, field: str) -> float:
    """Validate and return one numeric config field as a float.

    Parameters
    ----------
    value : Any
        Raw config value to validate.
    field : str
        Field name used in validation errors.

    Returns
    -------
    float
        Validated numeric value converted to ``float``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(f"{field} must be numeric.")
    return float(value)


def _as_nonempty_str(value: Any, field: str) -> str:
    """Validate and return one non-empty string config field.

    Parameters
    ----------
    value : Any
        Raw config value to validate.
    field : str
        Field name used in validation errors.

    Returns
    -------
    str
        Stripped non-empty string value.
    """
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"{field} must be a non-empty string.")
    return value.strip()


def _as_string_list(value: Any, field: str) -> list[str]:
    """Validate and normalize a deduplicated list of non-empty strings.

    Parameters
    ----------
    value : Any
        Candidate list value from the config payload.
    field : str
        Fully qualified config field name used in validation errors.

    Returns
    -------
    list[str]
        Normalized string list with leading/trailing whitespace removed.
    """
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
    """Validate and normalize a named method from the config payload.

    Parameters
    ----------
    value : Any
        Candidate method name from the config payload.
    field : str
        Fully qualified config field name used in validation errors.
    allowed : set[str] or None, optional
        Explicit method-name allow-list. When omitted, the shared
        normalization-method set is used.

    Returns
    -------
    str
        Lower-cased validated method name.
    """
    method_name = _as_nonempty_str(value, field).lower()
    allowed_methods = _ALLOWED_NORMALIZATION_METHODS if allowed is None else allowed
    if method_name not in allowed_methods:
        raise ConfigValidationError(
            f"{field} must be one of {sorted(allowed_methods)}, got {method_name!r}."
        )
    return method_name


def get_chemistry_type(config: dict[str, Any]) -> str:
    """Return the validated chemistry target family from the config.

    Parameters
    ----------
    config : dict[str, Any]
        Top-level config payload.

    Returns
    -------
    str
        Lower-cased chemistry type, either ``"fastchem"`` or ``"vulcan"``.
    """
    chemistry_type = _as_nonempty_str(
        config.get("chemistry_type"),
        "chemistry_type",
    ).lower()
    if chemistry_type not in CHEMISTRY_TYPES:
        raise ConfigValidationError(
            f"chemistry_type must be one of {CHEMISTRY_TYPES}, got {chemistry_type!r}."
        )
    return chemistry_type


def get_model_type(config: dict[str, Any]) -> str:
    """Return the validated prediction architecture family from the config.

    Parameters
    ----------
    config : dict[str, Any]
        Top-level config payload.

    Returns
    -------
    str
        Lower-cased model type, either ``"mlp"`` or ``"transformer"``.
    """
    model_type = _as_nonempty_str(config.get("model_type"), "model_type").lower()
    if model_type in {"equilibrium", "full_vulcan"}:
        raise ConfigValidationError(
            "Legacy model_type values 'equilibrium' and 'full_vulcan' are no longer "
            "supported. Use model_type='mlp' or model_type='transformer'."
        )
    if model_type not in MODEL_TYPES:
        raise ConfigValidationError(
            f"model_type must be one of {MODEL_TYPES}, got {model_type!r}."
        )
    return model_type


def uses_fastchem(config: dict[str, Any]) -> bool:
    """Report whether the config targets FastChem equilibrium chemistry.

    Parameters
    ----------
    config : dict[str, Any]
        Validated pipeline config.

    Returns
    -------
    bool
        ``True`` when ``chemistry_type`` resolves to ``"fastchem"``.
    """
    return get_chemistry_type(config) == "fastchem"


def uses_vulcan_chemistry(config: dict[str, Any]) -> bool:
    """Report whether the config targets converged VULCAN chemistry outputs.

    Parameters
    ----------
    config : dict[str, Any]
        Validated pipeline config.

    Returns
    -------
    bool
        ``True`` when ``chemistry_type`` resolves to ``"vulcan"``.
    """
    return get_chemistry_type(config) == "vulcan"


def uses_mlp(config: dict[str, Any]) -> bool:
    """Report whether the config selects the FiLM-conditioned MLP model.

    Parameters
    ----------
    config : dict[str, Any]
        Validated pipeline config.

    Returns
    -------
    bool
        ``True`` when ``model_type`` resolves to ``"mlp"``.
    """
    return get_model_type(config) == "mlp"


def uses_transformer(config: dict[str, Any]) -> bool:
    """Report whether the config selects the FiLM-conditioned Transformer.

    Parameters
    ----------
    config : dict[str, Any]
        Validated pipeline config.

    Returns
    -------
    bool
        ``True`` when ``model_type`` resolves to ``"transformer"``.
    """
    return get_model_type(config) == "transformer"


def _validate_science_presets(
    presets: Any,
    *,
    default_physics: dict[str, bool],
    default_atm_base: str,
    scope: str,
) -> list[dict[str, Any]]:
    """Validate curated full-VULCAN science presets.

    Parameters
    ----------
    presets : Any
        User-provided preset payload or ``None``.
    default_physics : dict[str, bool]
        Default public physics toggles used to backfill omitted presets.
    default_atm_base : str
        Default atmosphere base used to backfill omitted presets.
    scope : str
        Config scope label used in validation errors.

    Returns
    -------
    list[dict[str, Any]]
        Normalized preset dictionaries containing ``name``, ``atm_base``, and
        ``physics_toggles``.
    """
    if presets is None:
        return [
            {
                "name": _DEFAULT_SCIENCE_PRESET_NAME,
                "atm_base": default_atm_base,
                "physics_toggles": dict(default_physics),
            }
        ]
    if not isinstance(presets, list) or not presets:
        raise ConfigValidationError(f"{scope} must be a non-empty list when provided.")

    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for idx, raw_preset in enumerate(presets):
        preset_scope = f"{scope}[{idx}]"
        if not isinstance(raw_preset, dict):
            raise ConfigValidationError(f"{preset_scope} must be a mapping.")
        name = _as_nonempty_str(raw_preset.get("name"), f"{preset_scope}.name")
        if name in seen_names:
            raise ConfigValidationError(f"{scope} contains duplicate preset name {name!r}.")
        seen_names.add(name)
        atm_base = _as_nonempty_str(
            raw_preset.get("atm_base", default_atm_base),
            f"{preset_scope}.atm_base",
        )
        if atm_base not in SUPPORTED_ATM_BASES:
            raise ConfigValidationError(
                f"{preset_scope}.atm_base must be one of {SUPPORTED_ATM_BASES}, got {atm_base!r}."
            )
        physics_overrides = raw_preset.get("physics_toggles", {})
        if not isinstance(physics_overrides, dict):
            raise ConfigValidationError(f"{preset_scope}.physics_toggles must be a mapping.")
        physics = dict(default_physics)
        for toggle_name, toggle_value in physics_overrides.items():
            if toggle_name not in PUBLIC_PHYSICS_TOGGLES:
                raise ConfigValidationError(
                    f"{preset_scope}.physics_toggles.{toggle_name} is not a supported public toggle."
                )
            physics[toggle_name] = _as_bool(
                toggle_value,
                f"{preset_scope}.physics_toggles.{toggle_name}",
            )
        normalized.append(
            {
                "name": name,
                "atm_base": atm_base,
                "physics_toggles": physics,
            }
        )
    return normalized


def resolve_conditioning_inputs(
    *,
    raw_global_inputs: dict[str, float],
    required_global_inputs: list[str],
) -> dict[str, float]:
    """Resolve and validate the required per-run conditioning inputs.

    Parameters
    ----------
    raw_global_inputs : dict[str, float]
        Raw globals mapping loaded from a run file.
    required_global_inputs : list[str]
        Ordered conditioning-input names required by the active data contract.

    Returns
    -------
    dict[str, float]
        Reduced mapping containing exactly the required conditioning inputs in
        Python float form.
    """
    resolved: dict[str, float] = {}
    for name in required_global_inputs:
        if name in raw_global_inputs:
            resolved[name] = float(raw_global_inputs[name])
        else:
            raise ConfigValidationError(
                f"Missing required conditioning input {name!r} in raw globals."
            )
    return resolved


def global_static_feature_order(config: dict[str, Any]) -> list[str]:
    """Return the ordered list of global conditioning feature names.

    Parameters
    ----------
    config : dict[str, Any]
        Validated pipeline config whose ``data_spec`` section defines the
        required global inputs.

    Returns
    -------
    list[str]
        Ordered global feature names used by preprocessing and model I/O.
    """
    return list(config["data_spec"]["required_global_inputs"])


def global_feature_order(config: dict[str, Any]) -> list[str]:
    """Return the full ordered global feature vector contract for the model.

    Parameters
    ----------
    config : dict[str, Any]
        Validated pipeline config whose ``data_spec`` section defines the
        required global inputs.

    Returns
    -------
    list[str]
        Ordered global feature names. In the current contract this matches
        :func:`global_static_feature_order`.
    """
    return list(config["data_spec"]["required_global_inputs"])


def dataset_raw_root(config: dict[str, Any]) -> str:
    """Return the configured raw-data root for the active dataset."""
    return str(config["paths"]["raw_root"])


def dataset_run_root(config: dict[str, Any]) -> str:
    """Return the shared dataset run root containing raw, processed, and info."""
    return str(Path(config["paths"]["raw_root"]).parent)


def dataset_info_root(config: dict[str, Any]) -> str:
    """Return the shared dataset metadata directory adjacent to raw/processed."""
    return str(Path(config["paths"]["raw_root"]).parent / "info")


def _validate_mlp_model_config(model: dict[str, Any], scope: str) -> dict[str, Any]:
    """Validate and normalize the FiLM-MLP model config section.

    Parameters
    ----------
    model : dict[str, Any]
        Raw model config block for the MLP architecture.
    scope : str
        Fully qualified config scope used in validation errors.

    Returns
    -------
    dict[str, Any]
        Normalized MLP config with concrete numeric types and defaults
        applied.
    """
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
    normalized["dropout_rate"] = _as_float(
        normalized.get("dropout_rate", 0.05),
        f"{scope}.dropout_rate",
    )
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
    if not 0.0 <= normalized["dropout_rate"] < 1.0:
        raise ConfigValidationError(f"{scope}.dropout_rate must be in [0, 1).")
    return normalized


def _validate_transformer_model_config(model: dict[str, Any], scope: str) -> dict[str, Any]:
    """Validate and normalize the FiLM-Transformer config section.

    Parameters
    ----------
    model : dict[str, Any]
        Raw model config block for the Transformer architecture.
    scope : str
        Fully qualified config scope used in validation errors.

    Returns
    -------
    dict[str, Any]
        Normalized Transformer config with concrete numeric types and
        defaults applied.
    """
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
    normalized["dropout_rate"] = _as_float(
        normalized.get("dropout_rate", 0.05),
        f"{scope}.dropout_rate",
    )
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
    if not 0.0 <= normalized["dropout_rate"] < 1.0:
        raise ConfigValidationError(f"{scope}.dropout_rate must be in [0, 1).")
    return normalized


def _validate_training_scheduler(scheduler: Any, scope: str) -> dict[str, Any]:
    """Validate and normalize the optional training-scheduler block.

    Parameters
    ----------
    scheduler : Any
        Raw scheduler payload, or ``None`` to accept defaults.
    scope : str
        Fully qualified config scope used in validation errors.

    Returns
    -------
    dict[str, Any]
        Normalized scheduler payload for either cosine decay or
        reduce-on-plateau scheduling.
    """
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
    """Validate a two-number range field and return normalized bounds.

    Parameters
    ----------
    value : Any
        Candidate two-element range payload.
    field : str
        Fully qualified config field name used in validation errors.
    allow_equal : bool, default=False
        Whether equal lower and upper bounds are permitted.

    Returns
    -------
    list[float]
        Normalized ``[lower, upper]`` bounds as floats.
    """
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
    """Validate a normal-distribution config block.

    Parameters
    ----------
    spec : Any
        Candidate mapping containing ``mean`` and ``std``.
    field : str
        Fully qualified config field name used in validation errors.

    Returns
    -------
    dict[str, float]
        Normalized distribution specification with float ``mean`` and ``std``.
    """
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
    """Validate the shared temperature-profile bounds block.

    Parameters
    ----------
    validation_config : Any
        Candidate mapping containing minimum and maximum allowed
        temperatures.
    scope : str
        Parent config scope used to construct error messages.

    Returns
    -------
    dict[str, float]
        Normalized validation block with ``min_temperature_k`` and
        ``max_temperature_k``.
    """
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

    Parameters
    ----------
    sampler_config : Any
        Raw config payload for ``temperature_profiles.analytic_sampler``.

    Returns
    -------
    dict[str, Any]
        Normalized analytic-sampler config with validated numeric ranges and
        distribution specifications.
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

    Parameters
    ----------
    profile_config : Any
        Raw ``temperature_profiles`` config payload.

    Returns
    -------
    dict[str, Any]
        Normalized temperature-profile config containing validated source-mode,
        filters, analytic sampler settings, and shared validity bounds.
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
    """Validate the train/val/test split policy under ``normalization``.

    Parameters
    ----------
    split : Any
        Candidate split mapping with fractions and RNG seed.

    Returns
    -------
    dict[str, Any]
        Normalized split payload with numeric fractions and integer seed.
    """
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


def _seed_vulcan_sampling_geometry_ranges(config: dict[str, Any]) -> None:
    """Backfill new sampling geometry ranges from legacy fixed config keys.

    Older VULCAN configs stored irradiation geometry under
    ``vulcan.stellar_spectrum`` as fixed scalars. The current contract samples
    those values per run from ``sampling`` ranges instead. This helper mirrors
    any legacy fixed values into equal-endpoint sampling ranges before the main
    sampling schema is validated.
    """
    if config.get("chemistry_type") != "vulcan":
        return
    sampling = config.get("sampling")
    vulcan = config.get("vulcan")
    if not isinstance(sampling, dict) or not isinstance(vulcan, dict):
        return
    spectrum = vulcan.get("stellar_spectrum")
    if not isinstance(spectrum, dict):
        return
    for legacy_key, sampling_key in _LEGACY_VULCAN_STELLAR_GEOMETRY_ALIASES.items():
        if sampling_key not in sampling and legacy_key in spectrum:
            sampling[sampling_key] = [spectrum[legacy_key], spectrum[legacy_key]]


def load_and_validate_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config file, apply defaults, and validate its full contract.

    This is the primary entry point for config loading.  It reads the JSON
    file, validates every section (paths, data_spec, sampling,
    temperature_profiles, generation, normalization, training, model, and the
    optional chemistry-specific block), derives internal aliases, and returns
    a fully normalized dict.

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

    required_root = [
        "chemistry_type",
        "model_type",
        "paths",
        "data_spec",
        "sampling",
        "temperature_profiles",
        "generation",
        "normalization",
        "training",
        "model",
    ]
    _require_keys(config, required_root, "root")
    chemistry_type = get_chemistry_type(config)
    model_type = get_model_type(config)
    if "task" in config:
        raise ConfigValidationError(
            "task has been removed. Use top-level chemistry_type and model_type instead."
        )
    if chemistry_type == "fastchem" and "vulcan" in config:
        raise ConfigValidationError(
            "vulcan must not be defined when chemistry_type='fastchem'."
        )
    if chemistry_type == "vulcan":
        _require_keys(config, ["vulcan"], "root")
    config["chemistry_type"] = chemistry_type
    config["model_type"] = model_type
    _seed_vulcan_sampling_geometry_ranges(config)

    paths = config["paths"]
    if "raw_root" in paths or "processed_root" in paths:
        raise ConfigValidationError(
            "paths.raw_root and paths.processed_root are no longer supported. "
            "Use run_root only (e.g. 'data/fastchem_mlp'). The validator auto-expands "
            "it to run_root/raw and run_root/processed."
        )
    _require_keys(paths, ["run_root", "checkpoints_root", "vulcan_source_root"], "paths")
    for key in paths:
        paths[key] = _as_nonempty_str(paths[key], f"paths.{key}")
    run_root = paths.pop("run_root")
    paths["raw_root"] = str(Path(run_root) / "raw")
    paths["processed_root"] = str(Path(run_root) / "processed")

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
        list(FASTCHEM_CORE_GLOBAL_INPUTS)
        if chemistry_type == "fastchem"
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
    if chemistry_type == "vulcan":
        required_sampling += [
            "gravity_range_cm_s2",
            "planet_radius_range_cm",
            "stellar_radius_range_rsun",
            "semi_major_axis_range_au",
            "zenith_angle_range_deg",
            "diurnal_factor_range",
            "kzz_cm2_s",
        ]
    _require_keys(sampling, required_sampling, "sampling")
    if chemistry_type == "fastchem":
        disallowed_sampling = [
            key for key in ("gravity_range_cm_s2", "planet_radius_range_cm", "kzz_cm2_s")
            if key in sampling
        ]
        if disallowed_sampling:
            raise ConfigValidationError(
                "sampling contains VULCAN-only keys for chemistry_type='fastchem': "
                f"{disallowed_sampling}"
            )
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
    if chemistry_type == "vulcan":
        range_keys.extend(
            [
                "gravity_range_cm_s2",
                "planet_radius_range_cm",
                "stellar_radius_range_rsun",
                "semi_major_axis_range_au",
                "zenith_angle_range_deg",
                "diurnal_factor_range",
            ]
        )
    allow_equal_range_keys = {
        "stellar_radius_range_rsun",
        "semi_major_axis_range_au",
        "zenith_angle_range_deg",
        "diurnal_factor_range",
    }
    for key in range_keys:
        values = sampling[key]
        if not isinstance(values, list) or len(values) != 2:
            raise ConfigValidationError(f"sampling.{key} must be a length-2 list.")
        low = _as_float(values[0], f"sampling.{key}[0]")
        high = _as_float(values[1], f"sampling.{key}[1]")
        if key in allow_equal_range_keys:
            if low > high:
                raise ConfigValidationError(f"sampling.{key} must be non-decreasing.")
        elif not low < high:
            raise ConfigValidationError(f"sampling.{key} must be strictly increasing.")
        sampling[key] = [low, high]
    if chemistry_type == "vulcan":
        for key in (
            "gravity_range_cm_s2",
            "planet_radius_range_cm",
            "stellar_radius_range_rsun",
            "semi_major_axis_range_au",
            "diurnal_factor_range",
        ):
            if sampling[key][0] <= 0.0:
                raise ConfigValidationError(f"sampling.{key} must be strictly positive.")
        if sampling["zenith_angle_range_deg"][0] < 0.0 or sampling["zenith_angle_range_deg"][1] >= 90.0:
            raise ConfigValidationError(
                "sampling.zenith_angle_range_deg must lie within [0, 90)."
            )
    if chemistry_type == "vulcan":
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
    if chemistry_type == "vulcan":
        required_normalization_keys.append("spectrum_floor")
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
    if chemistry_type == "fastchem":
        expected_sequence_methods = {"pressure_bar", "temperature_k"}
    else:
        expected_sequence_methods = {"pressure_bar", "temperature_k", "kzz_cm2_s"}
        normalization["spectrum_floor"] = _as_float(
            normalization.get("spectrum_floor"),
            "normalization.spectrum_floor",
        )
        if normalization["spectrum_floor"] <= 0.0:
            raise ConfigValidationError("normalization.spectrum_floor must be positive.")
        if "spectrum_method" in normalization and normalization["spectrum_method"] is not None:
            normalization["spectrum_method"] = _normalized_method_name(
                normalization["spectrum_method"],
                "normalization.spectrum_method",
            )
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
    training["early_stopping_patience"] = _as_int(
        training.get("early_stopping_patience", 30),
        "training.early_stopping_patience",
    )
    for key in ("learning_rate", "min_lr", "weight_decay", "gradient_clip"):
        training[key] = _as_float(training[key], f"training.{key}")
    training["scheduler"] = _validate_training_scheduler(
        training.get("scheduler"),
        "training.scheduler",
    )
    if training["batch_size"] < 1 or training["epochs"] < 1:
        raise ConfigValidationError("training.batch_size and training.epochs must be >= 1.")
    if training["early_stopping_patience"] < 1:
        raise ConfigValidationError("training.early_stopping_patience must be >= 1.")
    if training["learning_rate"] <= 0.0 or training["min_lr"] <= 0.0:
        raise ConfigValidationError("Learning rates must be positive.")
    if training["min_lr"] > training["learning_rate"]:
        raise ConfigValidationError("training.min_lr cannot exceed training.learning_rate.")
    if training["gradient_clip"] <= 0.0:
        raise ConfigValidationError("training.gradient_clip must be positive.")
    loss = training["loss"]
    loss_required = ["lambda_z", "lambda_phys"]
    _require_keys(loss, loss_required, "training.loss")
    for key in loss_required:
        loss[key] = _as_float(loss[key], f"training.loss.{key}")
        if loss[key] < 0.0:
            raise ConfigValidationError(f"training.loss.{key} must be non-negative.")
    if "lambda_spectrum" in loss:
        loss["lambda_spectrum"] = _as_float(
            loss["lambda_spectrum"],
            "training.loss.lambda_spectrum",
        )
        if loss["lambda_spectrum"] < 0.0:
            raise ConfigValidationError("training.loss.lambda_spectrum must be non-negative.")
    else:
        loss["lambda_spectrum"] = 0.0

    model = config["model"]

    if not isinstance(model, dict):
        raise ConfigValidationError("model must be a mapping.")
    if model_type == "mlp":
        model = _validate_mlp_model_config(model, "model")
    else:
        model = _validate_transformer_model_config(model, "model")
    config["model"] = model
    config["training"]["model"] = dict(model)

    if chemistry_type == "vulcan":
        section = config["vulcan"]
        if not isinstance(section, dict):
            raise ConfigValidationError("vulcan must be a mapping.")
        _require_keys(
            section,
            [
                "physics_toggles",
                "runtime",
                "stellar_spectrum",
            ],
            "vulcan",
        )
        physics = section["physics_toggles"]
        if not isinstance(physics, dict):
            raise ConfigValidationError("vulcan.physics_toggles must be a mapping.")
        for name in PUBLIC_PHYSICS_TOGGLES:
            physics[name] = _as_bool(
                physics.get(name, False),
                f"vulcan.physics_toggles.{name}",
            )
        runtime = section["runtime"]
        _require_keys(
            runtime,
            [
                "chemistry_file",
                "t_cross_sp",
            ],
            "vulcan.runtime",
        )
        runtime["chemistry_file"] = _as_nonempty_str(
            runtime["chemistry_file"],
            "vulcan.runtime.chemistry_file",
        )
        runtime["t_cross_sp"] = _as_string_list(
            runtime["t_cross_sp"],
            "vulcan.runtime.t_cross_sp",
        )
        runtime["python_executable"] = _as_nonempty_str(
            runtime.get("python_executable", _INTERNAL_VULCAN_RUNTIME_DEFAULTS["python_executable"]),
            "vulcan.runtime.python_executable",
        )
        runtime["cfg_file"] = _as_nonempty_str(
            runtime.get("cfg_file", _INTERNAL_VULCAN_RUNTIME_DEFAULTS["cfg_file"]),
            "vulcan.runtime.cfg_file",
        )
        runtime["worker_root"] = _as_nonempty_str(
            runtime.get("worker_root", _INTERNAL_VULCAN_RUNTIME_DEFAULTS["worker_root"]),
            "vulcan.runtime.worker_root",
        )
        runtime["regenerate_chem_funs"] = _as_bool(
            runtime.get(
                "regenerate_chem_funs",
                _INTERNAL_VULCAN_RUNTIME_DEFAULTS["regenerate_chem_funs"],
            ),
            "vulcan.runtime.regenerate_chem_funs",
        )
        cfg_assignments = runtime.get(
            "cfg_assignments",
            _INTERNAL_VULCAN_RUNTIME_DEFAULTS["cfg_assignments"],
        )
        if not isinstance(cfg_assignments, dict):
            raise ConfigValidationError("vulcan.runtime.cfg_assignments must be a mapping.")
        runtime["cfg_assignments"] = dict(cfg_assignments)
        runtime["use_lowT_limit_rates"] = _as_bool(
            runtime.get(
                "use_lowT_limit_rates",
                _INTERNAL_VULCAN_RUNTIME_DEFAULTS["use_lowT_limit_rates"],
            ),
            "vulcan.runtime.use_lowT_limit_rates",
        )
        runtime["use_adaptive_rtol"] = _as_bool(
            runtime.get(
                "use_adaptive_rtol",
                _INTERNAL_VULCAN_RUNTIME_DEFAULTS["use_adaptive_rtol"],
            ),
            "vulcan.runtime.use_adaptive_rtol",
        )
        runtime["rocky"] = _as_bool(
            runtime.get("rocky", _INTERNAL_VULCAN_RUNTIME_DEFAULTS["rocky"]),
            "vulcan.runtime.rocky",
        )
        top_bc_flux_file = runtime.get(
            "top_bc_flux_file",
            _INTERNAL_VULCAN_RUNTIME_DEFAULTS["top_bc_flux_file"],
        )
        runtime["top_bc_flux_file"] = (
            None
            if top_bc_flux_file is None
            else _as_nonempty_str(top_bc_flux_file, "vulcan.runtime.top_bc_flux_file")
        )
        bot_bc_flux_file = runtime.get(
            "bot_bc_flux_file",
            _INTERNAL_VULCAN_RUNTIME_DEFAULTS["bot_bc_flux_file"],
        )
        runtime["bot_bc_flux_file"] = (
            None
            if bot_bc_flux_file is None
            else _as_nonempty_str(bot_bc_flux_file, "vulcan.runtime.bot_bc_flux_file")
        )
        default_atm_base = _as_nonempty_str(
            runtime.get("atm_base", "H2"),
            "vulcan.runtime.atm_base",
        )
        if default_atm_base not in SUPPORTED_ATM_BASES:
            raise ConfigValidationError(
                f"vulcan.runtime.atm_base must be one of {SUPPORTED_ATM_BASES}, got {default_atm_base!r}."
            )
        runtime["atm_base"] = default_atm_base
        science_presets = _validate_science_presets(
            section.get("science_presets"),
            default_physics=dict(physics),
            default_atm_base=default_atm_base,
            scope="vulcan.science_presets",
        )

        spectrum = section["stellar_spectrum"]
        _require_keys(
            spectrum,
            [
                "template_name",
                "template_file",
                "max_tokens",
                "wavelength_min_nm",
                "wavelength_max_nm",
                "encoder_mode",
                "latent_dim",
                "hidden_dim",
                "num_latents",
                "num_layers",
                "num_heads",
                "fourier_features",
            ],
            "vulcan.stellar_spectrum",
        )
        spectrum["template_name"] = _as_nonempty_str(
            spectrum["template_name"],
            "vulcan.stellar_spectrum.template_name",
        )
        spectrum["template_file"] = _as_nonempty_str(
            spectrum["template_file"],
            "vulcan.stellar_spectrum.template_file",
        )
        library_glob = spectrum.get("library_glob")
        if library_glob is None:
            spectrum["library_glob"] = None
        else:
            spectrum["library_glob"] = _as_nonempty_str(
                library_glob,
                "vulcan.stellar_spectrum.library_glob",
            )
        spectrum["max_tokens"] = _as_int(
            spectrum["max_tokens"],
            "vulcan.stellar_spectrum.max_tokens",
        )
        spectrum["latent_dim"] = _as_int(
            spectrum["latent_dim"],
            "vulcan.stellar_spectrum.latent_dim",
        )
        spectrum["hidden_dim"] = _as_int(
            spectrum["hidden_dim"],
            "vulcan.stellar_spectrum.hidden_dim",
        )
        spectrum["num_latents"] = _as_int(
            spectrum["num_latents"],
            "vulcan.stellar_spectrum.num_latents",
        )
        spectrum["num_layers"] = _as_int(
            spectrum["num_layers"],
            "vulcan.stellar_spectrum.num_layers",
        )
        spectrum["num_heads"] = _as_int(
            spectrum["num_heads"],
            "vulcan.stellar_spectrum.num_heads",
        )
        spectrum["fourier_features"] = _as_int(
            spectrum["fourier_features"],
            "vulcan.stellar_spectrum.fourier_features",
        )
        if spectrum["max_tokens"] < 8:
            raise ConfigValidationError("vulcan.stellar_spectrum.max_tokens must be >= 8.")
        if spectrum["latent_dim"] < 1:
            raise ConfigValidationError("vulcan.stellar_spectrum.latent_dim must be >= 1.")
        if spectrum["hidden_dim"] < 8:
            raise ConfigValidationError("vulcan.stellar_spectrum.hidden_dim must be >= 8.")
        if spectrum["num_latents"] < 1:
            raise ConfigValidationError("vulcan.stellar_spectrum.num_latents must be >= 1.")
        if spectrum["num_layers"] < 1:
            raise ConfigValidationError("vulcan.stellar_spectrum.num_layers must be >= 1.")
        if spectrum["num_heads"] < 1:
            raise ConfigValidationError("vulcan.stellar_spectrum.num_heads must be >= 1.")
        if spectrum["hidden_dim"] % spectrum["num_heads"] != 0:
            raise ConfigValidationError(
                "vulcan.stellar_spectrum.hidden_dim must be divisible by num_heads."
            )
        if spectrum["fourier_features"] < 1:
            raise ConfigValidationError("vulcan.stellar_spectrum.fourier_features must be >= 1.")
        spectrum["wavelength_min_nm"] = _as_float(
            spectrum["wavelength_min_nm"],
            "vulcan.stellar_spectrum.wavelength_min_nm",
        )
        spectrum["wavelength_max_nm"] = _as_float(
            spectrum["wavelength_max_nm"],
            "vulcan.stellar_spectrum.wavelength_max_nm",
        )
        if spectrum["wavelength_min_nm"] >= spectrum["wavelength_max_nm"]:
            raise ConfigValidationError(
                "vulcan.stellar_spectrum.wavelength_min_nm must be smaller than wavelength_max_nm."
            )
        spectrum.setdefault("dbin1_nm", 0.1)
        spectrum["dbin1_nm"] = _as_float(
            spectrum["dbin1_nm"],
            "vulcan.stellar_spectrum.dbin1_nm",
        )
        spectrum.setdefault("dbin2_nm", 2.0)
        spectrum["dbin2_nm"] = _as_float(
            spectrum["dbin2_nm"],
            "vulcan.stellar_spectrum.dbin2_nm",
        )
        spectrum.setdefault("dbin_12trans_nm", 240.0)
        spectrum["dbin_12trans_nm"] = _as_float(
            spectrum["dbin_12trans_nm"],
            "vulcan.stellar_spectrum.dbin_12trans_nm",
        )
        if spectrum["dbin1_nm"] <= 0.0:
            raise ConfigValidationError("vulcan.stellar_spectrum.dbin1_nm must be positive.")
        if spectrum["dbin2_nm"] <= 0.0:
            raise ConfigValidationError("vulcan.stellar_spectrum.dbin2_nm must be positive.")
        spectrum["encoder_mode"] = _as_nonempty_str(
            spectrum["encoder_mode"],
            "vulcan.stellar_spectrum.encoder_mode",
        ).lower()
        if spectrum["encoder_mode"] not in _ALLOWED_SPECTRUM_ENCODERS:
            raise ConfigValidationError(
                f"vulcan.stellar_spectrum.encoder_mode must be one of {_ALLOWED_SPECTRUM_ENCODERS}."
            )
        if "teff_k" in spectrum and spectrum["teff_k"] is not None:
            spectrum["teff_k"] = _as_float(
                spectrum["teff_k"],
                "vulcan.stellar_spectrum.teff_k",
            )
            if spectrum["teff_k"] <= 0.0:
                raise ConfigValidationError("vulcan.stellar_spectrum.teff_k must be positive.")
        else:
            spectrum["teff_k"] = None
        if "radius_rsun" in spectrum:
            spectrum["radius_rsun"] = _as_float(
                spectrum["radius_rsun"],
                "vulcan.stellar_spectrum.radius_rsun",
            )
            if spectrum["radius_rsun"] <= 0.0:
                raise ConfigValidationError("vulcan.stellar_spectrum.radius_rsun must be positive.")
        if "semi_major_axis_au" in spectrum:
            spectrum["semi_major_axis_au"] = _as_float(
                spectrum["semi_major_axis_au"],
                "vulcan.stellar_spectrum.semi_major_axis_au",
            )
            if spectrum["semi_major_axis_au"] <= 0.0:
                raise ConfigValidationError("vulcan.stellar_spectrum.semi_major_axis_au must be positive.")
        if "diurnal_factor" in spectrum:
            spectrum["diurnal_factor"] = _as_float(
                spectrum["diurnal_factor"],
                "vulcan.stellar_spectrum.diurnal_factor",
            )
            if spectrum["diurnal_factor"] <= 0.0:
                raise ConfigValidationError("vulcan.stellar_spectrum.diurnal_factor must be positive.")
        if "zenith_angle_deg" in spectrum:
            spectrum["zenith_angle_deg"] = _as_float(
                spectrum["zenith_angle_deg"],
                "vulcan.stellar_spectrum.zenith_angle_deg",
            )
            if spectrum["zenith_angle_deg"] < 0.0 or spectrum["zenith_angle_deg"] >= 90.0:
                raise ConfigValidationError(
                    "vulcan.stellar_spectrum.zenith_angle_deg must lie in [0, 90)."
                )

        section["physics_toggles"] = physics
        section["science_presets"] = science_presets
        section["runtime"] = runtime
        section["stellar_spectrum"] = spectrum
        config["vulcan"] = section
        config["physics_toggles"] = dict(physics)
        config["science_presets"] = list(science_presets)
        config["default_science_preset"] = dict(science_presets[0])
        config["vulcan_runtime"] = dict(runtime)
        config["stellar_spectrum"] = dict(spectrum)

    config["training"] = {
        "seed": training["seed"],
        "batch_size": training["batch_size"],
        "epochs": training["epochs"],
        "learning_rate": training["learning_rate"],
        "min_lr": training["min_lr"],
        "warmup_epochs": training["warmup_epochs"],
        "early_stopping_patience": training["early_stopping_patience"],
        "scheduler": training["scheduler"],
        "weight_decay": training["weight_decay"],
        "gradient_clip": training["gradient_clip"],
        "model": training["model"],
        "loss": training["loss"],
    }
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

    config["data_spec"]["state_dim"] = len(state_species)
    config["data_spec"]["target_dim"] = len(output_species)
    if chemistry_type == "fastchem":
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
