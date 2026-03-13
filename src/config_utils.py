from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUPPORTED_ATM_BASES = ("H2", "N2", "O2", "CO2", "H2O")
SUPPORTED_PHYSICS_TOGGLES = (
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
)
CORE_GLOBAL_INPUTS = ("gravity_cm_s2", "metallicity_log10", "c_to_o", "log10_dt_s")
OPTIONAL_GLOBAL_INPUTS = (
    *SUPPORTED_PHYSICS_TOGGLES,
    *tuple(f"atm_base_{name}" for name in SUPPORTED_ATM_BASES),
)
DEFAULT_REQUIRED_GLOBAL_INPUTS = (*CORE_GLOBAL_INPUTS, *OPTIONAL_GLOBAL_INPUTS)
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
_ALLOWED_SPECTRUM_ENCODERS = {"autoencoder", "linear", "none"}
_ALLOWED_EQ_ANCHOR_SOURCES = {"trajectory", "flat"}
_ALLOWED_EQ_ANCHOR_SPLITS = {"train", "val", "test"}


class ConfigValidationError(ValueError):
    """Raised when a configuration file violates the required contract."""


def _require_keys(mapping: dict[str, Any], keys: tuple[str, ...] | list[str], scope: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigValidationError(f"Missing required keys in {scope}: {missing}")


def _as_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigValidationError(f"{field} must be a boolean.")
    return value


def _as_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(f"{field} must be an integer.")
    return int(value)


def _as_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(f"{field} must be numeric.")
    return float(value)


def _as_nonempty_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"{field} must be a non-empty string.")
    return value.strip()


def _as_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ConfigValidationError(f"{field} must be a non-empty list.")
    result = [_as_nonempty_str(item, field) for item in value]
    if len(set(result)) != len(result):
        raise ConfigValidationError(f"{field} contains duplicate entries.")
    return result


def static_conditioning_defaults(config: dict[str, Any]) -> dict[str, float]:
    physics = config["physics_toggles"]
    runtime = config["vulcan_runtime"]
    defaults = {name: float(bool(physics[name])) for name in SUPPORTED_PHYSICS_TOGGLES}
    atm_base = _as_nonempty_str(runtime["atm_base"], "vulcan_runtime.atm_base")
    if atm_base not in SUPPORTED_ATM_BASES:
        raise ConfigValidationError(
            f"vulcan_runtime.atm_base must be one of {SUPPORTED_ATM_BASES}, got {atm_base!r}."
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
    defaults = static_conditioning_defaults(config)
    resolved: dict[str, float] = {}
    for name in required_global_inputs:
        if name == "log10_dt_s":
            continue
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
    return [
        name
        for name in config["data_spec"]["required_global_inputs"]
        if name != "log10_dt_s"
    ]


def global_feature_order(config: dict[str, Any]) -> list[str]:
    return list(config["data_spec"]["required_global_inputs"])


def load_and_validate_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    _require_keys(
        config,
        [
            "paths",
            "data_spec",
            "physics_toggles",
            "vulcan_runtime",
            "sampling",
            "stellar_spectrum",
            "generation",
            "trajectory_sampling",
            "preprocessing",
            "normalization",
            "training",
        ],
        "root",
    )

    paths = config["paths"]
    _require_keys(
        paths,
        [
            "raw_root",
            "processed_root",
            "checkpoints_root",
            "jax_export_root",
            "vulcan_source_root",
        ],
        "paths",
    )
    for key in paths:
        paths[key] = _as_nonempty_str(paths[key], f"paths.{key}")

    data_spec = config["data_spec"]
    state_species = _as_string_list(
        data_spec.get("state_species", list(DEFAULT_STATE_SPECIES)),
        "data_spec.state_species",
    )
    output_species = _as_string_list(
        data_spec.get("output_species", list(state_species)),
        "data_spec.output_species",
    )
    required_global_inputs = _as_string_list(
        data_spec.get("required_global_inputs", list(DEFAULT_REQUIRED_GLOBAL_INPUTS)),
        "data_spec.required_global_inputs",
    )
    if "log10_dt_s" not in required_global_inputs:
        raise ConfigValidationError("data_spec.required_global_inputs must contain 'log10_dt_s'.")
    data_spec["state_species"] = state_species
    data_spec["output_species"] = output_species
    data_spec["required_global_inputs"] = required_global_inputs

    physics = config["physics_toggles"]
    for name in SUPPORTED_PHYSICS_TOGGLES:
        physics[name] = _as_bool(physics.get(name, False), f"physics_toggles.{name}")

    runtime = config["vulcan_runtime"]
    _require_keys(
        runtime,
        [
            "python_executable",
            "cfg_file",
            "chemistry_file",
            "worker_root",
            "regenerate_chem_funs",
            "atm_base",
            "t_cross_sp",
            "cfg_assignments",
        ],
        "vulcan_runtime",
    )
    runtime["python_executable"] = _as_nonempty_str(
        runtime["python_executable"], "vulcan_runtime.python_executable"
    )
    runtime["cfg_file"] = _as_nonempty_str(runtime["cfg_file"], "vulcan_runtime.cfg_file")
    runtime["chemistry_file"] = _as_nonempty_str(
        runtime["chemistry_file"], "vulcan_runtime.chemistry_file"
    )
    runtime["worker_root"] = _as_nonempty_str(runtime["worker_root"], "vulcan_runtime.worker_root")
    runtime["regenerate_chem_funs"] = _as_bool(
        runtime["regenerate_chem_funs"], "vulcan_runtime.regenerate_chem_funs"
    )
    runtime["atm_base"] = _as_nonempty_str(runtime["atm_base"], "vulcan_runtime.atm_base")
    if runtime["atm_base"] not in SUPPORTED_ATM_BASES:
        raise ConfigValidationError(
            f"vulcan_runtime.atm_base must be one of {SUPPORTED_ATM_BASES}, got {runtime['atm_base']!r}."
        )
    runtime["t_cross_sp"] = _as_string_list(runtime["t_cross_sp"], "vulcan_runtime.t_cross_sp")
    if not isinstance(runtime["cfg_assignments"], dict):
        raise ConfigValidationError("vulcan_runtime.cfg_assignments must be a mapping.")

    sampling = config["sampling"]
    _require_keys(
        sampling,
        [
            "num_levels",
            "pressure_top_bar",
            "pressure_bottom_bar",
            "temperature_range_k",
            "gravity_range_cm_s2",
            "metallicity_log10_range",
            "c_to_o_range",
            "kzz_cm2_s",
            "num_time_steps",
            "time_step_log10_min_s",
            "time_step_log10_max_s",
        ],
        "sampling",
    )
    sampling["num_levels"] = _as_int(sampling["num_levels"], "sampling.num_levels")
    sampling["num_time_steps"] = _as_int(
        sampling["num_time_steps"], "sampling.num_time_steps"
    )
    if sampling["num_levels"] < 4:
        raise ConfigValidationError("sampling.num_levels must be >= 4.")
    if sampling["num_time_steps"] < 3:
        raise ConfigValidationError("sampling.num_time_steps must be >= 3.")
    sampling["pressure_top_bar"] = _as_float(
        sampling["pressure_top_bar"], "sampling.pressure_top_bar"
    )
    sampling["pressure_bottom_bar"] = _as_float(
        sampling["pressure_bottom_bar"], "sampling.pressure_bottom_bar"
    )
    if sampling["pressure_top_bar"] <= 0.0 or sampling["pressure_bottom_bar"] <= 0.0:
        raise ConfigValidationError("Pressure bounds must be positive.")
    if sampling["pressure_top_bar"] >= sampling["pressure_bottom_bar"]:
        raise ConfigValidationError("pressure_top_bar must be smaller than pressure_bottom_bar.")
    for key in (
        "temperature_range_k",
        "gravity_range_cm_s2",
        "metallicity_log10_range",
        "c_to_o_range",
    ):
        values = sampling[key]
        if not isinstance(values, list) or len(values) != 2:
            raise ConfigValidationError(f"sampling.{key} must be a length-2 list.")
        low = _as_float(values[0], f"sampling.{key}[0]")
        high = _as_float(values[1], f"sampling.{key}[1]")
        if not low < high:
            raise ConfigValidationError(f"sampling.{key} must be strictly increasing.")
        sampling[key] = [low, high]
    sampling["kzz_cm2_s"] = _as_float(sampling["kzz_cm2_s"], "sampling.kzz_cm2_s")
    if sampling["kzz_cm2_s"] <= 0.0:
        raise ConfigValidationError("sampling.kzz_cm2_s must be positive.")
    sampling["time_step_log10_min_s"] = _as_float(
        sampling["time_step_log10_min_s"], "sampling.time_step_log10_min_s"
    )
    sampling["time_step_log10_max_s"] = _as_float(
        sampling["time_step_log10_max_s"], "sampling.time_step_log10_max_s"
    )
    if not sampling["time_step_log10_min_s"] < sampling["time_step_log10_max_s"]:
        raise ConfigValidationError(
            "sampling.time_step_log10_min_s must be smaller than sampling.time_step_log10_max_s."
        )

    spectrum = config["stellar_spectrum"]
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
        "stellar_spectrum",
    )
    spectrum["enabled"] = _as_bool(spectrum["enabled"], "stellar_spectrum.enabled")
    spectrum["template_name"] = _as_nonempty_str(
        spectrum["template_name"], "stellar_spectrum.template_name"
    )
    spectrum["template_file"] = _as_nonempty_str(
        spectrum["template_file"], "stellar_spectrum.template_file"
    )
    spectrum["num_bins"] = _as_int(spectrum["num_bins"], "stellar_spectrum.num_bins")
    spectrum["latent_dim"] = _as_int(spectrum["latent_dim"], "stellar_spectrum.latent_dim")
    spectrum["hidden_dim"] = _as_int(spectrum["hidden_dim"], "stellar_spectrum.hidden_dim")
    if spectrum["num_bins"] < 8:
        raise ConfigValidationError("stellar_spectrum.num_bins must be >= 8.")
    if spectrum["latent_dim"] < 1 or spectrum["latent_dim"] > spectrum["num_bins"]:
        raise ConfigValidationError(
            "stellar_spectrum.latent_dim must lie in [1, num_bins]."
        )
    spectrum["wavelength_min_nm"] = _as_float(
        spectrum["wavelength_min_nm"], "stellar_spectrum.wavelength_min_nm"
    )
    spectrum["wavelength_max_nm"] = _as_float(
        spectrum["wavelength_max_nm"], "stellar_spectrum.wavelength_max_nm"
    )
    if spectrum["wavelength_min_nm"] >= spectrum["wavelength_max_nm"]:
        raise ConfigValidationError(
            "stellar_spectrum.wavelength_min_nm must be smaller than wavelength_max_nm."
        )
    spectrum["encoder_mode"] = _as_nonempty_str(
        spectrum["encoder_mode"], "stellar_spectrum.encoder_mode"
    ).lower()
    if spectrum["encoder_mode"] not in _ALLOWED_SPECTRUM_ENCODERS:
        raise ConfigValidationError(
            f"stellar_spectrum.encoder_mode must be one of {_ALLOWED_SPECTRUM_ENCODERS}."
        )
    for key in ("teff_k", "radius_rsun", "semi_major_axis_au", "diurnal_factor"):
        spectrum[key] = _as_float(spectrum[key], f"stellar_spectrum.{key}")
        if spectrum[key] <= 0.0:
            raise ConfigValidationError(f"stellar_spectrum.{key} must be positive.")
    spectrum["zenith_angle_deg"] = _as_float(
        spectrum["zenith_angle_deg"], "stellar_spectrum.zenith_angle_deg"
    )
    if spectrum["zenith_angle_deg"] < 0.0 or spectrum["zenith_angle_deg"] >= 90.0:
        raise ConfigValidationError("stellar_spectrum.zenith_angle_deg must lie in [0, 90).")

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
        generation["reuse_raw_if_present"], "generation.reuse_raw_if_present"
    )
    generation["parallel_workers"] = _as_int(
        generation["parallel_workers"], "generation.parallel_workers"
    )
    if generation["num_runs"] < 1:
        raise ConfigValidationError("generation.num_runs must be >= 1.")
    if generation["parallel_workers"] < 1:
        raise ConfigValidationError("generation.parallel_workers must be >= 1.")

    traj = config["trajectory_sampling"]
    _require_keys(
        traj,
        ["dt_min_s", "dt_max_s", "min_future_saved_steps", "num_logdt_bins"],
        "trajectory_sampling",
    )
    traj["dt_min_s"] = _as_float(traj["dt_min_s"], "trajectory_sampling.dt_min_s")
    traj["dt_max_s"] = _as_float(traj["dt_max_s"], "trajectory_sampling.dt_max_s")
    traj["min_future_saved_steps"] = _as_int(
        traj["min_future_saved_steps"], "trajectory_sampling.min_future_saved_steps"
    )
    traj["num_logdt_bins"] = _as_int(
        traj["num_logdt_bins"], "trajectory_sampling.num_logdt_bins"
    )
    if traj["dt_min_s"] <= 0.0 or traj["dt_max_s"] <= 0.0:
        raise ConfigValidationError("trajectory_sampling dt bounds must be positive.")
    if traj["dt_min_s"] >= traj["dt_max_s"]:
        raise ConfigValidationError("trajectory_sampling.dt_min_s must be < dt_max_s.")
    if traj["min_future_saved_steps"] < 1:
        raise ConfigValidationError("min_future_saved_steps must be >= 1.")
    if traj["num_logdt_bins"] < 1:
        raise ConfigValidationError("trajectory_sampling.num_logdt_bins must be >= 1.")

    inference = config.get("inference", {})
    if inference is None:
        inference = {}
    if not isinstance(inference, dict):
        raise ConfigValidationError("inference must be a mapping.")
    eq_anchor = inference.get("equilibrium_anchor", {})
    if eq_anchor is None:
        eq_anchor = {}
    if not isinstance(eq_anchor, dict):
        raise ConfigValidationError("inference.equilibrium_anchor must be a mapping.")
    eq_source = _as_nonempty_str(
        eq_anchor.get("source", "trajectory"),
        "inference.equilibrium_anchor.source",
    ).lower()
    if eq_source not in _ALLOWED_EQ_ANCHOR_SOURCES:
        raise ConfigValidationError(
            f"inference.equilibrium_anchor.source must be one of {_ALLOWED_EQ_ANCHOR_SOURCES}."
        )
    eq_split = _as_nonempty_str(
        eq_anchor.get("split", "train"),
        "inference.equilibrium_anchor.split",
    ).lower()
    if eq_split not in _ALLOWED_EQ_ANCHOR_SPLITS:
        raise ConfigValidationError(
            f"inference.equilibrium_anchor.split must be one of {_ALLOWED_EQ_ANCHOR_SPLITS}."
        )
    eq_run_id = eq_anchor.get("run_id")
    if eq_run_id is not None:
        eq_run_id = _as_nonempty_str(
            eq_run_id,
            "inference.equilibrium_anchor.run_id",
        )
    eq_step_index = eq_anchor.get("step_index")
    if eq_step_index is not None:
        eq_step_index = _as_int(
            eq_step_index,
            "inference.equilibrium_anchor.step_index",
        )
        if eq_step_index < 0:
            raise ConfigValidationError("inference.equilibrium_anchor.step_index must be >= 0.")
    else:
        eq_step_index = 0
    eq_anchor["source"] = eq_source
    eq_anchor["split"] = eq_split
    eq_anchor["run_id"] = eq_run_id
    eq_anchor["step_index"] = eq_step_index
    inference["equilibrium_anchor"] = eq_anchor
    config["inference"] = inference

    prep = config["preprocessing"]
    _require_keys(prep, ["train_fraction", "val_fraction", "test_fraction", "seed"], "preprocessing")
    for key in ("train_fraction", "val_fraction", "test_fraction"):
        prep[key] = _as_float(prep[key], f"preprocessing.{key}")
        if prep[key] <= 0.0:
            raise ConfigValidationError(f"preprocessing.{key} must be positive.")
    total_fraction = prep["train_fraction"] + prep["val_fraction"] + prep["test_fraction"]
    if abs(total_fraction - 1.0) > 1.0e-6:
        raise ConfigValidationError("preprocessing fractions must sum to 1.")
    prep["seed"] = _as_int(prep["seed"], "preprocessing.seed")

    norm = config["normalization"]
    _require_keys(norm, ["state_floor", "spectrum_floor", "sequence_methods"], "normalization")
    norm["state_floor"] = _as_float(norm["state_floor"], "normalization.state_floor")
    norm["spectrum_floor"] = _as_float(norm["spectrum_floor"], "normalization.spectrum_floor")
    if norm["state_floor"] <= 0.0 or norm["spectrum_floor"] <= 0.0:
        raise ConfigValidationError("normalization floors must be positive.")
    if not isinstance(norm["sequence_methods"], dict):
        raise ConfigValidationError("normalization.sequence_methods must be a mapping.")
    expected_sequence_methods = {"pressure_bar", "temperature_k", "kzz_cm2_s"}
    if set(norm["sequence_methods"].keys()) != expected_sequence_methods:
        raise ConfigValidationError(
            "normalization.sequence_methods must define exactly pressure_bar, temperature_k, and kzz_cm2_s."
        )
    for key, method in norm["sequence_methods"].items():
        method_name = _as_nonempty_str(method, f"normalization.sequence_methods.{key}").lower()
        if method_name not in {"standard", "log-standard", "none"}:
            raise ConfigValidationError(
                f"Unsupported normalization method for {key}: {method_name}."
            )
        norm["sequence_methods"][key] = method_name

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
            "live_sampling",
            "model",
            "loss",
        ],
        "training",
    )
    for key in ("seed", "batch_size", "epochs", "warmup_epochs"):
        training[key] = _as_int(training[key], f"training.{key}")
    for key in ("learning_rate", "min_lr", "weight_decay", "gradient_clip"):
        training[key] = _as_float(training[key], f"training.{key}")
    if training["batch_size"] < 1 or training["epochs"] < 1:
        raise ConfigValidationError("training.batch_size and training.epochs must be >= 1.")
    if training["learning_rate"] <= 0.0 or training["min_lr"] <= 0.0:
        raise ConfigValidationError("Learning rates must be positive.")
    if training["min_lr"] > training["learning_rate"]:
        raise ConfigValidationError("training.min_lr cannot exceed training.learning_rate.")
    if training["gradient_clip"] <= 0.0:
        raise ConfigValidationError("training.gradient_clip must be positive.")

    live = training["live_sampling"]
    _require_keys(live, ["train_pairs_per_run_per_epoch", "eval_pairs_per_run"], "training.live_sampling")
    live["train_pairs_per_run_per_epoch"] = _as_int(
        live["train_pairs_per_run_per_epoch"], "training.live_sampling.train_pairs_per_run_per_epoch"
    )
    live["eval_pairs_per_run"] = _as_int(
        live["eval_pairs_per_run"], "training.live_sampling.eval_pairs_per_run"
    )
    if live["train_pairs_per_run_per_epoch"] < 1 or live["eval_pairs_per_run"] < 1:
        raise ConfigValidationError("live sampling budgets must be >= 1.")

    model = training["model"]
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
        "training.model",
    )
    for key in ("d_model", "nhead", "num_layers", "dim_feedforward", "conditioning_hidden_dim", "output_head_divisor"):
        model[key] = _as_int(model[key], f"training.model.{key}")
    model["film_clamp"] = _as_float(model["film_clamp"], "training.model.film_clamp")
    if model["d_model"] < 8 or model["nhead"] < 1 or model["num_layers"] < 1:
        raise ConfigValidationError("training.model dimensions are too small.")
    if model["d_model"] % model["nhead"] != 0:
        raise ConfigValidationError("training.model.d_model must be divisible by nhead.")
    if model["dim_feedforward"] < model["d_model"]:
        raise ConfigValidationError("dim_feedforward must be >= d_model.")
    if model["output_head_divisor"] < 1:
        raise ConfigValidationError("output_head_divisor must be >= 1.")

    loss = training["loss"]
    _require_keys(loss, ["lambda_z", "lambda_phys", "lambda_spectrum"], "training.loss")
    for key in ("lambda_z", "lambda_phys", "lambda_spectrum"):
        loss[key] = _as_float(loss[key], f"training.loss.{key}")
        if loss[key] < 0.0:
            raise ConfigValidationError(f"training.loss.{key} must be non-negative.")

    config["data_spec"]["state_dim"] = len(state_species)
    config["data_spec"]["target_dim"] = len(output_species)
    config["data_spec"]["sequence_static_feature_order"] = [
        "pressure_bar",
        "temperature_k",
        "kzz_cm2_s",
    ]
    config["data_spec"]["global_static_feature_order"] = global_static_feature_order(config)
    config["data_spec"]["global_feature_order"] = global_feature_order(config)
    config["data_spec"]["dt_feature_index"] = config["data_spec"]["global_feature_order"].index(
        "log10_dt_s"
    )
    return config
