"""Configuration loading and strict validation."""

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


def _require_keys(container: dict[str, Any], required: set[str], scope: str) -> None:
    """Require a fixed key set in one config mapping."""
    missing = sorted(required - set(container.keys()))
    if missing:
        raise ConfigValidationError(f"Missing required keys in {scope}: {missing}")


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
        precision["stats_accumulation_dtype"], "precision.stats_accumulation_dtype"
    )
    model_dtype = _parse_dtype_name(precision["model_dtype"], "precision.model_dtype")
    forward_dtype = _parse_dtype_name(precision["forward_dtype"], "precision.forward_dtype")
    loss_dtype = _parse_dtype_name(precision["loss_dtype"], "precision.loss_dtype")
    optimizer_dtype = _parse_dtype_name(
        precision["optimizer_state_dtype"], "precision.optimizer_state_dtype"
    )
    amp_dtype = _parse_dtype_name(
        precision["amp_autocast_dtype"], "precision.amp_autocast_dtype", allow_none=True
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
    _require_keys(
        paths_cfg,
        {"project_root", "vulcan_source_path", "data_root", "models_root", "logs_root"},
        "paths",
    )
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
    _require_keys(
        cfg,
        {
            "num_runs",
            "num_workers",
            "snapshots_per_run",
            "snapshot_spacing",
            "split_ratios",
            "random_seed",
            "keep_vulcan_outputs_debug",
            "failure_policy",
            "run_timeout_seconds",
            "manifest_filename",
            "split_filename",
            "shard_size",
            "worker_root",
            "runs_root",
            "save_evo_frq",
        },
        "generation",
    )

    if _as_int(cfg["num_runs"], "generation.num_runs") <= 0:
        raise ConfigValidationError("generation.num_runs must be > 0.")
    if _as_int(cfg["num_workers"], "generation.num_workers") <= 0:
        raise ConfigValidationError("generation.num_workers must be > 0.")
    if _as_int(cfg["snapshots_per_run"], "generation.snapshots_per_run") <= 0:
        raise ConfigValidationError("generation.snapshots_per_run must be > 0.")
    _ = _as_int(cfg["random_seed"], "generation.random_seed")
    _ = _as_bool(cfg["keep_vulcan_outputs_debug"], "generation.keep_vulcan_outputs_debug")
    if str(cfg["snapshot_spacing"]) != "log_time":
        raise ConfigValidationError("generation.snapshot_spacing must be 'log_time'.")
    if str(cfg["failure_policy"]) != "fail_on_first_error":
        raise ConfigValidationError("generation.failure_policy must be 'fail_on_first_error'.")
    if _as_int(cfg["run_timeout_seconds"], "generation.run_timeout_seconds") <= 0:
        raise ConfigValidationError("generation.run_timeout_seconds must be > 0.")
    if _as_int(cfg["shard_size"], "generation.shard_size") <= 0:
        raise ConfigValidationError("generation.shard_size must be > 0.")
    if _as_int(cfg["save_evo_frq"], "generation.save_evo_frq") <= 0:
        raise ConfigValidationError("generation.save_evo_frq must be > 0.")

    for path_key in ("worker_root", "runs_root", "manifest_filename", "split_filename"):
        path_value = cfg[path_key]
        if not isinstance(path_value, str) or not path_value:
            raise ConfigValidationError(f"generation.{path_key} must be a non-empty string path.")
        if Path(path_value).is_absolute():
            raise ConfigValidationError(
                f"generation.{path_key} must be relative, got: {path_value}"
            )

    ratios = cfg["split_ratios"]
    _require_keys(ratios, {"train", "val", "test"}, "generation.split_ratios")
    train = _as_float(ratios["train"], "generation.split_ratios.train")
    val = _as_float(ratios["val"], "generation.split_ratios.val")
    test = _as_float(ratios["test"], "generation.split_ratios.test")
    if min(train, val, test) <= 0:
        raise ConfigValidationError("Split ratios must all be > 0.")
    if abs((train + val + test) - 1.0) > 1e-9:
        raise ConfigValidationError("Split ratios must sum to 1.0.")


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
    if p_top <= 0 or p_bottom <= 0 or p_bottom <= p_top:
        raise ConfigValidationError("Pressure bounds must satisfy 0 < p_top < p_bottom.")

    temp_limits = tp_cfg["temperature_limits_k"]
    _require_keys(temp_limits, {"min", "max"}, "tp_sampler.temperature_limits_k")
    t_min = _as_float(temp_limits["min"], "tp_sampler.temperature_limits_k.min")
    t_max = _as_float(temp_limits["max"], "tp_sampler.temperature_limits_k.max")
    if t_min <= 0 or t_max <= t_min:
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
    if floor <= 0 or cap <= floor:
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
    if co_hi <= co_lo or co_lo <= 0:
        raise ConfigValidationError("Invalid C/O range.")

    solar = abund_cfg["solar_abundances"]
    _require_keys(solar, {"O_H", "N_H", "He_H", "S_H"}, "abundance_sampler.solar_abundances")
    for key, value in solar.items():
        if _as_float(value, f"abundance_sampler.solar_abundances.{key}") <= 0:
            raise ConfigValidationError(f"Solar abundance {key} must be > 0.")


def _validate_data_spec(data_spec: dict[str, Any]) -> None:
    """Validate the explicit v1 input/output feature contract."""
    _require_keys(
        data_spec,
        {
            "target_species",
            "required_input_profiles",
            "required_global_inputs",
            "required_state_inputs",
            "time_input_transform",
            "strict_non_finite",
        },
        "data_spec",
    )

    species = data_spec["target_species"]
    if not isinstance(species, list) or len(species) != 20:
        raise ConfigValidationError(
            "data_spec.target_species must be a list of exactly 20 species."
        )
    if len(set(species)) != len(species):
        raise ConfigValidationError("data_spec.target_species contains duplicates.")
    if any((not isinstance(sp, str) or not sp.strip()) for sp in species):
        raise ConfigValidationError("All target species must be non-empty strings.")

    if str(data_spec["time_input_transform"]) != "log10_time_seconds":
        raise ConfigValidationError("data_spec.time_input_transform must be 'log10_time_seconds'.")
    _ = _as_bool(data_spec["strict_non_finite"], "data_spec.strict_non_finite")

    expected_inputs = ["pressure_bar", "temperature_k", "kzz_cm2_s"]
    if list(data_spec["required_input_profiles"]) != expected_inputs:
        raise ConfigValidationError(
            f"data_spec.required_input_profiles must equal {expected_inputs}."
        )

    expected_globals = ["gravity_cm_s2", "metallicity_log10", "c_to_o", "log10_time_s"]
    if list(data_spec["required_global_inputs"]) != expected_globals:
        raise ConfigValidationError(
            f"data_spec.required_global_inputs must equal {expected_globals}."
        )

    if list(data_spec["required_state_inputs"]) != ["initial_ymix"]:
        raise ConfigValidationError("data_spec.required_state_inputs must equal ['initial_ymix'].")


def _validate_normalization(norm_cfg: dict[str, Any]) -> None:
    """Validate explicit normalization methods for all feature groups."""
    _require_keys(
        norm_cfg,
        {"epsilon", "sequence_methods", "global_methods", "target_method"},
        "normalization",
    )
    if _as_float(norm_cfg["epsilon"], "normalization.epsilon") <= 0:
        raise ConfigValidationError("normalization.epsilon must be > 0.")

    allowed = {"standard", "log-standard", "log-min-max", "none"}

    sequence_methods = norm_cfg["sequence_methods"]
    if not isinstance(sequence_methods, dict) or not sequence_methods:
        raise ConfigValidationError("normalization.sequence_methods must be a non-empty mapping.")
    required_sequence = {"pressure_bar", "temperature_k", "kzz_cm2_s", "initial_ymix"}
    if set(sequence_methods) != required_sequence:
        raise ConfigValidationError(
            "normalization.sequence_methods must define exactly keys "
            f"{sorted(required_sequence)}, got {sorted(sequence_methods)}."
        )
    for key, method in sequence_methods.items():
        if method not in allowed:
            raise ConfigValidationError(
                "Unsupported normalization method "
                f"'{method}' in normalization.sequence_methods.{key}."
            )

    global_methods = norm_cfg["global_methods"]
    if not isinstance(global_methods, dict) or not global_methods:
        raise ConfigValidationError("normalization.global_methods must be a non-empty mapping.")
    required_globals = {"gravity_cm_s2", "metallicity_log10", "c_to_o", "log10_time_s"}
    if set(global_methods) != required_globals:
        raise ConfigValidationError(
            "normalization.global_methods must define exactly keys "
            f"{sorted(required_globals)}, got {sorted(global_methods)}."
        )
    for key, method in global_methods.items():
        if method not in allowed:
            raise ConfigValidationError(
                "Unsupported normalization method "
                f"'{method}' in normalization.global_methods.{key}."
            )
    if global_methods["metallicity_log10"] not in {"standard", "none"}:
        raise ConfigValidationError(
            "normalization.global_methods.metallicity_log10 cannot use log-based methods "
            "because metallicity_log10 is already base-10 transformed."
        )
    if global_methods["log10_time_s"] not in {"standard", "none"}:
        raise ConfigValidationError(
            "normalization.global_methods.log10_time_s cannot use log-based methods "
            "because time_input_transform already produces log10_time_s."
        )

    target_method = str(norm_cfg["target_method"])
    if target_method not in allowed:
        raise ConfigValidationError("Unsupported normalization.target_method.")


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
    if lr <= 0 or min_lr <= 0 or min_lr > lr:
        raise ConfigValidationError("training.min_lr must be >0 and <= training.learning_rate.")
    if _as_float(training["weight_decay"], "training.weight_decay") < 0.0:
        raise ConfigValidationError("training.weight_decay must be >= 0.")

    if _as_float(training["gradient_clip"], "training.gradient_clip") <= 0:
        raise ConfigValidationError("training.gradient_clip must be > 0.")
    if _as_int(training["num_workers"], "training.num_workers") < 0:
        raise ConfigValidationError("training.num_workers must be >= 0.")
    _as_int(training["seed"], "training.seed")
    _as_bool(training["use_amp"], "training.use_amp")

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
    if (
        _as_int(
            data_loading["max_cached_shards"],
            "training.data_loading.max_cached_shards",
        )
        <= 0
    ):
        raise ConfigValidationError("training.data_loading.max_cached_shards must be > 0.")
    if (
        _as_int(
            data_loading["large_shard_mmap_bytes"],
            "training.data_loading.large_shard_mmap_bytes",
        )
        <= 0
    ):
        raise ConfigValidationError("training.data_loading.large_shard_mmap_bytes must be > 0.")
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
    if _as_float(model_cfg["dropout"], "training.model.dropout") < 0:
        raise ConfigValidationError("training.model.dropout must be >= 0.")
    if _as_float(model_cfg["film_clamp"], "training.model.film_clamp") <= 0:
        raise ConfigValidationError("training.model.film_clamp must be > 0.")
    if _as_int(model_cfg["output_head_divisor"], "training.model.output_head_divisor") <= 0:
        raise ConfigValidationError("training.model.output_head_divisor must be > 0.")
    if _as_int(model_cfg["max_sequence_length"], "training.model.max_sequence_length") <= 0:
        raise ConfigValidationError("training.model.max_sequence_length must be > 0.")
    output_folder = training["output_folder"]
    if not isinstance(output_folder, str) or not output_folder:
        raise ConfigValidationError("training.output_folder must be a non-empty relative path.")
    if Path(output_folder).is_absolute():
        raise ConfigValidationError("training.output_folder must be relative.")


def _validate_physics_toggles(physics: dict[str, Any]) -> None:
    """Validate optional physics toggles against the supported v1 subset."""
    _require_keys(
        physics,
        {
            "use_photochemistry",
            "use_ion_chemistry",
            "use_transport",
            "use_boundary_conditions",
            "use_condensation_optional",
        },
        "physics_toggles",
    )
    for key in (
        "use_photochemistry",
        "use_ion_chemistry",
        "use_transport",
        "use_boundary_conditions",
        "use_condensation_optional",
    ):
        _ = _as_bool(physics[key], f"physics_toggles.{key}")

    if physics["use_photochemistry"]:
        raise ConfigValidationError(
            "v1 does not support photochemistry; set use_photochemistry=false."
        )
    if physics["use_ion_chemistry"]:
        raise ConfigValidationError(
            "v1 does not support ion chemistry; set use_ion_chemistry=false."
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
    # Enforce explicit base-10 naming where logarithmic controls are expected.
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
    _validate_tp_sampler(config["tp_sampler"])
    _validate_gravity_sampler(config["gravity_sampler"])
    _validate_kzz_sampler(config["kzz_sampler"])
    _validate_abundance_sampler(config["abundance_sampler"])
    _validate_data_spec(config["data_spec"])
    _validate_normalization(config["normalization"])
    _validate_training(config["training"])
    _validate_physics_toggles(config["physics_toggles"])
    _validate_boundary_conditions(
        config.get("boundary_conditions"),
        required=bool(config["physics_toggles"]["use_boundary_conditions"]),
    )
    _validate_log10_convention(config)
    _ = resolve_precision(config)

    return config
