"""Pydantic schema for the full emulator config contract.

The top-level :data:`Config` is a discriminated union over ``chemistry_type``
(``fastchem`` or ``vulcan``). Each branch pulls in the shared sub-schemas
(paths, sampling, temperature profiles, training, model) plus any
chemistry-specific extensions (the ``vulcan`` block for VULCAN runs, extra
sampling keys).

Validation rules are expressed as ``model_config = ConfigDict(extra="forbid")``
plus ``@model_validator`` / ``@field_validator`` hooks. Call sites in
:mod:`src.utils.config` validate their dict slice against this schema, format
any ``ValidationError`` into the ``ConfigValidationError`` shape the rest of
the codebase expects, and emit a plain ``dict`` via ``model_dump(mode="python")``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from ..constants import (
    DBIN1_NM_DEFAULT,
    DBIN2_NM_DEFAULT,
    DBIN_12TRANS_NM_DEFAULT,
    DEFAULT_REQUIRED_GLOBAL_INPUTS,
    DEFAULT_STATE_SPECIES,
    FASTCHEM_CORE_GLOBAL_INPUTS,
    FRACTION_SUM_TOLERANCE,
    PUBLIC_PHYSICS_TOGGLES,
    SUPPORTED_ATM_BASES,
    VULCAN_SUPPORTED_CONDENSATE_SPECIES,
    _ALLOWED_ACTIVATIONS,
    _ALLOWED_NORMALIZATION_METHODS,
    _ALLOWED_TEMPERATURE_PROFILE_BOOLEAN_FILTER_KEYS,
    _ALLOWED_TEMPERATURE_PROFILE_FILTER_KEYS,
    _ALLOWED_TEMPERATURE_PROFILE_NUMERIC_FILTER_KEYS,
    _DEFAULT_SCIENCE_PRESET_NAME,
    _INTERNAL_VULCAN_RUNTIME_DEFAULTS,
)


# ---------------------------------------------------------------------------
# Shared base and helpers
# ---------------------------------------------------------------------------


class _StrictModel(BaseModel):
    """Base model with strict extra-key rejection."""

    model_config = ConfigDict(extra="forbid")


NormalizationMethod = Literal["standard", "log-standard", "log-minmax", "none"]
Activation = Literal[
    "elu", "gelu", "leaky_relu", "relu", "selu", "silu", "softplus", "tanh"
]
NormType = Literal["layernorm", "rmsnorm"]
FfnType = Literal["dense", "swiglu"]
AtmBase = Literal["H2", "N2", "O2", "CO2", "H2O"]


def _range_pair(low_name: str, high_name: str, *, allow_equal: bool):
    """Return a field validator that enforces a monotonic [low, high] pair."""

    def _validate(value: Any) -> list[float]:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("must be a length-2 list")
        if any(isinstance(v, bool) for v in value):
            raise ValueError("must be a length-2 list of numbers")
        try:
            lo = float(value[0])
            hi = float(value[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("must be a length-2 list of numbers") from exc
        if allow_equal:
            if hi < lo:
                raise ValueError("must satisfy lower <= upper")
        else:
            if hi <= lo:
                raise ValueError("must satisfy lower < upper")
        return [lo, hi]

    return _validate


_strict_range = _range_pair("lower", "upper", allow_equal=False)
_inclusive_range = _range_pair("lower", "upper", allow_equal=True)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


class PathsConfig(_StrictModel):
    """Dataset and runtime paths. ``run_root`` replaces the legacy split roots."""

    run_root: str = Field(..., min_length=1)
    checkpoints_root: str = Field(..., min_length=1)
    vulcan_source_root: str = Field(..., min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_roots(cls, data: Any) -> Any:
        if isinstance(data, dict) and ("raw_root" in data or "processed_root" in data):
            raise ValueError(
                "paths.raw_root and paths.processed_root are no longer supported. "
                "Use run_root only (e.g. 'data/fastchem_mlp'); it is auto-expanded "
                "to run_root/raw and run_root/processed."
            )
        return data


# ---------------------------------------------------------------------------
# Data spec
# ---------------------------------------------------------------------------


class DataSpecConfig(_StrictModel):
    """State and output species contract. Derived fields are added post-validation."""

    state_species: list[str] = Field(default_factory=lambda: list(DEFAULT_STATE_SPECIES))
    output_species: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_derived_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for forbidden in ("required_global_inputs", "element_input_order"):
                if forbidden in data:
                    raise ValueError(
                        f"data_spec.{forbidden} is derived internally and must not be set in the user config."
                    )
        return data

    @field_validator("state_species", "output_species", mode="after")
    @classmethod
    def _nonempty_unique(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        if not value:
            raise ValueError("must be a non-empty list")
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("must contain non-empty strings")
            if item in seen:
                raise ValueError(f"contains duplicate entry {item!r}")
            seen.add(item)
        return list(value)

    @model_validator(mode="after")
    def _default_outputs(self) -> "DataSpecConfig":
        if self.output_species is None:
            object.__setattr__(self, "output_species", list(self.state_species))
        return self


# ---------------------------------------------------------------------------
# Sampling (chemistry-conditional)
# ---------------------------------------------------------------------------


_FRAC_RANGE_KEYS = (
    "he_frac_range",
    "c_frac_range",
    "o_frac_range",
    "n_frac_range",
    "s_frac_range",
)


class CornerCoverageConfig(_StrictModel):
    """Optional targeted sampling for high-sensitivity chemistry corners."""

    enabled: bool = False
    fraction: float = 0.2
    abundance_quantile_width: float = 0.2
    hot_tmax_k: PositiveFloat = 2500.0
    large_trange_k: PositiveFloat = 800.0
    max_profile_resample_attempts: PositiveInt = 50

    @model_validator(mode="after")
    def _check_corner_coverage(self) -> "CornerCoverageConfig":
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("corner_coverage.fraction must lie in [0, 1]")
        if not 0.0 < self.abundance_quantile_width <= 0.5:
            raise ValueError(
                "corner_coverage.abundance_quantile_width must lie in (0, 0.5]"
            )
        return self


class _SamplingBase(_StrictModel):
    """Shared sampling fields used by both FastChem and VULCAN configs."""

    num_levels_range: list[int]
    pressure_top_bar_range: list[float]
    pressure_bottom_bar_range: list[float]
    temperature_range_k: list[float]
    he_frac_range: list[float]
    c_frac_range: list[float]
    o_frac_range: list[float]
    n_frac_range: list[float]
    s_frac_range: list[float]
    scales: dict[str, Literal["log", "linear"]] | None = None
    corner_coverage: CornerCoverageConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_scalars(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for legacy_key in ("num_levels", "pressure_top_bar", "pressure_bottom_bar"):
                if legacy_key in data:
                    raise ValueError(
                        f"sampling.{legacy_key} is no longer supported. Use "
                        f"sampling.{legacy_key}_range with a [lower, upper] pair instead."
                    )
            if "kzz_cm2_s" in data:
                raise ValueError(
                    "sampling.kzz_cm2_s (scalar) is no longer supported. Use "
                    "sampling.kzz_range_cm2_s with a [lower, upper] pair (log-sampled by default)."
                )
        return data

    @field_validator("num_levels_range", mode="after")
    @classmethod
    def _num_levels(cls, value: list[int]) -> list[int]:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("must be a length-2 list")
        if any(isinstance(v, bool) for v in value):
            raise ValueError("must be a length-2 list of integers")
        try:
            lo = int(value[0])
            hi = int(value[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("must be a length-2 list of integers") from exc
        if lo < 4:
            raise ValueError("num_levels_range[0] must be >= 4")
        if lo > hi:
            raise ValueError("must satisfy lower <= upper")
        return [lo, hi]

    @field_validator(
        "pressure_top_bar_range",
        "pressure_bottom_bar_range",
        mode="after",
    )
    @classmethod
    def _positive_range(cls, value: list[float]) -> list[float]:
        bounds = _strict_range(value)
        if bounds[0] <= 0.0 or bounds[1] <= 0.0:
            raise ValueError("bounds must be strictly positive (bar)")
        return bounds

    @field_validator("temperature_range_k", mode="after")
    @classmethod
    def _temperature_range(cls, value: list[float]) -> list[float]:
        return _strict_range(value)

    @field_validator(*_FRAC_RANGE_KEYS, mode="after")
    @classmethod
    def _frac_range(cls, value: list[float]) -> list[float]:
        return _strict_range(value)

    @model_validator(mode="after")
    def _check_pressure_ordering_and_frac_sum(self) -> "_SamplingBase":
        if self.pressure_top_bar_range[1] >= self.pressure_bottom_bar_range[0]:
            raise ValueError(
                "sampling.pressure_top_bar_range must lie strictly below "
                "sampling.pressure_bottom_bar_range so every sampled column has p_top < p_bottom."
            )
        frac_upper_sum = sum(getattr(self, k)[1] for k in _FRAC_RANGE_KEYS)
        if frac_upper_sum >= 1.0:
            raise ValueError(
                f"The upper bounds of the elemental fraction ranges sum to {frac_upper_sum:.6f}, "
                "which must be < 1.0 so that H_frac = 1 - sum remains positive."
            )
        return self


class FastChemSamplingConfig(_SamplingBase):
    """Sampling block for ``chemistry_type='fastchem'``. Rejects VULCAN-only keys."""

    @model_validator(mode="before")
    @classmethod
    def _reject_vulcan_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            vulcan_only = [
                key
                for key in (
                    "gravity_range_cm_s2",
                    "planet_radius_range_cm",
                    "stellar_radius_range_rsun",
                    "semi_major_axis_range_au",
                    "zenith_angle_range_deg",
                    "diurnal_factor_range",
                    "kzz_range_cm2_s",
                )
                if key in data
            ]
            if vulcan_only:
                raise ValueError(
                    "sampling contains VULCAN-only keys for "
                    f"chemistry_type='fastchem': {vulcan_only}"
                )
        return data


class VulcanSamplingConfig(_SamplingBase):
    """Sampling block for ``chemistry_type='vulcan'``. Adds planetary/stellar/kzz keys."""

    gravity_range_cm_s2: list[float]
    planet_radius_range_cm: list[float]
    stellar_radius_range_rsun: list[float]
    semi_major_axis_range_au: list[float]
    zenith_angle_range_deg: list[float]
    diurnal_factor_range: list[float]
    kzz_range_cm2_s: list[float]

    @field_validator(
        "gravity_range_cm_s2",
        "planet_radius_range_cm",
        mode="after",
    )
    @classmethod
    def _positive_strict(cls, value: list[float]) -> list[float]:
        bounds = _strict_range(value)
        if bounds[0] <= 0.0:
            raise ValueError("must be strictly positive")
        return bounds

    @field_validator("kzz_range_cm2_s", mode="after")
    @classmethod
    def _kzz_range(cls, value: list[float]) -> list[float]:
        bounds = _strict_range(value)
        if bounds[0] <= 0.0:
            raise ValueError("kzz_range_cm2_s must be strictly positive")
        return bounds

    @field_validator(
        "stellar_radius_range_rsun",
        "semi_major_axis_range_au",
        "diurnal_factor_range",
        mode="after",
    )
    @classmethod
    def _positive_inclusive(cls, value: list[float]) -> list[float]:
        bounds = _inclusive_range(value)
        if bounds[0] <= 0.0:
            raise ValueError("must be strictly positive")
        return bounds

    @field_validator("zenith_angle_range_deg", mode="after")
    @classmethod
    def _zenith_range(cls, value: list[float]) -> list[float]:
        bounds = _inclusive_range(value)
        if bounds[0] < 0.0 or bounds[1] >= 90.0:
            raise ValueError("zenith_angle_range_deg must lie within [0, 90)")
        return bounds


# ---------------------------------------------------------------------------
# Temperature profiles
# ---------------------------------------------------------------------------


class AnalyticSamplerConfig(_StrictModel):
    """Analytic PT sampler.

    Each analytic draw is one of two shapes, gated by ``power_law_probability``:

    * Modified Guillot (Piette & Madhusudhan 2019, default) — driven by the
      six ``t_int_k_range`` … ``log10_p_trans_bar_range`` fields plus the
      optional convective adjustment.
    * Pure power-law ``T(P) = T0 * (P / power_law_p_ref_bar) ** alpha`` —
      driven by ``power_law_t0_range_k`` and ``power_law_alpha_range``.
      Mirrors the ExoJAX ``art.powerlaw_temperature(T0, alpha)`` family
      used by retrieval consumers, so the training distribution covers the
      power-law prior tails NUTS walks through.

    Power-law fields are optional but required together when
    ``power_law_probability > 0``.
    """

    t_int_k_range: list[float]
    t_eq_k_range: list[float]
    log10_delta_range: list[float]
    log10_gamma_range: list[float]
    alpha_range: list[float]
    log10_p_trans_bar_range: list[float]
    convection_probability: float
    adiabatic_gradient_range: list[float]
    power_law_probability: float = 0.0
    power_law_t0_range_k: list[float] | None = None
    power_law_alpha_range: list[float] | None = None
    power_law_p_ref_bar: float = 1.0

    @field_validator(
        "t_int_k_range",
        "t_eq_k_range",
        "log10_delta_range",
        "log10_gamma_range",
        "log10_p_trans_bar_range",
        "adiabatic_gradient_range",
        mode="after",
    )
    @classmethod
    def _strict(cls, value: list[float]) -> list[float]:
        return _strict_range(value)

    @field_validator("alpha_range", mode="after")
    @classmethod
    def _alpha_range(cls, value: list[float]) -> list[float]:
        bounds = _strict_range(value)
        if bounds[0] < 0.0 or bounds[1] >= 1.0:
            raise ValueError("alpha_range must lie within [0, 1)")
        return bounds

    @field_validator("power_law_t0_range_k", mode="after")
    @classmethod
    def _power_law_t0(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return value
        bounds = _strict_range(value)
        if bounds[0] <= 0.0:
            raise ValueError("power_law_t0_range_k[0] must be > 0")
        return bounds

    @field_validator("power_law_alpha_range", mode="after")
    @classmethod
    def _power_law_alpha(cls, value: list[float] | None) -> list[float] | None:
        # Alpha may be negative (weak inversion) so no positivity floor here.
        if value is None:
            return value
        return _strict_range(value)

    @model_validator(mode="after")
    def _positive_floors(self) -> "AnalyticSamplerConfig":
        if self.t_int_k_range[0] <= 0.0:
            raise ValueError("t_int_k_range[0] must be > 0")
        if self.t_eq_k_range[0] <= 0.0:
            raise ValueError("t_eq_k_range[0] must be > 0")
        if self.adiabatic_gradient_range[0] <= 0.0:
            raise ValueError("adiabatic_gradient_range[0] must be > 0")
        if not 0.0 <= self.convection_probability <= 1.0:
            raise ValueError("convection_probability must lie in [0, 1]")
        if not 0.0 <= self.power_law_probability <= 1.0:
            raise ValueError("power_law_probability must lie in [0, 1]")
        if self.power_law_p_ref_bar <= 0.0:
            raise ValueError("power_law_p_ref_bar must be > 0")
        if self.power_law_probability > 0.0 and (
            self.power_law_t0_range_k is None
            or self.power_law_alpha_range is None
        ):
            raise ValueError(
                "power_law_probability > 0 requires both power_law_t0_range_k "
                "and power_law_alpha_range to be set."
            )
        return self


class TPValidationConfig(_StrictModel):
    min_temperature_k: float
    max_temperature_k: float

    @model_validator(mode="after")
    def _check(self) -> "TPValidationConfig":
        if self.min_temperature_k <= 0.0:
            raise ValueError("validation.min_temperature_k must be > 0")
        if self.max_temperature_k <= self.min_temperature_k:
            raise ValueError(
                "validation.max_temperature_k must be greater than min_temperature_k"
            )
        return self


def _validate_filter_entry(field: str, raw_value: Any) -> float | tuple[float, float] | bool:
    if field in _ALLOWED_TEMPERATURE_PROFILE_BOOLEAN_FILTER_KEYS:
        if not isinstance(raw_value, bool):
            raise ValueError(f"filters.{field} must be a boolean")
        return raw_value
    if isinstance(raw_value, list):
        if len(raw_value) != 2:
            raise ValueError(
                f"filters.{field} must be a numeric scalar or a two-number inclusive range."
            )
        if any(isinstance(v, bool) for v in raw_value):
            raise ValueError(
                f"filters.{field} must be a numeric scalar or a two-number inclusive range."
            )
        try:
            lo = float(raw_value[0])
            hi = float(raw_value[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"filters.{field} must be a numeric scalar or a two-number inclusive range."
            ) from exc
        if hi < lo:
            raise ValueError(
                f"filters.{field} range upper bound must be >= lower bound."
            )
        return (lo, hi)
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(
            f"filters.{field} must be a numeric scalar or a two-number inclusive range."
        )
    return float(raw_value)


class _TPBase(_StrictModel):
    """Shared temperature-profile fields."""

    validation: TPValidationConfig
    filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("filters", mode="before")
    @classmethod
    def _validate_filters(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("temperature_profiles.filters must be a mapping")
        result: dict[str, Any] = {}
        for raw_key, raw_val in value.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise ValueError("temperature_profiles.filters must use non-empty string keys")
            key = raw_key.strip()
            if key not in _ALLOWED_TEMPERATURE_PROFILE_FILTER_KEYS:
                raise ValueError(
                    f"temperature_profiles.filters.{key} is not supported. "
                    f"Allowed keys are {sorted(_ALLOWED_TEMPERATURE_PROFILE_FILTER_KEYS)}."
                )
            result[key] = _validate_filter_entry(key, raw_val)
        return result


class AnalyticTPConfig(_TPBase):
    source_mode: Literal["analytic"]
    analytic_sampler: AnalyticSamplerConfig


class MixedTPConfig(_TPBase):
    source_mode: Literal["mixed"]
    analytic_sampler: AnalyticSamplerConfig
    data_glob: str = Field(..., min_length=1)
    analytic_probability: float = 0.5

    @field_validator("analytic_probability", mode="after")
    @classmethod
    def _probability(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "temperature_profiles.analytic_probability must be between 0 and 1 "
                "when temperature_profiles.source_mode='mixed'."
            )
        return value


class PTLibraryTPConfig(_TPBase):
    source_mode: Literal["pt_library"]
    data_glob: str = Field(..., min_length=1)
    analytic_sampler: AnalyticSamplerConfig | None = None


TemperatureProfilesConfig = Annotated[
    Union[AnalyticTPConfig, MixedTPConfig, PTLibraryTPConfig],
    Field(discriminator="source_mode"),
]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class BackfillConfig(_StrictModel):
    enabled: bool = True
    max_retries: NonNegativeInt = 3


class GenerationConfig(_StrictModel):
    num_runs: PositiveInt
    seed: int
    overwrite: bool
    reuse_raw_if_present: bool
    parallel_workers: NonNegativeInt
    sample_chunk_size: PositiveInt = 1000
    fastchem_timeout_seconds: PositiveFloat = 30.0
    vulcan_timeout_seconds: PositiveFloat = 1800.0
    backfill: BackfillConfig = Field(default_factory=BackfillConfig)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class SplitConfig(_StrictModel):
    train_fraction: float
    val_fraction: float
    test_fraction: float
    seed: int

    @field_validator("train_fraction", "val_fraction", "test_fraction", mode="after")
    @classmethod
    def _positive(cls, value: float, info) -> float:
        if value <= 0.0:
            raise ValueError(f"normalization.split.{info.field_name} must be positive.")
        return value

    @model_validator(mode="after")
    def _sum_to_one(self) -> "SplitConfig":
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > FRACTION_SUM_TOLERANCE:
            raise ValueError("normalization.split fractions must sum to 1.")
        return self


class NormalizationConfig(_StrictModel):
    split: SplitConfig
    state_floor: PositiveFloat
    sequence_methods: dict[str, NormalizationMethod]
    global_methods: dict[str, NormalizationMethod]
    target_method: NormalizationMethod


# ---------------------------------------------------------------------------
# Scheduler & training
# ---------------------------------------------------------------------------


class CosineSchedulerConfig(_StrictModel):
    name: Literal["cosine"]


class ReduceOnPlateauSchedulerConfig(_StrictModel):
    name: Literal["reduce_on_plateau"]
    factor: float
    patience: NonNegativeInt
    threshold: NonNegativeFloat

    @field_validator("factor", mode="after")
    @classmethod
    def _factor_range(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("factor must lie strictly between 0 and 1.")
        return value


SchedulerConfig = Annotated[
    Union[CosineSchedulerConfig, ReduceOnPlateauSchedulerConfig],
    Field(discriminator="name"),
]


class _LossBase(_StrictModel):
    """Common fields across every loss variant."""

    lambda_z: NonNegativeFloat


class MAELossConfig(_LossBase):
    """MAE in log10 mixing-ratio space plus MSE in normalized space."""

    type: Literal["mae"]
    lambda_log10_mae: NonNegativeFloat


class HuberLossConfig(_LossBase):
    """Huber in log10 mixing-ratio space plus MSE in normalized space."""

    type: Literal["huber"]
    lambda_log10_huber: NonNegativeFloat
    huber_delta_log10: PositiveFloat


LossConfig = Annotated[
    Union[MAELossConfig, HuberLossConfig],
    Field(discriminator="type"),
]


class EmaConfig(_StrictModel):
    enabled: bool = False
    decay: float = 0.0

    @model_validator(mode="after")
    def _check_decay(self) -> "EmaConfig":
        if self.enabled and not (0.0 <= self.decay < 1.0):
            raise ValueError("training.ema.decay must be in [0, 1) when enabled.")
        return self


class TrainingConfig(_StrictModel):
    seed: int
    batch_size: PositiveInt
    epochs: PositiveInt
    learning_rate: PositiveFloat
    min_lr: PositiveFloat
    warmup_epochs: NonNegativeInt
    early_stopping_patience: PositiveInt
    scheduler: SchedulerConfig
    weight_decay: NonNegativeFloat
    gradient_clip: PositiveFloat
    loss: LossConfig
    ema: EmaConfig = Field(default_factory=EmaConfig)

    @model_validator(mode="after")
    def _check_lrs(self) -> "TrainingConfig":
        if self.min_lr > self.learning_rate:
            raise ValueError("training.min_lr cannot exceed training.learning_rate.")
        return self


# ---------------------------------------------------------------------------
# Model (FiLM Transformer)
# ---------------------------------------------------------------------------


class ModelConfig(_StrictModel):
    d_model: int
    nhead: PositiveInt
    num_layers: PositiveInt
    dim_feedforward: PositiveInt
    conditioning_hidden_dim: PositiveInt
    film_clamp: PositiveFloat
    output_head_divisor: PositiveInt
    activation: Activation
    dropout_rate: float
    norm_type: NormType
    use_qk_norm: bool
    ffn_type: FfnType
    zero_init_film: bool

    @field_validator("d_model", mode="after")
    @classmethod
    def _d_model_floor(cls, value: int) -> int:
        if value < 8:
            raise ValueError("model.d_model must be >= 8.")
        return value

    @field_validator("dropout_rate", mode="after")
    @classmethod
    def _dropout_range(cls, value: float) -> float:
        if not 0.0 <= value < 1.0:
            raise ValueError("model.dropout_rate must be in [0, 1).")
        return value

    @model_validator(mode="after")
    def _arithmetic(self) -> "ModelConfig":
        if self.d_model % self.nhead != 0:
            raise ValueError("model.d_model must be divisible by nhead.")
        if self.dim_feedforward < self.d_model:
            raise ValueError("model.dim_feedforward must be >= d_model.")
        return self


# ---------------------------------------------------------------------------
# VULCAN section
# ---------------------------------------------------------------------------


class PhysicsTogglesConfig(_StrictModel):
    """Public VULCAN physics toggles exposed in the config."""

    use_photochemistry: bool = False
    use_ion_chemistry: bool = False
    use_eddy_diffusion: bool = False
    use_molecular_diffusion: bool = False
    use_upwind_molecular_diffusion: bool = False
    use_boundary_conditions: bool = False
    use_condensation: bool = False
    use_settling: bool = False
    use_initial_cold_trap: bool = False
    use_sat_surface_h2o: bool = False


class SciencePresetInput(_StrictModel):
    """Raw science-preset payload before default merging."""

    name: str = Field(..., min_length=1)
    atm_base: AtmBase | None = None
    physics_toggles: dict[str, bool] = Field(default_factory=dict)

    @field_validator("physics_toggles", mode="after")
    @classmethod
    def _check_toggle_names(cls, value: dict[str, bool]) -> dict[str, bool]:
        for name, flag in value.items():
            if name not in PUBLIC_PHYSICS_TOGGLES:
                raise ValueError(
                    f"physics_toggles.{name} is not a supported public toggle."
                )
            if not isinstance(flag, bool):
                raise ValueError(f"physics_toggles.{name} must be a boolean.")
        return value


class CondensationConfig(_StrictModel):
    """Condensation-specific knobs passed through to ``vulcan_cfg.py``.

    Required when any science preset has ``use_condensation=True``. Every
    species in ``condense_sp`` must be in
    :data:`~src.constants.VULCAN_SUPPORTED_CONDENSATE_SPECIES` (the species
    set for which VULCAN ships saturation-pressure data); the matching
    condensate label in ``non_gas_sp`` (e.g. ``H2O_l_s``) must appear in
    ``data_spec.state_species`` and ``data_spec.output_species`` so the
    surrogate predicts the condensate VMR column.
    """

    condense_sp: list[str]
    non_gas_sp: list[str]
    fix_species: list[str] = Field(default_factory=list)
    use_relax: list[str] = Field(default_factory=list)
    humidity: float = 1.0
    start_conden_time: PositiveFloat = 1.0e6
    stop_conden_time: PositiveFloat = 1.0e8
    fix_species_time: PositiveFloat = 1.0e8
    fix_species_from_coldtrap_lev: bool = True
    post_conden_rtol: PositiveFloat = 0.1
    r_p: dict[str, PositiveFloat] = Field(default_factory=dict)
    rho_p: dict[str, PositiveFloat] = Field(default_factory=dict)

    @field_validator("condense_sp", "non_gas_sp", mode="after")
    @classmethod
    def _nonempty_unique(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must be a non-empty list")
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("must contain non-empty strings")
            if item in seen:
                raise ValueError(f"contains duplicate entry {item!r}")
            seen.add(item)
        return list(value)

    @field_validator("condense_sp", mode="after")
    @classmethod
    def _condense_sp_supported(cls, value: list[str]) -> list[str]:
        unsupported = [sp for sp in value if sp not in VULCAN_SUPPORTED_CONDENSATE_SPECIES]
        if unsupported:
            raise ValueError(
                f"condense_sp entries {unsupported} are not in "
                f"{sorted(VULCAN_SUPPORTED_CONDENSATE_SPECIES)} — VULCAN ships "
                "saturation-pressure data only for those species."
            )
        return value

    @model_validator(mode="after")
    def _check_paired_lists(self) -> "CondensationConfig":
        if len(self.condense_sp) != len(self.non_gas_sp):
            raise ValueError(
                "condense_sp and non_gas_sp must have the same length "
                f"(got {len(self.condense_sp)} and {len(self.non_gas_sp)})."
            )
        return self


class VulcanRuntimeConfig(_StrictModel):
    """Runtime configuration passed to the VULCAN worker."""

    chemistry_file: str = Field(..., min_length=1)
    t_cross_sp: list[str]
    python_executable: str = Field(
        default=_INTERNAL_VULCAN_RUNTIME_DEFAULTS["python_executable"], min_length=1
    )
    cfg_file: str = Field(
        default=_INTERNAL_VULCAN_RUNTIME_DEFAULTS["cfg_file"], min_length=1
    )
    worker_root: str = Field(
        default=_INTERNAL_VULCAN_RUNTIME_DEFAULTS["worker_root"], min_length=1
    )
    regenerate_chem_funs: bool = _INTERNAL_VULCAN_RUNTIME_DEFAULTS["regenerate_chem_funs"]
    cfg_assignments: dict[str, Any] = Field(default_factory=dict)
    use_lowT_limit_rates: bool = _INTERNAL_VULCAN_RUNTIME_DEFAULTS["use_lowT_limit_rates"]
    use_adaptive_rtol: bool = _INTERNAL_VULCAN_RUNTIME_DEFAULTS["use_adaptive_rtol"]
    rocky: bool = _INTERNAL_VULCAN_RUNTIME_DEFAULTS["rocky"]
    top_bc_flux_file: str | None = _INTERNAL_VULCAN_RUNTIME_DEFAULTS["top_bc_flux_file"]
    bot_bc_flux_file: str | None = _INTERNAL_VULCAN_RUNTIME_DEFAULTS["bot_bc_flux_file"]
    atm_base: AtmBase = "H2"
    condensation: CondensationConfig | None = None

    @field_validator("t_cross_sp", mode="after")
    @classmethod
    def _nonempty_strings(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("t_cross_sp must be a non-empty list")
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("t_cross_sp must contain non-empty strings")
            if item in seen:
                raise ValueError(f"t_cross_sp contains duplicate entry {item!r}")
            seen.add(item)
        return list(value)

    @field_validator("top_bc_flux_file", "bot_bc_flux_file", mode="after")
    @classmethod
    def _nonempty_or_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string or null")
        return value


class StellarSpectrumConfig(_StrictModel):
    """Stellar spectrum block used by data generation (not the model)."""

    template_name: str = Field(..., min_length=1)
    max_tokens: int
    wavelength_min_nm: float
    wavelength_max_nm: float
    template_file: str | None = None
    library_glob: str | None = None
    dbin1_nm: PositiveFloat = DBIN1_NM_DEFAULT
    dbin2_nm: PositiveFloat = DBIN2_NM_DEFAULT
    dbin_12trans_nm: float = DBIN_12TRANS_NM_DEFAULT
    teff_k: PositiveFloat = 5485.0
    radius_rsun: PositiveFloat = 0.939
    semi_major_axis_au: PositiveFloat = 0.04858

    @field_validator("template_file", "library_glob", mode="after")
    @classmethod
    def _nonempty_or_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string or null")
        return value

    @field_validator("max_tokens", mode="after")
    @classmethod
    def _max_tokens_floor(cls, value: int) -> int:
        if value < 8:
            raise ValueError("max_tokens must be >= 8.")
        return value

    @model_validator(mode="after")
    def _check_wavelengths(self) -> "StellarSpectrumConfig":
        if self.wavelength_min_nm >= self.wavelength_max_nm:
            raise ValueError(
                "wavelength_min_nm must be smaller than wavelength_max_nm."
            )
        return self


class VulcanSection(_StrictModel):
    physics_toggles: PhysicsTogglesConfig
    runtime: VulcanRuntimeConfig
    science_presets: list[SciencePresetInput] | None = None
    stellar_spectrum: StellarSpectrumConfig | None = None

    @model_validator(mode="after")
    def _expand_presets_and_reject_photochem(self) -> "VulcanSection":
        default_physics = self.physics_toggles.model_dump()
        default_atm_base = self.runtime.atm_base
        presets = self.science_presets
        if presets is None:
            expanded = [
                SciencePresetInput.model_validate(
                    {
                        "name": _DEFAULT_SCIENCE_PRESET_NAME,
                        "atm_base": default_atm_base,
                        "physics_toggles": dict(default_physics),
                    }
                )
            ]
        else:
            if not presets:
                raise ValueError("vulcan.science_presets must be a non-empty list when provided.")
            seen_names: set[str] = set()
            expanded = []
            for idx, preset in enumerate(presets):
                if preset.name in seen_names:
                    raise ValueError(
                        f"vulcan.science_presets contains duplicate preset name {preset.name!r}."
                    )
                seen_names.add(preset.name)
                atm_base = preset.atm_base if preset.atm_base is not None else default_atm_base
                merged = dict(default_physics)
                merged.update(preset.physics_toggles)
                expanded.append(
                    SciencePresetInput.model_validate(
                        {
                            "name": preset.name,
                            "atm_base": atm_base,
                            "physics_toggles": merged,
                        }
                    )
                )
        object.__setattr__(self, "science_presets", expanded)

        if any(p.physics_toggles.get("use_photochemistry", False) for p in expanded):
            raise ValueError(
                "Photochemistry is not currently supported. All science presets must "
                "have use_photochemistry=False (or 0). Support will be added in a future release."
            )
        return self


# ---------------------------------------------------------------------------
# Top-level Config (discriminated on chemistry_type)
# ---------------------------------------------------------------------------


class _ConfigBase(_StrictModel):
    """Shared sections common to both FastChem and VULCAN top-level configs."""

    model_type: Literal["transformer"]
    paths: PathsConfig
    data_spec: DataSpecConfig
    temperature_profiles: TemperatureProfilesConfig
    generation: GenerationConfig
    normalization: NormalizationConfig
    training: TrainingConfig
    model: ModelConfig


class FastChemConfig(_ConfigBase):
    chemistry_type: Literal["fastchem"]
    sampling: FastChemSamplingConfig

    @model_validator(mode="before")
    @classmethod
    def _reject_vulcan_block(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "vulcan" in data:
                raise ValueError(
                    "vulcan must not be defined when chemistry_type='fastchem'."
                )
            if "task" in data:
                raise ValueError(
                    "task has been removed. Use top-level chemistry_type and model_type instead."
                )
        return data

    @model_validator(mode="after")
    def _check_normalization_keys(self) -> "FastChemConfig":
        _check_fastchem_normalization_keys(self.normalization, self.data_spec)
        _check_fastchem_temperature_floor(self.sampling, self.temperature_profiles)
        _check_corner_coverage_temperature_thresholds(
            self.sampling, self.temperature_profiles,
        )
        return self


class VulcanConfig(_ConfigBase):
    chemistry_type: Literal["vulcan"]
    sampling: VulcanSamplingConfig
    vulcan: VulcanSection

    @model_validator(mode="before")
    @classmethod
    def _reject_task(cls, data: Any) -> Any:
        if isinstance(data, dict) and "task" in data:
            raise ValueError(
                "task has been removed. Use top-level chemistry_type and model_type instead."
            )
        return data

    @model_validator(mode="after")
    def _check_normalization_keys(self) -> "VulcanConfig":
        _check_vulcan_normalization_keys(self.normalization, self.data_spec)
        _check_corner_coverage_temperature_thresholds(
            self.sampling, self.temperature_profiles,
        )
        _check_vulcan_condensation_contract(self.vulcan, self.data_spec)
        return self


Config = Annotated[
    Union[FastChemConfig, VulcanConfig],
    Field(discriminator="chemistry_type"),
]

ConfigAdapter: TypeAdapter[Any] = TypeAdapter(Config)


# ---------------------------------------------------------------------------
# Post-schema cross-section checks
# ---------------------------------------------------------------------------


def _check_fastchem_normalization_keys(
    normalization: NormalizationConfig, data_spec: DataSpecConfig
) -> None:
    expected_seq = {"pressure_bar", "temperature_k"}
    if set(normalization.sequence_methods.keys()) != expected_seq:
        raise ValueError(
            f"normalization.sequence_methods must define exactly {expected_seq}."
        )
    expected_globals = set(FASTCHEM_CORE_GLOBAL_INPUTS)
    if set(normalization.global_methods.keys()) != expected_globals:
        raise ValueError(
            f"normalization.global_methods must define exactly {expected_globals}."
        )


def _check_fastchem_temperature_floor(
    sampling: FastChemSamplingConfig,
    temperature_profiles: TemperatureProfilesConfig,
) -> None:
    min_sampling_temperature = float(sampling.temperature_range_k[0])
    min_validation_temperature = float(temperature_profiles.validation.min_temperature_k)
    if min_sampling_temperature < 100.0:
        raise ValueError(
            "FastChem sampling.temperature_range_k[0] must be >= 100.0 K."
        )
    if min_validation_temperature < 100.0:
        raise ValueError(
            "FastChem temperature_profiles.validation.min_temperature_k must be >= 100.0 K."
        )


def _check_corner_coverage_temperature_thresholds(
    sampling: _SamplingBase,
    temperature_profiles: TemperatureProfilesConfig,
) -> None:
    corner = sampling.corner_coverage
    if corner is None or not corner.enabled:
        return
    validation = temperature_profiles.validation
    t_min = float(validation.min_temperature_k)
    t_max = float(validation.max_temperature_k)
    if float(corner.hot_tmax_k) > t_max:
        raise ValueError(
            "corner_coverage.hot_tmax_k must be <= "
            "temperature_profiles.validation.max_temperature_k."
        )
    if float(corner.hot_tmax_k) < t_min:
        raise ValueError(
            "corner_coverage.hot_tmax_k must be >= "
            "temperature_profiles.validation.min_temperature_k."
        )
    if float(corner.large_trange_k) > (t_max - t_min):
        raise ValueError(
            "corner_coverage.large_trange_k must be <= the validation "
            "temperature span."
        )


def _check_vulcan_normalization_keys(
    normalization: NormalizationConfig, data_spec: DataSpecConfig
) -> None:
    expected_seq = {"pressure_bar", "temperature_k", "kzz_cm2_s"}
    if set(normalization.sequence_methods.keys()) != expected_seq:
        raise ValueError(
            f"normalization.sequence_methods must define exactly {expected_seq}."
        )
    expected_globals = set(DEFAULT_REQUIRED_GLOBAL_INPUTS)
    if set(normalization.global_methods.keys()) != expected_globals:
        raise ValueError(
            f"normalization.global_methods must define exactly {expected_globals}."
        )


def _check_vulcan_condensation_contract(
    vulcan: VulcanSection, data_spec: DataSpecConfig
) -> None:
    """Cross-validate the condensation block against physics_toggles and species lists.

    Three rules:
    1. If any expanded science preset has ``use_condensation=True``, the runtime
       must carry a ``condensation`` block (no silent default — condensation
       requires explicit ``condense_sp`` / ``non_gas_sp`` choices).
    2. Every condensate label in ``non_gas_sp`` must appear in both
       ``data_spec.state_species`` and ``data_spec.output_species`` so the
       trained surrogate predicts the condensate VMR column.
    3. When ``use_settling=True`` for any preset, ``r_p`` and ``rho_p`` must
       cover every entry in ``non_gas_sp`` (settling needs particle radius
       and density per condensate).
    """
    presets = vulcan.science_presets or []
    cond_on = any(p.physics_toggles.get("use_condensation", False) for p in presets)
    settling_on = any(p.physics_toggles.get("use_settling", False) for p in presets)
    block = vulcan.runtime.condensation

    if cond_on and block is None:
        raise ValueError(
            "vulcan.runtime.condensation is required when any science preset has "
            "use_condensation=True."
        )
    if not cond_on and block is not None:
        raise ValueError(
            "vulcan.runtime.condensation is set but no science preset has "
            "use_condensation=True; remove the block or enable the toggle."
        )
    if block is None:
        return

    state_set = set(data_spec.state_species or [])
    output_set = set(data_spec.output_species or [])
    missing_state = [sp for sp in block.non_gas_sp if sp not in state_set]
    missing_output = [sp for sp in block.non_gas_sp if sp not in output_set]
    if missing_state:
        raise ValueError(
            f"non_gas_sp entries {missing_state} must appear in data_spec.state_species."
        )
    if missing_output:
        raise ValueError(
            f"non_gas_sp entries {missing_output} must appear in data_spec.output_species."
        )

    if settling_on:
        missing_r_p = [sp for sp in block.non_gas_sp if sp not in block.r_p]
        missing_rho_p = [sp for sp in block.non_gas_sp if sp not in block.rho_p]
        if missing_r_p or missing_rho_p:
            raise ValueError(
                "use_settling=True requires r_p and rho_p entries for every condensate; "
                f"missing r_p for {missing_r_p}, missing rho_p for {missing_rho_p}."
            )
