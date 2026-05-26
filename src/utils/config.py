"""Configuration loading, validation, and normalization for emulator configs.

This module is the single source of truth for config schema validation. It:

1. Loads a JSON config file from disk.
2. Delegates schema validation to :mod:`src.utils.schemas` (Pydantic).
3. Derives internal aliases (``roth_sampler``,
   ``data_spec`` feature orders) so downstream code can rely on a normalized,
   validated structure.
4. Returns a dict that is safe to pass to any pipeline stage.

Validation is strict and fail-fast: any schema violation raises
``ConfigValidationError`` with a descriptive message identifying the
offending field.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..constants import (  # noqa: F401 — re-exported for downstream consumers
    CHEMISTRY_TYPES,
    DBIN1_NM_DEFAULT,
    DBIN2_NM_DEFAULT,
    DBIN_12TRANS_NM_DEFAULT,
    DEFAULT_REQUIRED_GLOBAL_INPUTS,
    DEFAULT_STATE_SPECIES,
    ELEMENT_INPUT_ORDER,
    FASTCHEM_CONDITIONING_INPUT_ORDER,
    FASTCHEM_CORE_GLOBAL_INPUTS,
    FRACTION_SUM_TOLERANCE,
    MODEL_TYPES,
    PUBLIC_PHYSICS_TOGGLES,
    SUPPORTED_ATM_BASES,
    VULCAN_CONDITIONING_INPUT_ORDER,
    VULCAN_CORE_GLOBAL_INPUTS,
    VULCAN_OPTIONAL_GLOBAL_INPUTS,
    VULCAN_STELLAR_GLOBAL_INPUTS,
)
from .schemas import (
    ConfigAdapter,
    ModelConfig,
)


class ConfigValidationError(ValueError):
    """Raised when a configuration file violates the required contract."""


_DISCRIMINATOR_ERROR_TYPES = {"union_tag_not_found", "union_tag_invalid"}
_TOP_LEVEL_DISCRIMINATORS = {"fastchem", "vulcan", "exogibbs"}


def _format_pydantic_error(exc: ValidationError, scope: str) -> str:
    """Render a Pydantic ``ValidationError`` as a ``<scope>.<field>: msg`` list.

    Keeps the dotted field-path error-message shape that tests and logs
    already rely on. Top-level discriminator tags (``fastchem`` /
    ``vulcan``, used by the ``chemistry_type`` discriminated union) are
    stripped from the leading position of the loc so users see paths
    relative to the root. Discriminator errors (tag missing or unknown)
    append the discriminator key to the loc so the error points at the
    missing field rather than the enclosing union.
    """
    parts: list[str] = []
    for err in exc.errors():
        loc = list(err["loc"])
        if loc and loc[0] in _TOP_LEVEL_DISCRIMINATORS:
            loc = loc[1:]
        if err["type"] in _DISCRIMINATOR_ERROR_TYPES:
            ctx = err.get("ctx") or {}
            disc = str(ctx.get("discriminator", "type")).strip("'\"")
            loc.append(disc)
        tail = ".".join(str(x) for x in loc)
        path = f"{scope}.{tail}" if tail else scope
        parts.append(f"{path}: {err['msg']}")
    return "; ".join(parts)


def get_chemistry_type(config: dict[str, Any]) -> str:
    """Return the validated chemistry target family from the config."""
    value = config.get("chemistry_type")
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError("chemistry_type: must be a non-empty string.")
    normalized = value.strip().lower()
    if normalized not in CHEMISTRY_TYPES:
        raise ConfigValidationError(
            f"chemistry_type must be one of {CHEMISTRY_TYPES}, got {normalized!r}."
        )
    return normalized


def get_model_type(config: dict[str, Any]) -> str:
    """Return the validated prediction architecture family from the config."""
    value = config.get("model_type")
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError("model_type: must be a non-empty string.")
    normalized = value.strip().lower()
    if normalized not in MODEL_TYPES:
        raise ConfigValidationError(
            f"model_type must be one of {MODEL_TYPES}, got {normalized!r}."
        )
    return normalized


def uses_fastchem(config: dict[str, Any]) -> bool:
    """Report whether the config targets FastChem equilibrium chemistry."""
    return get_chemistry_type(config) == "fastchem"


def uses_exogibbs(config: dict[str, Any]) -> bool:
    """Report whether the config targets ExoGibbs equilibrium chemistry."""
    return get_chemistry_type(config) == "exogibbs"


def uses_vulcan_chemistry(config: dict[str, Any]) -> bool:
    """Report whether the config targets converged VULCAN chemistry outputs."""
    return get_chemistry_type(config) == "vulcan"


def uses_transformer(config: dict[str, Any]) -> bool:
    """Report whether the config selects the FiLM-conditioned Transformer."""
    return get_model_type(config) == "transformer"


def resolve_conditioning_inputs(
    *,
    raw_global_inputs: dict[str, float],
    required_global_inputs: list[str],
) -> dict[str, float]:
    """Resolve and validate the required per-run conditioning inputs."""
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
    """Return the ordered list of global conditioning feature names."""
    return list(config["data_spec"]["required_global_inputs"])


def global_feature_order(config: dict[str, Any]) -> list[str]:
    """Return the full ordered global feature vector contract for the model."""
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


def validate_transformer_model_config(
    model: dict[str, Any], scope: str = "model"
) -> dict[str, Any]:
    """Validate a Transformer ``model`` dict against :class:`ModelConfig`.

    Used by :mod:`src.tuning` Optuna trials to re-validate per-trial
    hyperparameter overrides on top of a validated base config.
    """
    try:
        return ModelConfig.model_validate(model).model_dump(mode="python")
    except ValidationError as exc:
        raise ConfigValidationError(_format_pydantic_error(exc, scope)) from exc


# Matches a JSON string literal OR a /* ... */ block comment OR a //... line
# comment. The string-literal alternative is listed first so re.sub keeps
# strings intact and only strips comments outside them.
_JSONC_COMMENT_PATTERN = re.compile(
    r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*',
    re.DOTALL,
)


def _strip_jsonc_comments(text: str) -> str:
    """Strip ``//`` line and ``/* */`` block comments from JSONC text.

    Preserves comment-like substrings inside JSON strings (e.g. a path that
    contains ``//``). Trailing commas are NOT stripped — keep configs valid
    JSON aside from comments.
    """

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        return token if token.startswith('"') else ""

    return _JSONC_COMMENT_PATTERN.sub(_replace, text)


def _strip_comment_keys(node: Any) -> Any:
    """Recursively drop any mapping key whose name starts with ``_comment``.

    The pipeline configs are strict JSON (no // or /* */), so structured
    notes are encoded as ``_comment``-prefixed keys. The schema is
    ``extra="forbid"``, so these must be removed before validation.
    """
    if isinstance(node, dict):
        return {
            k: _strip_comment_keys(v)
            for k, v in node.items()
            if not (isinstance(k, str) and k.startswith("_comment"))
        }
    if isinstance(node, list):
        return [_strip_comment_keys(v) for v in node]
    return node


def load_and_validate_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config file, validate against the schema, and derive aliases.

    Validation is delegated to :data:`src.utils.schemas.Config` (a
    discriminated union over ``chemistry_type``). Derived aliases expected
    by downstream pipeline stages are appended post-validation:

    - ``paths.raw_root`` / ``paths.processed_root`` (from ``run_root``)
    - ``data_spec.element_input_order``, ``required_global_inputs``,
      ``state_dim``, ``target_dim``, sequence/global feature orders
    - ``roth_sampler`` (extract from ``temperature_profiles``)
    - ``physics_toggles``, ``science_presets``, ``default_science_preset``,
      ``vulcan_runtime``, ``stellar_spectrum`` (VULCAN mirrors)
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = _strip_comment_keys(json.loads(_strip_jsonc_comments(handle.read())))

    try:
        validated = ConfigAdapter.validate_python(raw)
    except ValidationError as exc:
        raise ConfigValidationError(_format_pydantic_error(exc, "root")) from exc

    config: dict[str, Any] = validated.model_dump(mode="python")
    chemistry_type = config["chemistry_type"]

    paths = config["paths"]
    run_root = paths.pop("run_root")
    paths["raw_root"] = str(Path(run_root) / "raw")
    paths["processed_root"] = str(Path(run_root) / "processed")

    data_spec = config["data_spec"]
    state_species = list(data_spec["state_species"])
    output_species = list(data_spec["output_species"])
    derived_globals = (
        list(FASTCHEM_CORE_GLOBAL_INPUTS)
        if chemistry_type in ("fastchem", "exogibbs")
        else list(DEFAULT_REQUIRED_GLOBAL_INPUTS)
    )
    data_spec["state_species"] = state_species
    data_spec["output_species"] = output_species
    data_spec["element_input_order"] = list(ELEMENT_INPUT_ORDER)
    data_spec["required_global_inputs"] = derived_globals
    data_spec["state_dim"] = len(state_species)
    data_spec["target_dim"] = len(output_species)
    data_spec["sequence_static_feature_order"] = (
        ["pressure_bar", "temperature_k"]
        if chemistry_type in ("fastchem", "exogibbs")
        else ["pressure_bar", "temperature_k", "kzz_cm2_s"]
    )
    data_spec["global_static_feature_order"] = list(derived_globals)
    data_spec["global_feature_order"] = list(derived_globals)

    temperature_profiles = config["temperature_profiles"]
    source_mode = temperature_profiles["source_mode"]
    config["roth_sampler"] = {
        "enabled": source_mode in {"pt_library", "mixed"},
        "source_mode": "mixed" if source_mode == "mixed" else "roth",
        "analytic_probability": temperature_profiles.get("analytic_probability"),
        "data_glob": temperature_profiles.get("data_glob", ""),
        "filters": dict(temperature_profiles["filters"]),
    }

    if chemistry_type == "vulcan":
        section = config["vulcan"]
        physics = dict(section["physics_toggles"])
        presets = list(section["science_presets"])
        config["physics_toggles"] = physics
        config["science_presets"] = presets
        config["default_science_preset"] = dict(presets[0])
        config["vulcan_runtime"] = dict(section["runtime"])
        if section.get("stellar_spectrum") is not None:
            config["stellar_spectrum"] = dict(section["stellar_spectrum"])

    return config
