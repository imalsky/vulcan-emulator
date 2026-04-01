from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.config import (
    DEFAULT_REQUIRED_GLOBAL_INPUTS,
    DEFAULT_STATE_SPECIES,
    ConfigValidationError,
    load_and_validate_config,
    resolve_conditioning_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
FASTCHEM_GLOBAL_ORDER = ["He_H", "C_H", "O_H", "N_H", "S_H"]
VULCAN_GLOBAL_ORDER = list(DEFAULT_REQUIRED_GLOBAL_INPUTS)


def _load_raw_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, name: str, payload: dict) -> Path:
    config_path = tmp_path / name
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return config_path


@pytest.mark.parametrize(
    ("filename", "chemistry_type", "model_type", "sequence_order", "global_order"),
    [
        (
            "fastchem_mlp_config.json",
            "fastchem",
            "mlp",
            ["pressure_bar", "temperature_k"],
            FASTCHEM_GLOBAL_ORDER,
        ),
        (
            "fastchem_transformer_config.json",
            "fastchem",
            "transformer",
            ["pressure_bar", "temperature_k"],
            FASTCHEM_GLOBAL_ORDER,
        ),
        (
            "vulcan_mlp_config.json",
            "vulcan",
            "mlp",
            ["pressure_bar", "temperature_k", "kzz_cm2_s"],
            VULCAN_GLOBAL_ORDER,
        ),
        (
            "vulcan_transformer_config.json",
            "vulcan",
            "transformer",
            ["pressure_bar", "temperature_k", "kzz_cm2_s"],
            VULCAN_GLOBAL_ORDER,
        ),
    ],
)
def test_shipped_configs_load_with_expected_contract(
    filename: str,
    chemistry_type: str,
    model_type: str,
    sequence_order: list[str],
    global_order: list[str],
):
    raw_config = _load_raw_json(ROOT / "config" / filename)
    assert "required_global_inputs" not in raw_config["data_spec"]
    assert "element_input_order" not in raw_config["data_spec"]

    config = load_and_validate_config(ROOT / "config" / filename)
    assert config["chemistry_type"] == chemistry_type
    assert config["model_type"] == model_type
    assert config["data_spec"]["state_species"] == list(DEFAULT_STATE_SPECIES)
    assert config["data_spec"]["element_input_order"] == FASTCHEM_GLOBAL_ORDER
    assert config["data_spec"]["required_global_inputs"] == global_order
    assert config["data_spec"]["sequence_static_feature_order"] == sequence_order
    assert config["data_spec"]["global_static_feature_order"] == global_order
    assert config["normalization"]["target_method"] == "log-standard"
    assert config["training"]["scheduler"] == {
        "name": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 10,
        "threshold": pytest.approx(1.0e-4),
    }


def test_shipped_fastchem_config_defaults_are_correct():
    config = load_and_validate_config(ROOT / "config" / "fastchem_mlp_config.json")
    assert config["normalization"]["global_methods"] == {
        "He_H": "standard",
        "C_H": "log-standard",
        "O_H": "log-standard",
        "N_H": "log-standard",
        "S_H": "log-standard",
    }
    assert config["temperature_profiles"]["source_mode"] == "mixed"
    assert config["temperature_profiles"]["analytic_probability"] == pytest.approx(0.5)
    assert config["temperature_profiles"]["analytic_sampler"]["t_int_k_normal"] == {
        "mean": 500.0,
        "std": 20.0,
    }
    assert config["temperature_profiles"]["validation"] == {
        "min_temperature_k": 1.0,
        "max_temperature_k": 3000.0,
    }
    assert config["roth_sampler"]["enabled"] is True
    assert config["roth_sampler"]["data_glob"] == "assets/PTprofiles/*.dat"
    assert config["training"]["model"]["activation"] == "leaky_relu"
    assert config["training"]["model"]["dropout_rate"] == pytest.approx(0.05)
    assert config["training"]["early_stopping_patience"] == 30


def test_shipped_vulcan_config_defaults_are_correct():
    raw_config = _load_raw_json(ROOT / "config" / "vulcan_transformer_config.json")
    assert "python_executable" not in raw_config["vulcan"]["runtime"]
    assert "cfg_file" not in raw_config["vulcan"]["runtime"]

    config = load_and_validate_config(ROOT / "config" / "vulcan_transformer_config.json")
    assert config["normalization"]["global_methods"]["gravity_cm_s2"] == "log-standard"
    assert config["vulcan_runtime"]["python_executable"] == "python"
    assert config["vulcan_runtime"]["cfg_file"] == "vulcan_cfg.py"
    assert config["vulcan_runtime"]["worker_root"] == "data/vulcan_workers"
    assert config["vulcan_runtime"]["use_lowT_limit_rates"] is True
    assert config["vulcan_runtime"]["use_adaptive_rtol"] is True
    assert len(config["science_presets"]) == 1
    assert config["science_presets"][0]["name"] == "basic_h2"
    assert config["science_presets"][0]["atm_base"] == "H2"
    assert config["science_presets"][0]["physics_toggles"]["use_photochemistry"] is False
    assert config["stellar_spectrum"]["library_glob"] == "assets/stellar_spectra/*.dat"


def test_resolve_conditioning_inputs_rejects_missing_vulcan_runtime_inputs():
    config = load_and_validate_config(ROOT / "config" / "vulcan_transformer_config.json")
    with pytest.raises(ConfigValidationError, match="use_photochemistry"):
        resolve_conditioning_inputs(
            raw_global_inputs={
                "gravity_cm_s2": 1.0e3,
                "He_H": 8.38e-2,
                "C_H": 2.95e-4,
                "O_H": 5.37e-4,
                "N_H": 7.08e-5,
                "S_H": 1.41e-5,
            },
            required_global_inputs=list(config["data_spec"]["required_global_inputs"]),
        )


def test_vulcan_block_is_required_only_for_vulcan(tmp_path):
    fastchem = _load_raw_json(ROOT / "config" / "fastchem_mlp_config.json")
    fastchem["vulcan"] = {"unexpected": True}
    with pytest.raises(ConfigValidationError, match="chemistry_type='fastchem'"):
        load_and_validate_config(_write_config(tmp_path, "fastchem_invalid.json", fastchem))

    vulcan = _load_raw_json(ROOT / "config" / "vulcan_mlp_config.json")
    vulcan.pop("vulcan")
    with pytest.raises(ConfigValidationError, match="Missing required keys in root"):
        load_and_validate_config(_write_config(tmp_path, "vulcan_missing.json", vulcan))


@pytest.mark.parametrize(
    ("legacy_key", "legacy_value", "message"),
    [
        ("task", {"kind": "equilibrium_only"}, "task.kind"),
        ("equilibrium_only", {"model": {}}, "equilibrium_only"),
        ("full_vulcan", {"model": {}}, "full_vulcan"),
    ],
)
def test_legacy_sections_are_rejected(tmp_path, legacy_key: str, legacy_value: object, message: str):
    payload = _load_raw_json(ROOT / "config" / "fastchem_mlp_config.json")
    payload[legacy_key] = legacy_value
    with pytest.raises(ConfigValidationError, match=message):
        load_and_validate_config(_write_config(tmp_path, "legacy_section.json", payload))


def test_legacy_model_type_values_are_rejected(tmp_path):
    payload = _load_raw_json(ROOT / "config" / "fastchem_mlp_config.json")
    payload["model_type"] = "equilibrium"
    with pytest.raises(ConfigValidationError, match="Legacy model_type values"):
        load_and_validate_config(_write_config(tmp_path, "legacy_model_type.json", payload))


@pytest.mark.parametrize(
    ("filename", "model_type"),
    [
        ("fastchem_mlp_config.json", "mlp"),
        ("fastchem_transformer_config.json", "transformer"),
    ],
)
def test_model_defaults_are_applied_for_each_model_family(tmp_path, filename: str, model_type: str):
    payload = _load_raw_json(ROOT / "config" / filename)
    payload["model"].pop("activation")
    payload["model"].pop("dropout_rate")
    payload["training"].pop("early_stopping_patience")
    config = load_and_validate_config(_write_config(tmp_path, filename, payload))
    assert config["model_type"] == model_type
    assert config["training"]["model"]["activation"] == "leaky_relu"
    assert config["training"]["model"]["dropout_rate"] == pytest.approx(0.05)
    assert config["training"]["early_stopping_patience"] == 30


@pytest.mark.parametrize(
    "activation",
    ["relu", "gelu", "silu", "tanh", "elu", "selu", "softplus", "leaky_relu"],
)
@pytest.mark.parametrize("filename", ["fastchem_mlp_config.json", "fastchem_transformer_config.json"])
def test_supported_activations_validate_for_both_model_families(
    tmp_path,
    activation: str,
    filename: str,
):
    payload = _load_raw_json(ROOT / "config" / filename)
    payload["model"]["activation"] = activation
    config = load_and_validate_config(_write_config(tmp_path, filename, payload))
    assert config["training"]["model"]["activation"] == activation


def test_invalid_mlp_activation_is_rejected(tmp_path):
    payload = _load_raw_json(ROOT / "config" / "fastchem_mlp_config.json")
    payload["model"]["activation"] = "bad_activation"
    with pytest.raises(ConfigValidationError, match="model.activation"):
        load_and_validate_config(_write_config(tmp_path, "invalid_activation.json", payload))


def test_invalid_dropout_rate_is_rejected(tmp_path):
    payload = _load_raw_json(ROOT / "config" / "fastchem_mlp_config.json")
    payload["model"]["dropout_rate"] = 1.0
    with pytest.raises(ConfigValidationError, match="dropout_rate"):
        load_and_validate_config(_write_config(tmp_path, "invalid_dropout.json", payload))


def test_invalid_early_stopping_patience_is_rejected(tmp_path):
    payload = _load_raw_json(ROOT / "config" / "fastchem_mlp_config.json")
    payload["training"]["early_stopping_patience"] = 0
    with pytest.raises(ConfigValidationError, match="early_stopping_patience"):
        load_and_validate_config(_write_config(tmp_path, "invalid_patience.json", payload))


def test_lambda_spectrum_validation_depends_on_chemistry_type(tmp_path):
    fastchem = _load_raw_json(ROOT / "config" / "fastchem_mlp_config.json")
    fastchem["training"]["loss"]["lambda_spectrum"] = 0.01
    with pytest.raises(ConfigValidationError, match="lambda_spectrum"):
        load_and_validate_config(_write_config(tmp_path, "fastchem_lambda_spectrum.json", fastchem))

    vulcan = _load_raw_json(ROOT / "config" / "vulcan_mlp_config.json")
    vulcan["training"]["loss"].pop("lambda_spectrum")
    with pytest.raises(ConfigValidationError, match="lambda_spectrum"):
        load_and_validate_config(_write_config(tmp_path, "vulcan_missing_lambda_spectrum.json", vulcan))


def test_invalid_mixed_temperature_profile_probability_is_rejected(tmp_path):
    payload = _load_raw_json(ROOT / "config" / "fastchem_mlp_config.json")
    payload["temperature_profiles"]["analytic_probability"] = 1.0
    with pytest.raises(ConfigValidationError, match="analytic_probability"):
        load_and_validate_config(_write_config(tmp_path, "invalid_temp_prob.json", payload))


def test_missing_analytic_sampler_for_mixed_temperature_profiles_is_rejected(tmp_path):
    payload = _load_raw_json(ROOT / "config" / "fastchem_mlp_config.json")
    payload["temperature_profiles"].pop("analytic_sampler")
    with pytest.raises(ConfigValidationError, match="analytic_sampler"):
        load_and_validate_config(_write_config(tmp_path, "missing_analytic_sampler.json", payload))


def test_valid_temperature_profile_filters_are_normalized(tmp_path):
    payload = _load_raw_json(ROOT / "config" / "fastchem_mlp_config.json")
    payload["temperature_profiles"]["filters"] = {
        "Teq": [1400.0, 1600.0],
        "LogDrag": 0.0,
        "TiOVO": False,
    }
    validated = load_and_validate_config(_write_config(tmp_path, "valid_filters.json", payload))
    assert validated["temperature_profiles"]["filters"] == {
        "Teq": (1400.0, 1600.0),
        "LogDrag": 0.0,
        "TiOVO": False,
    }


def test_invalid_temperature_profile_filter_key_is_rejected(tmp_path):
    payload = _load_raw_json(ROOT / "config" / "fastchem_mlp_config.json")
    payload["temperature_profiles"]["filters"] = {"phase": ["global_mean", "disk_mean"]}
    with pytest.raises(ConfigValidationError, match="temperature_profiles.filters"):
        load_and_validate_config(_write_config(tmp_path, "invalid_filter_key.json", payload))


def test_invalid_analytic_temperature_sampler_exponent_range_is_rejected(tmp_path):
    payload = _load_raw_json(ROOT / "config" / "fastchem_mlp_config.json")
    payload["temperature_profiles"]["analytic_sampler"]["power_law_n_range"] = [0.0, 0.5]
    with pytest.raises(ConfigValidationError, match="power_law_n_range"):
        load_and_validate_config(_write_config(tmp_path, "invalid_power_law.json", payload))


def test_invalid_temperature_profile_filter_range_is_rejected(tmp_path):
    payload = _load_raw_json(ROOT / "config" / "fastchem_mlp_config.json")
    payload["temperature_profiles"]["filters"] = {"Teq": [1800.0, 1200.0]}
    with pytest.raises(ConfigValidationError, match="Teq"):
        load_and_validate_config(_write_config(tmp_path, "invalid_filter_range.json", payload))


def test_explicit_cosine_scheduler_is_preserved(tmp_path):
    payload = _load_raw_json(ROOT / "config" / "fastchem_mlp_config.json")
    payload["training"]["scheduler"] = {"name": "cosine"}
    validated = load_and_validate_config(_write_config(tmp_path, "cosine_scheduler.json", payload))
    assert validated["training"]["scheduler"] == {"name": "cosine"}


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (("training", "live_sampling"), "training.live_sampling"),
        (("sampling", "num_time_steps"), "sampling.num_time_steps"),
        (("generation", "target_mode"), "generation.target_mode"),
        (("normalization", "state_method"), "normalization.state_method"),
        (("vulcan", "trajectory_sampling"), "vulcan.trajectory_sampling"),
    ],
)
def test_removed_legacy_keys_raise_targeted_errors(tmp_path, path: tuple[str, str], message: str):
    payload = _load_raw_json(ROOT / "config" / "vulcan_transformer_config.json")
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = {"legacy": True} if path[-1] == "trajectory_sampling" else 1
    with pytest.raises(ConfigValidationError, match=message):
        load_and_validate_config(_write_config(tmp_path, "legacy_key.json", payload))
