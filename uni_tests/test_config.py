from __future__ import annotations

import copy
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
FIXTURE_ROOT = ROOT / "uni_tests" / "fixtures"
FASTCHEM_GLOBAL_ORDER = ["He_H", "C_H", "O_H", "N_H", "S_H"]
VULCAN_GLOBAL_ORDER = list(DEFAULT_REQUIRED_GLOBAL_INPUTS)


def _load_raw_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, name: str, payload: dict) -> Path:
    config_path = tmp_path / name
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return config_path


@pytest.mark.parametrize(
    "filename",
    [
        "vulcan_no_condensation.json",
        "vulcan_condensation.json",
    ],
)
def test_shipped_configs_use_single_dataset_root_layout(filename: str):
    raw_config = _load_raw_json(ROOT / "config" / filename)
    run_root = Path(raw_config["paths"]["run_root"])

    assert run_root.parent.name == "data"


def test_config_rejects_legacy_data_root_keys(tmp_path):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload["paths"]["raw_root"] = "data/raw/fastchem"
    payload["paths"]["processed_root"] = "data/processed/fastchem_mlp"
    with pytest.raises(ConfigValidationError, match="no longer supported|run_root only"):
        load_and_validate_config(_write_config(tmp_path, "split_roots.json", payload))


def test_config_requires_run_root(tmp_path):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload["paths"].pop("run_root")
    with pytest.raises(ConfigValidationError, match="Missing required keys in paths"):
        load_and_validate_config(_write_config(tmp_path, "missing_run_root.json", payload))


def test_shipped_vulcan_config_defaults_are_correct():
    raw_config = _load_raw_json(ROOT / "config" / "vulcan_condensation.json")
    assert "python_executable" not in raw_config["vulcan"]["runtime"]
    assert "cfg_file" not in raw_config["vulcan"]["runtime"]
    assert raw_config["vulcan"]["runtime"]["chemistry_file"] == "thermo/SNCHO_photo_network_2025.txt"
    assert raw_config["vulcan"]["runtime"]["regenerate_chem_funs"] is True
    assert raw_config["vulcan"]["runtime"]["rocky"] is False
    assert raw_config["vulcan"]["runtime"]["cfg_assignments"]["condense_sp"] == ["H2O", "S8"]
    assert raw_config["vulcan"]["runtime"]["cfg_assignments"]["non_gas_sp"] == ["H2O_l_s", "S8_l_s"]
    assert "enabled" not in raw_config["vulcan"]["stellar_spectrum"]

    config = load_and_validate_config(ROOT / "config" / "vulcan_condensation.json")
    assert config["normalization"]["global_methods"]["gravity_cm_s2"] == "log-standard"
    assert config["normalization"]["global_methods"]["planet_radius_cm"] == "log-standard"
    assert config["vulcan_runtime"]["python_executable"] == "python"
    assert config["vulcan_runtime"]["cfg_file"] == "vulcan_cfg.py"
    assert config["vulcan_runtime"]["worker_root"] == "data/vulcan_workers"
    assert config["vulcan_runtime"]["chemistry_file"] == "thermo/SNCHO_photo_network_2025.txt"
    assert config["vulcan_runtime"]["regenerate_chem_funs"] is True
    assert config["vulcan_runtime"]["use_lowT_limit_rates"] is True
    assert config["vulcan_runtime"]["use_adaptive_rtol"] is True
    assert config["vulcan_runtime"]["rocky"] is False
    assert config["vulcan_runtime"]["top_bc_flux_file"] is None
    assert config["vulcan_runtime"]["bot_bc_flux_file"] is None
    assert config["vulcan_runtime"]["cfg_assignments"]["condense_sp"] == ["H2O", "S8"]
    assert config["vulcan_runtime"]["cfg_assignments"]["non_gas_sp"] == ["H2O_l_s", "S8_l_s"]
    assert len(config["science_presets"]) == 1
    assert config["science_presets"][0]["name"] == "basic_h2"
    assert config["science_presets"][0]["atm_base"] == "H2"
    assert config["science_presets"][0]["physics_toggles"]["use_photochemistry"] is False
    assert config["science_presets"][0]["physics_toggles"]["use_condensation"] is True
    assert config["stellar_spectrum"]["template_file"] is None
    assert config["stellar_spectrum"]["library_glob"] is None
    assert config["stellar_spectrum"]["teff_k"] == pytest.approx(5485.0)
    assert config["stellar_spectrum"]["max_tokens"] == 2610


def test_resolve_conditioning_inputs_rejects_missing_vulcan_runtime_inputs():
    config = load_and_validate_config(ROOT / "config" / "vulcan_condensation.json")
    with pytest.raises(ConfigValidationError, match="use_photochemistry"):
        resolve_conditioning_inputs(
            raw_global_inputs={
                "gravity_cm_s2": 1.0e3,
                "planet_radius_cm": 9.0e9,
                "He_H": 7.84e-2,
                "C_H": 2.69e-4,
                "O_H": 4.90e-4,
                "N_H": 6.76e-5,
                "S_H": 1.32e-5,
                "r_star_rsun": 0.939,
                "semi_major_axis_au": 0.04858,
                "zenith_angle_deg": 48.0,
                "diurnal_factor": 1.0,
            },
            required_global_inputs=list(config["data_spec"]["required_global_inputs"]),
        )


def test_vulcan_config_allows_missing_template_file_when_photochemistry_disabled(tmp_path):
    payload = _load_raw_json(ROOT / "config" / "vulcan_condensation.json")
    payload["vulcan"]["stellar_spectrum"]["template_file"] = None
    payload["vulcan"]["stellar_spectrum"]["library_glob"] = None

    config = load_and_validate_config(_write_config(tmp_path, "vulcan_no_template.json", payload))

    assert config["stellar_spectrum"]["template_file"] is None


def test_vulcan_config_rejects_photochemistry_enabled(tmp_path):
    payload = _load_raw_json(ROOT / "config" / "vulcan_condensation.json")
    payload["vulcan"]["physics_toggles"]["use_photochemistry"] = True
    payload["vulcan"]["science_presets"] = [
        {
            "name": "photo_h2",
            "atm_base": "H2",
            "physics_toggles": {
                "use_photochemistry": True,
                "use_ion_chemistry": False,
                "use_eddy_diffusion": True,
                "use_molecular_diffusion": False,
                "use_upwind_molecular_diffusion": False,
                "use_boundary_conditions": False,
                "use_condensation": True,
                "use_settling": False,
                "use_initial_cold_trap": False,
                "use_sat_surface_h2o": False,
            },
        }
    ]

    with pytest.raises(ConfigValidationError, match="Photochemistry is not currently supported"):
        load_and_validate_config(_write_config(tmp_path, "vulcan_photochem.json", payload))


def test_vulcan_block_is_required_only_for_vulcan(tmp_path):
    fastchem = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    fastchem["vulcan"] = {"unexpected": True}
    with pytest.raises(ConfigValidationError, match="chemistry_type='fastchem'"):
        load_and_validate_config(_write_config(tmp_path, "fastchem_invalid.json", fastchem))

    vulcan = _load_raw_json(ROOT / "config" / "vulcan_condensation.json")
    vulcan.pop("vulcan")
    with pytest.raises(ConfigValidationError, match="Missing required keys in root"):
        load_and_validate_config(_write_config(tmp_path, "vulcan_missing.json", vulcan))


@pytest.mark.parametrize(
    ("section", "key", "error_fragment"),
    [
        ("model", "activation", "model"),
        ("model", "dropout_rate", "model"),
        ("training", "early_stopping_patience", "training"),
        ("training", "scheduler", "training"),
    ],
)
def test_required_hyperparameters_must_be_explicit(tmp_path, section, key, error_fragment):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload = copy.deepcopy(payload)
    payload[section].pop(key)
    with pytest.raises(ConfigValidationError, match=error_fragment):
        load_and_validate_config(_write_config(tmp_path, "missing_hparam.json", payload))


@pytest.mark.parametrize(
    "activation",
    ["relu", "gelu", "silu", "tanh", "elu", "selu", "softplus", "leaky_relu"],
)
def test_supported_activations_validate(tmp_path, activation: str):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload = copy.deepcopy(payload)
    payload["model"]["activation"] = activation
    config = load_and_validate_config(_write_config(tmp_path, "fastchem_transformer_config.json", payload))
    assert config["training"]["model"]["activation"] == activation


def test_invalid_activation_is_rejected(tmp_path):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload["model"]["activation"] = "bad_activation"
    with pytest.raises(ConfigValidationError, match="model.activation"):
        load_and_validate_config(_write_config(tmp_path, "invalid_activation.json", payload))


def test_invalid_model_type_is_rejected(tmp_path):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload["model_type"] = "mlp"
    with pytest.raises(ConfigValidationError, match="model_type"):
        load_and_validate_config(_write_config(tmp_path, "invalid_model_type.json", payload))


def test_invalid_dropout_rate_is_rejected(tmp_path):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload["model"]["dropout_rate"] = 1.0
    with pytest.raises(ConfigValidationError, match="dropout_rate"):
        load_and_validate_config(_write_config(tmp_path, "invalid_dropout.json", payload))


def test_invalid_early_stopping_patience_is_rejected(tmp_path):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload["training"]["early_stopping_patience"] = 0
    with pytest.raises(ConfigValidationError, match="early_stopping_patience"):
        load_and_validate_config(_write_config(tmp_path, "invalid_patience.json", payload))


def test_vulcan_requires_lambda_z_and_lambda_phys_only(tmp_path):
    payload = _load_raw_json(ROOT / "config" / "vulcan_condensation.json")
    payload["training"]["loss"].pop("lambda_phys")
    with pytest.raises(ConfigValidationError, match="lambda_phys"):
        load_and_validate_config(_write_config(tmp_path, "vulcan_missing_lambda_phys.json", payload))


def test_vulcan_requires_planet_radius_sampling_range(tmp_path):
    payload = _load_raw_json(ROOT / "config" / "vulcan_condensation.json")
    payload["sampling"].pop("planet_radius_range_cm")
    with pytest.raises(ConfigValidationError, match="planet_radius_range_cm"):
        load_and_validate_config(_write_config(tmp_path, "missing_planet_radius.json", payload))


def test_fastchem_rejects_vulcan_only_sampling_keys(tmp_path):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload["sampling"]["planet_radius_range_cm"] = [7.0e9, 1.0e10]
    with pytest.raises(ConfigValidationError, match="chemistry_type='fastchem'"):
        load_and_validate_config(_write_config(tmp_path, "fastchem_planet_radius.json", payload))


def test_invalid_mixed_temperature_profile_probability_is_rejected(tmp_path):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload["temperature_profiles"]["analytic_probability"] = 1.5
    with pytest.raises(ConfigValidationError, match="analytic_probability"):
        load_and_validate_config(_write_config(tmp_path, "invalid_temp_prob.json", payload))


def test_missing_analytic_sampler_for_mixed_temperature_profiles_is_rejected(tmp_path):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload["temperature_profiles"].pop("analytic_sampler")
    with pytest.raises(ConfigValidationError, match="analytic_sampler"):
        load_and_validate_config(_write_config(tmp_path, "missing_analytic_sampler.json", payload))


def test_valid_temperature_profile_filters_are_normalized(tmp_path):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
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
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload["temperature_profiles"]["filters"] = {"phase": ["global_mean", "disk_mean"]}
    with pytest.raises(ConfigValidationError, match="temperature_profiles.filters"):
        load_and_validate_config(_write_config(tmp_path, "invalid_filter_key.json", payload))


def test_invalid_analytic_temperature_sampler_t_int_range_is_rejected(tmp_path):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload["temperature_profiles"]["analytic_sampler"]["t_int_k_range"] = [0.0, 500.0]
    with pytest.raises(ConfigValidationError, match="t_int_k_range"):
        load_and_validate_config(_write_config(tmp_path, "invalid_t_int.json", payload))


def test_invalid_temperature_profile_filter_range_is_rejected(tmp_path):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload["temperature_profiles"]["filters"] = {"Teq": [1800.0, 1200.0]}
    with pytest.raises(ConfigValidationError, match="Teq"):
        load_and_validate_config(_write_config(tmp_path, "invalid_filter_range.json", payload))


def test_explicit_cosine_scheduler_is_preserved(tmp_path):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload["training"]["scheduler"] = {"name": "cosine"}
    validated = load_and_validate_config(_write_config(tmp_path, "cosine_scheduler.json", payload))
    assert validated["training"]["scheduler"] == {"name": "cosine"}


def test_run_pbs_defaults_to_no_condensation_config() -> None:
    script_text = (ROOT / "run.pbs").read_text(encoding="utf-8")
    assert 'CONFIG_PATH="${CONFIG_PATH:-config/vulcan_no_condensation.json}"' in script_text
    assert "#PBS -N vulcan_emulator" in script_text
