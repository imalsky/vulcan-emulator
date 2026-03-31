from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.config import DEFAULT_STATE_SPECIES, ConfigValidationError, load_and_validate_config


def _load_raw_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_shipped_equilibrium_config_loads():
    root = Path(__file__).resolve().parents[1]
    raw_config = _load_raw_json(root / "config" / "equilibrium_only_config.json")
    assert "required_global_inputs" not in raw_config["data_spec"]
    assert "element_input_order" not in raw_config["data_spec"]
    config = load_and_validate_config(root / "config" / "equilibrium_only_config.json")
    assert config["task"]["kind"] == "equilibrium_only"
    assert config["model_type"] == "equilibrium"
    assert config["data_spec"]["state_species"] == list(DEFAULT_STATE_SPECIES)
    assert config["data_spec"]["required_global_inputs"] == ["He_H", "C_H", "O_H", "N_H", "S_H"]
    assert config["data_spec"]["element_input_order"] == ["He_H", "C_H", "O_H", "N_H", "S_H"]
    assert config["normalization"]["target_method"] == "log-standard"
    assert config["normalization"]["global_methods"] == {
        "He_H": "log-standard",
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
    assert config["temperature_profiles"]["analytic_sampler"]["power_law_n_range"] == [0.5, 2.0]
    assert config["roth_sampler"]["enabled"] is True
    assert config["roth_sampler"]["data_glob"] == "assets/PTprofiles/*.dat"
    assert config["training"]["model"]["activation"] == "relu"
    assert config["training"]["scheduler"] == {
        "name": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 10,
        "threshold": pytest.approx(1.0e-4),
    }


def test_shipped_full_vulcan_config_loads():
    root = Path(__file__).resolve().parents[1]
    raw_config = _load_raw_json(root / "config" / "full_vulcan_config.json")
    assert "python_executable" not in raw_config["full_vulcan"]["vulcan_runtime"]
    assert "cfg_file" not in raw_config["full_vulcan"]["vulcan_runtime"]
    config = load_and_validate_config(root / "config" / "full_vulcan_config.json")
    assert config["task"]["kind"] == "full_vulcan"
    assert config["model_type"] == "full_vulcan"
    assert config["data_spec"]["required_global_inputs"] == [
        "gravity_cm_s2",
        "He_H",
        "C_H",
        "O_H",
        "N_H",
        "S_H",
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
        "atm_base_H2",
        "atm_base_N2",
        "atm_base_O2",
        "atm_base_CO2",
        "atm_base_H2O",
    ]
    assert config["vulcan_runtime"]["python_executable"] == "python"
    assert config["vulcan_runtime"]["cfg_file"] == "vulcan_cfg.py"
    assert config["vulcan_runtime"]["worker_root"] == "data/vulcan_workers"
    assert config["vulcan_runtime"]["use_lowT_limit_rates"] is True
    assert config["vulcan_runtime"]["use_adaptive_rtol"] is True
    assert "use_lowT_limit_rates" not in config["physics_toggles"]
    assert "use_adaptive_rtol" not in config["physics_toggles"]


def test_missing_task_specific_section_is_rejected(tmp_path):
    root = Path(__file__).resolve().parents[1]
    payload = _load_raw_json(root / "config" / "equilibrium_only_config.json")
    payload.pop("equilibrium_only")
    config_path = tmp_path / "equilibrium_only_config.json"
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(config_path)


def test_forbidden_task_specific_section_is_rejected(tmp_path):
    root = Path(__file__).resolve().parents[1]
    payload = _load_raw_json(root / "config" / "equilibrium_only_config.json")
    payload["full_vulcan"] = {"unexpected": True}
    config_path = tmp_path / "equilibrium_only_config.json"
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(config_path)


def test_invalid_equilibrium_activation_is_rejected(tmp_path):
    root = Path(__file__).resolve().parents[1]
    payload = _load_raw_json(root / "config" / "equilibrium_only_config.json")
    payload["equilibrium_only"]["model"]["activation"] = "bad_activation"
    config_path = tmp_path / "equilibrium_only_config.json"
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(config_path)


@pytest.mark.parametrize(
    "activation",
    ["relu", "gelu", "silu", "tanh", "elu", "selu", "softplus", "leaky_relu"],
)
def test_supported_activations_validate_for_both_model_families(tmp_path, activation):
    root = Path(__file__).resolve().parents[1]
    equilibrium = _load_raw_json(root / "config" / "equilibrium_only_config.json")
    equilibrium["equilibrium_only"]["model"]["activation"] = activation
    equilibrium_path = tmp_path / f"equilibrium_{activation}.json"
    equilibrium_path.write_text(json.dumps(equilibrium, indent=2) + "\n", encoding="utf-8")
    assert load_and_validate_config(equilibrium_path)["training"]["model"]["activation"] == activation

    full_vulcan = _load_raw_json(root / "config" / "full_vulcan_config.json")
    full_vulcan["full_vulcan"]["model"]["activation"] = activation
    full_vulcan_path = tmp_path / f"full_vulcan_{activation}.json"
    full_vulcan_path.write_text(json.dumps(full_vulcan, indent=2) + "\n", encoding="utf-8")
    assert load_and_validate_config(full_vulcan_path)["training"]["model"]["activation"] == activation


def test_invalid_mixed_temperature_profile_probability_is_rejected(tmp_path):
    root = Path(__file__).resolve().parents[1]
    payload = _load_raw_json(root / "config" / "equilibrium_only_config.json")
    payload["temperature_profiles"]["analytic_probability"] = 1.0
    config_path = tmp_path / "equilibrium_only_config.json"
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(config_path)


def test_missing_analytic_sampler_for_mixed_temperature_profiles_is_rejected(tmp_path):
    root = Path(__file__).resolve().parents[1]
    payload = _load_raw_json(root / "config" / "equilibrium_only_config.json")
    payload["temperature_profiles"].pop("analytic_sampler")
    config_path = tmp_path / "equilibrium_only_config.json"
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(config_path)


def test_valid_temperature_profile_filters_are_normalized(tmp_path):
    root = Path(__file__).resolve().parents[1]
    payload = _load_raw_json(root / "config" / "equilibrium_only_config.json")
    payload["temperature_profiles"]["filters"] = {
        "Teq": [1400.0, 1600.0],
        "LogDrag": 0.0,
        "TiOVO": False,
    }
    config_path = tmp_path / "equilibrium_only_config.json"
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    validated = load_and_validate_config(config_path)
    assert validated["temperature_profiles"]["filters"] == {
        "Teq": (1400.0, 1600.0),
        "LogDrag": 0.0,
        "TiOVO": False,
    }


def test_invalid_temperature_profile_filter_key_is_rejected(tmp_path):
    root = Path(__file__).resolve().parents[1]
    payload = _load_raw_json(root / "config" / "equilibrium_only_config.json")
    payload["temperature_profiles"]["filters"] = {"phase": ["global_mean", "disk_mean"]}
    config_path = tmp_path / "equilibrium_only_config.json"
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(config_path)


def test_invalid_analytic_temperature_sampler_exponent_range_is_rejected(tmp_path):
    root = Path(__file__).resolve().parents[1]
    payload = _load_raw_json(root / "config" / "equilibrium_only_config.json")
    payload["temperature_profiles"]["analytic_sampler"]["power_law_n_range"] = [0.0, 0.5]
    config_path = tmp_path / "equilibrium_only_config.json"
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(config_path)


def test_invalid_temperature_profile_filter_range_is_rejected(tmp_path):
    root = Path(__file__).resolve().parents[1]
    payload = _load_raw_json(root / "config" / "equilibrium_only_config.json")
    payload["temperature_profiles"]["filters"] = {"Teq": [1800.0, 1200.0]}
    config_path = tmp_path / "equilibrium_only_config.json"
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(config_path)


def test_explicit_cosine_scheduler_is_preserved(tmp_path):
    root = Path(__file__).resolve().parents[1]
    payload = _load_raw_json(root / "config" / "equilibrium_only_config.json")
    payload["training"]["scheduler"] = {"name": "cosine"}
    config_path = tmp_path / "equilibrium_only_config.json"
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    validated = load_and_validate_config(config_path)
    assert validated["training"]["scheduler"] == {"name": "cosine"}


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (("training", "live_sampling"), "training.live_sampling"),
        (("sampling", "num_time_steps"), "sampling.num_time_steps"),
        (("generation", "target_mode"), "generation.target_mode"),
        (("normalization", "state_method"), "normalization.state_method"),
        (("full_vulcan", "trajectory_sampling"), "full_vulcan.trajectory_sampling"),
    ],
)
def test_removed_legacy_keys_raise_targeted_errors(tmp_path, path, message):
    root = Path(__file__).resolve().parents[1]
    payload = _load_raw_json(root / "config" / "full_vulcan_config.json")
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = {"legacy": True} if path[-1] == "trajectory_sampling" else 1
    config_path = tmp_path / "legacy_config.json"
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError, match=message):
        load_and_validate_config(config_path)
