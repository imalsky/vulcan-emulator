from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from src.utils.config import ConfigValidationError, load_and_validate_config

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "uni_tests" / "fixtures"


def _load_raw_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, name: str, payload: dict) -> Path:
    config_path = tmp_path / name
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return config_path


def _add_corner_coverage(payload: dict, **overrides) -> None:
    payload["sampling"]["corner_coverage"] = {
        "enabled": True,
        "fraction": 0.2,
        "abundance_quantile_width": 0.2,
        "hot_tmax_k": 2500.0,
        "large_trange_k": 800.0,
        "max_profile_resample_attempts": 50,
        **overrides,
    }


@pytest.mark.parametrize(
    ("config_name", "chemistry_type", "run_root_name"),
    [
        ("exogibbs_luhman16a_10k.json", "exogibbs", "exogibbs_luhman16a_10k"),
        ("vulcan_luhman16a_10k.json", "vulcan", "vulcan_luhman16a_10k"),
    ],
)
def test_luhman16a_10k_configs_use_analytic_bd_profiles(
    config_name: str,
    chemistry_type: str,
    run_root_name: str,
):
    config = load_and_validate_config(ROOT / "config" / config_name)
    assert config["chemistry_type"] == chemistry_type
    assert config["generation"]["num_runs"] == 10000
    assert Path(config["paths"]["raw_root"]).parent.name == run_root_name
    assert config["temperature_profiles"]["source_mode"] == "analytic"
    assert config["roth_sampler"]["enabled"] is False
    assert config["sampling"]["temperature_range_k"] == [200.0, 4500.0]
    assert config["temperature_profiles"]["validation"]["max_temperature_k"] == 4500.0
    sampler = config["temperature_profiles"]["analytic_sampler"]
    assert sampler["power_law_probability"] == 0.3
    assert sampler["t_int_k_range"] == [500.0, 1800.0]
    assert sampler["t_eq_k_range"] == [1.0, 100.0]
    assert sampler["log10_p_trans_bar_range"] == [-5.0, 2.3]
    assert sampler["power_law_t0_range_k"] == [600.0, 2500.0]
    assert sampler["power_law_alpha_range"] == [0.0, 0.15]
    if chemistry_type == "vulcan":
        assert config["vulcan_runtime"]["backend"] == "vulcan_jax"


def test_vulcan_runtime_rejects_unknown_backend(tmp_path):
    payload = _load_raw_json(ROOT / "config" / "vulcan_luhman16a_10k.json")
    payload = copy.deepcopy(payload)
    payload["vulcan"]["runtime"]["backend"] = "bad_backend"
    with pytest.raises(ConfigValidationError, match="backend"):
        load_and_validate_config(_write_config(tmp_path, "bad_backend.json", payload))


def test_config_requires_run_root(tmp_path):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload["paths"].pop("run_root")
    with pytest.raises(ConfigValidationError, match="paths.run_root"):
        load_and_validate_config(_write_config(tmp_path, "missing_run_root.json", payload))


def test_config_rejects_legacy_data_root_keys(tmp_path):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload["paths"]["raw_root"] = "data/raw/fastchem"
    payload["paths"]["processed_root"] = "data/processed/fastchem_mlp"
    with pytest.raises(ConfigValidationError, match="no longer supported|run_root only"):
        load_and_validate_config(_write_config(tmp_path, "split_roots.json", payload))


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("model", "activation"),
        ("model", "dropout_rate"),
        ("training", "scheduler"),
    ],
)
def test_required_hyperparameters_must_be_explicit(tmp_path, section, key):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload = copy.deepcopy(payload)
    payload[section].pop(key)
    with pytest.raises(ConfigValidationError, match=section):
        load_and_validate_config(_write_config(tmp_path, "missing_hparam.json", payload))


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


def test_fastchem_rejects_vulcan_only_sampling_keys(tmp_path):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload["sampling"]["planet_radius_range_cm"] = [7.0e9, 1.0e10]
    with pytest.raises(ConfigValidationError, match="chemistry_type='fastchem'"):
        load_and_validate_config(_write_config(tmp_path, "fastchem_planet_radius.json", payload))


def test_fastchem_rejects_sub_100_k_temperature_floor(tmp_path):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload["temperature_profiles"]["validation"]["min_temperature_k"] = 99.0
    with pytest.raises(ConfigValidationError, match="min_temperature_k.*100.0"):
        load_and_validate_config(_write_config(tmp_path, "fastchem_cold_validation.json", payload))


def test_fastchem_rejects_sub_100_k_sampling_range(tmp_path):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload["sampling"]["temperature_range_k"] = [99.0, 3000.0]
    with pytest.raises(ConfigValidationError, match="temperature_range_k.*100.0"):
        load_and_validate_config(_write_config(tmp_path, "fastchem_cold_sampling.json", payload))


@pytest.mark.parametrize(
    "overrides",
    [
        {"fraction": -0.1},
        {"fraction": 1.1},
        {"abundance_quantile_width": 0.0},
        {"abundance_quantile_width": 0.6},
        {"hot_tmax_k": 50.0},
        {"hot_tmax_k": 4000.0},
        {"large_trange_k": 3000.0},
    ],
)
def test_corner_coverage_rejects_invalid_values(tmp_path, overrides):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    _add_corner_coverage(payload, **overrides)
    with pytest.raises(ConfigValidationError, match="corner_coverage"):
        load_and_validate_config(_write_config(tmp_path, "invalid_corner_coverage.json", payload))


def test_invalid_mixed_temperature_profile_probability_is_rejected(tmp_path):
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload["temperature_profiles"]["analytic_probability"] = 1.5
    with pytest.raises(ConfigValidationError, match="analytic_probability"):
        load_and_validate_config(_write_config(tmp_path, "invalid_temp_prob.json", payload))


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


def test_power_law_probability_requires_t0_and_alpha_ranges(tmp_path):
    """Cross-field schema rule: enabling the power-law branch requires both ranges."""
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    sampler = payload["temperature_profiles"]["analytic_sampler"]
    sampler["power_law_probability"] = 0.1
    # power_law_t0_range_k and power_law_alpha_range deliberately omitted
    with pytest.raises(ConfigValidationError, match="power_law_t0_range_k|power_law_alpha_range"):
        load_and_validate_config(_write_config(tmp_path, "missing_power_law_ranges.json", payload))


def test_huber_loss_requires_huber_delta(tmp_path):
    """Loss-type cross-field validation — the Huber path needs huber_delta_log10."""
    payload = _load_raw_json(FIXTURE_ROOT / "fastchem_transformer_config.json")
    payload["training"]["loss"] = {
        "type": "huber",
        "lambda_z": 1.0,
        "lambda_log10_huber": 0.25,
    }
    with pytest.raises(ConfigValidationError, match="huber_delta_log10"):
        load_and_validate_config(_write_config(tmp_path, "missing_huber_delta.json", payload))


def test_public_surfaces_default_to_shipped_luhman16a_config():
    """The shell scripts, CLI, tuning entrypoint, and docs must agree on
    the default shipped config name. Drift here is a contract bug.
    """
    cli_text = (ROOT / "src" / "utils" / "cli.py").read_text(encoding="utf-8")
    tuning_text = (ROOT / "src" / "tuning" / "__main__.py").read_text(encoding="utf-8")
    run_pbs_text = (ROOT / "supercomputer_cmds" / "run.pbs").read_text(encoding="utf-8")

    assert 'default="config/vulcan_luhman16a_10k.json"' in cli_text
    assert 'default="config/vulcan_luhman16a_10k.json"' in tuning_text
    assert 'CONFIG_PATH="${CONFIG_PATH:-config/vulcan_luhman16a_10k.json}"' in run_pbs_text

    for path in (ROOT / "docs" / "README.md", ROOT / "docs" / "config_guide.md"):
        text = path.read_text(encoding="utf-8")
        assert "config/vulcan_luhman16a_10k.json" in text
