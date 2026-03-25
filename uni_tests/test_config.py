from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config_utils import DEFAULT_STATE_SPECIES, ConfigValidationError, load_and_validate_config


def test_shipped_config_loads():
    root = Path(__file__).resolve().parents[1]
    config = load_and_validate_config(root / "config" / "config.json")
    assert config["physics_toggles"]["use_photochemistry"] is True
    assert config["data_spec"]["state_species"] == list(DEFAULT_STATE_SPECIES)
    assert "use_photochemistry" in config["data_spec"]["required_global_inputs"]
    assert "atm_base_H2" in config["data_spec"]["required_global_inputs"]
    assert config["data_spec"]["dt_feature_index"] == config["data_spec"]["global_feature_order"].index("log10_dt_s")
    assert float(config["sampling"]["kzz_cm2_s"]) > 0.0
    assert config["generation"]["target_mode"] == "equilibrium_only"
    assert config["inference"]["equilibrium_anchor"]["source"] == "flat"
    assert config["inference"]["equilibrium_anchor"]["split"] == "train"
    assert config["inference"]["equilibrium_anchor"]["step_index"] == 0


def test_invalid_target_mode_is_rejected(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = load_and_validate_config(root / "config" / "config.json")
    config["generation"]["target_mode"] = "bad_mode"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(config_path)
