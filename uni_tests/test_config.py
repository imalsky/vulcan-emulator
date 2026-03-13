from __future__ import annotations

from pathlib import Path

from src.config_utils import DEFAULT_STATE_SPECIES, load_and_validate_config


def test_shipped_config_loads():
    root = Path(__file__).resolve().parents[1]
    config = load_and_validate_config(root / "config" / "config.json")
    assert config["physics_toggles"]["use_photochemistry"] is True
    assert config["data_spec"]["state_species"] == list(DEFAULT_STATE_SPECIES)
    assert "use_photochemistry" in config["data_spec"]["required_global_inputs"]
    assert "atm_base_H2" in config["data_spec"]["required_global_inputs"]
    assert config["data_spec"]["dt_feature_index"] == config["data_spec"]["global_feature_order"].index("log10_dt_s")
    assert float(config["sampling"]["kzz_cm2_s"]) > 0.0
    assert config["inference"]["equilibrium_anchor"]["source"] == "trajectory"
    assert config["inference"]["equilibrium_anchor"]["split"] == "train"
    assert config["inference"]["equilibrium_anchor"]["step_index"] == 0
