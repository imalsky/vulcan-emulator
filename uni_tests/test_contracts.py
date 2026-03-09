#!/usr/bin/env python3
"""Contract tests for config validation and processed-data provenance."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_utils import ConfigValidationError, load_and_validate_config
from provenance import (
    PROCESSED_FINGERPRINT_FILENAME,
    build_processed_fingerprint,
    stable_config_sha256,
    validate_processed_artifacts,
)
from trainer import TrainingError, _cast_optimizer_state, validate_processed_split_contract
from vulcan_runner import VulcanRuntimeError, resolve_boundary_conditions

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


def _load_base_config() -> dict:
    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class ContractTests(unittest.TestCase):
    """Regression tests for config validation and processed-data contracts."""

    def test_missing_required_paths_key_is_rejected(self) -> None:
        config = _load_base_config()
        config["paths"].pop("logs_root")
        with tempfile.TemporaryDirectory(prefix="ve_cfg_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ConfigValidationError):
                load_and_validate_config(config_path)

    def test_boundary_conditions_section_required_when_enabled(self) -> None:
        config = _load_base_config()
        config["physics_toggles"]["use_boundary_conditions"] = True
        config.pop("boundary_conditions", None)
        with tempfile.TemporaryDirectory(prefix="ve_cfg_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ConfigValidationError):
                load_and_validate_config(config_path)

    def test_boundary_condition_files_must_exist(self) -> None:
        config = _load_base_config()
        config["physics_toggles"]["use_boundary_conditions"] = True
        config["boundary_conditions"] = {
            "use_topflux": True,
            "use_botflux": False,
            "top_BC_flux_file": "atm/does_not_exist.txt",
            "bot_BC_flux_file": "atm/BC_bot_mars.txt",
            "use_fix_sp_bot": {},
        }
        with tempfile.TemporaryDirectory(prefix="ve_cfg_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            loaded = load_and_validate_config(config_path)
            with self.assertRaises(VulcanRuntimeError):
                resolve_boundary_conditions(loaded, Path(tmpdir_name))

    def test_log10_dt_globals_cannot_use_log_based_normalization(self) -> None:
        config = _load_base_config()
        config["normalization"]["global_methods"]["log10_dt_s"] = "log-standard"
        with tempfile.TemporaryDirectory(prefix="ve_cfg_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ConfigValidationError):
                load_and_validate_config(config_path)

    def test_invalid_processed_fingerprint_is_rejected(self) -> None:
        config = _load_base_config()
        config["paths"]["data_root"] = "data"
        with tempfile.TemporaryDirectory(prefix="ve_prov_") as tmpdir_name:
            root = Path(tmpdir_name)
            data_root = root / "data"
            processed_root = data_root / "processed"
            raw_run = data_root / "raw" / "run_000000.h5"
            raw_run.parent.mkdir(parents=True, exist_ok=True)
            raw_run.write_bytes(b"stub")

            split_meta = {
                "split": "train",
                "num_runs": 1,
                "max_steps": 2,
                "total_valid_candidates": 1,
                "sequence_length": 2,
                "input_dim": 5,
                "global_dim": 4,
                "global_static_dim": 3,
                "dt_feature_index": 3,
                "target_dim": 2,
                "state_dim": 2,
                "sequence_feature_order": ["pressure_bar", "temperature_k", "kzz_cm2_s", "anchor_ymix:H2", "anchor_ymix:He"],
                "global_feature_order": ["gravity_cm_s2", "metallicity_log10", "c_to_o", "log10_dt_s"],
                "global_static_feature_order": ["gravity_cm_s2", "metallicity_log10", "c_to_o"],
                "state_species_order": ["H2", "He"],
                "output_species_order": ["H2", "He"],
                "output_from_state_indices": [0, 1],
                "normalization_fingerprint": "a" * 64,
                "dt_min_s": 1.0,
                "dt_max_s": 10.0,
            }
            for split_name in ("train", "val", "test"):
                split_dir = processed_root / split_name
                split_dir.mkdir(parents=True, exist_ok=True)
                meta = deepcopy(split_meta)
                meta["split"] = split_name
                (split_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

            normalization_path = processed_root / "normalization_metadata.json"
            normalization_path.write_text(json.dumps({"epsilon": 1e-30}), encoding="utf-8")
            summary_path = processed_root / "processed_summary.json"
            summary_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
            manifest_path = data_root / "dataset_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "num_runs": 1,
                        "run_files": ["data/raw/run_000000.h5"],
                        "split": {"train": [0], "val": [], "test": []},
                        "state_species": ["H2", "He"],
                        "output_species": ["H2", "He"],
                    }
                ),
                encoding="utf-8",
            )
            split_path = data_root / "splits.json"
            split_path.write_text(json.dumps({"train": [0], "val": [], "test": []}), encoding="utf-8")

            fingerprint = build_processed_fingerprint(
                config=config,
                project_root=root,
                raw_run_files=[raw_run],
                manifest_path=manifest_path,
                split_path=split_path,
                normalization_path=normalization_path,
                summary_path=summary_path,
                split_metadata_paths={
                    split_name: processed_root / split_name / "metadata.json"
                    for split_name in ("train", "val", "test")
                },
            )
            fingerprint["config_sha256"] = "0" * 64
            (processed_root / PROCESSED_FINGERPRINT_FILENAME).write_text(json.dumps(fingerprint), encoding="utf-8")

            paths = SimpleNamespace(root=root, data_root=data_root, processed_root=processed_root)
            with self.assertRaises(RuntimeError):
                validate_processed_artifacts(config=config, paths=paths)

    def test_mismatched_species_order_is_rejected(self) -> None:
        config = _load_base_config()
        state_species = list(config["data_spec"]["state_species"])
        output_species = list(config["data_spec"]["output_species"])
        global_feature_order = list(config["data_spec"]["required_global_inputs"])
        state_dim = len(state_species)
        target_dim = len(output_species)
        expected_sequence_order = [
            "pressure_bar",
            "temperature_k",
            "kzz_cm2_s",
            *[f"anchor_ymix:{species_name}" for species_name in state_species],
        ]
        metadata = {
            "sequence_length": 16,
            "input_dim": 3 + state_dim,
            "global_dim": len(global_feature_order),
            "global_static_dim": len(global_feature_order) - 1,
            "dt_feature_index": global_feature_order.index("log10_dt_s"),
            "target_dim": target_dim,
            "state_dim": state_dim,
            "num_runs": 2,
            "max_steps": 8,
            "total_valid_candidates": 6,
            "sequence_feature_order": expected_sequence_order,
            "global_feature_order": global_feature_order,
            "global_static_feature_order": [name for name in global_feature_order if name != "log10_dt_s"],
            "state_species_order": state_species,
            "output_species_order": output_species,
            "output_from_state_indices": list(range(target_dim)),
            "normalization_fingerprint": "a" * 64,
            "dt_min_s": 1.0,
            "dt_max_s": 10.0,
        }
        bad_train_meta = deepcopy(metadata)
        bad_train_meta["output_species_order"] = [output_species[1], output_species[0], *output_species[2:]]
        with self.assertRaises(TrainingError):
            validate_processed_split_contract(
                config=config,
                train_meta=bad_train_meta,
                val_meta=metadata,
                test_meta=metadata,
                expected_norm_fingerprint="a" * 64,
            )

    def test_live_sampling_budgets_do_not_change_processed_config_fingerprint(self) -> None:
        config = _load_base_config()
        baseline = stable_config_sha256(config)
        config["training"]["live_sampling"]["train_pairs_per_run_per_epoch"] += 7
        config["training"]["live_sampling"]["eval_pairs_per_run"] += 3
        self.assertEqual(stable_config_sha256(config), baseline)

    def test_dt_bounds_do_change_processed_config_fingerprint(self) -> None:
        config = _load_base_config()
        baseline = stable_config_sha256(config)
        config["trajectory_sampling"]["dt_max_s"] = float(config["trajectory_sampling"]["dt_max_s"]) * 2.0
        self.assertNotEqual(stable_config_sha256(config), baseline)

    def test_optimizer_state_cast_helper_applies_requested_dtype(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        optimizer = torch.optim.AdamW([parameter], lr=1.0e-3)
        loss = (parameter.square()).sum()
        loss.backward()
        optimizer.step()
        _cast_optimizer_state(optimizer, torch.float64)
        float_states = [
            value
            for state in optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value) and torch.is_floating_point(value)
        ]
        self.assertTrue(float_states)
        for value in float_states:
            self.assertEqual(value.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
