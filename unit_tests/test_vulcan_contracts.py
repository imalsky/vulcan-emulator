#!/usr/bin/env python3
"""Unit tests for bundled VULCAN/runtime contract validation."""

from __future__ import annotations

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sampling import RunSpec
from vulcan_runner import (
    VulcanRuntimeError,
    WorkerSettings,
    _extract_run_payload,
    validate_target_species_available,
)

VULCAN_SOURCE = PROJECT_ROOT.parent / "VULCAN-master"


def _load_config(path: Path) -> dict:
    """Load one config file used by the VULCAN runtime contract tests."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class VulcanContractTests(unittest.TestCase):
    """Unit tests for bundled VULCAN integration assumptions."""

    def test_bundled_vulcan_supports_shipped_target_species(self) -> None:
        for config_name in ("config.json", "tiny_train_smoke.json"):
            config = _load_config(PROJECT_ROOT / "config" / config_name)
            validate_target_species_available(
                VULCAN_SOURCE,
                tuple(config["data_spec"]["target_species"]),
            )

    def test_multiline_species_list_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_species_") as tmpdir_name:
            vulcan_source = Path(tmpdir_name)
            (vulcan_source / "chem_funs.py").write_text(
                "spec_list = [\n"
                "    'H2',\n"
                "    'He',\n"
                "    'H2O',\n"
                "]\n",
                encoding="utf-8",
            )

            validate_target_species_available(vulcan_source, ("H2", "He"))

    def test_missing_target_species_are_reported_early(self) -> None:
        with self.assertRaisesRegex(
            VulcanRuntimeError,
            "Configured target species are not available",
        ):
            validate_target_species_available(VULCAN_SOURCE, ("H2", "H2S"))

    def test_extract_run_payload_preserves_initial_ymix_axis_order(self) -> None:
        species = ["H2", "He", "H2O", "CO2"]
        run_spec = RunSpec(
            run_id=3,
            pressure_bar=np.array([10.0, 1.0], dtype=np.float64),
            temperature_k=np.array([900.0, 800.0], dtype=np.float64),
            kzz_cm2_s=np.array([1.0e9, 1.0e9], dtype=np.float64),
            gravity_cm_s2=1.0e3,
            metallicity_log10=0.0,
            c_to_o=0.55,
            abundances={
                "O_H": 5.37e-4,
                "C_H": 2.95e-4,
                "N_H": 7.08e-5,
                "S_H": 1.41e-5,
                "He_H": 8.38e-2,
                "fastchem_met_scale": 1.0,
            },
            tp_params={},
            kzz_params={},
        )
        settings = WorkerSettings(
            vulcan_source=str(VULCAN_SOURCE),
            worker_root="unused",
            runs_root="unused",
            target_species=("H2", "H2O"),
            use_transport=True,
            boundary_conditions=None,
            use_condensation_optional=True,
            save_evo_frq=1,
            snapshots_per_run=2,
            keep_vulcan_outputs_debug=False,
            run_timeout_seconds=1,
            num_workers=1,
        )
        y_ini = np.array(
            [
                [2.0, 1.0, 3.0, 4.0],
                [5.0, 1.0, 4.0, 10.0],
            ],
            dtype=np.float64,
        )  # shape: (nz=2, species=4)
        y_time = np.array(
            [
                [
                    [4.0, 1.0, 2.0, 3.0],
                    [8.0, 1.0, 5.0, 6.0],
                ],
                [
                    [6.0, 1.0, 2.0, 1.0],
                    [9.0, 1.0, 6.0, 4.0],
                ],
            ],
            dtype=np.float64,
        )  # shape: (time=2, nz=2, species=4)
        payload_data = {
            "variable": {
                "species": species,
                "y_ini": y_ini,
                "y_time": y_time,
                "t_time": np.array([1.0, 100.0], dtype=np.float64),
            }
        }

        with tempfile.TemporaryDirectory(prefix="ve_payload_") as tmpdir_name:
            output_file = Path(tmpdir_name) / "run.vul"
            with output_file.open("wb") as handle:
                pickle.dump(payload_data, handle)

            payload = _extract_run_payload(output_file, run_spec, settings)

        expected_initial = np.array(
            [
                [0.2, 0.3],
                [0.25, 0.2],
            ],
            dtype=np.float64,
        )
        self.assertEqual(payload["initial_ymix"].shape, (2, 2))
        np.testing.assert_allclose(payload["initial_ymix"], expected_initial)


if __name__ == "__main__":
    unittest.main()
