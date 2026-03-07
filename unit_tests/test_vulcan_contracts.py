#!/usr/bin/env python3
"""Unit tests for VULCAN runtime contract helpers."""

from __future__ import annotations

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
    SpeciesSelection,
    VulcanRuntimeError,
    WorkerSettings,
    _apply_run_config,
    _extract_run_payload,
    validate_species_available,
)


def _run_spec() -> RunSpec:
    return RunSpec(
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


def _settings(tmpdir: Path) -> WorkerSettings:
    return WorkerSettings(
        vulcan_source=str(tmpdir),
        worker_root=str(tmpdir / "workers"),
        runs_root=str(tmpdir / "runs"),
        species=SpeciesSelection(
            state_species=("H2", "H2O", "CO2"),
            output_species=("H2O", "H2"),
        ),
        use_transport=True,
        boundary_conditions=None,
        use_condensation_optional=True,
        save_evo_frq=1,
        keep_vulcan_outputs_debug=False,
        run_timeout_seconds=1,
        num_workers=1,
        runtime=1.0e6,
        dt_min=1.0e-14,
        dt_max=1.0e3,
        count_max=123,
        trun_min=0.0,
        count_min=0,
        y_time_freq=7,
    )


class VulcanContractTests(unittest.TestCase):
    """Unit tests for bundled VULCAN integration assumptions."""

    def test_multiline_species_list_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_species_") as tmpdir_name:
            vulcan_source = Path(tmpdir_name)
            (vulcan_source / "chem_funs.py").write_text(
                "spec_list = [\n"
                "    'H2',\n"
                "    'He',\n"
                "    'H2O',\n"
                "    'CO2',\n"
                "]\n",
                encoding="utf-8",
            )
            validate_species_available(
                vulcan_source,
                state_species=("H2", "H2O"),
                output_species=("H2O",),
            )

    def test_missing_species_are_reported_early(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_species_missing_") as tmpdir_name:
            vulcan_source = Path(tmpdir_name)
            (vulcan_source / "chem_funs.py").write_text("spec_list = ['H2', 'He']\n", encoding="utf-8")
            with self.assertRaisesRegex(VulcanRuntimeError, "Configured species are not available"):
                validate_species_available(
                    vulcan_source,
                    state_species=("H2", "H2O"),
                    output_species=("H2",),
                )

    def test_apply_run_config_includes_runtime_overrides(self) -> None:
        baseline_cfg = "\n".join(
            [
                "save_evolution = False",
                "save_evo_frq = 10",
                "y_time_freq = 10",
                "runtime = 1.0",
                "dt_min = 1e-6",
                "dt_max = 1e2",
                "count_max = 1",
                "trun_min = 1.0",
                "count_min = 1",
                "use_photo = True",
                "use_ion = True",
                "use_live_plot = True",
                "use_live_flux = True",
                "use_plot_end = True",
                "use_plot_evo = True",
                "use_save_movie = True",
                "use_flux_movie = True",
                "use_print_prog = True",
                "output_humanread = True",
                "atm_type = 'P_ana'",
                "Kzz_prof = 'const'",
                "atm_file = 'atm/atm_HD189_Kzz.txt'",
                "out_name = 'old.vul'",
                "output_dir = 'output/'",
                "plot_dir = 'plot/'",
                "movie_dir = 'plot/movie/'",
                "ini_mix = 'EQ'",
                "use_solar = True",
                "use_Kzz = False",
                "use_moldiff = False",
                "use_topflux = False",
                "use_botflux = False",
                "use_fix_sp_bot = {}",
                "use_condense = False",
                "use_settling = False",
                "nz = 10",
                "P_b = 1.0",
                "P_t = 1.0e-8",
                "gs = 1.0e3",
                "O_H = 1.0",
                "C_H = 1.0",
                "N_H = 1.0",
                "S_H = 1.0",
                "He_H = 1.0",
                "fastchem_met_scale = 1.0",
            ]
        )
        with tempfile.TemporaryDirectory(prefix="ve_cfg_apply_") as tmpdir_name:
            settings = _settings(Path(tmpdir_name))
            rendered = _apply_run_config(
                baseline_cfg=baseline_cfg,
                run_spec=_run_spec(),
                settings=settings,
                atm_relpath="atm/generated/run_000003.txt",
                out_name="run_000003.vul",
            )
        self.assertIn("runtime = 1000000.0", rendered)
        self.assertIn("dt_max = 1000.0", rendered)
        self.assertIn("count_max = 123", rendered)
        self.assertIn("y_time_freq = 7", rendered)
        self.assertIn("save_evolution = True", rendered)

    def test_extract_run_payload_preserves_full_trajectory_and_species_order(self) -> None:
        species = ["H2", "He", "H2O", "CO2"]
        y_ini = np.array(
            [
                [2.0, 1.0, 3.0, 4.0],
                [5.0, 1.0, 4.0, 10.0],
            ],
            dtype=np.float64,
        )
        y_time = np.array(
            [
                [
                    [4.0, 1.0, 2.0, 3.0],
                    [8.0, 1.0, 5.0, 6.0],
                ],
                [
                    [5.0, 1.0, 3.0, 1.0],
                    [9.0, 1.0, 6.0, 4.0],
                ],
                [
                    [6.0, 1.0, 4.0, 2.0],
                    [10.0, 1.0, 7.0, 5.0],
                ],
            ],
            dtype=np.float64,
        )
        payload_data = {
            "variable": {
                "species": species,
                "y_ini": y_ini,
                "y_time": y_time,
                "t_time": np.array([1.0, 1.0, 100.0], dtype=np.float64),
            }
        }
        with tempfile.TemporaryDirectory(prefix="ve_payload_") as tmpdir_name:
            tmpdir = Path(tmpdir_name)
            output_file = tmpdir / "run.vul"
            with output_file.open("wb") as handle:
                pickle.dump(payload_data, handle)
            payload = _extract_run_payload(output_file, _run_spec(), _settings(tmpdir))

        self.assertEqual(payload["state_species"], ["H2", "H2O", "CO2"])
        self.assertEqual(payload["output_species"], ["H2O", "H2"])
        np.testing.assert_allclose(payload["time_s"], np.array([0.0, 1.0, 100.0], dtype=np.float64))
        self.assertEqual(payload["ymix_state"].shape, (3, 2, 3))
        self.assertEqual(payload["ymix_output"].shape, (3, 2, 2))
        expected_initial_state = np.array(
            [
                [0.2, 0.3, 0.4],
                [0.25, 0.2, 0.5],
            ],
            dtype=np.float64,
        )
        expected_initial_output = np.array(
            [
                [0.3, 0.2],
                [0.2, 0.25],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(payload["ymix_state"][0], expected_initial_state)
        np.testing.assert_allclose(payload["ymix_output"][0], expected_initial_output)


if __name__ == "__main__":
    unittest.main()
