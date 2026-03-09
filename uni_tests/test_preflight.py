#!/usr/bin/env python3
"""Preflight tests for VULCAN runtime smoke validation."""

from __future__ import annotations

import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vulcan_runner import SpeciesSelection, VulcanRuntimeError, WorkerSettings, preflight_vulcan_source


def _settings(root: Path) -> WorkerSettings:
    return WorkerSettings(
        vulcan_source=str(root),
        worker_root=str(root / "workers"),
        raw_root=str(root / "raw"),
        species=SpeciesSelection(
            state_species=("H2", "H2O"),
            output_species=("H2", "H2O"),
        ),
        boundary_conditions=None,
        use_eddy_diffusion=True,
        use_molecular_diffusion=True,
        use_upwind_molecular_diffusion=False,
        use_condensation=False,
        use_settling=False,
        use_initial_cold_trap=True,
        use_sat_surface_h2o=False,
        use_lowT_limit_rates=False,
        use_adaptive_rtol=True,
        ini_mix="EQ",
        atm_base="H2",
        save_evo_frq=1,
        keep_vulcan_outputs_debug=False,
        run_timeout_seconds=1,
        num_workers=1,
        runtime=1.0e6,
        dt_min=1.0e-14,
        dt_max=1.0e3,
        count_max=32,
        trun_min=0.0,
        count_min=0,
        max_trajectory_snapshots=0,
    )


def _write_minimal_vulcan_tree(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "vulcan.py").write_text("print('stub')\n", encoding="utf-8")
    cfg_lines = [
        "save_evolution = False",
        "save_evo_frq = 1",
        "runtime = 1.0",
        "dt_min = 1e-14",
        "dt_max = 1e2",
        "count_max = 1",
        "trun_min = 0.0",
        "count_min = 0",
        "use_lowT_limit_rates = False",
        "use_photo = False",
        "use_ion = False",
        "use_live_plot = False",
        "use_live_flux = False",
        "use_plot_end = False",
        "use_plot_evo = False",
        "use_save_movie = False",
        "use_flux_movie = False",
        "use_print_prog = False",
        "output_humanread = False",
        "atm_type = 'file'",
        "Kzz_prof = 'file'",
        "vz_prof = 'const'",
        "const_vz = 0.0",
        "atm_file = 'atm/generated/preflight_profile.txt'",
        "out_name = 'preflight_smoke.vul'",
        "output_dir = 'output/'",
        "plot_dir = 'plot/'",
        "movie_dir = 'plot/movie/'",
        "ini_mix = 'EQ'",
        "use_ini_cold_trap = True",
        "atm_base = 'H2'",
        "use_solar = False",
        "use_Kzz = True",
        "use_moldiff = True",
        "use_vm_mol = False",
        "use_vz = False",
        "use_topflux = False",
        "use_botflux = False",
        "top_BC_flux_file = 'atm/BC_top.txt'",
        "bot_BC_flux_file = 'atm/BC_bot.txt'",
        "use_fix_sp_bot = {}",
        "use_sat_surfaceH2O = False",
        "use_condense = False",
        "use_settling = False",
        "use_adapt_rtol = True",
        "nz = 16",
        "P_b = 1.0e9",
        "P_t = 1.0",
        "gs = 1.0e3",
        "O_H = 1.0",
        "C_H = 1.0",
        "N_H = 1.0",
        "S_H = 1.0",
        "He_H = 1.0",
        "fastchem_met_scale = 1.0",
    ]
    (root / "vulcan_cfg.py").write_text("\n".join(cfg_lines) + "\n", encoding="utf-8")
    fastchem_dir = root / "fastchem_vulcan"
    fastchem_dir.mkdir(parents=True, exist_ok=True)
    fastchem = fastchem_dir / "fastchem"
    fastchem.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fastchem.chmod(fastchem.stat().st_mode | stat.S_IXUSR)
    return root


class PreflightTests(unittest.TestCase):
    """Contract tests for VULCAN runtime preflight validation."""

    def test_missing_vulcan_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_missing_vulcan_") as tmpdir_name:
            missing = Path(tmpdir_name) / "not_here"
            with self.assertRaises(VulcanRuntimeError):
                preflight_vulcan_source(
                    missing,
                    settings=_settings(missing),
                    timeout_seconds=1,
                )

    def test_smoke_timeout_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_preflight_timeout_") as tmpdir_name:
            root = _write_minimal_vulcan_tree(Path(tmpdir_name) / "vulcan")
            with mock.patch(
                "vulcan_runner.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["python", "vulcan.py", "-n"], timeout=1),
            ):
                with self.assertRaises(VulcanRuntimeError):
                    preflight_vulcan_source(
                        root,
                        settings=_settings(root),
                        timeout_seconds=1,
                    )

    def test_smoke_nonzero_exit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_preflight_fail_") as tmpdir_name:
            root = _write_minimal_vulcan_tree(Path(tmpdir_name) / "vulcan")
            failed = subprocess.CompletedProcess(
                args=[sys.executable, "vulcan.py", "-n"],
                returncode=1,
                stdout="stdout",
                stderr="stderr",
            )
            with mock.patch("vulcan_runner.subprocess.run", return_value=failed):
                with self.assertRaises(VulcanRuntimeError):
                    preflight_vulcan_source(
                        root,
                        settings=_settings(root),
                        timeout_seconds=1,
                    )


if __name__ == "__main__":
    unittest.main()
