from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_and_validate_config  # noqa: E402
from src.data_generation.spectrum import generate_wasp39b_template, write_vulcan_spectrum_txt  # noqa: E402


def _write_full_vulcan_test_config(config_path: Path) -> None:
    fixture_glob = str(ROOT / "uni_tests" / "fixtures" / "pt_profiles" / "*.dat")
    config_path.write_text(
        json.dumps(
            {
                "task": {"kind": "full_vulcan"},
                "paths": {
                    "raw_root": "data/raw",
                    "processed_root": "data/processed",
                    "checkpoints_root": "models/default",
                    "vulcan_source_root": "../VULCAN-master",
                },
                "data_spec": {
                    "state_species": [
                        "H2",
                        "He",
                        "H",
                        "O",
                        "OH",
                        "H2O",
                        "CO",
                        "CO2",
                        "CH4",
                        "N2",
                        "NH3",
                        "H2S",
                        "SH",
                        "S",
                        "SO",
                        "SO2",
                        "S2",
                    ],
                    "output_species": [
                        "H2",
                        "He",
                        "H",
                        "O",
                        "OH",
                        "H2O",
                        "CO",
                        "CO2",
                        "CH4",
                        "N2",
                        "NH3",
                        "H2S",
                        "SH",
                        "S",
                        "SO",
                        "SO2",
                        "S2",
                    ],
                },
                "sampling": {
                    "num_levels": 64,
                    "pressure_top_bar": 1e-07,
                    "pressure_bottom_bar": 100.0,
                    "temperature_range_k": [1.0, 4000.0],
                    "gravity_range_cm_s2": [300.0, 900.0],
                    "metallicity_log10_range": [0.0, 2.0],
                    "c_to_o_range": [0.25, 1.1],
                    "s_to_o_range": [0.005, 0.1],
                    "kzz_cm2_s": 1.0e8,
                    "num_time_steps": 24,
                    "time_step_log10_min_s": 1.0,
                    "time_step_log10_max_s": 5.0,
                },
                "temperature_profiles": {
                    "source_mode": "mixed",
                    "analytic_probability": 0.5,
                    "data_glob": fixture_glob,
                    "filters": {"Teq": [1200.0, 1200.0], "LogMet": 0.0, "TiOVO": False},
                    "validation": {
                        "min_temperature_k": 1.0,
                        "max_temperature_k": 3000.0,
                    },
                    "analytic_sampler": {
                        "reference_gravity_m_s2": 24.79,
                        "t_int_k_normal": {"mean": 500.0, "std": 150.0},
                        "t_irr_k_normal": {"mean": 1800.0, "std": 500.0},
                        "log10_kappa_ir_m2_kg_normal": {"mean": -2.5, "std": 2.5},
                        "power_law_n_range": [0.5, 2.0],
                        "log10_gamma_1_range": [-2.0, 2.0],
                        "log10_gamma_2_range": [-2.0, 2.0],
                        "alpha_range": [0.0, 1.0],
                        "temperature_shift_k_range": [-600.0, 600.0],
                        "convection_probability": 1.0 / 3.0,
                        "adiabatic_gradient_range": [0.25, 0.35],
                    },
                },
                "generation": {
                    "mode": "vulcan",
                    "num_runs": 128,
                    "seed": 7,
                    "overwrite": False,
                    "reuse_raw_if_present": True,
                    "parallel_workers": 8,
                },
                "normalization": {
                    "split": {
                        "train_fraction": 0.67,
                        "val_fraction": 0.17,
                        "test_fraction": 0.16,
                        "seed": 123,
                    },
                    "state_floor": 1e-30,
                    "spectrum_floor": 1e-30,
                    "state_method": "log-standard",
                    "target_method": "log-standard",
                    "log10_dt_method": "standard",
                    "spectrum_method": "log-standard",
                    "sequence_methods": {
                        "pressure_bar": "log-standard",
                        "temperature_k": "standard",
                        "kzz_cm2_s": "log-standard",
                    },
                    "global_methods": {
                        "gravity_cm_s2": "log-standard",
                        "He_H": "log-standard",
                        "C_H": "log-standard",
                        "O_H": "log-standard",
                        "N_H": "log-standard",
                        "S_H": "log-standard",
                        "use_photochemistry": "none",
                        "use_ion_chemistry": "none",
                        "use_eddy_diffusion": "none",
                        "use_molecular_diffusion": "none",
                        "use_upwind_molecular_diffusion": "none",
                        "use_boundary_conditions": "none",
                        "use_condensation": "none",
                        "use_settling": "none",
                        "use_initial_cold_trap": "none",
                        "use_sat_surface_h2o": "none",
                        "use_lowT_limit_rates": "none",
                        "use_adaptive_rtol": "none",
                        "atm_base_H2": "none",
                        "atm_base_N2": "none",
                        "atm_base_O2": "none",
                        "atm_base_CO2": "none",
                        "atm_base_H2O": "none"
                    }
                },
                "training": {
                    "seed": 123,
                    "batch_size": 32,
                    "epochs": 20,
                    "learning_rate": 0.0002,
                    "min_lr": 1e-05,
                    "warmup_epochs": 0,
                    "weight_decay": 1e-05,
                    "gradient_clip": 1.0,
                    "live_sampling": {
                        "train_pairs_per_run_per_epoch": 128,
                        "eval_pairs_per_run": 32,
                    },
                    "loss": {
                        "lambda_z": 1.0,
                        "lambda_phys": 0.1,
                        "lambda_spectrum": 0.01,
                    },
                },
                "full_vulcan": {
                    "model": {
                        "d_model": 128,
                        "nhead": 8,
                        "num_layers": 4,
                        "dim_feedforward": 256,
                        "conditioning_hidden_dim": 256,
                        "film_clamp": 1.5,
                        "output_head_divisor": 2,
                    },
                    "physics_toggles": {
                        "use_photochemistry": True,
                        "use_ion_chemistry": False,
                        "use_eddy_diffusion": True,
                        "use_molecular_diffusion": False,
                        "use_upwind_molecular_diffusion": False,
                        "use_boundary_conditions": False,
                        "use_condensation": False,
                        "use_settling": False,
                        "use_initial_cold_trap": False,
                        "use_sat_surface_h2o": False,
                        "use_lowT_limit_rates": False,
                        "use_adaptive_rtol": False,
                    },
                    "vulcan_runtime": {
                        "python_executable": "python",
                        "cfg_file": "vulcan_cfg.py",
                        "chemistry_file": "thermo/SNCHO_photo_network_2025.txt",
                        "worker_root": "data/vulcan_workers",
                        "regenerate_chem_funs": True,
                        "atm_base": "H2",
                        "t_cross_sp": ["H2O", "H2S", "SH", "SO2", "S2"],
                        "cfg_assignments": {},
                    },
                    "stellar_spectrum": {
                        "enabled": True,
                        "template_name": "wasp39b_frances_surface_flux",
                        "template_file": "assets/spectra/wasp39b/test_surface_flux.txt",
                        "num_bins": 256,
                        "wavelength_min_nm": 0.05,
                        "wavelength_max_nm": 700.0,
                        "encoder_mode": "autoencoder",
                        "latent_dim": 16,
                        "hidden_dim": 64,
                        "teff_k": 5485.0,
                        "radius_rsun": 0.939,
                        "semi_major_axis_au": 0.04858,
                        "zenith_angle_deg": 48.0,
                        "diurnal_factor": 1.0,
                    },
                    "trajectory_sampling": {
                        "dt_min_s": 100.0,
                        "dt_max_s": 100000.0,
                        "min_future_saved_steps": 1,
                        "num_logdt_bins": 8,
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture()
def tiny_config(tmp_path):
    config_path = tmp_path / "full_vulcan_config.json"
    _write_full_vulcan_test_config(config_path)
    config = load_and_validate_config(config_path)
    config = copy.deepcopy(config)

    spectrum_file = tmp_path / "test_surface_flux.txt"
    write_vulcan_spectrum_txt(
        generate_wasp39b_template(
            num_points=64,
            wavelength_min_nm=float(config["stellar_spectrum"]["wavelength_min_nm"]),
            wavelength_max_nm=float(config["stellar_spectrum"]["wavelength_max_nm"]),
            teff_k=float(config["stellar_spectrum"]["teff_k"]),
            radius_rsun=float(config["stellar_spectrum"]["radius_rsun"]),
            semi_major_axis_au=float(config["stellar_spectrum"]["semi_major_axis_au"]),
            name="test_surface_flux",
        ),
        spectrum_file,
    )

    config["paths"]["raw_root"] = str(tmp_path / "raw")
    config["paths"]["processed_root"] = str(tmp_path / "processed")
    config["paths"]["checkpoints_root"] = str(tmp_path / "checkpoints")
    config["paths"]["vulcan_source_root"] = str(tmp_path / "VULCAN")

    config["full_vulcan"]["stellar_spectrum"]["template_file"] = str(spectrum_file)
    config["full_vulcan"]["stellar_spectrum"]["template_name"] = "test_surface_flux"
    config["stellar_spectrum"]["template_file"] = str(spectrum_file)
    config["stellar_spectrum"]["template_name"] = "test_surface_flux"

    config["_project_root"] = ROOT
    config["generation"]["mode"] = "synthetic"
    config["generation"]["num_runs"] = 4
    config["generation"]["seed"] = 11
    config["generation"]["parallel_workers"] = 1
    config["sampling"]["num_levels"] = 12
    config["sampling"]["num_time_steps"] = 8

    config["full_vulcan"]["stellar_spectrum"]["num_bins"] = 32
    config["full_vulcan"]["stellar_spectrum"]["hidden_dim"] = 16
    config["full_vulcan"]["stellar_spectrum"]["latent_dim"] = 4
    config["stellar_spectrum"]["num_bins"] = 32
    config["stellar_spectrum"]["hidden_dim"] = 16
    config["stellar_spectrum"]["latent_dim"] = 4

    config["training"]["batch_size"] = 4
    config["training"]["epochs"] = 1
    config["training"]["live_sampling"]["train_pairs_per_run_per_epoch"] = 4
    config["training"]["live_sampling"]["eval_pairs_per_run"] = 2
    config["training"]["model"]["d_model"] = 16
    config["training"]["model"]["nhead"] = 4
    config["training"]["model"]["num_layers"] = 1
    config["training"]["model"]["dim_feedforward"] = 32
    config["training"]["model"]["conditioning_hidden_dim"] = 32

    config["full_vulcan"]["model"]["d_model"] = 16
    config["full_vulcan"]["model"]["nhead"] = 4
    config["full_vulcan"]["model"]["num_layers"] = 1
    config["full_vulcan"]["model"]["dim_feedforward"] = 32
    config["full_vulcan"]["model"]["conditioning_hidden_dim"] = 32
    return config
