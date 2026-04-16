from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_generation.spectrum import (  # noqa: E402
    generate_blackbody_template,
    write_vulcan_spectrum_txt,
)
from src.utils.config import load_and_validate_config  # noqa: E402


def _write_vulcan_transformer_test_config(config_path: Path) -> None:
    fixture_glob = str(ROOT / "uni_tests" / "fixtures" / "pt_profiles" / "*.dat")
    config_path.write_text(
        json.dumps(
            {
                "chemistry_type": "vulcan",
                "model_type": "transformer",
                "paths": {
                    "run_root": "data/test_vulcan_transformer",
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
                    "num_levels_range": [40, 60],
                    "pressure_top_bar_range": [1e-07, 1e-05],
                    "pressure_bottom_bar_range": [50.0, 500.0],
                    "temperature_range_k": [1.0, 4000.0],
                    "gravity_range_cm_s2": [300.0, 900.0],
                    "planet_radius_range_cm": [7.0e9, 1.1e10],
                    "he_frac_range": [0.06, 0.12],
                    "c_frac_range": [1e-5, 5e-3],
                    "o_frac_range": [1e-5, 5e-3],
                    "n_frac_range": [1e-6, 1e-3],
                    "s_frac_range": [1e-7, 5e-4],
                    "kzz_range_cm2_s": [1.0e6, 1.0e10],
                    "stellar_radius_range_rsun": [0.8, 1.2],
                    "semi_major_axis_range_au": [0.03, 0.06],
                    "zenith_angle_range_deg": [0.0, 89.0],
                    "diurnal_factor_range": [0.5, 1.0],
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
                        "t_int_k_range": [100.0, 800.0],
                        "t_eq_k_range": [300.0, 4000.0],
                        "log10_delta_range": [-6.0, 6.0],
                        "log10_gamma_range": [-2.0, 2.0],
                        "alpha_range": [0.0, 0.99],
                        "log10_p_trans_bar_range": [-5.0, 1.0],
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
                    "target_method": "log-standard",
                    "sequence_methods": {
                        "pressure_bar": "log-standard",
                        "temperature_k": "standard",
                        "kzz_cm2_s": "log-standard",
                    },
                    "global_methods": {
                        "gravity_cm_s2": "log-standard",
                        "planet_radius_cm": "log-standard",
                        "He_H": "standard",
                        "C_H": "log-standard",
                        "O_H": "log-standard",
                        "N_H": "log-standard",
                        "S_H": "log-standard",
                        "r_star_rsun": "log-standard",
                        "semi_major_axis_au": "log-standard",
                        "zenith_angle_deg": "standard",
                        "diurnal_factor": "none",
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
                    "loss": {
                        "lambda_z": 1.0,
                        "lambda_phys": 0.1,
                    },
                },
                "model": {
                    "d_model": 128,
                    "nhead": 8,
                    "num_layers": 4,
                    "dim_feedforward": 256,
                    "conditioning_hidden_dim": 256,
                    "film_clamp": 1.5,
                    "output_head_divisor": 2,
                },
                "vulcan": {
                    "physics_toggles": {
                        "use_photochemistry": False,
                        "use_ion_chemistry": False,
                        "use_eddy_diffusion": True,
                        "use_molecular_diffusion": False,
                        "use_upwind_molecular_diffusion": False,
                        "use_boundary_conditions": False,
                        "use_condensation": False,
                        "use_settling": False,
                        "use_initial_cold_trap": False,
                        "use_sat_surface_h2o": False,
                    },
                    "science_presets": [
                        {
                            "name": "thermochem_h2",
                            "atm_base": "H2",
                            "physics_toggles": {
                                "use_photochemistry": False,
                                "use_eddy_diffusion": True,
                            },
                        },
                        {
                            "name": "thermochem_n2",
                            "atm_base": "N2",
                            "physics_toggles": {
                                "use_photochemistry": False,
                                "use_eddy_diffusion": True,
                            },
                        },
                    ],
                    "runtime": {
                        "chemistry_file": "thermo/SNCHO_photo_network_2025.txt",
                        "rocky": False,
                        "atm_base": "H2",
                        "t_cross_sp": ["H2O", "H2S", "SH", "SO2", "S2"],
                    },
                    "stellar_spectrum": {
                        "template_name": "test_blackbody",
                        "template_file": "assets/spectra/test_surface_flux.txt",
                        "library_glob": "assets/spectra/library/*.txt",
                        "max_tokens": 32,
                        "wavelength_min_nm": 400.0,
                        "wavelength_max_nm": 450.0,
                        "teff_k": 5485.0,
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
    config_path = tmp_path / "vulcan_transformer_config.json"
    _write_vulcan_transformer_test_config(config_path)
    config = load_and_validate_config(config_path)
    config = copy.deepcopy(config)

    spectrum_dir = tmp_path / "spectra"
    spectrum_dir.mkdir(parents=True, exist_ok=True)
    spectrum_file = spectrum_dir / "test_surface_flux.txt"
    spectrum_file_alt = spectrum_dir / "test_surface_flux_alt.txt"
    write_vulcan_spectrum_txt(
        generate_blackbody_template(
            num_points=64,
            wavelength_min_nm=float(config["stellar_spectrum"]["wavelength_min_nm"]),
            wavelength_max_nm=float(config["stellar_spectrum"]["wavelength_max_nm"]),
            teff_k=float(config["stellar_spectrum"].get("teff_k", 5485.0)),
            name="test_surface_flux",
        ),
        spectrum_file,
    )
    write_vulcan_spectrum_txt(
        generate_blackbody_template(
            num_points=64,
            wavelength_min_nm=float(config["stellar_spectrum"]["wavelength_min_nm"]),
            wavelength_max_nm=float(config["stellar_spectrum"]["wavelength_max_nm"]),
            teff_k=float(config["stellar_spectrum"].get("teff_k", 5485.0)) + 250.0,
            name="test_surface_flux_alt",
        ),
        spectrum_file_alt,
    )

    config["paths"]["raw_root"] = str(tmp_path / "dataset" / "raw")
    config["paths"]["processed_root"] = str(tmp_path / "dataset" / "processed")
    config["paths"]["checkpoints_root"] = str(tmp_path / "checkpoints")
    config["paths"]["vulcan_source_root"] = str(tmp_path / "VULCAN")

    config["vulcan"]["stellar_spectrum"]["template_file"] = str(spectrum_file)
    config["vulcan"]["stellar_spectrum"]["template_name"] = "test_surface_flux"
    config["vulcan"]["stellar_spectrum"]["library_glob"] = str(spectrum_dir / "*.txt")
    config["stellar_spectrum"]["template_file"] = str(spectrum_file)
    config["stellar_spectrum"]["template_name"] = "test_surface_flux"
    config["stellar_spectrum"]["library_glob"] = str(spectrum_dir / "*.txt")

    config["_project_root"] = ROOT
    config["generation"]["mode"] = "vulcan"
    config["generation"]["num_runs"] = 4
    config["generation"]["seed"] = 11
    config["generation"]["parallel_workers"] = 1
    config["sampling"]["num_levels_range"] = [12, 12]

    config["vulcan"]["stellar_spectrum"]["max_tokens"] = 32
    config["stellar_spectrum"]["max_tokens"] = 32

    config["training"]["batch_size"] = 4
    config["training"]["epochs"] = 1
    config["training"]["model"]["d_model"] = 16
    config["training"]["model"]["nhead"] = 4
    config["training"]["model"]["num_layers"] = 1
    config["training"]["model"]["dim_feedforward"] = 32
    config["training"]["model"]["conditioning_hidden_dim"] = 32

    config["model"]["d_model"] = 16
    config["model"]["nhead"] = 4
    config["model"]["num_layers"] = 1
    config["model"]["dim_feedforward"] = 32
    config["model"]["conditioning_hidden_dim"] = 32
    return config
