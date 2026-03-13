from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_utils import load_and_validate_config  # noqa: E402
from src.spectrum import generate_wasp39b_template, write_vulcan_spectrum_txt  # noqa: E402


@pytest.fixture()
def tiny_config(tmp_path):
    config = load_and_validate_config(ROOT / "config" / "config.json")
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
    config["paths"]["jax_export_root"] = str(tmp_path / "jax")
    config["paths"]["vulcan_source_root"] = str(tmp_path / "VULCAN")
    config["stellar_spectrum"]["template_file"] = str(spectrum_file)
    config["stellar_spectrum"]["template_name"] = "test_surface_flux"
    config["_project_root"] = ROOT
    config["generation"]["mode"] = "synthetic"
    config["generation"]["num_runs"] = 4
    config["generation"]["seed"] = 11
    config["generation"]["parallel_workers"] = 1
    config["sampling"]["num_levels"] = 12
    config["sampling"]["num_time_steps"] = 8
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
    return config
