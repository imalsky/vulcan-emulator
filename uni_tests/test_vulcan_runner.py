from __future__ import annotations

import copy
import pickle

import h5py
import numpy as np
import pytest

from src.sampling import sample_run_specifications
from src.vulcan_runner import (
    _patch_vulcan_cfg,
    convert_vulcan_output_to_hdf5,
    generate_raw_dataset,
    generate_synthetic_raw_runs,
    patch_python_assignments,
)


def test_patch_python_assignments():
    text = "use_photo = False\nnetwork = 'old.txt'\n"
    patched = patch_python_assignments(text, {"use_photo": True, "network": "new.txt", "extra_key": 3})
    assert "use_photo = True" in patched
    assert "network = 'new.txt'" in patched
    assert "extra_key = 3" in patched


def test_generate_synthetic_raw_runs(tiny_config):
    artifact = generate_synthetic_raw_runs(tiny_config, project_root=tiny_config["_project_root"])
    assert len(artifact.run_files) == tiny_config["generation"]["num_runs"]
    assert artifact.manifest_path is not None and artifact.manifest_path.exists()
    assert artifact.coverage_path is not None and artifact.coverage_path.exists()
    with h5py.File(artifact.run_files[0], "r") as handle:
        species = [item.decode("utf-8") for item in handle["inputs/state_species"][:]]
        assert "OH" in species and "H2S" in species and "SO2" in species
        time_s = np.asarray(handle["trajectory/time_s"])
        assert np.all(np.diff(time_s) > 0.0)
        ymix = np.asarray(handle["trajectory/ymix_state"])
        assert ymix.shape[0] == time_s.size
        assert np.all(np.isfinite(ymix))
        spectrum = np.asarray(handle["spectrum/flux_erg_cm2_s_nm"])
        assert spectrum.size == tiny_config["stellar_spectrum"]["num_bins"] or spectrum.size > 10


def test_generate_synthetic_raw_runs_requires_configured_spectrum_template(tiny_config):
    config = copy.deepcopy(tiny_config)
    config["stellar_spectrum"]["template_file"] = str(
        tiny_config["_project_root"] / "does_not_exist_surface_flux.txt"
    )
    with pytest.raises(FileNotFoundError):
        generate_synthetic_raw_runs(config, project_root=config["_project_root"])


def test_vulcan_generation_requires_configured_checkout(tiny_config):
    config = copy.deepcopy(tiny_config)
    config["generation"]["mode"] = "vulcan"
    with pytest.raises(FileNotFoundError):
        generate_raw_dataset(config, project_root=config["_project_root"])


def test_patch_vulcan_cfg_uses_profile_kzz_and_cross_sections(tmp_path, tiny_config):
    spec = sample_run_specifications(
        config=tiny_config,
        project_root=tiny_config["_project_root"],
        num_runs=1,
        seed=5,
    )[0]
    cfg_file = tmp_path / "vulcan_cfg.py"
    cfg_file.write_text(
        "atm_type = 'isothermal'\nKzz_prof = 'const'\nT_cross_sp = []\nconst_Kzz = 1.0\n",
        encoding="utf-8",
    )
    tp_file = tmp_path / "atm" / "profile.txt"
    spectrum_file = tmp_path / "atm" / "stellar_flux" / "surface_flux.txt"
    tp_file.parent.mkdir(parents=True, exist_ok=True)
    spectrum_file.parent.mkdir(parents=True, exist_ok=True)
    tp_file.write_text("", encoding="utf-8")
    spectrum_file.write_text("", encoding="utf-8")

    _patch_vulcan_cfg(
        cfg_file,
        spec=spec,
        config=tiny_config,
        tp_file=tp_file,
        spectrum_file=spectrum_file,
    )

    patched = cfg_file.read_text(encoding="utf-8")
    assert "atm_type = 'file'" in patched
    assert "Kzz_prof = 'file'" in patched
    assert "T_cross_sp = ['H2O', 'H2S', 'SH', 'SO2', 'S2']" in patched


def test_convert_fake_vulcan_output_to_hdf5(tmp_path, tiny_config):
    specs = sample_run_specifications(config=tiny_config, project_root=tiny_config["_project_root"], num_runs=1, seed=5)
    spec = specs[0]
    species = list(tiny_config["data_spec"]["state_species"])
    nt = spec.time_s.size
    nz = spec.pressure_bar.size
    state_dim = len(species)
    ymix_time = np.repeat(spec.initial_ymix[None, :, :], nt, axis=0)
    fake = {
        "variable": {
            "species": species,
            "ymix_time": ymix_time,
            "t_time": spec.time_s,
        },
        "atm": {
            "pco": spec.pressure_bar * 1.0e6,
            "Tco": spec.temperature_k,
            "Kzz": spec.kzz_cm2_s[:-1],
        },
    }
    vul_path = tmp_path / "fake.vul"
    with vul_path.open("wb") as handle:
        pickle.dump(fake, handle, protocol=pickle.HIGHEST_PROTOCOL)
    out_h5 = tmp_path / "converted.h5"
    spec = type(spec)(
        run_id=spec.run_id,
        pressure_bar=spec.pressure_bar,
        temperature_k=spec.temperature_k,
        kzz_cm2_s=spec.kzz_cm2_s,
        initial_ymix=spec.initial_ymix,
        time_s=spec.time_s,
        globals=spec.globals,
        spectrum=spec.spectrum,
        metadata={**spec.metadata, "state_species": species, "output_species": species},
    )
    convert_vulcan_output_to_hdf5(vul_path, output_h5_path=out_h5, spec=spec, config=tiny_config)
    with h5py.File(out_h5, "r") as handle:
        ymix = np.asarray(handle["trajectory/ymix_state"])
        assert ymix.shape == (nt, nz, state_dim)
