from __future__ import annotations

import copy
import pickle
from pathlib import Path

import h5py
import numpy as np
import pytest

import src.data_generation.sampling as sampling_module
import src.data_generation.generation as vulcan_runner_module
from src.data_generation.sampling import sample_run_specifications
from src.data_generation.generation import (
    build_flat_h2_he_anchor,
    _copy_fastchem_runtime,
    _patch_vulcan_cfg,
    convert_fastchem_output_to_hdf5,
    convert_vulcan_output_to_hdf5,
    generate_raw_dataset,
    generate_synthetic_raw_runs,
    patch_python_assignments,
    run_vulcan_generation,
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
        assert time_s.size == int(tiny_config["sampling"]["num_time_steps"])
        ymix = np.asarray(handle["trajectory/ymix_state"])
        assert ymix.shape[0] == time_s.size
        assert np.all(np.isfinite(ymix))
        reference = np.asarray(handle["inputs/reference_ymix_state"])
        assert np.allclose(ymix[0], reference)
        assert handle["inputs/target_mode"][()].decode("utf-8") == tiny_config["generation"]["target_mode"]
        spectrum = np.asarray(handle["spectrum/flux_erg_cm2_s_nm"])
        assert spectrum.size > 10


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
    config = copy.deepcopy(tiny_config)
    config["generation"]["target_mode"] = "trajectory"
    specs = sample_run_specifications(config=tiny_config, project_root=tiny_config["_project_root"], num_runs=1, seed=5)
    spec = specs[0]
    species = list(tiny_config["data_spec"]["state_species"])
    nt = 3
    nz = spec.pressure_bar.size
    state_dim = len(species)
    reference = np.asarray(spec.initial_ymix, dtype=np.float64)
    ymix_time = np.repeat(reference[None, :, :], nt, axis=0)
    t_time = np.asarray([10.0, 100.0, 1000.0], dtype=np.float64)
    n0 = np.full(nz, 1.0e12, dtype=np.float64)
    fake = {
        "variable": {
            "species": species,
            "ymix_time": ymix_time,
            "y_ini": reference * n0[:, None],
            "t_time": t_time,
        },
        "atm": {
            "pco": spec.pressure_bar * 1.0e6,
            "Tco": spec.temperature_k,
            "Kzz": spec.kzz_cm2_s[:-1],
            "n_0": n0,
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
    convert_vulcan_output_to_hdf5(vul_path, output_h5_path=out_h5, spec=spec, config=config)
    with h5py.File(out_h5, "r") as handle:
        time_s = np.asarray(handle["trajectory/time_s"])
        ymix = np.asarray(handle["trajectory/ymix_state"])
        reference_out = np.asarray(handle["inputs/reference_ymix_state"])
        assert ymix.shape == (nt + 1, nz, state_dim)
        assert np.isclose(time_s[0], 0.0)
        assert np.allclose(ymix[0], reference_out)
        assert np.allclose(ymix[1:], ymix_time)


def test_convert_fake_vulcan_output_to_hdf5_equilibrium_only_shell(tmp_path, tiny_config):
    config = copy.deepcopy(tiny_config)
    config["generation"]["target_mode"] = "equilibrium_only"
    specs = sample_run_specifications(config=config, project_root=config["_project_root"], num_runs=1, seed=5)
    spec = specs[0]
    species = list(config["data_spec"]["state_species"])
    nz = spec.pressure_bar.size
    state_dim = len(species)
    reference = np.asarray(spec.initial_ymix, dtype=np.float64)
    n0 = np.full(nz, 1.0e12, dtype=np.float64)
    fake = {
        "variable": {
            "species": species,
            "ymix_time": np.repeat(reference[None, :, :], 2, axis=0),
            "y_ini": reference * n0[:, None],
            "t_time": np.asarray([10.0, 100.0], dtype=np.float64),
        },
        "atm": {
            "pco": spec.pressure_bar * 1.0e6,
            "Tco": spec.temperature_k,
            "Kzz": spec.kzz_cm2_s[:-1],
            "n_0": n0,
        },
    }
    vul_path = tmp_path / "fake_eq.vul"
    with vul_path.open("wb") as handle:
        pickle.dump(fake, handle, protocol=pickle.HIGHEST_PROTOCOL)
    out_h5 = tmp_path / "converted_eq.h5"
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
    convert_vulcan_output_to_hdf5(vul_path, output_h5_path=out_h5, spec=spec, config=config)
    with h5py.File(out_h5, "r") as handle:
        time_s = np.asarray(handle["trajectory/time_s"])
        ymix = np.asarray(handle["trajectory/ymix_state"])
        reference_out = np.asarray(handle["inputs/reference_ymix_state"])
        assert ymix.shape == (2, nz, state_dim)
        assert np.allclose(time_s, np.array([0.0, 1.0]))
        assert np.allclose(ymix[0], build_flat_h2_he_anchor(species, nz=nz))
        assert np.allclose(ymix[1], reference_out)


def test_convert_fake_fastchem_output_to_hdf5_equilibrium_only_shell(tmp_path, tiny_config):
    config = copy.deepcopy(tiny_config)
    config["generation"]["target_mode"] = "equilibrium_only"
    specs = sample_run_specifications(config=config, project_root=config["_project_root"], num_runs=1, seed=5)
    spec = specs[0]
    species = list(config["data_spec"]["state_species"])
    nz = spec.pressure_bar.size
    reference = build_flat_h2_he_anchor(species, nz=nz)
    idx = {name: i for i, name in enumerate(species)}
    reference[:, idx["H2O"]] = 1.0e-3
    reference[:, idx["CH4"]] = 5.0e-4
    reference[:, idx["H2"]] -= 1.5e-3
    reference /= np.sum(reference, axis=1, keepdims=True)

    fastchem_output = tmp_path / "vulcan_EQ.dat"
    with fastchem_output.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(species) + "\n")
        for row in reference:
            handle.write(" ".join(f"{value:.8e}" for value in row) + "\n")

    out_h5 = tmp_path / "converted_fastchem.h5"
    convert_fastchem_output_to_hdf5(
        fastchem_output,
        output_h5_path=out_h5,
        spec=spec,
        config=config,
    )

    with h5py.File(out_h5, "r") as handle:
        time_s = np.asarray(handle["trajectory/time_s"])
        ymix = np.asarray(handle["trajectory/ymix_state"])
        reference_out = np.asarray(handle["inputs/reference_ymix_state"])
        assert np.allclose(time_s, np.array([0.0, 1.0]))
        assert np.allclose(ymix[0], build_flat_h2_he_anchor(species, nz=nz))
        assert np.allclose(ymix[1], reference)
        assert np.allclose(reference_out, reference)


def test_copy_fastchem_runtime_copies_minimal_runtime(tmp_path):
    source_root = tmp_path / "source"
    fastchem_root = source_root / "fastchem_vulcan"
    (fastchem_root / "input").mkdir(parents=True)
    (fastchem_root / "fastchem_src" / "chem_input").mkdir(parents=True)
    (fastchem_root / "obj").mkdir(parents=True)
    (fastchem_root / "fastchem").write_text("binary", encoding="utf-8")
    (fastchem_root / "input" / "config.input").write_text("cfg", encoding="utf-8")
    (fastchem_root / "fastchem_src" / "chem_input" / "chemical_elements.dat").write_text(
        "elements",
        encoding="utf-8",
    )
    (fastchem_root / "obj" / "unused.o").write_text("unused", encoding="utf-8")

    copied_root = _copy_fastchem_runtime(source_root, tmp_path / "worker")

    assert (copied_root / "fastchem").exists()
    assert (copied_root / "input" / "config.input").exists()
    assert (copied_root / "fastchem_src" / "chem_input" / "chemical_elements.dat").exists()
    assert (copied_root / "output").exists()
    assert not (copied_root / "obj").exists()


def test_run_vulcan_generation_equilibrium_only_skips_vulcan_runtime(tmp_path, tiny_config, monkeypatch):
    config = copy.deepcopy(tiny_config)
    config["generation"]["mode"] = "vulcan"
    config["generation"]["target_mode"] = "equilibrium_only"
    config["generation"]["num_runs"] = 1
    config["paths"]["vulcan_source_root"] = str(tmp_path / "VULCAN")

    fastchem_root = Path(config["paths"]["vulcan_source_root"]) / "fastchem_vulcan"
    (fastchem_root / "input").mkdir(parents=True)
    (fastchem_root / "fastchem_src" / "chem_input").mkdir(parents=True)
    (fastchem_root / "fastchem").write_text("binary", encoding="utf-8")
    (fastchem_root / "input" / "config.input").write_text("cfg", encoding="utf-8")
    (fastchem_root / "fastchem_src" / "chem_input" / "chemical_elements.dat").write_text(
        "elements",
        encoding="utf-8",
    )

    def _unexpected_initial_ymix(*args, **kwargs):
        raise AssertionError("Equilibrium-only VULCAN generation should not sample an initial ymix.")

    def _unexpected_time_grid(*args, **kwargs):
        raise AssertionError("Equilibrium-only VULCAN generation should not sample a time grid.")

    monkeypatch.setattr(sampling_module, "sample_initial_ymix", _unexpected_initial_ymix)
    monkeypatch.setattr(sampling_module, "sample_time_grid", _unexpected_time_grid)

    def _unexpected_vulcan(*args, **kwargs):
        raise AssertionError("Equilibrium-only generation should not invoke the VULCAN runtime.")

    def _fake_fastchem(spec, *, source_root, worker_base, runs_dir, config):
        del source_root
        species = list(config["data_spec"]["state_species"])
        reference = build_flat_h2_he_anchor(species, nz=spec.pressure_bar.size)
        idx = {name: i for i, name in enumerate(species)}
        reference[:, idx["H2O"]] = 1.0e-3
        reference[:, idx["CH4"]] = 5.0e-4
        reference[:, idx["H2"]] -= 1.5e-3
        reference /= np.sum(reference, axis=1, keepdims=True)
        fastchem_output = worker_base / f"{spec.run_id}_fake_fastchem.dat"
        fastchem_output.parent.mkdir(parents=True, exist_ok=True)
        with fastchem_output.open("w", encoding="utf-8") as handle:
            handle.write(" ".join(species) + "\n")
            for row in reference:
                handle.write(" ".join(f"{value:.8e}" for value in row) + "\n")
        return convert_fastchem_output_to_hdf5(
            fastchem_output,
            output_h5_path=runs_dir / f"{spec.run_id}.h5",
            spec=spec,
            config=config,
        )

    monkeypatch.setattr(vulcan_runner_module, "_run_single_vulcan_spec", _unexpected_vulcan)
    monkeypatch.setattr(vulcan_runner_module, "_run_single_fastchem_spec", _fake_fastchem)

    artifact = run_vulcan_generation(config, project_root=config["_project_root"])

    assert len(artifact.run_files) == 1
    with h5py.File(artifact.run_files[0], "r") as handle:
        time_s = np.asarray(handle["trajectory/time_s"])
        assert np.allclose(time_s, np.array([0.0, 1.0]))
        assert handle["inputs/target_mode"][()].decode("utf-8") == "equilibrium_only"
