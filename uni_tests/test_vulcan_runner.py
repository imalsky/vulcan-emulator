from __future__ import annotations

import copy
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import src.data_generation.generation as generation_module
from src.data_generation.generation import (
    _copy_fastchem_runtime,
    _patch_vulcan_cfg,
    convert_fastchem_output_to_hdf5,
    convert_vulcan_output_to_hdf5,
    generate_raw_dataset,
    run_vulcan_generation,
    write_equilibrium_hdf5,
)
from synthetic_fixture import generate_synthetic_raw_runs
from src.data_generation.sampling import sample_run_specifications
from src.data_generation.spectrum import (
    generate_blackbody_template,
    write_vulcan_spectrum_txt,
)
from src.utils.config import dataset_info_root, load_and_validate_config


def _open_first_run(artifact: generation_module.GeneratedRawDataset) -> h5py.Group:
    """Return the first stored run group from a raw-generation artifact."""
    root = h5py.File(artifact.consolidated_path, "r")
    return root[sorted(root.keys())[0]]


def _make_equilibrium_config(tmp_path: Path) -> dict:
    """Build one small FastChem config suitable for unit tests."""
    root = Path(__file__).resolve().parents[1]
    config = load_and_validate_config(root / "uni_tests" / "fixtures" / "fastchem_transformer_config.json")
    config = copy.deepcopy(config)
    config["paths"]["raw_root"] = str(tmp_path / "dataset" / "raw")
    config["paths"]["processed_root"] = str(tmp_path / "dataset" / "processed")
    config["paths"]["checkpoints_root"] = str(tmp_path / "checkpoints")
    config["paths"]["vulcan_source_root"] = str(tmp_path / "VULCAN")
    config["generation"]["mode"] = "vulcan"
    config["generation"]["num_runs"] = 1
    config["generation"]["parallel_workers"] = 1
    config["temperature_profiles"]["source_mode"] = "analytic"
    config["temperature_profiles"].pop("data_glob", None)
    config["temperature_profiles"].pop("analytic_probability", None)
    config["roth_sampler"] = {
        "enabled": False,
        "source_mode": "roth",
        "analytic_probability": None,
        "data_glob": "",
        "filters": {},
    }
    config["_project_root"] = root
    return config


def test_generate_synthetic_raw_runs_writes_final_state_only(tiny_config):
    artifact = generate_synthetic_raw_runs(tiny_config, project_root=tiny_config["_project_root"])
    assert len(artifact.run_ids) == tiny_config["generation"]["num_runs"]
    assert not (artifact.raw_root / "runs").exists()
    assert artifact.manifest_path is not None and artifact.manifest_path.exists()
    assert artifact.coverage_path is not None and artifact.coverage_path.exists()

    handle = _open_first_run(artifact)
    try:
        species = [item.decode("utf-8") for item in handle["inputs/state_species"][:]]
        element_labels = [item.decode("utf-8") for item in handle["inputs/element_input_order"][:]]
        final_state = np.asarray(handle["final_state/ymix_output"])
        assert "trajectory" not in handle
        nl_lo, nl_hi = tiny_config["sampling"]["num_levels_range"]
        assert nl_lo <= final_state.shape[0] <= nl_hi
        nz = int(final_state.shape[0])
        assert final_state.shape[1] == len(tiny_config["data_spec"]["output_species"])
        assert np.all(np.isfinite(final_state))
        assert species == list(tiny_config["data_spec"]["state_species"])
        assert element_labels == ["He_H", "C_H", "O_H", "N_H", "S_H"]
        assert np.asarray(handle["inputs/elemental_abundances_frac"]).shape == (
            nz,
            len(element_labels),
        )
        assert np.asarray(handle["inputs/gravity_cm_s2"]).shape == (nz,)
        spectrum = np.asarray(handle["spectrum/flux_erg_cm2_s_nm"])
        assert spectrum.size > 10
        assert "target_mode" not in handle["inputs"]
    finally:
        handle.file.close()


def test_generate_synthetic_raw_runs_reuse_keeps_consolidated_format(tiny_config):
    artifact = generate_synthetic_raw_runs(tiny_config, project_root=tiny_config["_project_root"])

    artifact = generate_synthetic_raw_runs(tiny_config, project_root=tiny_config["_project_root"])

    assert artifact.consolidated_path.exists()


def test_generate_synthetic_raw_runs_requires_configured_spectrum_template(tiny_config):
    config = copy.deepcopy(tiny_config)
    config["stellar_spectrum"]["library_glob"] = None
    config["vulcan"]["stellar_spectrum"]["library_glob"] = None
    config["stellar_spectrum"]["template_file"] = str(
        tiny_config["_project_root"] / "does_not_exist_surface_flux.txt"
    )
    config["vulcan"]["stellar_spectrum"]["template_file"] = config["stellar_spectrum"]["template_file"]
    with pytest.raises(FileNotFoundError):
        generate_synthetic_raw_runs(config, project_root=config["_project_root"])


def test_generate_synthetic_raw_runs_can_synthesize_template_without_asset(tiny_config):
    config = copy.deepcopy(tiny_config)
    config["stellar_spectrum"]["library_glob"] = None
    config["vulcan"]["stellar_spectrum"]["library_glob"] = None
    config["stellar_spectrum"]["template_file"] = None
    config["vulcan"]["stellar_spectrum"]["template_file"] = None

    artifact = generate_synthetic_raw_runs(config, project_root=config["_project_root"])

    assert artifact.consolidated_path.exists()


def test_vulcan_generation_requires_configured_checkout(tiny_config):
    config = copy.deepcopy(tiny_config)
    config["generation"]["mode"] = "vulcan"
    with pytest.raises(FileNotFoundError):
        generate_raw_dataset(config, project_root=config["_project_root"])


def test_patch_vulcan_cfg_uses_profile_kzz_and_cross_sections(tmp_path, tiny_config):
    config = copy.deepcopy(tiny_config)
    config["vulcan"]["physics_toggles"]["use_photochemistry"] = False
    config["vulcan"]["physics_toggles"]["use_condensation"] = True
    config["vulcan"]["physics_toggles"]["use_initial_cold_trap"] = True
    config["physics_toggles"] = dict(config["vulcan"]["physics_toggles"])
    config["vulcan"]["science_presets"] = [
        {
            "name": "basic_h2",
            "atm_base": "H2",
            "physics_toggles": dict(config["vulcan"]["physics_toggles"]),
        }
    ]
    config["science_presets"] = copy.deepcopy(config["vulcan"]["science_presets"])
    config["default_science_preset"] = copy.deepcopy(config["science_presets"][0])
    config["vulcan"]["runtime"]["chemistry_file"] = "thermo/SNCHO_photo_network_2025.txt"
    config["vulcan"]["runtime"]["regenerate_chem_funs"] = True
    config["vulcan"]["runtime"]["cfg_assignments"] = {
        "condense_sp": ["H2O", "S8"],
        "non_gas_sp": ["H2O_l_s", "S8_l_s"],
        "use_relax": ["H2O"],
        "humidity": 1.0,
        "r_p": {"H2O_l_s": 5e-5, "S8_l_s": 1e-4},
        "rho_p": {"H2O_l_s": 0.9, "S8_l_s": 2.07},
        "start_conden_time": 0,
        "runtime": 1.0e12,
    }
    config["vulcan_runtime"] = copy.deepcopy(config["vulcan"]["runtime"])

    spec = sample_run_specifications(
        config=config,
        project_root=config["_project_root"],
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
        config=config,
        tp_file=tp_file,
        spectrum_file=spectrum_file,
    )

    patched = cfg_file.read_text(encoding="utf-8")
    expected_atm_base = next(
        name for name in ("H2", "N2", "O2", "CO2", "H2O")
        if float(spec.globals.get(f"atm_base_{name}", 0.0)) > 0.5
    )
    assert "atm_type = 'file'" in patched
    assert "Kzz_prof = 'file'" in patched
    assert "use_photo = False" in patched
    assert "use_condense = True" in patched
    assert "use_ini_cold_trap = True" in patched
    assert "network = 'thermo/SNCHO_photo_network_2025.txt'" in patched
    assert "condense_sp = ['H2O', 'S8']" in patched
    assert "non_gas_sp = ['H2O_l_s', 'S8_l_s']" in patched
    assert "use_relax = ['H2O']" in patched
    assert "humidity = 1.0" in patched
    assert "'H2O_l_s': 5e-05" in patched
    assert "'S8_l_s': 0.0001" in patched
    assert "'S8_l_s': 2.07" in patched
    assert "runtime = 1000000000000.0" in patched
    assert "T_cross_sp = ['H2O', 'H2S', 'SH', 'SO2', 'S2']" in patched
    assert f"atm_base = '{expected_atm_base}'" in patched
    assert f"gs = {float(spec.globals['gravity_cm_s2'])}" in patched
    assert f"Rp = {float(spec.globals['planet_radius_cm'])}" in patched
    assert "rocky = False" in patched
    assert "use_lowT_limit_rates = True" in patched
    assert "use_adapt_rtol = True" in patched


def test_convert_fake_vulcan_output_to_hdf5_writes_final_state_only(tmp_path, tiny_config):
    specs = sample_run_specifications(
        config=tiny_config,
        project_root=tiny_config["_project_root"],
        num_runs=1,
        seed=5,
    )
    spec = specs[0]
    species = list(tiny_config["data_spec"]["state_species"])
    nz = spec.pressure_bar.size
    state_dim = len(species)
    reference = np.full((nz, state_dim), 1.0e-8, dtype=np.float64)
    reference[:, species.index("H2")] = 0.84
    reference[:, species.index("He")] = 0.15
    reference[:, species.index("H2O")] = 1.0e-3
    reference /= np.sum(reference, axis=1, keepdims=True)
    runtime_gravity = np.linspace(2.0e3, 1.7e3, nz, dtype=np.float64)
    fake = {
        "variable": {
            "species": species,
            "ymix": reference,
        },
        "atm": {
            "pco": spec.pressure_bar * 1.0e6,
            "Tco": spec.temperature_k,
            "Kzz": spec.kzz_cm2_s[:-1],
            "g": runtime_gravity,
        },
    }
    vul_path = tmp_path / "fake.vul"
    with vul_path.open("wb") as handle:
        pickle.dump(fake, handle, protocol=pickle.HIGHEST_PROTOCOL)
    out_h5 = tmp_path / "converted.h5"
    converted_spec = type(spec)(
        run_id=spec.run_id,
        pressure_bar=spec.pressure_bar,
        temperature_k=spec.temperature_k,
        kzz_cm2_s=spec.kzz_cm2_s,
        globals=spec.globals,
        spectrum=spec.spectrum,
        metadata={**spec.metadata, "state_species": species, "output_species": species},
        elemental_abundances_frac=spec.elemental_abundances_frac,
        gravity_cm_s2=spec.gravity_cm_s2,
    )
    convert_vulcan_output_to_hdf5(
        vul_path,
        output_h5_path=out_h5,
        spec=converted_spec,
        config=tiny_config,
    )
    with h5py.File(out_h5, "r") as handle:
        assert "trajectory" not in handle
        assert "target_mode" not in handle["inputs"]
        final_state = np.asarray(handle["final_state/ymix_output"])
        assert final_state.shape == (nz, state_dim)
        np.testing.assert_allclose(final_state, reference, atol=1.0e-12)
        assert np.asarray(handle["inputs/kzz_cm2_s"]).shape == (nz,)
        np.testing.assert_allclose(np.asarray(handle["inputs/gravity_cm_s2"]), runtime_gravity, atol=1.0e-12)



def test_convert_fake_fastchem_output_to_hdf5_writes_equilibrium_contract(tmp_path, tiny_config):
    specs = sample_run_specifications(
        config=tiny_config,
        project_root=tiny_config["_project_root"],
        num_runs=1,
        seed=7,
    )
    spec = specs[0]
    species = list(tiny_config["data_spec"]["state_species"])
    nz = spec.pressure_bar.size
    equilibrium_ymix = np.full((nz, len(species)), 1.0e-8, dtype=np.float64)
    equilibrium_ymix[:, species.index("H2")] = 0.84
    equilibrium_ymix[:, species.index("He")] = 0.15
    equilibrium_ymix[:, species.index("CO")] = 5.0e-4
    equilibrium_ymix[:, species.index("H2O")] = 1.0e-3
    equilibrium_ymix /= np.sum(equilibrium_ymix, axis=1, keepdims=True)

    fastchem_output = tmp_path / "vulcan_EQ.dat"
    with fastchem_output.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(species) + "\n")
        for row in equilibrium_ymix:
            handle.write(" ".join(f"{value:.8e}" for value in row) + "\n")

    out_h5 = tmp_path / "converted_fastchem.h5"
    convert_fastchem_output_to_hdf5(
        fastchem_output,
        output_h5_path=out_h5,
        spec=spec,
        config=tiny_config,
    )

    with h5py.File(out_h5, "r") as handle:
        assert "trajectory" not in handle
        assert "target_mode" not in handle["inputs"]
        np.testing.assert_allclose(np.asarray(handle["equilibrium/ymix"]), equilibrium_ymix, atol=1.0e-12)
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


def test_run_vulcan_generation_fastchem_skips_vulcan_runtime(tmp_path, monkeypatch):
    config = _make_equilibrium_config(tmp_path)

    fastchem_root = Path(config["paths"]["vulcan_source_root"]) / "fastchem_vulcan"
    (fastchem_root / "input").mkdir(parents=True)
    (fastchem_root / "fastchem_src" / "chem_input").mkdir(parents=True)
    (fastchem_root / "fastchem").write_text("binary", encoding="utf-8")
    (fastchem_root / "input" / "config.input").write_text("cfg", encoding="utf-8")
    (fastchem_root / "fastchem_src" / "chem_input" / "chemical_elements.dat").write_text(
        "elements",
        encoding="utf-8",
    )

    def _unexpected_vulcan(*args, **kwargs):
        raise AssertionError("FastChem generation should not invoke the VULCAN runtime.")

    def _fake_fastchem(spec, *, source_root, worker_base, runs_dir, config):
        del source_root, worker_base
        nz = spec.pressure_bar.size
        species = list(config["data_spec"]["output_species"])
        equilibrium_ymix = np.full((nz, len(species)), 1.0e-8, dtype=np.float64)
        equilibrium_ymix[:, species.index("H2")] = 0.84
        equilibrium_ymix[:, species.index("He")] = 0.15
        equilibrium_ymix[:, species.index("H2O")] = 1.0e-3
        equilibrium_ymix /= np.sum(equilibrium_ymix, axis=1, keepdims=True)
        return write_equilibrium_hdf5(
            runs_dir / f"{spec.run_id}.h5",
            spec=spec,
            equilibrium_ymix=equilibrium_ymix,
            state_species=list(config["data_spec"]["state_species"]),
            output_species=species,
        )

    monkeypatch.setattr(generation_module, "_run_single_vulcan_spec", _unexpected_vulcan)
    monkeypatch.setattr(generation_module, "_run_single_fastchem_spec", _fake_fastchem)

    artifact = run_vulcan_generation(config, project_root=config["_project_root"])

    assert len(artifact.run_ids) == 1
    assert not (artifact.raw_root / "runs").exists()
    handle = _open_first_run(artifact)
    try:
        assert "equilibrium" in handle
        assert "final_state" not in handle
        assert "target_mode" not in handle["inputs"]
    finally:
        handle.file.close()


def test_run_vulcan_generation_backfill_assigns_unique_run_ids(tmp_path, monkeypatch):
    config = _make_equilibrium_config(tmp_path)
    config["generation"]["num_runs"] = 2
    config["generation"]["backfill"] = {"enabled": True, "max_retries": 1}

    def _fake_validated_paths(config, *, project_root):
        del config, project_root
        return tmp_path, tmp_path / "fastchem"

    def _fake_sample_run_specifications(*, config, project_root, num_runs=None, seed=None):
        del config, project_root, seed
        total = int(num_runs or 0)
        pressure_bar = np.array([1.0, 0.1], dtype=np.float64)
        temperature_k = np.array([1200.0, 1000.0], dtype=np.float64)
        elemental_abundances_frac = np.repeat(
            np.array([[7.84e-2, 2.69e-4, 4.90e-4, 6.76e-5, 1.32e-5]], dtype=np.float64),
            pressure_bar.size,
            axis=0,
        )
        gravity_cm_s2 = np.full(pressure_bar.shape, 2.5e3, dtype=np.float64)
        globals_map = {
            "He_H": 7.84e-2,
            "C_H": 2.69e-4,
            "O_H": 4.90e-4,
            "N_H": 6.76e-5,
            "S_H": 1.32e-5,
        }
        return [
            generation_module.RunSpecification(
                run_id=f"run_{idx:05d}",
                pressure_bar=pressure_bar,
                temperature_k=temperature_k,
                globals=dict(globals_map),
                metadata={},
                elemental_abundances_frac=elemental_abundances_frac,
                gravity_cm_s2=gravity_cm_s2,
            )
            for idx in range(total)
        ]

    def _fake_fastchem(spec, *, source_root, worker_base, runs_dir, config):
        del source_root, worker_base
        if spec.run_id == "run_00000":
            raise RuntimeError("synthetic failure")
        nz = spec.pressure_bar.size
        species = list(config["data_spec"]["output_species"])
        equilibrium_ymix = np.full((nz, len(species)), 1.0e-8, dtype=np.float64)
        equilibrium_ymix[:, species.index("H2")] = 0.84
        equilibrium_ymix[:, species.index("He")] = 0.15
        equilibrium_ymix[:, species.index("H2O")] = 1.0e-3
        equilibrium_ymix /= np.sum(equilibrium_ymix, axis=1, keepdims=True)
        return write_equilibrium_hdf5(
            runs_dir / f"{spec.run_id}.h5",
            spec=spec,
            equilibrium_ymix=equilibrium_ymix,
            state_species=list(config["data_spec"]["state_species"]),
            output_species=species,
        )

    monkeypatch.setattr(generation_module, "_validated_vulcan_paths", _fake_validated_paths)
    monkeypatch.setattr(generation_module, "sample_run_specifications", _fake_sample_run_specifications)
    monkeypatch.setattr(generation_module, "_run_single_fastchem_spec", _fake_fastchem)

    artifact = run_vulcan_generation(config, project_root=config["_project_root"])

    assert artifact.consolidated_path is not None and artifact.consolidated_path.exists()
    with h5py.File(artifact.consolidated_path, "r") as handle:
        assert sorted(handle.keys()) == ["run_00001", "run_00002"]


def test_run_vulcan_generation_hard_fails_on_shortfall(tmp_path, tiny_config, monkeypatch):
    config = copy.deepcopy(tiny_config)
    config["generation"]["mode"] = "vulcan"
    config["generation"]["num_runs"] = 2
    config["generation"]["parallel_workers"] = 1
    config["generation"]["backfill"] = {"enabled": True, "max_retries": 1}

    def _fake_validated_paths(config, *, project_root):
        del config, project_root
        return tmp_path, tmp_path / "network.txt"

    def _always_fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(generation_module, "_validated_vulcan_paths", _fake_validated_paths)
    monkeypatch.setattr(generation_module, "_run_single_vulcan_spec", _always_fail)

    with pytest.raises(RuntimeError, match="successful runs short"):
        run_vulcan_generation(config, project_root=config["_project_root"])

    failed_log = Path(dataset_info_root(config)) / "failed_runs.json"
    assert failed_log.exists()
