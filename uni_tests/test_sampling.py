from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from src.utils.config import load_and_validate_config
from src.data_generation.roth_sampling import _interpolate_profile, load_roth_profiles
from src.data_generation.sampling import (
    _sample_temperature_profile_record,
    sample_kzz_profile,
    sample_pressure_grid,
    sample_run_specifications,
    sample_temperature_profile,
)


FIXTURE_PT_PATH = (
    Path(__file__).resolve().parents[1]
    / "uni_tests"
    / "fixtures"
    / "pt_profiles"
    / "PTprofiles-Teq_1200-LogMet_0.0-LogDrag_0-Mstar_0.8-Rp_1.3-logG_1.3-TiOVO_false.dat"
)


def test_roth_temperature_sampling_requires_matching_profiles(tiny_config):
    config = copy.deepcopy(tiny_config)
    config["roth_sampler"] = {
        "enabled": True,
        "data_glob": str(tiny_config["_project_root"] / "missing_roth_profiles" / "*.npz"),
        "filters": {},
    }
    pressure_bar = sample_pressure_grid(
        num_levels=int(config["sampling"]["num_levels"]),
        pressure_top_bar=float(config["sampling"]["pressure_top_bar"]),
        pressure_bottom_bar=float(config["sampling"]["pressure_bottom_bar"]),
    )

    with pytest.raises(FileNotFoundError):
        sample_temperature_profile(
            pressure_bar,
            config=config,
            rng=np.random.default_rng(0),
        )


def test_roth_temperature_sampling_rejects_over_restrictive_filters(tiny_config):
    config = copy.deepcopy(tiny_config)
    config["roth_sampler"] = {
        "enabled": True,
        "data_glob": str(FIXTURE_PT_PATH),
        "filters": {"Teq": (9999.0, 10000.0), "TiOVO": False},
    }
    pressure_bar = sample_pressure_grid(
        num_levels=int(config["sampling"]["num_levels"]),
        pressure_top_bar=float(config["sampling"]["pressure_top_bar"]),
        pressure_bottom_bar=float(config["sampling"]["pressure_bottom_bar"]),
    )

    with pytest.raises(FileNotFoundError):
        sample_temperature_profile(
            pressure_bar,
            config=config,
            rng=np.random.default_rng(0),
        )


def test_kzz_sampling_is_constant_with_depth(tiny_config):
    pressure_bar = sample_pressure_grid(
        num_levels=int(tiny_config["sampling"]["num_levels"]),
        pressure_top_bar=float(tiny_config["sampling"]["pressure_top_bar"]),
        pressure_bottom_bar=float(tiny_config["sampling"]["pressure_bottom_bar"]),
    )

    kzz = sample_kzz_profile(
        pressure_bar,
        config=tiny_config,
        rng=np.random.default_rng(0),
    )

    assert kzz.shape == pressure_bar.shape
    np.testing.assert_allclose(kzz, np.full_like(kzz, kzz[0]))
    np.testing.assert_allclose(kzz[0], float(tiny_config["sampling"]["kzz_cm2_s"]))


def test_full_vulcan_sampling_emits_elemental_globals_and_curated_presets(tiny_config):
    specs = sample_run_specifications(
        config=tiny_config,
        project_root=tiny_config["_project_root"],
        num_runs=16,
        seed=3,
    )

    preset_names = {spec.metadata["science_preset_name"] for spec in specs}
    assert preset_names == {preset["name"] for preset in tiny_config["science_presets"]}
    assert len({spec.metadata["spectrum_name"] for spec in specs}) >= 2
    for spec in specs:
        assert spec.elemental_abundances_x_h.shape[-1] == len(tiny_config["data_spec"]["element_input_order"])
        for name in ("He_H", "C_H", "O_H", "N_H", "S_H"):
            assert name in spec.globals
        assert "gravity_cm_s2" in spec.globals
        assert sum(
            int(float(spec.globals[f"atm_base_{name}"]) > 0.5)
            for name in ("H2", "N2", "O2", "CO2", "H2O")
        ) == 1


def test_shipped_equilibrium_config_uses_fixture_temperature_profiles():
    root = Path(__file__).resolve().parents[1]
    config = load_and_validate_config(root / "config" / "fastchem_mlp_config.json")
    config["_project_root"] = root
    config["temperature_profiles"]["data_glob"] = str(FIXTURE_PT_PATH)
    config["temperature_profiles"]["filters"] = {"Teq": (1200.0, 1200.0), "LogMet": 0.0, "TiOVO": False}
    config["roth_sampler"]["data_glob"] = str(FIXTURE_PT_PATH)
    config["roth_sampler"]["filters"] = dict(config["temperature_profiles"]["filters"])
    assert config["roth_sampler"]["source_mode"] == "mixed"
    assert config["roth_sampler"]["analytic_probability"] == pytest.approx(0.5)
    assert config["temperature_profiles"]["analytic_sampler"]["log10_kappa_ir_m2_kg_normal"] == {
        "mean": -2.5,
        "std": 2.5,
    }
    pressure_bar = sample_pressure_grid(
        num_levels=int(config["sampling"]["num_levels"]),
        pressure_top_bar=float(config["sampling"]["pressure_top_bar"]),
        pressure_bottom_bar=float(config["sampling"]["pressure_bottom_bar"]),
    )
    profile = sample_temperature_profile(
        pressure_bar,
        config=config,
        rng=np.random.default_rng(0),
    )
    assert profile.shape == pressure_bar.shape
    assert np.all(np.isfinite(profile))
    assert float(np.min(profile)) >= float(config["temperature_profiles"]["validation"]["min_temperature_k"])
    assert float(np.max(profile)) <= float(config["temperature_profiles"]["validation"]["max_temperature_k"])


def test_analytic_temperature_sampling_uses_line_sampler_metadata_and_bounds(tiny_config):
    config = copy.deepcopy(tiny_config)
    config["roth_sampler"] = {"enabled": False}
    config["temperature_profiles"]["source_mode"] = "analytic"
    config["temperature_profiles"]["analytic_sampler"]["convection_probability"] = 1.0
    pressure_bar = sample_pressure_grid(
        num_levels=int(config["sampling"]["num_levels"]),
        pressure_top_bar=float(config["sampling"]["pressure_top_bar"]),
        pressure_bottom_bar=float(config["sampling"]["pressure_bottom_bar"]),
    )

    profile, metadata = _sample_temperature_profile_record(
        pressure_bar,
        config=config,
        rng=np.random.default_rng(0),
    )

    hard_bounds = config["temperature_profiles"]["validation"]
    assert profile.shape == pressure_bar.shape
    assert np.all(np.isfinite(profile))
    assert float(np.min(profile)) >= float(hard_bounds["min_temperature_k"])
    assert float(np.max(profile)) <= float(hard_bounds["max_temperature_k"])
    assert metadata["temperature_profile_source"] == "analytic"
    assert metadata["temperature_profile_analytic_profile_type"] == "line_2013"
    assert metadata["temperature_profile_analytic_convective_adjustment_applied"] is True
    assert "temperature_profile_analytic_t_int_k" in metadata
    assert "temperature_profile_analytic_gamma_1" in metadata
    assert "temperature_profile_analytic_alpha" in metadata


def test_pt_library_sampling_rejects_profiles_outside_shared_temperature_bounds(tiny_config, tmp_path):
    config = copy.deepcopy(tiny_config)
    pressure_bar = sample_pressure_grid(
        num_levels=int(config["sampling"]["num_levels"]),
        pressure_top_bar=float(config["sampling"]["pressure_top_bar"]),
        pressure_bottom_bar=float(config["sampling"]["pressure_bottom_bar"]),
    )
    invalid_profile_path = tmp_path / "profile_invalid.npz"
    np.savez(
        invalid_profile_path,
        pressure_bar=np.array([100.0, 1.0, 1.0e-7], dtype=np.float64),
        temperature_k=np.array([3200.0, 2500.0, 1500.0], dtype=np.float64),
    )
    config["roth_sampler"] = {
        "enabled": True,
        "source_mode": "roth",
        "data_glob": str(invalid_profile_path),
        "filters": {},
    }

    with pytest.raises(FileNotFoundError):
        sample_temperature_profile(
            pressure_bar,
            config=config,
            rng=np.random.default_rng(0),
        )


def test_pt_dat_loader_parses_metadata_and_column_profiles():
    pressure_bar = sample_pressure_grid(
        num_levels=16,
        pressure_top_bar=1.0e-7,
        pressure_bottom_bar=100.0,
    )
    profiles = load_roth_profiles(
        str(FIXTURE_PT_PATH),
        pressure_grid_bar=pressure_bar,
        filters={"Teq": (1200.0, 1200.0), "LogMet": 0.0, "TiOVO": False},
    )

    assert profiles
    assert len(profiles) == 2
    assert len({(profile.metadata["lon"], profile.metadata["lat"]) for profile in profiles}) == len(profiles)
    for profile in profiles:
        assert profile.pressure_bar.shape == pressure_bar.shape
        assert profile.temperature_k.shape == pressure_bar.shape
        assert profile.metadata["Teq"] == pytest.approx(1200.0)
        assert profile.metadata["LogMet"] == pytest.approx(0.0)
        assert profile.metadata["LogDrag"] == pytest.approx(0.0)
        assert profile.metadata["Mstar"] == pytest.approx(0.8)
        assert profile.metadata["Rp"] == pytest.approx(1.3)
        assert profile.metadata["logG"] == pytest.approx(1.3)
        assert profile.metadata["TiOVO"] is False
        assert isinstance(profile.metadata["lon"], float)
        assert isinstance(profile.metadata["lat"], float)
        assert np.all(np.isfinite(profile.temperature_k))


def test_roth_interpolation_uses_smoother_than_linear_log_pressure_fit():
    source_pressure_bar = np.array([100.0, 1.0, 0.1, 0.001, 1.0e-6], dtype=np.float64)
    source_temperature_k = np.array([1500.0, 1325.0, 1040.0, 910.0, 760.0], dtype=np.float64)
    pressure_bar = sample_pressure_grid(
        num_levels=21,
        pressure_top_bar=1.0e-6,
        pressure_bottom_bar=100.0,
    )

    interpolated = _interpolate_profile(
        pressure_bar,
        source_pressure_bar,
        source_temperature_k,
    )
    linear = np.interp(
        np.log10(pressure_bar),
        np.log10(source_pressure_bar)[::-1],
        source_temperature_k[::-1],
    )

    assert interpolated.shape == pressure_bar.shape
    assert np.all(np.isfinite(interpolated))
    assert float(np.min(interpolated)) >= float(np.min(source_temperature_k)) - 1.0e-6
    assert float(np.max(interpolated)) <= float(np.max(source_temperature_k)) + 1.0e-6
    assert not np.allclose(interpolated, linear)


def test_mixed_temperature_sampling_supports_analytic_and_pt_paths(tiny_config):
    config = copy.deepcopy(tiny_config)
    pressure_bar = sample_pressure_grid(
        num_levels=int(config["sampling"]["num_levels"]),
        pressure_top_bar=float(config["sampling"]["pressure_top_bar"]),
        pressure_bottom_bar=float(config["sampling"]["pressure_bottom_bar"]),
    )
    config["roth_sampler"] = {
        "enabled": True,
        "source_mode": "mixed",
        "analytic_probability": 0.75,
        "data_glob": str(FIXTURE_PT_PATH),
        "filters": {"Teq": (1200.0, 1200.0), "LogMet": 0.0, "TiOVO": False},
    }

    analytic_profile = sample_temperature_profile(
        pressure_bar,
        config=config,
        rng=np.random.default_rng(0),
    )
    assert analytic_profile.shape == pressure_bar.shape
    assert np.all(np.isfinite(analytic_profile))

    config["roth_sampler"]["analytic_probability"] = 0.25
    sampled_pt_profile = sample_temperature_profile(
        pressure_bar,
        config=config,
        rng=np.random.default_rng(0),
    )
    assert sampled_pt_profile.shape == pressure_bar.shape
    assert np.all(np.isfinite(sampled_pt_profile))
    assert not np.allclose(sampled_pt_profile, analytic_profile)
