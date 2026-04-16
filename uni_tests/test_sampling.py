from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from src.data_generation.roth_sampling import (
    _interpolate_profile,
    load_roth_profiles,
    load_roth_profiles_native,
    roth_library_pressure_bounds,
)
from src.data_generation.generation import _element_profile_from_spec
from src.data_generation.sampling import (
    _decide_temperature_profile_source,
    _sample_column_pressure_grid,
    _sample_temperature_profile_record,
    build_sampling_plan,
    sample_kzz_profile,
    sample_pressure_grid,
    sample_run_specifications,
    sample_run_specifications_slice,
    sample_temperature_profile,
)
from src.utils.config import load_and_validate_config


def _grid_from_config(config, seed=0):
    """Draw one in-range pressure grid from a validated config for tests."""
    return _sample_column_pressure_grid(
        config["sampling"], rng=np.random.default_rng(seed),
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
    pressure_bar = _grid_from_config(config)

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
    pressure_bar = _grid_from_config(config)

    with pytest.raises(FileNotFoundError):
        sample_temperature_profile(
            pressure_bar,
            config=config,
            rng=np.random.default_rng(0),
        )


def test_kzz_sampling_is_constant_with_depth(tiny_config):
    pressure_bar = _grid_from_config(tiny_config)

    kzz = sample_kzz_profile(
        pressure_bar,
        config=tiny_config,
        rng=np.random.default_rng(0),
    )

    assert kzz.shape == pressure_bar.shape
    np.testing.assert_allclose(kzz, np.full_like(kzz, kzz[0]))
    lo, hi = (float(x) for x in tiny_config["sampling"]["kzz_range_cm2_s"])
    assert lo <= float(kzz[0]) <= hi


def test_vulcan_sampling_emits_elemental_globals_and_curated_presets(tiny_config):
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
        profile = _element_profile_from_spec(spec)
        assert profile.shape[-1] == len(tiny_config["data_spec"]["element_input_order"])
        for name in ("He_H", "C_H", "O_H", "N_H", "S_H"):
            assert name in spec.globals
        assert "gravity_cm_s2" in spec.globals
        assert "planet_radius_cm" in spec.globals
        assert float(tiny_config["sampling"]["planet_radius_range_cm"][0]) <= float(spec.globals["planet_radius_cm"])
        assert float(spec.globals["planet_radius_cm"]) <= float(tiny_config["sampling"]["planet_radius_range_cm"][1])
        assert sum(
            int(float(spec.globals[f"atm_base_{name}"]) > 0.5)
            for name in ("H2", "N2", "O2", "CO2", "H2O")
        ) == 1


def test_analytic_temperature_sampling_uses_piette_sampler_metadata_and_bounds(tiny_config):
    config = copy.deepcopy(tiny_config)
    config["roth_sampler"] = {"enabled": False}
    config["temperature_profiles"]["source_mode"] = "analytic"
    config["temperature_profiles"]["analytic_sampler"]["convection_probability"] = 1.0
    pressure_bar = _grid_from_config(config)

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
    assert metadata["temperature_profile_analytic_profile_type"] == "piette_2019"
    assert metadata["temperature_profile_analytic_convective_adjustment_applied"] is True
    assert "temperature_profile_analytic_t_int_k" in metadata
    assert "temperature_profile_analytic_t_eq_k" in metadata
    assert "temperature_profile_analytic_gamma" in metadata
    assert "temperature_profile_analytic_alpha" in metadata
    assert "temperature_profile_analytic_log10_delta" in metadata
    assert "temperature_profile_analytic_log10_p_trans_bar" in metadata


def test_pt_library_sampling_rejects_profiles_outside_shared_temperature_bounds(tiny_config, tmp_path):
    config = copy.deepcopy(tiny_config)
    pressure_bar = _grid_from_config(config)
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
    pressure_bar = _grid_from_config(config)
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


def test_roth_library_pressure_bounds_returns_union_range():
    profiles = load_roth_profiles_native(str(FIXTURE_PT_PATH))
    assert profiles
    native_min, native_max = roth_library_pressure_bounds(profiles)
    observed_min = min(float(np.min(p.pressure_bar)) for p in profiles)
    observed_max = max(float(np.max(p.pressure_bar)) for p in profiles)
    assert native_min == pytest.approx(observed_min)
    assert native_max == pytest.approx(observed_max)


def test_sample_column_pressure_grid_clips_to_native_bounds():
    """When native bounds are tighter than the configured ranges, the sampled
    grid must sit strictly inside [native_min, native_max]."""
    sampling_cfg = {
        "num_levels_range": [40, 60],
        "pressure_top_bar_range": [1.0e-9, 1.0e-3],
        "pressure_bottom_bar_range": [10.0, 1000.0],
    }
    native_bounds = (1.0e-6, 10.0)
    for seed in range(20):
        grid = _sample_column_pressure_grid(
            sampling_cfg,
            rng=np.random.default_rng(seed),
            native_p_bounds=native_bounds,
        )
        assert float(grid.min()) >= native_bounds[0] - 1e-12
        assert float(grid.max()) <= native_bounds[1] + 1e-12


def test_decide_temperature_profile_source_honors_mode():
    rng = np.random.default_rng(0)
    assert _decide_temperature_profile_source({"enabled": False}, rng=rng) == "analytic"
    assert _decide_temperature_profile_source(
        {"enabled": True, "source_mode": "roth"}, rng=rng,
    ) == "roth"
    # Mixed mode: analytic_probability=0 -> always Roth.
    always_roth = {"enabled": True, "source_mode": "mixed", "analytic_probability": 0.0}
    assert _decide_temperature_profile_source(always_roth, rng=rng) == "roth"
    # Mixed mode: analytic_probability=1 -> always analytic.
    always_analytic = {"enabled": True, "source_mode": "mixed", "analytic_probability": 1.0}
    assert _decide_temperature_profile_source(always_analytic, rng=rng) == "analytic"


def test_roth_profile_has_no_constant_plateau_near_toa(tiny_config):
    """With upfront source decision + native clipping, a Roth-chosen run must
    interpolate rather than clamp at the TOA, so consecutive temperatures
    near the top should not be bit-identical."""
    config = copy.deepcopy(tiny_config)
    config["sampling"]["pressure_top_bar_range"] = [1.0e-9, 1.0e-5]
    config["sampling"]["pressure_bottom_bar_range"] = [10.0, 200.0]
    config["sampling"]["num_levels_range"] = [52, 52]
    config["roth_sampler"] = {
        "enabled": True,
        "source_mode": "mixed",
        "analytic_probability": 0.0,
        "data_glob": str(FIXTURE_PT_PATH),
        "filters": {"Teq": (1200.0, 1200.0), "LogMet": 0.0, "TiOVO": False},
    }
    specs = sample_run_specifications(
        config=config,
        project_root=config["_project_root"],
        num_runs=4,
        seed=7,
    )
    for spec in specs:
        assert spec.metadata.get("temperature_profile_source") == "pt_library"
        # Inside the native library range — no clamp should occur.
        assert spec.pressure_bar.min() >= 1.0e-6 - 1e-12
        assert spec.pressure_bar.max() <= 10.0 + 1e-12
        top_slice = spec.temperature_k[-5:]
        # Require that the top levels vary (no constant plateau).
        assert np.any(np.abs(np.diff(top_slice)) > 1.0e-6)


def test_analytic_sampler_ranges_match_repo_config():
    """Pin the narrowed analytic-sampler ranges — changes should be explicit.

    Catches accidental reversions to the old wide bounds that produced the
    extreme ``run_07390`` inversion.
    """
    import json

    root = Path(__file__).resolve().parents[1]
    for cfg_name in ("vulcan_no_condensation.json", "vulcan_condensation.json"):
        cfg_path = root / "config" / cfg_name
        cfg = json.loads(cfg_path.read_text())
        sampler = cfg["temperature_profiles"]["analytic_sampler"]
        assert sampler["log10_gamma_range"] == [-1.0, 1.0], cfg_name
        assert sampler["log10_delta_range"] == [-3.0, 1.0], cfg_name
        assert sampler["alpha_range"] == [0.0, 0.9], cfg_name
        assert sampler["log10_p_trans_bar_range"] == [-2.0, 2.0], cfg_name
        assert sampler["t_int_k_range"] == [50.0, 1000.0], cfg_name
        assert sampler["t_eq_k_range"] == [300.0, 3500.0], cfg_name


def _specs_equal(a, b):
    """Compare two RunSpecification objects structurally for test equality."""
    assert a.run_id == b.run_id
    np.testing.assert_array_equal(a.pressure_bar, b.pressure_bar)
    np.testing.assert_array_equal(a.temperature_k, b.temperature_k)
    np.testing.assert_array_equal(a.gravity_cm_s2, b.gravity_cm_s2)
    assert a.globals == b.globals
    assert a.metadata == b.metadata
    if a.kzz_cm2_s is None:
        assert b.kzz_cm2_s is None
    else:
        np.testing.assert_array_equal(a.kzz_cm2_s, b.kzz_cm2_s)


def test_chunked_slice_matches_one_shot_sampling(tiny_config):
    """Streaming the plan in slices reproduces ``sample_run_specifications`` byte-for-byte.

    This protects against subtle regressions in the chunked-generation pipeline
    (``generation.py``) where per-run sampling moved from a single ThreadPool
    call to a loop of ``sample_run_specifications_slice(plan, start, end)``
    invocations. Identical seeds must produce identical specs regardless of
    how the total_runs interval is partitioned.
    """
    total_runs = 12
    baseline = sample_run_specifications(
        config=tiny_config,
        project_root=tiny_config["_project_root"],
        num_runs=total_runs,
        seed=7,
    )
    plan = build_sampling_plan(
        config=tiny_config,
        project_root=tiny_config["_project_root"],
        num_runs=total_runs,
        seed=7,
    )
    # Multiple slice partitions must all reproduce the one-shot result.
    for partition in ([0, 4, 9, total_runs], [0, 12], [0, 1, 2, total_runs]):
        streamed = []
        for lo, hi in zip(partition[:-1], partition[1:]):
            streamed.extend(sample_run_specifications_slice(plan, start=lo, end=hi))
        assert len(streamed) == len(baseline)
        for a, b in zip(baseline, streamed):
            _specs_equal(a, b)
