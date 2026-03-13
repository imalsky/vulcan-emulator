from __future__ import annotations

import copy

import numpy as np
import pytest

from src.sampling import sample_kzz_profile, sample_pressure_grid, sample_temperature_profile


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
