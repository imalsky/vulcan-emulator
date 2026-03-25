from __future__ import annotations

from unittest.mock import patch

import numpy as np

from src.anchor_states import build_flat_h2_he_anchor
from src.data_loader import load_processed_dataset
from src.inference import PhysicalSpaceStandaloneModel
from src.preprocess import inverse_block, preprocess_raw_dataset
from src.vulcan_runner import generate_synthetic_raw_runs


def _prepare_processed_dataset(config: dict) -> tuple[dict, dict]:
    generate_synthetic_raw_runs(config, project_root=config["_project_root"])
    preprocess_raw_dataset(config, project_root=config["_project_root"])
    splits, normalization, _ = load_processed_dataset(config["paths"]["processed_root"])
    return splits, normalization


def _echo_anchor(self, **kwargs):
    return np.asarray(kwargs["ymix_state"], dtype=np.float64)


def test_equilibrium_uses_flat_anchor_in_equilibrium_only_mode(tiny_config):
    splits, normalization = _prepare_processed_dataset(tiny_config)
    train = splits["train"]
    nz = train.state_trajectories.shape[2]
    expected_anchor = build_flat_h2_he_anchor(
        tiny_config["data_spec"]["state_species"],
        nz=nz,
    )
    model = PhysicalSpaceStandaloneModel(
        params=None,
        dims=None,
        normalization=normalization,
        contract={"state_species_order": tiny_config["data_spec"]["state_species"]},
        config=tiny_config,
        project_root=tiny_config["_project_root"],
    )
    with patch.object(PhysicalSpaceStandaloneModel, "predict", autospec=True) as predict_mock:
        predict_mock.side_effect = _echo_anchor
        result = model.equilibrium(
            pressure_bar=np.logspace(2.0, -7.0, nz),
            temperature_K=np.linspace(900.0, 1200.0, nz),
            eddy_diffusion_cm2_s=np.full(nz, 1.0e8),
            global_inputs={},
            spectrum_inputs=np.ones(8, dtype=np.float32),
        )
    np.testing.assert_allclose(result, expected_anchor)


def test_equilibrium_prefers_earliest_trajectory_anchor(tiny_config):
    tiny_config["inference"]["equilibrium_anchor"]["source"] = "trajectory"
    splits, normalization = _prepare_processed_dataset(tiny_config)
    train = splits["train"]
    valid_steps = np.flatnonzero(train.valid_steps_mask[0])
    expected_anchor = inverse_block(
        train.state_trajectories[0, valid_steps[0]],
        normalization["state"],
    )
    model = PhysicalSpaceStandaloneModel(
        params=None,
        dims=None,
        normalization=normalization,
        contract={"state_species_order": tiny_config["data_spec"]["state_species"]},
        config=tiny_config,
        project_root=tiny_config["_project_root"],
    )
    nz = expected_anchor.shape[0]
    with patch.object(PhysicalSpaceStandaloneModel, "predict", autospec=True) as predict_mock:
        predict_mock.side_effect = _echo_anchor
        result = model.equilibrium(
            pressure_bar=np.logspace(2.0, -7.0, nz),
            temperature_K=np.linspace(900.0, 1200.0, nz),
            eddy_diffusion_cm2_s=np.full(nz, 1.0e8),
            global_inputs={},
            spectrum_inputs=np.ones(8, dtype=np.float32),
        )
    np.testing.assert_allclose(result, expected_anchor)


def test_equilibrium_anchor_step_index_uses_closest_saved_step(tiny_config):
    tiny_config["inference"]["equilibrium_anchor"]["source"] = "trajectory"
    tiny_config["inference"]["equilibrium_anchor"]["step_index"] = 3
    splits, normalization = _prepare_processed_dataset(tiny_config)
    train = splits["train"]
    valid_steps = np.flatnonzero(train.valid_steps_mask[0])
    selected_step = int(valid_steps[int(np.argmin(np.abs(valid_steps - 3)))])
    expected_anchor = inverse_block(
        train.state_trajectories[0, selected_step],
        normalization["state"],
    )
    model = PhysicalSpaceStandaloneModel(
        params=None,
        dims=None,
        normalization=normalization,
        contract={"state_species_order": tiny_config["data_spec"]["state_species"]},
        config=tiny_config,
        project_root=tiny_config["_project_root"],
    )
    nz = expected_anchor.shape[0]
    with patch.object(PhysicalSpaceStandaloneModel, "predict", autospec=True) as predict_mock:
        predict_mock.side_effect = _echo_anchor
        result = model.equilibrium(
            pressure_bar=np.logspace(2.0, -7.0, nz),
            temperature_K=np.linspace(900.0, 1200.0, nz),
            eddy_diffusion_cm2_s=np.full(nz, 1.0e8),
            global_inputs={},
            spectrum_inputs=np.ones(8, dtype=np.float32),
        )
    np.testing.assert_allclose(result, expected_anchor)
