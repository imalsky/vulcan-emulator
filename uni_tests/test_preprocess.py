from __future__ import annotations

import copy

import numpy as np

from src.data_generation.generation import generate_synthetic_raw_runs
from src.data_generation.data_loader import build_batch, load_processed_dataset
from src.data_generation.preprocess import preprocess_raw_dataset


def test_preprocess_and_batch(tiny_config):
    generate_synthetic_raw_runs(tiny_config, project_root=tiny_config["_project_root"])
    artifact = preprocess_raw_dataset(tiny_config, project_root=tiny_config["_project_root"])
    assert "processed_root" in artifact

    splits, normalization, contract = load_processed_dataset(tiny_config["paths"]["processed_root"])
    assert set(splits.keys()) == {"train", "val", "test"}
    assert contract["target_dim"] == len(tiny_config["data_spec"]["output_species"])
    assert contract["global_static_feature_order"] == tiny_config["data_spec"]["global_static_feature_order"]
    assert contract["global_static_feature_order"][:6] == [
        "gravity_cm_s2",
        "He_H",
        "C_H",
        "O_H",
        "N_H",
        "S_H",
    ]

    train = splits["train"]
    indices = np.arange(min(1, train.num_runs))
    batch = build_batch(train, indices)
    assert batch["sequence"].shape[-1] == contract["sequence_dim"]
    assert batch["target"].shape[-1] == contract["target_dim"]
    assert np.all(np.isfinite(batch["sequence"]))


def test_preprocess_respects_distinct_output_species(tiny_config):
    config = copy.deepcopy(tiny_config)
    config["data_spec"]["output_species"] = ["H2O", "CO", "CO2", "CH4"]
    generate_synthetic_raw_runs(config, project_root=config["_project_root"])
    preprocess_raw_dataset(config, project_root=config["_project_root"])

    splits, normalization, contract = load_processed_dataset(config["paths"]["processed_root"])
    train = splits["train"]
    indices = np.arange(min(1, train.num_runs))
    batch = build_batch(train, indices)
    assert batch["sequence"].shape[-1] == 3  # P, T, Kzz
    assert batch["target"].shape[-1] == len(config["data_spec"]["output_species"])
    assert contract["target_dim"] == len(config["data_spec"]["output_species"])
    assert "target_mode" not in contract
