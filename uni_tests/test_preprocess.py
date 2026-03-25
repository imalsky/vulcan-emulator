from __future__ import annotations

import copy

import numpy as np

from src.config_utils import effective_transition_sampling
from src.data_loader import build_batch_from_rows, load_processed_dataset
from src.preprocess import preprocess_raw_dataset
from src.transition_sampling import build_candidate_table
from src.vulcan_runner import generate_synthetic_raw_runs


def test_preprocess_and_live_batch(tiny_config):
    generate_synthetic_raw_runs(tiny_config, project_root=tiny_config["_project_root"])
    artifact = preprocess_raw_dataset(tiny_config, project_root=tiny_config["_project_root"])
    assert "processed_root" in artifact

    splits, normalization, contract = load_processed_dataset(tiny_config["paths"]["processed_root"])
    assert set(splits.keys()) == {"train", "val", "test"}
    assert contract["target_dim"] == len(tiny_config["data_spec"]["output_species"])
    assert len(normalization["state"]["mean"]) == len(tiny_config["data_spec"]["state_species"])

    train = splits["train"]
    transition_sampling = effective_transition_sampling(tiny_config)
    candidate_table = build_candidate_table(
        train.time_s,
        train.valid_steps_mask,
        dt_min_s=float(transition_sampling["dt_min_s"]),
        dt_max_s=float(transition_sampling["dt_max_s"]),
        min_future_saved_steps=int(transition_sampling["min_future_saved_steps"]),
        log10_dt_stats={
            "mean": float(normalization["log10_dt_s"]["mean"][0]),
            "std": float(normalization["log10_dt_s"]["std"][0]),
        },
    )
    rows = candidate_table.rows_for_run(0)[:1]
    batch = build_batch_from_rows(train, candidate_table, rows)
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
    transition_sampling = effective_transition_sampling(config)
    candidate_table = build_candidate_table(
        train.time_s,
        train.valid_steps_mask,
        dt_min_s=float(transition_sampling["dt_min_s"]),
        dt_max_s=float(transition_sampling["dt_max_s"]),
        min_future_saved_steps=int(transition_sampling["min_future_saved_steps"]),
        log10_dt_stats={
            "mean": float(normalization["log10_dt_s"]["mean"][0]),
            "std": float(normalization["log10_dt_s"]["std"][0]),
        },
    )
    rows = candidate_table.rows_for_run(0)[:1]
    batch = build_batch_from_rows(train, candidate_table, rows)
    assert batch["sequence"].shape[-1] == 3 + len(config["data_spec"]["state_species"])
    assert batch["target"].shape[-1] == len(config["data_spec"]["output_species"])
    assert contract["target_dim"] == len(config["data_spec"]["output_species"])
    assert contract["target_mode"] == config["generation"]["target_mode"]
