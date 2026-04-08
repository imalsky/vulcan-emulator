from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from src.data_generation.data_loader import (
    build_batch,
    load_processed_dataset,
    processed_info_dir,
)
from src.data_generation.generation import generate_synthetic_raw_runs
from src.data_generation.preprocess import preprocess_raw_dataset


def test_preprocess_and_batch(tiny_config):
    generate_synthetic_raw_runs(tiny_config, project_root=tiny_config["_project_root"])
    processed_root = Path(tiny_config["paths"]["processed_root"])
    processed_root.mkdir(parents=True, exist_ok=True)
    for filename in ("data_contract.json", "normalization.json", "processed_manifest.json", "splits.json"):
        (processed_root / filename).write_text("{}", encoding="utf-8")

    artifact = preprocess_raw_dataset(tiny_config, project_root=tiny_config["_project_root"])
    assert "processed_root" in artifact

    info_dir = processed_info_dir(processed_root)
    assert info_dir.is_dir()
    for filename in ("data_contract.json", "normalization.json", "processed_manifest.json", "splits.json"):
        assert (info_dir / filename).is_file()
        assert not (processed_root / filename).exists()

    splits, normalization, contract = load_processed_dataset(tiny_config["paths"]["processed_root"])
    del normalization
    assert set(splits.keys()) == {"train", "val", "test"}
    assert contract["target_dim"] == len(tiny_config["data_spec"]["output_species"])
    assert contract["global_static_feature_order"] == tiny_config["data_spec"]["global_static_feature_order"]
    assert contract["global_static_feature_order"][:7] == [
        "gravity_cm_s2",
        "planet_radius_cm",
        "He_H",
        "C_H",
        "O_H",
        "N_H",
        "S_H",
    ]
    assert contract["spectrum_max_tokens"] == tiny_config["stellar_spectrum"]["max_tokens"]
    assert contract["spectrum_variable_length"] is True

    train = splits["train"]
    batch = build_batch(train, np.arange(min(1, train.num_runs)))
    assert batch["sequence"].shape[-1] == contract["sequence_dim"]
    assert batch["target"].shape[-1] == contract["target_dim"]
    assert batch["spectrum_wavelengths_nm"].shape[-1] == contract["spectrum_max_tokens"]
    assert batch["spectrum_fluxes_erg_cm2_s_nm"].shape[-1] == contract["spectrum_max_tokens"]
    assert batch["spectrum_mask"].shape[-1] == contract["spectrum_max_tokens"]
    assert np.all(np.isfinite(batch["sequence"]))
    assert np.all(batch["spectrum_fluxes_erg_cm2_s_nm"][~batch["spectrum_mask"]] == 0.0)


def test_preprocess_respects_distinct_output_species(tiny_config):
    config = copy.deepcopy(tiny_config)
    config["data_spec"]["output_species"] = ["H2O", "CO", "CO2", "CH4"]
    generate_synthetic_raw_runs(config, project_root=config["_project_root"])
    preprocess_raw_dataset(config, project_root=config["_project_root"])

    splits, normalization, contract = load_processed_dataset(config["paths"]["processed_root"])
    del normalization
    train = splits["train"]
    batch = build_batch(train, np.arange(min(1, train.num_runs)))
    assert batch["sequence"].shape[-1] == 3
    assert batch["target"].shape[-1] == len(config["data_spec"]["output_species"])
    assert contract["target_dim"] == len(config["data_spec"]["output_species"])
    assert "target_mode" not in contract


def test_dataset_run_layout_is_flat_after_generation_and_preprocess(tiny_config):
    generate_synthetic_raw_runs(tiny_config, project_root=tiny_config["_project_root"])
    preprocess_raw_dataset(tiny_config, project_root=tiny_config["_project_root"])

    run_root = Path(tiny_config["paths"]["raw_root"]).parent
    child_dirs = sorted(path.name for path in run_root.iterdir() if path.is_dir())

    assert "raw" in child_dirs
    assert "processed" in child_dirs
    processed_root = Path(tiny_config["paths"]["processed_root"])
    proc_dirs = sorted(path.name for path in processed_root.iterdir() if path.is_dir())
    assert "train" in proc_dirs
    assert "val" in proc_dirs
    assert "test" in proc_dirs
    assert "info" in proc_dirs
    assert (run_root / "raw" / "runs.h5").is_file()
    assert (run_root / "info" / "generation_manifest.json").is_file()
    assert (run_root / "info" / "sampling_coverage.json").is_file()
    assert (processed_root / "info" / "processed_manifest.json").is_file()
