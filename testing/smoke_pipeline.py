#!/usr/bin/env python3
"""Run a full tiny end-to-end generation, training, export, and inference smoke test."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# Prevent duplicate OpenMP runtime aborts before importing torch via src modules.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import iter_split_shards
from inference import VulcanPredictor, load_physical_space_model, physical_inputs_from_processed_arrays

CONFIG_PATH = PROJECT_ROOT / "config" / "tiny_train_smoke.json"


def _run(command: list[str]) -> None:
    """Run one subprocess from the project root and fail on non-zero exit."""
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


def main() -> None:
    """Execute the tiny end-to-end smoke pipeline and validate key artifacts."""
    if os.environ.get("CONDA_DEFAULT_ENV") != "nn":
        raise RuntimeError("Run this smoke script inside the 'nn' conda environment.")

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    data_root = PROJECT_ROOT / str(config["paths"]["data_root"])
    logs_root = PROJECT_ROOT / str(config["paths"]["logs_root"])
    run_dir = PROJECT_ROOT / str(config["paths"]["models_root"]) / str(config["training"]["output_folder"])
    processed_root = data_root / "processed"
    raw_runs_root = data_root / "raw" / "runs"

    for path in (data_root, logs_root, run_dir):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    config_relpath = str(CONFIG_PATH.relative_to(PROJECT_ROOT))
    _run([sys.executable, "src/main.py", "--config", config_relpath, "--gen"])
    _run([sys.executable, "src/main.py", "--config", config_relpath, "--train"])
    _run(
        [
            sys.executable,
            "testing/export.py",
            "--run-dir",
            str(run_dir.relative_to(PROJECT_ROOT)),
        ]
    )

    required_paths = [
        data_root / "dataset_manifest.json",
        data_root / "splits.json",
        processed_root / "processed_summary.json",
        processed_root / "processed_fingerprint.json",
        run_dir / "best.pt",
        run_dir / "last.pt",
        run_dir / "data_contract.json",
        run_dir / "normalization_metadata.json",
        run_dir / "processed_fingerprint.json",
        run_dir / "standalone_model.pt2",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Smoke pipeline missing required artifacts: {missing}")

    raw_runs = sorted(raw_runs_root.glob("run_*.h5"))
    expected_runs = int(config["generation"]["num_runs"])
    if len(raw_runs) != expected_runs:
        raise RuntimeError(f"Expected {expected_runs} raw runs, found {len(raw_runs)}.")

    model, normalization_metadata, data_contract = load_physical_space_model(run_dir)
    seq, glb, _tgt = next(iter(iter_split_shards(processed_root=processed_root, split="test")))
    example = physical_inputs_from_processed_arrays(
        sequence_inputs=seq[0],
        global_inputs=glb[0],
        normalization_metadata=normalization_metadata,
        data_contract=data_contract,
    )
    predictor = VulcanPredictor.from_run_dir(run_dir)
    prediction = predictor.predict(
        pressure_bar=example["pressure_bar"],
        temperature_k=example["temperature_k"],
        kzz_cm2_s=example["kzz_cm2_s"],
        initial_ymix=example["initial_ymix"],
        gravity_cm_s2=example["gravity_cm_s2"],
        metallicity_log10=example["metallicity_log10"],
        c_to_o=example["c_to_o"],
        time_s=example["time_s"],
    )
    if prediction.shape != (seq.shape[1], data_contract["target_dim"]):
        raise RuntimeError(
            "Unexpected predictor output shape: "
            f"{prediction.shape} != {(seq.shape[1], data_contract['target_dim'])}"
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError("Predictor produced non-finite physical-space outputs.")
    if len(model.target_species) != data_contract["target_dim"]:
        raise RuntimeError("Loaded target species count does not match data contract.")

    print("Smoke pipeline completed successfully.")


if __name__ == "__main__":
    main()
