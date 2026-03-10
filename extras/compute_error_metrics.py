#!/usr/bin/env python3
"""Compute physical-unit error metrics on a processed split."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
import torch

from script_utils import (
    PROJECT_ROOT,
    build_model_from_checkpoint,
    denormalize,
    iter_fixed_split_batches,
    load_checkpoint,
    load_json,
    load_split_metadata,
    resolve_run_dir,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
RUN_DIR_OVERRIDE: Path | None = None
FIGURES_SUBDIR = "figures"


def main() -> None:
    run_dir, config_path = resolve_run_dir(config_path=CONFIG_PATH, run_dir=RUN_DIR_OVERRIDE)
    split = "test"
    checkpoint_name = "best.pt"
    batch_size = 256

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    out_dir = (run_dir / FIGURES_SUBDIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(run_dir=run_dir, checkpoint_name=checkpoint_name)
    config = checkpoint["config"]
    processed_root = (PROJECT_ROOT / str(config["paths"]["processed_root"])).resolve()
    split_meta = load_split_metadata(processed_root=processed_root, split=split)
    model, forward_dtype = build_model_from_checkpoint(
        checkpoint=checkpoint,
        split_metadata=split_meta,
        device=torch.device("cpu"),
    )

    norm_meta = load_json(processed_root / "normalization_metadata.json")
    target_stats = norm_meta["targets"]["ymix"]
    species = list(split_meta["output_species_order"])
    n_species = len(species)

    abs_sum = np.zeros(n_species, dtype=np.float64)
    sq_sum = np.zeros(n_species, dtype=np.float64)
    abs_pct_sum = np.zeros(n_species, dtype=np.float64)
    rows = 0

    for seq, glb, tgt, _dt in iter_fixed_split_batches(
        processed_root=processed_root,
        split=split,
        config=config,
        normalization_metadata=norm_meta,
        batch_size=batch_size,
    ):
        seq_batch = torch.from_numpy(seq).to(dtype=forward_dtype)
        glb_batch = torch.from_numpy(glb).to(dtype=forward_dtype)
        tgt_batch = np.asarray(tgt, dtype=np.float64)

        with torch.inference_mode():
            pred = model(seq_batch, glb_batch, padding_mask=None)

        pred_phys = denormalize(pred.detach().cpu().numpy(), target_stats)
        tgt_phys = denormalize(tgt_batch, target_stats)
        diff = pred_phys - tgt_phys
        abs_sum += np.sum(np.abs(diff), axis=(0, 1))
        sq_sum += np.sum(diff * diff, axis=(0, 1))
        denom = np.maximum(np.abs(tgt_phys), 1e-30)
        abs_pct_sum += np.sum(np.abs(diff) / denom, axis=(0, 1))
        rows += diff.shape[0] * diff.shape[1]

    if rows <= 0:
        raise RuntimeError("No samples were evaluated.")

    total_elements = rows * n_species
    per_species = []
    for idx, name in enumerate(species):
        per_species.append(
            {
                "species": name,
                "mae": abs_sum[idx] / rows,
                "rmse": float(np.sqrt(sq_sum[idx] / rows)),
                "mape_percent": (abs_pct_sum[idx] / rows) * 100.0,
            }
        )

    summary = {
        "split": split,
        "checkpoint": checkpoint_name,
        "samples": rows,
        "species_count": n_species,
        "overall_mae": float(abs_sum.sum() / total_elements),
        "overall_rmse": float(np.sqrt(sq_sum.sum() / total_elements)),
        "overall_mape_percent": float((abs_pct_sum.sum() / total_elements) * 100.0),
        "per_species": per_species,
    }

    json_path = out_dir / "error_metrics.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    csv_path = out_dir / "error_metrics_per_species.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["species", "mae", "rmse", "mape_percent"])
        for item in per_species:
            writer.writerow([item["species"], item["mae"], item["rmse"], item["mape_percent"]])

    print(f"Saved error summary: {json_path}")
    print(f"Saved per-species metrics: {csv_path}")
    print(f"Resolved config path: {config_path if config_path is not None else 'explicit run dir override'}")


if __name__ == "__main__":
    main()
