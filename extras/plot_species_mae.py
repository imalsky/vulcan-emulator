#!/usr/bin/env python3
"""Plot per-species MAE on one processed split."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vulcan_emulator_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "vulcan_emulator_cache"))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
import torch

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise RuntimeError("matplotlib is required for plotting. Install project dependencies.") from exc

from script_utils import (
    build_model_from_checkpoint,
    denormalize,
    iter_split_shards,
    load_checkpoint,
    load_json,
    load_split_metadata,
    resolve_processed_root_from_checkpoint,
    resolve_run_dir,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
RUN_DIR_OVERRIDE: Path | None = None
TEST_SPLIT = "test"
CHECKPOINT_NAME = "best.pt"
BATCH_SIZE = 256
FIGURES_SUBDIR = "figures"
FILENAME = "species_mae.png"
TOP_K = 20
STYLE_PATH = Path(__file__).with_name("science.mplstyle")

plt.style.use(str(STYLE_PATH))


def main() -> None:
    """Compute and plot per-species MAE."""
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be > 0.")
    if TOP_K <= 0:
        raise ValueError("TOP_K must be > 0.")

    run_dir, config_path = resolve_run_dir(config_path=CONFIG_PATH, run_dir=RUN_DIR_OVERRIDE)
    processed_root = resolve_processed_root_from_checkpoint(run_dir=run_dir, checkpoint_name=CHECKPOINT_NAME)
    figures_dir = run_dir / FIGURES_SUBDIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(run_dir=run_dir, checkpoint_name=CHECKPOINT_NAME)
    split_meta = load_split_metadata(processed_root=processed_root, split=TEST_SPLIT)
    model, forward_dtype = build_model_from_checkpoint(
        checkpoint=checkpoint,
        split_metadata=split_meta,
        device=torch.device("cpu"),
    )
    target_stats = load_json(processed_root / "normalization_metadata.json")["targets"]["ymix"]
    species = list(split_meta["output_species_order"])

    abs_sum = np.zeros(len(species), dtype=np.float64)
    total_rows = 0
    for seq, glb, tgt, _dt in iter_split_shards(processed_root=processed_root, split=TEST_SPLIT):
        shard_samples = int(seq.shape[0])
        for start in range(0, shard_samples, BATCH_SIZE):
            end = min(start + BATCH_SIZE, shard_samples)
            seq_batch = torch.from_numpy(seq[start:end]).to(dtype=forward_dtype)
            glb_batch = torch.from_numpy(glb[start:end]).to(dtype=forward_dtype)
            tgt_batch = np.asarray(tgt[start:end], dtype=np.float64)

            with torch.inference_mode():
                pred = model(seq_batch, glb_batch, padding_mask=None)

            pred_phys = denormalize(pred.detach().cpu().numpy(), target_stats)
            tgt_phys = denormalize(tgt_batch, target_stats)
            abs_sum += np.sum(np.abs(pred_phys - tgt_phys), axis=(0, 1))
            total_rows += int(pred_phys.shape[0] * pred_phys.shape[1])

    if total_rows <= 0:
        raise RuntimeError("No samples were evaluated.")

    mae = abs_sum / float(total_rows)
    ranking = np.argsort(mae)[::-1]
    selected = ranking[: min(TOP_K, len(species))]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.bar(
        np.arange(len(selected), dtype=np.int64),
        np.clip(mae[selected], 1.0e-30, None),
        color="#2a6f97",
    )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(selected), dtype=np.int64))
    ax.set_xticklabels([species[idx] for idx in selected], rotation=45, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title(f"Per-Species MAE | split={TEST_SPLIT}")
    ax.set_box_aspect(1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = figures_dir / FILENAME
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Per-species MAE plot")
    print(f"  Config path : {config_path if config_path is not None else 'explicit run dir override'}")
    print(f"  Run dir     : {run_dir}")
    print(f"  Checkpoint  : {CHECKPOINT_NAME}")
    print(f"  Split       : {TEST_SPLIT}")
    print(f"  Output      : {output_path}")
    print("  Largest MAE species:")
    for idx in selected:
        print(f"    {species[idx]:<12} {mae[idx]:.6e}")


if __name__ == "__main__":
    main()
