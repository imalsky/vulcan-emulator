#!/usr/bin/env python3
"""Plot training progression curves from training_log.csv."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
if str(TESTING_DIR) not in sys.path:
    sys.path.insert(0, str(TESTING_DIR))

from common import PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vulcan_emulator_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "vulcan_emulator_cache"))

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise RuntimeError("matplotlib is required for plotting. Install project dependencies.") from exc


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training-curve plotting."""
    parser = argparse.ArgumentParser(description="Plot training progression.")
    parser.add_argument("--run-dir", type=Path, default=Path("models/trained_model"))
    parser.add_argument("--out-dir", type=Path, default=Path("testing/figures"))
    parser.add_argument("--filename", type=str, default="training_progression.png")
    return parser.parse_args()


def _load_log(path: Path) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    """Load epoch metrics from one CSV training log."""
    epochs: list[int] = []
    train_mse: list[float] = []
    val_mse: list[float] = []
    val_mae: list[float] = []
    lrs: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_mse.append(float(row["train_mse"]))
            val_mse.append(float(row["val_mse"]))
            val_mae.append(float(row["val_mae"]))
            lrs.append(float(row["lr"]))

    if not epochs:
        raise RuntimeError(f"No rows found in training log: {path}")
    return epochs, train_mse, val_mse, val_mae, lrs


def main() -> None:
    """Render training and learning-rate curves from one run directory."""
    args = _parse_args()
    run_dir = (PROJECT_ROOT / args.run_dir).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "training_log.csv"
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing training log: {log_path}")

    epochs, train_mse, val_mse, val_mae, lrs = _load_log(log_path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(epochs, train_mse, label="train_mse", linewidth=2.0)
    axes[0].plot(epochs, val_mse, label="val_mse", linewidth=2.0)
    axes[0].plot(epochs, val_mae, label="val_mae", linewidth=2.0, linestyle="--")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Progression")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, lrs, color="#444444", linewidth=2.0, label="learning_rate")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    output_path = out_dir / args.filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved training progression figure: {output_path}")


if __name__ == "__main__":
    main()
