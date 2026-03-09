#!/usr/bin/env python3
"""Plot training progression curves from training_log.csv."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from script_utils import PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vulcan_emulator_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "vulcan_emulator_cache"))

RUN_DIR = PROJECT_ROOT / "models" / "trained_model"
FIGURES_SUBDIR = "figures"
FILENAME = "training_progression.png"
STYLE_PATH = Path(__file__).with_name("science.mplstyle")


def _load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting. Install project dependencies.") from exc
    plt.style.use(str(STYLE_PATH))
    return plt


def _load_log(
    path: Path,
) -> tuple[list[int], list[float], list[float], list[float], list[float], list[float]]:
    """Load epoch metrics from one CSV training log."""
    epochs: list[int] = []
    train_mse: list[float] = []
    val_mse: list[float] = []
    train_mae_log10: list[float] = []
    val_mae_log10: list[float] = []
    lrs: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_mse.append(float(row["train_mse"]))
            val_mse.append(float(row["val_mse"]))
            train_mae_log10.append(float(row["train_mae_log10"]))
            val_mae_log10.append(float(row["val_mae_log10"]))
            lrs.append(float(row["lr"]))

    if not epochs:
        raise RuntimeError(f"No rows found in training log: {path}")
    return epochs, train_mse, val_mse, train_mae_log10, val_mae_log10, lrs


def main() -> None:
    """Render training and learning-rate curves from one run directory."""
    plt = _load_pyplot()
    run_dir = RUN_DIR.resolve()
    figures_dir = run_dir / FIGURES_SUBDIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "training_log.csv"
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing training log: {log_path}")

    epochs, train_mse, val_mse, train_mae_log10, val_mae_log10, lrs = _load_log(log_path)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(epochs, train_mse, label="train_mse", linewidth=2.0)
    axes[0].plot(epochs, val_mse, label="val_mse", linewidth=2.0)
    axes[0].plot(epochs, train_mae_log10, label="train_mae_log10", linewidth=2.0, linestyle="--")
    axes[0].plot(epochs, val_mae_log10, label="val_mae_log10", linewidth=2.0, linestyle=":")
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

    output_path = figures_dir / FILENAME
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Training progression plot")
    print(f"  Run dir : {run_dir}")
    print(f"  Log     : {log_path}")
    print(f"  Output  : {output_path}")


if __name__ == "__main__":
    main()
