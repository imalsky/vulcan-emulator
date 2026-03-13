#!/usr/bin/env python3
"""Plot one true-vs-predicted mixing-ratio profile in log-log space."""

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

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise RuntimeError("matplotlib is required for plotting. Install project dependencies.") from exc

from inference import VulcanPredictor, load_physical_space_model, physical_inputs_from_processed_arrays
from script_utils import (
    denormalize,
    iter_fixed_split_batches,
    load_checkpoint,
    load_fixed_split_sample,
    load_json,
    resolve_processed_root_from_checkpoint,
)

MODEL_DIR_NAME = "v4"
TEST_SPLIT = "test"
CHECKPOINT_NAME = "best.pt"
RANDOM_SEED: int | None = None
FIGURES_SUBDIR = "figures"
PROFILE_FIGURE_STEM = "true_vs_pred_profile"
TP_FIGURE_STEM = "tp_profile"
MAX_SPECIES = 8
DEVICE = "cpu"
SAMPLE_SELECTION_MODE = "uniform_log_dt"
LOG_DT_BIN_COUNT = 24
DT_SCAN_BATCH_SIZE = 2048
STYLE_PATH = Path(__file__).with_name("science.mplstyle")

plt.style.use(str(STYLE_PATH))


def _resolve_run_dir() -> Path:
    """Resolve the selected trained model directory from ``models/<MODEL_DIR_NAME>``."""
    model_dir_name = MODEL_DIR_NAME.strip()
    if not model_dir_name:
        raise ValueError("MODEL_DIR_NAME must be a non-empty model folder name.")

    run_dir = (PROJECT_ROOT / "models" / model_dir_name).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {run_dir}")
    return run_dir


def _choose_sample_index(
    *,
    processed_root: Path,
    config: dict,
    normalization_metadata: dict,
) -> int:
    """Select one fixed-eval sample index, optionally balancing coverage in log10(dt)."""
    dt_batches: list[np.ndarray] = []
    for _seq, _glb, _tgt, dt in iter_fixed_split_batches(
        processed_root=processed_root,
        split=TEST_SPLIT,
        config=config,
        normalization_metadata=normalization_metadata,
        batch_size=DT_SCAN_BATCH_SIZE,
    ):
        dt_batches.append(np.asarray(dt, dtype=np.float64).reshape(-1))
    if not dt_batches:
        raise RuntimeError(f"No samples are available in split '{TEST_SPLIT}'.")

    dt_values = np.concatenate(dt_batches, axis=0)
    total_samples = int(dt_values.size)
    if total_samples <= 0:
        raise RuntimeError(f"No samples are available in split '{TEST_SPLIT}'.")

    rng = np.random.default_rng(RANDOM_SEED)
    if SAMPLE_SELECTION_MODE == "uniform":
        return int(rng.integers(0, total_samples))
    if SAMPLE_SELECTION_MODE != "uniform_log_dt":
        raise ValueError(
            f"Unsupported SAMPLE_SELECTION_MODE={SAMPLE_SELECTION_MODE!r}. "
            "Use 'uniform' or 'uniform_log_dt'."
        )

    log_dt = np.log10(np.clip(dt_values, np.finfo(np.float64).tiny, None))
    log_dt_min = float(np.min(log_dt))
    log_dt_max = float(np.max(log_dt))
    if not np.isfinite(log_dt_min) or not np.isfinite(log_dt_max):
        raise RuntimeError("Encountered non-finite dt values while selecting a plot sample.")
    if log_dt_max <= log_dt_min:
        return int(rng.integers(0, total_samples))

    bin_edges = np.linspace(log_dt_min, log_dt_max, num=LOG_DT_BIN_COUNT + 1, dtype=np.float64)
    bin_ids = np.digitize(log_dt, bin_edges[1:-1], right=False)
    bin_counts = np.bincount(bin_ids, minlength=LOG_DT_BIN_COUNT)
    non_empty_bins = np.flatnonzero(bin_counts > 0)
    if non_empty_bins.size == 0:
        raise RuntimeError("Failed to identify any non-empty log-dt bins for sample selection.")

    chosen_bin = int(rng.choice(non_empty_bins))
    candidate_indices = np.flatnonzero(bin_ids == chosen_bin)
    if candidate_indices.size == 0:
        raise RuntimeError(f"Selected empty log-dt bin {chosen_bin} during sample selection.")
    return int(rng.choice(candidate_indices))


def main() -> None:
    """Render one true-vs-predicted profile comparison."""
    if MAX_SPECIES <= 0:
        raise ValueError("MAX_SPECIES must be > 0.")

    run_dir = _resolve_run_dir()
    checkpoint = load_checkpoint(run_dir=run_dir, checkpoint_name=CHECKPOINT_NAME)
    processed_root = resolve_processed_root_from_checkpoint(run_dir=run_dir, checkpoint_name=CHECKPOINT_NAME)
    normalization_metadata = load_json(processed_root / "normalization_metadata.json")
    sample_index = _choose_sample_index(
        processed_root=processed_root,
        config=checkpoint["config"],
        normalization_metadata=normalization_metadata,
    )
    figures_dir = run_dir / FIGURES_SUBDIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    seq, glb, tgt, dt_s = load_fixed_split_sample(
        processed_root=processed_root,
        split=TEST_SPLIT,
        config=checkpoint["config"],
        normalization_metadata=normalization_metadata,
        sample_index=sample_index,
    )
    wrapper, normalization_metadata, data_contract = load_physical_space_model(
        run_dir,
        checkpoint_name=CHECKPOINT_NAME,
        device=DEVICE,
    )
    predictor = VulcanPredictor(wrapper, device=wrapper.epsilon.device)
    example = physical_inputs_from_processed_arrays(
        sequence_inputs=seq,
        global_inputs=glb,
        normalization_metadata=normalization_metadata,
        data_contract=data_contract,
    )
    target_phys = denormalize(tgt, normalization_metadata["targets"]["ymix"])
    prediction = predictor.predict(
        pressure_bar=example["pressure_bar"],
        temperature_k=example["temperature_k"],
        kzz_cm2_s=example["kzz_cm2_s"],
        anchor_ymix=example["anchor_ymix"],
        gravity_cm_s2=example["gravity_cm_s2"],
        metallicity_log10=example["metallicity_log10"],
        c_to_o=example["c_to_o"],
        dt_s=example["dt_s"],
    )

    species = list(example["output_species"])
    ranking = np.argsort(np.max(target_phys, axis=0))[::-1]
    selected = ranking[: min(MAX_SPECIES, len(species))]
    pressure_bar = np.clip(np.asarray(example["pressure_bar"], dtype=np.float64), 1.0e-30, None)

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap("tab10")
    for plot_idx, species_idx in enumerate(selected):
        color = cmap(plot_idx % 10)
        species_name = species[species_idx]
        true_profile = np.clip(target_phys[:, species_idx], 1.0e-30, None)
        pred_profile = np.clip(prediction[:, species_idx], 1.0e-30, None)
        ax.plot(true_profile, pressure_bar, color=color, linewidth=2.0, label=f"{species_name} true")
        ax.plot(
            pred_profile,
            pressure_bar,
            color=color,
            linewidth=2.0,
            linestyle="--",
            label=f"{species_name} pred",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlim(1e-10, 2)
    ax.set_box_aspect(1.0)

    ax.set_xlabel("Mixing Ratio")
    ax.set_ylabel("Pressure (bar)")
    ax.set_title(
        f"True vs Predicted Mixing Ratios | split={TEST_SPLIT} sample={sample_index} | dt={float(dt_s):.2e} s"
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    output_path = figures_dir / f"{PROFILE_FIGURE_STEM}_sample_{sample_index:05d}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    tp_fig, tp_ax = plt.subplots(figsize=(7, 7))
    tp_ax.plot(example["temperature_k"], pressure_bar, color="#2753DB", linewidth=2.0)
    tp_ax.set_yscale("log")
    tp_ax.invert_yaxis()
    tp_ax.set_xlim(200, 2500)
    tp_ax.set_box_aspect(1.0)
    tp_ax.set_xlabel("Temperature (K)")
    tp_ax.set_ylabel("Pressure (bar)")
    tp_ax.set_title(f"T-P Profile | split={TEST_SPLIT} sample={sample_index}")
    tp_fig.tight_layout()

    tp_output_path = figures_dir / f"{TP_FIGURE_STEM}_sample_{sample_index:05d}.png"
    tp_fig.savefig(tp_output_path, dpi=180, bbox_inches="tight")
    plt.close(tp_fig)

    print("True vs predicted profile")
    print("  Config path  : checkpoint config")
    print(f"  Model dir    : {MODEL_DIR_NAME}")
    print(f"  Run dir      : {run_dir}")
    print(f"  Checkpoint   : {CHECKPOINT_NAME}")
    print(f"  Split/sample : {TEST_SPLIT} / {sample_index}")
    print(f"  Selection    : {SAMPLE_SELECTION_MODE}")
    print(f"  Device       : {DEVICE}")
    print(f"  dt_s         : {float(dt_s):.6e}")
    print(f"  Profile plot : {output_path}")
    print(f"  T-P plot     : {tp_output_path}")
    print(f"  Species      : {', '.join(species[idx] for idx in selected)}")


if __name__ == "__main__":
    main()
