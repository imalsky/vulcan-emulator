"""Run inference on a random FastChem test profile and plot mixing ratios.

Loads a trained checkpoint and the processed test split, picks one profile
at random, runs the forward pass, denormalizes predictions back to log10
mixing-ratio space, and plots them against the true values.

Usage:
    python extras/inference_example.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.utils.numpy_compat import patch_numpy_asarray_copy

patch_numpy_asarray_copy()

import matplotlib.pyplot as plt
import numpy as np

_STYLE = _ROOT / "extras" / "science.mplstyle"

# -- Configuration -----------------------------------------------------------
CHECKPOINT = _ROOT / "models" / "fastchem_mlp" / "best.pt"
PROCESSED_ROOT = _ROOT / "data/processed/fastchem_mlp"
# ---------------------------------------------------------------------------


def _is_fastchem(payload: dict[str, Any]) -> bool:
    """Return True when the checkpoint payload targets FastChem chemistry."""
    return str(payload.get("config", {}).get("chemistry_type", "")) == "fastchem"


def _restore_target_log10(values: np.ndarray, normalization: dict[str, Any]) -> np.ndarray:
    """Convert normalized target outputs back to log10 mixing-ratio space."""
    block = normalization["target"]
    method = str(block["method"])
    mean = np.asarray(block["mean"], dtype=np.float32)
    std = np.asarray(block["std"], dtype=np.float32)
    if method == "log-standard":
        return values * std + mean
    if method == "standard":
        linear = values * std + mean
        return np.log10(np.clip(linear, 1.0e-30, None))
    if method == "none":
        return np.log10(np.clip(values, 1.0e-30, None))
    raise ValueError(f"Unsupported target normalization method: {method}")


def _recover_column(seq_row: np.ndarray, normalization: dict[str, Any], col: int) -> np.ndarray:
    """Undo one saved sequence-feature normalization block back to physical space."""
    block = normalization["sequence_static"]["blocks"][col]
    mean = np.asarray(block["mean"], dtype=np.float64)
    std = np.asarray(block["std"], dtype=np.float64)
    normed = seq_row[:, col]
    if block["method"] == "log-standard":
        return 10.0 ** (normed * std + mean)
    if block["method"] == "standard":
        return normed * std + mean
    return normed


def _plots_dir(checkpoint: Path) -> Path:
    """Return the plots directory colocated with the selected checkpoint."""
    plots = checkpoint.parent / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    return plots


def _plot_profiles(
    pressure_bar: np.ndarray,
    temperature_k: np.ndarray,
    pred_log10: np.ndarray,
    true_log10: np.ndarray,
    species: list[str],
    run_id: str,
    output_dir: Path,
):
    plt.style.use(str(_STYLE))
    fig, (ax_mix, ax_pt) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    import matplotlib.ticker as mticker

    # Reduce number of y-axis ticks (pressure)
    ax_mix.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
    ax_mix.yaxis.set_minor_locator(mticker.NullLocator())


    colors = plt.cm.tab20(np.linspace(0, 1, len(species)))
    for i, name in enumerate(species):
        c = colors[i]
        ax_mix.plot(10 ** true_log10[:, i], pressure_bar, "-", color=c, lw=1.4, label=name)
        ax_mix.plot(10 ** pred_log10[:, i], pressure_bar, "--", color=c, lw=1.1)
    ax_mix.set_yscale("log")
    ax_mix.set_xscale("log")
    ax_mix.set_xlim(1e-20, 3)
    ax_mix.invert_yaxis()
    ax_mix.set_xlabel("Mixing Ratio")
    ax_mix.set_ylabel("Pressure [bar]")
    #ax_mix.set_title("Mixing ratios  (solid = true, dashed = predicted)")
    ax_mix.legend(fontsize=7, ncol=3, loc="best")

    ax_pt.plot(temperature_k, pressure_bar, "k-", lw=1.8)
    ax_pt.set_xlabel("Temperature [K]")
    #ax_pt.set_title("P-T profile")
    ax_pt.set_xlim(0, 3000)

    #fig.suptitle(f"Run: {run_id}")
    fig.tight_layout()
    out_path = output_dir / f"{run_id}_profiles.png"
    fig.savefig(out_path)
    print(f"Saved plot to {out_path}")
    plt.close(fig)


def run_fastchem(payload: dict[str, Any]) -> None:
    """Inference for a FastChem model on one random test profile."""
    import jax
    import jax.numpy as jnp
    from src.models.jax_model import (
        MLPDimensions,
        TransformerDimensions,
        apply_mlp,
        apply_transformer_model,
    )

    jax_device = jax.devices("cpu")[0]
    model_type = str(payload["config"]["model_type"])
    if model_type == "mlp":
        dims = MLPDimensions.from_dict(payload["model_dimensions"])
    else:
        dims = TransformerDimensions.from_dict(payload["model_dimensions"])
    params = jax.device_put(payload["params"], jax_device)
    normalization = payload["normalization"]
    contract = payload["data_contract"]

    test_dir = PROCESSED_ROOT / "test"
    seq = np.load(test_dir / "sequence_inputs.npy")
    targets = np.load(test_dir / "target_outputs.npy")
    globs = np.load(test_dir / "global_inputs.npy")
    run_ids = json.loads((test_dir / "run_ids.json").read_text())

    idx = np.random.default_rng().integers(0, seq.shape[0])
    print(f"Selected test profile: {run_ids[idx]} (index {idx}/{seq.shape[0]})")

    if model_type == "mlp":
        pred, _ = apply_mlp(
            params,
            jnp.asarray(seq[idx:idx + 1]),
            jnp.asarray(globs[idx:idx + 1]),
            dims,
        )
    else:
        pred, _ = apply_transformer_model(
            params,
            jnp.asarray(seq[idx:idx + 1]),
            jnp.asarray(globs[idx:idx + 1]),
            None,
            dims,
        )

    pred_log10 = _restore_target_log10(np.asarray(pred[0]), normalization)
    true_log10 = _restore_target_log10(targets[idx], normalization)
    pressure_bar = _recover_column(seq[idx], normalization, col=0)
    temperature_k = _recover_column(seq[idx], normalization, col=1)

    _plot_profiles(
        pressure_bar, temperature_k, pred_log10, true_log10,
        contract["output_species_order"], run_ids[idx],
        output_dir=_plots_dir(CHECKPOINT),
    )


def main() -> None:
    with CHECKPOINT.open("rb") as f:
        payload = pickle.load(f)

    if not _is_fastchem(payload):
        raise NotImplementedError(
            "extras/inference_example.py currently supports FastChem checkpoints only. "
            "Use the ExoJAX wrapper or a dedicated VULCAN example for vulcan bundles."
        )
    print(f"Detected FastChem {payload['config']['model_type']} model")
    run_fastchem(payload)


if __name__ == "__main__":
    main()
