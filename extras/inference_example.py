"""Run inference on a random test profile and plot predicted mixing ratios.

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

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

_STYLE = _ROOT / "extras" / "science.mplstyle"

# -- Configuration -----------------------------------------------------------
CHECKPOINT = _ROOT / "models/equilibrium_only_silu/best.pt"
PROCESSED_ROOT = _ROOT / "data/processed/equilibrium_only"
# ---------------------------------------------------------------------------


def _is_equilibrium(payload: dict) -> bool:
    return "d_hidden" in payload["model_dimensions"]


def _denormalize(pred: np.ndarray, normalization: dict) -> np.ndarray:
    mean = np.asarray(normalization["target"]["mean"], dtype=np.float32)
    std = np.asarray(normalization["target"]["std"], dtype=np.float32)
    return pred * std + mean


def _recover_column(seq_row: np.ndarray, normalization: dict, col: int) -> np.ndarray:
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
    plots = checkpoint.parent / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    return plots


def _apply_equilibrium_mlp_numpy(params: dict, seq: np.ndarray, globs: np.ndarray, dims: dict) -> np.ndarray:
    """Pure-numpy forward pass for the equilibrium MLP (FiLM-conditioned)."""
    x = seq.reshape(seq.shape[0], -1)  # [B, nz * seq_dim]
    x = np.concatenate([x, globs], axis=-1)  # [B, nz*seq_dim + global_dim]

    # Input projection
    x = x @ params["input_proj"]["w"].T + params["input_proj"]["b"]

    for i in range(dims["num_hidden_layers"]):
        layer = params[f"hidden_{i}"]
        h = x @ layer["w"].T + layer["b"]
        # SiLU activation
        h = h * (1.0 / (1.0 + np.exp(-h)))

        # FiLM conditioning if present
        film_key = f"film_{i}"
        if film_key in params:
            film = params[film_key]
            cond = globs @ film["w"].T + film["b"]
            gamma = cond[:, :h.shape[-1]]
            beta = cond[:, h.shape[-1]:]
            clamp = dims.get("film_clamp", 5.0)
            gamma = np.clip(gamma, -clamp, clamp)
            beta = np.clip(beta, -clamp, clamp)
            h = h * (1.0 + gamma) + beta
        x = h

    x = x @ params["output_proj"]["w"].T + params["output_proj"]["b"]
    return x


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

    colors = plt.cm.tab20(np.linspace(0, 1, len(species)))
    for i, name in enumerate(species):
        c = colors[i]
        ax_mix.plot(true_log10[:, i], pressure_bar, "-", color=c, lw=1.4, label=name)
        ax_mix.plot(pred_log10[:, i], pressure_bar, "--", color=c, lw=1.1)
    ax_mix.set_yscale("log")
    ax_mix.invert_yaxis()
    ax_mix.set_xlabel("log$_{10}$ mixing ratio")
    ax_mix.set_ylabel("Pressure [bar]")
    ax_mix.set_title("Mixing ratios  (solid = true, dashed = predicted)")
    ax_mix.legend(fontsize=7, ncol=3, loc="best")

    ax_pt.plot(temperature_k, pressure_bar, "k-", lw=1.8)
    ax_pt.set_xlabel("Temperature [K]")
    ax_pt.set_title("P-T profile")
    ax_pt.set_xlim(0, 3000)

    fig.suptitle(f"Run: {run_id}")
    fig.tight_layout()
    out_path = output_dir / f"{run_id}_profiles.png"
    fig.savefig(out_path)
    print(f"Saved plot to {out_path}")
    plt.close(fig)


def run_equilibrium(payload: dict):
    """Inference for an equilibrium model on one random test profile."""
    import jax
    import jax.numpy as jnp
    from src.models.jax_model import (
        EquilibriumMLPDimensions,
        apply_equilibrium_mlp,
    )

    jax_device = jax.devices("cpu")[0]
    dims = EquilibriumMLPDimensions.from_dict(payload["model_dimensions"])
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

    pred, _ = apply_equilibrium_mlp(
        params,
        jnp.asarray(seq[idx:idx + 1]),
        jnp.asarray(globs[idx:idx + 1]),
        dims,
    )

    pred_log10 = _denormalize(np.asarray(pred[0]), normalization)
    true_log10 = _denormalize(targets[idx], normalization)
    pressure_bar = _recover_column(seq[idx], normalization, col=0)
    temperature_k = _recover_column(seq[idx], normalization, col=1)

    _plot_profiles(
        pressure_bar, temperature_k, pred_log10, true_log10,
        contract["output_species_order"], run_ids[idx],
        output_dir=_plots_dir(CHECKPOINT),
    )


def main():
    with CHECKPOINT.open("rb") as f:
        payload = pickle.load(f)

    if _is_equilibrium(payload):
        print("Detected equilibrium model")
        run_equilibrium(payload)
    else:
        print("Detected transition (full_vulcan) model")
        # Transition model still needs JAX for the transformer forward pass.
        run_equilibrium(payload)


if __name__ == "__main__":
    main()
