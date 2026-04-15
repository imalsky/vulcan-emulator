"""Minimal standalone example: run the exported FastChem emulator.

Shows the simplest possible inference path: load the bundle, predict mixing
ratios, print results.  No src/ imports — the inference module is extracted
directly from the bundle.

Usage
-----
    python extras/stand_alone_example.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = PROJECT_ROOT / "models" / "fastchem_transformer" / "best_exported.npz"

# ---------------------------------------------------------------------------
# Load the inference module from the bundle — no src/ import needed.
# ---------------------------------------------------------------------------
_src = bytes(np.load(BUNDLE_PATH, allow_pickle=False)["meta/vulcan_emulator_src"]).decode()
_mod = types.ModuleType("vulcan_emulator")
sys.modules["vulcan_emulator"] = _mod
exec(compile(_src, "vulcan_emulator", "exec"), _mod.__dict__)
load_model = _mod.load_model

# ---------------------------------------------------------------------------
# Load model and define atmospheric inputs.
# ---------------------------------------------------------------------------
model = load_model(BUNDLE_PATH)
print(f"Loaded: {model.chemistry_type} {model.model_type}  ({len(model.species)} species)")

# 50-level pressure grid matching the training domain: 100 → 1e-7 bar.
pressure_bar = np.logspace(2, -7, 50)
temperature_k = np.full_like(pressure_bar, 1500.0)

global_inputs = {
    "He_H": 7.84e-2,
    "C_H":  2.69e-4,
    "O_H":  4.90e-4,
    "N_H":  6.76e-5,
    "S_H":  1.32e-5,
}

# ---------------------------------------------------------------------------
# Run inference.
# ---------------------------------------------------------------------------
predictions_log10 = np.asarray(
    model.predict_fastchem(
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        global_inputs=global_inputs,
        return_log10=True,
    )
)

idx = int(np.argmin(np.abs(pressure_bar - 0.1)))
print(f"Output shape: {predictions_log10.shape}  |  species: {', '.join(model.species)}")
print("\nMixing ratios at P ≈ 0.1 bar (log10):")
for name in ["H2", "H2O", "CO", "CO2", "CH4", "NH3", "H2S"]:
    if name in model.species:
        col = model.species.index(name)
        print(f"  {name:>5s} = {predictions_log10[idx, col]:+.3f}")
