"""
standalone_basic.py
===================
The simplest possible example of using the VULCAN equilibrium emulator.

Run from the vulcan-emulator root:

    conda activate nn
    python extras/standalone_basic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make 'src' importable.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
from src.models.export_bundle import load_exported_model

# ---------------------------------------------------------------------------
# 1. Load the model
# ---------------------------------------------------------------------------
# The .npz bundle is entirely self-contained: model weights, architecture,
# normalisation statistics, and the species/feature ordering are all inside.
# Nothing from the training codebase is needed.
model = load_exported_model(_ROOT / "models/equilibrium_only_silu/best_exported.npz")

# ---------------------------------------------------------------------------
# 2. Build a T-P profile
# ---------------------------------------------------------------------------
# 50 pressure levels, log-spaced from 100 bar (deep) to 1e-5 bar (top).
pressure_bar  = np.logspace(2, -5, 50)

# Simple power-law temperature profile: warm at depth, cooler at the top.
temperature_k = 800.0 + 1200.0 * (pressure_bar / 100.0) ** 0.08

# ---------------------------------------------------------------------------
# 3. Call the model
# ---------------------------------------------------------------------------
# predict_equilibrium_profile handles all normalisation internally.
# Inputs are raw physical units; outputs are log10 mixing ratios by species.
mixing_ratios_log10 = model.predict_equilibrium_profile(
    pressure_bar=pressure_bar,
    temperature_k=temperature_k,
    global_inputs={
        "He_H": 8.38e-2,  # solar helium abundance
        "C_H": 2.95e-4,   # solar carbon abundance
        "O_H": 5.37e-4,   # solar oxygen abundance
        "N_H": 7.08e-5,   # solar nitrogen abundance
        "S_H": 1.41e-5,   # solar sulphur abundance
    },
    return_log10=True,  # return log10(mixing ratio); set False for linear
)
# mixing_ratios_log10 is a JAX array with shape (50, 17) — one column per species.

# ---------------------------------------------------------------------------
# 4. Print a quick summary
# ---------------------------------------------------------------------------
species = list(model.data_contract["output_species_order"])
phot_idx = int(np.argmin(np.abs(pressure_bar - 0.1)))   # level nearest 0.1 bar

print(f"Output shape : {mixing_ratios_log10.shape}  (levels × species)")
print(f"\nMixing ratios at P ≈ {pressure_bar[phot_idx]:.2f} bar (photosphere):")
for name, log10_x in zip(species, mixing_ratios_log10[phot_idx]):
    print(f"  {name:>5s}  log10(X) = {float(log10_x):+.3f}")
