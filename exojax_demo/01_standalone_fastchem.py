#!/usr/bin/env python3
"""Minimal example: load an exported FastChem emulator and predict mixing ratios.

This is the simplest possible usage.  No ExoJAX dependency, no plotting,
no CLI arguments -- just load, predict, and print.

Usage
-----
    python 01_standalone_fastchem.py
"""

from __future__ import annotations

import numpy as np

from vulcan_emulator import BUNDLE_PATH, SOLAR_ABUNDANCES, load_model

NUM_LEVELS = 50
PRESSURE_BOTTOM_BAR = 1.0e2
PRESSURE_TOP_BAR = 1.0e-7
ISOTHERMAL_TEMPERATURE_K = 1500.0
PHOTOSPHERE_PRESSURE_BAR = 0.1
DISPLAY_SPECIES = ("H2", "H2O", "CO", "CO2", "CH4", "NH3", "H2S")

# ===================================================================
# Step 1 -- Load the exported model bundle
# ===================================================================
# The .npz file is fully self-contained: model weights, architecture
# config, and normalization statistics are all inside.

model = load_model(BUNDLE_PATH)
print(f"Loaded: {model.chemistry_type} {model.model_type}  ({len(model.species)} species)")

# ===================================================================
# Step 2 -- Define atmospheric inputs (physical units)
# ===================================================================
# Pressure grid matching the training domain: 50 levels, 100 → 1e-7 bar.
# Native ordering is bottom-to-top (high pressure first).
pressure_bar = np.logspace(
    np.log10(PRESSURE_BOTTOM_BAR),
    np.log10(PRESSURE_TOP_BAR),
    NUM_LEVELS,
)

# Temperature: isothermal at 1500 K for simplicity.
temperature_k = np.full_like(pressure_bar, ISOTHERMAL_TEMPERATURE_K)
global_inputs = dict(SOLAR_ABUNDANCES)

print(f"Inputs: {len(pressure_bar)} levels, {pressure_bar[0]:.0e}–{pressure_bar[-1]:.0e} bar, {temperature_k[0]:.0f} K isothermal, solar abundances")
predictions_log10 = np.asarray(
    model.predict_fastchem(
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        global_inputs=global_inputs,
        return_log10=True,
    )
)

species = model.species
idx = int(np.argmin(np.abs(pressure_bar - PHOTOSPHERE_PRESSURE_BAR)))
print(f"\nMixing ratios at P ≈ {PHOTOSPHERE_PRESSURE_BAR:.1f} bar  (log10, shape={predictions_log10.shape}):")
for name in DISPLAY_SPECIES:
    if name in species:
        col = species.index(name)
        print(f"  {name:>5s} = {predictions_log10[idx, col]:+.3f} dex")
