#!/usr/bin/env python3
"""Compare the emulator against stored ground-truth test profiles.

For each bundled test case: loads the stored P-T profile and true FastChem
mixing ratios, runs the emulator on the same inputs, and saves a two-panel
figure (P-T profile + stored vs emulator mixing ratios).

Usage
-----
    python 03_compare_test_profiles.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
PLOTS_DIR = DEMO_DIR / "plots"
STYLE_PATH = DEMO_DIR / "science.mplstyle"
TEST_DATA_PATH = DEMO_DIR / "bundle" / "test_profiles.npz"

from vulcan_emulator import BUNDLE_PATH, load_model

VMR_FLOOR = 1.0e-30
TEMPERATURE_PLOT_MIN_K = 0.0
TEMPERATURE_PLOT_MAX_K = 3000.0
MIXING_RATIO_PLOT_MIN = 1.0e-20
MIXING_RATIO_PLOT_MAX = 3.0
FIGURE_SIZE = (12, 6)
PT_LINE_WIDTH = 2.0
TARGET_PROFILE_LINE_WIDTH = 1.6
EMULATOR_PROFILE_LINE_WIDTH = 1.2
LEGEND_FONT_SIZE = 8
COMPARISON_LEGEND_FONT_SIZE = 14
PRESSURE_LOG_TICKS = 8
PLOT_DPI = 160

plt.style.use(str(STYLE_PATH))

model = load_model(BUNDLE_PATH)
test_data = np.load(TEST_DATA_PATH, allow_pickle=False)

species = [s.item() if hasattr(s, "item") else str(s) for s in test_data["species"]]
global_keys = [k.item() if hasattr(k, "item") else str(k) for k in test_data["global_keys"]]
num_cases = int(test_data["num_cases"].item())

PLOTS_DIR.mkdir(exist_ok=True)

colors = plt.cm.tab20(np.linspace(0, 1, len(species)))

for i in range(num_cases):
    prefix = f"case{i}"
    run_id = str(test_data[f"{prefix}/run_id"].item())
    pressure_bar = test_data[f"{prefix}/pressure_bar"]
    temperature_k = test_data[f"{prefix}/temperature_k"]
    target_ymix = test_data[f"{prefix}/target_ymix"]
    global_array = test_data[f"{prefix}/global_inputs"]
    global_inputs = {key: float(global_array[j]) for j, key in enumerate(global_keys)}

    pred_ymix = np.asarray(model.predict_fastchem(
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        global_inputs=global_inputs,
        return_log10=False,
    ))

    fig, (ax_pt, ax_mix) = plt.subplots(1, 2, figsize=FIGURE_SIZE, sharey=True)

    ax_pt.plot(temperature_k, pressure_bar, color="black", lw=PT_LINE_WIDTH)
    ax_pt.set_xlabel("Temperature [K]")
    ax_pt.set_ylabel("Pressure [bar]")
    ax_pt.set_yscale("log")
    ax_pt.invert_yaxis()
    ax_pt.set_xlim(TEMPERATURE_PLOT_MIN_K, TEMPERATURE_PLOT_MAX_K)
    ax_pt.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=PRESSURE_LOG_TICKS))
    ax_pt.yaxis.set_minor_locator(mticker.NullLocator())

    for j, name in enumerate(species):
        color = colors[j]
        ax_mix.plot(
            np.clip(target_ymix[:, j], VMR_FLOOR, None),
            pressure_bar,
            color=color, lw=TARGET_PROFILE_LINE_WIDTH, label=name,
        )
        ax_mix.plot(
            np.clip(pred_ymix[:, j], VMR_FLOOR, None),
            pressure_bar,
            color=color, lw=EMULATOR_PROFILE_LINE_WIDTH, ls="--",
        )

    ax_mix.set_xscale("log")
    ax_mix.set_xlim(MIXING_RATIO_PLOT_MIN, MIXING_RATIO_PLOT_MAX)
    ax_mix.set_xlabel("Mixing Ratio")

    species_legend = ax_mix.legend(fontsize=LEGEND_FONT_SIZE, ncol=3, loc="best")
    ax_mix.add_artist(species_legend)
    line_handles = [
        Line2D([0], [0], color="black", lw=TARGET_PROFILE_LINE_WIDTH, ls="-", label="Test Profile"),
        Line2D([0], [0], color="black", lw=EMULATOR_PROFILE_LINE_WIDTH, ls="--", label="Emulator Profile"),
    ]
    ax_mix.legend(handles=line_handles, fontsize=COMPARISON_LEGEND_FONT_SIZE, loc="upper right")

    fig.tight_layout()

    out_path = PLOTS_DIR / f"{run_id}_compare.png"
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
