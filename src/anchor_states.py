from __future__ import annotations

import numpy as np


def build_flat_h2_he_anchor(
    species_order: list[str],
    *,
    nz: int,
    h2_fraction: float = 0.85,
    he_fraction: float = 0.15,
    floor: float = 1.0e-12,
) -> np.ndarray:
    """Build a flat H2/He-dominated anchor state on the requested species grid."""
    state = np.full((int(nz), len(species_order)), float(floor), dtype=np.float64)
    idx = {name: i for i, name in enumerate(species_order)}
    if "H2" in idx:
        state[:, idx["H2"]] = float(h2_fraction)
    if "He" in idx:
        state[:, idx["He"]] = float(he_fraction)
    state /= np.sum(state, axis=1, keepdims=True)
    return state
