from __future__ import annotations

from pathlib import Path
from typing import Any

from .inference import load_physical_space_model


def load_exojax_transition(export_root: str | Path) -> dict[str, Any]:
    model = load_physical_space_model(export_root)
    return {
        "transition_fn": model.transition_fn,
        "global_static_feature_order": list(model.contract["global_static_feature_order"]),
        "global_feature_order": list(model.contract["global_feature_order"]),
        "state_species_order": list(model.contract["state_species_order"]),
        "spectrum_wavelength_nm": list(model.contract["spectrum_wavelength_nm"]),
    }
