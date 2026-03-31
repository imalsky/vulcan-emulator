"""Export a training checkpoint to a portable JAX NPZ bundle.

Usage:
    python extras/export.py
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# -- Configuration -----------------------------------------------------------
CHECKPOINT = _ROOT / "models/equilibrium_only_silu/best.pt"
# ---------------------------------------------------------------------------

sys.path.insert(0, str(_ROOT))

from src.utils.numpy_compat import patch_numpy_asarray_copy

patch_numpy_asarray_copy()

from src.models.export_bundle import export_checkpoint_to_npz

_EXPECTED_EQUILIBRIUM_GLOBAL_ORDER = ["metallicity_log10", "c_to_o", "s_to_o"]


def _validate_checkpoint_contract(checkpoint_path: Path) -> None:
    """Reject equilibrium checkpoints that do not match the current export contract."""
    with checkpoint_path.open("rb") as handle:
        payload = pickle.load(handle)
    contract = payload.get("data_contract", {})
    global_order = list(contract.get("global_static_feature_order", []))
    model_type = str(contract.get("model_type", ""))
    if model_type == "equilibrium" and global_order != _EXPECTED_EQUILIBRIUM_GLOBAL_ORDER:
        raise RuntimeError(
            "extras/export.py requires an equilibrium checkpoint with the current "
            f"ratio-global contract {_EXPECTED_EQUILIBRIUM_GLOBAL_ORDER}, but "
            f"the selected checkpoint stores {global_order}. Regenerate the checkpoint "
            "with the current pipeline before exporting."
        )


def main():
    start = time.perf_counter()
    _validate_checkpoint_contract(CHECKPOINT)
    output_path = export_checkpoint_to_npz(CHECKPOINT)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported JAX bundle to {output_path} ({size_mb:.1f} MB) in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
