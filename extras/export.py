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
CHECKPOINT = _ROOT / "models" / "eq" / "best.pt"
# ---------------------------------------------------------------------------

sys.path.insert(0, str(_ROOT))

from src.utils.numpy_compat import patch_numpy_asarray_copy

patch_numpy_asarray_copy()

from src.models.export_bundle import export_checkpoint_to_npz
from src.utils.config import DEFAULT_REQUIRED_GLOBAL_INPUTS, ELEMENT_INPUT_ORDER

_EXPECTED_EQUILIBRIUM_GLOBAL_ORDER = list(ELEMENT_INPUT_ORDER)
_EXPECTED_FULL_VULCAN_GLOBAL_ORDER = list(DEFAULT_REQUIRED_GLOBAL_INPUTS)


def _validate_checkpoint_contract(checkpoint_path: Path) -> None:
    """Reject checkpoints that do not match the current exported input contracts.

    Parameters
    ----------
    checkpoint_path : Path
        Source checkpoint that will be exported.

    Returns
    -------
    None
        The function returns silently when the checkpoint contract matches the
        current exported global-input ordering.
    """
    with checkpoint_path.open("rb") as handle:
        payload = pickle.load(handle)
    contract = payload.get("data_contract", {})
    global_order = list(contract.get("global_static_feature_order", []))
    chemistry_type = str(contract.get("chemistry_type", payload.get("config", {}).get("chemistry_type", "")))
    if chemistry_type == "fastchem" and global_order != _EXPECTED_EQUILIBRIUM_GLOBAL_ORDER:
        raise RuntimeError(
            "extras/export.py requires a FastChem checkpoint with the current "
            f"`X/H` contract {_EXPECTED_EQUILIBRIUM_GLOBAL_ORDER}, but "
            f"the selected checkpoint stores {global_order}. Regenerate the checkpoint "
            "with the current pipeline before exporting."
        )
    if chemistry_type == "vulcan" and global_order != _EXPECTED_FULL_VULCAN_GLOBAL_ORDER:
        raise RuntimeError(
            "extras/export.py requires a VULCAN checkpoint with the current "
            f"runtime-conditioning contract {_EXPECTED_FULL_VULCAN_GLOBAL_ORDER}, but "
            f"the selected checkpoint stores {global_order}. Regenerate the checkpoint "
            "with the current pipeline before exporting."
        )


def main():
    """Validate the configured checkpoint and export it to an NPZ bundle.

    Returns
    -------
    None
        Export status and output-path information are printed to stdout.
    """
    start = time.perf_counter()
    _validate_checkpoint_contract(CHECKPOINT)
    output_path = export_checkpoint_to_npz(CHECKPOINT)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported JAX bundle to {output_path} ({size_mb:.1f} MB) in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
