"""Export a training checkpoint to a portable JAX NPZ bundle.

Usage:
    python extras/export.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# -- Configuration -----------------------------------------------------------
CHECKPOINT = _ROOT / "models/equilibrium_only_silu/best.pt"
# ---------------------------------------------------------------------------

sys.path.insert(0, str(_ROOT))

from src.models.export_bundle import export_checkpoint_to_npz


def main():
    start = time.perf_counter()
    output_path = export_checkpoint_to_npz(CHECKPOINT)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported JAX bundle to {output_path} ({size_mb:.1f} MB) in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
