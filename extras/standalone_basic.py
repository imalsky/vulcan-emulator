"""Minimal standalone FastChem emulator example.

Usage:
    python extras/standalone_basic.py
    python extras/standalone_basic.py --bundle models/fastchem_mlp/best_exported.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
from src.models.export_bundle import load_exported_model  # noqa: E402
from src.utils.config import ELEMENT_INPUT_ORDER, load_and_validate_config  # noqa: E402
from src.utils.helpers import resolve_path, resolve_project_root  # noqa: E402

_DEFAULT_CONFIG = _ROOT / "config" / "fastchem_mlp_config.json"
_SOLAR_ELEMENT_ABUNDANCES = {
    "He_H": 8.38e-2,
    "C_H": 2.95e-4,
    "O_H": 5.37e-4,
    "N_H": 7.08e-5,
    "S_H": 1.41e-5,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the standalone example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="FastChem config used to derive the default export-bundle path.",
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help="Optional explicit path to an exported FastChem bundle.",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=50,
        help="Number of pressure levels in the synthetic profile.",
    )
    return parser.parse_args(argv)


def _resolve_bundle_path(config: dict[str, object], project_root: Path, explicit: str | None) -> Path:
    """Resolve the export-bundle path, preferring the canonical best bundle."""
    if explicit is not None:
        return resolve_path(explicit, project_root)

    checkpoints_root = resolve_path(config["paths"]["checkpoints_root"], project_root)
    default_bundle = checkpoints_root / "best_exported.npz"
    if default_bundle.exists():
        return default_bundle

    candidates = sorted(checkpoints_root.glob("*_exported.npz"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No exported bundle found under {checkpoints_root}. "
            "Pass --bundle explicitly or export a checkpoint first."
        )
    raise FileNotFoundError(
        f"Multiple exported bundles found under {checkpoints_root}: {candidates}. "
        "Pass --bundle explicitly."
    )


def main(argv: list[str] | None = None) -> int:
    """Run the minimal standalone FastChem inference demo."""
    args = _parse_args(argv)
    project_root = resolve_project_root(Path(__file__).resolve())
    config = load_and_validate_config(resolve_path(args.config, project_root))
    bundle_path = _resolve_bundle_path(config, project_root, args.bundle)

    model = load_exported_model(bundle_path)
    if not model.uses_fastchem:
        raise RuntimeError(f"Expected a FastChem export bundle, got {bundle_path}.")

    expected_global_order = list(ELEMENT_INPUT_ORDER)
    global_order = list(model.data_contract["global_static_feature_order"])
    if global_order != expected_global_order:
        raise RuntimeError(
            "extras/standalone_basic.py requires an exported FastChem bundle with "
            f"global inputs {expected_global_order}, but the bundle stores {global_order}."
        )

    pressure_bar = np.logspace(2.0, -5.0, int(args.num_levels))
    temperature_k = 800.0 + 1200.0 * (pressure_bar / 100.0) ** 0.08

    mixing_ratios_log10 = np.asarray(
        model.predict_fastchem_profile(
            pressure_bar=pressure_bar,
            temperature_k=temperature_k,
            global_inputs=dict(_SOLAR_ELEMENT_ABUNDANCES),
            return_log10=True,
        )
    )

    species = list(model.data_contract["output_species_order"])
    phot_idx = int(np.argmin(np.abs(pressure_bar - 0.1)))

    print(f"Bundle path   : {bundle_path}")
    print(f"Output shape  : {mixing_ratios_log10.shape}  (levels × species)")
    print(f"\nMixing ratios at P ≈ {pressure_bar[phot_idx]:.2f} bar:")
    for name, log10_x in zip(species, mixing_ratios_log10[phot_idx]):
        print(f"  {name:>5s}  log10(X) = {float(log10_x):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
