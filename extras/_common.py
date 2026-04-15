"""Shared constants and utilities for the extras scripts.

Centralizes duplicated logic (bundle resolution, saved test-case loading,
VULCAN source lookup, solar abundances, plot style) so individual scripts
stay focused on their own purpose.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Project root and sys.path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation.data_loader import (  # noqa: E402
    ProcessedSplit,
    load_processed_dataset,
)
from src.data_generation.generation import list_run_ids_from_consolidated  # noqa: E402
from src.data_generation.preprocess import (  # noqa: E402
    inverse_block,
    inverse_mixed_block,
)
from src.utils.helpers import resolve_path  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STYLE_PATH = ROOT / "extras" / "science.mplstyle"
DEFAULT_BUNDLE = ROOT / "models" / "fastchem_transformer" / "best_exported.npz"

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
SOLAR_ELEMENT_ABUNDANCES: dict[str, float] = {
    "He_H": 7.84e-2,
    "C_H": 2.69e-4,
    "O_H": 4.90e-4,
    "N_H": 6.76e-5,
    "S_H": 1.32e-5,
}

# Elements whose FastChem abundances are scaled by metallicity.
FASTCHEM_METALLICITY_SCALED_ELEMENTS: set[str] = {
    "C", "N", "O", "S", "P", "Si", "Ti", "V",
    "Cl", "K", "Na", "Mg", "F", "Ca", "Fe",
}

# Small additive floor for log-space residuals to avoid log(0) issues.
EPSILON = 1.0e-30

# Compact labels for the exported global feature names.
GLOBAL_LABELS: dict[str, str] = {
    "He_H": "He/H",
    "C_H": "C/H",
    "O_H": "O/H",
    "N_H": "N/H",
    "S_H": "S/H",
}


# ---------------------------------------------------------------------------
# Extras test-data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FastChemTestContext:
    """Saved processed test split plus optional matching raw-data sidecars."""

    bundle_path: Path
    processed_root: Path
    raw_root: Path | None
    split: ProcessedSplit
    normalization: dict[str, Any]
    contract: dict[str, Any]
    run_id_to_index: dict[str, int]
    raw_run_ids: frozenset[str]


@dataclass(frozen=True)
class FastChemTestCase:
    """One inverse-normalized FastChem test example."""

    run_id: str
    pressure_bar: np.ndarray
    temperature_k: np.ndarray
    global_inputs: dict[str, float]
    stored_target_ymix: np.ndarray
    output_species: list[str]
    raw_globals: dict[str, float] | None
    raw_metadata: dict[str, Any] | None


# ---------------------------------------------------------------------------
# Bundle / path resolution helpers
# ---------------------------------------------------------------------------
def resolve_bundle_path(project_root: Path, explicit: str | None) -> Path:
    """Resolve the exported transformer bundle.

    Parameters
    ----------
    project_root : Path
        Repository root directory.
    explicit : str or None
        CLI-supplied path override; when *None* the default bundle is used.

    Returns
    -------
    Path
        Validated path to the ``.npz`` export bundle.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist on disk.
    """
    bundle_path = (
        resolve_path(explicit, project_root)
        if explicit is not None
        else DEFAULT_BUNDLE
    )
    if not bundle_path.exists():
        raise FileNotFoundError(f"Exported bundle not found: {bundle_path}")
    return bundle_path


def _unique_paths(paths: list[Path]) -> list[Path]:
    """Deduplicate candidate paths while preserving order."""
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _resolve_processed_root(
    project_root: Path,
    *,
    bundle_path: Path,
    config: dict[str, object],
) -> Path:
    """Locate the processed dataset that matches the selected bundle."""
    paths = config.get("paths", {})
    candidates: list[Path] = []
    if isinstance(paths, dict):
        processed_value = paths.get("processed_root")
        if isinstance(processed_value, str) and processed_value:
            candidates.append(resolve_path(processed_value, project_root))
    candidates.append(project_root / "data" / bundle_path.parent.name / "processed")

    for candidate in _unique_paths(candidates):
        if (candidate / "test" / "metadata.json").exists():
            return candidate
    raise FileNotFoundError(
        "No processed test split found in any expected location: "
        + ", ".join(str(path) for path in _unique_paths(candidates))
    )


def _resolve_raw_root(
    project_root: Path,
    *,
    bundle_path: Path,
    config: dict[str, object],
    processed_root: Path,
    require_raw: bool,
) -> tuple[Path | None, frozenset[str]]:
    """Locate the consolidated raw dataset that matches the processed split."""
    paths = config.get("paths", {})
    candidates: list[Path] = []
    if isinstance(paths, dict):
        raw_value = paths.get("raw_root")
        if isinstance(raw_value, str) and raw_value:
            candidates.append(resolve_path(raw_value, project_root))
    candidates.append(processed_root.parent / "raw")
    candidates.append(project_root / "data" / bundle_path.parent.name / "raw")
    candidates.append(project_root / "data" / "raw")

    for candidate in _unique_paths(candidates):
        consolidated_path = candidate / "runs.h5"
        if not consolidated_path.exists():
            continue
        run_ids = frozenset(list_run_ids_from_consolidated(consolidated_path))
        if run_ids:
            return candidate, run_ids

    if require_raw:
        raise FileNotFoundError(
            "No consolidated raw dataset found in any expected location: "
            + ", ".join(str(path) for path in _unique_paths(candidates))
        )
    return None, frozenset()


def load_fastchem_test_context(
    project_root: Path,
    *,
    bundle_path: Path,
    config: dict[str, object],
    require_raw: bool = False,
) -> FastChemTestContext:
    """Load the saved processed FastChem test split and optional raw sidecars."""
    processed_root = _resolve_processed_root(
        project_root, bundle_path=bundle_path, config=config,
    )
    splits, normalization, contract = load_processed_dataset(processed_root)
    if "test" not in splits:
        raise FileNotFoundError(f"Processed dataset is missing the test split: {processed_root}")
    split = splits["test"]
    if str(contract.get("chemistry_type", "")).lower() != "fastchem":
        raise RuntimeError(f"Expected a FastChem processed dataset under {processed_root}.")
    raw_root, raw_run_ids = _resolve_raw_root(
        project_root,
        bundle_path=bundle_path,
        config=config,
        processed_root=processed_root,
        require_raw=require_raw,
    )
    return FastChemTestContext(
        bundle_path=bundle_path,
        processed_root=processed_root,
        raw_root=raw_root,
        split=split,
        normalization=normalization,
        contract=contract,
        run_id_to_index={run_id: idx for idx, run_id in enumerate(split.run_ids)},
        raw_run_ids=raw_run_ids,
    )


def select_fastchem_test_run_id(
    context: FastChemTestContext,
    *,
    run_id: str | None,
    require_raw: bool = False,
    rng: np.random.Generator | None = None,
) -> str:
    """Choose a saved test run ID, optionally requiring raw-sidecar coverage."""
    if run_id is not None:
        if run_id not in context.run_id_to_index:
            raise KeyError(f"Requested run ID {run_id!r} is not present in the saved test split.")
        if require_raw and run_id not in context.raw_run_ids:
            raise KeyError(f"Requested run ID {run_id!r} is not present in the raw dataset.")
        return run_id

    candidates = list(context.split.run_ids)
    if require_raw:
        if context.raw_root is None:
            raise FileNotFoundError("This extras workflow requires the matching raw dataset.")
        candidates = [candidate for candidate in candidates if candidate in context.raw_run_ids]
    if not candidates:
        raise FileNotFoundError("No saved test runs are available for the requested selection mode.")

    generator = np.random.default_rng() if rng is None else rng
    return str(candidates[int(generator.integers(0, len(candidates)))])


def _decode_hdf5_scalar(value: Any) -> Any:
    """Convert one HDF5 scalar payload into plain Python types."""
    scalar = np.asarray(value)[()]
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    if isinstance(scalar, np.bool_):
        return bool(scalar)
    if isinstance(scalar, np.integer):
        return int(scalar)
    if isinstance(scalar, np.floating):
        return float(scalar)
    return scalar


def _decode_hdf5_labels(values: np.ndarray) -> list[str]:
    """Decode an HDF5 string array into Python strings."""
    return [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in values
    ]


def _load_raw_sidecars(
    raw_root: Path,
    *,
    run_id: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Load raw globals and scalar metadata for one stored run."""
    consolidated_path = raw_root / "runs.h5"
    with h5py.File(consolidated_path, "r") as handle:
        if run_id not in handle:
            raise KeyError(f"Run ID {run_id!r} not found in {consolidated_path}")
        group = handle[run_id]
        globals_map = {
            key: float(np.asarray(group[f"globals/{key}"]))
            for key in group["globals"].keys()
        }
        if "inputs/element_input_order" in group and "inputs/elemental_abundances_frac" in group:
            element_labels = _decode_hdf5_labels(np.asarray(group["inputs/element_input_order"]))
            element_profile = np.asarray(group["inputs/elemental_abundances_frac"], dtype=np.float64)
            for index, label in enumerate(element_labels):
                globals_map[label] = float(element_profile[0, index])
        metadata: dict[str, Any] = {}
        if "metadata" in group:
            metadata = {
                key: _decode_hdf5_scalar(group[f"metadata/{key}"][()])
                for key in group["metadata"].keys()
            }
    return globals_map, metadata


def load_fastchem_raw_metadata_map(
    raw_root: Path,
    run_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Load raw scalar metadata for a collection of saved test runs."""
    consolidated_path = raw_root / "runs.h5"
    metadata_map: dict[str, dict[str, Any]] = {}
    with h5py.File(consolidated_path, "r") as handle:
        for run_id in run_ids:
            if run_id not in handle:
                raise KeyError(f"Run ID {run_id!r} not found in {consolidated_path}")
            group = handle[run_id]
            if "metadata" not in group:
                metadata_map[run_id] = {}
                continue
            metadata_map[run_id] = {
                key: _decode_hdf5_scalar(group[f"metadata/{key}"][()])
                for key in group["metadata"].keys()
            }
    return metadata_map


def classify_temperature_profile_bucket(raw_metadata: dict[str, Any]) -> str:
    """Classify one stored run into the plot-profile display buckets."""
    source = raw_metadata.get("temperature_profile_source")
    if source == "pt_library":
        return "pt_library"
    if source == "analytic":
        key = "temperature_profile_analytic_convective_adjustment_applied"
        if key not in raw_metadata:
            raise KeyError(
                "Analytic raw metadata is missing "
                "'temperature_profile_analytic_convective_adjustment_applied'."
            )
        return "analytic_convective" if bool(raw_metadata[key]) else "analytic_radiative"
    raise KeyError("Raw metadata is missing a supported temperature_profile_source value.")


def load_fastchem_test_case(
    context: FastChemTestContext,
    *,
    run_id: str,
) -> FastChemTestCase:
    """Inverse-normalize one saved FastChem test example back to physical units."""
    if run_id not in context.run_id_to_index:
        raise KeyError(f"Run ID {run_id!r} is not present in the saved test split.")

    index = context.run_id_to_index[run_id]
    sequence_inputs = np.asarray(context.split.sequence_inputs[index], dtype=np.float64)
    global_inputs = np.asarray(context.split.global_inputs[index : index + 1], dtype=np.float64)
    target_outputs = np.asarray(context.split.target_outputs[index], dtype=np.float64)

    pressure_bar = inverse_block(
        sequence_inputs[:, 0:1],
        context.normalization["sequence_static"]["blocks"][0],
    )[:, 0]
    temperature_k = inverse_block(
        sequence_inputs[:, 1:2],
        context.normalization["sequence_static"]["blocks"][1],
    )[:, 0]
    global_vector = inverse_mixed_block(
        global_inputs,
        context.normalization["global_static"],
    )[0]
    global_order = list(context.contract["global_static_feature_order"])
    global_map = {
        name: float(global_vector[idx])
        for idx, name in enumerate(global_order)
    }
    stored_target_ymix = inverse_block(
        target_outputs,
        context.normalization["target"],
    )

    raw_globals: dict[str, float] | None = None
    raw_metadata: dict[str, Any] | None = None
    if context.raw_root is not None and run_id in context.raw_run_ids:
        raw_globals, raw_metadata = _load_raw_sidecars(context.raw_root, run_id=run_id)

    return FastChemTestCase(
        run_id=run_id,
        pressure_bar=np.asarray(pressure_bar, dtype=np.float64),
        temperature_k=np.asarray(temperature_k, dtype=np.float64),
        global_inputs=global_map,
        stored_target_ymix=np.asarray(stored_target_ymix, dtype=np.float64),
        output_species=list(context.contract["output_species_order"]),
        raw_globals=raw_globals,
        raw_metadata=raw_metadata,
    )


def resolve_vulcan_source_root(
    project_root: Path,
    *,
    config: dict[str, object],
    explicit_root: str | None,
) -> Path:
    """Resolve the VULCAN-master source checkout.

    Looks first at an explicit CLI override, then falls back to the
    ``paths.vulcan_source_root`` entry inside the exported bundle config.

    Parameters
    ----------
    project_root : Path
        Repository root directory.
    config : dict
        Exported bundle configuration dictionary.
    explicit_root : str or None
        CLI-supplied override path.

    Returns
    -------
    Path
        Resolved path to the VULCAN source tree.

    Raises
    ------
    KeyError
        If no source root can be determined.
    """
    if explicit_root is not None:
        return resolve_path(explicit_root, project_root)

    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        raise KeyError("Exported bundle config does not include a paths section.")
    vulcan_source_root = paths.get("vulcan_source_root")
    if not isinstance(vulcan_source_root, str) or not vulcan_source_root:
        raise KeyError("Exported bundle config is missing paths.vulcan_source_root.")
    return resolve_path(vulcan_source_root, project_root)


def plots_dir_for_bundle(bundle_path: Path) -> Path:
    """Return (and create) the ``plots/`` directory next to the bundle.

    All extras scripts save figures here so output lives with the model,
    not inside the extras folder.

    Parameters
    ----------
    bundle_path : Path
        Path to the exported ``.npz`` bundle file.

    Returns
    -------
    Path
        The ``plots/`` directory adjacent to the bundle.
    """
    plots_dir = bundle_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def global_label(name: str) -> str:
    """Map an exported global feature name to a compact display label."""
    return GLOBAL_LABELS.get(name, name)


def apply_style() -> None:
    """Load the shared matplotlib style if available."""
    if STYLE_PATH.exists():
        import matplotlib.pyplot as plt

        plt.style.use(str(STYLE_PATH))
