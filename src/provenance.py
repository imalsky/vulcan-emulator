"""Artifact provenance helpers for generated and processed VULCAN datasets.

Ensures that processed training data stays in sync with its source config
and raw VULCAN outputs via SHA-256 fingerprinting.  Before training begins,
the fingerprint of the processed artifacts is recomputed and compared to
the stored fingerprint; any mismatch (stale data, modified config, missing
files) raises an error requiring ``--gen`` re-execution.

This prevents subtle bugs from training on processed data that was generated
under a different configuration or from a different set of raw runs.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

PROCESSED_FINGERPRINT_FILENAME = "processed_fingerprint.json"


def _display_path(path: Path, project_root: Path) -> str:
    """Render one path relative to the project root when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(project_root.resolve()))
    except ValueError:
        return str(resolved)


def _sha256_bytes(data: bytes) -> str:
    """Hash one in-memory byte payload with SHA-256."""
    return sha256(data).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    """Hash one file incrementally to avoid loading large artifacts into memory."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_dict(path: Path) -> dict[str, Any]:
    """Load one JSON file and require an object payload."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}, found {type(payload).__name__}.")
    return payload


def _preprocess_relevant_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract only config sections that affect generated processed artifacts."""
    generation = config["generation"]
    relevant_generation = {
        "num_runs": generation["num_runs"],
        "split_ratios": generation["split_ratios"],
        "random_seed": generation["random_seed"],
        "save_evo_frq": generation["save_evo_frq"],
        "max_trajectory_snapshots": generation["max_trajectory_snapshots"],
        "keep_vulcan_outputs_debug": generation["keep_vulcan_outputs_debug"],
        "run_timeout_seconds": generation["run_timeout_seconds"],
    }
    sampling = config["trajectory_sampling"]
    relevant_sampling = {
        "mode": sampling["mode"],
        "dt_min_s": sampling["dt_min_s"],
        "dt_max_s": sampling["dt_max_s"],
        "min_future_saved_steps": sampling["min_future_saved_steps"],
    }
    return {
        "generation": relevant_generation,
        "trajectory_sampling": relevant_sampling,
        "vulcan_runtime": config["vulcan_runtime"],
        "tp_sampler": config["tp_sampler"],
        "gravity_sampler": config["gravity_sampler"],
        "kzz_sampler": config["kzz_sampler"],
        "abundance_sampler": config["abundance_sampler"],
        "physics_toggles": config["physics_toggles"],
        "boundary_conditions": config.get("boundary_conditions"),
        "data_spec": config["data_spec"],
        "normalization": config["normalization"],
    }


def stable_config_sha256(config: dict[str, Any]) -> str:
    """Hash the preprocessing-relevant subset of the configuration deterministically."""
    payload = json.dumps(
        _preprocess_relevant_config(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _sha256_bytes(payload.encode("utf-8"))


def resolve_manifest_run_files(manifest: dict[str, Any], project_root: Path) -> list[Path]:
    """Resolve and validate raw-run paths listed in the dataset manifest."""
    run_files = manifest.get("run_files")
    if not isinstance(run_files, list) or not run_files:
        raise RuntimeError("dataset_manifest.json must contain a non-empty 'run_files' list.")

    resolved_paths: list[Path] = []
    for item in run_files:
        if not isinstance(item, str) or not item:
            raise RuntimeError("dataset_manifest.json contains an invalid run file entry.")
        path = Path(item)
        if not path.is_absolute():
            path = (project_root / path).resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            raise RuntimeError(f"Raw run file listed in dataset_manifest.json is missing: {path}")
        resolved_paths.append(path)
    return resolved_paths


def _file_entry(path: Path, project_root: Path) -> dict[str, Any]:
    """Build one provenance record for a single on-disk artifact."""
    stat = path.stat()
    return {
        "path": _display_path(path, project_root),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def build_processed_fingerprint(
    *,
    config: dict[str, Any],
    project_root: Path,
    raw_run_files: list[Path],
    manifest_path: Path,
    split_path: Path,
    normalization_path: Path,
    summary_path: Path,
    split_metadata_paths: dict[str, Path],
) -> dict[str, Any]:
    """Build the fingerprint that ties processed data back to raw inputs and config."""
    raw_entries = []
    for path in raw_run_files:
        stat = path.stat()
        raw_entries.append(
            {
                "path": _display_path(path, project_root),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )

    metadata_entries = {
        split_name: _file_entry(path, project_root)
        for split_name, path in sorted(split_metadata_paths.items())
    }

    return {
        "version": 2,
        "config_sha256": stable_config_sha256(config),
        "manifest": _file_entry(manifest_path, project_root),
        "splits": _file_entry(split_path, project_root),
        "normalization": _file_entry(normalization_path, project_root),
        "processed_summary": _file_entry(summary_path, project_root),
        "split_metadata": metadata_entries,
        "raw_run_files": raw_entries,
    }


def compare_processed_fingerprints(
    *,
    expected: dict[str, Any],
    existing: dict[str, Any],
) -> tuple[bool, str]:
    """Compare two processed-data fingerprints and explain the first mismatch."""
    try:
        if existing["version"] != expected["version"]:
            return False, "fingerprint version mismatch"
        if existing["config_sha256"] != expected["config_sha256"]:
            return False, "preprocessing-relevant config differs from processed artifacts"

        for section_name in ("manifest", "splits", "normalization", "processed_summary"):
            actual_section = existing[section_name]
            if not isinstance(actual_section, dict):
                return False, f"invalid {section_name} section in processed fingerprint"
            if actual_section != expected[section_name]:
                return False, f"{section_name} artifact differs from processed fingerprint"

        actual_split_metadata = existing["split_metadata"]
        if not isinstance(actual_split_metadata, dict):
            return False, "invalid split_metadata section in processed fingerprint"
        if actual_split_metadata != expected["split_metadata"]:
            return False, "processed split metadata differs from processed fingerprint"

        if existing["raw_run_files"] != expected["raw_run_files"]:
            return False, "raw run file list/size/mtime differs from processed fingerprint"
        return True, ""
    except KeyError as exc:
        return False, f"fingerprint missing required field: {exc}"


def validate_processed_artifacts(
    *,
    config: dict[str, Any],
    paths: Any,
) -> dict[str, Any]:
    """Validate processed artifacts and return the resolved provenance bundle."""
    generation = config["generation"]
    processed_root = paths.processed_root
    manifest_path = paths.data_root / str(generation["manifest_filename"])
    split_path = paths.data_root / str(generation["split_filename"])
    normalization_path = processed_root / "normalization_metadata.json"
    summary_path = processed_root / "processed_summary.json"
    fingerprint_path = processed_root / PROCESSED_FINGERPRINT_FILENAME
    split_metadata_paths = {
        split_name: processed_root / split_name / "metadata.json"
        for split_name in ("train", "val", "test")
    }

    required_paths = [
        manifest_path,
        split_path,
        normalization_path,
        summary_path,
        fingerprint_path,
        *split_metadata_paths.values(),
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Processed artifact validation failed. Missing required files: "
            f"{missing}. Regenerate data with --gen."
        )

    manifest = load_json_dict(manifest_path)
    raw_run_files = resolve_manifest_run_files(manifest, project_root=paths.root)
    existing_fingerprint = load_json_dict(fingerprint_path)
    expected_fingerprint = build_processed_fingerprint(
        config=config,
        project_root=paths.root,
        raw_run_files=raw_run_files,
        manifest_path=manifest_path,
        split_path=split_path,
        normalization_path=normalization_path,
        summary_path=summary_path,
        split_metadata_paths=split_metadata_paths,
    )
    matches, reason = compare_processed_fingerprints(
        expected=expected_fingerprint,
        existing=existing_fingerprint,
    )
    if not matches:
        raise RuntimeError(
            "Processed artifacts do not match the current dataset provenance: "
            f"{reason}. Regenerate data with --gen."
        )

    return {
        "manifest": manifest,
        "expected_fingerprint": expected_fingerprint,
        "fingerprint_path": fingerprint_path,
        "normalization_path": normalization_path,
        "summary_path": summary_path,
        "raw_run_files": raw_run_files,
        "split_path": split_path,
        "split_metadata_paths": split_metadata_paths,
    }
