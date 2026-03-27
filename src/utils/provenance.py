"""SHA-256 hashing and file manifest generation for data provenance.

Used by the generation and preprocessing stages to record which raw
files were consumed and what normalization was applied, so that a
trained model can always be traced back to its exact training data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_json_dumps(payload: Any) -> str:
    """Serialize JSON deterministically for hashing and manifests."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    """Hash a UTF-8 text payload with SHA-256."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file incrementally with SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def manifest_for_files(paths: list[Path]) -> list[dict[str, Any]]:
    """Build a deterministic manifest for a list of filesystem paths.

    Each entry records the filename, full path, byte size, mtime, and
    SHA-256 hash — enough to detect any change to the source data.
    """
    manifest: list[dict[str, Any]] = []
    for path in sorted(paths):
        stat = path.stat()
        manifest.append(
            {
                "name": path.name,
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(path),
            }
        )
    return manifest


def fingerprint_payload(payload: Any) -> str:
    """Hash a JSON-serializable payload using the stable serializer."""
    return sha256_text(stable_json_dumps(payload))
