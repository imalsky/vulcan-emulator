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
    """Serialize a JSON-compatible payload with deterministic formatting.

    Parameters
    ----------
    payload : Any
        JSON-serializable object such as a manifest or metadata dictionary.

    Returns
    -------
    str
        Canonical JSON string with sorted keys and stable separators suitable
        for hashing.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    """Compute the SHA-256 digest of a UTF-8 text payload.

    Parameters
    ----------
    text : str
        Text payload to hash.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file incrementally with SHA-256.

    Parameters
    ----------
    path : Path
        File to hash.
    chunk_size : int, default=1024 * 1024
        Number of bytes to read per iteration while streaming the file.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest of the file contents.
    """
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

    Parameters
    ----------
    paths : list[Path]
        Filesystem paths to include in the manifest.

    Returns
    -------
    list[dict[str, Any]]
        Sorted manifest entries containing the filename, full path, size,
        nanosecond mtime, and SHA-256 hash for each file.

    Notes
    -----
    ``mtime_ns`` is not reproducible across machines or fresh checkouts, so
    callers that need a content-stable fingerprint should strip it before
    passing the manifest to :func:`fingerprint_payload`. The ``sha256`` field
    is the authoritative content identifier.
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
    """Compute a deterministic fingerprint for a JSON-serializable payload.

    Parameters
    ----------
    payload : Any
        JSON-serializable manifest or metadata payload.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest of the canonical JSON serialization.
    """
    return sha256_text(stable_json_dumps(payload))
