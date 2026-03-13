from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def manifest_for_files(paths: list[Path]) -> list[dict[str, Any]]:
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
    return sha256_text(stable_json_dumps(payload))
