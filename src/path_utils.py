from __future__ import annotations

from pathlib import Path


PROJECT_MARKERS = ("pyproject.toml", "spec.md")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_project_root(start: Path | None = None) -> Path:
    cursor = (start or Path.cwd()).resolve()
    if cursor.is_file():
        cursor = cursor.parent
    for candidate in (cursor, *cursor.parents):
        if all((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    raise FileNotFoundError("Could not resolve project root from the current path.")


def resolve_path(path_like: str | Path, project_root: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()
