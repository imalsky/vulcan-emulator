#!/usr/bin/env python3
"""Print the configured model training log as an aligned table."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}.")
    return payload


def _format_cell(value: str) -> str:
    """Format one CSV cell for terminal display."""
    try:
        numeric = float(value)
    except ValueError:
        return value
    if value.isdigit():
        return value
    return f"{numeric:.2e}"


def _print_csv(path: Path) -> None:
    """Print one CSV file with aligned columns."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise RuntimeError(f"Empty CSV file: {path}")

    header = rows[0]
    body = [[_format_cell(cell) for cell in row] for row in rows[1:]]
    widths = [len(name) for name in header]
    for row in body:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    print("  ".join(f"{name:<{widths[index]}}" for index, name in enumerate(header)))
    for row in body:
        print("  ".join(f"{cell:>{widths[index]}}" for index, cell in enumerate(row)))


def main() -> int:
    """Read and print the config-selected training log."""
    config = _load_json(ROOT / "config" / "config.json")
    run_dir = (
        ROOT
        / str(config["paths"].get("project_root", "."))
        / str(config["paths"]["models_root"])
        / str(config["training"]["output_folder"])
    ).resolve()
    log_path = run_dir / "training_log.csv"
    print(f"run: {run_dir}")
    print(f"file: {log_path}\n")
    _print_csv(log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
