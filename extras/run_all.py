#!/usr/bin/env python3
"""Run every extras script in sequence and report results.

A convenience wrapper that executes each standalone demo / diagnostic
script from the ``extras/`` directory, printing a clear pass/fail
summary at the end.  Any CLI arguments are forwarded to every script
(e.g. ``--bundle``).

Usage
-----
    python extras/run_all.py
    python extras/run_all.py --bundle models/fastchem_transformer/best_exported.npz
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# Scripts to run, in order.  Each entry is (display name, module path).
_EXTRAS_DIR = Path(__file__).resolve().parent
_SCRIPTS: list[tuple[str, Path]] = [
    ("stand_alone_example", _EXTRAS_DIR / "stand_alone_example.py"),
    ("fastchem_saved_test_profile_demo", _EXTRAS_DIR / "fastchem_saved_test_profile_demo.py"),
    ("plot_saved_test_profiles", _EXTRAS_DIR / "plot_saved_test_profiles.py"),
    ("compare_saved_test_profile_fastchem", _EXTRAS_DIR / "compare_saved_test_profile_fastchem.py"),
]


def _run_script(
    name: str,
    script: Path,
    extra_args: list[str],
) -> tuple[str, bool, float, str]:
    """Run one extras script in a subprocess.

    Returns
    -------
    tuple[str, bool, float, str]
        (name, success, elapsed_seconds, captured_output)
    """
    cmd = [sys.executable, str(script)] + extra_args
    print(f"\n{'=' * 72}")
    print(f"  Running: {name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 72}\n")

    t0 = time.monotonic()
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.monotonic() - t0

    # Stream the output so the user sees progress.
    if result.stdout:
        print(result.stdout, end="")

    success = result.returncode == 0
    tag = "PASS" if success else f"FAIL (exit {result.returncode})"
    print(f"\n  [{tag}] {name}  ({elapsed:.1f}s)")
    return name, success, elapsed, result.stdout


def main() -> int:
    """Run all extras scripts and print a summary."""
    # Forward all CLI args (e.g. --bundle) to every script.
    extra_args = sys.argv[1:]

    results: list[tuple[str, bool, float, str]] = []
    for name, script in _SCRIPTS:
        if not script.exists():
            print(f"  SKIP — {script.name} not found")
            continue
        results.append(_run_script(name, script, extra_args))

    # --- Summary ---
    print(f"\n{'=' * 72}")
    print("  SUMMARY")
    print(f"{'=' * 72}")
    total_time = 0.0
    failures = 0
    for name, success, elapsed, _ in results:
        status = "PASS" if success else "FAIL"
        print(f"    [{status}]  {name:30s}  {elapsed:6.1f}s")
        total_time += elapsed
        if not success:
            failures += 1

    print(f"\n  {len(results)} scripts, {failures} failed, {total_time:.1f}s total")
    print(f"{'=' * 72}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
