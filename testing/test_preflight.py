#!/usr/bin/env python3
"""Preflight tests for VULCAN runtime smoke validation."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vulcan_runner import VulcanRuntimeError, preflight_vulcan_source


class PreflightTests(unittest.TestCase):
    """Contract tests for VULCAN runtime preflight validation."""

    def test_missing_vulcan_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_missing_vulcan_") as tmpdir_name:
            missing = Path(tmpdir_name) / "not_here"
            with self.assertRaises(VulcanRuntimeError):
                preflight_vulcan_source(
                    missing,
                    boundary_conditions=None,
                    use_transport=True,
                    use_condensation_optional=True,
                    timeout_seconds=1,
                )

    def test_smoke_timeout_is_rejected(self) -> None:
        real_source = PROJECT_ROOT.parent / "VULCAN-master"
        with mock.patch(
            "vulcan_runner.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["python", "vulcan.py", "-n"], timeout=1),
        ):
            with self.assertRaises(VulcanRuntimeError):
                preflight_vulcan_source(
                    real_source,
                    boundary_conditions=None,
                    use_transport=True,
                    use_condensation_optional=True,
                    timeout_seconds=1,
                )

    def test_smoke_nonzero_exit_is_rejected(self) -> None:
        real_source = PROJECT_ROOT.parent / "VULCAN-master"
        failed = subprocess.CompletedProcess(
            args=["python", "vulcan.py", "-n"],
            returncode=1,
            stdout="stdout",
            stderr="stderr",
        )
        with mock.patch("vulcan_runner.subprocess.run", return_value=failed):
            with self.assertRaises(VulcanRuntimeError):
                preflight_vulcan_source(
                    real_source,
                    boundary_conditions=None,
                    use_transport=True,
                    use_condensation_optional=True,
                    timeout_seconds=1,
                )


if __name__ == "__main__":
    unittest.main()
