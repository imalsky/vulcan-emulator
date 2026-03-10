#!/usr/bin/env python3
"""Unit tests for raw-run reuse and preprocessing contracts."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocess import PreprocessError, _build_trajectory_specs, discover_existing_raw_run_files, load_raw_run_file


def _write_raw_run(
    path: Path,
    *,
    run_id: int,
    source_kind: str = "analytic",
    state_species: list[str],
    output_species: list[str],
    time_s: np.ndarray,
    ymix_state: np.ndarray,
    ymix_output: np.ndarray,
    pressure: np.ndarray | None = None,
    kzz: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    nz = int(ymix_state.shape[1])
    pressure_values = (
        np.asarray(pressure, dtype=np.float64)
        if pressure is not None
        else np.logspace(1.0, -1.0, nz, dtype=np.float64)
    )
    temperature = np.full((nz,), 1000.0, dtype=np.float64)
    kzz_values = (
        np.asarray(kzz, dtype=np.float64)
        if kzz is not None
        else np.full((nz,), 1.0e9, dtype=np.float64)
    )

    with h5py.File(path, "w") as handle:
        handle.attrs["run_id"] = int(run_id)
        inputs = handle.create_group("inputs")
        inputs.create_dataset("pressure_bar", data=pressure_values)
        inputs.create_dataset("temperature_k", data=temperature)
        inputs.create_dataset("kzz_cm2_s", data=kzz_values)
        inputs.create_dataset("state_species", data=np.asarray(state_species, dtype=str_dtype))
        inputs.create_dataset("output_species", data=np.asarray(output_species, dtype=str_dtype))

        globals_group = handle.create_group("globals")
        globals_group.create_dataset("gravity_cm_s2", data=np.float64(1.0e3))
        globals_group.create_dataset("metallicity_log10", data=np.float64(0.0))
        globals_group.create_dataset("c_to_o", data=np.float64(0.55))

        trajectory = handle.create_group("trajectory")
        trajectory.create_dataset("time_s", data=np.asarray(time_s, dtype=np.float64))
        trajectory.create_dataset("ymix_state", data=np.asarray(ymix_state, dtype=np.float64))
        trajectory.create_dataset("ymix_output", data=np.asarray(ymix_output, dtype=np.float64))
        sampler = handle.create_group("sampler")
        sampler.create_dataset("source_kind", data=np.asarray(source_kind, dtype=str_dtype))


class PreprocessContractsTests(unittest.TestCase):
    """Unit tests for raw run discovery and pair sampling reuse behavior."""

    def test_load_raw_run_file_accepts_species_subsets(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_preprocess_subset_") as tmpdir_name:
            run_path = Path(tmpdir_name) / "run_000123.h5"
            state_species = ["H2", "He", "H2O", "CO"]
            time_s = np.array([0.0, 10.0, 20.0], dtype=np.float64)
            ymix_state = np.arange(3 * 2 * 4, dtype=np.float64).reshape(3, 2, 4) + 1.0
            ymix_output = ymix_state.copy()
            _write_raw_run(
                run_path,
                run_id=123,
                state_species=state_species,
                output_species=state_species,
                time_s=time_s,
                ymix_state=ymix_state,
                ymix_output=ymix_output,
            )

            raw = load_raw_run_file(
                run_path,
                state_species=["H2", "H2O"],
                output_species=["CO", "H2"],
            )

        self.assertEqual(raw.run_id, 123)
        self.assertEqual(raw.ymix_state.shape, (3, 2, 2))
        self.assertEqual(raw.ymix_output.shape, (3, 2, 2))
        np.testing.assert_allclose(raw.ymix_state, ymix_state[:, :, [0, 2]])
        np.testing.assert_allclose(raw.ymix_output, ymix_output[:, :, [3, 0]])

    def test_discover_existing_raw_run_files_only_uses_flat_raw_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_preprocess_discover_") as tmpdir_name:
            raw_root = Path(tmpdir_name) / "raw"
            direct_run = raw_root / "run_000001.h5"
            nested_legacy_run = raw_root / "runs" / "run_000002.h5"
            direct_run.parent.mkdir(parents=True, exist_ok=True)
            nested_legacy_run.parent.mkdir(parents=True, exist_ok=True)
            direct_run.write_bytes(b"")
            nested_legacy_run.write_bytes(b"")

            discovered = discover_existing_raw_run_files(raw_root)

        self.assertEqual(discovered, [direct_run])

    def test_load_raw_run_file_rejects_nonpositive_pressure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_preprocess_pressure_") as tmpdir_name:
            run_path = Path(tmpdir_name) / "run_000123.h5"
            species = ["H2", "He"]
            ymix = np.full((3, 2, 2), 0.5, dtype=np.float64)
            _write_raw_run(
                run_path,
                run_id=123,
                state_species=species,
                output_species=species,
                time_s=np.array([0.0, 10.0, 20.0], dtype=np.float64),
                ymix_state=ymix,
                ymix_output=ymix,
                pressure=np.array([1.0, 0.0], dtype=np.float64),
            )

            with self.assertRaisesRegex(PreprocessError, "pressure_bar must be strictly positive"):
                load_raw_run_file(
                    run_path,
                    state_species=species,
                    output_species=species,
                )

    def test_load_raw_run_file_rejects_nonpositive_kzz(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_preprocess_kzz_") as tmpdir_name:
            run_path = Path(tmpdir_name) / "run_000123.h5"
            species = ["H2", "He"]
            ymix = np.full((3, 2, 2), 0.5, dtype=np.float64)
            _write_raw_run(
                run_path,
                run_id=123,
                state_species=species,
                output_species=species,
                time_s=np.array([0.0, 10.0, 20.0], dtype=np.float64),
                ymix_state=ymix,
                ymix_output=ymix,
                kzz=np.array([1.0e9, -1.0], dtype=np.float64),
            )

            with self.assertRaisesRegex(PreprocessError, "kzz_cm2_s must be strictly positive"):
                load_raw_run_file(
                    run_path,
                    state_species=species,
                    output_species=species,
                )

    def test_load_raw_run_file_requires_source_kind(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_preprocess_source_kind_") as tmpdir_name:
            run_path = Path(tmpdir_name) / "run_000123.h5"
            species = ["H2", "He"]
            ymix = np.full((3, 2, 2), 0.5, dtype=np.float64)
            _write_raw_run(
                run_path,
                run_id=123,
                state_species=species,
                output_species=species,
                time_s=np.array([0.0, 10.0, 20.0], dtype=np.float64),
                ymix_state=ymix,
                ymix_output=ymix,
            )
            with h5py.File(run_path, "a") as handle:
                del handle["sampler"]

            with self.assertRaisesRegex(PreprocessError, "sampler/source_kind is required"):
                load_raw_run_file(
                    run_path,
                    state_species=species,
                    output_species=species,
                )

    def test_build_trajectory_specs_skips_runs_without_valid_log_dt_pairs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_preprocess_pairs_") as tmpdir_name:
            tmpdir = Path(tmpdir_name)
            species = ["H2", "He", "H2O"]
            valid_run_path = tmpdir / "run_000001.h5"
            invalid_run_path = tmpdir / "run_000002.h5"
            ymix_state = np.full((3, 2, 3), 1.0 / 3.0, dtype=np.float64)

            _write_raw_run(
                valid_run_path,
                run_id=1,
                state_species=species,
                output_species=species,
                time_s=np.array([0.0, 10.0, 20.0], dtype=np.float64),
                ymix_state=ymix_state,
                ymix_output=ymix_state,
            )
            _write_raw_run(
                invalid_run_path,
                run_id=2,
                state_species=species,
                output_species=species,
                time_s=np.array([0.0, 1.0, 2.0], dtype=np.float64),
                ymix_state=ymix_state,
                ymix_output=ymix_state,
            )

            config: dict[str, Any] = {
                "trajectory_sampling": {
                    "dt_min_s": 10.0,
                    "dt_max_s": 20.0,
                    "min_future_saved_steps": 1,
                },
            }

            bundles = _build_trajectory_specs(
                run_files=[invalid_run_path, valid_run_path],
                config=config,
                state_species=species,
                output_species=species,
            )

        self.assertEqual(len(bundles), 1)
        self.assertEqual(bundles[0].run_id, 1)
        self.assertEqual(bundles[0].run_file, valid_run_path)


if __name__ == "__main__":
    unittest.main()
