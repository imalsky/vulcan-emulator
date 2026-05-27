from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import h5py
import numpy as np
import pytest
from src.data_generation import exogibbs_backend, generation
from src.data_generation.sampling import RunSpecification
from src.utils.config import load_and_validate_config

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "uni_tests" / "fixtures"


def _load_fastchem_fixture_config(tmp_path: Path) -> dict:
    payload = json.loads(
        (FIXTURE_ROOT / "fastchem_transformer_config.json").read_text(encoding="utf-8")
    )
    payload["paths"]["run_root"] = str(tmp_path / "fastchem")
    config_path = tmp_path / "fastchem_config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return load_and_validate_config(config_path)


def _minimal_fastchem_spec() -> RunSpecification:
    return RunSpecification(
        run_id="run_00000",
        pressure_bar=np.array([1.0, 0.1], dtype=np.float64),
        temperature_k=np.array([1000.0, 900.0], dtype=np.float64),
        globals={
            "He_H": 0.1,
            "C_H": 1.0e-4,
            "O_H": 2.0e-4,
            "N_H": 5.0e-5,
            "S_H": 1.0e-5,
        },
        metadata={},
        gravity_cm_s2=np.ones(2, dtype=np.float64),
    )


def _minimal_vulcan_spec(run_idx: int) -> RunSpecification:
    pressure = np.array([1.0, 0.1], dtype=np.float64)
    return RunSpecification(
        run_id=f"run_{run_idx:06d}",
        pressure_bar=pressure,
        temperature_k=np.array([1000.0, 900.0], dtype=np.float64),
        globals={
            "He_H": 0.1,
            "C_H": 1.0e-4,
            "O_H": 2.0e-4,
            "N_H": 5.0e-5,
            "S_H": 1.0e-5,
            "gravity_cm_s2": 1.0e3,
            "planet_radius_cm": 7.0e9,
        },
        metadata={},
        kzz_cm2_s=np.full(pressure.shape, 1.0e6, dtype=np.float64),
        gravity_cm_s2=np.full(pressure.shape, 1.0e3, dtype=np.float64),
    )


def test_fastchem_monitor_fail_mask_parser(tmp_path):
    monitor = tmp_path / "monitor_output.dat"
    monitor.write_text(
        "\n".join(
            [
                "grid c_iter c_conv P T n_tot n_g m H C",
                "0 1 ok 1.0 1000.0 1.0 1.0 1.0 ok ok",
                "1 1 ok 0.1 900.0 1.0 1.0 1.0 ok fail",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fail_mask = generation.read_fastchem_monitor_fail_mask(monitor)

    np.testing.assert_array_equal(
        fail_mask,
        np.array([False, True], dtype=bool),
    )


def test_fastchem_generation_uses_configured_timeout_and_rejects_monitor_failures(
    tmp_path,
    monkeypatch,
):
    config = _load_fastchem_fixture_config(tmp_path)
    config["generation"]["fastchem_timeout_seconds"] = 12.5
    fastchem_root = tmp_path / "worker" / "fastchem_vulcan"
    observed: dict[str, float] = {}

    def fake_ensure(source_root: Path, worker_root: Path) -> Path:
        del source_root, worker_root
        (fastchem_root / "output").mkdir(parents=True, exist_ok=True)
        return fastchem_root

    def fake_chemistry_subprocess(cmd, *, cwd, run_id, timeout, env=None, label="Chemistry"):
        observed["timeout"] = float(timeout)
        output_dir = Path(cwd) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "monitor_output.dat").write_text(
            "\n".join(
                [
                    "grid c_iter c_conv P T n_tot n_g m H C",
                    "0 1 ok 1.0 1000.0 1.0 1.0 1.0 ok fail",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(generation, "_ensure_fastchem_worker_tree", fake_ensure)
    monkeypatch.setattr(generation, "_reset_fastchem_worker_between_runs", lambda _: None)
    monkeypatch.setattr(generation, "_write_fastchem_element_abundances", lambda *a, **k: None)
    monkeypatch.setattr(generation, "_write_fastchem_tp_profile", lambda *a, **k: None)
    monkeypatch.setattr(generation, "_run_chemistry_subprocess", fake_chemistry_subprocess)

    with pytest.raises(RuntimeError, match="non-converged levels.*1/1"):
        generation._run_single_fastchem_spec(
            _minimal_fastchem_spec(),
            source_root=tmp_path / "source",
            worker_base=tmp_path / "workers",
            runs_dir=tmp_path / "runs",
            config=config,
        )

    assert observed["timeout"] == 12.5


def test_patch_python_assignments_replaces_multiline_assignment():
    text = "sl_angle = (\n    48 / 180.0 * 3.14159\n)\natom_list = ['H']\n"

    patched = generation.patch_python_assignments(text, {"sl_angle": 0.5})

    namespace = {}
    exec(compile(patched, "<patched_vulcan_cfg>", "exec"), namespace)
    assert namespace["sl_angle"] == 0.5
    assert "48 / 180.0" not in patched


def test_vulcan_jax_backend_requires_importable_package(monkeypatch):
    def fake_import_module(name: str):
        if name == "vulcan_jax":
            raise ImportError("missing vulcan_jax")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(generation.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="requires an importable vulcan_jax"):
        generation._discover_installed_vulcan_jax_root()


def test_vulcan_jax_generation_warms_cache_before_parallel_fanout(
    tmp_path,
    monkeypatch,
):
    config = load_and_validate_config(ROOT / "config" / "vulcan_luhman16a_10k.json")
    config["paths"]["run_root"] = str(tmp_path / "vulcan")
    config["paths"]["raw_root"] = str(tmp_path / "vulcan" / "raw")
    config["paths"]["processed_root"] = str(tmp_path / "vulcan" / "processed")
    config["generation"]["num_runs"] = 3
    config["generation"]["overwrite"] = True
    config["generation"]["sample_chunk_size"] = 3
    config["generation"]["parallel_workers"] = 2
    config["generation"]["backfill"] = {"enabled": False, "max_retries": 0}
    config["vulcan_runtime"]["worker_root"] = str(tmp_path / "workers")

    specs = [_minimal_vulcan_spec(i) for i in range(3)]
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        generation,
        "_validated_vulcan_paths",
        lambda *_args, **_kwargs: (tmp_path / "source", tmp_path / "source" / "chem.txt"),
    )
    monkeypatch.setattr(generation, "build_sampling_plan", lambda **_kwargs: object())
    monkeypatch.setattr(
        generation,
        "sample_run_specifications_slice",
        lambda _plan, *, start, end: specs[start:end],
    )

    def fake_run_single(spec, *, source_root, worker_base, runs_dir, config):
        del source_root, worker_base, config
        calls.append((spec.run_id, threading.current_thread().name))
        output_path = Path(runs_dir) / f"{spec.run_id}.h5"
        with h5py.File(output_path, "w") as handle:
            handle.create_dataset("dummy", data=np.array([1.0], dtype=np.float64))
        return output_path

    monkeypatch.setattr(generation, "_run_single_vulcan_jax_spec", fake_run_single)

    artifact = generation.run_vulcan_generation(config, project_root=ROOT)

    assert calls[0] == ("run_000000", "MainThread")
    assert sorted(run_id for run_id, _thread_name in calls) == [
        "run_000000",
        "run_000001",
        "run_000002",
    ]
    assert artifact.consolidated_path.exists()


def test_exogibbs_generation_honors_shard_slice(tmp_path, monkeypatch):
    config = _load_fastchem_fixture_config(tmp_path)
    config["chemistry_type"] = "exogibbs"
    config["paths"]["run_root"] = str(tmp_path / "exogibbs")
    config["paths"]["raw_root"] = str(tmp_path / "exogibbs" / "raw")
    config["paths"]["processed_root"] = str(tmp_path / "exogibbs" / "processed")
    config["generation"]["num_runs"] = 4
    config["generation"]["overwrite"] = True
    config["generation"]["sample_chunk_size"] = 2
    config["generation"]["parallel_workers"] = 1
    config["generation"]["backfill"] = {"enabled": False, "max_retries": 0}
    config["temperature_profiles"]["source_mode"] = "analytic"
    config["roth_sampler"]["enabled"] = False

    monkeypatch.setattr(exogibbs_backend, "_init_runtime", lambda _: object())

    def fake_profile(runtime, spec):
        del runtime
        return np.full(
            (spec.pressure_bar.size, len(config["data_spec"]["output_species"])),
            1.0e-8,
            dtype=np.float64,
        )

    monkeypatch.setattr(exogibbs_backend, "_run_single_profile", fake_profile)

    artifact = exogibbs_backend.run_exogibbs_generation(
        config,
        project_root=ROOT,
        shard_id=1,
        num_shards=2,
        staging_root=tmp_path / "scratch",
    )

    fragment = json.loads(Path(artifact.manifest_path).read_text(encoding="utf-8"))
    assert artifact.run_ids == ["run_000002", "run_000003"]
    assert fragment["shard_start"] == 2
    assert fragment["shard_end"] == 4
    assert fragment["deterministic_run_ids"] == ["run_000002", "run_000003"]
    assert not (Path(config["paths"]["raw_root"]) / "runs.h5").exists()
