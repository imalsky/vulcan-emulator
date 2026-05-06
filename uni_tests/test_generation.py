from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.data_generation import generation
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

    def fake_run(args, *, cwd, check, timeout):
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
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(generation, "_ensure_fastchem_worker_tree", fake_ensure)
    monkeypatch.setattr(generation, "_reset_fastchem_worker_between_runs", lambda _: None)
    monkeypatch.setattr(generation, "_write_fastchem_element_abundances", lambda *a, **k: None)
    monkeypatch.setattr(generation, "_write_fastchem_tp_profile", lambda *a, **k: None)
    monkeypatch.setattr(generation.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="non-converged levels.*1/1"):
        generation._run_single_fastchem_spec(
            _minimal_fastchem_spec(),
            source_root=tmp_path / "source",
            worker_base=tmp_path / "workers",
            runs_dir=tmp_path / "runs",
            config=config,
        )

    assert observed["timeout"] == 12.5
