from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from src.utils.cli import main


def test_cli_rejects_legacy_subcommands():
    with pytest.raises(SystemExit):
        main(["--config", "config/fastchem_mlp_config.json", "generation"])
    with pytest.raises(SystemExit):
        main(["--config", "config/fastchem_mlp_config.json", "--stage", "migrate"])


def test_cli_normalization_stage_dispatches_preprocess(monkeypatch, capsys, tmp_path):
    expected_config = {"chemistry_type": "fastchem", "model_type": "mlp"}
    called: dict[str, object] = {}

    monkeypatch.setattr("src.utils.cli.resolve_project_root", lambda _start: tmp_path)
    monkeypatch.setattr("src.utils.cli._load_config", lambda _path, _root: expected_config)

    def _fake_preprocess(config: dict, *, project_root):
        called["config"] = config
        called["project_root"] = project_root
        return {"run_root": "data/demo"}

    monkeypatch.setattr("src.utils.cli.preprocess_raw_dataset", _fake_preprocess)

    exit_code = main(["--config", "config/fastchem_mlp_config.json", "--stage", "normalization"])
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert called["config"] is expected_config
    assert called["project_root"] == tmp_path
    assert json.loads(stdout)["run_root"] == "data/demo"


def test_cli_generation_stage_reports_only_run_root_layout(monkeypatch, capsys, tmp_path):
    expected_config = {"chemistry_type": "fastchem", "model_type": "mlp"}
    called: dict[str, object] = {}

    monkeypatch.setattr("src.utils.cli.resolve_project_root", lambda _start: tmp_path)
    monkeypatch.setattr("src.utils.cli._load_config", lambda _path, _root: expected_config)

    artifact = SimpleNamespace(
        run_root=tmp_path / "dataset",
        run_ids=["run_00000"],
        consolidated_path=tmp_path / "dataset" / "raw" / "runs.h5",
        manifest_path=tmp_path / "dataset" / "info" / "generation_manifest.json",
        coverage_path=tmp_path / "dataset" / "info" / "sampling_coverage.json",
    )

    def _fake_generate(config: dict, *, project_root):
        called["config"] = config
        called["project_root"] = project_root
        return artifact

    monkeypatch.setattr("src.utils.cli.generate_raw_dataset", _fake_generate)

    exit_code = main(["--config", "config/fastchem_mlp_config.json", "--stage", "generation"])
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert called["config"] is expected_config
    assert called["project_root"] == tmp_path
    assert json.loads(stdout) == {
        "run_root": str(artifact.run_root),
        "num_runs": 1,
        "manifest_path": str(artifact.manifest_path),
        "coverage_path": str(artifact.coverage_path),
    }
