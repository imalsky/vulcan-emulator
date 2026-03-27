from __future__ import annotations

import json

import pytest

from src.utils.cli import main


def test_cli_rejects_legacy_subcommands():
    with pytest.raises(SystemExit):
        main(["--config", "config/equilibrium_only_config.json", "generation"])


def test_cli_normalization_stage_dispatches_preprocess(monkeypatch, capsys, tmp_path):
    expected_config = {"task": {"kind": "equilibrium_only"}}
    called: dict[str, object] = {}

    monkeypatch.setattr("src.utils.cli.resolve_project_root", lambda _start: tmp_path)
    monkeypatch.setattr("src.utils.cli._load_config", lambda _path, _root: expected_config)

    def _fake_preprocess(config: dict, *, project_root):
        called["config"] = config
        called["project_root"] = project_root
        return {"processed_root": "processed/demo"}

    monkeypatch.setattr("src.utils.cli.preprocess_raw_dataset", _fake_preprocess)

    exit_code = main(["--config", "config/equilibrium_only_config.json", "--stage", "normalization"])
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert called["config"] is expected_config
    assert called["project_root"] == tmp_path
    assert json.loads(stdout)["processed_root"] == "processed/demo"
