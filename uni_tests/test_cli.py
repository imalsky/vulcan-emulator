from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.utils.cli import main


def test_cli_rejects_legacy_subcommands():
    with pytest.raises(SystemExit):
        main(["--config", "config/fastchem_mlp_config.json", "generation"])


def test_cli_normalization_stage_dispatches_preprocess(monkeypatch, capsys, tmp_path):
    expected_config = {"chemistry_type": "fastchem", "model_type": "mlp"}
    called: dict[str, object] = {}

    monkeypatch.setattr("src.utils.cli.resolve_project_root", lambda _start: tmp_path)
    monkeypatch.setattr("src.utils.cli._load_config", lambda _path, _root: expected_config)

    def _fake_preprocess(config: dict, *, project_root):
        called["config"] = config
        called["project_root"] = project_root
        return {"processed_root": "processed/demo"}

    monkeypatch.setattr("src.utils.cli.preprocess_raw_dataset", _fake_preprocess)

    exit_code = main(["--config", "config/fastchem_mlp_config.json", "--stage", "normalization"])
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert called["config"] is expected_config
    assert called["project_root"] == tmp_path
    assert json.loads(stdout)["processed_root"] == "processed/demo"


def test_direct_module_imports_do_not_trigger_cli_circular_imports():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import src.data_generation.preprocess; "
                "import src.data_generation.generation; "
                "from src.utils import main; "
                "assert callable(main)"
            ),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
