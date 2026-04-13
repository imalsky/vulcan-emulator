from __future__ import annotations

from pathlib import Path

from src.utils.helpers import resolve_project_root


def test_resolve_project_root_falls_back_to_current_working_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project_root"
    (project_root / "src").mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")

    detached_src_file = tmp_path / "detached_copy" / "src" / "utils" / "cli.py"
    detached_src_file.parent.mkdir(parents=True)
    detached_src_file.write_text("", encoding="utf-8")

    monkeypatch.chdir(project_root)

    assert resolve_project_root(detached_src_file) == project_root.resolve()
