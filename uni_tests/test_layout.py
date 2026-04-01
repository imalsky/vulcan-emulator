from __future__ import annotations

import json
from pathlib import Path


def test_asset_layout_uses_assets_for_immutable_inputs():
    root = Path(__file__).resolve().parents[1]
    fastchem = json.loads((root / "config" / "fastchem_mlp_config.json").read_text(encoding="utf-8"))
    vulcan = json.loads((root / "config" / "vulcan_transformer_config.json").read_text(encoding="utf-8"))
    assert fastchem["temperature_profiles"]["data_glob"].startswith("assets/")
    assert vulcan["vulcan"]["stellar_spectrum"]["template_file"].startswith("assets/")


def test_real_assets_are_gitignored_and_fixtures_are_local():
    root = Path(__file__).resolve().parents[1]
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    fixture = (
        root
        / "uni_tests"
        / "fixtures"
        / "pt_profiles"
        / "PTprofiles-Teq_1200-LogMet_0.0-LogDrag_0-Mstar_0.8-Rp_1.3-logG_1.3-TiOVO_false.dat"
    )
    assert "assets/**" in gitignore
    assert fixture.exists()
