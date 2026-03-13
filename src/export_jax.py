from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import numpy as np


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): _json_ready(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(value) for value in obj]
    if isinstance(obj, tuple):
        return [_json_ready(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _flatten_tree(obj: Any, prefix: tuple[str, ...], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if isinstance(obj, dict):
        return {
            "type": "dict",
            "items": {
                key: _flatten_tree(value, prefix + (str(key),), arrays)
                for key, value in obj.items()
            },
        }
    if isinstance(obj, list):
        return {
            "type": "list",
            "items": [
                _flatten_tree(value, prefix + (str(index),), arrays)
                for index, value in enumerate(obj)
            ],
        }
    key = "__".join(prefix)
    arrays[key] = np.asarray(obj)
    return {"type": "array", "key": key}


def _unflatten_tree(spec: dict[str, Any], arrays: dict[str, np.ndarray]) -> Any:
    spec_type = spec["type"]
    if spec_type == "dict":
        return {key: _unflatten_tree(value, arrays) for key, value in spec["items"].items()}
    if spec_type == "list":
        return [_unflatten_tree(value, arrays) for value in spec["items"]]
    if spec_type == "array":
        return arrays[spec["key"]]
    raise ValueError(f"Unsupported flattened tree spec type: {spec_type}")


def export_checkpoint_payload(payload: dict[str, Any], export_root: str | Path) -> Path:
    root = Path(export_root)
    root.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    structure = _flatten_tree(payload["params"], ("params",), arrays)
    np.savez(root / "params.npz", **arrays)
    (root / "structure.json").write_text(json.dumps(structure, indent=2) + "\n", encoding="utf-8")
    (root / "contract.json").write_text(json.dumps(_json_ready(payload["data_contract"]), indent=2) + "\n", encoding="utf-8")
    (root / "normalization.json").write_text(json.dumps(_json_ready(payload["normalization"]), indent=2) + "\n", encoding="utf-8")
    (root / "config.json").write_text(json.dumps(_json_ready(payload["config"]), indent=2) + "\n", encoding="utf-8")
    (root / "model_dimensions.json").write_text(
        json.dumps(_json_ready(payload["model_dimensions"]), indent=2) + "\n",
        encoding="utf-8",
    )
    return root


def load_export_bundle(export_root: str | Path) -> dict[str, Any]:
    root = Path(export_root)
    arrays = {key: value for key, value in np.load(root / "params.npz").items()}
    structure = json.loads((root / "structure.json").read_text(encoding="utf-8"))
    params_tree = _unflatten_tree(structure, arrays)
    if isinstance(params_tree, dict) and "params" in params_tree:
        params = params_tree["params"]
    else:
        params = params_tree
    return {
        "params": jax.tree_util.tree_map(lambda x: x, params),
        "data_contract": json.loads((root / "contract.json").read_text(encoding="utf-8")),
        "normalization": json.loads((root / "normalization.json").read_text(encoding="utf-8")),
        "config": json.loads((root / "config.json").read_text(encoding="utf-8")),
        "model_dimensions": json.loads((root / "model_dimensions.json").read_text(encoding="utf-8")),
    }
