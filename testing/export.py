#!/usr/bin/env python3
"""Export a trained checkpoint to a standalone PT2 module with physical-space inputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
TESTING_DIR = Path(__file__).resolve().parent
if str(TESTING_DIR) not in sys.path:
    sys.path.insert(0, str(TESTING_DIR))
import torch

from common import PROJECT_ROOT, iter_split_shards, load_checkpoint
from inference import load_physical_space_model, physical_inputs_from_processed_arrays


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export standalone PT2 model.")
    parser.add_argument("--run-dir", type=Path, default=Path("models/trained_model"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--output", type=str, default="standalone_model.pt2")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = (PROJECT_ROOT / args.run_dir).resolve()
    checkpoint = load_checkpoint(run_dir=run_dir, checkpoint_name=args.checkpoint)
    config = checkpoint["config"]
    processed_root_cfg = str(
        config["paths"].get("processed_root", str(Path(config["paths"]["data_root"]) / "processed"))
    )
    processed_root = (PROJECT_ROOT / processed_root_cfg).resolve()

    model, normalization_metadata, data_contract = load_physical_space_model(
        run_dir,
        checkpoint_name=args.checkpoint,
        device=torch.device("cpu"),
    )

    try:
        seq, glb, _tgt, _dt = next(iter(iter_split_shards(processed_root=processed_root, split=args.split)))
    except StopIteration as exc:
        raise RuntimeError(f"No shards found for split '{args.split}'.") from exc

    example = physical_inputs_from_processed_arrays(
        sequence_inputs=seq[:1],
        global_inputs=glb[:1],
        normalization_metadata=normalization_metadata,
        data_contract=data_contract,
    )
    example_tensors = (
        torch.as_tensor(example["pressure_bar"], dtype=torch.float32),
        torch.as_tensor(example["temperature_k"], dtype=torch.float32),
        torch.as_tensor(example["kzz_cm2_s"], dtype=torch.float32),
        torch.as_tensor(example["anchor_ymix"], dtype=torch.float32),
        torch.as_tensor(example["gravity_cm_s2"], dtype=torch.float32),
        torch.as_tensor(example["metallicity_log10"], dtype=torch.float32),
        torch.as_tensor(example["c_to_o"], dtype=torch.float32),
        torch.as_tensor(example["dt_s"], dtype=torch.float32),
    )

    with torch.inference_mode():
        exported_program = torch.export.export(model, args=example_tensors, strict=True)
        reference = model(*example_tensors)

    output_path = run_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported_program, str(output_path))

    loaded_program = torch.export.load(str(output_path))
    loaded_module = loaded_program.module()
    with torch.inference_mode():
        prediction = loaded_module(*example_tensors)
    torch.testing.assert_close(reference, prediction, rtol=1e-4, atol=1e-5)

    summary = {
        "run_dir": str(run_dir),
        "split": args.split,
        "checkpoint": args.checkpoint,
        "output": str(output_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
