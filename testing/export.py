#!/usr/bin/env python3
"""Export a trained checkpoint to a standalone PT2 module with physical-space inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Prevent duplicate OpenMP runtime aborts before importing torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

from common import PROJECT_ROOT, iter_split_shards, load_checkpoint
from inference import load_physical_space_model, physical_inputs_from_processed_arrays


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone PT2 export."""
    parser = argparse.ArgumentParser(description="Export standalone PT2 model.")
    parser.add_argument("--run-dir", type=Path, default=Path("models/trained_model"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--output", type=str, default="standalone_model.pt2")
    return parser.parse_args()


def main() -> None:
    """Export one trained run to PT2 and verify round-trip parity."""
    args = _parse_args()
    run_dir = (PROJECT_ROOT / args.run_dir).resolve()
    checkpoint = load_checkpoint(run_dir=run_dir, checkpoint_name=args.checkpoint)
    config = checkpoint["config"]
    data_root = (PROJECT_ROOT / str(config["paths"]["data_root"])).resolve()
    processed_root = data_root / "processed"

    model, normalization_metadata, data_contract = load_physical_space_model(
        run_dir,
        checkpoint_name=args.checkpoint,
        device=torch.device("cpu"),
    )

    try:
        seq, glb, _tgt = next(iter(iter_split_shards(processed_root=processed_root, split=args.split)))
    except StopIteration as exc:
        raise RuntimeError(f"No shards found for split '{args.split}'.") from exc

    if seq.shape[0] <= 0:
        raise RuntimeError(f"First shard for split '{args.split}' has zero samples.")

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
        torch.as_tensor(example["initial_ymix"], dtype=torch.float32),
        torch.as_tensor(example["gravity_cm_s2"], dtype=torch.float32),
        torch.as_tensor(example["metallicity_log10"], dtype=torch.float32),
        torch.as_tensor(example["c_to_o"], dtype=torch.float32),
        torch.as_tensor(example["time_s"], dtype=torch.float32),
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

    metadata = {
        "checkpoint": args.checkpoint,
        "exported_model": str(output_path),
        "split_used_for_trace": args.split,
        "example_pressure_shape": list(example_tensors[0].shape),
        "example_initial_ymix_shape": list(example_tensors[3].shape),
        "example_time_shape": list(example_tensors[7].shape),
        "target_species": list(data_contract["target_species_order"]),
    }
    metadata_path = output_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved standalone PT2 model: {output_path}")
    print(f"Saved export metadata: {metadata_path}")


if __name__ == "__main__":
    main()
