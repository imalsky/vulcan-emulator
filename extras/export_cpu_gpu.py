#!/usr/bin/env python3
"""Export the trained physical-space model on CPU and CUDA when available.

The exported ``.pt2`` files are fully standalone: all normalization statistics
(mean, std, min, max per variable) are baked into the model as registered
buffers, so the file operates entirely in physical space without needing any
external metadata files (normalization_metadata.json, data_contract.json, etc.).

Input signature (all physical units):
    pressure_bar      [batch, nz]              Pressure in bar
    temperature_k     [batch, nz]              Temperature in Kelvin
    kzz_cm2_s         [batch, nz]              Eddy diffusion in cm^2/s
    anchor_ymix       [batch, nz, n_species]   Mixing ratios (dimensionless)
    gravity_cm_s2     [batch]                  Surface gravity in cm/s^2
    metallicity_log10 [batch]                  log10(metallicity / solar)
    c_to_o            [batch]                  Carbon-to-oxygen ratio
    dt_s              [batch]                  Time step in seconds

Output:
    predicted_ymix    [batch, nz, n_output]    Predicted mixing ratios (physical)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
import torch

from inference import load_physical_space_model, physical_inputs_from_processed_arrays
from script_utils import (
    load_checkpoint,
    load_fixed_split_sample,
    resolve_processed_root_from_checkpoint,
    resolve_run_dir,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
RUN_DIR_OVERRIDE: Path | None = None
TEST_SPLIT = "test"
SAMPLE_INDEX = 0
CHECKPOINT_NAME = "best.pt"
OUTPUT_PREFIX = "standalone_model"


def _example_tensors(
    *,
    run_dir: Path,
    processed_root: Path,
    sample_index: int,
    checkpoint_name: str,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """Build one representative physical-space example for export validation."""
    wrapper, normalization_metadata, data_contract = load_physical_space_model(
        run_dir,
        checkpoint_name=checkpoint_name,
        device=device,
    )
    checkpoint = load_checkpoint(run_dir=run_dir, checkpoint_name=checkpoint_name)
    seq, glb, _tgt, _dt = load_fixed_split_sample(
        processed_root=processed_root,
        split=TEST_SPLIT,
        config=checkpoint["config"],
        normalization_metadata=normalization_metadata,
        sample_index=sample_index,
    )
    example = physical_inputs_from_processed_arrays(
        sequence_inputs=seq,
        global_inputs=glb,
        normalization_metadata=normalization_metadata,
        data_contract=data_contract,
    )
    model_dtype = next(wrapper.parameters()).dtype
    tensors = (
        torch.as_tensor(example["pressure_bar"][None, :], dtype=model_dtype, device=device),
        torch.as_tensor(example["temperature_k"][None, :], dtype=model_dtype, device=device),
        torch.as_tensor(example["kzz_cm2_s"][None, :], dtype=model_dtype, device=device),
        torch.as_tensor(example["anchor_ymix"][None, :, :], dtype=model_dtype, device=device),
        torch.as_tensor([example["gravity_cm_s2"]], dtype=model_dtype, device=device),
        torch.as_tensor([example["metallicity_log10"]], dtype=model_dtype, device=device),
        torch.as_tensor([example["c_to_o"]], dtype=model_dtype, device=device),
        torch.as_tensor([example["dt_s"]], dtype=model_dtype, device=device),
    )
    return tensors, {"wrapper": wrapper}


def _export_one_device(
    *,
    run_dir: Path,
    processed_root: Path,
    sample_index: int,
    checkpoint_name: str,
    output_prefix: str,
    device: torch.device,
) -> dict[str, Any]:
    """Export and validate one standalone PT2 model on a single device."""
    example_tensors, payload = _example_tensors(
        run_dir=run_dir,
        processed_root=processed_root,
        sample_index=sample_index,
        checkpoint_name=checkpoint_name,
        device=device,
    )
    wrapper = payload["wrapper"]

    with torch.inference_mode():
        exported_program = torch.export.export(wrapper, args=example_tensors, strict=True)
        reference = wrapper(*example_tensors)

    output_path = run_dir / f"{output_prefix}_{device.type}.pt2"
    torch.export.save(exported_program, str(output_path))

    loaded_program = torch.export.load(str(output_path))
    loaded_module = loaded_program.module()
    if hasattr(loaded_module, "to"):
        loaded_module = loaded_module.to(device=device)
    with torch.inference_mode():
        prediction = loaded_module(*example_tensors)
    torch.testing.assert_close(reference, prediction, rtol=1e-4, atol=1e-5)

    return {
        "device": device.type,
        "output": str(output_path),
        "status": "ok",
    }


def main() -> None:
    """Export CPU and optional CUDA standalone models."""
    run_dir, config_path = resolve_run_dir(config_path=CONFIG_PATH, run_dir=RUN_DIR_OVERRIDE)
    processed_root = resolve_processed_root_from_checkpoint(
        run_dir=run_dir,
        checkpoint_name=CHECKPOINT_NAME,
    )

    results = [
        _export_one_device(
            run_dir=run_dir,
            processed_root=processed_root,
            sample_index=SAMPLE_INDEX,
            checkpoint_name=CHECKPOINT_NAME,
            output_prefix=OUTPUT_PREFIX,
            device=torch.device("cpu"),
        )
    ]
    if torch.cuda.is_available():
        results.append(
            _export_one_device(
                run_dir=run_dir,
                processed_root=processed_root,
                sample_index=SAMPLE_INDEX,
                checkpoint_name=CHECKPOINT_NAME,
                output_prefix=OUTPUT_PREFIX,
                device=torch.device("cuda"),
            )
        )
    else:
        results.append({"device": "cuda", "status": "skipped", "reason": "CUDA unavailable"})

    print("Standalone export")
    print(f"  Config path  : {config_path if config_path is not None else 'explicit run dir override'}")
    print(f"  Run dir      : {run_dir}")
    print(f"  Checkpoint   : {CHECKPOINT_NAME}")
    print(f"  Split        : {TEST_SPLIT}")
    print(f"  Sample index : {SAMPLE_INDEX}")
    for item in results:
        if item["status"] == "ok":
            print(f"  {item['device'].upper():<11}: wrote {item['output']}")
        else:
            print(f"  {item['device'].upper():<11}: skipped ({item['reason']})")


if __name__ == "__main__":
    main()
