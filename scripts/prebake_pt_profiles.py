"""Prebake the Roth PT-library .dat files into a single .npz bundle.

The .dat parse path in roth_sampling.py is dominated by Python-level line
filtering plus np.genfromtxt — for a 4+ GB library this can stall the
generation pipeline for many minutes before any worker emits a run.

This one-shot script expands every .dat into its per-(lon, lat) profiles
and packs them into a single bundle file (concatenated arrays plus an
offsets index and per-profile JSON metadata) that loads in milliseconds.

The runtime preference for the bundle is wired up in
``src/data_generation/sampling.py:_resolve_roth_data_glob`` and
``src/data_generation/roth_sampling.py:load_roth_profiles_native``.

Usage:

    python scripts/prebake_pt_profiles.py
    python scripts/prebake_pt_profiles.py \
        --input-dir assets/PTprofiles --output-path assets/PTprofiles.bundle.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_generation.roth_sampling import _load_pt_dat_profiles  # noqa: E402

BUNDLE_VERSION = 1


def prebake(input_dir: Path, output_path: Path) -> tuple[int, int]:
    dat_files = sorted(input_dir.glob("*.dat"))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in {input_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pressures: list[np.ndarray] = []
    temperatures: list[np.ndarray] = []
    offsets: list[int] = [0]
    metadata_blobs: list[str] = []

    started = time.time()
    files_processed = 0
    for index, dat_path in enumerate(dat_files, start=1):
        try:
            profiles = _load_pt_dat_profiles(dat_path)
        except Exception as exc:
            print(f"  [{index}/{len(dat_files)}] FAILED {dat_path.name}: {exc}")
            continue
        for profile in profiles:
            p = np.asarray(profile.pressure_bar, dtype=np.float64)
            t = np.asarray(profile.temperature_k, dtype=np.float64)
            pressures.append(p)
            temperatures.append(t)
            offsets.append(offsets[-1] + p.size)
            metadata_blobs.append(json.dumps(profile.metadata))
        files_processed += 1
        if index % 25 == 0 or index == len(dat_files):
            elapsed = time.time() - started
            print(
                f"  [{index}/{len(dat_files)}] {dat_path.name} "
                f"(profiles={len(metadata_blobs)} elapsed={elapsed:.1f}s)"
            )

    if not metadata_blobs:
        raise RuntimeError("No profiles were extracted; refusing to write empty bundle.")

    pressure_concat = np.concatenate(pressures, axis=0)
    temperature_concat = np.concatenate(temperatures, axis=0)
    offsets_array = np.asarray(offsets, dtype=np.int64)
    metadata_array = np.asarray(metadata_blobs, dtype=np.object_)

    print(
        f"Writing bundle to {output_path} "
        f"(profiles={len(metadata_blobs)} samples={pressure_concat.size})"
    )
    np.savez(
        output_path,
        version=np.array(BUNDLE_VERSION, dtype=np.int64),
        pressure_bar=pressure_concat,
        temperature_k=temperature_concat,
        offsets=offsets_array,
        metadata=metadata_array,
    )
    return files_processed, len(metadata_blobs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "PTprofiles",
        help="Directory containing the original Roth .dat files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "assets" / "PTprofiles.bundle.npz",
        help="Destination path for the single PT-library bundle.",
    )
    args = parser.parse_args()

    print(f"Prebaking PT-library: {args.input_dir} -> {args.output_path}")
    files, profiles = prebake(args.input_dir.resolve(), args.output_path.resolve())
    print(f"Done. Files processed: {files}. Profiles packed: {profiles}.")


if __name__ == "__main__":
    main()
