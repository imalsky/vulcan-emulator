#!/usr/bin/env python3
"""
Randomly subsamples sequences from a large HDF5 file and splits the result
into N shards (default: 5) named *_1.h5, *_2.h5, ... for downstream parallelism.

Single-pass over the source file; reads sequential batches and writes the
selected rows to the correct shard(s) to minimize random I/O.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def _derive_shard_paths(dest_path: Path, num_shards: int) -> List[Path]:
    stem, suffix = dest_path.stem, dest_path.suffix
    # If original had no suffix, we still add numeric suffixes without extra dot.
    return [dest_path.with_name(f"{stem}_{i+1}{suffix}") for i in range(num_shards)]


def _split_counts(total: int, num_shards: int) -> List[int]:
    base = total // num_shards
    rem = total % num_shards
    # First `rem` shards get one extra
    return [base + (1 if i < rem else 0) for i in range(num_shards)]


def subsample_hdf5_batched_sharded(
    source_path: Path,
    dest_path: Path,
    num_samples: int,
    batch_size: int,
    num_shards: int = 5,
):
    """
    Subsample `num_samples` rows (without replacement) from root-level datasets in `source_path`
    and write them evenly into `num_shards` HDF5 files derived from `dest_path`.

    Assumptions:
    - All datasets share the same first dimension (number of sequences).
    - Datasets are at the file root (no groups). Only root-level h5py.Dataset entries are used.
    """
    if not source_path.exists():
        logging.error(f"Source file not found: {source_path}")
        return

    logging.info(f"Opening source file: {source_path}")
    with h5py.File(source_path, "r") as hf_in:
        # Determine dataset keys at the root and pick one to infer length
        dataset_keys = [k for k, v in hf_in.items() if isinstance(v, h5py.Dataset)]
        if not dataset_keys:
            logging.error(f"Source file '{source_path}' has no root-level datasets. Aborting.")
            return

        first_dset_key = dataset_keys[0]
        total_sequences = hf_in[first_dset_key].shape[0]
        logging.info(f"Source file contains {total_sequences} total sequences.")

        # Validate request
        if num_samples > total_sequences:
            logging.error(f"Requested {num_samples} samples, but only {total_sequences} are available. Aborting.")
            return
        if num_samples <= 0:
            logging.error("Number of samples must be a positive number. Aborting.")
            return
        if num_shards <= 0:
            logging.error("Number of shards must be a positive integer. Aborting.")
            return

        # Sample once (sorted for efficient batching)
        logging.info(f"Generating {num_samples} unique random indices...")
        random_indices = np.sort(np.random.choice(total_sequences, size=num_samples, replace=False))

        # Compute per-shard sizes and index slices
        shard_counts = _split_counts(num_samples, num_shards)
        shard_offsets = np.cumsum([0] + shard_counts)
        shard_slices = [random_indices[shard_offsets[i]:shard_offsets[i+1]] for i in range(num_shards)]

        shard_paths = _derive_shard_paths(dest_path, num_shards)
        for p in shard_paths:
            if p.exists():
                logging.warning(f"Destination shard '{p}' already exists. It will be overwritten.")
                p.unlink()

        # Create and prepare shard files/datasets
        logging.info(f"Creating {num_shards} destination shards:")
        shard_files = []
        shard_ds_handles = []  # list of dicts: per-shard {key: dset_handle}
        shard_cursors = np.zeros(num_shards, dtype=np.int64)

        for si, (p, nrows) in enumerate(zip(shard_paths, shard_counts)):
            logging.info(f"  - {p.name}: {nrows} rows")
            hf_out = h5py.File(p, "w")

            # Copy file-level attributes
            for k, v in hf_in.attrs.items():
                hf_out.attrs[k] = v

            # Create datasets with preserved layout / filters
            ds_map = {}
            for key in dataset_keys:
                src = hf_in[key]
                dest_shape = (nrows,) + src.shape[1:]

                # Preserve dataset creation properties when possible
                ds = hf_out.create_dataset(
                    name=key,
                    shape=dest_shape,
                    dtype=src.dtype,
                    chunks=src.chunks,
                    compression=src.compression,
                    compression_opts=src.compression_opts,
                    shuffle=getattr(src, "shuffle", False),
                    fletcher32=getattr(src, "fletcher32", False),
                )
                # Copy dataset attributes
                for ak, av in src.attrs.items():
                    ds.attrs[ak] = av

                ds_map[key] = ds

            shard_files.append(hf_out)
            shard_ds_handles.append(ds_map)

        # Stream source in sequential batches and dispatch rows to shards
        progress_bar = tqdm(range(0, total_sequences, batch_size), desc="Processing in batches")
        for start_idx in progress_bar:
            end_idx = min(start_idx + batch_size, total_sequences)

            # For each shard, find which requested indices fall in this batch
            # Keep (shard_id, rel_indices) pairs for which we have work
            per_shard_rel = []
            for si in range(num_shards):
                sidx = shard_slices[si]
                left = np.searchsorted(sidx, start_idx, side="left")
                right = np.searchsorted(sidx, end_idx, side="left")
                if right > left:
                    abs_idx = sidx[left:right]
                    rel_idx = abs_idx - start_idx  # indices within [start_idx:end_idx)
                    per_shard_rel.append((si, rel_idx))

            if not per_shard_rel:
                continue

            # Read each dataset chunk once, then scatter to shards
            for key in dataset_keys:
                chunk = hf_in[key][start_idx:end_idx]  # contiguous read

                for si, rel_idx in per_shard_rel:
                    n = len(rel_idx)
                    if n == 0:
                        continue
                    cursor = shard_cursors[si]
                    shard_ds_handles[si][key][cursor:cursor + n] = chunk[rel_idx]

            # Advance cursors
            for si, rel_idx in per_shard_rel:
                shard_cursors[si] += len(rel_idx)

        # Close shard files
        for hf_out in shard_files:
            hf_out.flush()
            hf_out.close()

    # Report sizes
    for p in shard_paths:
        size_mb = p.stat().st_size / (1024 ** 2)
        logging.info(f"Created: {p} ({size_mb:.2f} MB)")

    logging.info("Subsampling & sharding complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly and efficiently subsample an HDF5 file and split into shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("source_file", type=str, help="Path to the large source HDF5 file (e.g., gcm_profiles.h5).")
    parser.add_argument("destination_file", type=str, help="Base path for output shards (e.g., gcm_profiles_sub.h5).")
    parser.add_argument("-n", "--num-samples", type=int, default=500000,
                        help="Total number of random sequences to select (distributed across shards).")
    parser.add_argument("-b", "--batch-size", type=int, default=10000,
                        help="Number of rows to process per batch.")
    parser.add_argument("-s", "--num-shards", type=int, default=5,
                        help="Number of output shards (files) to create.")

    args = parser.parse_args()

    subsample_hdf5_batched_sharded(
        source_path=Path(args.source_file),
        dest_path=Path(args.destination_file),
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_shards=args.num_shards,
    )
