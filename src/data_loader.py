"""Processed-split data loading utilities with explicit RAM/disk policies.

Supports three loading modes for processed ``.npy`` transition shards:

- **ram**: Loads all shards into host memory at init time. Best for small
  datasets that fit comfortably in RAM.
- **disk**: Lazy-loads shards on demand with an LRU shard cache. Suitable
  for datasets too large for RAM; large shards can optionally use mmap.
- **auto**: Estimates available RAM and selects ram/disk automatically.

Each sample is a 5-tuple: ``(sequence_inputs, globals, targets, padding_mask, dt_s)``.

For CUDA training, :class:`DevicePrefetchLoader` wraps any DataLoader with
asynchronous host->device transfers on a separate CUDA stream, overlapping
data movement with compute.
"""

from __future__ import annotations

import bisect
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


class DataLoadingError(RuntimeError):
    """Raised when processed-split loading contracts are violated."""


@dataclass(frozen=True)
class DataLoadingConfig:
    """Configurable dataset-loading policy."""

    mode: str
    max_cached_shards: int
    large_shard_mmap_bytes: int
    ram_safety_fraction: float
    copy_mmap_slices: bool
    use_device_prefetch: bool


def load_split_metadata(split_dir: Path) -> dict[str, Any]:
    """Load and validate one processed-split metadata file."""
    metadata_path = split_dir / "metadata.json"
    if not metadata_path.is_file():
        raise DataLoadingError(f"Missing split metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    integer_keys = {
        "num_shards",
        "sequence_length",
        "input_dim",
        "global_dim",
        "target_dim",
        "state_dim",
        "total_samples",
    }
    for key in integer_keys:
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise DataLoadingError(f"Invalid metadata integer field '{key}' in {metadata_path}.")
        if value <= 0:
            raise DataLoadingError(f"Metadata field '{key}' must be > 0 in {metadata_path}.")

    for key in (
        "sequence_feature_order",
        "global_feature_order",
        "state_species_order",
        "output_species_order",
    ):
        value = metadata.get(key)
        if not isinstance(value, list) or not value:
            raise DataLoadingError(f"Invalid metadata list field '{key}' in {metadata_path}.")
        if any((not isinstance(item, str) or not item.strip()) for item in value):
            raise DataLoadingError(f"Metadata field '{key}' contains an invalid entry.")

    output_from_state_indices = metadata.get("output_from_state_indices")
    if not isinstance(output_from_state_indices, list) or not output_from_state_indices:
        raise DataLoadingError(
            f"Invalid metadata list field 'output_from_state_indices' in {metadata_path}."
        )
    if any(isinstance(item, bool) or not isinstance(item, int) for item in output_from_state_indices):
        raise DataLoadingError(
            f"Metadata field 'output_from_state_indices' must contain integers in {metadata_path}."
        )

    fingerprint = metadata.get("normalization_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise DataLoadingError(
            "Invalid metadata field 'normalization_fingerprint' in "
            f"{metadata_path}: expected 64-character sha256 hex string."
        )
    try:
        int(fingerprint, 16)
    except ValueError as exc:
        raise DataLoadingError(
            f"Invalid normalization_fingerprint in {metadata_path}: must be hex."
        ) from exc

    if len(metadata["sequence_feature_order"]) != int(metadata["input_dim"]):
        raise DataLoadingError("Invalid sequence feature order length in split metadata.")
    if len(metadata["global_feature_order"]) != int(metadata["global_dim"]):
        raise DataLoadingError("Invalid global feature order length in split metadata.")
    if len(metadata["state_species_order"]) != int(metadata["state_dim"]):
        raise DataLoadingError("Invalid state species order length in split metadata.")
    if len(metadata["output_species_order"]) != int(metadata["target_dim"]):
        raise DataLoadingError("Invalid output species order length in split metadata.")
    if len(output_from_state_indices) != int(metadata["target_dim"]):
        raise DataLoadingError(
            "Invalid output_from_state_indices length in split metadata."
        )
    if any(index < 0 or index >= int(metadata["state_dim"]) for index in output_from_state_indices):
        raise DataLoadingError(
            "output_from_state_indices contains an out-of-range state index."
        )
    return metadata


def _to_tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.from_numpy(array)
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _estimate_available_ram_bytes() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    if hasattr(os, "sysconf"):
        names = os.sysconf_names
        if "SC_PAGE_SIZE" in names and "SC_AVPHYS_PAGES" in names:
            try:
                page_size = int(os.sysconf("SC_PAGE_SIZE"))
                available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
                if page_size > 0 and available_pages > 0:
                    return page_size * available_pages
            except (ValueError, OSError):
                pass
    return -1


def load_full_split_arrays(
    split_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load an entire processed split eagerly into host memory."""
    metadata = load_split_metadata(split_dir)
    num_shards = int(metadata["num_shards"])
    expected_seq_len = int(metadata["sequence_length"])
    expected_input_dim = int(metadata["input_dim"])
    expected_global_dim = int(metadata["global_dim"])
    expected_target_dim = int(metadata["target_dim"])

    seq_parts: list[np.ndarray] = []
    glb_parts: list[np.ndarray] = []
    tgt_parts: list[np.ndarray] = []
    dt_parts: list[np.ndarray] = []

    for shard_idx in range(num_shards):
        seq_path = split_dir / "sequence_inputs" / f"shard_{shard_idx:05d}.npy"
        glb_path = split_dir / "globals" / f"shard_{shard_idx:05d}.npy"
        tgt_path = split_dir / "targets" / f"shard_{shard_idx:05d}.npy"
        dt_path = split_dir / "dt_s" / f"shard_{shard_idx:05d}.npy"
        for path in (seq_path, glb_path, tgt_path, dt_path):
            if not path.is_file():
                raise DataLoadingError(f"Missing shard file: {path}")
        seq = np.load(seq_path, allow_pickle=False)
        glb = np.load(glb_path, allow_pickle=False)
        tgt = np.load(tgt_path, allow_pickle=False)
        dt = np.load(dt_path, allow_pickle=False)

        if seq.ndim != 3 or glb.ndim != 2 or tgt.ndim != 3 or dt.ndim != 1:
            raise DataLoadingError(f"Unexpected shard ranks in {split_dir} shard {shard_idx}.")
        if not (seq.shape[0] == glb.shape[0] == tgt.shape[0] == dt.shape[0]):
            raise DataLoadingError(f"Shard sample-count mismatch in {split_dir} shard {shard_idx}.")
        if seq.shape[1:] != (expected_seq_len, expected_input_dim):
            raise DataLoadingError(
                f"Sequence shape mismatch in {split_dir} shard {shard_idx}: {seq.shape}."
            )
        if glb.shape[1] != expected_global_dim:
            raise DataLoadingError(
                f"Global shape mismatch in {split_dir} shard {shard_idx}: {glb.shape}."
            )
        if tgt.shape[1:] != (expected_seq_len, expected_target_dim):
            raise DataLoadingError(
                f"Target shape mismatch in {split_dir} shard {shard_idx}: {tgt.shape}."
            )
        if np.any(~np.isfinite(seq)) or np.any(~np.isfinite(glb)) or np.any(~np.isfinite(tgt)) or np.any(~np.isfinite(dt)):
            raise DataLoadingError(f"Non-finite values found in processed split: {split_dir}")

        seq_parts.append(seq)
        glb_parts.append(glb)
        tgt_parts.append(tgt)
        dt_parts.append(dt)

    if not seq_parts:
        raise DataLoadingError(f"No shard data found for split: {split_dir}")

    sequence = np.concatenate(seq_parts, axis=0)
    globals_ = np.concatenate(glb_parts, axis=0)
    targets = np.concatenate(tgt_parts, axis=0)
    dt_s = np.concatenate(dt_parts, axis=0)
    if int(sequence.shape[0]) != int(metadata["total_samples"]):
        raise DataLoadingError(
            f"Split sample count mismatch in {split_dir}: metadata={metadata['total_samples']}, loaded={sequence.shape[0]}"
        )
    return sequence, globals_, targets, dt_s, metadata


class ProcessedSplitDataset(Dataset):
    """Sharded dataset with configurable RAM/disk/auto loading behavior.

    Reads processed ``.npy`` shard files written by ``preprocess.py``.
    In disk mode, maintains an ordered LRU cache of loaded shards to
    avoid repeated I/O for sequential access patterns.  Returns a fixed
    all-False padding mask (no padding in v1 since all runs share nz).
    """

    def __init__(
        self,
        *,
        split_dir: Path,
        metadata: dict[str, Any],
        input_dtype: torch.dtype,
        target_dtype: torch.dtype,
        loading: DataLoadingConfig,
    ) -> None:
        super().__init__()
        self.split_dir = split_dir
        self.input_dtype = input_dtype
        self.target_dtype = target_dtype
        self.loading = loading

        self.num_shards = int(metadata["num_shards"])
        self.sequence_length = int(metadata["sequence_length"])
        self.input_dim = int(metadata["input_dim"])
        self.global_dim = int(metadata["global_dim"])
        self.target_dim = int(metadata["target_dim"])
        self.total_samples = int(metadata["total_samples"])

        self.seq_paths = [split_dir / "sequence_inputs" / f"shard_{idx:05d}.npy" for idx in range(self.num_shards)]
        self.glb_paths = [split_dir / "globals" / f"shard_{idx:05d}.npy" for idx in range(self.num_shards)]
        self.tgt_paths = [split_dir / "targets" / f"shard_{idx:05d}.npy" for idx in range(self.num_shards)]
        self.dt_paths = [split_dir / "dt_s" / f"shard_{idx:05d}.npy" for idx in range(self.num_shards)]

        for seq_path, glb_path, tgt_path, dt_path in zip(
            self.seq_paths,
            self.glb_paths,
            self.tgt_paths,
            self.dt_paths,
            strict=True,
        ):
            for path in (seq_path, glb_path, tgt_path, dt_path):
                if not path.is_file():
                    raise DataLoadingError(f"Missing shard file: {path}")

        self._row_counts: list[int] = []
        for idx, (seq_path, glb_path, tgt_path, dt_path) in enumerate(
            zip(self.seq_paths, self.glb_paths, self.tgt_paths, self.dt_paths, strict=True)
        ):
            seq = np.load(seq_path, mmap_mode="r", allow_pickle=False)
            glb = np.load(glb_path, mmap_mode="r", allow_pickle=False)
            tgt = np.load(tgt_path, mmap_mode="r", allow_pickle=False)
            dt = np.load(dt_path, mmap_mode="r", allow_pickle=False)
            if seq.ndim != 3 or glb.ndim != 2 or tgt.ndim != 3 or dt.ndim != 1:
                raise DataLoadingError(f"Unexpected shard ranks in {split_dir} shard {idx}.")
            if not (seq.shape[0] == glb.shape[0] == tgt.shape[0] == dt.shape[0]):
                raise DataLoadingError(f"Shard sample-count mismatch in {split_dir} shard {idx}.")
            if seq.shape[1:] != (self.sequence_length, self.input_dim):
                raise DataLoadingError(f"Sequence shape mismatch in {split_dir} shard {idx}: {seq.shape[1:]}")
            if glb.shape[1] != self.global_dim:
                raise DataLoadingError(f"Global shape mismatch in {split_dir} shard {idx}: {glb.shape[1]}")
            if tgt.shape[1:] != (self.sequence_length, self.target_dim):
                raise DataLoadingError(f"Target shape mismatch in {split_dir} shard {idx}: {tgt.shape[1:]}")
            self._row_counts.append(int(seq.shape[0]))

        self._offsets = [0]
        for count in self._row_counts:
            self._offsets.append(self._offsets[-1] + count)
        if self._offsets[-1] != self.total_samples:
            raise DataLoadingError(
                f"Total samples mismatch in {split_dir}: metadata={self.total_samples}, shards={self._offsets[-1]}"
            )

        selected_mode = self.loading.mode
        if selected_mode == "auto":
            available = _estimate_available_ram_bytes()
            if available <= 0:
                raise DataLoadingError(
                    "Unable to estimate available RAM for mode='auto'. Set training.data_loading.mode to 'ram' or 'disk'."
                )
            safe_available = int(available * self.loading.ram_safety_fraction)
            required = sum(
                path.stat().st_size
                for path in [*self.seq_paths, *self.glb_paths, *self.tgt_paths, *self.dt_paths]
            )
            selected_mode = "ram" if required <= safe_available else "disk"

        if selected_mode not in {"ram", "disk"}:
            raise DataLoadingError(f"Unsupported loading mode '{selected_mode}'.")
        self.mode = selected_mode

        self._ram_seq: np.ndarray | None = None
        self._ram_glb: np.ndarray | None = None
        self._ram_tgt: np.ndarray | None = None
        self._ram_dt: np.ndarray | None = None
        self._cache: OrderedDict[int, dict[str, Any]] | None = None
        self._cache_size = min(self.loading.max_cached_shards, self.num_shards)
        if self._cache_size <= 0:
            raise DataLoadingError("training.data_loading.max_cached_shards must be > 0.")
        self._fixed_padding_mask = torch.zeros((self.sequence_length,), dtype=torch.bool)

        if self.mode == "ram":
            self._load_all_to_ram()
        else:
            self._cache = OrderedDict()

    def _load_all_to_ram(self) -> None:
        seq, glb, tgt, dt, _ = load_full_split_arrays(self.split_dir)
        self._ram_seq = seq
        self._ram_glb = glb
        self._ram_tgt = tgt
        self._ram_dt = dt

    def _load_shard(self, shard_idx: int) -> dict[str, Any]:
        if self._cache is None:
            raise DataLoadingError("Disk cache is not initialized.")
        cached = self._cache.pop(shard_idx, None)
        if cached is not None:
            self._cache[shard_idx] = cached
            return cached

        seq_path = self.seq_paths[shard_idx]
        glb_path = self.glb_paths[shard_idx]
        tgt_path = self.tgt_paths[shard_idx]
        dt_path = self.dt_paths[shard_idx]
        use_mmap = seq_path.stat().st_size >= self.loading.large_shard_mmap_bytes
        mmap_mode = "r" if use_mmap else None
        shard = {
            "sequence": np.load(seq_path, mmap_mode=mmap_mode, allow_pickle=False),
            "globals": np.load(glb_path, mmap_mode=mmap_mode, allow_pickle=False),
            "targets": np.load(tgt_path, mmap_mode=mmap_mode, allow_pickle=False),
            "dt_s": np.load(dt_path, mmap_mode=mmap_mode, allow_pickle=False),
            "mmap_backed": bool(use_mmap),
        }
        if (
            np.any(~np.isfinite(shard["sequence"]))
            or np.any(~np.isfinite(shard["globals"]))
            or np.any(~np.isfinite(shard["targets"]))
            or np.any(~np.isfinite(shard["dt_s"]))
        ):
            raise DataLoadingError(f"Non-finite values found in split data: {self.split_dir}")
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[shard_idx] = shard
        return shard

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range for dataset size {self.total_samples}.")

        if self.mode == "ram":
            if self._ram_seq is None or self._ram_glb is None or self._ram_tgt is None or self._ram_dt is None:
                raise DataLoadingError("RAM mode arrays are not initialized.")
            seq_np = self._ram_seq[idx]
            glb_np = self._ram_glb[idx]
            tgt_np = self._ram_tgt[idx]
            dt_np = self._ram_dt[idx]
        else:
            shard_idx = bisect.bisect_right(self._offsets, idx) - 1
            within = idx - self._offsets[shard_idx]
            shard = self._load_shard(shard_idx)
            seq_np = shard["sequence"][within]
            glb_np = shard["globals"][within]
            tgt_np = shard["targets"][within]
            dt_np = shard["dt_s"][within]
            if self.loading.copy_mmap_slices and bool(shard.get("mmap_backed", False)):
                seq_np = seq_np.copy()
                glb_np = glb_np.copy()
                tgt_np = tgt_np.copy()
                dt_np = np.array(dt_np, copy=True)

        seq = torch.from_numpy(seq_np)
        glb = torch.from_numpy(glb_np)
        tgt = torch.from_numpy(tgt_np)
        dt = torch.as_tensor(dt_np)
        if seq.dtype != self.input_dtype:
            seq = seq.to(dtype=self.input_dtype)
        if glb.dtype != self.input_dtype:
            glb = glb.to(dtype=self.input_dtype)
        if tgt.dtype != self.target_dtype:
            tgt = tgt.to(dtype=self.target_dtype)
        if dt.dtype != self.target_dtype:
            dt = dt.to(dtype=self.target_dtype)
        return seq, glb, tgt, self._fixed_padding_mask, dt


class DevicePrefetchLoader:
    """Asynchronous CUDA prefetch wrapper for `(seq, glb, tgt, mask, dt)` batches."""

    prefetches_to_device = True

    def __init__(
        self,
        loader: DataLoader,
        *,
        device: torch.device,
        forward_dtype: torch.dtype,
        loss_dtype: torch.dtype,
    ) -> None:
        self.loader = loader
        self.device = device
        self.forward_dtype = forward_dtype
        self.loss_dtype = loss_dtype
        if device.type != "cuda":
            raise DataLoadingError("DevicePrefetchLoader requires a CUDA device.")

    def _to_device(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq, glb, tgt, mask, dt = batch
        seq = seq.to(device=self.device, dtype=self.forward_dtype, non_blocking=True)
        glb = glb.to(device=self.device, dtype=self.forward_dtype, non_blocking=True)
        tgt = tgt.to(device=self.device, dtype=self.loss_dtype, non_blocking=True)
        mask = mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        dt = dt.to(device=self.device, dtype=self.loss_dtype, non_blocking=True)
        return seq, glb, tgt, mask, dt

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True
        current = None
        for next_batch in self.loader:
            with torch.cuda.stream(stream):
                next_ready = self._to_device(next_batch)
            if not first:
                yield current
            else:
                first = False
            torch.cuda.current_stream().wait_stream(stream)
            current = next_ready
        if current is not None:
            yield current

    def __len__(self) -> int:
        return len(self.loader)


def build_training_loader(
    *,
    split_dir: Path,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    preload_to_device: bool,
    num_workers: int,
    input_dtype: torch.dtype,
    target_dtype: torch.dtype,
    loading: DataLoadingConfig,
) -> tuple[DataLoader | DevicePrefetchLoader, dict[str, Any]]:
    """Build the configured loader variant for one processed split."""
    metadata = load_split_metadata(split_dir)
    if preload_to_device:
        if num_workers != 0:
            raise DataLoadingError("training.gpu_preload=true requires training.num_workers=0.")
        seq_np, glb_np, tgt_np, dt_np, _ = load_full_split_arrays(split_dir)
        seq = _to_tensor(seq_np, dtype=input_dtype)
        glb = _to_tensor(glb_np, dtype=input_dtype)
        tgt = _to_tensor(tgt_np, dtype=target_dtype)
        dt = _to_tensor(dt_np.astype(np.float32, copy=False), dtype=target_dtype)
        mask = torch.zeros((seq.shape[0], seq.shape[1]), dtype=torch.bool)
        try:
            seq = seq.to(device=device, non_blocking=False)
            glb = glb.to(device=device, non_blocking=False)
            tgt = tgt.to(device=device, non_blocking=False)
            dt = dt.to(device=device, non_blocking=False)
            mask = mask.to(device=device, non_blocking=False)
        except RuntimeError as exc:
            raise DataLoadingError(
                "GPU preload failed (likely OOM). Set training.gpu_preload=false to use host-memory loading."
            ) from exc
        loader = DataLoader(
            TensorDataset(seq, glb, tgt, mask, dt),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=0,
            pin_memory=False,
        )
        return loader, metadata

    dataset = ProcessedSplitDataset(
        split_dir=split_dir,
        metadata=metadata,
        input_dtype=input_dtype,
        target_dtype=target_dtype,
        loading=loading,
    )
    base_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    if loading.use_device_prefetch and device.type == "cuda":
        return (
            DevicePrefetchLoader(
                base_loader,
                device=device,
                forward_dtype=input_dtype,
                loss_dtype=target_dtype,
            ),
            metadata,
        )
    return base_loader, metadata
