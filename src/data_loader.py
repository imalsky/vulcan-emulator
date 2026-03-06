"""Processed-split data loading utilities with explicit RAM/disk policies."""

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
    """Configurable dataset-loading policy.

    Attributes:
        mode: Loader mode in `{"auto", "ram", "disk"}`.
        max_cached_shards: Maximum number of disk-backed shards kept in the LRU cache.
        large_shard_mmap_bytes: Minimum shard size in bytes before `np.load(..., mmap_mode="r")`
            is used in disk mode.
        ram_safety_fraction: Fraction of estimated free host RAM that auto mode is allowed to use.
        copy_mmap_slices: Whether per-sample slices from memory-mapped shards are copied before
            Torch conversion.
        use_device_prefetch: Whether to wrap a host loader with asynchronous CUDA prefetching.
    """

    mode: str
    max_cached_shards: int
    large_shard_mmap_bytes: int
    ram_safety_fraction: float
    copy_mmap_slices: bool
    use_device_prefetch: bool


def load_split_metadata(split_dir: Path) -> dict[str, Any]:
    """Load and validate one processed-split metadata file.

    Args:
        split_dir: Directory containing `metadata.json` for one processed split.

    Returns:
        Metadata dictionary describing split sizes, feature ordering, and normalization
        fingerprint values. Integer fields remain Python `int` values.
    """
    metadata_path = split_dir / "metadata.json"
    if not metadata_path.is_file():
        raise DataLoadingError(f"Missing split metadata: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    for key in (
        "num_shards",
        "sequence_length",
        "input_dim",
        "global_dim",
        "target_dim",
        "total_samples",
    ):
        if key not in metadata:
            raise DataLoadingError(f"Missing key '{key}' in split metadata: {metadata_path}")
        value = metadata[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise DataLoadingError(
                "Invalid metadata type for "
                f"'{key}' in {metadata_path}: expected integer, "
                f"got {type(value).__name__}."
            )
        if value <= 0:
            raise DataLoadingError(
                f"Invalid metadata value for '{key}' in {metadata_path}: expected > 0, got {value}."
            )

    for key in ("sequence_feature_order", "global_feature_order", "target_species_order"):
        value = metadata.get(key)
        if not isinstance(value, list) or not value:
            raise DataLoadingError(
                f"Invalid metadata field '{key}' in {metadata_path}: expected non-empty list."
            )
        if any((not isinstance(item, str) or not item.strip()) for item in value):
            raise DataLoadingError(
                f"Invalid metadata field '{key}' in {metadata_path}: "
                "all entries must be non-empty strings."
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
            f"Invalid metadata field 'normalization_fingerprint' in {metadata_path}: must be hex."
        ) from exc

    expected_input_dim = int(metadata["input_dim"])
    if len(metadata["sequence_feature_order"]) != expected_input_dim:
        raise DataLoadingError(
            "Invalid sequence feature order length in "
            f"{metadata_path}: input_dim={expected_input_dim}, "
            f"len(sequence_feature_order)={len(metadata['sequence_feature_order'])}"
        )
    expected_global_dim = int(metadata["global_dim"])
    if len(metadata["global_feature_order"]) != expected_global_dim:
        raise DataLoadingError(
            "Invalid global feature order length in "
            f"{metadata_path}: global_dim={expected_global_dim}, "
            f"len(global_feature_order)={len(metadata['global_feature_order'])}"
        )
    expected_target_dim = int(metadata["target_dim"])
    if len(metadata["target_species_order"]) != expected_target_dim:
        raise DataLoadingError(
            "Invalid target species order length in "
            f"{metadata_path}: target_dim={expected_target_dim}, "
            f"len(target_species_order)={len(metadata['target_species_order'])}"
        )
    return metadata


def _to_tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Convert one NumPy array to Torch and cast only when needed."""
    tensor = torch.from_numpy(array)
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _estimate_available_ram_bytes() -> int:
    """Estimate available host RAM for `mode='auto'` loader selection."""
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


def _load_split_arrays(
    split_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load an entire split eagerly into host memory with contract checks.

    Args:
        split_dir: Directory containing sharded `sequence_inputs`, `globals`, and `targets`
            `.npy` files for one split.

    Returns:
        Tuple `(sequence, globals_, targets, metadata)` where:
        - `sequence` has shape `[samples, nz, input_dim]`
        - `globals_` has shape `[samples, global_dim]`
        - `targets` has shape `[samples, nz, target_dim]`
        - `metadata` is the validated split metadata dictionary
    """
    metadata = load_split_metadata(split_dir)
    num_shards = int(metadata["num_shards"])
    expected_seq_len = int(metadata["sequence_length"])
    expected_input_dim = int(metadata["input_dim"])
    expected_global_dim = int(metadata["global_dim"])
    expected_target_dim = int(metadata["target_dim"])
    seq_dir = split_dir / "sequence_inputs"
    glb_dir = split_dir / "globals"
    tgt_dir = split_dir / "targets"

    seq_parts: list[np.ndarray] = []
    glb_parts: list[np.ndarray] = []
    tgt_parts: list[np.ndarray] = []

    for shard_idx in range(num_shards):
        seq_path = seq_dir / f"shard_{shard_idx:05d}.npy"
        glb_path = glb_dir / f"shard_{shard_idx:05d}.npy"
        tgt_path = tgt_dir / f"shard_{shard_idx:05d}.npy"
        if not seq_path.is_file() or not glb_path.is_file() or not tgt_path.is_file():
            raise DataLoadingError(
                f"Missing shard file set for index {shard_idx} in {split_dir}"
            )

        seq = np.load(seq_path, allow_pickle=False)
        glb = np.load(glb_path, allow_pickle=False)
        tgt = np.load(tgt_path, allow_pickle=False)

        if seq.ndim != 3 or glb.ndim != 2 or tgt.ndim != 3:
            raise DataLoadingError(
                f"Unexpected shard ranks in split {split_dir}, shard {shard_idx}"
            )
        if not (seq.shape[0] == glb.shape[0] == tgt.shape[0]):
            raise DataLoadingError(
                f"Shard sample-count mismatch in split {split_dir}, shard {shard_idx}"
            )
        if seq.shape[1:] != (expected_seq_len, expected_input_dim):
            raise DataLoadingError(
                "Sequence shard shape mismatch in "
                f"{split_dir}, shard {shard_idx}: expected "
                f"(*,{expected_seq_len},{expected_input_dim}), got {seq.shape}"
            )
        if glb.shape[1] != expected_global_dim:
            raise DataLoadingError(
                "Global shard shape mismatch in "
                f"{split_dir}, shard {shard_idx}: expected "
                f"(*,{expected_global_dim}), got {glb.shape}"
            )
        if tgt.shape[1:] != (expected_seq_len, expected_target_dim):
            raise DataLoadingError(
                "Target shard shape mismatch in "
                f"{split_dir}, shard {shard_idx}: expected "
                f"(*,{expected_seq_len},{expected_target_dim}), got {tgt.shape}"
            )

        seq_parts.append(seq)
        glb_parts.append(glb)
        tgt_parts.append(tgt)

    if not seq_parts:
        raise DataLoadingError(f"No shard data found for split: {split_dir}")

    sequence = np.concatenate(seq_parts, axis=0)
    globals_ = np.concatenate(glb_parts, axis=0)
    targets = np.concatenate(tgt_parts, axis=0)
    expected_total = int(metadata["total_samples"])
    if sequence.shape[0] != expected_total:
        raise DataLoadingError(
            "Split sample count mismatch in "
            f"{split_dir}: metadata={expected_total}, loaded={sequence.shape[0]}"
        )
    if sequence.shape[1:] != (expected_seq_len, expected_input_dim):
        raise DataLoadingError(
            "Loaded sequence shape mismatch in "
            f"{split_dir}: expected (*,{expected_seq_len},{expected_input_dim}), "
            f"got {sequence.shape}"
        )
    if globals_.shape[1] != expected_global_dim:
        raise DataLoadingError(
            "Loaded globals shape mismatch in "
            f"{split_dir}: expected (*,{expected_global_dim}), got {globals_.shape}"
        )
    if targets.shape[1:] != (expected_seq_len, expected_target_dim):
        raise DataLoadingError(
            "Loaded targets shape mismatch in "
            f"{split_dir}: expected (*,{expected_seq_len},{expected_target_dim}), "
            f"got {targets.shape}"
        )
    if (
        np.any(~np.isfinite(sequence))
        or np.any(~np.isfinite(globals_))
        or np.any(~np.isfinite(targets))
    ):
        raise DataLoadingError(f"Non-finite values found in processed split: {split_dir}")
    return sequence, globals_, targets, metadata


class ProcessedSplitDataset(Dataset):
    """Sharded dataset with explicit RAM/disk/auto loading behavior.

    Each sample is returned as a tuple of Torch tensors:
    - `sequence`: shape `[nz, input_dim]`
    - `globals`: shape `[global_dim]`
    - `targets`: shape `[nz, target_dim]`
    - `padding_mask`: shape `[nz]`

    Padding convention:
    - `True` in the returned mask means "padding position".
    - v1 shards are fixed-length; returned masks are all `False` unless a future
      variable-length preprocessing path is added.
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
        """Prepare either RAM-backed or shard-backed access for one split."""
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

        self.seq_paths = [
            split_dir / "sequence_inputs" / f"shard_{idx:05d}.npy"
            for idx in range(self.num_shards)
        ]
        self.glb_paths = [
            split_dir / "globals" / f"shard_{idx:05d}.npy" for idx in range(self.num_shards)
        ]
        self.tgt_paths = [
            split_dir / "targets" / f"shard_{idx:05d}.npy"
            for idx in range(self.num_shards)
        ]

        for seq_path, glb_path, tgt_path in zip(
            self.seq_paths, self.glb_paths, self.tgt_paths, strict=True
        ):
            if not seq_path.is_file() or not glb_path.is_file() or not tgt_path.is_file():
                raise DataLoadingError(
                    f"Missing shard file set: {seq_path.name}, {glb_path.name}, {tgt_path.name}"
                )

        self._row_counts: list[int] = []
        for idx, (seq_path, glb_path, tgt_path) in enumerate(
            zip(self.seq_paths, self.glb_paths, self.tgt_paths, strict=True)
        ):
            seq = np.load(seq_path, mmap_mode="r", allow_pickle=False)
            glb = np.load(glb_path, mmap_mode="r", allow_pickle=False)
            tgt = np.load(tgt_path, mmap_mode="r", allow_pickle=False)

            if seq.ndim != 3 or glb.ndim != 2 or tgt.ndim != 3:
                raise DataLoadingError(f"Unexpected shard ranks in {split_dir} shard {idx}.")
            if seq.shape[0] != glb.shape[0] or seq.shape[0] != tgt.shape[0]:
                raise DataLoadingError(f"Shard sample-count mismatch in {split_dir} shard {idx}.")
            if seq.shape[1:] != (self.sequence_length, self.input_dim):
                raise DataLoadingError(
                    f"Sequence shape mismatch in {split_dir} shard {idx}: {seq.shape[1:]}"
                )
            if glb.shape[1] != self.global_dim:
                raise DataLoadingError(
                    f"Global shape mismatch in {split_dir} shard {idx}: {glb.shape[1]}"
                )
            if tgt.shape[1:] != (self.sequence_length, self.target_dim):
                raise DataLoadingError(
                    f"Target shape mismatch in {split_dir} shard {idx}: {tgt.shape[1:]}"
                )
            self._row_counts.append(int(seq.shape[0]))

        self._offsets = [0]
        for count in self._row_counts:
            self._offsets.append(self._offsets[-1] + count)
        if self._offsets[-1] != self.total_samples:
            raise DataLoadingError(
                f"Total samples mismatch in {split_dir}: metadata={self.total_samples}, "
                f"shards={self._offsets[-1]}"
            )

        selected_mode = self.loading.mode
        if selected_mode == "auto":
            available = _estimate_available_ram_bytes()
            if available <= 0:
                raise DataLoadingError(
                    "Unable to estimate available RAM for mode='auto'. "
                    "Set training.data_loading.mode to 'ram' or 'disk'."
                )
            safe_available = int(available * self.loading.ram_safety_fraction)
            required = sum(
                path.stat().st_size for path in [*self.seq_paths, *self.glb_paths, *self.tgt_paths]
            )
            selected_mode = "ram" if required <= safe_available else "disk"

        if selected_mode not in {"ram", "disk"}:
            raise DataLoadingError(
                f"Unsupported loading mode '{selected_mode}'. Use one of ['auto', 'ram', 'disk']."
            )
        self.mode = selected_mode

        self._ram_seq: np.ndarray | None = None
        self._ram_glb: np.ndarray | None = None
        self._ram_tgt: np.ndarray | None = None
        self._cache: OrderedDict[int, dict[str, Any]] | None = None
        self._cache_size = min(self.loading.max_cached_shards, self.num_shards)
        if self._cache_size <= 0:
            raise DataLoadingError("training.data_loading.max_cached_shards must be > 0.")

        # Pre-allocate a single all-False mask since v1 shards are fixed-length
        # (no actual padding). Shared across all __getitem__ calls to avoid
        # repeated tensor allocation.
        self._fixed_padding_mask = torch.zeros((self.sequence_length,), dtype=torch.bool)

        if self.mode == "ram":
            self._load_all_to_ram()
        else:
            self._cache = OrderedDict()

    def _load_all_to_ram(self) -> None:
        """Load all shard arrays into concatenated in-memory buffers."""
        seq_parts: list[np.ndarray] = []
        glb_parts: list[np.ndarray] = []
        tgt_parts: list[np.ndarray] = []
        for seq_path, glb_path, tgt_path in zip(
            self.seq_paths, self.glb_paths, self.tgt_paths, strict=True
        ):
            seq = np.load(seq_path, allow_pickle=False)
            glb = np.load(glb_path, allow_pickle=False)
            tgt = np.load(tgt_path, allow_pickle=False)
            if np.any(~np.isfinite(seq)) or np.any(~np.isfinite(glb)) or np.any(~np.isfinite(tgt)):
                raise DataLoadingError(f"Non-finite values found in split data: {self.split_dir}")
            seq_parts.append(seq)
            glb_parts.append(glb)
            tgt_parts.append(tgt)

        self._ram_seq = np.concatenate(seq_parts, axis=0)
        self._ram_glb = np.concatenate(glb_parts, axis=0)
        self._ram_tgt = np.concatenate(tgt_parts, axis=0)

    def _load_shard(self, shard_idx: int) -> dict[str, Any]:
        """Load or retrieve one shard from the small LRU cache."""
        if self._cache is None:
            raise DataLoadingError("Disk cache is not initialized.")
        cached = self._cache.pop(shard_idx, None)
        if cached is not None:
            self._cache[shard_idx] = cached
            return cached

        seq_path = self.seq_paths[shard_idx]
        glb_path = self.glb_paths[shard_idx]
        tgt_path = self.tgt_paths[shard_idx]
        use_mmap = seq_path.stat().st_size >= self.loading.large_shard_mmap_bytes
        mmap_mode = "r" if use_mmap else None
        shard = {
            "sequence": np.load(seq_path, mmap_mode=mmap_mode, allow_pickle=False),
            "globals": np.load(glb_path, mmap_mode=mmap_mode, allow_pickle=False),
            "targets": np.load(tgt_path, mmap_mode=mmap_mode, allow_pickle=False),
            "mmap_backed": bool(use_mmap),
        }
        if (
            np.any(~np.isfinite(shard["sequence"]))
            or np.any(~np.isfinite(shard["globals"]))
            or np.any(~np.isfinite(shard["targets"]))
        ):
            raise DataLoadingError(f"Non-finite values found in split data: {self.split_dir}")

        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[shard_idx] = shard
        return shard

    def __len__(self) -> int:
        """Return the total number of samples in the split."""
        return self.total_samples

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one sample tuple plus the fixed-length padding mask.

        Args:
            idx: Zero-based sample index in `[0, total_samples)`.

        Returns:
            Tuple `(seq, glb, tgt, padding_mask)` with shapes `[nz, input_dim]`,
            `[global_dim]`, `[nz, target_dim]`, and `[nz]`.
        """
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range for dataset size {self.total_samples}.")

        if self.mode == "ram":
            if self._ram_seq is None or self._ram_glb is None or self._ram_tgt is None:
                raise DataLoadingError("RAM mode arrays are not initialized.")
            seq_np = self._ram_seq[idx]
            glb_np = self._ram_glb[idx]
            tgt_np = self._ram_tgt[idx]
        else:
            shard_idx = bisect.bisect_right(self._offsets, idx) - 1
            within = idx - self._offsets[shard_idx]
            shard = self._load_shard(shard_idx)
            seq_np = shard["sequence"][within]
            glb_np = shard["globals"][within]
            tgt_np = shard["targets"][within]
            if self.loading.copy_mmap_slices and bool(shard.get("mmap_backed", False)):
                seq_np = seq_np.copy()
                glb_np = glb_np.copy()
                tgt_np = tgt_np.copy()

        seq = torch.from_numpy(seq_np)
        glb = torch.from_numpy(glb_np)
        tgt = torch.from_numpy(tgt_np)
        if seq.dtype != self.input_dtype:
            seq = seq.to(dtype=self.input_dtype)
        if glb.dtype != self.input_dtype:
            glb = glb.to(dtype=self.input_dtype)
        if tgt.dtype != self.target_dtype:
            tgt = tgt.to(dtype=self.target_dtype)

        return seq, glb, tgt, self._fixed_padding_mask


class DevicePrefetchLoader:
    """Asynchronous CUDA prefetch wrapper for `(seq, glb, tgt, mask)` batches."""

    prefetches_to_device = True

    def __init__(
        self,
        loader: DataLoader,
        *,
        device: torch.device,
        forward_dtype: torch.dtype,
        loss_dtype: torch.dtype,
    ) -> None:
        """Wrap one host loader with asynchronous CUDA copies."""
        self.loader = loader
        self.device = device
        self.forward_dtype = forward_dtype
        self.loss_dtype = loss_dtype
        self.is_cuda = device.type == "cuda"
        if not self.is_cuda:
            raise DataLoadingError("DevicePrefetchLoader requires a CUDA device.")

    def _to_device(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move one batch tuple onto the target CUDA device."""
        seq, glb, tgt, mask = batch
        seq = seq.to(device=self.device, dtype=self.forward_dtype, non_blocking=True)
        glb = glb.to(device=self.device, dtype=self.forward_dtype, non_blocking=True)
        tgt = tgt.to(device=self.device, dtype=self.loss_dtype, non_blocking=True)
        mask = mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        return seq, glb, tgt, mask

    def __iter__(self):
        """Prefetch one batch ahead on a dedicated CUDA stream."""
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
        """Forward the underlying loader length."""
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
    """Build the configured loader variant for one processed split.

    Args:
        split_dir: Processed split directory containing shard files and `metadata.json`.
        batch_size: Number of samples per emitted batch.
        shuffle: Whether to shuffle sample order.
        device: Target Torch device for optional preload/prefetch paths.
        preload_to_device: Whether to preload the full split onto `device`.
        num_workers: DataLoader worker count for host-side loading.
        input_dtype: Torch dtype for sequence/global conditioning tensors.
        target_dtype: Torch dtype for target tensors.
        loading: Resolved data-loading policy.

    Returns:
        Tuple `(loader, metadata)` where `loader` is either a `DataLoader` or
        `DevicePrefetchLoader` yielding `(seq, glb, tgt, mask)` batches with shapes
        `[batch, nz, input_dim]`, `[batch, global_dim]`, `[batch, nz, target_dim]`,
        and `[batch, nz]`.
    """
    metadata = load_split_metadata(split_dir)
    if preload_to_device:
        if num_workers != 0:
            raise DataLoadingError("training.gpu_preload=true requires training.num_workers=0.")
        seq_np, glb_np, tgt_np, _metadata = _load_split_arrays(split_dir)
        seq = _to_tensor(seq_np, dtype=input_dtype)
        glb = _to_tensor(glb_np, dtype=input_dtype)
        tgt = _to_tensor(tgt_np, dtype=target_dtype)
        mask = torch.zeros((seq.shape[0], seq.shape[1]), dtype=torch.bool)
        try:
            seq = seq.to(device=device, non_blocking=False)
            glb = glb.to(device=device, non_blocking=False)
            tgt = tgt.to(device=device, non_blocking=False)
            mask = mask.to(device=device, non_blocking=False)
        except RuntimeError as exc:
            raise DataLoadingError(
                "GPU preload failed (likely OOM). Set training.gpu_preload=false "
                "to use host-memory loading."
            ) from exc

        loader = DataLoader(
            TensorDataset(seq, glb, tgt, mask),
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
