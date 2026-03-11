"""GPU-resident live transition-pair sampling from normalized trajectory splits.

Loads processed split arrays onto the target device once, builds a candidate
table of all valid (anchor, target) index pairs, and provides per-epoch
resampling for training and deterministic fixed sampling for evaluation.
Batch assembly happens entirely on-device with no host-to-device copies in
the training hot path.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_loader import ProcessedSplitArrays, load_processed_split_arrays
from transition_sampling import TransitionSamplingError, build_candidate_pairs

logger = logging.getLogger(__name__)


class LiveSamplingError(RuntimeError):
    """Raised when live-sampling setup or batch construction fails."""


@dataclass(frozen=True)
class CandidateTable:
    """All valid transition candidates for one processed split."""

    run_index: torch.Tensor
    anchor_index: torch.Tensor
    target_index: torch.Tensor
    actual_dt_s: torch.Tensor
    normalized_log10_dt_s: torch.Tensor
    sampling_weights: torch.Tensor
    per_run_offsets: torch.Tensor
    per_run_counts: torch.Tensor

    def __len__(self) -> int:
        return int(self.run_index.numel())


class LivePairBatchLoader:
    """Simple iterable over already-device-resident live-sampled transition batches."""

    prefetches_to_device = True

    def __init__(
        self,
        *,
        store: "ProcessedTrajectoryStore",
        selected_candidate_indices: torch.Tensor,
        batch_size: int,
    ) -> None:
        if batch_size <= 0:
            raise LiveSamplingError("batch_size must be > 0.")
        self.store = store
        self.selected_candidate_indices = selected_candidate_indices
        self.batch_size = int(batch_size)
        self._full_padding_mask = torch.zeros(
            (self.batch_size, self.store.sequence_length),
            dtype=torch.bool,
            device=self.store.device,
        )

    def __len__(self) -> int:
        return max(1, math.ceil(int(self.selected_candidate_indices.numel()) / self.batch_size))

    def __iter__(self):
        total = int(self.selected_candidate_indices.numel())
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            yield self.store.build_batch(self.selected_candidate_indices[start:end], self._full_padding_mask)


class ProcessedTrajectoryStore:
    """One processed split loaded into memory/device for live candidate sampling.

    Holds normalized trajectory tensors and a precomputed candidate table on the
    target device.  Provides train-epoch resampling (with log-uniform weights)
    and fixed eval sampling (deterministic seed, sampled once and reused).
    """

    def __init__(
        self,
        *,
        split_arrays: ProcessedSplitArrays,
        normalization_metadata: dict[str, Any],
        config: dict[str, Any],
        device: torch.device,
        tensor_dtype: torch.dtype,
    ) -> None:
        self.metadata = split_arrays.metadata
        self.device = device
        self.sequence_length = int(self.metadata["sequence_length"])
        self.global_static_dim = int(self.metadata["global_static_dim"])
        self.dt_feature_index = int(self.metadata["dt_feature_index"])
        self.output_from_state_indices = torch.as_tensor(
            self.metadata["output_from_state_indices"],
            dtype=torch.long,
            device=device,
        )

        self.static_inputs = torch.as_tensor(
            split_arrays.static_inputs,
            dtype=tensor_dtype,
            device=device,
        )
        self.state_ymix = torch.as_tensor(
            split_arrays.state_ymix,
            dtype=tensor_dtype,
            device=device,
        )
        self.global_inputs = torch.as_tensor(
            split_arrays.global_inputs,
            dtype=tensor_dtype,
            device=device,
        )
        self.run_ids = torch.as_tensor(split_arrays.run_ids, dtype=torch.long, device=device)
        self.candidates = _build_candidate_table(
            split_arrays=split_arrays,
            normalization_metadata=normalization_metadata,
            config=config,
            device=device,
        )
        self._warned_shortfall_runs: set[int] = set()

    @classmethod
    def from_split_dir(
        cls,
        *,
        split_dir: Path,
        normalization_metadata: dict[str, Any],
        config: dict[str, Any],
        device: torch.device,
        tensor_dtype: torch.dtype,
    ) -> "ProcessedTrajectoryStore":
        return cls(
            split_arrays=load_processed_split_arrays(split_dir),
            normalization_metadata=normalization_metadata,
            config=config,
            device=device,
            tensor_dtype=tensor_dtype,
        )

    def selected_pairs_per_epoch(self, pairs_per_run: int) -> int:
        return int(
            sum(min(int(count), int(pairs_per_run)) for count in self.candidates.per_run_counts.tolist())
        )

    def select_train_candidate_indices(self, *, pairs_per_run: int, seed: int) -> torch.Tensor:
        return self._sample_candidate_indices(
            pairs_per_run=pairs_per_run,
            seed=seed,
            warn_shortfall=True,
        )

    def select_fixed_candidate_indices(self, *, pairs_per_run: int, seed: int) -> torch.Tensor:
        return self._sample_candidate_indices(
            pairs_per_run=pairs_per_run,
            seed=seed,
            warn_shortfall=False,
        )

    def build_train_epoch_loader(
        self,
        *,
        batch_size: int,
        pairs_per_run: int,
        seed: int,
    ) -> LivePairBatchLoader:
        return LivePairBatchLoader(
            store=self,
            selected_candidate_indices=self.select_train_candidate_indices(
                pairs_per_run=pairs_per_run,
                seed=seed,
            ),
            batch_size=batch_size,
        )

    def build_fixed_loader(
        self,
        *,
        batch_size: int,
        pairs_per_run: int,
        seed: int,
    ) -> LivePairBatchLoader:
        return LivePairBatchLoader(
            store=self,
            selected_candidate_indices=self.select_fixed_candidate_indices(
                pairs_per_run=pairs_per_run,
                seed=seed,
            ),
            batch_size=batch_size,
        )

    def build_batch(
        self,
        selected_candidate_indices: torch.Tensor,
        full_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble one batch of (sequence, globals, target, padding_mask, dt_s) on device.

        Gathers static profiles + anchor state into the sequence tensor, inserts
        the normalized log10(dt) into the global vector at ``dt_feature_index``,
        and extracts the target output-species subset from the target time step.
        """
        candidate_ids = selected_candidate_indices.to(device=self.device, dtype=torch.long)
        run_idx = self.candidates.run_index[candidate_ids]
        anchor_idx = self.candidates.anchor_index[candidate_ids]
        target_idx = self.candidates.target_index[candidate_ids]
        normalized_dt = self.candidates.normalized_log10_dt_s[candidate_ids]
        actual_dt_s = self.candidates.actual_dt_s[candidate_ids]

        static_block = self.static_inputs[run_idx]
        anchor_state = self.state_ymix[run_idx, anchor_idx]
        target_state = self.state_ymix[run_idx, target_idx]
        target = torch.index_select(target_state, dim=-1, index=self.output_from_state_indices)

        if self.dt_feature_index == 0:
            globals_ = torch.cat([normalized_dt.unsqueeze(-1), self.global_inputs[run_idx]], dim=-1)
        elif self.dt_feature_index == self.global_static_dim:
            globals_ = torch.cat([self.global_inputs[run_idx], normalized_dt.unsqueeze(-1)], dim=-1)
        else:
            globals_ = torch.cat(
                [
                    self.global_inputs[run_idx, : self.dt_feature_index],
                    normalized_dt.unsqueeze(-1),
                    self.global_inputs[run_idx, self.dt_feature_index :],
                ],
                dim=-1,
            )

        sequence = torch.cat([static_block, anchor_state], dim=-1)
        if full_padding_mask is None:
            padding_mask = torch.zeros(
                (sequence.shape[0], self.sequence_length),
                dtype=torch.bool,
                device=self.device,
            )
        else:
            padding_mask = full_padding_mask[: sequence.shape[0]]
        return sequence, globals_, target, padding_mask, actual_dt_s

    def _sample_candidate_indices(
        self,
        *,
        pairs_per_run: int,
        seed: int,
        warn_shortfall: bool,
    ) -> torch.Tensor:
        """Sample up to ``pairs_per_run`` candidates per run using log-uniform weights.

        Samples without replacement when possible.  If a run has fewer valid
        candidates than requested, all of them are used and a warning is logged
        once.  The selected indices are globally shuffled before return.
        """
        if pairs_per_run <= 0:
            raise LiveSamplingError("pairs_per_run must be > 0.")
        generator_device = self.device.type if self.device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(int(seed))

        selected_chunks: list[torch.Tensor] = []
        for run_idx, count in enumerate(self.candidates.per_run_counts.tolist()):
            count_int = int(count)
            if count_int <= 0:
                continue
            offset = int(self.candidates.per_run_offsets[run_idx].item())
            local_indices = torch.arange(
                offset,
                offset + count_int,
                device=self.device,
                dtype=torch.long,
            )
            if count_int <= pairs_per_run:
                if warn_shortfall and run_idx not in self._warned_shortfall_runs:
                    logger.info(
                        "Split %s run %d has %d valid candidates; using all of them instead of requested %d.",
                        self.metadata["split"],
                        int(self.run_ids[run_idx].item()),
                        count_int,
                        int(pairs_per_run),
                    )
                    self._warned_shortfall_runs.add(run_idx)
                selected_chunks.append(local_indices)
                continue
            weights = self.candidates.sampling_weights[offset : offset + count_int]
            local_choice = torch.multinomial(
                weights,
                num_samples=int(pairs_per_run),
                replacement=False,
                generator=generator,
            )
            selected_chunks.append(local_indices[local_choice])

        if not selected_chunks:
            raise LiveSamplingError(f"No candidate pairs available for split '{self.metadata['split']}'.")

        selected = torch.cat(selected_chunks, dim=0)
        shuffle = torch.randperm(int(selected.numel()), generator=generator, device=self.device)
        return selected[shuffle]


def _normalize_log10_dt(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    """Apply normalization to pre-computed log10(dt) values using the stored dt stats."""
    method = str(stats["method"])
    data = np.asarray(values, dtype=np.float64)
    if method == "none":
        return data
    if method == "standard":
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        return (data - mean) / std
    raise LiveSamplingError(f"Unsupported log10_dt normalization method for live sampling: {method}")


def _build_candidate_table(
    *,
    split_arrays: ProcessedSplitArrays,
    normalization_metadata: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> CandidateTable:
    """Build the full candidate table for one split from its time arrays.

    Enumerates all valid (anchor, target) pairs per run, computes actual dt,
    normalized log10(dt), and log-uniform sampling weights (1/dt), then
    transfers everything to the target device as contiguous tensors.
    """
    sampling_cfg = config["trajectory_sampling"]
    dt_stats = normalization_metadata["globals"]["log10_dt_s"]

    run_index_parts: list[np.ndarray] = []
    anchor_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    dt_parts: list[np.ndarray] = []
    normalized_dt_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    per_run_counts: list[int] = []

    for run_idx in range(int(split_arrays.run_ids.shape[0])):
        valid_steps = int(np.sum(split_arrays.valid_steps_mask[run_idx], dtype=np.int64))
        times_s = np.asarray(split_arrays.time_s[run_idx, :valid_steps], dtype=np.float64)
        try:
            candidates = build_candidate_pairs(
                times_s=times_s,
                dt_min_s=float(sampling_cfg["dt_min_s"]),
                dt_max_s=float(sampling_cfg["dt_max_s"]),
                min_future_saved_steps=int(sampling_cfg["min_future_saved_steps"]),
            )
        except TransitionSamplingError as exc:
            raise LiveSamplingError(
                f"Processed split '{split_arrays.metadata['split']}' contains an invalid run "
                f"{int(split_arrays.run_ids[run_idx])}: {exc}"
            ) from exc

        per_run_counts.append(int(candidates.actual_dt_s.size))
        run_index_parts.append(
            np.full(candidates.actual_dt_s.shape, run_idx, dtype=np.int64)
        )
        anchor_parts.append(np.asarray(candidates.anchor_index, dtype=np.int64))
        target_parts.append(np.asarray(candidates.target_index, dtype=np.int64))
        dt_parts.append(np.asarray(candidates.actual_dt_s, dtype=np.float64))
        normalized_dt_parts.append(
            _normalize_log10_dt(np.log10(candidates.actual_dt_s).reshape(-1, 1), dt_stats).reshape(-1)
        )
        weight_parts.append(
            1.0 / np.maximum(np.asarray(candidates.actual_dt_s, dtype=np.float64), np.finfo(np.float64).tiny)
        )

    per_run_offsets = np.zeros((len(per_run_counts),), dtype=np.int64)
    if per_run_counts:
        per_run_offsets[1:] = np.cumsum(np.asarray(per_run_counts[:-1], dtype=np.int64))

    return CandidateTable(
        run_index=torch.as_tensor(np.concatenate(run_index_parts), dtype=torch.long, device=device),
        anchor_index=torch.as_tensor(np.concatenate(anchor_parts), dtype=torch.long, device=device),
        target_index=torch.as_tensor(np.concatenate(target_parts), dtype=torch.long, device=device),
        actual_dt_s=torch.as_tensor(np.concatenate(dt_parts), dtype=torch.float32, device=device),
        normalized_log10_dt_s=torch.as_tensor(
            np.concatenate(normalized_dt_parts),
            dtype=torch.float32,
            device=device,
        ),
        sampling_weights=torch.as_tensor(
            np.concatenate(weight_parts),
            dtype=torch.float64,
            device=device,
        ),
        per_run_offsets=torch.as_tensor(per_run_offsets, dtype=torch.long, device=device),
        per_run_counts=torch.as_tensor(np.asarray(per_run_counts, dtype=np.int64), dtype=torch.long, device=device),
    )
