"""Transition-pair utilities for saved VULCAN trajectories.

The VULCAN solver saves states at irregular times. For the current emulator
regime, every valid ordered pair of saved snapshots is a candidate transition
example, subject to a configurable actual-dt range.

This supports the intended single-step operator:

- input: current atmospheric state + conditioning inputs + dt
- output: future atmospheric state after that dt
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class TransitionSamplingError(ValueError):
    """Raised when transition-pair sampling cannot satisfy the configured contract."""


@dataclass(frozen=True)
class CandidatePairs:
    """All valid ordered candidate pairs for one trajectory."""

    anchor_index: np.ndarray
    target_index: np.ndarray
    actual_dt_s: np.ndarray
    anchor_time_s: np.ndarray
    target_time_s: np.ndarray


def _validate_times(times_s: np.ndarray) -> np.ndarray:
    """Validate and return strictly increasing saved times."""
    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1 or times.size < 2:
        raise TransitionSamplingError("times_s must be a rank-1 array with at least 2 entries.")
    if np.any(~np.isfinite(times)):
        raise TransitionSamplingError("times_s must contain only finite values.")
    if np.any(np.diff(times) <= 0.0):
        raise TransitionSamplingError("times_s must be strictly increasing.")
    if float(times[0]) < 0.0:
        raise TransitionSamplingError("times_s must be non-negative.")
    return times


def build_candidate_pairs(
    *,
    times_s: np.ndarray,
    dt_min_s: float,
    dt_max_s: float,
    min_future_saved_steps: int,
) -> CandidatePairs:
    """Enumerate all valid ordered pairs within the configured actual-dt range."""
    times = _validate_times(times_s)
    dt_min = float(dt_min_s)
    dt_max = float(dt_max_s)
    if dt_min <= 0.0:
        raise TransitionSamplingError("dt_min_s must be > 0.")
    if dt_max <= dt_min:
        raise TransitionSamplingError("dt_max_s must be > dt_min_s.")
    if min_future_saved_steps < 1:
        raise TransitionSamplingError("min_future_saved_steps must be >= 1.")

    anchor_index, target_index = np.triu_indices(times.size, k=int(min_future_saved_steps))
    if anchor_index.size == 0:
        raise TransitionSamplingError("Trajectory does not contain enough future saved states.")

    anchor_time_s = times[anchor_index]
    target_time_s = times[target_index]
    actual_dt_s = target_time_s - anchor_time_s
    keep = (actual_dt_s >= dt_min) & (actual_dt_s <= dt_max)
    if not np.any(keep):
        raise TransitionSamplingError(
            "No saved trajectory pairs satisfy the configured dt range. "
            "Lower trajectory_sampling.dt_min_s, raise dt_max_s, or save denser trajectories."
        )

    anchor_index = anchor_index[keep].astype(np.int64, copy=False)
    target_index = target_index[keep].astype(np.int64, copy=False)
    actual_dt_s = actual_dt_s[keep]
    anchor_time_s = anchor_time_s[keep]
    target_time_s = target_time_s[keep]
    if np.any(actual_dt_s <= 0.0):
        raise TransitionSamplingError("Sampled non-positive actual dt.")

    return CandidatePairs(
        anchor_index=anchor_index,
        target_index=target_index,
        actual_dt_s=actual_dt_s,
        anchor_time_s=anchor_time_s,
        target_time_s=target_time_s,
    )
