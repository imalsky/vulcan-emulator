"""Log-uniform transition-pair sampling from saved VULCAN trajectories.

The VULCAN solver saves states at irregular times. For the current emulator
regime, every valid ordered pair of saved snapshots is a candidate training
example, subject to a configurable actual-dt range. Training pairs are sampled
approximately uniformly in log(dt) by weighting candidate pairs with 1 / dt.

This supports the intended operator:

- input: current atmospheric state + conditioning inputs + dt
- output: future atmospheric state after that dt
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class TransitionSamplingError(ValueError):
    """Raised when transition-pair sampling cannot satisfy the configured contract."""


@dataclass(frozen=True)
class TransitionPairs:
    """A batch of anchor/target trajectory pairs."""

    anchor_index: np.ndarray
    target_index: np.ndarray
    actual_dt_s: np.ndarray
    anchor_time_s: np.ndarray
    target_time_s: np.ndarray


@dataclass(frozen=True)
class _CandidatePairs:
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


def _candidate_pairs(
    *,
    times_s: np.ndarray,
    dt_min_s: float,
    dt_max_s: float,
    min_future_saved_steps: int,
) -> _CandidatePairs:
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

    return _CandidatePairs(
        anchor_index=anchor_index,
        target_index=target_index,
        actual_dt_s=actual_dt_s,
        anchor_time_s=anchor_time_s,
        target_time_s=target_time_s,
    )


def sample_transition_pairs(
    *,
    rng: np.random.Generator,
    times_s: np.ndarray,
    n_pairs: int,
    dt_min_s: float,
    dt_max_s: float,
    min_future_saved_steps: int = 1,
) -> TransitionPairs:
    """Sample up to ``n_pairs`` trajectory pairs with approximate log-uniform dt coverage.

    Uniform density in log(dt) implies p(dt) ∝ 1 / dt in linear dt space.
    We approximate this by weighting each candidate pair by 1 / actual_dt_s.
    """
    if n_pairs <= 0:
        raise TransitionSamplingError("n_pairs must be > 0.")

    candidates = _candidate_pairs(
        times_s=times_s,
        dt_min_s=dt_min_s,
        dt_max_s=dt_max_s,
        min_future_saved_steps=min_future_saved_steps,
    )
    weights = 1.0 / np.maximum(candidates.actual_dt_s, np.finfo(np.float64).tiny)
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise TransitionSamplingError("Failed to construct finite log-uniform sampling weights.")
    weights = weights / weight_sum

    sample_size = min(int(n_pairs), int(candidates.anchor_index.size))
    choice = rng.choice(candidates.anchor_index.size, size=sample_size, replace=False, p=weights)
    choice = np.asarray(choice, dtype=np.int64)

    anchor_index = candidates.anchor_index[choice]
    target_index = candidates.target_index[choice]
    actual_dt_s = candidates.actual_dt_s[choice]
    anchor_time_s = candidates.anchor_time_s[choice]
    target_time_s = candidates.target_time_s[choice]

    order = np.lexsort((target_index, anchor_index))
    return TransitionPairs(
        anchor_index=anchor_index[order],
        target_index=target_index[order],
        actual_dt_s=actual_dt_s[order],
        anchor_time_s=anchor_time_s[order],
        target_time_s=target_time_s[order],
    )


def build_rollout_indices(*, times_s: np.ndarray, max_points: int) -> np.ndarray:
    """Build a deterministic log-spaced rollout path through a saved trajectory."""
    times = _validate_times(times_s)
    if max_points < 2:
        raise TransitionSamplingError("max_points must be >= 2 for rollout evaluation.")
    if times.size <= max_points:
        return np.arange(times.size, dtype=np.int64)

    positive_times = times[1:]
    if np.any(positive_times <= 0.0):
        raise TransitionSamplingError("Saved times beyond t=0 must be strictly positive.")

    target_times = np.logspace(
        np.log10(float(positive_times[0])),
        np.log10(float(positive_times[-1])),
        int(max_points - 1),
        dtype=np.float64,
    )
    sampled = np.searchsorted(times, target_times, side="left")
    sampled = np.clip(sampled, 1, times.size - 1)
    indices = np.unique(np.concatenate([np.array([0], dtype=np.int64), sampled.astype(np.int64)]))
    if indices[-1] != times.size - 1:
        indices = np.append(indices, times.size - 1)
    if indices.size > max_points:
        interior = indices[1:-1]
        keep = np.linspace(0, max(interior.size - 1, 0), max_points - 2, dtype=np.int64)
        indices = np.concatenate(
            [
                np.array([indices[0]], dtype=np.int64),
                interior[keep] if interior.size > 0 else np.array([], dtype=np.int64),
                np.array([indices[-1]], dtype=np.int64),
            ]
        )
    return indices.astype(np.int64, copy=False)
