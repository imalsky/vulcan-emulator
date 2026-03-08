"""Fixed-requested-dt sampling from saved VULCAN trajectories.

The bundled VULCAN solver advances with adaptive internal timesteps and saves
states at irregular times.  For the current emulator regime we intentionally
train on a *single requested jump size* after a configurable late-time cutoff.
Each requested jump is snapped to the nearest saved future solver snapshot, and
that snapped ``actual_dt_s`` is what the model ultimately sees and predicts.

This gives a clean operating regime for the surrogate while preserving modest
``dt`` variation from the solver's irregular save cadence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class TransitionSamplingError(ValueError):
    """Raised when transition-pair sampling cannot satisfy the configured contract."""


@dataclass(frozen=True)
class TransitionPairs:
    """A batch of anchor/target trajectory pairs.

    Attributes:
        anchor_index: Time-axis indices into the saved trajectory for anchors.
        target_index: Time-axis indices for targets (always > ``anchor_index``).
        requested_dt_s: Requested time jump before snapping to a saved state.
        actual_dt_s: True jump after snapping to the nearest saved future state.
        anchor_time_s: Absolute simulation times of anchor snapshots.
        target_time_s: Absolute simulation times of target snapshots.
    """

    anchor_index: np.ndarray
    target_index: np.ndarray
    requested_dt_s: np.ndarray
    actual_dt_s: np.ndarray
    anchor_time_s: np.ndarray
    target_time_s: np.ndarray


@dataclass(frozen=True)
class _CandidatePairs:
    """Deterministic fixed-dt pair mapping for every valid anchor."""

    anchor_index: np.ndarray
    target_index: np.ndarray
    requested_dt_s: np.ndarray
    actual_dt_s: np.ndarray
    anchor_time_s: np.ndarray
    target_time_s: np.ndarray
    relative_dt_error: np.ndarray


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


def _resolved_anchor_time_threshold(
    times_s: np.ndarray,
    *,
    post_equilibrium_time_min_s: float,
    post_equilibrium_min_fraction_of_final_time: float,
) -> float:
    """Return the effective anchor-time threshold for late-time sampling."""
    absolute_min = float(post_equilibrium_time_min_s)
    if absolute_min < 0.0:
        raise TransitionSamplingError("post_equilibrium_time_min_s must be >= 0.")

    min_fraction = float(post_equilibrium_min_fraction_of_final_time)
    if not (0.0 <= min_fraction < 1.0):
        raise TransitionSamplingError(
            "post_equilibrium_min_fraction_of_final_time must be in [0, 1)."
        )

    return max(absolute_min, min_fraction * float(times_s[-1]))


def _candidate_pairs(
    *,
    times_s: np.ndarray,
    fixed_requested_dt_s: float,
    post_equilibrium_time_min_s: float,
    post_equilibrium_min_fraction_of_final_time: float,
    min_future_saved_steps: int,
    max_target_relative_dt_error: float | None,
) -> _CandidatePairs:
    """Build the deterministic fixed-dt anchor->target mapping for one trajectory."""
    times = _validate_times(times_s)
    requested_dt_s = float(fixed_requested_dt_s)
    if requested_dt_s <= 0.0:
        raise TransitionSamplingError("fixed_requested_dt_s must be > 0.")
    if min_future_saved_steps < 1:
        raise TransitionSamplingError("min_future_saved_steps must be >= 1.")
    if max_target_relative_dt_error is not None and float(max_target_relative_dt_error) < 0.0:
        raise TransitionSamplingError("max_target_relative_dt_error must be >= 0 when provided.")

    threshold_time_s = _resolved_anchor_time_threshold(
        times,
        post_equilibrium_time_min_s=post_equilibrium_time_min_s,
        post_equilibrium_min_fraction_of_final_time=post_equilibrium_min_fraction_of_final_time,
    )

    max_anchor_index = times.size - int(min_future_saved_steps)
    if max_anchor_index <= 0:
        raise TransitionSamplingError("Trajectory does not contain enough future saved states.")

    anchor_index = np.arange(max_anchor_index, dtype=np.int64)
    anchor_time_s = times[anchor_index]
    valid_anchor_mask = anchor_time_s >= threshold_time_s
    if not np.any(valid_anchor_mask):
        raise TransitionSamplingError(
            "No saved trajectory states satisfy the configured post-equilibrium anchor threshold."
        )

    anchor_index = anchor_index[valid_anchor_mask]
    anchor_time_s = anchor_time_s[valid_anchor_mask]
    min_target_index = anchor_index + int(min_future_saved_steps)

    requested_target_time_s = anchor_time_s + requested_dt_s
    right_index = np.searchsorted(times, requested_target_time_s, side="left")
    right_index = np.clip(right_index, min_target_index, times.size - 1)
    left_index = np.maximum(right_index - 1, min_target_index)

    left_delta = np.abs(times[left_index] - requested_target_time_s)
    right_delta = np.abs(times[right_index] - requested_target_time_s)
    choose_left = left_delta <= right_delta
    target_index = np.where(choose_left, left_index, right_index).astype(np.int64)
    target_index = np.maximum(target_index, min_target_index)

    target_time_s = times[target_index]
    actual_dt_s = target_time_s - anchor_time_s
    if np.any(actual_dt_s <= 0.0):
        raise TransitionSamplingError("Sampled non-positive actual dt after target selection.")

    requested_dt_block = np.full(anchor_index.shape, requested_dt_s, dtype=np.float64)
    relative_dt_error = np.abs(actual_dt_s - requested_dt_block) / requested_dt_block
    if max_target_relative_dt_error is not None:
        max_relative_error = float(max_target_relative_dt_error)
        keep = relative_dt_error <= max_relative_error
        if not np.any(keep):
            raise TransitionSamplingError(
                "No candidate fixed-dt pairs satisfy max_target_relative_dt_error. "
                "Increase snapshot density, relax the tolerance, lower the requested dt, "
                "or lower the post-equilibrium anchor threshold."
            )
        anchor_index = anchor_index[keep]
        target_index = target_index[keep]
        requested_dt_block = requested_dt_block[keep]
        actual_dt_s = actual_dt_s[keep]
        anchor_time_s = anchor_time_s[keep]
        target_time_s = target_time_s[keep]
        relative_dt_error = relative_dt_error[keep]

    if anchor_index.size == 0:
        raise TransitionSamplingError("No valid fixed-dt transition pairs remain after filtering.")

    return _CandidatePairs(
        anchor_index=anchor_index,
        target_index=target_index,
        requested_dt_s=requested_dt_block,
        actual_dt_s=actual_dt_s,
        anchor_time_s=anchor_time_s,
        target_time_s=target_time_s,
        relative_dt_error=relative_dt_error,
    )


def sample_transition_pairs(
    *,
    rng: np.random.Generator,
    times_s: np.ndarray,
    n_pairs: int,
    fixed_requested_dt_s: float,
    post_equilibrium_time_min_s: float,
    post_equilibrium_min_fraction_of_final_time: float = 0.0,
    min_future_saved_steps: int = 1,
    max_target_relative_dt_error: float | None = None,
) -> TransitionPairs:
    """Sample fixed-requested-dt transition pairs from one saved trajectory.

    The requested jump size is constant for all examples.  Anchors are sampled
    uniformly from the subset of saved late-time states that admit a valid
    snapped future target.
    """
    if n_pairs <= 0:
        raise TransitionSamplingError("n_pairs must be > 0.")

    candidates = _candidate_pairs(
        times_s=times_s,
        fixed_requested_dt_s=fixed_requested_dt_s,
        post_equilibrium_time_min_s=post_equilibrium_time_min_s,
        post_equilibrium_min_fraction_of_final_time=post_equilibrium_min_fraction_of_final_time,
        min_future_saved_steps=min_future_saved_steps,
        max_target_relative_dt_error=max_target_relative_dt_error,
    )

    choice = rng.choice(candidates.anchor_index.size, size=int(n_pairs), replace=True).astype(np.int64)
    anchor_index = candidates.anchor_index[choice]
    target_index = candidates.target_index[choice]
    requested_dt_s = candidates.requested_dt_s[choice]
    actual_dt_s = candidates.actual_dt_s[choice]
    anchor_time_s = candidates.anchor_time_s[choice]
    target_time_s = candidates.target_time_s[choice]

    order = np.lexsort((target_index, anchor_index))
    return TransitionPairs(
        anchor_index=anchor_index[order],
        target_index=target_index[order],
        requested_dt_s=requested_dt_s[order],
        actual_dt_s=actual_dt_s[order],
        anchor_time_s=anchor_time_s[order],
        target_time_s=target_time_s[order],
    )


def build_rollout_indices(
    *,
    times_s: np.ndarray,
    fixed_requested_dt_s: float,
    post_equilibrium_time_min_s: float,
    post_equilibrium_min_fraction_of_final_time: float = 0.0,
    min_future_saved_steps: int = 1,
    max_target_relative_dt_error: float | None = None,
    max_points: int,
) -> np.ndarray:
    """Build one deterministic fixed-dt rollout path through a saved trajectory.

    The rollout starts at the earliest valid late-time anchor and repeatedly
    applies the same fixed requested dt, snapping to the nearest saved future
    state at each step.  Returned indices always remain strictly increasing.
    """
    if max_points < 2:
        raise TransitionSamplingError("max_points must be >= 2 for rollout evaluation.")

    candidates = _candidate_pairs(
        times_s=times_s,
        fixed_requested_dt_s=fixed_requested_dt_s,
        post_equilibrium_time_min_s=post_equilibrium_time_min_s,
        post_equilibrium_min_fraction_of_final_time=post_equilibrium_min_fraction_of_final_time,
        min_future_saved_steps=min_future_saved_steps,
        max_target_relative_dt_error=max_target_relative_dt_error,
    )

    mapping = {
        int(anchor_idx): int(target_idx)
        for anchor_idx, target_idx in zip(candidates.anchor_index, candidates.target_index, strict=True)
    }
    start_index = int(np.min(candidates.anchor_index))
    indices = [start_index]
    current_index = start_index
    while len(indices) < int(max_points):
        next_index = mapping.get(current_index)
        if next_index is None or next_index <= current_index:
            break
        indices.append(int(next_index))
        current_index = int(next_index)

    return np.asarray(indices, dtype=np.int64)
