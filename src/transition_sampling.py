"""Irregular-time transition-pair sampling from saved VULCAN trajectories.

VULCAN saves chemistry snapshots at irregular time intervals determined by
its adaptive solver.  This module samples (anchor, target) index pairs with
log-uniform requested time jumps, then snaps each target to the nearest
actual saved snapshot.  The log-uniform distribution ensures broad coverage
across the many orders of magnitude of chemical timescales (~1e-6 to ~1e16 s).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class TransitionSamplingError(ValueError):
    """Raised when transition-pair sampling cannot satisfy the configured contract."""


@dataclass(frozen=True)
class TransitionPairs:
    """One batch of sampled anchor/target trajectory pairs.

    Each pair maps an anchor snapshot to a future target snapshot from the
    same VULCAN trajectory.  The ``requested_dt_s`` is the ideal log-uniform
    time jump; ``actual_dt_s`` is the true dt after snapping to the nearest
    saved state.  All arrays share the same length (n_pairs).

    Attributes:
        anchor_index: Time-axis indices into the saved trajectory for anchors.
        target_index: Time-axis indices for targets (always > anchor_index).
        requested_dt_s: Ideal log-uniform dt before nearest-snapshot matching.
        actual_dt_s: True dt = target_time_s - anchor_time_s (used as labels).
        anchor_time_s: Absolute simulation times of anchor snapshots.
        target_time_s: Absolute simulation times of target snapshots.
    """

    anchor_index: np.ndarray
    target_index: np.ndarray
    requested_dt_s: np.ndarray
    actual_dt_s: np.ndarray
    anchor_time_s: np.ndarray
    target_time_s: np.ndarray


def _strictly_increasing_future_mask(times_s: np.ndarray, min_future_saved_steps: int) -> np.ndarray:
    """Return anchors that admit at least one valid future target."""
    if min_future_saved_steps < 1:
        raise TransitionSamplingError("min_future_saved_steps must be >= 1.")
    if times_s.ndim != 1 or times_s.size < 2:
        raise TransitionSamplingError("times_s must be a rank-1 array with at least 2 entries.")
    deltas = np.diff(times_s)
    if np.any(deltas <= 0.0):
        raise TransitionSamplingError("times_s must be strictly increasing.")
    valid = np.zeros(times_s.shape, dtype=bool)
    valid[: times_s.size - min_future_saved_steps] = True
    return valid


def sample_transition_pairs(
    *,
    rng: np.random.Generator,
    times_s: np.ndarray,
    n_pairs: int,
    requested_dt_min_s: float,
    requested_dt_max_s: float,
    allow_anchor_at_t0: bool,
    min_future_saved_steps: int = 1,
) -> TransitionPairs:
    """Sample log-uniform time jumps and map them to actual saved VULCAN states.

    The requested time jump is drawn from a log-uniform distribution. The returned target
    is always an actual saved state from the irregular VULCAN trajectory; the stored
    ``actual_dt_s`` is therefore the true dt corresponding to the target label.
    """
    times = np.asarray(times_s, dtype=np.float64)
    if n_pairs <= 0:
        raise TransitionSamplingError("n_pairs must be > 0.")
    if requested_dt_min_s <= 0.0 or requested_dt_max_s <= requested_dt_min_s:
        raise TransitionSamplingError("requested dt bounds must satisfy 0 < min < max.")

    valid_anchor_mask = _strictly_increasing_future_mask(times, min_future_saved_steps)
    if not allow_anchor_at_t0:
        valid_anchor_mask &= times > 0.0
    valid_anchor_indices = np.flatnonzero(valid_anchor_mask)
    if valid_anchor_indices.size == 0:
        raise TransitionSamplingError("No valid anchor indices remain after applying constraints.")

    anchor_index = rng.choice(valid_anchor_indices, size=n_pairs, replace=True).astype(np.int64)
    log_min = np.log10(float(requested_dt_min_s))
    log_max = np.log10(float(requested_dt_max_s))
    requested_dt_s = np.power(10.0, rng.uniform(log_min, log_max, size=n_pairs)).astype(np.float64)

    anchor_time_s = times[anchor_index]
    max_dt_s = times[-1] - anchor_time_s
    requested_dt_s = np.minimum(requested_dt_s, max_dt_s)

    # Guarantee that each sampled pair advances by at least ``min_future_saved_steps``.
    min_target_index = np.minimum(anchor_index + int(min_future_saved_steps), times.size - 1)
    min_advance_dt_s = times[min_target_index] - anchor_time_s
    requested_dt_s = np.maximum(requested_dt_s, min_advance_dt_s)
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

    order = np.lexsort((target_index, anchor_index))
    return TransitionPairs(
        anchor_index=anchor_index[order],
        target_index=target_index[order],
        requested_dt_s=requested_dt_s[order],
        actual_dt_s=actual_dt_s[order],
        anchor_time_s=anchor_time_s[order],
        target_time_s=target_time_s[order],
    )
