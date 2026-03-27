"""Candidate transition pair extraction, log-dt binning, and row sampling.

The transition model predicts mixing-ratio changes between an *anchor*
time step and a *target* time step separated by a variable dt.  This
module:

1. **Enumerates** all valid ``(anchor, target)`` pairs from each saved
   trajectory, subject to ``dt_min_s``, ``dt_max_s``, and
   ``min_future_saved_steps`` constraints.
2. **Indexes** them in a flat ``CandidateTable`` with per-row weights
   proportional to ``1/dt`` (so short transitions, which are more
   common but carry less information, can be down-sampled).
3. **Stratifies** sampling across log-dt bins so that each epoch sees
   a balanced distribution of time horizons rather than collapsing
   onto the shortest available transitions.
4. **Provides** epoch-dependent (training) and deterministic (eval)
   row samplers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CandidateTable:
    """Indexed table of valid transition pairs extracted from saved trajectories.

    Each row represents one ``(run, anchor_step, target_step)`` triple.
    Arrays are flat (one element per candidate pair) and share a common
    length.  ``run_row_bounds`` is a ``(num_runs + 1,)`` array of
    cumulative row counts for per-run slicing.
    """
    run_index: np.ndarray
    anchor_index: np.ndarray
    target_index: np.ndarray
    dt_s: np.ndarray
    log10_dt_s: np.ndarray
    normalized_log10_dt_s: np.ndarray
    weight: np.ndarray
    run_row_bounds: np.ndarray

    def num_rows(self) -> int:
        """Return the number of candidate transition rows."""
        return int(self.run_index.size)

    def rows_for_run(self, run: int) -> np.ndarray:
        """Return the candidate row indices belonging to one run."""
        start = int(self.run_row_bounds[run])
        end = int(self.run_row_bounds[run + 1])
        return np.arange(start, end, dtype=np.int32)


def fit_log10_dt_normalization(
    time_s: np.ndarray,
    valid_steps_mask: np.ndarray,
    *,
    dt_min_s: float,
    dt_max_s: float,
    min_future_saved_steps: int,
) -> dict[str, float]:
    """Fit weighted mean and std of ``log10(dt)`` for the dt feature.

    Statistics are weighted by ``1/dt`` so that the normalization is
    not dominated by the (fewer but larger) long-horizon pairs.  These
    statistics are stored in ``normalization.json`` and reused at
    inference time to normalize the dt conditioning input.
    """
    dt_values: list[np.ndarray] = []
    weight_values: list[np.ndarray] = []
    for run in range(time_s.shape[0]):
        valid_steps = np.flatnonzero(valid_steps_mask[run])
        if valid_steps.size < 2:
            continue
        for offset, anchor in enumerate(valid_steps[:-1]):
            targets = valid_steps[offset + 1 :]
            gap_mask = (targets - anchor) >= min_future_saved_steps
            targets = targets[gap_mask]
            if targets.size == 0:
                continue
            dt_s = time_s[run, targets] - time_s[run, anchor]
            keep = (dt_s >= dt_min_s) & (dt_s <= dt_max_s)
            if np.any(keep):
                valid_dt = np.asarray(dt_s[keep], dtype=np.float64)
                dt_values.append(valid_dt)
                weight_values.append(1.0 / np.clip(valid_dt, 1.0e-30, None))
    if not dt_values:
        raise ValueError("No valid dt values were found while fitting dt normalization.")
    dt = np.concatenate(dt_values)
    weights = np.concatenate(weight_values)
    log10_dt = np.log10(dt)
    mean = np.average(log10_dt, weights=weights)
    variance = np.average((log10_dt - mean) ** 2, weights=weights)
    std = float(max(np.sqrt(variance), 1.0e-8))
    return {"mean": float(mean), "std": std}


def build_candidate_table(
    time_s: np.ndarray,
    valid_steps_mask: np.ndarray,
    *,
    dt_min_s: float,
    dt_max_s: float,
    min_future_saved_steps: int,
    log10_dt_stats: dict[str, float],
) -> CandidateTable:
    """Enumerate all valid ``(anchor, target)`` pairs for each run.

    Pairs are filtered by the dt range and minimum step gap, then
    stored in a flat table.  ``normalized_log10_dt_s`` is z-scored
    using the previously fitted statistics so it can be fed directly
    as a model input feature.
    """
    run_index: list[np.ndarray] = []
    anchor_index: list[np.ndarray] = []
    target_index: list[np.ndarray] = []
    dt_s_values: list[np.ndarray] = []
    log10_dt_values: list[np.ndarray] = []
    weight_values: list[np.ndarray] = []
    run_row_bounds = [0]
    total_rows = 0
    mean = float(log10_dt_stats["mean"])
    std = float(log10_dt_stats["std"])
    for run in range(time_s.shape[0]):
        rows_run: list[tuple[int, int, float]] = []
        valid_steps = np.flatnonzero(valid_steps_mask[run])
        for offset, anchor in enumerate(valid_steps[:-1]):
            targets = valid_steps[offset + 1 :]
            if targets.size == 0:
                continue
            targets = targets[(targets - anchor) >= int(min_future_saved_steps)]
            if targets.size == 0:
                continue
            dt_s = time_s[run, targets] - time_s[run, anchor]
            keep = (dt_s >= dt_min_s) & (dt_s <= dt_max_s)
            for target, dt in zip(targets[keep], dt_s[keep]):
                rows_run.append((int(anchor), int(target), float(dt)))
        rows_run.sort(key=lambda row: (row[0], row[1]))
        if rows_run:
            anchors = np.array([row[0] for row in rows_run], dtype=np.int32)
            targets = np.array([row[1] for row in rows_run], dtype=np.int32)
            dt_s = np.array([row[2] for row in rows_run], dtype=np.float64)
            log10_dt = np.log10(np.clip(dt_s, 1.0e-30, None))
            # Short transitions are common, so keep their weights explicit for later
            # balanced sampling rather than baking the policy into row ordering.
            weight = 1.0 / np.clip(dt_s, 1.0e-30, None)
            run_index.append(np.full_like(anchors, fill_value=run))
            anchor_index.append(anchors)
            target_index.append(targets)
            dt_s_values.append(dt_s)
            log10_dt_values.append(log10_dt)
            weight_values.append(weight)
            total_rows += anchors.size
        run_row_bounds.append(total_rows)

    if total_rows == 0:
        zeros_i = np.zeros((0,), dtype=np.int32)
        zeros_f = np.zeros((0,), dtype=np.float64)
        return CandidateTable(
            run_index=zeros_i,
            anchor_index=zeros_i,
            target_index=zeros_i,
            dt_s=zeros_f,
            log10_dt_s=zeros_f,
            normalized_log10_dt_s=zeros_f,
            weight=zeros_f,
            run_row_bounds=np.asarray(run_row_bounds, dtype=np.int32),
        )

    run_idx = np.concatenate(run_index)
    anchor_idx = np.concatenate(anchor_index)
    target_idx = np.concatenate(target_index)
    dt_all = np.concatenate(dt_s_values)
    log10_dt_all = np.concatenate(log10_dt_values)
    normalized = (log10_dt_all - mean) / std
    weight_all = np.concatenate(weight_values)
    return CandidateTable(
        run_index=run_idx,
        anchor_index=anchor_idx,
        target_index=target_idx,
        dt_s=dt_all,
        log10_dt_s=log10_dt_all,
        normalized_log10_dt_s=normalized.astype(np.float64),
        weight=weight_all.astype(np.float64),
        run_row_bounds=np.asarray(run_row_bounds, dtype=np.int32),
    )


def _logdt_bin_ids(log10_dt_s: np.ndarray, *, num_bins: int) -> np.ndarray:
    """Assign each ``log10_dt_s`` value to a stratification bin.

    Bin edges are linearly spaced between the observed min and max of
    ``log10_dt_s``.  If all values are identical, everything maps to
    bin 0.
    """
    if log10_dt_s.size == 0:
        return np.zeros((0,), dtype=np.int32)
    log10_min = float(np.min(log10_dt_s))
    log10_max = float(np.max(log10_dt_s))
    if np.isclose(log10_min, log10_max):
        return np.zeros(log10_dt_s.shape, dtype=np.int32)
    edges = np.linspace(log10_min, log10_max, int(num_bins) + 1, dtype=np.float64)
    return np.clip(
        np.searchsorted(edges[1:-1], log10_dt_s, side="right"),
        0,
        int(num_bins) - 1,
    ).astype(np.int32)


def _allocate_stratified_counts(
    bin_sizes: np.ndarray,
    *,
    total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Distribute a per-run sampling budget across non-empty log-dt bins."""
    counts = np.zeros_like(bin_sizes, dtype=np.int32)
    remaining = min(int(total), int(np.sum(bin_sizes)))
    while remaining > 0:
        active_bins = np.flatnonzero(bin_sizes > counts)
        if active_bins.size == 0:
            break
        for bin_id in rng.permutation(active_bins):
            if remaining == 0:
                break
            counts[bin_id] += 1
            remaining -= 1
    return counts


def _sample_weighted_rows(
    rows: np.ndarray,
    *,
    weights: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample row ids without replacement using optional per-row weights."""
    if rows.size <= count:
        return np.asarray(rows, dtype=np.int32)
    probabilities = np.asarray(weights, dtype=np.float64)
    total_weight = float(np.sum(probabilities))
    if total_weight <= 0.0 or not np.isfinite(total_weight):
        probabilities = None
    else:
        probabilities = probabilities / total_weight
    chosen = rng.choice(rows, size=int(count), replace=False, p=probabilities)
    return np.sort(np.asarray(chosen, dtype=np.int32))


def sample_rows_per_run(
    candidate_table: CandidateTable,
    *,
    pairs_per_run: int,
    num_logdt_bins: int,
    rng: np.random.Generator,
    deterministic: bool,
) -> np.ndarray:
    """Sample candidate rows per run with log-dt stratification.

    For each run, up to ``pairs_per_run`` rows are drawn with the
    sampling budget distributed evenly across log-dt bins.  Within
    each bin, rows are drawn without replacement using ``1/dt``
    weights.  If ``deterministic`` is False the final row order is
    shuffled; otherwise rows are returned sorted.
    """
    row_bin_ids = _logdt_bin_ids(candidate_table.log10_dt_s, num_bins=int(num_logdt_bins))
    selected: list[np.ndarray] = []
    num_runs = int(candidate_table.run_row_bounds.size - 1)
    for run in range(num_runs):
        rows = candidate_table.rows_for_run(run)
        if rows.size == 0:
            continue
        if rows.size <= pairs_per_run:
            chosen = rows
        else:
            # Stratify within each run so eval and training cover the dt range
            # instead of collapsing onto the shortest available transitions.
            run_bin_ids = row_bin_ids[rows]
            bin_sizes = np.bincount(run_bin_ids, minlength=int(num_logdt_bins))
            target_counts = _allocate_stratified_counts(
                bin_sizes,
                total=int(pairs_per_run),
                rng=rng,
            )
            sampled_rows: list[np.ndarray] = []
            for bin_id, count in enumerate(target_counts):
                if count == 0:
                    continue
                bin_rows = rows[run_bin_ids == bin_id]
                sampled_rows.append(
                    _sample_weighted_rows(
                        bin_rows,
                        weights=candidate_table.weight[bin_rows],
                        count=int(count),
                        rng=rng,
                    )
                )
            chosen = np.sort(np.concatenate(sampled_rows)).astype(np.int32)
        selected.append(chosen.astype(np.int32))
    if not selected:
        return np.zeros((0,), dtype=np.int32)
    merged = np.concatenate(selected)
    if not deterministic and merged.size > 1:
        rng.shuffle(merged)
    return merged.astype(np.int32)


# ---------------------------------------------------------------------------
# Live sampling helpers (merged from training/live_sampling.py)
# ---------------------------------------------------------------------------


def sample_train_rows(
    candidate_table: CandidateTable,
    *,
    pairs_per_run: int,
    num_logdt_bins: int,
    seed: int,
    epoch: int,
) -> np.ndarray:
    """Sample training rows with epoch-dependent randomness."""
    rng = np.random.default_rng(int(seed) + int(epoch))
    return sample_rows_per_run(
        candidate_table,
        pairs_per_run=int(pairs_per_run),
        num_logdt_bins=int(num_logdt_bins),
        rng=rng,
        deterministic=False,
    )


def sample_eval_rows(
    candidate_table: CandidateTable,
    *,
    pairs_per_run: int,
    num_logdt_bins: int,
    seed: int,
) -> np.ndarray:
    """Sample evaluation rows deterministically from the candidate table."""
    rng = np.random.default_rng(int(seed))
    return sample_rows_per_run(
        candidate_table,
        pairs_per_run=int(pairs_per_run),
        num_logdt_bins=int(num_logdt_bins),
        rng=rng,
        deterministic=True,
    )
