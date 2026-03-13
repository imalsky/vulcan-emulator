from __future__ import annotations

import numpy as np

from src.live_sampling import sample_eval_rows, sample_train_rows
from src.transition_sampling import CandidateTable


def _candidate_table(dt_s: list[float]) -> CandidateTable:
    dt = np.asarray(dt_s, dtype=np.float64)
    rows = np.arange(dt.size, dtype=np.int32)
    return CandidateTable(
        run_index=np.zeros(dt.size, dtype=np.int32),
        anchor_index=np.zeros(dt.size, dtype=np.int32),
        target_index=rows + 1,
        dt_s=dt,
        log10_dt_s=np.log10(dt),
        normalized_log10_dt_s=np.zeros(dt.size, dtype=np.float64),
        weight=(1.0 / dt).astype(np.float64),
        run_row_bounds=np.asarray([0, dt.size], dtype=np.int32),
    )


def test_eval_sampling_is_stratified_over_logdt():
    candidate_table = _candidate_table([10.0, 12.0, 15.0, 1.0e4, 1.2e4, 1.5e4])
    rows_a = sample_eval_rows(
        candidate_table,
        pairs_per_run=4,
        num_logdt_bins=2,
        seed=17,
    )
    rows_b = sample_eval_rows(
        candidate_table,
        pairs_per_run=4,
        num_logdt_bins=2,
        seed=17,
    )

    assert np.array_equal(rows_a, rows_b)
    assert np.sum(rows_a < 3) == 2
    assert np.sum(rows_a >= 3) == 2


def test_train_sampling_uses_dt_weights_within_each_bin():
    candidate_table = _candidate_table([1.0, 1.0e3, 1.0e3, 1.0e3])
    leading_hits = 0
    for seed in range(64):
        chosen = sample_train_rows(
            candidate_table,
            pairs_per_run=1,
            num_logdt_bins=1,
            seed=seed,
            epoch=0,
        )
        leading_hits += int(chosen[0] == 0)

    assert leading_hits >= 60
