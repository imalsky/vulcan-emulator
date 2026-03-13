from __future__ import annotations

import numpy as np

from .transition_sampling import CandidateTable, sample_rows_per_run


def sample_train_rows(
    candidate_table: CandidateTable,
    *,
    pairs_per_run: int,
    num_logdt_bins: int,
    seed: int,
    epoch: int,
) -> np.ndarray:
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
    rng = np.random.default_rng(int(seed))
    return sample_rows_per_run(
        candidate_table,
        pairs_per_run=int(pairs_per_run),
        num_logdt_bins=int(num_logdt_bins),
        rng=rng,
        deterministic=True,
    )
