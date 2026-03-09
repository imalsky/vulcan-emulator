# Config Reference

Short reference for the current `config/config.json` schema. For the full
pipeline contract, see [`spec.md`](/Users/imalsky/Desktop/VULCAN_Project/vulcan-emulator/spec.md).

## `trajectory_sampling`

This section now defines the valid transition-candidate space only.

Required keys:

- `mode`
- `dt_min_s`
- `dt_max_s`
- `min_future_saved_steps`
- `rollout_eval_points`

Notes:

- `mode` must be `"log_uniform_all_pairs"`
- `pairs_per_run` is no longer a valid config key
- `rollout_eval_points` affects rollout evaluation only, not processed data generation

## `generation`

Notes:

- `shard_size` is no longer a valid config key because processed pair shards were removed
- `failure_policy` is no longer a valid config key; `--gen` always drops failed VULCAN runs when any usable runs survive
- `max_trajectory_snapshots` must be `0` or `>= 2`; `0` means no cap

## `training.live_sampling`

This section controls live pair sampling during training and fixed eval-pair
selection during validation/test.

Required keys:

- `train_pairs_per_run_per_epoch`
- `eval_pairs_per_run`

Notes:

- train pairs are resampled every epoch
- val/test pairs are sampled once deterministically from `training.seed`
- changing these budgets does not require rerunning `--gen`

## `training`

Relevant current keys:

- `device`
- `batch_size`
- `epochs`
- `learning_rate`
- `min_lr`
- `warmup_epochs`
- `weight_decay`
- `gradient_clip`
- `use_amp`
- `seed`
- `live_sampling`
- `model`
- `loss`
- `output_folder`

Notes:

- `device` must be `"cuda"`
- `gpu_preload`, `num_workers`, and `training.data_loading` are legacy keys and are rejected
- `training.model.dropout` must be within `[0, 1]`

## `normalization`

Notes:

- `normalization.sequence_methods.anchor_ymix` must be `"log-standard"`
- `normalization.target_method` must be `"log-standard"`

## Processed Data

`--gen` writes normalized trajectory splits under `data/processed`:

- `static_inputs.npy`
- `state_ymix.npy`
- `global_inputs.npy`
- `time_s.npy`
- `valid_steps_mask.npy`
- `run_ids.npy`
- `metadata.json`

There are no processed pair shards anymore.

## Provenance

Processed-data provenance depends on:

- raw run files
- split assignments
- normalization metadata
- candidate-validity config

Processed-data provenance does not depend on:

- `training.live_sampling.train_pairs_per_run_per_epoch`
- `training.live_sampling.eval_pairs_per_run`
- `trajectory_sampling.rollout_eval_points`
