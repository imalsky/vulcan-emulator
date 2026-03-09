# VULCAN Emulator Spec

## 1. Purpose

This repository trains a transformer surrogate for VULCAN 1D thermochemical
trajectories. The learned operator is a variable-dt state transition:

- input: atmospheric profiles, conditioning scalars, current chemistry state, realized dt
- output: future chemistry state after that dt

The core workflow is:

1. generate or reuse raw VULCAN trajectories,
2. split runs into train/val/test with no run leakage,
3. normalize and persist full trajectories,
4. train with live transition-pair sampling on GPU,
5. evaluate single-step behavior,
6. export or inspect trained runs with local utility scripts.

This is not a steady-state regressor, not a photochemical emulator, and not a
backward-compatible multi-layout pipeline.

## 2. Current Operating Regime

The shipped config is intentionally narrow:

- `trajectory_sampling.mode = "log_uniform_all_pairs"`
- one model is trained over one bounded dt range configured by
  `trajectory_sampling.dt_min_s` and `trajectory_sampling.dt_max_s`
- training is CUDA-only
- the current state/output species set is the reduced top-8 set

Current shipped species:

- `H2`
- `He`
- `H2O`
- `CO`
- `CO2`
- `CH4`
- `N2`
- `NH3`

## 3. Raw Trajectory Semantics

Each raw HDF5 run stores one VULCAN trajectory with:

- `inputs/pressure_bar`, `inputs/temperature_k`, `inputs/kzz_cm2_s`
- `inputs/state_species`, `inputs/output_species`
- `globals/gravity_cm_s2`, `globals/metallicity_log10`, `globals/c_to_o`
- `trajectory/time_s`, `trajectory/ymix_state`, `trajectory/ymix_output`

The raw loader requires:

- finite numeric arrays,
- strictly increasing `time_s`,
- at least one saved future state beyond `t=0`,
- requested species to exist in the stored species lists.

The raw files remain the authoritative source of trajectory content. They are
never rewritten by training.

## 4. Transition Candidate Space

For a trajectory with `T` saved snapshots, every ordered pair `(i, j)` with
`j > i` is a candidate transition. This gives `T * (T - 1) / 2` possible pairs
before filtering.

The only supported candidate regime is:

- `trajectory_sampling.mode = "log_uniform_all_pairs"`

Candidate validity is defined by:

- `trajectory_sampling.dt_min_s`
- `trajectory_sampling.dt_max_s`
- `trajectory_sampling.min_future_saved_steps`

Rules:

- pairs are filtered by actual realized `dt = time_s[j] - time_s[i]`
- the actual dt must lie in `[dt_min_s, dt_max_s]`
- `min_future_saved_steps` becomes the minimum index gap between anchor and target
- trajectories with no valid candidates are skipped during `--gen`

## 5. Split And Leakage Contract

Splitting is always done at the run level, never at the sampled-pair level.

Consequences:

- every raw trajectory belongs to exactly one of `train`, `val`, or `test`
- live-sampled anchor/target pairs from one run never cross split boundaries
- normalization is fit on the train split only

This prevents state leakage across train and evaluation.

## 6. Processed Data Design

`data/raw` stays raw. `data/processed` contains normalized split trajectories,
not precomputed sampled transition pairs.

Each processed split directory contains:

- `static_inputs.npy` with shape `[num_runs, nz, 3]`
- `state_ymix.npy` with shape `[num_runs, max_steps, nz, state_dim]`
- `global_inputs.npy` with shape `[num_runs, global_static_dim]`
- `time_s.npy` with shape `[num_runs, max_steps]`
- `valid_steps_mask.npy` with shape `[num_runs, max_steps]`
- `run_ids.npy` with shape `[num_runs]`
- `metadata.json`

There are no processed `sequence_inputs/`, `globals/`, `targets/`, or `dt_s/`
pair-shard directories anymore.

### Split Metadata Contract

Each split metadata file records:

- `num_runs`
- `max_steps`
- `total_valid_candidates`
- `sequence_length`
- `input_dim`
- `global_dim`
- `global_static_dim`
- `dt_feature_index`
- `target_dim`
- `state_dim`
- `sequence_feature_order`
- `global_feature_order`
- `global_static_feature_order`
- `state_species_order`
- `output_species_order`
- `output_from_state_indices`
- `normalization_fingerprint`
- candidate-range fields (`sampling_mode`, `dt_sampling_min_s`, `dt_sampling_max_s`,
  `min_future_saved_steps`, `dt_min_s`, `dt_max_s`)

The fixed feature orders are:

- sequence:
  - `pressure_bar`
  - `temperature_k`
  - `kzz_cm2_s`
  - `anchor_ymix:<species>` for each configured state species
- globals:
  - `gravity_cm_s2`
  - `metallicity_log10`
  - `c_to_o`
  - `log10_dt_s`
  - 10 physics-toggle indicators
  - 5 `atm_base_*` one-hot indicators

`global_inputs.npy` excludes `log10_dt_s`. That dt feature is assembled live per
sample at training/evaluation time and inserted at `dt_feature_index`.

## 7. Normalization Contract

Normalization statistics are fit on the train split only.

Fitting rules:

- `pressure_bar`, `temperature_k`, `kzz_cm2_s`:
  - fit over all train-split runs
- `anchor_ymix`:
  - fit over all saved train-split states across all saved times
- targets:
  - reuse the `anchor_ymix` statistics subset selected by `output_from_state_indices`
- non-dt globals:
  - fit once per train run
- `log10_dt_s`:
  - fit over all valid train candidate pairs with weights proportional to `1 / dt`

This keeps dt normalization aligned with the live sampler’s log-uniform target
distribution.

Supported normalization methods remain:

- `standard`
- `log-standard`
- `log-min-max`
- `none`

`anchor_ymix` normalization is fixed to `log-standard`.

`normalization.target_method` is also fixed to `log-standard`, matching
`normalization.sequence_methods.anchor_ymix`.

## 8. Live Sampling Runtime

Training no longer consumes a fixed precomputed pair dataset.

At `--train` time:

1. each processed split is loaded once,
2. each split is copied to GPU once,
3. valid candidate tables are built once from `time_s` and `valid_steps_mask`,
4. train pairs are resampled every epoch,
5. val/test pairs are sampled once deterministically and then reused.

Each candidate table stores:

- `run_index`
- `anchor_index`
- `target_index`
- `actual_dt_s`
- normalized `log10_dt_s`
- sampling weights `1 / dt`
- per-run offsets and counts

### Train Sampling

Train sampling is controlled by:

- `training.live_sampling.train_pairs_per_run_per_epoch`

Per epoch, for each train run:

- sample up to `train_pairs_per_run_per_epoch` valid candidates
- sample without replacement when possible
- if a run has fewer valid candidates, use all of them and log once
- concatenate all selected candidates across runs
- globally shuffle the selected candidate rows

The train sample set intentionally changes across epochs.

### Fixed Eval Sampling

Validation and test sampling are controlled by:

- `training.live_sampling.eval_pairs_per_run`

One fixed candidate table is built for each evaluation split at training start:

- val uses seed `training.seed + 1`
- test uses seed `training.seed + 2`

The same fixed eval pairs are reused for:

- epoch-by-epoch validation,
- final `metrics.json`,
- local helper scripts under `extras/`.

## 9. GPU Hot Path

The training hot path is GPU-resident:

- normalized split tensors live on GPU,
- candidate indices live on GPU,
- batch assembly happens on GPU,
- no `DataLoader`, worker pool, shard cache, mmap, or per-batch host-to-device copy
  is used in the live training path.

Each live batch is assembled as:

- `sequence = concat(static_inputs[run], state_ymix[run, anchor], dim=-1)`
- `globals = concat(global_inputs[run], normalized_log10_dt)` with dt inserted at
  `dt_feature_index`
- `target = state_ymix[run, target][..., output_from_state_indices]`
- `dt_s = actual_dt_s`
- `padding_mask = all_false`, because `nz` is fixed and batching is by pair, not
  by variable vertical length

## 10. Training Contract

Training is CUDA-only.

`training` now requires:

- `device = "cuda"`
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

`training.live_sampling` requires:

- `train_pairs_per_run_per_epoch`
- `eval_pairs_per_run`

`training.loss` requires:

- `lambda_z`
- `lambda_phys`

Optimizer and loss behavior remain unchanged:

- optimizer: AdamW with explicit bias/norm no-decay grouping
- schedule: linear warmup then cosine decay to `min_lr`
- gradient clipping: global norm clip
- checkpointing: `best.pt` (lowest validation combined loss) and `last.pt`

The current shipped model hyperparameters are:

- `d_model = 256`
- `nhead = 8`
- `num_layers = 6`
- `dim_feedforward = 768`
- `dropout = 0.0`
- `film_clamp = 10.0`
- `output_head_divisor = 2`
- `max_sequence_length = 64`
- `conditioning_hidden_dim = 256`

## 11. Evaluation Contract

Single-step validation/test metrics are computed on the fixed eval pairs.

Reported metrics remain:

- normalized-space MSE
- normalized-space MAE
- physical-space `mae_log10` derived from target normalization stats
- combined loss using `training.loss.lambda_z` and `training.loss.lambda_phys`

Final dt-bin summaries are built from the actual dt values present in the fixed
eval pair tables, not from the full split candidate range.

Future work may reintroduce multi-step rollout evaluation or autoregressive
utilities, but the current contract intentionally excludes them.

## 12. Generation And Reuse Rules

`--gen` behaves as follows:

1. if `raw_root` already contains `run_*.h5`, raw generation is skipped and those files are reused
2. otherwise, new run specs are sampled and VULCAN jobs are launched
3. failed VULCAN jobs are dropped as long as at least one usable run survives
4. runs with no valid transition candidates for the current dt range are skipped
5. only usable runs are split into train/val/test
6. `processed_root` is deleted and rebuilt as normalized trajectory splits

`generation.max_trajectory_snapshots` must be `0` (no cap) or `>= 2`.

`--train` does not regenerate data. It requires a matching processed fingerprint.

Changing only:

- `training.live_sampling.train_pairs_per_run_per_epoch`
- `training.live_sampling.eval_pairs_per_run`

does not require rerunning `--gen`.

Changing candidate validity, normalization, raw data, or run splits does require
rerunning `--gen`.

## 13. Provenance Contract

Processed provenance depends on:

- raw run file list, size, and mtime
- dataset manifest
- split assignment file
- normalization metadata
- per-split metadata
- preprocessing-relevant config:
  - generation controls that affect raw/processed artifacts
  - candidate-validity settings
  - VULCAN runtime and samplers
  - physics toggles and boundary conditions
  - data spec
  - normalization config

Processed provenance explicitly excludes:

- live train pair budget
- live eval pair budget

Those settings change training/evaluation behavior, not processed tensors.

## 14. Filesystem Layout

```text
data/
  raw/
    run_000000.h5
    run_000001.h5
    ...
  processed/
    train/
      static_inputs.npy
      state_ymix.npy
      global_inputs.npy
      time_s.npy
      valid_steps_mask.npy
      run_ids.npy
      metadata.json
    val/
      ...
    test/
      ...
    normalization_metadata.json
    processed_summary.json
    processed_fingerprint.json
  dataset_manifest.json
  splits.json

models/
  <output_folder>/
    best.pt
    last.pt
    metrics.json
    data_contract.json
    normalization_metadata.json
    processed_fingerprint.json
    training_log.csv
    figures/

logs/
  runtime_config_<action>_<timestamp>.json
  <action>_<timestamp>.log
  training_progress_<output_folder>.log
```

`generation.worker_root` remains temporary scratch space for VULCAN workers.

## 15. Inference And Utility Scripts

Core inference remains in `src/inference.py`:

- `PhysicalSpaceStandaloneModel`
- `VulcanPredictor`

These operate on physical-space inputs and outputs for single-step transitions.
Checkpoint format, inference artifacts, and the transformer architecture remain
unchanged.

Local scripts under `extras/` now build deterministic fixed eval pairs from the
processed trajectory splits instead of reading pair shards directly.

## 16. Failure Philosophy

The repository is intentionally strict:

- missing config fields are errors,
- incompatible processed artifacts are errors,
- non-finite arrays are errors,
- missing files are errors,
- unsupported old layouts are not silently revived.

Generation is strict about end-state usability, but individual failed VULCAN
runs are dropped when enough successful trajectories remain to build valid
splits.

The goal is fail-fast scientific correctness, not compatibility shims.

## 17. Environment

### Local

- `python src/main.py --gen --config config/config.json`
- `python src/main.py --train --config config/config.json`

### Batch / HPC

Environment variables may override path resolution:

- `VULCAN_EMULATOR_PROJECT_ROOT`
- `VULCAN_EMULATOR_VULCAN_SOURCE`
- `VULCAN_EMULATOR_CONDA_ENV`

The shipped batch script assumes GPU training.

## 18. Conventions

- all logarithms are base-10
- padding mask follows PyTorch convention: `True = padding`
- pressure units are bar inside the emulator
- temperature is Kelvin
- Kzz is cm^2/s
- gravity is cm/s^2
- config keys with `log10_` prefixes are already base-10 transformed
