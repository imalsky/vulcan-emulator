# Photochemical VULCAN Surrogate Spec

## Scope

This repository builds a JAX-native surrogate for 1D VULCAN chemistry trajectories.
The intended production path is a strict VULCAN-backed photochemical workflow for hot
Jupiter profiles, with explicit stellar-spectrum conditioning and an export path that
can be embedded in ExoJAX.

The differentiable part of the system is the trained surrogate and its exported
physical-space transition function. Raw VULCAN generation and preprocessing are offline
data-pipeline steps and are not part of the differentiable runtime.

## End-to-End Pipeline

The codebase is organized around four stages:

1. `gen`
   Generate raw trajectory files in HDF5, either from a real VULCAN checkout or from the
   synthetic smoke-test generator.
2. `preprocess`
   Align raw runs to the configured species contract, resample spectra to a fixed grid,
   split runs into train/val/test, and write normalized NumPy tensors plus metadata.
3. `train`
   Build live transition samples from the processed trajectories, train the transformer
   surrogate, write checkpoints, and export the best model to a standalone JAX bundle.
4. `inference` / `exojax_adapter`
   Load either a checkpoint or an export bundle and expose a pure-JAX physical-space
   transition operator.

## CLI Contract

The CLI entrypoint is [src/main.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/main.py).

Supported commands:

- `python -m src.main --config <path> gen`
- `python -m src.main --config <path> preprocess`
- `python -m src.main --config <path> train`
- `python -m src.main --config <path> hyperparam`
- `python -m src.main --config <path> show-config`

Command behavior:

- `gen` prints the raw root, run count, and generation metadata paths.
- `preprocess` prints the processed-root summary payload.
- `train` prints the checkpoint, export, history, and metrics paths.
- `hyperparam` runs the built-in small hyperparameter sweep.
- `show-config` prints the validated configuration after defaults and validation-derived
  fields are applied.

The default CLI config is `config/config.json`. For smoke runs, use the
same file and set `generation.mode = "synthetic"`.

## Configuration Contract

The validated configuration is defined in
[src/config_utils.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/config_utils.py).

### Current Working Assumptions

The shipped configuration and runtime are currently set up around the following choices:

- stellar spectrum: use the WASP-39b Frances stellar surface-flux file as the input spectrum
- Kzz: use a single constant `sampling.kzz_cm2_s` value at every pressure level
- equilibrium-first inference: use `equilibrium()` to predict near-EQ abundances from the
  physical inputs, while keeping `predict()` and `step_from_equilibrium()` available for
  future disequilibrium-focused training and inference
- equilibrium anchor selection: default to `inference.equilibrium_anchor.step_index = 0`,
  so the anchor is the first saved trajectory profile and therefore as close as possible
  to the EQ state at trajectory start
- spectrum compression: run the fixed-grid stellar spectrum through the configurable
  internal encoder, with the shipped config using `encoder_mode = "autoencoder"`
- extensibility: the physical-space API still accepts full per-level `kzz_cm2_s` inputs
  and arbitrary anchor states, so later retraining on disequilibrium trajectories does
  not require changing the exported interface

Required top-level sections:

- `paths`
- `data_spec`
- `physics_toggles`
- `vulcan_runtime`
- `sampling`
- `stellar_spectrum`
- `generation`
- `trajectory_sampling`
- `preprocessing`
- `normalization`
- `training`

Important derived contracts:

- `data_spec.state_species` defines the ordered per-level anchor-state basis.
- `data_spec.output_species` defines the ordered per-level target basis.
- `data_spec.required_global_inputs` defines the full conditioning order and must include
  `log10_dt_s`.
- `data_spec.global_static_feature_order` is the same order with `log10_dt_s` removed.
- `data_spec.global_feature_order` is the full global order including `log10_dt_s`.
- `data_spec.dt_feature_index` is the insertion point for normalized `log10_dt_s`.

Default species order:

- `H2`
- `He`
- `H`
- `O`
- `OH`
- `H2O`
- `CO`
- `CO2`
- `CH4`
- `N2`
- `NH3`
- `H2S`
- `SH`
- `S`
- `SO`
- `SO2`
- `S2`

Global-conditioning semantics:

- `gravity_cm_s2`, `metallicity_log10`, `c_to_o`, and `log10_dt_s` are the core physical
  conditioning scalars.
- `use_*` toggles are persisted into the dataset and conditioned on directly.
- `atm_base_*` one-hot features encode the selected atmospheric base gas.
- `metallicity_log10` is kept in physical units in normalization, while `gravity_cm_s2`
  is log-standardized and `c_to_o` is standardized.

## Module Responsibilities

Primary code paths:

- [src/vulcan_runner.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/vulcan_runner.py):
  raw generation, VULCAN patching, raw HDF5 writing, generation metadata.
- [src/sampling.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/sampling.py):
  Latin-hypercube run sampling, temperature/Kzz/time/spectrum/initial-state sampling.
- [src/preprocess.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/preprocess.py):
  raw-run loading, split creation, normalization fitting, processed tensor writing.
- [src/transition_sampling.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/transition_sampling.py):
  candidate transition construction and weighted, stratified row sampling.
- [src/data_loader.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/data_loader.py):
  processed split loading and batch assembly.
- [src/jax_model.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/jax_model.py):
  FiLM-conditioned transformer definition and parameter initialization.
- [src/trainer.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/trainer.py):
  training loop, evaluation, checkpoint writing, export.
- [src/export_jax.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/export_jax.py):
  standalone export bundle serialization.
- [src/inference.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/inference.py):
  physical-space runtime wrapper around checkpoints or exports.
- [src/exojax_adapter.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/exojax_adapter.py):
  ExoJAX-oriented export loader.

## Raw Generation Contract

### Sampling

Run specifications are sampled from the ranges in `sampling`:

- pressure grid: log-spaced from `pressure_bottom_bar` to `pressure_top_bar`
- temperature profile: analytic hot-Jupiter profile, or Roth profiles if
  `roth_sampler.enabled=true`
- Kzz profile: constant `sampling.kzz_cm2_s` applied at every pressure level
- gravity, metallicity, C/O: Latin-hypercube sampled
- time grid: saved-time grid with log-uniform adjacent steps between
  `time_step_log10_min_s` and `time_step_log10_max_s`
- stellar spectrum: loaded from the configured template file and saved into the library

Strict input rules:

- if `roth_sampler.enabled=true`, at least one profile must match the configured glob
  and filters or generation fails;
- `stellar_spectrum.template_file` must exist or generation fails;
- if `generation.mode="vulcan"`, the configured VULCAN checkout, cfg file, and chemistry
  file must exist or generation fails.

### Generation Modes

`generation.mode="vulcan"`:

- copies the configured VULCAN source tree into per-run worker directories;
- writes an atmosphere file with `Pressure Temp Kzz` columns;
- writes a VULCAN-format stellar surface-flux file;
- optionally regenerates `chem_funs.py`;
- runs VULCAN and converts the resulting `.vul` pickle to raw HDF5.

`generation.mode="synthetic"`:

- exists only for tests and smoke runs;
- uses a deterministic toy sulfur photochemistry with oxidation radicals (`H`, `O`,
  `OH`) and vertical mixing;
- is not intended for science training corpora.

Both modes support CPU-side parallel raw generation via `generation.parallel_workers`.

### Initialization & Equilibrium-First Approach

Every VULCAN run is explicitly initialized from thermochemical equilibrium:

- `ini_mix = 'EQ'` — FastChem computes the equilibrium abundances at each P-T level
- `use_condense = False` — no condensation during initialization or integration
- `use_photo = True` — photochemistry is enabled (driven by the patched stellar spectrum)

These three settings are patched directly into `vulcan_cfg.py` by the runner and do not
rely on VULCAN-checkout defaults.  This means `trajectory[0]` in the raw HDF5 output is
always the FastChem equilibrium state for the sampled conditions.

The inference API mirrors this equilibrium-first design:

- `equilibrium()` returns near-EQ abundances for a given P-T profile and conditioning
  inputs by calling the surrogate with a minimal timestep (`log10_dt_s = 0`, i.e. dt = 1 s).
- `step_from_equilibrium(dt_s=...)` first computes equilibrium, then evolves the state
  forward by the user-supplied timestep.
- `predict()` remains available for arbitrary anchor-state transitions.

The default workflow is: call `equilibrium()` to get the EQ state, then optionally call
`predict()` with a user-defined dt to evolve forward from that anchor.

### VULCAN Config Patching

When running in real VULCAN mode, the runner patches `vulcan_cfg.py` to set:

- `ini_mix = 'EQ'` (FastChem equilibrium initialization)
- chemistry toggles such as `use_photo`, `use_ion`, `use_Kzz`, `use_moldiff`
- `use_condense = False` (no condensation)
- `network`
- `atm_file`
- `sflux_file`
- `atm_base`
- `atm_type = 'file'`
- `Kzz_prof = 'file'`
- `T_cross_sp = vulcan_runtime.t_cross_sp`
- `nz`, `P_b`, `P_t`
- `gs`, `r_star`, `orbit_radius`, `sl_angle`, `f_diurnal`
- elemental abundances derived from metallicity and C/O
- output/save flags needed to persist time-history trajectories

`vulcan_runtime.t_cross_sp` must only contain species supported by the target VULCAN
checkout. The shipped WASP-39b config uses the sulfur-relevant supported subset:
`H2O`, `H2S`, `SH`, `SO2`, and `S2`.

### Raw HDF5 Layout

Each raw run stores:

- `inputs/pressure_bar`
- `inputs/temperature_k`
- `inputs/kzz_cm2_s`
- `inputs/state_species`
- `inputs/output_species`
- `globals/<name>` for every persisted global scalar
- `trajectory/time_s`
- `trajectory/ymix_state`
- `trajectory/ymix_output`
- `spectrum/name`
- `spectrum/wavelength_nm`
- `spectrum/flux_erg_cm2_s_nm`

For real VULCAN conversion:

- the converter accepts `variable.ymix_time` directly when present;
- otherwise it derives mixing ratios from `variable.y_time / atm.n_0`;
- species are then reindexed into the configured `state_species` and `output_species`
  orders before writing HDF5.

### Raw Dataset Metadata

Each raw dataset also writes:

- `generation_manifest.json`
- `sampling_coverage.json`

`generation_manifest.json` records the generated files and provenance hashes.
`sampling_coverage.json` records realized min/max coverage for gravity, metallicity, C/O,
temperature, Kzz, adjacent saved `dt`, and pressure.

## Preprocessing Contract

Preprocessing is defined in
[src/preprocess.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/preprocess.py).

### Split Logic

- raw runs are shuffled with `preprocessing.seed`;
- train/val/test fractions come from config;
- the splitter enforces at least one run in each split.

### Spectrum Handling

- each raw spectrum is resampled to a fixed linear wavelength grid defined by
  `stellar_spectrum.wavelength_min_nm`, `stellar_spectrum.wavelength_max_nm`, and
  `stellar_spectrum.num_bins`;
- the resampled wavelength grid is written into both `normalization.json` and
  `data_contract.json`.

### Normalization Contract

Normalization is fit on the train split only.

Blocks:

- `sequence_static`
  three one-feature blocks for `pressure_bar`, `temperature_k`, and `kzz_cm2_s`, using
  the methods from `normalization.sequence_methods`
- `state`
  log-standard normalization of anchor-state trajectories with `state_floor`
- `target`
  log-standard normalization of target trajectories with `state_floor`
- `global_static`
  mixed normalization over the static global vector
- `log10_dt_s`
  standard normalization using weighted transition statistics from valid train-split pairs
- `spectrum`
  log-standard normalization with `spectrum_floor`

`log10_dt_s` statistics are fit using transition weights proportional to `1 / dt`.

### Processed Files

Each processed split directory contains:

- `sequence_inputs.npy` with shape `[num_runs, nz, 3]`
- `state_trajectories.npy` with shape `[num_runs, max_steps, nz, state_dim]`
- `target_outputs.npy` with shape `[num_runs, max_steps, nz, target_dim]`
- `global_inputs.npy` with shape `[num_runs, global_static_dim]`
- `spectrum_inputs.npy` with shape `[num_runs, spectrum_dim]`
- `time_s.npy` with shape `[num_runs, max_steps]`
- `valid_steps_mask.npy` with shape `[num_runs, max_steps]`
- `run_ids.json`
- `metadata.json`

Top-level processed metadata:

- `normalization.json`
- `data_contract.json`
- `splits.json`
- `processed_manifest.json`

The current processed-data contract version is `3`.

### Data Contract Fields

`data_contract.json` includes:

- `processed_data_version`
- `state_species_order`
- `output_species_order`
- `sequence_static_feature_order`
- `global_feature_order`
- `global_static_feature_order`
- `dt_feature_index`
- `sequence_dim`
- `target_dim`
- `spectrum_dim`
- `spectrum_wavelength_nm`

`sequence_dim` is always `3 + state_dim`, because batching concatenates normalized
`[pressure, temperature, kzz]` with the normalized anchor state.

## Transition Sampling Contract

Transition sampling is defined in
[src/transition_sampling.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/transition_sampling.py)
and [src/live_sampling.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/live_sampling.py).

Candidate-row construction:

- valid rows are `(anchor_index, target_index)` pairs taken from saved trajectory steps;
- `target_index` must be at least `min_future_saved_steps` after `anchor_index`;
- `dt_s` must lie in `[dt_min_s, dt_max_s]`.

Each candidate row stores:

- `run_index`
- `anchor_index`
- `target_index`
- `dt_s`
- `log10_dt_s`
- normalized `log10_dt_s`
- `weight = 1 / dt_s`

Row selection for both train and eval:

- bins rows by `log10_dt_s` using `trajectory_sampling.num_logdt_bins`
- allocates a near-uniform quota across non-empty bins
- samples without replacement inside each bin using the stored `1 / dt` weights
- uses a seeded RNG for reproducibility

Train rows are resampled each epoch. Eval rows are deterministic because the RNG seed is
fixed, not because the code takes the first sorted candidates.

## Batch Contract

Batch assembly is defined in
[src/data_loader.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/data_loader.py).

For each sampled row:

- `sequence`
  normalized `[pressure, temperature, kzz, anchor_state]` with shape
  `[batch, nz, 3 + state_dim]`
- `global_inputs`
  normalized static globals with normalized `log10_dt_s` inserted at `dt_feature_index`
- `spectrum_inputs`
  normalized fixed-grid spectrum
- `target`
  normalized target state at the sampled future step
- `dt_s`
  physical timestep in seconds

Anchor states are always drawn from `state_trajectories.npy`.
Targets are always drawn from `target_outputs.npy`.

## Model Contract

The surrogate architecture is implemented in
[src/jax_model.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/jax_model.py).

Architecture summary:

- project the per-level input sequence to `d_model`
- add sinusoidal vertical positional encoding
- encode the stellar spectrum with mode `autoencoder`, `linear`, or `none`
- concatenate `[global_inputs, spectrum_latent]`
- project that context to FiLM `gamma` and `beta` parameters for each transformer layer
- run a stack of pre-norm self-attention blocks with FiLM modulation
- project the final hidden state to `target_dim`

Model dimensions are built from the processed contract plus `training.model` and
`stellar_spectrum` settings.

The model path is pure JAX and is compatible with `jax.grad` and `jax.jvp`.

## Training Contract

Training is defined in
[src/trainer.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/trainer.py).

Behavior:

- if a compatible processed dataset already exists, training reuses it;
- otherwise preprocessing is rerun;
- if no raw dataset exists, raw generation is triggered first using the current config.

Loss terms:

- normalized-space MSE on the predicted target tensor
- physical log-space MSE after inverse normalization
- optional spectrum autoencoder reconstruction MSE

Optimization:

- AdamW implemented directly in JAX
- cosine decay with optional warmup
- global gradient clipping

Artifacts written under `paths.checkpoints_root`:

- `best.pt`
- `last.pt`
- `history.json`
- `metrics.json`

`metrics.json` includes:

- `best_val_combined_loss`
- `test`
- `num_train_candidates`
- `num_val_candidates`
- `num_test_candidates`
- `parameter_count`

## Export Contract

The standalone export bundle is written by
[src/export_jax.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/export_jax.py)
under `paths.jax_export_root`.

Files:

- `params.npz`
- `structure.json`
- `contract.json`
- `normalization.json`
- `config.json`
- `model_dimensions.json`

The export contains everything needed to reconstruct the pure-JAX transition operator
without re-running training code.

## Inference Contract

Inference is defined in
[src/inference.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/inference.py).

`load_physical_space_model()` accepts either:

- a checkpoint file such as `best.pt`
- an export directory containing the standalone bundle

The exposed `transition_fn` accepts physical-space inputs:

- `pressure_bar` with shape `[nz]`
- `temperature_k` with shape `[nz]`
- `kzz_cm2_s` with shape `[nz]`
- `anchor_ymix` with shape `[nz, state_dim]`
- `global_static_vector` with shape `[global_static_dim]`
- `spectrum_inputs` with shape `[spectrum_dim]`
- `log10_dt_s` as a scalar

It returns a physical-space prediction with shape `[nz, target_dim]`.

The wrapper is responsible for:

- applying sequence, state, global, dt, and spectrum normalization
- inserting normalized `log10_dt_s` at `dt_feature_index`
- calling the JAX model
- inverse-transforming the target prediction back to physical mixing ratios

### Equilibrium-First Convenience API

`PhysicalSpaceStandaloneModel` exposes two additional methods on top of `predict()`:

- `equilibrium(pressure_bar, temperature_K, eddy_diffusion_cm2_s, global_inputs,
  spectrum_inputs)` — returns near-EQ abundances by calling the surrogate with
  `log10_dt_s = 0.0` from the configured equilibrium anchor. The shipped config uses
  the first saved trajectory profile (`step_index = 0`), which is the closest saved
  state to the EQ initialization. This is the recommended default entry point when you
  only need the equilibrium state.

- `step_from_equilibrium(dt_s, pressure_bar, temperature_K, eddy_diffusion_cm2_s,
  global_inputs, spectrum_inputs)` — calls `equilibrium()` to compute the anchor, then
  evolves forward by the user-supplied `dt_s` seconds.  Convenience wrapper for the
  common EQ-then-evolve workflow.

## ExoJAX Adapter Contract

The ExoJAX-facing adapter is
[src/exojax_adapter.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/exojax_adapter.py).

`load_exojax_transition(export_root)` returns:

- `transition_fn`
- `global_static_feature_order`
- `global_feature_order`
- `state_species_order`
- `spectrum_wavelength_nm`

This is the intended integration surface for ExoJAX-side callers that need both the
function and the canonical feature/species ordering metadata.

## Hyperparameter Search Contract

The built-in hyperparameter sweep is defined in
[src/hyperparam_testing.py](/Users/imalsky/Desktop/VULCAN_JAX/vulcan_emulator_photochem/src/hyperparam_testing.py).

It is intentionally small and writes results under `models/hyperparam_testing`:

- per-trial checkpoints and exports
- `logs/trial_XXX.json`
- `best_config.json`

This is a utility path for quick internal sweeps, not a distributed experiment manager.

## Production Assumptions

- The production dataset path is real VULCAN generation, not synthetic generation.
- The shipped WASP-39b config expects the adjacent `../VULCAN-master` checkout, the
  Frances stellar surface-flux file, and the sulfur-enabled
  `thermo/SNCHO_photo_network_2025.txt` network.
- Existing processed datasets from earlier species contracts must be regenerated when the
  processed-data version changes.
