# Vulcan-Emulator Specification

## 1. Objective

This repository implements a GPU-first surrogate for the non-photochemical part of the current local VULCAN codebase.

The current operating mode is intentionally narrower than a fully arbitrary-time emulator:

1. generate VULCAN trajectories with as much non-photochemical physics as is practical from simple config toggles,
2. sample only late-time transitions,
3. train the model on one fixed requested jump size,
4. condition the model on the snapped **actual** jump size that comes from the saved VULCAN trajectory.

The shipped main configuration uses:

- `trajectory_sampling.fixed_requested_dt_s = 1e12` seconds,
- `trajectory_sampling.post_equilibrium_time_min_s = 1e12` seconds,
- `trajectory_sampling.target_selection = "nearest_saved_snapshot"`,
- `generation.save_evo_frq = 1`.

This is a deliberate "single operational horizon after equilibration" design, not a true arbitrary-`dt` emulator.

## 2. Scope

Included in scope:

- VULCAN thermochemistry from the currently compiled local chemistry network,
- sampled temperature-pressure profiles,
- sampled eddy-diffusion profiles,
- sampled gravity and elemental abundance controls,
- optional molecular diffusion,
- optional upwind molecular-diffusion discretization,
- optional boundary conditions,
- optional condensation and settling,
- optional cold-trap initialization,
- optional adaptive ODE relative tolerance,
- optional low-temperature-limited rates,
- configurable atmospheric base gas,
- fixed-`dt` late-time surrogate training,
- GPU-preloaded and vectorized surrogate training/inference.

Explicitly excluded in this branch:

- photochemistry,
- ion chemistry,
- arbitrary-time requested supervision,
- vertical advection via `use_vz`,
- non-equilibrium initialization modes other than `ini_mix = "EQ"`,
- automatic chemistry-network regeneration from VULCAN source templates.

## 3. Design Principles

The repository follows these rules.

- Fail fast on missing files, invalid config keys, invalid ranges, inconsistent artifacts, and unsupported mode combinations.
- Keep all important science/runtime controls in config rather than hidden in code.
- Prefer deterministic preprocessing and small explicit helper functions over implicit magic.
- Vectorize preprocessing and training data construction wherever practical.
- Keep training GPU-first, but do not pretend VULCAN generation itself is GPU-accelerated; the external VULCAN subprocess remains CPU-bound.
- Remove dead configuration surface rather than keeping misleading knobs.

## 4. Repository Responsibilities

`src/sampling.py`
: Samples TP, gravity, Kzz, and abundance parameters and materializes `RunSpec` objects.

`src/vulcan_runner.py`
: Validates the local VULCAN tree, creates isolated worker copies, patches `vulcan_cfg.py`, launches `vulcan.py -n`, and converts `.vul` outputs into canonical raw HDF5 trajectories.

`src/transition_sampling.py`
: Implements fixed requested-`dt` late-time transition sampling and deterministic fixed-`dt` rollout-path construction.

`src/preprocess.py`
: Loads raw HDF5 runs, builds processed training pairs, computes normalization statistics, and writes sharded arrays plus metadata.

`src/model.py`
: Defines the transformer + FiLM transition model.

`src/data_loader.py`
: Loads processed shards in RAM/disk/auto modes and supports CUDA device prefetch.

`src/trainer.py`
: Trains the surrogate, evaluates validation/test metrics, and computes fixed-`dt` rollout metrics against raw VULCAN trajectories.

`src/main.py`
: Entry point for `--gen` and `--train`.

`config/config.json`
: Main training/generation configuration. This now defaults to the full 69-species list present in the current local `chem_funs.py`.

`config/tiny_train_smoke.json`
: Tiny CPU-safe smoke config.

## 5. Important Current Constraints

### 5.1 Chemistry-network identity

The generation path runs `python vulcan.py -n` inside a copied VULCAN source tree. That means the effective chemistry network comes from the already compiled/generated VULCAN files in that tree, especially `chem_funs.py`.

This repository does **not** rebuild the chemistry network automatically.

If the VULCAN chemistry network changes, the user must:

1. regenerate the VULCAN chemistry artifacts in the VULCAN tree,
2. make sure `data_spec.state_species` and `data_spec.output_species` still match that compiled network,
3. regenerate raw and processed data.

### 5.2 Full-state closure versus reduced species

The shipped main config uses the full currently compiled 69-species list from the local `chem_funs.py`, because that is the most faithful non-photochemical state representation available without further architectural changes.

Reduced species subsets are still supported, but they are not fully Markov-closed with respect to the full VULCAN chemistry state.

### 5.3 Equilibrium is treated as an operational regime, not a proof

`post_equilibrium_time_min_s = 1e12` seconds is a practical late-time cutoff, not a mathematically guaranteed equilibrium detector.

The code intentionally avoids a per-run equilibrium heuristic because that would add run-dependent ambiguity, extra passes over trajectories, and more complicated contracts. Instead, the late-time cutoff is explicit and user-controlled in config.

If the chosen sampler ranges or physics toggles produce trajectories that are still drifting materially after `1e12` seconds, increase the threshold and regenerate data.

## 6. Configuration Surface

All important runtime, generation, and training controls are exposed through config.

### 6.1 `paths`

- `project_root`
- `vulcan_source_path`
- `data_root`
- `models_root`
- `logs_root`

All configured paths must be relative.

### 6.2 `generation`

Controls the VULCAN-run stage and dataset sharding.

Important keys:

- `num_runs`
- `num_workers`
- `split_ratios`
- `random_seed`
- `run_timeout_seconds`
- `manifest_filename`
- `split_filename`
- `shard_size`
- `worker_root`
- `runs_root`
- `save_evo_frq`
- `keep_vulcan_outputs_debug`
- `failure_policy`

Important note: `save_evo_frq` is the effective saved-trajectory thinning control in the bundled VULCAN code used here.

### 6.3 `trajectory_sampling`

This section now defines a fixed late-time training regime.

Required keys:

- `mode = "fixed_dt_post_equilibrium"`
- `pairs_per_run`
- `fixed_requested_dt_s`
- `post_equilibrium_time_min_s`
- `post_equilibrium_min_fraction_of_final_time`
- `anchor_sampling = "uniform_valid_anchors"`
- `target_selection = "nearest_saved_snapshot"`
- `min_future_saved_steps`
- `max_target_relative_dt_error`
- `rollout_eval_points`

Semantics:

- `fixed_requested_dt_s`: the single requested horizon used to build training pairs.
- `post_equilibrium_time_min_s`: minimum absolute anchor time allowed for training/evaluation sampling.
- `post_equilibrium_min_fraction_of_final_time`: optional additional late-time threshold as a fraction of the trajectory end time.
- `min_future_saved_steps`: target must be at least this many saved steps after the anchor.
- `max_target_relative_dt_error`: maximum allowed relative mismatch between the requested `dt` and the snapped saved-state `actual_dt`.

### 6.4 `vulcan_runtime`

Controls the underlying VULCAN integration run.

Required keys:

- `runtime`
- `dt_min`
- `dt_max`
- `count_max`
- `trun_min`
- `count_min`
- `ini_mix`
- `atm_base`

Current repository rules:

- `ini_mix` is explicit in config but currently restricted to `"EQ"`.
- `atm_base` is exposed and validated against `H2`, `N2`, `O2`, `CO2`, and `H2O`.

### 6.5 `tp_sampler`

Defines the TP-profile family and its parameter distributions.

Important exposed knobs include:

- pressure grid: `nz`, `p_top_bar`, `p_bottom_bar`,
- temperature bounds,
- `max_sampling_attempts`,
- `adiabatic_gradient`,
- `convective_adjustment_probability`,
- `log10_kappa_ir`,
- `kappa_pressure_power_exponent`,
- `log10_gamma1`,
- `log10_gamma2`,
- `alpha_partition`,
- `t_int`,
- `t_irr`,
- `temperature_shift`.

All of these are sampled directly from config-defined distributions.

### 6.6 `gravity_sampler`

Supported modes:

- `uniform`
- `fixed`

### 6.7 `kzz_sampler`

Current mode:

- `power_law_profile`

Important exposed knobs:

- `log10_kzz_at_1bar_min`
- `log10_kzz_at_1bar_max`
- `beta_min`
- `beta_max`
- `kzz_floor_cm2_s`
- `kzz_cap_cm2_s`
- `pressure_unit_for_profile`

### 6.8 `abundance_sampler`

Important exposed knobs:

- `metallicity_mode`
- `log10_metallicity_min`
- `log10_metallicity_max`
- `c_to_o_min`
- `c_to_o_max`
- `solar_abundances`

The code samples metallicity and C/O, derives elemental abundances, and always runs VULCAN with `use_solar = False` so that these sampled values actually control the chemistry input.

### 6.9 `physics_toggles`

This section now exposes the simple non-photochemical VULCAN on/off controls that are structurally safe to map directly.

Supported keys:

- `use_photochemistry`
- `use_ion_chemistry`
- `use_eddy_diffusion`
- `use_molecular_diffusion`
- `use_upwind_molecular_diffusion`
- `use_boundary_conditions`
- `use_condensation`
- `use_settling`
- `use_initial_cold_trap`
- `use_sat_surface_h2o`
- `use_lowT_limit_rates`
- `use_adaptive_rtol`

Validation rules:

- `use_photochemistry` must currently be `false`.
- `use_ion_chemistry` must currently be `false`.
- `use_settling=true` requires `use_condensation=true`.
- `use_upwind_molecular_diffusion=true` requires `use_molecular_diffusion=true`.

### 6.10 `boundary_conditions`

Required only when `physics_toggles.use_boundary_conditions = true`.

Required keys:

- `use_topflux`
- `use_botflux`
- `top_BC_flux_file`
- `bot_BC_flux_file`
- `use_fix_sp_bot`

### 6.11 `data_spec`

Important keys:

- `state_species`
- `output_species`
- `required_input_profiles`
- `required_global_inputs`
- `required_state_inputs`
- `time_input_transform`
- `strict_non_finite`

The current main config uses:

- full compiled state/output species list,
- profile inputs: pressure, temperature, Kzz,
- global inputs: gravity, metallicity, C/O, `log10_dt_s`,
- state input: anchor `ymix`,
- `time_input_transform = "log10_dt_seconds"`.

### 6.12 `normalization`

Explicit per-variable normalization rules.

The main config currently uses:

- `pressure_bar`: `log-min-max`
- `temperature_k`: `standard`
- `kzz_cm2_s`: `log-standard`
- `anchor_ymix`: `log-standard`
- `gravity_cm_s2`: `log-standard`
- `metallicity_log10`: `standard`
- `c_to_o`: `standard`
- `log10_dt_s`: `standard`
- targets: `log-standard`

### 6.13 `precision`

Controls numeric dtypes for input tensors, stats accumulation, forward pass, loss, optimizer state, and optional autocast.

### 6.14 `training`

Important keys:

- `device`
- `gpu_preload`
- `batch_size`
- `epochs`
- `learning_rate`
- `min_lr`
- `warmup_epochs`
- `weight_decay`
- `gradient_clip`
- `use_amp`
- `num_workers`
- `seed`
- `data_loading`
- `model`
- `output_folder`

The shipped main config is GPU-prioritized:

- `device = "cuda"`
- `gpu_preload = true`
- `data_loading.mode = "ram"`
- `data_loading.use_device_prefetch = true`

## 7. VULCAN Physics Toggle Mapping

This section states exactly what each exposed non-photochemical switch does.

### 7.1 `use_eddy_diffusion`

Mapped to VULCAN `use_Kzz`.

Effect: includes vertical eddy-diffusion transport using the sampled `Kzz(p)` profile.

### 7.2 `use_molecular_diffusion`

Mapped to VULCAN `use_moldiff`.

Effect: includes species-dependent molecular diffusion in addition to bulk eddy transport.

### 7.3 `use_upwind_molecular_diffusion`

Mapped to VULCAN `use_vm_mol`.

Effect: enables VULCAN's alternate upwind treatment for molecular diffusion. The upstream VULCAN config comments label this path as under testing, so it is exposed but defaults to `false`.

### 7.4 `use_boundary_conditions`

Activates the optional `boundary_conditions` section.

Effect: allows top fluxes, bottom fluxes, and/or fixed lower-boundary species mixing ratios to be applied in VULCAN.

### 7.5 `use_condensation`

Mapped to VULCAN `use_condense`.

Effect: enables condensation reactions for configured condensable species.

### 7.6 `use_settling`

Mapped to VULCAN `use_settling`.

Effect: includes gravitational settling of condensate particles. This is only valid when condensation is also enabled.

### 7.7 `use_initial_cold_trap`

Mapped to VULCAN `use_ini_cold_trap`.

Effect: applies cold-trap logic during initialization of the equilibrium abundance field.

### 7.8 `use_sat_surface_h2o`

Mapped to VULCAN `use_sat_surfaceH2O`.

Effect: enables the surface H2O saturation treatment used by VULCAN for atmospheres where near-surface H2O should be saturation-limited.

### 7.9 `use_lowT_limit_rates`

Mapped to VULCAN `use_lowT_limit_rates`.

Effect: applies low-temperature limiting behavior to reaction rates where VULCAN supports it.

### 7.10 `use_adaptive_rtol`

Mapped to VULCAN `use_adapt_rtol`.

Effect: allows VULCAN to adjust the relative tolerance during long integrations.

### 7.11 `atm_base`

Mapped directly to VULCAN `atm_base`.

Effect: changes the assumed bulk atmospheric gas, which in turn changes molecular-diffusion coefficients, thermal-diffusion factors, and settling behavior inside VULCAN.

### 7.12 Pinned or intentionally unsupported controls

The following are intentionally pinned in this emulator branch.

- `use_photo = False`
- `use_ion = False`
- `use_vz = False`
- `atm_type = "file"`
- `Kzz_prof = "file"`
- `vz_prof = "const"`
- `const_vz = 0.0`
- `use_solar = False`
- `ini_mix = "EQ"`

Reasons:

- photochemistry and ion chemistry require a different conditioning surface and different data semantics,
- vertical advection requires an explicit velocity-profile contract that is not part of the current data model,
- `atm_type` and `Kzz_prof` are pinned to file-based inputs because the repository already generates explicit TP/Kzz profiles,
- `use_solar` is pinned off because abundances are sampled explicitly in this repository,
- non-`EQ` initialization modes require additional file/grid contracts that are intentionally excluded for now.

## 8. Raw Data Contract

Each successful VULCAN run is converted into one HDF5 file under the raw runs directory.

Required content:

- `inputs/pressure_bar`
- `inputs/temperature_k`
- `inputs/kzz_cm2_s`
- `inputs/state_species`
- `inputs/output_species`
- `globals/gravity_cm_s2`
- `globals/metallicity_log10`
- `globals/c_to_o`
- `trajectory/time_s`
- `trajectory/ymix_state`
- `trajectory/ymix_output`

Important details:

- number densities are converted to mixing ratios before writing,
- duplicate or non-increasing saved times are removed,
- the initial state at `t=0` is prepended from VULCAN `y_ini`,
- `ymix_output` is a possibly reordered subset of the full state according to `output_species`.

## 9. Processed Data Contract

Processed splits are stored in sharded `.npy` arrays with a metadata file per split.

Per-sample tensors are:

- sequence input: `[pressure_bar, temperature_k, kzz_cm2_s, anchor_ymix...]`
- globals: `[gravity_cm_s2, metallicity_log10, c_to_o, log10_dt_s]`
- targets: future `ymix` for `output_species`
- stored `dt_s`: the snapped **actual** `dt`, not the nominal requested `dt`

Per-split metadata includes:

- feature order,
- species order,
- output-from-state index mapping,
- normalization fingerprint,
- `dt_min_s`,
- `dt_max_s`,
- fixed requested `dt`,
- post-equilibrium thresholds,
- target-selection policy,
- mean and max relative `dt` snapping error.

## 10. Fixed-`dt` Late-Time Sampling

### 10.1 Current algorithm

For each raw VULCAN trajectory:

1. load the strictly increasing saved times,
2. compute the late-time anchor threshold as
   `max(post_equilibrium_time_min_s, post_equilibrium_min_fraction_of_final_time * t_final)`,
3. keep only anchors at or after that threshold and with at least `min_future_saved_steps` future saved states,
4. for each valid anchor, form one requested target time `t_anchor + fixed_requested_dt_s`,
5. snap that request to the nearest saved future state,
6. compute `actual_dt_s = t_target - t_anchor`,
7. optionally reject the pair if the relative mismatch exceeds `max_target_relative_dt_error`,
8. sample `pairs_per_run` anchors uniformly with replacement from the remaining valid anchor set.

The model is conditioned on `log10(actual_dt_s)`, not on the nominal requested `dt`.

### 10.2 Why a fixed requested `dt` is acceptable here

A fixed requested `dt` is acceptable for the current objective because the intended operating regime is a repeated coarse jump after the chemistry has largely relaxed.

This has several advantages.

- It simplifies the learning problem. The model does not need to disentangle a broad multi-decade time-scale family from the state transition problem at the same time.
- It improves data efficiency. Every sampled pair teaches the same nominal transition horizon.
- It stabilizes optimization. The model sees a narrower target family and can focus capacity on chemistry-state dependence.
- It still leaves some practical `dt` flexibility. Because the saved VULCAN times are irregular, the snapped `actual_dt_s` varies around the nominal requested value, and that snapped value is what the model actually conditions on.

This is therefore a **single-horizon surrogate with moderate local `dt` variation**, not a strict delta-function in `dt` and not a true arbitrary-time emulator.

### 10.3 Why the shipped config uses `1e12` seconds

The current default chooses `1e12` seconds for both the requested jump and the late-time cutoff because the repository is now targeting a late, slowly evolving regime rather than the full transient approach to equilibrium.

This is a heuristic default, not a theorem. It was chosen to make the training target coarse, operationally useful, and aligned with the long-runtime VULCAN integrations already configured in the project.

If the saved trajectories still show meaningful drift at or after `1e12` seconds for the sampled parameter space, the user should raise `post_equilibrium_time_min_s` and regenerate the data.

### 10.4 Rollout evaluation

Held-out rollout evaluation now uses the same fixed-`dt` late-time path construction rather than arbitrary full-trajectory checkpoints from `t=0`.

That makes evaluation consistent with the training regime.

## 11. Flexible `dt` Would Require More Than Snapping

The current code does **not** implement true flexible requested-time supervision.

To support flexible `dt` properly, the following changes would be required.

### 11.1 Requested-time supervision instead of nearest saved snapshot only

The label must correspond to the requested time itself, not merely to the nearest saved VULCAN snapshot. There are two clean ways to do that.

- Save trajectories densely enough that snapping error becomes negligible over the target range.
- Interpolate or otherwise reconstruct target states at the requested times from much denser saved solver trajectories.

### 11.2 Multi-`dt` sampling policy

The fixed-`dt` sampler would need to be replaced with a multi-horizon sampler, for example log-uniform or staged sampling over a configured `dt` range.

That means exposing and training over a `requested_dt` distribution rather than a single scalar.

### 11.3 Explicit requested-versus-actual semantics

The current processed data stores only the snapped `actual_dt_s` as the conditioning feature. A true flexible-`dt` design should usually retain both:

- requested `dt`,
- actual/interpolated label `dt`,
- snapping or interpolation error diagnostics.

### 11.4 Time-conditioning upgrades

The current scalar `log10_dt_s` conditioning is sufficient for the fixed-horizon regime. For broad multi-decade `dt` generalization, richer time embeddings may help, such as:

- a deeper conditioning MLP,
- learned time embeddings,
- random Fourier features,
- curriculum training over increasingly wide `dt` ranges.

### 11.5 Evaluation changes

Validation and test metrics would need to be stratified by requested `dt`, not just by observed snapped `dt`, and rollout tests would need to cover multiple horizon schedules.

## 12. Model Architecture

The surrogate model in `src/model.py` is an encoder-only transformer with FiLM conditioning.

Sequence inputs per level:

- pressure,
- temperature,
- Kzz,
- anchor chemistry state.

Global conditioning inputs:

- gravity,
- metallicity,
- C/O,
- `log10_dt_s`.

Architecture summary:

1. static profile channels and state channels are projected separately into `d_model`,
2. the sum is normalized and receives sinusoidal positional encoding over vertical level,
3. global scalars are passed through an MLP conditioning projector,
4. FiLM is applied once before the encoder and again after every transformer block,
5. an output MLP predicts a delta,
6. that delta is added to the anchor subset through a residual skip connection.

Important consequences:

- the model predicts a transition, not an absolute state from scratch,
- the time signal is emphasized globally through FiLM rather than being appended at every level,
- bidirectional attention is appropriate because the vertical column is not causal in level index,
- the design remains compatible with larger full-network state vectors.

## 13. Training and GPU Policy

The repository prioritizes GPU-resident training.

Main config defaults:

- CUDA device,
- RAM loading of processed shards,
- preload entire split tensors to device when requested,
- asynchronous device prefetch when loading from CPU-backed DataLoaders.

Operational rules:

- `gpu_preload=true` requires CUDA,
- `use_device_prefetch=true` requires CUDA,
- invalid precision/device combinations fail immediately,
- training uses vectorized tensor operations only; no Python per-sample chemistry logic is executed inside the model forward path.

The model is trained on normalized processed arrays. Validation/test metrics and rollout metrics are computed after decoding back to physical-space mixing ratios.

## 14. Logarithm Convention

All logarithms in this repository are base-10.

This includes:

- configuration keys named `log10_*`,
- abundance controls,
- Kzz controls,
- `log10_dt_s`,
- log-based normalization modes.

## 15. Correctness Fixes Implemented In This Revision

This revision intentionally fixes the following issues.

1. Removed the misleading variable-`dt` training semantics. The training pipeline now explicitly uses a single fixed requested horizon plus late-time anchors.
2. Aligned evaluation rollouts with the training semantics.
3. Removed `y_time_freq` from the emulator config surface because the bundled VULCAN implementation used here does not actually honor it for saved chemistry evolution; `save_evo_frq` is the meaningful control.
4. Split the old coarse runtime booleans into the actual VULCAN toggle surface: eddy diffusion, molecular diffusion, upwind molecular diffusion, boundary conditions, condensation, settling, cold trap, surface H2O saturation, low-temperature rate limits, adaptive tolerance.
5. Exposed `atm_base` and kept `ini_mix` explicit in config.
6. Fixed preflight so it exercises the same runtime-setting surface used by real generation.
7. Added fixed-`dt` metadata diagnostics, including mean/max relative snapping error.
8. Updated smoke tests and contract tests to match the new semantics.
9. Switched the main config to the full currently compiled species list instead of the earlier reduced 20-species subset.

## 16. Known Limitations

- Photochemistry is still disabled by design.
- Ion chemistry is still disabled by design.
- Vertical advection is not wired.
- Only `ini_mix = "EQ"` is supported.
- Flexible requested-time supervision is not implemented.
- The chemistry network is whatever the local VULCAN tree has already compiled; this repository does not rebuild it automatically.
- The late-time threshold is heuristic and must be checked against the user's sampled regime.

## 17. Operational Commands

### 17.1 Local Development

Generate raw data and processed shards:

```bash
python src/main.py --gen
```

Train the surrogate:

```bash
python src/main.py --train
```

The expected local workflow is:

1. validate or edit `config/config.json`,
2. run `--gen`,
3. inspect generated-data summaries and late-time `dt` diagnostics,
4. run `--train`,
5. evaluate saved rollout metrics and held-out physical-space errors.

### 17.2 HPC / PBS Batch Deployment

The repository ships `run.pbs` for PBS-managed HPC clusters. Submit with:

```bash
qsub run.pbs
```

#### Environment Variables

Three environment variables control HPC-specific path and environment resolution. All are optional for local development but are set automatically by `run.pbs` on cluster nodes.

| Variable | Purpose | Default (if unset) |
|---|---|---|
| `VULCAN_EMULATOR_PROJECT_ROOT` | Override the project root directory. When set, `src/main.py` and `src/path_utils.py` use this instead of deriving root from `__file__`. Paths are normalized with `os.path.normpath` (no symlink resolution). | Derived from `Path(__file__).resolve().parent.parent` |
| `VULCAN_EMULATOR_VULCAN_SOURCE` | Override the VULCAN source tree path. Takes precedence over `paths.vulcan_source_path` in config. | Joined from project root + config `paths.vulcan_source_path` |
| `VULCAN_EMULATOR_CONDA_ENV` | Override the expected conda environment name for the preflight check in `src/main.py`. | `"nn"` |

#### Why `os.path.normpath` Instead of `Path.resolve()`

HPC filesystems frequently use symlinks for canonical mount points (e.g., `/nobackupp19/` vs. `/home5/`). `Path.resolve()` follows symlinks, which can rewrite the project root to a canonical path that differs from the PBS working directory (`$PBS_O_WORKDIR`). This breaks sibling-directory references like `../VULCAN`.

All path joins in `src/path_utils.py`, `src/main.py`, and `src/preprocess.py` use `os.path.normpath` to clean `..` segments without symlink traversal when an environment-variable override is active.

#### `run.pbs` Workflow

The shipped `run.pbs` script performs the following steps:

1. Sets `VULCAN_EMULATOR_PROJECT_ROOT` from `$PBS_O_WORKDIR`.
2. Loads the site conda module and activates the configured environment (default: `pyt2_8_gh`).
3. Exports `VULCAN_EMULATOR_CONDA_ENV` so the Python preflight accepts the HPC environment.
4. Validates the VULCAN source tree and builds FastChem if the binary is missing.
5. Runs a Python preflight that checks all required packages and GPU availability.
6. Executes `python -u src/main.py --gen` (data generation).
7. Executes `python -u src/main.py --train` (model training).

Either stage can be skipped by commenting out the corresponding line in `run.pbs`.

#### Resource Defaults

The shipped PBS directives request:

- Queue: `gpu_long`
- 1 node, 32 CPUs, 1 GPU (GH200), 400 GB RAM
- 72-hour walltime

Adjust these to match your site's queue names and hardware.
