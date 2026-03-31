# Config Reference

The repository ships one canonical config:

- `equilibrium_only_config.json`

All paths are resolved relative to the project root (the directory containing `spec.md`
and `pyproject.toml`).

---

## Top-Level Schema

Every config uses the same shared top-level sections:

- `task`
- `paths`
- `data_spec`
- `sampling`
- `temperature_profiles`
- `generation`
- `normalization`
- `training`

Task-specific settings live under exactly one additional block:

- `full_vulcan` (when `task.kind = "full_vulcan"`)
- `equilibrium_only` (when `task.kind = "equilibrium_only"`)

The config for the *other* task kind must **not** be present.

---

## `task`

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `kind` | string | yes | `"full_vulcan"` or `"equilibrium_only"`. Controls model selection, data contract, validation rules, and which task-specific config section is expected. |

---

## `paths`

All paths are strings, resolved relative to the project root unless absolute.

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `raw_root` | string | yes | Directory for raw HDF5 run files and generation manifests. Example: `"data/raw/equilibrium_only"`. |
| `processed_root` | string | yes | Directory for processed tensors, normalization metadata, and split directories (train/val/test). Example: `"data/processed/equilibrium_only"`. |
| `checkpoints_root` | string | yes | Directory for model checkpoints (`best.pt`, `last.pt`), training history, and metrics. Example: `"models/equilibrium_only"`. |
| `vulcan_source_root` | string | yes | Path to the VULCAN source tree (used by the generation stage to locate `vulcan.py` and `vulcan_cfg.py`). Example: `"../VULCAN-master"`. |

---

## `data_spec`

Defines the chemical species tracked by the emulator.

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `state_species` | list[string] | yes | Ordered list of chemical species tracked in the atmospheric state vector. These are the species VULCAN evolves. Default: 17 NCHO+S species (H2, He, H, O, OH, H2O, CO, CO2, CH4, N2, NH3, H2S, SH, S, SO, SO2, S2). |
| `output_species` | list[string] | yes | Ordered list of species the model predicts. Usually identical to `state_species` but can be a subset. |

**Internal derived metadata** (computed during validation, not valid user config keys):

- `state_dim`: number of state species.
- `target_dim`: number of output species.
- `element_input_order`: fixed FastChem-native elemental abundance channels, always `["He_H", "C_H", "O_H", "N_H", "S_H"]`.
- `required_global_inputs`: derived conditioning scalar order. For equilibrium this is the fixed element list above. For full_vulcan it prepends `gravity_cm_s2`, appends `log10_dt_s`, and includes the physics-toggle and atmosphere-base flags.
- `sequence_static_feature_order`: ordered names of per-level input columns.
- `global_static_feature_order`: global feature names excluding `log10_dt_s`.
- `global_feature_order`: full global feature names including `log10_dt_s`.
- `dt_feature_index`: index of `log10_dt_s` in the global vector (null for equilibrium).

These fields appear in validated configs, processed metadata, and exported bundles for
self-description, but they must not be provided manually in the user JSON.

---

## `sampling`

Controls the atmospheric parameter space from which training data is drawn.

### Shared fields (both tasks)

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `num_levels` | int | yes | Number of vertical pressure levels in each atmospheric column. Must be >= 4. Example: `64`. |
| `pressure_top_bar` | float | yes | Lowest pressure in bar (top of atmosphere). Must be positive and less than `pressure_bottom_bar`. Example: `1e-7`. |
| `pressure_bottom_bar` | float | yes | Highest pressure in bar (bottom of atmosphere). Must be positive. Example: `100.0`. |
| `temperature_range_k` | [float, float] | yes | Hard bounds on temperature in Kelvin. Used for validation. Example: `[500.0, 2500.0]`. |
| `metallicity_log10_range` | [float, float] | yes | Generation-only sampler range for the latent metallicity control used to synthesize explicit elemental abundances. Example: `[0.0, 1.0]`. |
| `c_to_o_range` | [float, float] | yes | Generation-only sampler range for the latent carbon-to-oxygen control used to synthesize explicit elemental abundances. Example: `[0.45, 0.85]`. |
| `s_to_o_range` | [float, float] | yes | Generation-only sampler range for the latent sulfur-to-oxygen control used to synthesize explicit elemental abundances. Example: `[0.01, 0.03]`. |

### Full-VULCAN only fields

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `gravity_range_cm_s2` | [float, float] | yes | Surface gravity range in cm/s^2. Example: `[500.0, 5000.0]`. |
| `kzz_cm2_s` | float | yes | Constant eddy diffusion coefficient in cm^2/s. Must be positive. Example: `1e10`. |
| `num_time_steps` | int | yes | Number of saved timesteps per VULCAN trajectory. Must be >= 3. |
| `time_step_log10_min_s` | float | yes | Minimum log10 timestep (seconds) for the random time grid. |
| `time_step_log10_max_s` | float | yes | Maximum log10 timestep (seconds) for the random time grid. |

---

## `temperature_profiles`

Unifies two temperature profile sources: analytic (parameterized) and PT-library
(externally computed GCM profiles).

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `source_mode` | string | yes | `"analytic"` (Line et al. 2013 only), `"pt_library"` (Roth .dat files only), or `"mixed"` (both, selected per-run by probability). |
| `analytic_probability` | float | mixed only | Probability of sampling an analytic profile in mixed mode. Must be strictly between 0 and 1. Example: `0.5`. |
| `data_glob` | string | pt_library/mixed | Glob pattern for Roth .dat files, relative to project root. Example: `"assets/PTprofiles/*.dat"`. |
| `filters` | object | no | Metadata filters applied to PT-library profiles. See below. |
| `analytic_sampler` | object | analytic/mixed | Parameters for the Line et al. 2013 analytic sampler. See below. |

### `temperature_profiles.filters`

Optional filters on Roth profile metadata parsed from filenames.

| Key | Type | Description |
|-----|------|-------------|
| `Teq` | float or [float, float] | Equilibrium temperature filter. Scalar = exact match, list = inclusive range. |
| `LogMet` | float or [float, float] | Log metallicity filter. |
| `LogDrag` | float or [float, float] | Log drag coefficient filter. |
| `Mstar` | float or [float, float] | Stellar mass filter. |
| `Rp` | float or [float, float] | Planet radius filter. |
| `logG` | float or [float, float] | Log surface gravity filter. |
| `TiOVO` | bool | Whether TiO/VO opacity is included. |

### `temperature_profiles.analytic_sampler`

Parameters controlling the Line et al. (2013) analytic PT profile generator.
All logarithms are base-10.

| Key | Type | Description |
|-----|------|-------------|
| `reference_gravity_m_s2` | float | Reference surface gravity for optical depth integration (m/s^2). Must be positive. Default: `25`. |
| `t_int_k_normal` | `{mean, std}` | Normal distribution for the internal heat flux temperature T_int (K). Default: `{mean: 500, std: 150}`. |
| `t_irr_k_normal` | `{mean, std}` | Normal distribution for the irradiation temperature T_irr (K). Default: `{mean: 1800, std: 500}`. |
| `log10_kappa_ir_m2_kg_normal` | `{mean, std}` | Normal distribution for log10 of the infrared opacity (m^2/kg). Default: `{mean: -2.5, std: 2.5}`. |
| `power_law_n_range` | [float, float] | Uniform range for the pressure power-law exponent n. Must be > 0. Default: `[0.5, 2.0]`. |
| `log10_gamma_1_range` | [float, float] | Uniform range for log10 of the first visible-channel opacity ratio. Default: `[-2.0, 2.0]`. |
| `log10_gamma_2_range` | [float, float] | Uniform range for log10 of the second visible-channel opacity ratio. Default: `[-2.0, 2.0]`. |
| `alpha_range` | [float, float] | Uniform range for the visible-channel partition fraction. Must lie within [0, 1]. Default: `[0.0, 1.0]`. |
| `temperature_shift_k_range` | [float, float] | Uniform range for a constant temperature offset (K) applied to the entire profile. Default: `[-600.0, 600.0]`. |
| `convection_probability` | float | Probability that convective adjustment is applied to a profile. Must lie in [0, 1]. Default: `0.333`. |
| `adiabatic_gradient_range` | [float, float] | Uniform range for the adiabatic gradient d(ln T)/d(ln P). Must be > 0. Default: `[0.25, 0.35]`. |
| `validation` | object | Rejection-sampling bounds. Profiles outside these limits are discarded and resampled. |
| `validation.min_temperature_k` | float | Minimum allowed temperature anywhere in the profile. Default: `50.0`. |
| `validation.max_temperature_k` | float | Maximum allowed temperature anywhere in the profile. Default: `4000.0`. |
| `validation.min_bottom_temperature_k` | float | Minimum temperature at the bottom of the column. Default: `1000.0`. |
| `validation.max_top_temperature_k` | float | Maximum temperature at the top of the column. Default: `2800.0`. |

---

## `generation`

Controls the raw data generation stage.

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `mode` | string | yes | `"vulcan"` (run VULCAN externally) or `"synthetic"` (in-process smoke generation). |
| `num_runs` | int | yes | Total number of atmospheric profiles to generate. Must be >= 1. |
| `seed` | int | yes | Random seed for reproducible sampling of run specifications. |
| `overwrite` | bool | yes | If true, regenerate raw data even if the output directory already exists. |
| `reuse_raw_if_present` | bool | yes | If true and raw data already exists, skip generation entirely. Takes precedence over `overwrite` when both are true. |
| `parallel_workers` | int | yes | Number of parallel VULCAN worker processes. Must be >= 1. Has no effect in `synthetic` mode. |

**Derived field** (set during validation):

- `target_mode`: `"equilibrium_only"` or `"trajectory"`, derived from `task.kind`.

---

## `normalization`

Controls the raw-to-processed conversion stage: train/val/test splitting, normalization
fitting, and tensor export.

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `split.train_fraction` | float | yes | Fraction of runs allocated to the training split. Must be positive. All three fractions must sum to 1. |
| `split.val_fraction` | float | yes | Fraction of runs allocated to the validation split. |
| `split.test_fraction` | float | yes | Fraction of runs allocated to the test split. |
| `split.seed` | int | yes | Random seed for the train/val/test shuffle. |
| `state_floor` | float | yes | Minimum value used before taking log10 of mixing ratios. Prevents log(0). Must be positive. Example: `1e-30`. |
| `spectrum_floor` | float | full_vulcan only | Minimum value for stellar spectrum normalization. Must be positive. |
| `sequence_methods` | object | yes | Per-column normalization method for sequence-static features. Keys must be exactly `{pressure_bar, temperature_k}` for equilibrium or `{pressure_bar, temperature_k, kzz_cm2_s}` for full_vulcan. Allowed values: `"standard"`, `"log-standard"`, `"none"`. |
| `global_methods` | object | yes | Per-feature normalization method for the derived global static feature order. Keys must match the derived `global_static_feature_order`. Allowed values: `"standard"`, `"log-standard"`, `"none"`. |
| `target_method` | string | yes | Normalization method for target mixing ratios. Allowed values: `"standard"`, `"log-standard"`, `"none"`. |
| `state_method` | string | full_vulcan only | Normalization method for trajectory state inputs. Allowed values: `"standard"`, `"log-standard"`, `"none"`. |
| `spectrum_method` | string | full_vulcan only | Normalization method for stellar spectrum inputs. Allowed values: `"standard"`, `"log-standard"`, `"none"`. |
| `log10_dt_method` | string | full_vulcan only | Normalization method for the already-log10 timestep feature. Allowed values: `"standard"`, `"none"`. |

---

## `training`

Controls the model training stage.

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `seed` | int | yes | Random seed for parameter initialization, data shuffling, and batch sampling. |
| `batch_size` | int | yes | Number of profiles (or transition pairs) per mini-batch. Must be >= 1. |
| `epochs` | int | yes | Number of full passes through the training data. Must be >= 1. |
| `learning_rate` | float | yes | Peak learning rate for the AdamW optimizer. Must be positive. |
| `min_lr` | float | yes | Minimum learning-rate floor used by cosine annealing or reduce-on-plateau. Must be positive and <= `learning_rate`. |
| `warmup_epochs` | int | yes | Number of epochs for linear warmup from 0 to `learning_rate`. |
| `scheduler.name` | string | no | Post-warmup scheduler. `"reduce_on_plateau"` is the default; `"cosine"` preserves the old cosine-annealing behavior. |
| `scheduler.factor` | float | reduce_on_plateau only | Multiplicative LR drop applied after a plateau. Must lie strictly between 0 and 1. Default: `0.5`. |
| `scheduler.patience` | int | reduce_on_plateau only | Number of post-warmup epochs without sufficient validation improvement before reducing LR. Must be >= 0. Default: `10`. |
| `scheduler.threshold` | float | reduce_on_plateau only | Minimum absolute validation-loss improvement that resets plateau tracking. Must be >= 0. Default: `1e-4`. |
| `weight_decay` | float | yes | L2 weight decay coefficient for AdamW. |
| `gradient_clip` | float | yes | Maximum global L2 norm for gradient clipping. Must be positive. |
| `live_sampling.train_pairs_per_run_per_epoch` | int | yes | Number of (anchor, target) transition pairs sampled per run per epoch. Must be >= 1. |
| `live_sampling.eval_pairs_per_run` | int | yes | Number of transition pairs per run for validation/test evaluation. Must be >= 1. |
| `loss.lambda_z` | float | yes | Weight on the MSE loss in normalized space (primary training signal). Must be >= 0. |
| `loss.lambda_phys` | float | yes | Weight on the MSE loss in log10 mixing-ratio space (physical-scale diagnostic). Must be >= 0. |
| `loss.lambda_spectrum` | float | full_vulcan only | Weight on the spectrum autoencoder reconstruction loss. Must be >= 0. Set to 0 if `spectrum_encoder_mode = "none"`. |

---

## `equilibrium_only`

Task-specific configuration for the equilibrium chemistry MLP.

### `equilibrium_only.model`

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `activation` | string | no | Hidden activation function: `"gelu"`, `"relu"`, or `"silu"`. Default: `"gelu"`. |
| `d_hidden` | int | yes | Width of each hidden layer in the per-level MLP. Must be >= 8. |
| `num_hidden_layers` | int | yes | Number of hidden layers in the per-level MLP. Must be >= 1. |
| `conditioning_hidden_dim` | int | yes | Width of the FiLM conditioning MLP that maps global inputs to per-layer gamma/beta. Must be >= 1. |
| `film_clamp` | float | yes | Symmetric clamp applied to FiLM gamma and beta values. Must be positive. Limits the magnitude of the affine modulation. |

---

## `full_vulcan`

Task-specific configuration for the full trajectory Transformer.

### `full_vulcan.model`

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `d_model` | int | yes | Hidden width of the Transformer backbone. Must be >= 8. |
| `nhead` | int | yes | Number of attention heads. Must divide `d_model` evenly. Must be >= 1. |
| `num_layers` | int | yes | Number of Transformer encoder blocks. Must be >= 1. |
| `dim_feedforward` | int | yes | Width of each block's feed-forward sub-layer. Must be >= `d_model`. |
| `conditioning_hidden_dim` | int | yes | Width of the FiLM conditioning MLP. Must be >= 1. |
| `film_clamp` | float | yes | Symmetric clamp on FiLM gamma/beta. Must be positive. |
| `output_head_divisor` | int | yes | The output head bottleneck width is `d_model // output_head_divisor`. Must be >= 1. |
| `activation` | string | no | Hidden activation: `"gelu"`, `"relu"`, or `"silu"`. Default: `"gelu"`. |

### `full_vulcan.physics_toggles`

Boolean flags controlling which VULCAN physics modules are enabled during generation.
Each defaults to `false` if omitted.

Supported toggles: `use_photochemistry`, `use_ion_chemistry`, `use_eddy_diffusion`,
`use_molecular_diffusion`, `use_upwind_molecular_diffusion`, `use_boundary_conditions`,
`use_condensation`, `use_settling`, `use_initial_cold_trap`, `use_sat_surface_h2o`,
`use_lowT_limit_rates`, `use_adaptive_rtol`.

### `full_vulcan.vulcan_runtime`

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `python_executable` | string | yes | Python interpreter for VULCAN subprocesses. |
| `cfg_file` | string | yes | Name of the VULCAN config file to patch. |
| `chemistry_file` | string | yes | Path to the VULCAN chemistry network file. |
| `worker_root` | string | yes | Temporary directory root for parallel VULCAN workers. |
| `regenerate_chem_funs` | bool | yes | Whether to regenerate `chem_funs.py` before the first run. |
| `atm_base` | string | yes | Bulk atmospheric gas: `"H2"`, `"N2"`, `"O2"`, `"CO2"`, or `"H2O"`. |
| `t_cross_sp` | list[string] | yes | Species included in the cross-section tables. |
| `cfg_assignments` | object | yes | Arbitrary key-value pairs injected into the VULCAN config. |

### `full_vulcan.stellar_spectrum`

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `enabled` | bool | yes | Whether stellar spectrum conditioning is active. |
| `template_name` | string | yes | Human-readable name for the spectrum template. |
| `template_file` | string | yes | Path to the spectrum text file (wavelength vs flux), relative to project root. |
| `num_bins` | int | yes | Number of wavelength bins for the resampled spectrum. Must be >= 8. |
| `wavelength_min_nm` | float | yes | Minimum wavelength in nm for the fixed grid. |
| `wavelength_max_nm` | float | yes | Maximum wavelength in nm. Must be > `wavelength_min_nm`. |
| `encoder_mode` | string | yes | Spectrum encoding strategy: `"autoencoder"`, `"linear"`, or `"none"`. |
| `latent_dim` | int | yes | Dimension of the spectrum latent vector. Must be in [1, num_bins]. |
| `hidden_dim` | int | yes | Hidden width inside the spectrum autoencoder (ignored if not autoencoder mode). |
| `teff_k` | float | yes | Stellar effective temperature in Kelvin. Must be positive. |
| `radius_rsun` | float | yes | Stellar radius in solar radii. Must be positive. |
| `semi_major_axis_au` | float | yes | Orbital semi-major axis in AU. Must be positive. |
| `zenith_angle_deg` | float | yes | Stellar zenith angle in degrees. Must be in [0, 90). |
| `diurnal_factor` | float | yes | Diurnal averaging factor for incident flux. Must be positive. |

### `full_vulcan.trajectory_sampling`

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `dt_min_s` | float | yes | Minimum allowed timestep in seconds for transition pairs. Must be positive. |
| `dt_max_s` | float | yes | Maximum allowed timestep. Must be > `dt_min_s`. |
| `min_future_saved_steps` | int | yes | Minimum number of saved steps after the anchor for a valid pair. Must be >= 1. |
| `num_logdt_bins` | int | yes | Number of log10(dt) bins for stratified sampling. Must be >= 1. |
