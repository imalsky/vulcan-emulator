# Full-VULCAN Config Helper

Use `full_vulcan_config.json` when you want the emulator to predict the final
converged VULCAN abundances for a full atmospheric column.

This mode is the final-state VULCAN path:

- output target: final converged VULCAN abundances
- model family: FiLM-conditioned Transformer
- vertical coupling: yes, through attention over the full column
- required inputs at inference time: pressure, temperature, Kzz, `metallicity_log10`, `c_to_o`, `s_to_o`, gravity, stellar spectrum
- additional conditioning: public physics toggles and atmosphere-base flags

There is no supported timestep-control, anchor-state, or trajectory mode in
this config.

## Top-Level Layout

The full-VULCAN config contains these sections:

- `task`
- `paths`
- `data_spec`
- `sampling`
- `temperature_profiles`
- `generation`
- `normalization`
- `training`
- `full_vulcan`

## `task`

- `kind`

Set `task.kind = "full_vulcan"`.

## `paths`

These control where the pipeline reads and writes data:

- `raw_root`: raw generated runs
- `processed_root`: processed train/val/test arrays
- `checkpoints_root`: saved checkpoints and metrics
- `vulcan_source_root`: location of the VULCAN source tree used during generation

## `data_spec`

These define the species layout used throughout the pipeline:

- `state_species`: full species ordering stored in raw and processed data
- `output_species`: species ordering predicted by the emulator

Do not add `required_global_inputs` or `element_input_order`. Those are
derived internally from the fixed public contract.

## `sampling`

These set the atmosphere and transport parameter ranges:

- `num_levels`: number of vertical layers
- `pressure_top_bar`: pressure at the top layer
- `pressure_bottom_bar`: pressure at the bottom layer
- `temperature_range_k`: allowed temperature range
- `metallicity_log10_range`: metallicity sampler range
- `c_to_o_range`: carbon-to-oxygen ratio range
- `s_to_o_range`: sulfur-to-oxygen ratio range
- `gravity_range_cm_s2`: gravity sampler range
- `kzz_cm2_s`: eddy-diffusion coefficient used in the sampled columns

As in the equilibrium path, the generator samples chemistry from metallicity,
C/O, and S/O and then materializes the fixed internal elemental channels for
VULCAN. The learned conditioning contract, however, uses the ratio globals
plus gravity and the selected public physics/base controls.

## `temperature_profiles`

This controls how temperature profiles are chosen:

- `source_mode`: `analytic`, `pt_library`, or `mixed`
- `validation`: hard temperature sanity bounds
- `filters`: optional PT-library filters
- `analytic_sampler`: parameters for the analytic profile generator
- `data_glob`: PT-library file pattern
- `analytic_probability`: only used when `source_mode = "mixed"`

## `generation`

This controls raw-dataset creation:

- `mode`: `synthetic` or `vulcan`
- `num_runs`: requested successful runs
- `seed`: generation seed
- `overwrite`: regenerate even if outputs already exist
- `reuse_raw_if_present`: skip generation when the raw dataset already exists
- `parallel_workers`: generation worker count
- `backfill.enabled`: retry failed generations with new samples
- `backfill.max_retries`: maximum retry rounds

Generation is final-state-only. Failed runs are excluded, and the generator
fails if it cannot reach the requested number of successful runs after
backfill.

## `normalization`

This controls how inputs and targets are scaled before training:

- `split`: train/val/test fractions and split seed
- `state_floor`: lower bound applied before log scaling chemistry targets
- `target_method`: target scaling method
- `sequence_methods`: scaling per sequence feature
- `global_methods`: scaling per global feature
- `spectrum_floor`: lower bound applied before log scaling stellar spectrum values
- `spectrum_method`: stellar spectrum scaling method

Supported normalization methods are:

- `standard`
- `log-standard`
- `none`

For the supported ratio-global chemistry contract, `metallicity_log10`,
`c_to_o`, and `s_to_o` are typically scaled with `standard`, while gravity
commonly uses `log-standard`.

## `training`

This controls optimization:

- `seed`
- `batch_size`
- `epochs`
- `learning_rate`
- `min_lr`
- `warmup_epochs`
- `early_stopping_patience`
- `scheduler`
- `weight_decay`
- `gradient_clip`
- `loss`

`loss` contains:

- `lambda_z`: weight on normalized-space chemistry MSE
- `lambda_phys`: weight on denormalized log10-space chemistry MSE
- `lambda_spectrum`: weight on stellar-spectrum reconstruction loss

Supported schedulers are:

- `reduce_on_plateau`
- `cosine`

`early_stopping_patience` stops training after that many consecutive epochs
without improvement in the validation checkpoint metric. The default is `30`.

## `full_vulcan.model`

This defines the Transformer:

- `d_model`: hidden width
- `nhead`: number of attention heads
- `num_layers`: number of Transformer blocks
- `dim_feedforward`: feedforward width inside each block
- `conditioning_hidden_dim`: hidden size of the FiLM conditioning network
- `film_clamp`: clamp applied to FiLM modulation values
- `output_head_divisor`: output-head width control
- `activation`: activation function, default `leaky_relu`
- `dropout_rate`: hidden-activation dropout probability, default `0.05`

Supported activations are:

- `relu`
- `gelu`
- `silu`
- `tanh`
- `elu`
- `selu`
- `softplus`
- `leaky_relu`

## `full_vulcan.physics_toggles`

These are the public scientific switches that become part of the model
conditioning vector:

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

Use this block to decide which VULCAN physical processes are active in the raw
dataset and in the learned emulator contract.

## `full_vulcan.vulcan_runtime`

These are the public scientific runtime knobs:

- `chemistry_file`: chemistry network file
- `atm_base`: base atmosphere
- `t_cross_sp`: species used in cross-section construction

Supported `atm_base` values are:

- `H2`
- `N2`
- `O2`
- `CO2`
- `H2O`

Internal runtime defaults such as `python_executable`, `cfg_file`,
`worker_root`, `regenerate_chem_funs`, `cfg_assignments`,
`use_lowT_limit_rates`, and `use_adaptive_rtol` are not part of the public
config surface.

## `full_vulcan.stellar_spectrum`

This block controls the stellar spectrum supplied to the emulator:

- `enabled`: include stellar-spectrum conditioning
- `template_name`: short label for the selected template
- `template_file`: source file for the stellar spectrum
- `num_bins`: output wavelength bins
- `latent_dim`: encoder latent width
- `hidden_dim`: encoder hidden width
- `teff_k`: stellar effective temperature
- `radius_rsun`: stellar radius
- `semi_major_axis_au`: orbital distance
- `diurnal_factor`: day-night averaging factor
- `wavelength_min_nm`: lower wavelength bound
- `wavelength_max_nm`: upper wavelength bound
- `encoder_mode`: spectrum encoder mode
- `zenith_angle_deg`: irradiation zenith angle

Use this block when you want the chemistry emulator to be conditioned on the
stellar irradiation field used by VULCAN.
