# Config Helper

## Supported Tasks

Use exactly one of:

- `task.kind = "equilibrium_only"`
- `task.kind = "full_vulcan"`

There is no supported transition, trajectory, timestep-control, or live-sampling mode.

## Shared Top-Level Sections

Every config contains:

- `task`
- `paths`
- `data_spec`
- `sampling`
- `temperature_profiles`
- `generation`
- `normalization`
- `training`

Plus exactly one task-specific section:

- `equilibrium_only`
- `full_vulcan`

## Fixed Internal Element Order

The elemental input order is fixed in code and must not be set in `data_spec`:

```python
["He_H", "C_H", "O_H", "N_H", "S_H"]
```

These values use the FastChem-native hydrogen-normalized number abundance `n_X / n_H`.

## `data_spec`

Public keys:

- `state_species`
- `output_species`

Do not set these keys manually:

- `required_global_inputs`
- `element_input_order`

They are derived internally.

## `sampling`

Shared keys:

- `num_levels`
- `pressure_top_bar`
- `pressure_bottom_bar`
- `temperature_range_k`
- `metallicity_log10_range`
- `c_to_o_range`
- `s_to_o_range`

Additional `full_vulcan` keys:

- `gravity_range_cm_s2`
- `kzz_cm2_s`

Removed legacy keys:

- `num_time_steps`
- `time_step_log10_min_s`
- `time_step_log10_max_s`

## `temperature_profiles`

Supported source modes:

- `analytic`
- `pt_library`
- `mixed`

Shared public keys:

- `source_mode`
- `validation`
- `filters`

For `analytic` or `mixed`, include:

- `analytic_sampler`

For `pt_library` or `mixed`, include:

- `data_glob`

For `mixed`, include:

- `analytic_probability`

## `generation`

Supported keys:

- `mode`: `synthetic` or `vulcan`
- `num_runs`
- `seed`
- `overwrite`
- `reuse_raw_if_present`
- `parallel_workers`
- `backfill.enabled`
- `backfill.max_retries`

Removed legacy key:

- `target_mode`

Generation is final-state-only. Failed runs are never included. If the exact requested successful run count is not reached after backfill, generation fails.

## `normalization`

Shared public keys:

- `split`
- `state_floor`
- `sequence_methods`
- `global_methods`
- `target_method`

Additional `full_vulcan` keys:

- `spectrum_floor`
- `spectrum_method`

Supported method names:

- `standard`
- `log-standard`
- `none`

Removed legacy keys:

- `state_method`
- `log10_dt_method`

Typical chemistry scaling is `log-standard`, which corresponds to `log10` plus z-score.

## `training`

Supported keys:

- `seed`
- `batch_size`
- `epochs`
- `learning_rate`
- `min_lr`
- `warmup_epochs`
- `scheduler`
- `weight_decay`
- `gradient_clip`
- `loss`

Removed legacy key:

- `live_sampling`

Supported scheduler names:

- `reduce_on_plateau`
- `cosine`

## `equilibrium_only.model`

Required keys:

- `d_hidden`
- `num_hidden_layers`
- `conditioning_hidden_dim`
- `film_clamp`

Optional key:

- `activation`

## `full_vulcan.model`

Required keys:

- `d_model`
- `nhead`
- `num_layers`
- `dim_feedforward`
- `conditioning_hidden_dim`
- `film_clamp`
- `output_head_divisor`

Optional key:

- `activation`

## Supported Activations

The same activation names are supported for both model families:

- `relu`
- `gelu`
- `silu`
- `tanh`
- `elu`
- `selu`
- `softplus`
- `leaky_relu`

Use lowercase strings. The default activation for both model families is `leaky_relu`.

## `full_vulcan.physics_toggles`

Public scientific toggles:

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

These toggles are part of the model conditioning contract.

## `full_vulcan.vulcan_runtime`

Public scientific keys:

- `chemistry_file`
- `atm_base`
- `t_cross_sp`

Internal defaults that are not part of the public config surface:

- `python_executable`
- `cfg_file`
- `worker_root`
- `regenerate_chem_funs`
- `cfg_assignments`
- `use_lowT_limit_rates`
- `use_adaptive_rtol`

Supported `atm_base` values:

- `H2`
- `N2`
- `O2`
- `CO2`
- `H2O`

## `full_vulcan.stellar_spectrum`

Required keys:

- `enabled`
- `template_name`
- `template_file`
- `num_bins`
- `encoder_mode`
- `latent_dim`
- `hidden_dim`
- `teff_k`
- `radius_rsun`
- `semi_major_axis_au`
- `zenith_angle_deg`
- `diurnal_factor`
- `wavelength_min_nm`
- `wavelength_max_nm`

Supported encoder modes:

- `autoencoder`
- `linear`
- `none`

## Normalization Expectations

The model input blocks are:

- equilibrium sequence: `[pressure_bar, temperature_k]`
- full-VULCAN sequence: `[pressure_bar, temperature_k, kzz_cm2_s]`
- equilibrium globals: fixed-order elemental abundances
- full-VULCAN globals: gravity, fixed-order elemental abundances, public physics toggles, atmosphere-base one-hot
- full-VULCAN spectrum: resampled stellar flux

Your `normalization.global_methods` keys must match the derived global feature order exactly.

## Shipped Configs

Supported example configs:

- `config/equilibrium_only_config.json`
- `config/full_vulcan_config.json`
