# VULCAN Emulator Spec

## Overview

This repository supports exactly two emulator tasks:

- `task.kind = "equilibrium_only"`: FastChem equilibrium chemistry from pressure, temperature, and fixed-order elemental abundances.
- `task.kind = "full_vulcan"`: final converged VULCAN chemistry from pressure, temperature, Kzz, gravity, elemental abundances, stellar spectrum, and public physics knobs.

There is no timestep control, no live pair sampling, no anchor-state contract, and no trajectory target mode in the supported pipeline.

## Public Pipeline

The public CLI stays:

```bash
python -m src.utils --config <path> --stage generation|normalization|training
```

The three stages are:

1. `generation`
2. `normalization`
3. `training`

`generation` must reach the exact requested number of successful runs. Failed VULCAN runs are never included in the dataset. If backfill retries still leave a shortfall, generation raises an error and writes `failed_runs.json` for debugging.

## Supported Config Surface

Every config contains these shared top-level sections:

- `task`
- `paths`
- `data_spec`
- `sampling`
- `temperature_profiles`
- `generation`
- `normalization`
- `training`

Exactly one task-specific section must be present:

- `equilibrium_only`
- `full_vulcan`

The only supported task kinds are:

- `equilibrium_only`
- `full_vulcan`

The fixed FastChem-native elemental input order is internal and not user-configurable:

```python
["He_H", "C_H", "O_H", "N_H", "S_H"]
```

These are hydrogen-normalized number abundances `n_X / n_H`, not total-gas mixing ratios and not relative-to-solar values.

## Full-VULCAN Public Knobs

The public full-VULCAN config surface includes:

- atmosphere base: `full_vulcan.vulcan_runtime.atm_base`
- chemistry network file: `full_vulcan.vulcan_runtime.chemistry_file`
- cross-section species list: `full_vulcan.vulcan_runtime.t_cross_sp`
- transport toggles:
  - `use_eddy_diffusion`
  - `use_molecular_diffusion`
  - `use_upwind_molecular_diffusion`
- boundary-condition toggle:
  - `use_boundary_conditions`
- condensation family:
  - `use_condensation`
  - `use_settling`
  - `use_initial_cold_trap`
  - `use_sat_surface_h2o`
- chemistry toggles:
  - `use_photochemistry`
  - `use_ion_chemistry`
- stellar/spectrum physical inputs:
  - template name/file
  - wavelength range
  - bin count
  - zenith angle
  - diurnal factor
  - stellar parameters
  - orbital distance

These runtime controls are internal defaults, not part of the public scientific surface:

- `python_executable`
- `cfg_file`
- `worker_root`
- `regenerate_chem_funs`
- `cfg_assignments`
- `use_lowT_limit_rates`
- `use_adaptive_rtol`

## Removed Legacy Config Keys

These keys are explicitly rejected with migration errors:

- `training.live_sampling`
- `sampling.num_time_steps`
- `sampling.time_step_log10_min_s`
- `sampling.time_step_log10_max_s`
- `generation.target_mode`
- `normalization.state_method`
- `normalization.log10_dt_method`
- `full_vulcan.trajectory_sampling`

## Raw Data Contract

The raw HDF5 layout is final-state-only.

### Equilibrium raw run

- `inputs/pressure_bar` `(nz,)`
- `inputs/temperature_k` `(nz,)`
- `inputs/element_input_order` `(n_elements,)`
- `inputs/elemental_abundances_x_h` `(nz, n_elements)`
- `inputs/gravity_cm_s2` `(nz,)`
- `inputs/state_species` `(n_state,)`
- `inputs/output_species` `(n_output,)`
- `globals/<name>` scalar datasets
- `equilibrium/ymix` `(nz, n_output)`

### Full-VULCAN raw run

- `inputs/pressure_bar` `(nz,)`
- `inputs/temperature_k` `(nz,)`
- `inputs/kzz_cm2_s` `(nz,)`
- `inputs/element_input_order` `(n_elements,)`
- `inputs/elemental_abundances_x_h` `(nz, n_elements)`
- `inputs/gravity_cm_s2` `(nz,)`
- `inputs/state_species` `(n_state,)`
- `inputs/output_species` `(n_output,)`
- `globals/<name>` scalar datasets
- `final_state/ymix_output` `(nz, n_output)`
- `spectrum/name` scalar string
- `spectrum/wavelength_nm` `(n_wavelength,)`
- `spectrum/flux_erg_cm2_s_nm` `(n_wavelength,)`

There is no `trajectory/` group and no `inputs/target_mode`.

## Processed Data Contract

Processed artifacts are versioned with:

```python
PROCESSED_DATA_VERSION = 9
```

### Equilibrium split files

- `sequence_inputs.npy` `(N, nz, 2)` for `[pressure_bar, temperature_k]`
- `target_outputs.npy` `(N, nz, target_dim)`
- `global_inputs.npy` `(N, global_dim)`
- `run_ids.json`
- `metadata.json`

### Full-VULCAN split files

- `sequence_inputs.npy` `(N, nz, 3)` for `[pressure_bar, temperature_k, kzz_cm2_s]`
- `target_outputs.npy` `(N, nz, target_dim)`
- `global_inputs.npy` `(N, global_dim)`
- `spectrum_inputs.npy` `(N, spectrum_dim)`
- `run_ids.json`
- `metadata.json`

The processed full-VULCAN contract contains only final-state targets plus spectrum and run metadata.

## Normalization

Normalization is fit on the training split only.

Supported methods:

- `standard`
- `log-standard`
- `none`
- `mixed` for the stored global-conditioning block

Config-selected normalization controls are:

- `normalization.sequence_methods`
- `normalization.global_methods`
- `normalization.target_method`
- `normalization.spectrum_method` for `full_vulcan`
- `normalization.state_floor`
- `normalization.spectrum_floor` for `full_vulcan`
- `normalization.split`

The elemental channels use the same normalization path as the other positive chemistry features: `log10` plus z-score when configured as `log-standard`.

## Model Families

### Equilibrium model

- FiLM-conditioned per-level MLP
- no neighbor coupling between vertical levels
- input features per level: pressure and temperature
- global conditioning: fixed-order elemental abundances

### Full-VULCAN model

- FiLM-conditioned Transformer
- sequence inputs per level: pressure, temperature, Kzz
- global conditioning: gravity, fixed-order elemental abundances, public physics toggles, atmosphere-base one-hot
- separate stellar spectrum input

Supported activations for both model families:

- `relu`
- `gelu`
- `silu`
- `tanh`
- `elu`
- `selu`
- `softplus`
- `leaky_relu`

Default activation for both model families: `leaky_relu`.

## Export Bundle

Training checkpoints can be exported to NPZ bundles with baked normalization.

The bundle API in `src/models/export_bundle.py` exposes:

- `predict_equilibrium_profile(...)`
- `predict_full_vulcan_profile(...)`

Both accept physical-unit inputs, apply normalization internally, and return physical mixing-ratio outputs.

## ExoJAX API

`src/models/exojax_api.py` provides the public differentiable JAX wrapper layer.

### Equilibrium wrapper

```python
make_equilibrium_vmr_fn(bundle) -> (vmr_fn, species_labels)

vmr_fn(
    temperatures_k,            # (nz,), top -> bottom
    pressures_bar,             # (nz,), top -> bottom
    elemental_abundances_x_h,  # (nz, n_elements), top -> bottom
    gravity_cm_s2,             # (nz,), top -> bottom
) -> species_abundances        # (nz, n_species)
```

### Full-VULCAN wrapper

```python
make_full_vulcan_vmr_fn(bundle) -> (vmr_fn, species_labels)

vmr_fn(
    temperatures_k,            # (nz,), top -> bottom
    pressures_bar,             # (nz,), top -> bottom
    elemental_abundances_x_h,  # (nz, n_elements), top -> bottom
    kzz_cm2_s,                 # (nz,), top -> bottom
    gravity_cm_s2,             # (nz,), top -> bottom
    spectrum_flux,             # (spectrum_dim,)
) -> species_abundances        # (nz, n_species)
```

Contract rules:

- public layer order is top-to-bottom
- internal model order is bottom-to-top
- wrappers reverse inputs and outputs as needed
- elemental abundances use FastChem-native `n_X / n_H`
- gravity is required in both APIs
- current trained models still assume vertically constant elemental-abundance and gravity profiles, so eager wrapper calls reject varying profiles

## Extras

The supported utility scripts in `extras/` use the current final-state contract and exported bundle interfaces. They are examples and diagnostics, not part of the public training pipeline.
