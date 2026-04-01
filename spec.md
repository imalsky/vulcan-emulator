# VULCAN Emulator Spec

> Note: I still need to figure something out with gravity and Kzz.

## Overview

The public config surface is now defined by:
- `chemistry_type`: `fastchem` or `vulcan`
- `model_type`: `mlp` or `transformer`

Supported combinations:
- `fastchem + mlp`
- `fastchem + transformer`
- `vulcan + mlp`
- `vulcan + transformer`

`chemistry_type` selects the target contract and learned inputs.
`model_type` selects the prediction architecture only.

Removed and rejected legacy config keys:
- `task.kind`
- `equilibrium_only`
- `full_vulcan`
- old internal `model_type` values `equilibrium` and `full_vulcan`
- `training.live_sampling`
- `sampling.num_time_steps`
- `sampling.time_step_log10_min_s`
- `sampling.time_step_log10_max_s`
- `generation.target_mode`
- `normalization.state_method`
- `normalization.log10_dt_method`
- `full_vulcan.trajectory_sampling`
- `vulcan.trajectory_sampling`

## Public Pipeline

```bash
python -m src.utils --config <path> --stage generation|normalization|training
```

Stages:
1. `generation`
2. `normalization`
3. `training`

`generation` must reach the requested number of successful runs. Failed VULCAN runs are never included in the final dataset.

## Config Surface

Every config contains:
- `chemistry_type`
- `model_type`
- `paths`
- `data_spec`
- `sampling`
- `temperature_profiles`
- `generation`
- `normalization`
- `training`
- `model`

`vulcan` is required only when `chemistry_type = "vulcan"`.
`vulcan` is invalid when `chemistry_type = "fastchem"`.

The fixed elemental order is internal and not user-configurable:

```python
["He_H", "C_H", "O_H", "N_H", "S_H"]
```

These are hydrogen-normalized absolute abundances `n_X / n_H`.

The generation sampler still draws `[M/H]`, `C/O`, and `S/O`, then converts them into the fixed `X/H` channels before preprocessing and training.

`temperature_profiles.source_mode` supports only:
- `analytic`
- `pt_library`
- `mixed`

Supported PT-library filter keys:
- numeric: `Teq`, `LogMet`, `LogDrag`, `Mstar`, `Rp`, `logG`
- boolean: `TiOVO`

### `model`

If `model_type = "mlp"`, valid keys are:
- `d_hidden`
- `num_hidden_layers`
- `conditioning_hidden_dim`
- `film_clamp`
- optional `activation`
- optional `dropout_rate`

If `model_type = "transformer"`, valid keys are:
- `d_model`
- `nhead`
- `num_layers`
- `dim_feedforward`
- `conditioning_hidden_dim`
- `film_clamp`
- `output_head_divisor`
- optional `activation`
- optional `dropout_rate`

### `vulcan`

Required child keys:
- `physics_toggles`
- `runtime`
- `stellar_spectrum`

Optional child key:
- `science_presets`

Public physics toggles:
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

Supported `atm_base` values:
- `H2`
- `N2`
- `O2`
- `CO2`
- `H2O`

`science_presets` is the public mechanism for per-run VULCAN science variation.
Each preset supports:
- `name`
- optional `atm_base`
- optional `physics_toggles`

Within a preset:
- omitted toggle values fall back to `vulcan.physics_toggles`
- omitted `atm_base` falls back to `vulcan.runtime.atm_base`

`vulcan.stellar_spectrum` supports:
- `enabled`
- `template_name`
- `template_file`
- optional `library_glob`
- `num_bins`
- `latent_dim`
- `hidden_dim`
- `encoder_mode`
- `wavelength_min_nm`
- `wavelength_max_nm`
- `teff_k`
- `radius_rsun`
- `semi_major_axis_au`
- `diurnal_factor`
- `zenith_angle_deg`

Supported `encoder_mode` values:
- `autoencoder`
- `linear`
- `none`

`enabled` remains a compatibility flag only. VULCAN chemistry always uses the spectrum conditioning path.

Internal runtime defaults that are not learned public inputs:
- `python_executable`
- `cfg_file`
- `worker_root`
- `regenerate_chem_funs`
- `cfg_assignments`
- `use_lowT_limit_rates`
- `use_adaptive_rtol`

## Raw Data Contract

### FastChem raw run

- `inputs/pressure_bar` `(nz,)`
- `inputs/temperature_k` `(nz,)`
- `inputs/element_input_order` `(n_elements,)`
- `inputs/elemental_abundances_x_h` `(nz, n_elements)`
- `inputs/gravity_cm_s2` `(nz,)`
- `inputs/state_species` `(n_state,)`
- `inputs/output_species` `(n_output,)`
- `globals/<name>` scalar datasets
- `equilibrium/ymix` `(nz, n_output)`

### VULCAN raw run

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
- `spectrum/name`
- `spectrum/wavelength_nm` `(n_wavelength,)`
- `spectrum/flux_erg_cm2_s_nm` `(n_wavelength,)`

There is no trajectory target mode and no timestep-control learning contract.

## Processed Data Contract

Processed artifacts are versioned with:

```python
PROCESSED_DATA_VERSION = 13
```

All processed datasets write:
- split directories `train/`, `val/`, and `test/`
- shared metadata under `info/`
  - `info/normalization.json`
  - `info/data_contract.json`
  - `info/splits.json`
  - `info/processed_manifest.json`

### FastChem processed split

- `sequence_inputs.npy` `(N, nz, 2)` for `[pressure_bar, temperature_k]`
- `target_outputs.npy` `(N, nz, target_dim)`
- `global_inputs.npy` `(N, 5)` for `[He_H, C_H, O_H, N_H, S_H]`
- no `spectrum_inputs.npy`

### VULCAN processed split

- `sequence_inputs.npy` `(N, nz, 3)` for `[pressure_bar, temperature_k, kzz_cm2_s]`
- `target_outputs.npy` `(N, nz, target_dim)`
- `global_inputs.npy` `(N, global_dim)`
- `spectrum_inputs.npy` `(N, spectrum_dim)`

`metadata.json` and `info/data_contract.json` store explicit:
- `chemistry_type`
- `model_type`
- `sequence_static_feature_order`
- `global_static_feature_order`
- `spectrum_dim` when present

## Training Contract

`training.loss` always requires:
- `lambda_z`
- `lambda_phys`

`training.loss.lambda_spectrum` is required only for `chemistry_type = "vulcan"` and invalid for `fastchem`.

Architecture behavior:
- `fastchem + mlp`: PT + `X/H`
- `fastchem + transformer`: PT + `X/H`, no spectrum path
- `vulcan + mlp`: PT + Kzz + runtime globals + spectrum latent
- `vulcan + transformer`: PT + Kzz + runtime globals + spectrum latent

## Architecture Contract

Both architectures use **FiLM (Feature-wise Linear Modulation)** to inject
global context (elemental abundances, gravity, physics toggles, spectrum
latent) into the per-level prediction pathway.  Layer norms and residual
connections are architectural defaults, not config-controlled.

### FiLM Conditioning (shared by both architectures)

1. **Spectrum encoding** (VULCAN only): compress stellar spectrum to a
   latent vector via autoencoder, linear projection, or zeros.
2. **Context MLP**: `[global_inputs, spectrum_latent]` →
   `act(Linear(conditioning_hidden_dim))` →
   `Linear(num_layers × 2 × width)` →
   reshape to `(batch, num_layers, 2, width)` producing per-layer
   gamma and beta.
3. **Modulation**: at each layer, `x = x * (1 + clip(gamma)) + clip(beta)`
   where clip bounds are `[-film_clamp, film_clamp]`.

### FiLM-Conditioned MLP

Per-level architecture with shared weights across the vertical grid.

```
Layer 0:  Linear(sequence_dim → d_hidden) → LayerNorm → FiLM → activation → dropout
Layer i≥1: Linear(d_hidden → d_hidden) → LayerNorm → FiLM → activation → dropout → residual add
Output:   Linear(d_hidden → target_dim)
```

- **LayerNorm before FiLM**: normalizes activations so gamma/beta operate
  on a standardized representation (unit variance), making them directly
  interpretable as relative scale/shift.
- **Residual connections** (layer ≥ 1): enable gradient flow through deep
  networks.  Layer 0 changes dimension (`sequence_dim → d_hidden`) so no
  residual there.

### FiLM-Conditioned Transformer

Pre-norm transformer with FiLM modulation and post-FiLM stabilization.

```
Input:    Linear(sequence_dim → d_model) + sinusoidal positional encoding

Per block:
  a. Pre-norm (ln1) → multi-head self-attention → dropout → residual add
  b. FiLM: x = x * (1 + gamma) + beta
  c. Post-FiLM norm (ln_film) → re-stabilize residual stream
  d. Pre-norm (ln2) → FFN (up-project, activation, dropout, down-project) → residual add

Output head: LayerNorm → activation → Linear(d_model → d_model // output_head_divisor) → dropout → Linear(→ target_dim)
```

- **Post-FiLM LayerNorm (ln_film)**: re-stabilizes the residual stream
  after FiLM's affine transform, preventing FiLM scaling from conflicting
  with the FFN pre-norm.
- **Pre-norm attention (ln1)** and **pre-norm FFN (ln2)**: standard
  pre-norm transformer pattern.
- **Residual connections**: after both the attention and FFN sub-layers.

## Export and Inference Contract

Export bundles embed explicit:
- `chemistry_type`
- `model_type`
- model dimensions
- normalization metadata
- data contract
- config

Legacy task-based bundles are rejected on load.

`ExportedJAXModel` exposes:
- `predict_fastchem_profile(...)`
- `predict_vulcan_profile(...)`

ExoJAX wrappers:
- `make_fastchem_vmr_fn(bundle) -> (vmr_fn, species_labels)`
- `make_vulcan_vmr_fn(bundle) -> (vmr_fn, species_labels)`

The public FastChem wrapper expects only profile-global `X/H`.
The public VULCAN wrapper expects:
- `kzz_cm2_s`
- `global_inputs` containing gravity, `X/H`, toggles, and `atm_base_*`
- `spectrum_flux`

Both wrappers preserve JAX differentiability.
