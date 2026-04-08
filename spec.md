# VULCAN Emulator Spec

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
- `normalization.spectrum_method`
- `training.loss.lambda_spectrum`

## Public Pipeline

```bash
python -m src.utils --config <path> --stage generation|normalization|training
```

Stages:
1. `generation`
2. `normalization`
3. `training`

`generation` must reach the requested number of successful runs. Failed VULCAN runs are never included in the final dataset.

## Module Boundaries

- Keep the current module boundaries stable within `src/`.
- Prefer in-file organization over file proliferation.
- Split a module only when it owns radically different responsibilities.
- Production hardening must preserve the current import paths and public
  entry points unless this spec explicitly changes them.

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

Path layout is standardized per run:

```text
data/<run_name>/
  raw/
  info/
  train/
  val/
  test/
```

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
- optional `library_glob` — glob pattern for multi-spectrum libraries
- `max_tokens`
- `latent_dim`
- `hidden_dim`
- `num_latents`
- `num_layers`
- `num_heads`
- `fourier_features`
- `encoder_mode`
- `wavelength_min_nm`
- `wavelength_max_nm`
- optional `teff_k` — provenance/template-generation metadata only
- optional legacy fixed geometry keys:
  `radius_rsun`, `semi_major_axis_au`, `diurnal_factor`, `zenith_angle_deg`

Per-run stellar irradiation geometry now lives under `sampling`:
- `stellar_radius_range_rsun`
- `semi_major_axis_range_au`
- `zenith_angle_range_deg`
- `diurnal_factor_range`

Each VULCAN run samples those four values, writes them into `globals/`,
uses them to patch the worker-local `vulcan_cfg.py`, and feeds them back to
the surrogate as learned global inputs. The stellar spectrum itself remains
the learned carrier of stellar-type / SED structure, so `teff_k` is no
longer part of the required model global-input contract.

Supported `encoder_mode` values:
- `perceiver`

Internal runtime defaults that are not learned public inputs:
- `python_executable`
- `cfg_file`
- `worker_root`
- `regenerate_chem_funs`
- `cfg_assignments`
- `use_lowT_limit_rates`
- `use_adaptive_rtol`
- `rocky`
- optional `top_bc_flux_file`
- optional `bot_bc_flux_file`

Worker runtime scratch directories are temporary and live outside `data/`.

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

FastChem stores a gravity profile in raw HDF5 for dataset uniformity, but the
learned FastChem contract does not consume gravity or Kzz.

### VULCAN raw run

- `inputs/pressure_bar` `(nz,)`
- `inputs/temperature_k` `(nz,)`
- `inputs/kzz_cm2_s` `(nz,)`
- `inputs/element_input_order` `(n_elements,)`
- `inputs/elemental_abundances_x_h` `(nz, n_elements)`
- `inputs/gravity_cm_s2` `(nz,)`, storing the layerwise gravity profile from
  the VULCAN runtime (`atm.g`) when available
- `inputs/state_species` `(n_state,)`
- `inputs/output_species` `(n_output,)`
- `globals/<name>` scalar datasets
- `final_state/ymix_output` `(nz, n_output)`
- `spectrum/name`
- `spectrum/wavelength_nm` `(n_wavelength,)`
- `spectrum/flux_erg_cm2_s_nm` `(n_wavelength,)`

There is no trajectory target mode and no timestep-control learning contract.
The learned VULCAN contract still uses scalar `globals/gravity_cm_s2` (surface
gravity) and scalar `globals/planet_radius_cm`; the raw layerwise gravity
profile is retained for provenance and diagnostics only.

## Processed Data Contract

Processed artifacts are versioned with:

```python
PROCESSED_DATA_VERSION = 17
```

All processed datasets write:
- split directories `train/`, `val/`, and `test/`
- shared metadata under `info/`
  - `info/normalization.json`
  - `info/data_contract.json`
  - `info/splits.json`
  - `info/processed_manifest.json`
  - `info/generation_manifest.json`
  - `info/sampling_coverage.json`
  - optional `info/failed_runs.json`

### FastChem processed split

- `sequence_inputs.npy` `(N, nz, 2)` for `[pressure_bar, temperature_k]`
- `target_outputs.npy` `(N, nz, target_dim)`
- `global_inputs.npy` `(N, 5)` for `[He_H, C_H, O_H, N_H, S_H]`
- no spectrum token arrays

### VULCAN processed split

- `sequence_inputs.npy` `(N, nz, 3)` for `[pressure_bar, temperature_k, kzz_cm2_s]`
- `target_outputs.npy` `(N, nz, target_dim)`
- `global_inputs.npy` `(N, global_dim)` for
  `[gravity_cm_s2, planet_radius_cm, He_H, C_H, O_H, N_H, S_H,
    r_star_rsun, semi_major_axis_au, zenith_angle_deg, diurnal_factor,
    <physics toggles>, <atm_base one-hots>]`
- `spectrum_wavelengths_nm.npy` `(N, spectrum_max_tokens)`
- `spectrum_fluxes_erg_cm2_s_nm.npy` `(N, spectrum_max_tokens)`
- `spectrum_mask.npy` `(N, spectrum_max_tokens)`

`metadata.json` and `info/data_contract.json` store explicit:
- `chemistry_type`
- `model_type`
- `sequence_static_feature_order`
- `global_static_feature_order`
- `spectrum_max_tokens`
- `spectrum_wavelength_min_nm`
- `spectrum_wavelength_max_nm`

## Training Contract

`training.loss` always requires:
- `lambda_z`
- `lambda_phys`

Architecture behavior:
- `fastchem + mlp`: PT + `X/H`
- `fastchem + transformer`: PT + `X/H`, no spectrum path
- `vulcan + mlp`: PT + Kzz + surface gravity + planet radius + sampled irradiation geometry + runtime globals + spectrum latent (FiLM only)
- `vulcan + transformer`: PT + Kzz + surface gravity + planet radius + sampled irradiation geometry + runtime globals + spectrum latent (FiLM) + spectrum cross-attention

## Architecture Contract

### Global Conditioning Inputs

Both architectures receive a global conditioning vector that encodes all
non-sequence, non-spectrum scalar inputs.  For VULCAN the global vector
contains (in order):

| Group                | Fields                                                          | Normalization     |
|----------------------|-----------------------------------------------------------------|-------------------|
| Planetary            | `gravity_cm_s2`, `planet_radius_cm`                             | log-standard      |
| Elemental abundances | `He_H`, `C_H`, `O_H`, `N_H`, `S_H`                            | standard or log-standard |
| Irradiation geometry | `r_star_rsun`, `semi_major_axis_au`, `zenith_angle_deg`, `diurnal_factor` | `log-standard`, `log-standard`, `standard`, `standard` |
| Physics toggles      | `use_photochemistry`, `use_ion_chemistry`, ...                  | none (binary)     |
| Atmosphere base      | `atm_base_H2`, `atm_base_N2`, `atm_base_O2`, `atm_base_CO2`, `atm_base_H2O` | none (one-hot) |

The irradiation-geometry globals (`r_star_rsun`, `semi_major_axis_au`,
`zenith_angle_deg`, `diurnal_factor`) are sampled per run from the VULCAN
`sampling` ranges, stored in the raw `globals/` group, used to configure the
VULCAN runtime, and promoted to learned model inputs. This keeps the training
data physically self-consistent: the surrogate sees the same stellar forcing
that generated the chemistry target.

`teff_k` is intentionally not a required learned global. The full stellar
spectrum is already provided to the Perceiver encoder, so stellar-type
information is learned from the SED itself rather than from a separate scalar
label.

For FastChem the global vector is `[He_H, C_H, O_H, N_H, S_H]` only.

### Spectrum Encoding: Perceiver Encoder

Both architectures use the same Perceiver-based spectrum encoder (VULCAN
only; disabled for FastChem).  The encoder compresses a variable-length
native-grid stellar spectrum into two outputs:

1. **Mean-pooled latent vector** `(batch, spectrum_latent_dim)` — used
   for FiLM conditioning in both MLP and Transformer.
2. **Un-pooled latent tokens** `(batch, num_latents, spectrum_hidden_dim)`
   — used for per-block cross-attention in the Transformer only.

#### Token preparation

Each spectrum sample `(wavelengths, fluxes, mask)` of up to
`spectrum_max_tokens` entries is converted to a feature vector per token:

| Feature                 | Dimension          | Description |
|-------------------------|--------------------|-------------|
| `log_wavelength`        | 1                  | `log10(wavelength_nm)` |
| `wavelength_step`       | 1                  | Difference in log-wavelength between adjacent tokens |
| `normalized_log_flux`   | 1                  | Per-spectrum z-normalized `log10(flux)` |
| `wavelength_embedding`  | 2 × `fourier_features` | Sin/cos Fourier features on `log10(wavelength)` with frequencies `2^0, 2^1, ..., 2^(F-1)` scaled by pi |

Invalid (padded) tokens are zeroed via the mask.

#### Summary scalars

Six per-spectrum summary scalars are computed from the valid tokens and
preserved alongside the latent tokens:

1. `log_flux_mean` — mean of `log10(flux)` over valid tokens
2. `log_flux_std` — standard deviation of `log10(flux)`
3. `log_integrated_flux` — `log10` of trapezoidal integration of flux over wavelength
4. `coverage_fraction` — fraction of tokens that are valid (not padding)
5. `log_wavelength_min` — `log10` of minimum valid wavelength
6. `log_wavelength_max` — `log10` of maximum valid wavelength

These are concatenated with the mean-pooled latent before the final
projection, preserving absolute-scale information that z-normalization
removes.

#### Perceiver architecture

```
Token embedding:  Linear(token_feature_dim → spectrum_hidden_dim) → LayerNorm → activation

Learnable latents:  (spectrum_num_latents, spectrum_hidden_dim) initialized N(0, 0.02)

Cross-attention:  LayerNorm(latents) → Q, tokens → K/V
                  Masked multi-head attention (spectrum_num_heads)
                  Residual add to latents

Self-attention blocks (× spectrum_num_layers):
  LayerNorm → self-attention → residual add
  LayerNorm → FFN(4× expansion) → residual add

Output:
  LayerNorm(latents) → mean-pool → concat(summary) → Linear → activation → Linear → latent_vector
  LayerNorm(latents) → latent_tokens  (shared normalization with mean-pool path)
```

When the spectrum encoder is disabled (FastChem), `latent_vector` is a
zero vector and `latent_tokens` is `None`.

### FiLM Conditioning (shared by both architectures)

1. **Context MLP**: `[global_inputs, spectrum_latent_vector]` →
   `act(Linear(conditioning_hidden_dim))` →
   `Linear(num_layers × 2 × width)` →
   reshape to `(batch, num_layers, 2, width)` producing per-layer
   gamma and beta.
2. **Modulation**: at each layer, `x = x * (1 + clip(gamma)) + clip(beta)`
   where clip bounds are `[-film_clamp, film_clamp]`.

### FiLM-Conditioned MLP

Per-level architecture with shared weights across the vertical grid.
Spectrum influence is via FiLM only (no cross-attention).

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

Pre-norm transformer with dual spectrum pathway: FiLM modulation from the
mean-pooled latent vector **plus** per-block cross-attention from the
un-pooled Perceiver latent tokens.

```
Input:    Linear(sequence_dim → d_model) + sinusoidal positional encoding

Per block:
  a. Pre-norm (ln1) → multi-head self-attention → dropout → residual add
  b. Pre-norm (ln_cross) → cross-attention(Q=sequence, KV=spectrum_latent_tokens)
     → dropout → residual add                               [VULCAN only; skipped for FastChem]
  c. FiLM: x = x * (1 + gamma) + beta
  d. Pre-norm (ln_ffn) → FFN (up-project, activation, dropout, down-project) → residual add

Output head: LayerNorm → activation → Linear(d_model → d_model // output_head_divisor) → dropout → Linear(→ target_dim)
```

#### Cross-attention to spectrum (step b)

Each transformer block performs cross-attention from the atmospheric
sequence (query) to the Perceiver's latent tokens (key/value).  This
allows every vertical level to directly attend to spectral features
rather than relying solely on the compressed FiLM signal.

- **Query**: sequence representation `(batch, nz, d_model)`, projected via `cross_q`
- **Key/Value**: Perceiver latent tokens `(batch, num_latents, spectrum_hidden_dim)`,
  projected via `cross_k` and `cross_v` to `d_model`
- **Output**: projected back to `d_model` via `cross_o`, added as residual
- **Heads**: same `nhead` as self-attention
- **No masking**: all Perceiver latent tokens are valid (padding was
  handled inside the Perceiver's cross-attention to raw tokens)
- **Skip condition**: when `spectrum_latent_tokens` is `None` (FastChem
  or encoder mode `"none"`), cross-attention is skipped entirely — no
  parameters are allocated and the block reduces to the FiLM-only pattern

This dual pathway means the transformer receives spectrum information
through two complementary channels:
1. **FiLM** (global modulation): the mean-pooled latent vector sets a
   per-layer scale/shift — a broadband summary of the stellar spectrum.
2. **Cross-attention** (per-level detail): each atmospheric level can
   selectively attend to specific spectral features in the Perceiver
   latent space — capturing fine-grained wavelength-dependent effects
   such as UV-driven photodissociation rates that vary with altitude.

#### Design notes

- **Pre-norm attention (ln1)** and **pre-norm FFN (ln_ffn)**: standard
  pre-norm transformer pattern. FiLM conditioning is applied directly
  to the residual stream before the FFN pre-norm.
- **Residual connections**: after self-attention, cross-attention, and FFN sub-layers.
- **Cross-attention parameters per layer**: `ln_cross` (LayerNorm),
  `cross_q` (d_model → d_model), `cross_k` (spectrum_hidden_dim → d_model),
  `cross_v` (spectrum_hidden_dim → d_model), `cross_o` (d_model → d_model).

### Multi-Spectrum Training

The data generation pipeline supports training on multiple stellar
spectra out of the box:

1. Place spectrum files (VULCAN `.dat` format) in a directory.
2. Set `vulcan.stellar_spectrum.library_glob` to a glob pattern matching
   those files (e.g. `"assets/stellar_spectra/*.dat"`).
3. During sampling, each run is assigned a randomly selected spectrum
   from the full library, while `r_star_rsun`, `semi_major_axis_au`,
   `zenith_angle_deg`, and `diurnal_factor` are independently sampled from
   the configured VULCAN `sampling` ranges and stored per-run.

Recommended global normalization for variable-star training:
- `r_star_rsun`: `log-standard`
- `semi_major_axis_au`: `log-standard`
- `zenith_angle_deg`: `standard`
- `diurnal_factor`: `standard`

The Perceiver encoder handles variable wavelength grids natively via
`pack_spectrum_tokens()` — spectra shorter than `max_tokens` are
zero-padded with a validity mask; longer spectra are flux-conserving
log-space rebinned.

### Normalization

Target outputs use `log-standard` normalization: `(log10(ymix) - mean) / std`.

Sequence features are normalized per-feature:
- `pressure_bar`: log-standard
- `temperature_k`: standard
- `kzz_cm2_s`: log-standard

Global features use mixed per-feature normalization as documented in the
table above.  Normalization statistics are fitted on the training split
only and applied identically to validation and test splits.

Spectrum fluxes are normalized on-the-fly per spectrum in the model
forward pass (not during preprocessing): `log10(flux)` is z-normalized
with a per-spectrum mean and standard deviation.  A configurable
`spectrum_floor` (default `1e-30`) prevents log of zero.

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
- `global_inputs` containing surface gravity, planet radius, `X/H`, stellar
  irradiation geometry (`r_star_rsun`, `semi_major_axis_au`,
  `zenith_angle_deg`, `diurnal_factor`), physics toggles, and `atm_base_*`
- `spectrum_wavelength_nm`
- `spectrum_flux_erg_cm2_s_nm`

Both wrappers preserve JAX differentiability for model inputs subject to the
following VULCAN spectrum contract:
- eager / non-traced inputs: arbitrary valid wavelength grids are sanitized,
  clipped to the configured interval, and packed through
  `pack_spectrum_tokens()`
- JAX-traced inputs: spectra must already be valid native-grid arrays with
  static `len(spectrum) <= spectrum_max_tokens`; the wrapper pads them inside
  JAX and does not sort, clip, or rebin them

PT-library metadata and analytic-sampler reference gravity remain PT-shape-only
inputs and are intentionally allowed to differ from the VULCAN runtime surface
gravity and radius.
