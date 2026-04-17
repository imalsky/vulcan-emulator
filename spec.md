# VULCAN Emulator Spec

## Overview

The public config surface is now defined by:
- `chemistry_type`: `fastchem` or `vulcan`
- `model_type`: `transformer`

Supported combinations:
- `fastchem + transformer`
- `vulcan + transformer`

Shipped configs:
- `config/vulcan_no_condensation.json`
- `config/vulcan_condensation.json`

`chemistry_type` selects the target contract and learned inputs.
`model_type` selects the prediction architecture only.

Photochemistry is not currently supported. The `use_photochemistry` toggle
is kept in the config schema but must be `False` for all science presets.
Support will be re-added in a future release.

## Constants

All shared constants live in `src/constants.py` — the single source of truth
for physical constants, solar abundances, species data, chemistry/model enums,
config validation allowlists, internal defaults, and export versioning.
Module-private constants (used by only one file) stay in their owning module.
`src/models/standalone_inference.py` is self-contained and does not import from
`constants.py`.

## Public Pipeline

```bash
python -m src.utils --config <path> --stage generation|normalization|training|export
```

Stages:
1. `generation`
2. `normalization`
3. `training`
4. `export`

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

Valid keys:
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

Optional child keys:
- `science_presets`
- `stellar_spectrum` (used by data generation only, not the model)

Public physics toggles:
- `use_photochemistry` (must be `False` for all presets — photochem not yet supported)
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

`vulcan.stellar_spectrum` is required because VULCAN reads `sflux_file`
unconditionally at startup, even when `use_photochemistry` is `False`.
When present, it supports:
- `template_name`
- `template_file`
- optional `library_glob` — glob pattern for multi-spectrum libraries
- `max_tokens`
- `wavelength_min_nm`
- `wavelength_max_nm`
- optional `dbin1_nm` — short-wavelength bin width in nm (default 0.1)
- optional `dbin2_nm` — long-wavelength bin width in nm (default 2.0)
- optional `dbin_12trans_nm` — transition wavelength between bin regimes in nm (default 240.0)
- optional `teff_k` — effective temperature for blackbody template generation (defaults to 5485 K)
- optional `radius_rsun` — provenance metadata only (per-run values come from `sampling`)
- optional `semi_major_axis_au` — provenance metadata only (per-run values come from `sampling`)

Per-run stellar irradiation geometry now lives under `sampling`:
- `stellar_radius_range_rsun`
- `semi_major_axis_range_au`
- `zenith_angle_range_deg`
- `diurnal_factor_range`

Each VULCAN run samples those four values, writes them into `globals/`,
uses them to patch the worker-local `vulcan_cfg.py`, and feeds them back to
the surrogate as learned global inputs.

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

The shipped `vulcan_transformer` example is a condensation-enabled,
photochemistry-disabled H2 setup using `thermo/SNCHO_photo_network_2025.txt`
and `vulcan.runtime.cfg_assignments` to pass the `H2O`/`S8` condensation
recipe through to worker-local `vulcan_cfg.py`.

Worker runtime scratch directories use `tempfile.mkdtemp()` for staging and
`data/vulcan_workers/` for per-worker VULCAN checkouts. No persistent
`runtime/` directory is created.

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
- `spectrum/name` (optional, used by VULCAN runtime only)
- `spectrum/wavelength_nm` `(n_wavelength,)` (optional)
- `spectrum/flux_erg_cm2_s_nm` `(n_wavelength,)` (optional)

There is no trajectory target mode and no timestep-control learning contract.
The learned VULCAN contract still uses scalar `globals/gravity_cm_s2` (surface
gravity) and scalar `globals/planet_radius_cm`; the raw layerwise gravity
profile is retained for provenance and diagnostics only.

## Processed Data Contract

Processed artifacts are versioned with:

```python
PROCESSED_DATA_VERSION = 18
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

### VULCAN processed split

- `sequence_inputs.npy` `(N, nz, 3)` for `[pressure_bar, temperature_k, kzz_cm2_s]`
- `target_outputs.npy` `(N, nz, target_dim)`
- `global_inputs.npy` `(N, global_dim)` for
  `[gravity_cm_s2, planet_radius_cm, He_H, C_H, O_H, N_H, S_H,
    r_star_rsun, semi_major_axis_au, zenith_angle_deg, diurnal_factor,
    <physics toggles>, <atm_base one-hots>]`

`metadata.json` and `info/data_contract.json` store explicit:
- `chemistry_type`
- `model_type`
- `sequence_static_feature_order`
- `global_static_feature_order`

## Training Contract

`training.loss` always requires:
- `lambda_z`
- `lambda_log10_mae`

Architecture behavior:
- `fastchem + transformer`: PT + `X/H`
- `vulcan + transformer`: PT + Kzz + surface gravity + planet radius + sampled irradiation geometry + runtime globals

## Architecture Contract

### Global Conditioning Inputs

The transformer receives a global conditioning vector that encodes all
non-sequence scalar inputs.  For VULCAN the global vector
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

For FastChem the global vector is `[He_H, C_H, O_H, N_H, S_H]` only.

### FiLM Conditioning

1. **Context MLP**: `global_inputs` →
   `act(Linear(conditioning_hidden_dim))` →
   `Linear(num_layers × 2 × d_model)` →
   reshape to `(batch, num_layers, 2, d_model)` producing per-layer
   gamma and beta.
2. **Modulation**: at each layer, `x = x * (1 + clip(gamma)) + clip(beta)`
   where clip bounds are `[-film_clamp, film_clamp]`.

### FiLM-Conditioned Transformer

Pre-norm transformer with FiLM modulation from global inputs.

```
Input:    Linear(sequence_dim → d_model) + sinusoidal positional encoding

Per block:
  a. Pre-norm (ln1) → multi-head self-attention → dropout → residual add
  b. FiLM: x = x * (1 + gamma) + beta
  c. Pre-norm (ln_ffn) → FFN (up-project, activation, dropout, down-project) → residual add

Output head: LayerNorm → activation → Linear(d_model → d_model // output_head_divisor) → dropout → Linear(→ target_dim)
```

#### Design notes

- **Pre-norm attention (ln1)** and **pre-norm FFN (ln_ffn)**: standard
  pre-norm transformer pattern. FiLM conditioning is applied directly
  to the residual stream before the FFN pre-norm.
- **Residual connections**: after self-attention and FFN sub-layers.

### Normalization

Target outputs use `log-standard` normalization: `(log10(ymix) - mean) / std`.

Sequence features are normalized per-feature:
- `pressure_bar`: log-standard
- `temperature_k`: standard
- `kzz_cm2_s`: log-standard

Global features use mixed per-feature normalization as documented in the
table above.  Normalization statistics are fitted on the training split
only and applied identically to validation and test splits.

## Export and Inference Contract

Export bundles embed explicit:
- `chemistry_type`
- `model_type`
- model dimensions
- normalization metadata
- data contract
- config

Bundles without explicit `chemistry_type` and `model_type` metadata are
rejected on load.

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

Both wrappers preserve JAX differentiability for model inputs.

PT-library metadata and analytic-sampler reference gravity remain PT-shape-only
inputs and are intentionally allowed to differ from the VULCAN runtime surface
gravity and radius.

## Testing

The test suite is intentionally minimal — five files covering the critical
paths:

| File | Scope |
|------|-------|
| `test_config.py` | Config validation safety net |
| `test_model.py` | JAX model forward pass, autodiff, training loop |
| `test_export_bundle.py` | Export/import round-trip, physical inference |
| `test_sampling.py` | Atmospheric sampling correctness |
| `test_vulcan_runner.py` | End-to-end generation and preprocessing |

New tests should be added only when they cover a genuinely distinct failure
mode not already exercised by the existing suite.
