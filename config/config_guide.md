# Config Guide

## Overview

Every config is defined by two top-level selectors:

| Key | Values | Selects |
|-----|--------|---------|
| `chemistry_type` | `fastchem`, `vulcan` | Target contract and learned inputs |
| `model_type` | `mlp`, `transformer` | Prediction architecture only |

The four supported combinations are:
- `fastchem + mlp`
- `fastchem + transformer`
- `vulcan + mlp`
- `vulcan + transformer`

Shipped example configs:
- `config/fastchem_transformer_config.json`
- `config/vulcan_transformer_config.json`

CLI:

```bash
python -m src.utils --config config/fastchem_transformer_config.json --stage generation
python -m src.utils --config config/fastchem_transformer_config.json --stage normalization
python -m src.utils --config config/fastchem_transformer_config.json --stage training
```

---

## Top-Level Keys

Every config requires all of these:

```
chemistry_type
model_type
paths
data_spec
sampling
temperature_profiles
generation
normalization
training
model
```

`vulcan` is required only when `chemistry_type = "vulcan"` and invalid for `fastchem`.

Removed and rejected legacy keys: `task.kind`, `equilibrium_only`, `full_vulcan`.

---

## Chemistry Contracts

### FastChem (equilibrium)

- Sequence inputs: `[pressure_bar, temperature_k]` (nz, 2)
- Global inputs: `[He_H, C_H, O_H, N_H, S_H]` (5 elemental abundances, hydrogen-normalized)
- Target: FastChem `equilibrium/ymix`
- No spectrum, no Kzz, no gravity, no runtime knobs

### VULCAN (full kinetic)

- Sequence inputs: `[pressure_bar, temperature_k, kzz_cm2_s]` (nz, 3)
- Global inputs: `[gravity_cm_s2, planet_radius_cm, He_H, C_H, O_H, N_H, S_H, r_star_rsun, semi_major_axis_au, zenith_angle_deg, diurnal_factor]` + 10 physics toggles + 5 atm_base one-hots (26 total)
- Spectrum inputs: padded native-grid `(wavelengths, fluxes, mask)` tokens up to `max_tokens`
- Target: converged VULCAN `final_state/ymix_output`
- Raw VULCAN HDF5 stores the layerwise gravity profile under `inputs/gravity_cm_s2`, but the learned contract uses scalar surface gravity and scalar planet radius from `globals/*`
- The stellar spectrum carries stellar-type / SED information; sampled irradiation geometry is learned through the four global inputs above

The fixed elemental order is internal and not configurable:

```python
["He_H", "C_H", "O_H", "N_H", "S_H"]
```

These are hydrogen-normalized absolute abundances `n_X / n_H`. The generation sampler draws `[M/H]`, `C/O`, and `S/O`, then converts to the fixed `X/H` channels.

---

## Architecture Contracts

Both architectures use FiLM (Feature-wise Linear Modulation) to inject global context into per-level predictions. Layer norms and residual connections are architectural defaults, not config-controlled.

### FiLM conditioning (shared)

1. Encode stellar spectrum to latent (VULCAN only; zeros for FastChem)
2. `[global_inputs, spectrum_latent]` -> conditioning MLP -> per-layer gamma/beta
3. Per layer: `x = x * (1 + clip(gamma)) + clip(beta)`, clipped to `[-film_clamp, film_clamp]`

### MLP

FiLM-conditioned per-level MLP with shared weights across the vertical grid.

```
Layer 0:   Linear(sequence_dim -> d_hidden) -> LayerNorm -> FiLM -> activation -> dropout
Layer i>=1: Linear(d_hidden -> d_hidden) -> LayerNorm -> FiLM -> activation -> dropout -> residual add
Output:    Linear(d_hidden -> target_dim)
```

### Transformer

Pre-norm FiLM-conditioned Transformer with dual spectrum pathways:
mean-pooled FiLM conditioning plus per-block cross-attention to the
Perceiver latent tokens.

```
Input: Linear(sequence_dim -> d_model) + sinusoidal positional encoding

Per block:
  a. Pre-norm (ln1) -> multi-head self-attention -> dropout -> residual add
  b. Pre-norm (ln_cross) -> cross-attention(Q=sequence, KV=spectrum_latent_tokens) -> dropout -> residual add   [VULCAN only]
  c. FiLM: x = x * (1 + gamma) + beta
  d. Pre-norm (ln_ffn) -> FFN (up-project, activation, dropout, down-project) -> residual add

Output head: LayerNorm -> activation -> bottleneck -> dropout -> Linear(-> target_dim)
```

---

## Key-by-Key Reference

### `paths`

| Key | Description |
|-----|-------------|
| `run_root` | Dataset run directory. Use `data/<run_name>` |
| `checkpoints_root` | Directory for model checkpoints |
| `vulcan_source_root` | Path to VULCAN-master (needed for generation) |

`paths.run_root` is the only supported dataset path key. The dataset layout is:

```text
data/<run_name>/
  raw/
  info/
  train/
  val/
  test/
```

### `data_spec`

| Key | Description |
|-----|-------------|
| `state_species` | Species tracked internally by the solver |
| `output_species` | Species the model predicts (subset of state_species) |

### `sampling`

| Key | Required for | Description |
|-----|-------------|-------------|
| `num_levels` | both | Number of vertical pressure levels |
| `pressure_top_bar` | both | Top-of-atmosphere pressure (bar) |
| `pressure_bottom_bar` | both | Bottom pressure (bar) |
| `temperature_range_k` | both | [min, max] temperature bounds (K) |
| `metallicity_log10_range` | both | [min, max] log10([M/H]) |
| `c_to_o_range` | both | [min, max] C/O ratio |
| `s_to_o_range` | both | [min, max] S/O ratio |
| `gravity_range_cm_s2` | vulcan only | [min, max] surface gravity (cm/s^2) |
| `planet_radius_range_cm` | vulcan only | [min, max] planet radius (cm) |
| `stellar_radius_range_rsun` | vulcan only | [min, max] stellar radius (solar radii); equal endpoints keep it fixed |
| `semi_major_axis_range_au` | vulcan only | [min, max] orbital separation (AU); equal endpoints keep it fixed |
| `zenith_angle_range_deg` | vulcan only | [min, max] stellar zenith angle in degrees; equal endpoints keep it fixed |
| `diurnal_factor_range` | vulcan only | [min, max] diurnal averaging factor; equal endpoints keep it fixed |
| `kzz_cm2_s` | vulcan only | Constant eddy diffusion coefficient |

### `temperature_profiles`

| Key | Description |
|-----|-------------|
| `source_mode` | `"analytic"`, `"pt_library"`, or `"mixed"` |
| `analytic_probability` | Fraction of profiles from analytic sampler (mixed mode) |
| `data_glob` | Glob pattern for PT-library .dat files |
| `filters` | Numeric/boolean filters on PT-library metadata |
| `validation` | `min_temperature_k`, `max_temperature_k` bounds |
| `analytic_sampler` | Parameters for Line et al. (2013) radiative-equilibrium profiles |

Supported filter keys: `Teq`, `LogMet`, `LogDrag`, `Mstar`, `Rp`, `logG` (numeric); `TiOVO` (boolean).

`temperature_profiles.analytic_sampler.reference_gravity_m_s2` controls only
the analytic PT-profile shape. It does not constrain sampled VULCAN surface
gravity or planet radius.

### `generation`

| Key | Description |
|-----|-------------|
| `mode` | `"vulcan"` (uses VULCAN source tree for both FastChem and VULCAN runs) |
| `num_runs` | Target number of successful runs |
| `seed` | Sampling seed |
| `overwrite` | Whether to overwrite existing raw data |
| `reuse_raw_if_present` | Skip generation if raw data exists |
| `parallel_workers` | Number of parallel generation workers |
| `backfill` | `{enabled, max_retries}` for VULCAN failure recovery |

### `normalization`

| Key | Description |
|-----|-------------|
| `split` | `{train_fraction, val_fraction, test_fraction, seed}` |
| `state_floor` | Floor value for log-standard normalization (prevents -inf) |
| `target_method` | Normalization method for targets (`"log-standard"`) |
| `sequence_methods` | Per-feature methods: `pressure_bar`, `temperature_k`, optionally `kzz_cm2_s` |
| `global_methods` | Per-feature methods for all global inputs |
| `spectrum_floor` | Positive floor used by on-the-fly spectrum log-flux normalization (vulcan only) |

Normalization methods:
- `"standard"`: z-score `(x - mean) / std`
- `"log-standard"`: `(log10(clip(x, floor)) - mean) / std`
- `"none"`: pass-through (for boolean/one-hot features)

### `training`

| Key | Description |
|-----|-------------|
| `seed` | Training seed |
| `batch_size` | Samples per batch |
| `epochs` | Maximum training epochs |
| `learning_rate` | Peak learning rate |
| `min_lr` | Minimum learning rate |
| `warmup_epochs` | Linear warmup epochs (0 -> learning_rate) |
| `early_stopping_patience` | Stop after this many epochs without improvement |
| `scheduler` | `{name, factor, patience, threshold}` or `{name: "cosine"}` |
| `weight_decay` | AdamW weight decay |
| `gradient_clip` | Global L2 gradient norm clip |
| `loss` | Loss weights (see below) |

### `training.loss`

| Key | Required for | Description |
|-----|-------------|-------------|
| `lambda_z` | both | Weight on MSE in normalized space (primary signal) |
| `lambda_phys` | both | Weight on MSE in log10 physical space |

### `model` (MLP)

When `model_type = "mlp"`:

| Key | Description |
|-----|-------------|
| `d_hidden` | Hidden width of per-level MLP |
| `num_hidden_layers` | Number of hidden layers |
| `conditioning_hidden_dim` | Hidden width of FiLM conditioning MLP |
| `film_clamp` | Symmetric clamp on FiLM gamma/beta |
| `activation` | Optional. Activation function (default: `"leaky_relu"`) |
| `dropout_rate` | Optional. Dropout probability (default: 0.05) |

### `model` (Transformer)

When `model_type = "transformer"`:

| Key | Description |
|-----|-------------|
| `d_model` | Transformer hidden width (must be divisible by `nhead`) |
| `nhead` | Number of attention heads |
| `num_layers` | Number of transformer blocks |
| `dim_feedforward` | FFN hidden width (must be >= `d_model`) |
| `conditioning_hidden_dim` | Hidden width of FiLM conditioning MLP |
| `film_clamp` | Symmetric clamp on FiLM gamma/beta |
| `output_head_divisor` | Output bottleneck: `d_model // output_head_divisor` |
| `activation` | Optional. Activation function (default: `"leaky_relu"`) |
| `dropout_rate` | Optional. Dropout probability (default: 0.05) |

Supported activations: `relu`, `gelu`, `silu`, `tanh`, `elu`, `selu`, `softplus`, `leaky_relu`.

### `vulcan` (required only for `chemistry_type = "vulcan"`)

#### `vulcan.physics_toggles`

10 boolean flags controlling VULCAN physics:

```
use_photochemistry, use_ion_chemistry, use_eddy_diffusion,
use_molecular_diffusion, use_upwind_molecular_diffusion,
use_boundary_conditions, use_condensation, use_settling,
use_initial_cold_trap, use_sat_surface_h2o
```

#### `vulcan.runtime`

| Key | Description |
|-----|-------------|
| `chemistry_file` | Path to VULCAN chemistry network file |
| `atm_base` | Default atmosphere base gas (`H2`, `N2`, `O2`, `CO2`, or `H2O`) |
| `t_cross_sp` | Cross-section species list |
| `rocky` | Fixed runtime flag passed through to `vulcan_cfg.py` |
| `top_bc_flux_file` | Optional top boundary-condition flux file override |
| `bot_bc_flux_file` | Optional bottom boundary-condition flux file override |

Optional internal keys (not learned inputs): `python_executable`, `cfg_file`, `worker_root`, `regenerate_chem_funs`, `cfg_assignments`, `use_lowT_limit_rates`, `use_adaptive_rtol`.

#### `vulcan.science_presets`

Optional list of per-run science variations. Each preset supports:
- `name` (required)
- `atm_base` (optional, falls back to `vulcan.runtime.atm_base`)
- `physics_toggles` (optional, omitted values fall back to `vulcan.physics_toggles`)

When omitted, config validation synthesizes a single default preset from
`vulcan.physics_toggles` and `vulcan.runtime.atm_base`.

#### `vulcan.stellar_spectrum`

This section is required for all VULCAN configs. The current generation,
preprocessing, and upstream VULCAN contracts still consume the stellar flux
file and sampled irradiation geometry even when `use_photochemistry = false`.

| Key | Description |
|-----|-------------|
| `template_name` | Name of default stellar spectrum |
| `template_file` | Path to default spectrum .dat file |
| `library_glob` | Optional glob for spectrum library; when omitted, VULCAN sampling falls back to `template_file` only |
| `max_tokens` | Maximum number of packed spectrum tokens |
| `latent_dim` | Spectrum encoder latent dimension |
| `hidden_dim` | Spectrum encoder hidden width |
| `num_latents` | Number of learned Perceiver latent queries |
| `num_layers` | Number of latent self-attention blocks |
| `num_heads` | Number of spectrum-attention heads |
| `fourier_features` | Number of log-wavelength Fourier feature pairs |
| `encoder_mode` | `"perceiver"` |
| `wavelength_min_nm` | Minimum wavelength (nm) |
| `wavelength_max_nm` | Maximum wavelength (nm) |
| `dbin1_nm` | Optional fine native-grid bin width below the transition wavelength; defaults to `0.1` |
| `dbin2_nm` | Optional coarse native-grid bin width above the transition wavelength; defaults to `2.0` |
| `dbin_12trans_nm` | Optional transition wavelength between `dbin1_nm` and `dbin2_nm`; defaults to `240.0` |
| `teff_k` | Optional provenance/template-generation metadata only |
| `radius_rsun` | Optional legacy fixed stellar radius; backfilled to `sampling.stellar_radius_range_rsun = [x, x]` |
| `semi_major_axis_au` | Optional legacy fixed orbital distance; backfilled to `sampling.semi_major_axis_range_au = [x, x]` |
| `diurnal_factor` | Optional legacy fixed diurnal factor; backfilled to `sampling.diurnal_factor_range = [x, x]` |
| `zenith_angle_deg` | Optional legacy fixed zenith angle; backfilled to `sampling.zenith_angle_range_deg = [x, x]` |

The spectrum pipeline preserves native wavelength grids through raw generation
and preprocessing. At training time, each spectrum is sanitized, clipped to the
configured wavelength interval, optionally compressed with flux-conserving
log-space bins when it exceeds `max_tokens`, and stored as padded wavelength,
flux, and mask arrays. The model normalizes spectra on-the-fly in log-flux
space and keeps absolute-scale summary scalars in the encoder head.

For variable-star training, the spectrum library is selected through
`library_glob` and the per-run irradiation geometry is sampled from the
`sampling.*range*` controls. Recommended `normalization.global_methods` for
those learned geometry inputs are:
- `r_star_rsun`: `log-standard`
- `semi_major_axis_au`: `log-standard`
- `zenith_angle_deg`: `standard`
- `diurnal_factor`: `standard`
