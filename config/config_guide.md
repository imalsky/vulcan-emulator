# Config Guide

## Overview

Every config is defined by two top-level selectors:

| Key | Values | Selects |
|-----|--------|---------|
| `chemistry_type` | `fastchem`, `vulcan` | Target contract and learned inputs |
| `model_type` | `transformer` | Prediction architecture |

The two supported combinations are:
- `fastchem + transformer`
- `vulcan + transformer`

Shipped configs:
- `config/vulcan_no_condensation.json` — VULCAN kinetic chemistry, gas-phase only
- `config/vulcan_condensation.json` — VULCAN kinetic chemistry with H2O/S8 condensation

CLI:

```bash
python -m src.utils --config config/vulcan_no_condensation.json --stage generation
python -m src.utils --config config/vulcan_no_condensation.json --stage normalization
python -m src.utils --config config/vulcan_no_condensation.json --stage training
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

---

## Chemistry Contracts

### FastChem (equilibrium)

- Sequence inputs: `[pressure_bar, temperature_k]` (nz, 2)
- Global inputs: `[He_H, C_H, O_H, N_H, S_H]` (5 elemental abundances, hydrogen-normalized)
- Target: FastChem `equilibrium/ymix`
- No Kzz, no gravity, no runtime knobs

### VULCAN (kinetic)

- Sequence inputs: `[pressure_bar, temperature_k, kzz_cm2_s]` (nz, 3)
- Global inputs: `[gravity_cm_s2, planet_radius_cm, He_H, C_H, O_H, N_H, S_H, r_star_rsun, semi_major_axis_au, zenith_angle_deg, diurnal_factor]` + 10 physics toggles + 5 atm_base one-hots (26 total)
- Target: converged VULCAN `final_state/ymix_output`
- Raw VULCAN HDF5 stores the layerwise gravity profile under `inputs/gravity_cm_s2`, but the learned contract uses scalar surface gravity and scalar planet radius from `globals/*`

**Note:** Photochemistry is not currently supported. The `use_photochemistry`
toggle must be `False` for all science presets. Support will be re-added in a
future release.

The fixed elemental order is internal and not configurable:

```python
["He_H", "C_H", "O_H", "N_H", "S_H"]
```

These are hydrogen-normalized absolute abundances `n_X / n_H`. The generation sampler draws `[M/H]`, `C/O`, and `S/O`, then converts to the fixed `X/H` channels.

---

## Architecture Contract

The Transformer uses FiLM (Feature-wise Linear Modulation) to inject global context into per-level predictions. Layer norms and residual connections are architectural defaults, not config-controlled.

### FiLM conditioning

1. `global_inputs` -> conditioning MLP -> per-layer gamma/beta
2. Per layer: `x = x * (1 + clip(gamma)) + clip(beta)`, clipped to `[-film_clamp, film_clamp]`

### Transformer

Pre-norm FiLM-conditioned Transformer.

```
Input: Linear(sequence_dim -> d_model) + sinusoidal positional encoding

Per block:
  a. Pre-norm (ln1) -> multi-head self-attention -> dropout -> residual add
  b. FiLM: x = x * (1 + gamma) + beta
  c. Pre-norm (ln_ffn) -> FFN (up-project, activation, dropout, down-project) -> residual add

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
| `stellar_radius_range_rsun` | vulcan only | [min, max] stellar radius (solar radii) |
| `semi_major_axis_range_au` | vulcan only | [min, max] orbital separation (AU) |
| `zenith_angle_range_deg` | vulcan only | [min, max] stellar zenith angle (degrees) |
| `diurnal_factor_range` | vulcan only | [min, max] diurnal averaging factor |
| `kzz_cm2_s` | vulcan only | Constant eddy diffusion coefficient |

### `temperature_profiles`

| Key | Description |
|-----|-------------|
| `source_mode` | `"analytic"`, `"pt_library"`, or `"mixed"` |
| `analytic_probability` | Fraction of profiles from analytic sampler (mixed mode) |
| `data_glob` | Glob pattern for PT-library .dat files |
| `filters` | Numeric/boolean filters on PT-library metadata |
| `validation` | `min_temperature_k`, `max_temperature_k` bounds |
| `analytic_sampler` | Parameters for Piette & Madhusudhan (2019) Guillot-based profiles |

Supported filter keys: `Teq`, `LogMet`, `LogDrag`, `Mstar`, `Rp`, `logG` (numeric); `TiOVO` (boolean).

### `generation`

| Key | Description |
|-----|-------------|
| `mode` | `"vulcan"` or `"synthetic"` |
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

### `model`

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

**Note:** `use_photochemistry` must be `False` for all science presets.

#### `vulcan.runtime`

| Key | Description |
|-----|-------------|
| `chemistry_file` | Path to VULCAN chemistry network file |
| `atm_base` | Default atmosphere base gas (`H2`, `N2`, `O2`, `CO2`, or `H2O`) |
| `t_cross_sp` | Cross-section species list |
| `rocky` | Fixed runtime flag passed through to `vulcan_cfg.py` |

Optional internal keys (not learned inputs): `python_executable`, `cfg_file`, `worker_root`, `regenerate_chem_funs`, `cfg_assignments`, `use_lowT_limit_rates`, `use_adaptive_rtol`.

#### `vulcan.science_presets`

Optional list of per-run science variations. Each preset supports:
- `name` (required)
- `atm_base` (optional, falls back to `vulcan.runtime.atm_base`)
- `physics_toggles` (optional, omitted values fall back to `vulcan.physics_toggles`)

#### `vulcan.stellar_spectrum`

Optional section used by the data generation pipeline (not the model).
When present, supports:

| Key | Description |
|-----|-------------|
| `template_name` | Name of default stellar spectrum |
| `template_file` | Optional path to default spectrum .dat file |
| `library_glob` | Optional glob for spectrum library |
| `max_tokens` | Maximum number of packed spectrum tokens |
| `wavelength_min_nm` | Minimum wavelength (nm) |
| `wavelength_max_nm` | Maximum wavelength (nm) |
| `teff_k` | Optional provenance/template-generation metadata |
