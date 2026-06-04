# Config Guide

## Overview

Every config is defined by two top-level selectors:

| Key | Values | Selects |
|-----|--------|---------|
| `chemistry_type` | `fastchem`, `vulcan`, `exogibbs` | Target contract and learned inputs |
| `model_type` | `transformer` | Prediction architecture |

The three supported combinations are:
- `fastchem + transformer`
- `vulcan + transformer`
- `exogibbs + transformer`

Shipped configs:
- `config/exogibbs_luhman16a_10k.json` — ExoGibbs thermochemical equilibrium for the Luhman 16A brown-dwarf retrieval case
- `config/vulcan_luhman16a_10k.json` — VULCAN-JAX vertical-mixing kinetics for the same 10k-profile Luhman 16A setup

CLI:

```bash
python -m src.utils --config config/vulcan_luhman16a_10k.json --stage generation
python -m src.utils --config config/vulcan_luhman16a_10k.json --stage normalization
python -m src.utils --config config/vulcan_luhman16a_10k.json --stage training
python -m src.utils --config config/vulcan_luhman16a_10k.json --stage export
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
  processed/
    train/
    val/
    test/
```

`raw/`, `info/`, and `processed/` are auto-expanded by
`load_and_validate_config()`; do not set `paths.raw_root` or
`paths.processed_root` in the config.

### `data_spec`

| Key | Description |
|-----|-------------|
| `state_species` | Species tracked internally by the solver |
| `output_species` | Species the model predicts (subset of state_species) |

### `sampling`

| Key | Required for | Description |
|-----|-------------|-------------|
| `num_levels_range` | both | `[min, max]` number of vertical pressure levels (per-run count is sampled uniformly) |
| `pressure_top_bar_range` | both | `[min, max]` top-of-atmosphere pressure (bar) |
| `pressure_bottom_bar_range` | both | `[min, max]` bottom pressure (bar) |
| `temperature_range_k` | both | [min, max] temperature bounds (K) |
| `he_frac_range` | both | [min, max] He number fraction (n_He / n_H) |
| `c_frac_range` | both | [min, max] C number fraction (n_C / n_H) |
| `o_frac_range` | both | [min, max] O number fraction (n_O / n_H) |
| `n_frac_range` | both | [min, max] N number fraction (n_N / n_H) |
| `s_frac_range` | both | [min, max] S number fraction (n_S / n_H) |
| `gravity_range_cm_s2` | vulcan only | [min, max] surface gravity (cm/s^2) |
| `planet_radius_range_cm` | vulcan only | [min, max] planet radius (cm) |
| `stellar_radius_range_rsun` | vulcan only | [min, max] stellar radius (solar radii) |
| `semi_major_axis_range_au` | vulcan only | [min, max] orbital separation (AU) |
| `zenith_angle_range_deg` | vulcan only | [min, max] stellar zenith angle (degrees) |
| `diurnal_factor_range` | vulcan only | [min, max] diurnal averaging factor |
| `kzz_range_cm2_s` | vulcan only | [min, max] per-run eddy diffusion coefficient (cm²/s). Log-sampled by default — override via `scales.kzz_cm2_s`. Kzz remains depth-constant within each run. |
| `scales` | optional | Per-parameter sampling scale, `"linear"` (default) or `"log"`. Applies to any of `he_frac`, `c_frac`, `o_frac`, `n_frac`, `s_frac`, `kzz_cm2_s`. Use `log` when the range spans more than ~1 dex so samples are not biased toward the upper bound. |
| `corner_coverage` | optional | FastChem-only targeted joint sampling for retrieval-sensitive chemistry corners. When enabled, a configured fraction of runs remaps C/O unit coordinates into carbon-rich, oxygen-rich, high-C+O, and near-unity C/O strata and resamples PT profiles toward hot or large-temperature-range columns. VULCAN configs may include the block, but it is currently ignored. |

The scalar `num_levels`, `pressure_top_bar`, and `pressure_bottom_bar` keys
are no longer accepted; config loading raises if they appear. Use the
`_range` variants above.

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
| `num_runs` | Target number of successful runs |
| `seed` | Sampling seed |
| `overwrite` | Whether to overwrite existing raw data |
| `reuse_raw_if_present` | Skip generation if raw data exists |
| `parallel_workers` | Number of parallel generation workers (`0` = auto-detect from PBS/SLURM/OS) |
| `sample_chunk_size` | Runs per streaming sample+execute chunk (default `1000`). Each chunk's per-run HDF5 files are merged into one `chunks/chunk_XXXXXX_YYYYYY.h5` immediately on completion and the originals deleted, so this also caps the peak per-run file count on disk. Lower values keep that peak smaller at the cost of more chunk files at the final merge step. |
| `fastchem_timeout_seconds` | FastChem subprocess timeout in seconds (default `30.0`) |
| `vulcan_timeout_seconds` | VULCAN subprocess timeout in seconds (default `1800.0`). Stalled condensation runs raise `subprocess.TimeoutExpired`, which the worker pool routes through the standard `backfill` path. |
| `backfill` | `{enabled, max_retries}` for VULCAN failure recovery |
| `gpu_batch` | GPU-batched VULCAN-JAX generation (see below). `{enabled, batch_size, require_converged, host_setup_workers}` |

#### `generation.gpu_batch` (GPU nodes, `chemistry_type = "vulcan"` only)

On a GPU node, set `gpu_batch.enabled: true` to run generation **in-process on the
GPU** — profiles are bucketed by `(nz, toggle-combo, atm_base)` and each bucket is
integrated in one `jax.vmap`'d call via `vulcan_jax.OuterLoop.run_batch`, instead of
one CPU subprocess per profile. The HPC scripts auto-detect a usable GPU and flip this
on (`$VULCAN_GEN_GPU_BATCH`), so no config edit is needed on the cluster. Requires
**`vulcan-jax >= 0.1.10`**. `parallel_workers` is ignored in this mode.

| Key | Description |
|-----|-------------|
| `enabled` | Turn on the GPU-batched path (default `false`) |
| `batch_size` | Profiles per `vmap` device call (default `64`); pad/cap to device memory |
| `require_converged` | Send non-converged runs to backfill (default `true`); non-finite always backfills |
| `host_setup_workers` | Size of the CPU pool that builds profiles (atmosphere + rates + FastChem) to feed the GPU (`0` = auto-detect cores). `$VULCAN_GEN_HOST_SETUP_WORKERS` / PBS `GEN_HOST_SETUP_WORKERS` overrides it. |

Per-profile host setup (dominated by the FastChem subprocess) is fanned out across the
CPU cores by a persistent **spawn `ProcessPool`** so it doesn't starve the GPU; each
worker holds its own `vulcan_jax` import + a private FastChem tree. Recommend
`sample_chunk_size` be a comfortable multiple of `batch_size`.

`generation.mode` is a deprecated compatibility field. When present it must
remain `"vulcan"`, but backend selection now comes from `chemistry_type` and
the field is ignored by generation/preprocessing.

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
- `"log-minmax"`: `(log10(clip(x, floor)) - min) / (max - min)`
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
| `ema` | Optional. `{enabled, decay}`. Omit or set `enabled: false` to train without EMA; when on, `decay=0.999` is the usual default and EMA shadow weights are used for validation/test/export. |

### `training.loss`

`type` selects the loss variant. Both variants combine a normalized-space
MSE term (`lambda_z`) with a log10-space penalty on the physical residual
`r = log10(pred_VMR / target_VMR)`.

| Key | Required for | Description |
|-----|-------------|-------------|
| `type` | both | `"mae"` or `"huber"`. Picks the log10-space loss shape. |
| `lambda_z` | both | Weight on MSE in normalized space (smoothness regularizer) |
| `lambda_log10_mae` | `mae` | Weight on mean `|r|` in log10 physical space. |
| `lambda_log10_huber` | `huber` | Weight on Huber loss in log10 physical space (primary signal on `r`). |
| `huber_delta_log10` | `huber` | Huber transition point in dex (must be `> 0`). Quadratic for `|r| <= delta`, linear otherwise. `0.1` is a sensible default (~26% fractional error at the transition). |

MAE is the `delta -> 0` limit of Huber and is the current default in the
shipped `config/vulcan_luhman16a_10k.json`. Huber is the preferred
variant when outliers in the log-ratio tail dominate training signal.

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
| `activation` | Optional. Activation function (default: `"gelu"`) |
| `dropout_rate` | Optional. Dropout probability (default: `0.0`) |
| `norm_type` | Optional. `"layernorm"` (default) or `"rmsnorm"`. RMSNorm drops mean subtraction and bias; ~10% faster and a modern default. |
| `use_qk_norm` | Optional. `false` (default) or `true`. Applies RMSNorm to Q and K along `head_dim` before the attention dot product; stabilizes logits in deeper stacks. |
| `ffn_type` | Optional. `"dense"` (default) or `"swiglu"`. SwiGLU gains a third projection; re-budget `dim_feedforward` to `~2/3 × dense` for param parity. |
| `zero_init_film` | Optional. `false` (default) or `true`. Zero-initializes the final FiLM projection so γ=β=0 at step 0 (AdaLN-Zero). Combine with a wide `conditioning_hidden_dim`; starves the composition pathway otherwise. |

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

Optional internal keys (not learned inputs): `python_executable`, `cfg_file`, `worker_root`, `regenerate_chem_funs`, `cfg_assignments`, `use_lowT_limit_rates`, `use_adaptive_rtol`, `condensation`.

#### `vulcan.runtime.condensation`

Required when any science preset has `use_condensation = True`; forbidden
otherwise (the validator rejects `use_condensation = False` configs that
carry the block, and vice versa).

| Key | Description |
|-----|-------------|
| `condense_sp` | Gas-phase species that condense. Must be a subset of `{H2O, NH3, H2SO4, S2, S4, S8, C, H2S}` (the species for which VULCAN ships saturation-pressure data). |
| `non_gas_sp` | Paired condensate labels (e.g. `H2O_l_s`, `S8_l_s`). Same length as `condense_sp`. Every entry must also appear in `data_spec.state_species` and `data_spec.output_species`. |
| `fix_species` | Optional. Species frozen after condensation–evaporation EQ. Typically `condense_sp ∪ non_gas_sp`. |
| `use_relax` | Optional. Subset of `condense_sp` using the H2O/NH3 relaxation path. |
| `humidity` | Optional. Default `1.0`. |
| `start_conden_time`, `stop_conden_time`, `fix_species_time` | Optional. Defaults `1e6`, `1e8`, `1e8` seconds. |
| `fix_species_from_coldtrap_lev` | Optional. Default `true`. |
| `post_conden_rtol` | Optional. Default `0.1`. |
| `r_p`, `rho_p` | Per-condensate particle radius (cm) and density (g/cm³) maps. Required to cover every entry in `non_gas_sp` when any preset has `use_settling = True`. |

#### `vulcan.science_presets`

Optional list of per-run science variations. Each preset supports:
- `name` (required)
- `atm_base` (optional, falls back to `vulcan.runtime.atm_base`)
- `physics_toggles` (optional, omitted values fall back to `vulcan.physics_toggles`)

#### `vulcan.stellar_spectrum`

Required for data generation because VULCAN reads `sflux_file` unconditionally
at startup, even when `use_photochemistry` is `False`. Not used by the model.

| Key | Description |
|-----|-------------|
| `template_name` | Name of default stellar spectrum |
| `template_file` | Optional path to default spectrum .dat file |
| `library_glob` | Optional glob for spectrum library |
| `max_tokens` | Maximum number of packed spectrum tokens |
| `wavelength_min_nm` | Minimum wavelength (nm) |
| `wavelength_max_nm` | Maximum wavelength (nm) |
| `dbin1_nm` | Short-wavelength bin width in nm (default: 0.1) |
| `dbin2_nm` | Long-wavelength bin width in nm (default: 2.0) |
| `dbin_12trans_nm` | Transition wavelength between bin regimes in nm (default: 240.0) |
| `teff_k` | Effective temperature for blackbody template generation (default: 5485 K) |
| `radius_rsun` | Optional provenance metadata: stellar radius in solar radii |
| `semi_major_axis_au` | Optional provenance metadata: orbital separation in AU |
