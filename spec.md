# VULCAN Emulator Spec

## Overview

The repository is organized around one shared data pipeline and two explicit emulator
tasks:

- `task.kind = "full_vulcan"`: trajectory emulation with timestep conditioning,
  stellar-spectrum inputs, and a Transformer.
- `task.kind = "equilibrium_only"`: equilibrium-state emulation with an MLP.

Both tasks must use the same generation, normalization, split, metadata, and artifact
layout. Task selection changes model/runtime details, not the surrounding pipeline.

The public control surface is intentionally minimal:

```bash
python -m src.utils --config <path> --stage generation|normalization|training
```

No subcommands are part of the public interface.

## Source Layout

Only four top-level source packages define the pipeline:

- `src/models`: model definitions (JAX Transformer and Equilibrium MLP),
  initialization, and portable NPZ export with baked normalization.
- `src/training`: training orchestration (AdamW optimizer, warmup plus
  configurable LR scheduling, gradient clipping) and checkpointing.
- `src/data_generation`: raw data generation, analytic and PT-library temperature
  profile sampling, stellar spectrum handling, normalization fitting, and
  dataset I/O.
- `src/utils`: CLI dispatch, config loading/validation, path helpers, logging,
  provenance hashing, and NumPy compatibility.

No other `src/*` package is part of the supported project structure.

## Config Contract

Every run is controlled by one JSON config. The public schema uses these shared top-level
sections:

- `task`
- `paths`
- `data_spec`
- `sampling`
- `temperature_profiles`
- `generation`
- `normalization`
- `training`

Task-specific settings live under exactly one of:

- `full_vulcan`
- `equilibrium_only`

The config contract is:

- `task.kind` is required and must be either `full_vulcan` or `equilibrium_only`.
- `full_vulcan` contains Transformer hyperparameters and runtime-specific blocks such as
  `vulcan_runtime`, `stellar_spectrum`, and `trajectory_sampling`.
- `equilibrium_only` contains the equilibrium MLP hyperparameters.
- `normalization` is the full preprocess contract: raw-run splitting, train-only
  normalization fitting, and processed tensor export.
- The FastChem-native elemental channels are fixed in code as
  `["He_H", "C_H", "O_H", "N_H", "S_H"]`; they are exported in metadata but are
  not user-configurable under `data_spec`.

Internally, config validation still derives compatibility aliases such as
`model_type`, `generation.target_mode`, `training.model`, and `roth_sampler` so the
rest of the implementation can remain coherent while the public schema stays stable.

Hyperparameter tuning is done manually. No automated search or tuning is part of this
pipeline.

## Pipeline Stages

The three pipeline stages are:

1. `generation`
   Sample shared atmospheric inputs, build TP profiles, and write raw HDF5 runs plus raw
   manifests.
2. `normalization`
   Split raw runs into train/val/test, fit train-only normalization, and write processed
   tensors, normalization metadata, and processed manifests.
3. `training`
   Build the task-selected model, train it against the processed dataset, and save the
   best checkpoint.

The `generation` and `normalization` stages are shared across both tasks. Divergence
between tasks is limited to task-specific inputs and model code.

## Analytic Temperature Profile Parameterization

The analytic branch generates pressure-temperature profiles following the Line et al.
(2013) radiative-equilibrium parameterization with Robinson & Catling (2012) thermal
opacity modifications.  The profile equation is:

```
T^4(tau) = (3 * T_int^4 / 4) * (2/3 + tau)
         + (3 * T_irr^4 / 4) * ((1 - alpha) * xi(gamma_1, tau) + alpha * xi(gamma_2, tau))
```

where tau is the pressure-dependent gray infrared optical depth, T_int is the internal
heat flux temperature, T_irr is the irradiation temperature, alpha partitions two
visible-stream channels, and gamma_1, gamma_2 are the ratios of visible to infrared
Planck mean opacities.

The visible-channel contribution function xi is defined as:

```
xi(gamma, tau) = 2/3
               + (2 / (3 * gamma)) * (1 + (gamma * tau / 2 - 1) * exp(-gamma * tau))
               + (2 * gamma / 3) * (1 - tau^2 / 2) * E_2(gamma * tau)
```

where E_2 is the second-order exponential integral, computed via E_2(x) = exp(-x) - x * E_1(x)
with E_1 evaluated by continued-fraction (x > 1) or power-series (x <= 1) expansion.

The gray optical depth is integrated from a power-law opacity profile:

```
tau(P) = kappa_IR * P_Pa * (P / P_ref)^n / (g * (n + 1))
```

where kappa_IR is the infrared opacity in m^2/kg, n is the pressure power-law exponent,
g is the reference gravity, and P_ref is the reference pressure.

### Sampled Parameters

| Parameter | Distribution | Range / Stats | Unit |
|-----------|-------------|---------------|------|
| kappa_IR power-law exponent | Uniform | [-0.5, 1.0] | -- |
| log10(kappa_IR) | Uniform | [-2.5, 2.5] | m^2/kg |
| log10(gamma_1) | Uniform | [-2, 2] | -- |
| log10(gamma_2) | Uniform | [-2, 2] | -- |
| alpha | Uniform | [0, 1] | -- |
| Temperature shift | Uniform | [-600, 600] | K |
| T_int | Normal | mean=300, std=700 | K |
| T_irr | Normal | mean=1800, std=500 | K |
| Orbital separation modifier | Normal | mean=1.0, std=0.5 | -- |
| Convective adjustment | Bernoulli | probability=1/3 | -- |

Hard temperature bounds are enforced at [1, 4000] K.  When convective adjustment is
applied, super-adiabatic layers are relaxed toward a dry adiabat with a randomly sampled
convective alpha in [0.6, 0.9] and adiabatic index 1.4.

Implementation: `src/data_generation/sampling.py`, function
`_sample_analytic_temperature_profile_record`.

## Normalization Methods

All normalization statistics are fitted on the training split exclusively to prevent
data leakage.  The following methods are supported:

- **`standard`**: Z-score normalization in linear space.
  `z = (x - mean) / std`

- **`log-standard`**: Z-score normalization in log10 space with a floor to avoid log(0).
  `z = (log10(max(x, floor)) - mean_log) / std_log`

- **`none`**: Identity pass-through (mean=0, std=1).

- **`mixed`**: Per-feature method selection within a single block.  Each feature column
  has its own method, mean, std, and floor.

Normalization-method selection lives entirely in the config:

- `normalization.sequence_methods` for per-column sequence-static features,
- `normalization.global_methods` for the mixed global-conditioning block,
- `normalization.target_method` for targets,
- `normalization.state_method`, `normalization.spectrum_method`, and
  `normalization.log10_dt_method` for full-VULCAN-specific blocks.

Implementation: `src/data_generation/preprocess.py`.  JAX-compatible versions for
inference live in `src/models/export_bundle.py`.

## Post-Training Policy

Training produces a pickle checkpoint (`best.pt`) containing all artifacts needed for
downstream use:

- Model parameters (JAX arrays converted to NumPy)
- Model dimensions dataclass (serialized as dict)
- Normalization metadata (fitted statistics per block)
- Data contract (species order, dimensions, feature indices)
- Full config (minus the transient Roth profile cache)

## Export Bundle

The export system (`src/models/export_bundle.py`) converts a training checkpoint into
a portable NPZ bundle with format `jax_physical_bundle` (version 1).  The bundle
embeds all metadata required for standalone inference from physical-unit inputs.

### NPZ Layout

- `params/<dotted.key>`: Flattened model weight arrays (e.g., `params/layers.0.q.weight`).
- `meta/export_format`: Format identifier string.
- `meta/export_version`: Integer version string.
- `meta/model_dimensions`: JSON-serialized model dimensions.
- `meta/normalization`: JSON-serialized normalization metadata.
- `meta/data_contract`: JSON-serialized data contract.
- `meta/config`: JSON-serialized config.

### ExportedJAXModel

`load_exported_model(path)` returns an `ExportedJAXModel` with two inference methods
that bake normalization into the forward pass:

- `predict_equilibrium_profile(pressure_bar, temperature_k, global_inputs)`:
  Accepts physical-unit inputs (1-D arrays), normalizes them internally, runs the
  equilibrium MLP, and returns mixing ratios in physical space (or log10 if
  `return_log10=True`).  `global_inputs` is the ordered FastChem-native
  hydrogen-normalized elemental abundance vector (`n_X / n_H`), typically
  `["He_H", "C_H", "O_H", "N_H", "S_H"]`.

- `predict_transition_profile(pressure_bar, temperature_k, kzz_cm2_s, anchor_state,
  global_inputs, spectrum_flux, dt_s)`: Same pattern for the transition Transformer.
  `global_inputs` contains gravity, explicit elemental abundances, and any
  additional static physics/base flags recorded in the bundle contract.

Both methods handle all normalization (sequence static, global mixed, target inverse)
transparently.

## Shared Data Contract

Raw and processed datasets share one common contract for both tasks. Common fields include
the atmospheric state, pressure-grid-aligned TP inputs, and run/provenance metadata.

### Processed Array Shapes

**Equilibrium task** (per split directory):

| File | Shape | Description |
|------|-------|-------------|
| `sequence_inputs.npy` | `(N, nz, 2)` | Normalized [pressure, temperature] per level |
| `target_outputs.npy` | `(N, nz, target_dim)` | Normalized log10 mixing ratios |
| `global_inputs.npy` | `(N, global_dim)` | Normalized FastChem-native elemental abundances (`n_X / n_H`) |
| `run_ids.json` | `(N,)` | String run identifiers |

**Full-VULCAN task** (per split directory):

| File | Shape | Description |
|------|-------|-------------|
| `sequence_inputs.npy` | `(N, nz, 3)` | Normalized [pressure, temperature, Kzz] per level |
| `state_trajectories.npy` | `(N, T, nz, state_dim)` | Normalized mixing-ratio trajectories |
| `target_outputs.npy` | `(N, T, nz, target_dim)` | Normalized target mixing ratios |
| `global_inputs.npy` | `(N, global_dim)` | Normalized gravity, FastChem-native elemental abundances (`n_X / n_H`), and static conditioning flags |
| `spectrum_inputs.npy` | `(N, spectrum_dim)` | Normalized stellar spectrum |
| `time_s.npy` | `(N, T)` | Raw time values in seconds |
| `valid_steps_mask.npy` | `(N, T)` | Boolean mask for valid timesteps |
| `run_ids.json` | `(N,)` | String run identifiers |

Processed artifacts are versioned (`PROCESSED_DATA_VERSION = 7`) so incompatible schema
changes force regeneration.

## Temperature Profiles

`temperature_profiles` unifies analytic and PT-library sampling. For PT-library `.dat`
files, each `(lon, lat)` column is treated as a separate physically usable 1D TP profile.

The conversion rules are:

1. parse source metadata from the filename,
2. expand every `(lon, lat)` column into its own candidate profile,
3. sort and interpret the source pressure axis consistently,
4. interpolate temperature onto the configured VULCAN pressure grid in log-pressure
   space using shape-preserving PCHIP interpolation, and
5. store Roth provenance through raw and processed manifests.

This keeps the interpolation scientifically consistent with atmospheric profiles that are
tabulated versus pressure while preserving the originating Roth metadata for auditability.

## Artifact Policy

- `assets/` holds immutable external inputs such as PT libraries and stellar-spectrum
  templates. Real asset contents are not tracked in git.
- `data/` holds generated raw runs, processed tensors, manifests, normalization outputs,
  and other training data artifacts.
- `models/` holds checkpoints and model-specific training outputs.
- `uni_tests/fixtures/` holds the only tracked tiny PT fixtures and other synthetic test
  assets required for CI.

## ExoJAX API

`src/models/exojax_api.py` provides the differentiable ExoJAX-facing interface for
both supported emulator branches.  It wraps exported bundles as pure JAX functions
that support `jax.jit`, `jax.grad`, `jax.vjp`, and `jax.vmap`.

### Public surface

```python
from src.models.exojax_api import make_equilibrium_vmr_fn, make_transition_vmr_fn
```

**`make_equilibrium_vmr_fn(bundle) → (vmr_fn, species_labels)`**

Factory function.  Binds one equilibrium export bundle and returns a plain pure-JAX
function with top-to-bottom layer order:

```python
vmr_fn(
    temperatures_k: jax.Array,           # (nz,), top -> bottom
    pressures_bar: jax.Array,            # (nz,), top -> bottom
    elemental_abundances_x_h: jax.Array, # (nz, n_elements), top -> bottom
    gravity_cm_s2: jax.Array,            # (nz,), top -> bottom
) -> jax.Array                           # (nz, n_species), linear VMR
```

`species_labels` is the companion `list[str]` in bundle output order.  The elemental
input order is fixed in code as `["He_H", "C_H", "O_H", "N_H", "S_H"]`.

**`make_transition_vmr_fn(bundle) → (vmr_fn, species_labels)`**

Factory function.  Binds one transition/full-VULCAN export bundle and returns:

```python
vmr_fn(
    temperatures_k: jax.Array,           # (nz,), top -> bottom
    pressures_bar: jax.Array,            # (nz,), top -> bottom
    elemental_abundances_x_h: jax.Array, # (nz, n_elements), top -> bottom
    kzz_cm2_s: jax.Array,                # (nz,), top -> bottom
    gravity_cm_s2: jax.Array,            # (nz,), top -> bottom
    anchor_state: jax.Array,             # (nz, state_dim), top -> bottom
    spectrum_flux: jax.Array,            # (spectrum_dim,)
    dt_s: jax.Array,                     # scalar
) -> jax.Array                           # (nz, n_species), linear VMR
```

### ExoJAX contract

- Public layer order is always top-to-bottom.
- Wrappers reverse to the repository's internal canonical order and reverse outputs
  back before returning them.
- `elemental_abundances_x_h` uses the FastChem-native hydrogen-normalized number abundance `n_X / n_H`, not a total-gas mixing ratio and not relative-to-solar.
- `gravity_cm_s2` is required in both wrappers for contract consistency.
- Current trained models still assume column-constant chemistry and constant-per-level
  gravity, so eager calls reject vertically varying elemental-abundance or gravity
  profiles.
- Layer geometry, altitude grids, and radius grids are intentionally out of scope for
  this API version.

### Current training-manifold assumptions

The public API accepts explicit per-layer elemental-abundance and gravity profiles, but
the current training data is still generated from a column-constant latent sampler:

- `metallicity_log10_range`
- `c_to_o_range`
- `s_to_o_range`

Those latent controls are generation-time only.  Raw data materializes them into
explicit `He_H/C_H/O_H/N_H/S_H` profiles, and preprocessing reduces those constant
profiles back to the explicit global conditioning vectors used by the trained models.

Implementation: `src/models/exojax_api.py`.

## Extras Scripts

The `extras/` directory contains standalone utility scripts that operate outside the
main three-stage pipeline.  They share a common boilerplate pattern: define `_ROOT`,
patch NumPy compatibility, and insert the project root onto `sys.path`.

- **`extras/plot_profiles.py`**: Randomly sample one Roth PT-library profile and one
  analytic PT profile, then plot them side by side.  Accepts `--config`, `--seed`, and
  `--output` arguments.

- **`extras/inference_example.py`**: Load a trained checkpoint and the processed test
  split, run the forward pass on one randomly selected profile, denormalize predictions
  back to log10 mixing ratios, and plot predicted vs. true profiles.

- **`extras/export.py`**: Convert a training checkpoint to a portable NPZ bundle using
  `export_checkpoint_to_npz`.

- **`extras/benchmark.py`**: Time JIT compilation and steady-state inference for the
  active model type across multiple batch sizes and devices.

- **`extras/standalone_basic.py`**: Minimal equilibrium-bundle example using
  `ExportedJAXModel.predict_equilibrium_profile` directly with explicit FastChem-native
  elemental abundances.

- **`extras/standalone_test.py`**: End-to-end equilibrium demonstration with forward
  inference, optional FastChem comparison, and JAX autodiff examples.

- **`extras/compare_test_profile_fastchem.py`**: Utility for comparing stored test/raw
  profiles against FastChem using the explicit elemental-abundance contract.
