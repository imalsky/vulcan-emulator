# VULCAN Emulator Spec

## 1. Purpose

This repository trains a transformer-based surrogate model for VULCAN 1D chemical kinetics.
The model is a single-dt-jump operator: given an atmospheric state at time t and a time
interval dt, it predicts the state at time t+dt. It learns over a range of dt values
(approximately log-spaced) so that the same model handles short and long jumps. This
makes it suitable for autoregressive rollout with variable step sizes.

The current scope is explicit:

- generate or reuse raw VULCAN trajectories,
- sample variable-dt transition pairs from those trajectories (all ordered snapshot pairs
  weighted for log-uniform dt coverage),
- train a transformer to predict future mixing ratios from the current state, atmospheric
  inputs, and realized dt,
- evaluate single-step behavior and optional autoregressive rollout against held-out
  VULCAN data,
- export and inspect trained runs with a small set of local utility scripts.

This is not:

- a direct steady-state regressor from initial conditions alone,
- a photochemical model,
- a legacy multi-layout codebase with implicit path inference.

## 2. Current Operating Regime

The shipped main config is intentionally scoped.

- `trajectory_sampling.mode = "log_uniform_all_pairs"`
- one model is trained over a bounded log-uniform dt range [1s, 10^12 s]
- the current species set is a reduced top-8 state/output set

Current shipped species:

- `H2`
- `He`
- `H2O`
- `CO`
- `CO2`
- `CH4`
- `N2`
- `NH3`

The model learns a variable-dt transition operator:

- inputs: atmospheric profiles, globals, current mixing ratios, realized dt
- targets: future VULCAN mixing ratios after that dt

## 3. Variable-dt Jump Design

### Pair Generation

Each VULCAN trajectory saves T snapshots at irregular times determined by the adaptive
solver. For one trajectory with T snapshots, every ordered pair (i, j) where j > i is
a candidate training example, giving T*(T-1)/2 possible pairs. These are filtered to
the configured dt range [dt_min_s, dt_max_s].

VULCAN time grids do not align across runs. This is intentional: the natural spread
of realized dt values from the adaptive solver provides continuous coverage across
the dt range without snapping to a fixed grid. The model is conditioned directly on
the realized actual_dt_s (via log10 transform), so it learns a continuous dt operator.

### Sampling Strategy

The sampler weights candidates by 1/dt to achieve approximate uniform density in
log10(dt) space. From each trajectory, `pairs_per_run` examples are drawn without
replacement using these weights.

### Training Size Controls

Two config keys directly control training set size:

- `generation.num_runs`: number of VULCAN simulations (each produces one trajectory)
- `trajectory_sampling.pairs_per_run`: transition pairs sampled per trajectory

Total training pairs ~ num_runs * pairs_per_run (modulo split ratios and unusable runs).

### dt Range

The dt range [dt_min_s, dt_max_s] is configured in `trajectory_sampling`. The shipped
range [1s, 10^12 s] spans 12 orders of magnitude, covering fast kinetics through
near-steady-state transitions. This range is approximately logarithmic in coverage
due to the 1/dt weighting.

### Autoregressive Use

Although trained as a single-step operator, the model supports autoregressive rollout
when `output_species == state_species` (closed-loop). The rollout evaluation validates
that single-step accuracy composes well over multiple sequential jumps.

## 4. Supported Physics

All VULCAN inputs that affect thermochemical evolution are exposed to the model either
as profile inputs, global conditioning scalars, or physics toggles.

### Included

- thermochemistry from the local VULCAN installation (NCHO network),
- sampled T-P profiles (modified Line-2013 two-stream parameterization),
- sampled Kzz profiles (power-law family with floor/cap),
- sampled gravity (uniform over [500, 20000] cm/s^2),
- sampled metallicity (log-uniform) and C/O ratio,
- atmosphere base composition (H2/N2/O2/CO2/H2O, one-hot encoded),
- eddy diffusion toggle,
- molecular diffusion toggle,
- upwind molecular diffusion toggle,
- condensation toggle,
- settling toggle (requires condensation),
- cold-trap initialization toggle,
- saturated-surface H2O toggle,
- low-temperature-limited rates toggle,
- adaptive relative tolerance toggle,
- boundary conditions (top/bottom flux, fixed bottom species).

### Excluded

- photochemistry (requires stellar flux, orbit, zenith angle - out of scope),
- ion chemistry (out of scope),
- vertical advection (hardcoded off),
- non-EQ initialization modes (require additional input artifacts),
- chemistry-network regeneration from this repo.

### Global Conditioning Inputs (ordered)

1. `gravity_cm_s2` (log-standard normalized)
2. `metallicity_log10` (standard normalized - already in log10)
3. `c_to_o` (standard normalized)
4. `log10_dt_s` (standard normalized - log10 of realized dt)
5. Physics toggles: 10 boolean flags (unnormalized 0/1)
6. `atm_base` one-hot: 5 indicators for H2/N2/O2/CO2/H2O (unnormalized 0/1)

Total: 19 global conditioning inputs.

## 5. Model Architecture

Encoder-only transformer with FiLM (Feature-wise Linear Modulation) conditioning.

### Data Flow

```
Sequence inputs [batch, nz, 3+state_dim]    Global inputs [batch, 19]
        |                                    (gravity, abundances, dt, toggles, ...)
  Profile projection (P,T,Kzz -> d_model)         |
  + State projection (ymix -> d_model)       MLP -> d_model
        |                                          |
  LayerNorm + Sinusoidal PE                        |
        |                                          |
  Initial FiLM conditioning  <---------------------+
        |                                          |
  N x [ TransformerEncoderLayer                    |
        + per-block FiLM ]   <---------------------+
        |
  Output head (d_model -> hidden -> target_dim)
        |
  + anchor_subset (residual skip)
        |
  Predictions [batch, nz, target_dim]
```

### Key Design Decisions

- **Separate profile/state projections**: P, T, Kzz projected independently from
  ymix then summed, allowing distinct feature learning.
- **Residual delta prediction**: Output head predicts a correction added to the anchor
  state. At initialization (weights=0), the model outputs identity. For short dt,
  delta approaches zero naturally.
- **Bidirectional attention**: No causal mask. All pressure levels are physically
  coupled through vertical transport in the continuity equation.
- **Pre-norm**: `norm_first=True` for stable gradients with FiLM conditioning.
- **FiLM tanh clamping**: gamma/beta are bounded by `clamp * tanh(x/clamp)` to
  prevent early-training instabilities. Default clamp=10.0.
- **Profile input dimension**: Hardcoded to 3 (pressure, temperature, Kzz). This is
  enforced by the data contract and cannot change without reprocessing.

### Default Hyperparameters

- d_model=128, nhead=8, num_layers=4, dim_feedforward=384
- dropout=0.0, film_clamp=10.0, output_head_divisor=2
- max_sequence_length=64, conditioning_hidden_dim=128

## 6. Config Contract

The config is strict. Missing required keys and unexpected keys in strict sections
are errors. There are no legacy config fallbacks, no implicit defaults, no deprecated
key revival. Booleans are type-checked (not coerced from int). All logarithms are base-10.

### Required Sections

- `paths`: 7 relative path keys
- `generation`: num_runs, num_workers, split_ratios, seed, failure_policy, timeout, shard_size, save_evo_frq, max_trajectory_snapshots
- `trajectory_sampling`: mode, pairs_per_run, dt_min_s, dt_max_s, min_future_saved_steps, rollout_eval_points
- `vulcan_runtime`: runtime, dt_min, dt_max, count_max, trun_min, count_min, ini_mix, atm_base
- `tp_sampler`: pressure_grid, temperature_limits_k, distribution specs for opacity/gamma/T params
- `gravity_sampler`: distribution + bounds
- `kzz_sampler`: power-law mode, log10_kzz range, beta range, floor/cap
- `abundance_sampler`: metallicity range, C/O range, solar abundances
- `physics_toggles`: 12 boolean flags (photochemistry and ion chemistry must be false)
- `data_spec`: state/output species, required inputs, time transform, strict_non_finite
- `normalization`: epsilon, per-variable methods, target method
- `precision`: per-stage dtype control (input, stats, model, forward, loss, optimizer, AMP)
- `training`: device, batch_size, epochs, lr, model hyperparameters, data_loading, output_folder

### Cross-Section Validation

- `normalization.target_method` must match `normalization.sequence_methods.anchor_ymix`
  (residual skip requires shared normalized space)
- `normalization.global_methods.metallicity_log10` cannot use log-based methods
  (already log10-transformed)
- `normalization.global_methods.log10_dt_s` cannot use log-based methods
  (already log10-transformed)
- `d_model` must be even and divisible by `nhead`
- AMP requires CUDA device and float32 model dtype
- `precision.forward_dtype` must match `precision.model_dtype`
- `output_species` must be a subset of `state_species`

## 7. Filesystem Layout

Flat data layout. No nested `raw/runs` structure.

```text
data/
  raw/
    run_000000.h5
    run_000001.h5
    ...
  processed/
    train/
    val/
    test/
    normalization_metadata.json
    processed_summary.json
    processed_fingerprint.json
  dataset_manifest.json
  splits.json

models/
  <output_folder>/
    best.pt
    last.pt
    metrics.json
    data_contract.json
    normalization_metadata.json
    processed_fingerprint.json
    training_log.csv
    figures/

logs/
  runtime_config_<action>_<timestamp>.json
  <action>_<timestamp>.log
  training_progress_<output_folder>.log
```

`generation.worker_root` is temporary scratch space for copied VULCAN trees. It is
removed after generation unless debug retention is enabled.

## 8. Raw Trajectory Contract

Each raw HDF5 file stores one trajectory with:

- `inputs/pressure_bar`, `inputs/temperature_k`, `inputs/kzz_cm2_s`
- `inputs/state_species`, `inputs/output_species`
- `globals/gravity_cm_s2`, `globals/metallicity_log10`, `globals/c_to_o`
- `globals/<physics_toggle>` for each of 10 toggles
- `globals/atm_base_<name>` for each of 5 atmosphere bases (one-hot)
- `trajectory/time_s`, `trajectory/ymix_state`, `trajectory/ymix_output`

The raw loader requires:

- finite numeric arrays,
- strictly increasing `time_s`,
- at least one saved future state beyond t=0,
- requested species to exist in the stored species lists.

The loader may read a subset of species from a larger stored raw file, but raw files
must still live directly under `raw_root`.

## 9. Generation And Reuse Rules

`--gen` behaves as follows:

1. If `raw_root` already contains `run_*.h5`, raw generation is skipped and those files are reused.
2. Otherwise, the code samples atmospheric/input conditions and launches VULCAN jobs.
3. Raw runs that fail outright may be tolerated depending on `generation.failure_policy`.
4. Raw runs that cannot produce valid transition pairs for the current dt range are skipped (not rejected - this avoids discarding useful data unnecessarily).
5. Only usable runs are split into train, val, and test.
6. Processed shards are rebuilt under `processed_root`.

Failure behavior:

- if no raw runs exist and no new runs succeed, generation fails,
- if raw files exist but none produce valid transition pairs, generation fails,
- processed reuse is allowed only when the processed fingerprint matches the current config and artifact set.

## 10. Transition Sampling Contract

The only supported training regime is:

- `trajectory_sampling.mode = "log_uniform_all_pairs"`

Sampling rules:

- all T*(T-1)/2 ordered saved-snapshot pairs are candidates for a trajectory with T snapshots,
- valid pairs are filtered by actual dt range [dt_min_s, dt_max_s],
- `pairs_per_run` controls how many pairs are sampled from each trajectory (without replacement),
- sampling is weighted by 1/dt to approximate uniform coverage in log10(dt) space,
- the model is conditioned directly on realized `actual_dt_s` (via log10 transform).

No profiles are rejected for having "wrong" time grids. The variable time grids
from VULCAN's adaptive solver provide natural dt diversity.

## 11. Processed Data Contract

Processed shards contain:

- sequence inputs with shape `[batch, nz, 3 + state_dim]`
- global inputs with shape `[batch, global_dim]`
- targets with shape `[batch, nz, output_dim]`
- raw `dt_s` values with shape `[batch]`

Sequence feature order is fixed:

- `pressure_bar`
- `temperature_k`
- `kzz_cm2_s`
- `anchor_ymix:<species>` for each configured state species

Global feature order:

- `gravity_cm_s2`, `metallicity_log10`, `c_to_o`, `log10_dt_s`
- 10 physics toggle flags
- 5 `atm_base` one-hot indicators

Normalization statistics are fit on the training split only (no data leakage).

## 12. Model And Training Contract

Inputs:

- pressure, temperature, Kzz profiles (3 channels),
- current mixing-ratio state (state_dim channels),
- gravity, metallicity, C/O, dt (4 scalars),
- physics toggles (10 booleans),
- atm_base one-hot (5 indicators).

Outputs:

- future mixing-ratio profiles for `output_species`.

Training:

- loss: MSE in normalized target space,
- optimizer: AdamW with explicit bias/norm no-decay groups,
- schedule: linear warmup + cosine decay to min_lr,
- gradient clipping: norm=1.0,
- checkpointing: best.pt (lowest val MSE), last.pt.

Rollout evaluation:

- allowed only when `output_species == state_species`,
- skipped otherwise because the predicted state is not closed under autoregressive iteration.

## 13. Inference And Export

Inference is handled through `src/inference.py`.

Supported interfaces:

- `PhysicalSpaceStandaloneModel`: torch module with physical-unit inputs and outputs,
  handles normalization roundtrip internally, compatible with torch.compile
- `VulcanPredictor`: numpy-facing convenience wrapper with predict() and rollout()

Root helper:

- `read.py` reads one trained run and prints a compact summary

Standalone export:

- `extras/export_cpu_gpu.py` exports PT2 models on CPU and CUDA when available

## 14. Local Utility Scripts

Non-core local scripts live under `extras/`.

Current scripts:

- `export_cpu_gpu.py`
- `plot_true_vs_pred_profile.py`
- `plot_species_mae.py`
- `training_progression.py`
- `benchmark.py`

Script behavior:

- configured by top-of-file globals rather than CLI argument parsing,
- plot-producing scripts use `extras/science.mplstyle`,
- plots are saved under `<run_dir>/figures/`,
- plots are square-format outputs.

## 15. Tests And Repo Structure

Unit tests live under `uni_tests/`.

Project split:

- `src/`: runtime code only
- `uni_tests/`: unit and contract tests, plus one local error-summary helper
- `extras/`: local export/plot/benchmark utilities

Shared script helpers live at the repo root in `script_utils.py`.

## 16. Failure Philosophy

The repository is intentionally strict. Limited defensive coding.

- missing required config keys are errors,
- unexpected keys in strict sections are errors,
- missing files are errors,
- non-finite arrays are errors,
- incompatible processed artifacts are errors,
- old path layouts and deprecated config keys are not silently revived.

Unexpected behavior is treated as a bug, not a compatibility feature. The goal is
to fail fast and loudly rather than produce subtly wrong results.

No fallbacks. No implicit defaults. No silent coercion. If something is wrong, the
pipeline stops immediately with a clear error message.

## 17. Environment

### Local Development

- Conda environment: `nn` (or override via `VULCAN_EMULATOR_CONDA_ENV`)
- Run: `python src/main.py --gen` and `python src/main.py --train`

### HPC / Batch

Environment variables override config for portability:

- `VULCAN_EMULATOR_PROJECT_ROOT`: project root directory
- `VULCAN_EMULATOR_VULCAN_SOURCE`: VULCAN source tree location
- `VULCAN_EMULATOR_CONDA_ENV`: expected conda environment name

Path joining uses `os.path.normpath` instead of `Path.resolve()` so that
symlink-based canonical-path rewriting on cluster filesystems does not break
sibling-directory references.

## 18. Conventions

- All logarithms are base-10 (no natural log in config, normalization, or model input semantics).
- Padding mask: True = padding position (PyTorch convention).
- Pressure units: bar throughout the emulator (VULCAN uses dyne/cm^2 internally; conversion happens in vulcan_runner.py).
- Temperature: Kelvin.
- Kzz: cm^2/s.
- Gravity: cm/s^2.
- Config keys using `log10_` prefix indicate the value is already log10-transformed.
