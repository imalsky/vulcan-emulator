# Vulcan-Emulator Scientific Development Spec

## 1. Purpose And Scope

This project builds a machine-learning surrogate for VULCAN 1D atmospheric chemistry.

- Primary objective (v1):
  - Emulate **time-conditioned thermochemical trajectories** from VULCAN.
  - Inputs: atmospheric setup + initial chemistry state + query time.
  - Outputs: top-20 configured species vertical `ymix` profiles at requested time.
- Long-term objective:
  - Expand toward broader VULCAN feature parity (including photochemistry), without redesigning core architecture.
- Explicit v1 exclusion:
  - Photochemistry and ion chemistry are disabled by default.

## 2. Engineering Principles

- Fail fast:
  - Missing files, invalid config, invalid value ranges, unsupported precision, or inconsistent artifacts must raise immediately.
- No hidden fallback:
  - No silent mode switching, no auto-downgrades, no "best effort" behavior.
  - If requested behavior is unsupported, terminate with actionable error.
- Minimal complexity in checks:
  - Validate contracts once, early, and explicitly.
  - Avoid layered defensive branching that hides root causes.
- Pythonic and maintainable:
  - Clear function boundaries, small modules, explicit typing, deterministic IO contracts.
- Config is the single source of truth:
  - All science/runtime behavior is config-defined, not hardcoded in execution paths.

## 3. Repository Layout

Project root:
- `.`

Required layout:
- `src/`: all runtime code.
- `config/`: required config files.
- `data/raw/`: canonical generated trajectory data (HDF5).
- `data/processed/`: normalized NPY shards for training.
- `models/`: checkpoints and run artifacts.
- `testing/`: validation and analysis scripts.
- `unit_tests/`: fast artifact-light unit tests suitable for local Codex verification.
- `spec.md`: this contract.

No runtime code is allowed outside `src/` for normal operation.

### 3.1 Path Convention

- Use relative paths everywhere (code, config, docs, logs).
- Resolve all relative paths against project root (`.`).
- Absolute paths are disallowed in normal runtime configuration.

## 4. Environment Contract

- All commands run in conda environment `nn`.
- VULCAN is **assumed pre-installed and runnable**.
- Default VULCAN source path for subprocess execution is `../VULCAN-master`.
- If VULCAN runtime prerequisites are missing at execution time, command fails immediately with explicit remediation text.

Canonical command examples:
- `conda run -n nn python src/main.py --gen`
- `conda run -n nn python src/main.py --train`

## 5. CLI Contract (`src/main.py`)

`src/main.py` supports only explicit action flags:

- `--gen`:
  - Run full data-prep stage in one command:
    1. generate raw HDF5 trajectories via VULCAN subprocess runs
    2. normalize and write processed NPY shards
- `--train`:
  - Train surrogate from processed shards.

Rules:
- Exactly one action is required (`--gen` or `--train`).
- No `--install-vulcan`, no `--normalize`, no `--all`.
- Missing prerequisites are hard failures.
- No implicit fallback to alternate modes.

## 6. VULCAN Runtime Contract

### 6.1 Source Path

- VULCAN source path is config-defined.
- Default: `../VULCAN-master` relative to project root.
- v1 uses **local source path execution** (`python vulcan.py` from source tree), not package import.

### 6.2 Runtime Preflight (No Installer)

At the start of `--gen`, code must validate:

1. Source tree exists.
2. Required runtime files exist:
  - `vulcan.py`
  - `vulcan_cfg.py`
  - `fastchem_vulcan/`
  - compiled `fastchem_vulcan/fastchem`
3. VULCAN run command is executable in `nn` via a real smoke run in an isolated worker copy.

Any preflight failure is fatal. The pipeline must not attempt auto-install or auto-build.

## 7. Configuration Contract

All required keys must exist with exact type/range validation.

### 7.1 Top-Level Sections

- `paths`
- `generation`
- `tp_sampler`
- `gravity_sampler`
- `kzz_sampler`
- `abundance_sampler`
- `physics_toggles`
- `boundary_conditions` (optional; required only when enabled)
- `data_spec`
- `normalization`
- `precision`
- `training`

### 7.2 Required Section Details

#### `paths`
- `project_root`
- `vulcan_source_path` (default `../VULCAN-master`)
- `data_root`
- `models_root`
- `logs_root`

Path rule for this section:
- all values must be relative paths.

#### `generation`
- `num_runs` (default first milestone: `1000`)
- `num_workers` (multiprocess CPU workers)
- `snapshots_per_run` (default `128`)
- `snapshot_spacing` (`log_time`)
- `split_ratios` (`train=0.70`, `val=0.15`, `test=0.15`)
- `random_seed`
- `keep_vulcan_outputs_debug` (`false` default)
- `failure_policy` (`fail_on_first_error`)

#### `tp_sampler`
- Uses modified Line-2013 style parameterization used in prior RT work.
- Required parameter distributions include:
  - `log10_kappa_ir`
  - `log10_gamma1`
  - `log10_gamma2`
  - `alpha_partition`
  - `t_int`
  - `t_irr`
  - `temperature_shift`
  - `kappa_pressure_power_exponent`
  - `convective_adjustment_probability`
- Must also define physical reject rules (e.g., non-positive temperatures).

#### `kzz_sampler` (global config knobs)
- Strategy: hybrid literature-inspired power-law family.
- Required keys:
  - `mode` = `power_law_profile`
  - `log10_kzz_at_1bar_min` (default `8.0`)
  - `log10_kzz_at_1bar_max` (default `10.0`)
  - `beta_min` (default `0.4`)
  - `beta_max` (default `0.6`)
  - `kzz_floor_cm2_s`
  - `kzz_cap_cm2_s`
  - `pressure_unit_for_profile` (`bar`)
- Every run must log sampled Kzz parameters and realized profile summary.

#### `abundance_sampler` (global config knobs)
- Required:
  - `metallicity_mode` (log-scale multiplier)
  - `log10_metallicity_min`
  - `log10_metallicity_max`
  - `c_to_o_min`
  - `c_to_o_max`
  - mapping rules into VULCAN elemental abundance fields

#### `physics_toggles`
- `use_photochemistry` (default `false`, required)
- `use_ion_chemistry` (default `false`, required)
- `use_transport` (default `true`)
- `use_boundary_conditions` (default `false`)
- `use_condensation_optional` (default `true`)

#### `boundary_conditions` (optional)
- Required when `physics_toggles.use_boundary_conditions=true`
- Required keys:
  - `use_topflux`
  - `use_botflux`
  - `top_BC_flux_file`
  - `bot_BC_flux_file`
  - `use_fix_sp_bot`
- Boundary-condition file paths are relative to the VULCAN source tree copied into each worker.
- Enabling boundary conditions with no active top flux, bottom flux, or fixed bottom mixing ratios is invalid.

#### `data_spec`
- Explicit top-20 `target_species` list is required (no default list).
- `required_input_profiles`:
  - `pressure_bar`
  - `temperature_k`
  - `kzz_cm2_s`
- `required_global_inputs`:
  - gravity + abundance params + run metadata used for conditioning
- `required_state_inputs`:
  - initial `ymix` of configured species
- `time_input_transform` = `log10_time_seconds`
- strict shape/dtype requirements and required HDF5 keys

#### `normalization`
- Explicit per-variable methods only, no implicit defaults.
- Target `ymix` policy:
  - log-space transform with epsilon, then configured scaling.

#### `precision`
- default all FP32
- optional AMP/BF16 only when explicitly configured and hardware-supported
- invalid combinations hard-fail

#### `training`
- model family: transformer encoder + FiLM
- objective: time-conditioned trajectory regression
- `gpu_preload` is CUDA-only and must hard-fail on non-CUDA devices
- explicit `data_loading` policy:
  - `mode`: `auto` | `ram` | `disk`
  - bounded shard cache size and mmap threshold
  - explicit RAM safety fraction for `auto`
  - `use_device_prefetch` is CUDA-only and must hard-fail on non-CUDA devices
- OOM behavior: hard-fail unless user explicitly changes loading policy

## 8. Logarithm Convention

All logarithms in this project are **base-10 only**.

- No natural-log conventions are allowed in config semantics, normalization semantics, metrics semantics, or model-input transforms.
- Any config key that implies logarithmic behavior must use explicit base-10 naming (for example `log10_*`).

## 9. Scientific Generation Design

### 9.1 VULCAN-Conditioned Inputs

Mandatory model inputs in v1:

1. Vertical pressure profile (`pressure_bar`)
2. Vertical temperature profile (`temperature_k`)
3. Vertical Kzz profile (`kzz_cm2_s`)
4. Gravity
5. Elemental abundance controls (metallicity + C/O derived settings)
6. Initial composition profile for configured target species (`ymix0`)
7. Query time encoded as `log10(time_s)`

### 9.2 TP Profile Strategy

v1 uses the same high-level strategy as prior RT profile generation:
- Modified Line-2013 style parameterized TP family.
- Broad but controlled sampling for hot-Jupiter-relevant regime.
- Explicit rejection criteria for non-physical outcomes.
- No silent clipping to "fix" invalid samples.

### 9.3 Kzz Parameterization Notes (v1)

Kzz is uncertain and model-dependent in exoplanet atmospheres. It varies with dynamics, pressure, temperature, and tracer behavior. For v1, this project uses a conservative, explicit family to ensure learnable coverage without pretending Kzz is uniquely known.

Default profile family:
- `Kzz(p_bar) = clamp( Kzz_1bar * p_bar^(-beta), floor, cap )`

Default conservative sampled ranges:
- `log10(Kzz_1bar [cm^2 s^-1]) in [8, 10]`
- `beta in [0.4, 0.6]`
- `floor` and `cap` are required config values.

Rationale:
- wide enough to teach Kzz sensitivity
- conservative enough for stable v1 generation
- fully configurable for future literature updates

Future extension modes (not v1 default):
- temperature-dependent Kzz families
- species-dependent effective mixing parameterizations
- externally supplied dynamical Kzz profiles

### 9.4 Generation Orchestration

- Multiprocess CPU worker pool is default.
- Each worker runs VULCAN subprocess jobs in isolated working directories.
- Worker failure policy:
  - any run crash/convergence failure aborts full generation job.
- Every run produces:
  - sampled parameter record
  - VULCAN output extraction payload
  - deterministic run id and provenance metadata

## 10. Data Contracts

### 10.1 Raw Canonical Data

Format:
- HDF5 only.

Contract:
- Data grouped by run id, then snapshots.
- Required fields must include:
  - run metadata
  - profile inputs (`pressure_bar`, `temperature_k`, `kzz_cm2_s`)
  - global inputs (gravity, abundance controls, BC/meta used by model)
  - initial state (`ymix0`)
  - snapshot time `time_s`
  - targets (`ymix_target_top20`)
- Any missing key, shape mismatch, NaN/Inf is fatal.

`.vul` retention:
- default: do not retain full `.vul` outputs
- optional debug retention via config (`keep_vulcan_outputs_debug=true`)

### 10.2 Processed Training Data

Format:
- NPY shards.

Layout:
- `data/processed/train/`
- `data/processed/val/`
- `data/processed/test/`

Per split:
- shard arrays for sequence inputs, global inputs, and targets
- metadata JSON with shard counts, feature ordering, normalization fingerprint
- `processed_fingerprint.json` linking processed artifacts to config + raw run provenance

Split policy:
- split by run/case only (never by snapshot)
- ratios default `70/15/15`
- split leakage across run ids is disallowed and fatal.

`--gen` responsibility:
- `--gen` must produce both raw and processed artifacts in one execution.
- `--train` must refuse processed artifacts whose fingerprint/config/raw-run provenance no longer matches the current config.

## 11. Normalization Policy

- All variables require explicit configured transform.
- Missing method/stats is fatal.
- Target `ymix` transform:
  - log-space with epsilon (base-10)
  - explicit scaler per config
- Statistics are computed on train split only.
- No implicit normalization defaults.

## 12. Precision And Numerical Policy

Defaults:
- input/stat/model/forward/loss/optimizer-state: `float32`

Optional mixed precision:
- allowed only when config requests it and backend supports it
- BF16/AMP requests on unsupported backend are fatal

No automatic dtype fallback is allowed.

## 13. Model Architecture

### 13.1 Overview

Architecture family: encoder-only transformer with FiLM (Feature-wise Linear Modulation) conditioning and a regression output head.

Data flow:
```
Sequence inputs [batch, nz, input_dim]    Global inputs [batch, global_dim]
        |                                         |
  Input Projection (Linear → d_model)             |
        |                                         |
  Sinusoidal Positional Encoding                   |
        |                                         |
  Initial FiLM Conditioning  <--------------------+
        |                                         |
  N × [ TransformerEncoderLayer                    |
        + per-block FiLM ]   <--------------------+
        |
  Output Head (Linear → hidden → GELU → Linear → target_dim)
        |
  Predictions [batch, nz, target_dim]
```

### 13.2 Components

- **Input projection**: Linear layer mapping `input_dim` to `d_model`.
- **Positional encoding**: Standard sinusoidal encoding (Vaswani et al. 2017).
- **FiLM conditioning**: Global features projected to per-channel scale (`gamma`) and shift (`beta`). Applied as `(1 + gamma) * x + beta`. Clamped for stability.
  - Initial FiLM on embeddings before the encoder stack.
  - Per-block FiLM after each transformer encoder layer.
- **Transformer encoder layers**: Pre-norm architecture (`norm_first=True`), multi-head self-attention, GELU activation, `batch_first=True`.
- **Output head**: Two-layer MLP (`d_model → hidden → target_dim`) with GELU activation and no final activation clamp.

### 13.3 Sequence Inputs

Per-layer features concatenated into `input_dim` channels:
1. `pressure_bar` (1 channel)
2. `temperature_k` (1 channel)
3. `kzz_cm2_s` (1 channel)
4. `initial_ymix` for each of the 20 target species (20 channels)

Total: `input_dim = 23`

### 13.4 Global Inputs

Per-sample scalar conditioning features stacked into `global_dim` channels:
1. `gravity_cm_s2`
2. `metallicity_log10`
3. `c_to_o`
4. `log10_time_s` (query time for time-conditioned prediction)

Total: `global_dim = 4`

### 13.5 Key Design Decisions

- **Time conditioning via FiLM**: `log10(time_s)` is a global feature that modulates the sequence representation at every depth, enabling the single model to predict profiles at any queried time.
- **No causal masking**: The encoder uses bidirectional self-attention across all pressure levels.
- **Padding mask**: `True = padding position`, following PyTorch convention. v1 uses fixed-length sequences so masks are all `False`.
- **Export compatibility**: Architecture avoids dynamic control flow for `torch.export`/`torch.compile` compatibility.

### 13.6 Hyperparameters

All hyperparameters are config-defined. Current baseline from `config/config.json`:
- `d_model = 128`, `nhead = 4`, `num_layers = 3`, `dim_feedforward = 512`
- `dropout = 0.0`, `film_clamp = 10.0`
- `max_sequence_length = 120`
- `output_head_divisor = 2`

## 14. Training And Inference Policy

- Objective: predict top-20 species `ymix` profile at requested time.
- Training loss:
  - masked MSE regression loss on valid (non-padding) elements.
- Optimizer: AdamW with cosine-warmup LR schedule.
  - Biases and norm parameters excluded from weight decay.
- GPU preload is default loading mode.
- If preload OOM occurs:
  - hard-fail unless loading mode is explicitly reconfigured.

No hard scientific threshold gate in v1:
- always report metrics
- do not block by fixed percent-error cutoff.

Post-training inference/export:
- Runtime code must expose a physical-space predictor interface that accepts profile/state inputs and query time directly.
- Standalone export must normalize inputs internally and return physical-space `ymix` predictions.

## 15. Verification, Static Analysis, And Test Matrix

All checks run in `nn`.

Fast local integrity loop for Codex work:
- `conda run -n nn pytest -q unit_tests`
- `conda run -n nn ruff check src testing unit_tests`
- `conda run -n nn vulture src testing unit_tests`

Broader validation before accepting substantial pipeline changes:
- `conda run -n nn pytest -q`

Static checks:
- `ruff check src testing unit_tests`
- `vulture src testing unit_tests`
- optional `pyflakes src testing unit_tests`

Fail-fast scenarios that must be tested:

1. Runtime preflight: missing VULCAN source tree/runtime files.
2. Runtime preflight: VULCAN command non-zero exit at preflight.
3. Generation: missing target species list.
4. Generation: missing sampler config or invalid ranges.
5. Generation: one worker run crash.
6. Raw schema: missing HDF5 key.
7. Raw schema: shape mismatch.
8. Raw schema: NaN/Inf encountered.
9. Split integrity: run id appears in multiple splits.
10. Precision: invalid AMP/BF16 combination.
11. Loader: GPU preload OOM with default policy.
12. Training: all-padding or invalid-mask batch.

Each scenario must raise explicit actionable errors.

## 16. Staged Roadmap

### v1 (this spec)
- Thermochemistry trajectory surrogate with broad Kzz + abundance coverage.
- Transport/BC coverage and optional condensation support.
- Photochemistry disabled.

### v2+
- Add photochemistry/ion-chemistry toggles and expanded data generation.
- Preserve same core contracts: explicit config, fail-fast, no hidden fallback.

## 17. Locked Defaults And Assumptions

1. Project root: `.`
2. Code location: all runtime code in `src/`
3. Photochemistry excluded in v1 by default
4. VULCAN default source path: `../VULCAN-master`
5. VULCAN execution path: local-source subprocess execution
6. CLI actions: **only** `--gen` and `--train`
7. `--gen` performs generation + normalization in one command
8. Model family baseline: transformer encoder + FiLM
9. Objective: time-conditioned full trajectory emulation
10. Snapshot policy: `128` log-time snapshots per run
11. Grid defaults: `nz=120`, `P_top=1e-8 bar`, `P_bottom=1e3 bar`
12. Abundance sampling: metallicity + C/O enabled
13. Split policy: run-level only, `70/15/15`
14. Raw format: HDF5
15. Processed format: NPY shards
16. Target species list is required and explicit (top-20 editable config)
17. Precision default: FP32, AMP/BF16 only by explicit valid config
18. Quality gate: report metrics, no hard pass/fail threshold in v1
19. All logarithms are base-10 only
20. Engineering style: clean, correct, pythonic, fail-fast, no unexpected fallback behavior
