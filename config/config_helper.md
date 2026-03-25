# Config reference

The shipped config is:

- `config.json`: VULCAN-oriented photochemical configuration for
  WASP-39b-like use with the Frances stellar spectrum and the sulfur-enabled 2025
  SNCHO network

Important sections:

## `data_spec`

- `state_species`: ordered input/output chemistry basis
- `output_species`: usually the same as `state_species`
- `required_global_inputs`: ordered list used to build the global conditioning vector

## `physics_toggles`

Boolean toggles that are both persisted into raw runs and injected into the surrogate
conditioning vector. `use_photochemistry` is expected to be `true` for the main use case.

## `vulcan_runtime`

Settings for an external VULCAN checkout:

- `python_executable`
- `cfg_file`
- `chemistry_file`
- `worker_root`
- `regenerate_chem_funs`
- `atm_base`
- `t_cross_sp`
- `cfg_assignments`

The VULCAN-backed workflow is strict: if the configured checkout or chemistry file is
missing, generation fails instead of silently falling back to synthetic data. The
configured stellar spectrum file is also required; no analytic spectrum fallback is used.
`t_cross_sp` should only include species supported by the target VULCAN checkout.

For smoke tests or local debugging, switch `generation.mode` from `"vulcan"` to
`"synthetic"` in the same config.

## `sampling`

Controls the pressure grid, T profile, Kzz profile, time sampling, and the scalar
parameter space that the raw-run generator covers. The shipped WASP-39b config uses a
Latin-hypercube design over gravity, metallicity, and C/O so the sampled dataset spans
the configured envelope instead of clustering randomly.

For now Kzz is configured as a single constant profile value:

- `kzz_cm2_s`: positive scalar eddy-diffusion coefficient applied at every pressure level

## `stellar_spectrum`

Controls the fixed spectrum grid and the internal spectrum encoder:

- `template_file`
- `num_bins`
- `wavelength_min_nm`
- `wavelength_max_nm`
- `encoder_mode`: `"autoencoder"`, `"linear"`, or `"none"`
- `latent_dim`
- `hidden_dim`
- `zenith_angle_deg`
- `diurnal_factor`

`template_file` should point to a VULCAN-format stellar surface-flux file. The default
WASP-39b config uses `../VULCAN-master/atm/stellar_flux/sflux-wasp39-frances.txt`.

## `generation`

- `mode`: `"synthetic"` or `"vulcan"` source generation
- `target_mode`: `"equilibrium_only"` or `"trajectory"` supervision target
- `num_runs`
- `overwrite`
- `reuse_raw_if_present`
- `parallel_workers`

Generation writes `generation_manifest.json` and `sampling_coverage.json` under
`paths.raw_root` so the realized parameter-space coverage is auditable.

`target_mode = "equilibrium_only"` keeps the same surrogate architecture and public API,
but trains on a two-step shell from a flat H2/He anchor to the exact FastChem
equilibrium profile. When `generation.mode = "vulcan"`, this path skips `vulcan.py`
entirely, copies only the FastChem runtime subset, and runs FastChem directly on the
sampled P-T profile and elemental abundances. `target_mode = "trajectory"` keeps the
full transition-learning path and prepends the exact FastChem state at `t = 0`.

## `trajectory_sampling`

Defines valid `(anchor, target)` transitions using realized dt and minimum saved-step gaps.

- `dt_min_s`
- `dt_max_s`
- `min_future_saved_steps`
- `num_logdt_bins`

Train and eval row selection both use weighted, stratified sampling across these
`log10_dt_s` bins.

## `inference.equilibrium_anchor`

Controls the anchor state used by `PhysicalSpaceStandaloneModel.equilibrium()`.

- `source`: `"trajectory"` or `"flat"`
- `split`: processed split to read when `source = "trajectory"`
- `run_id`: optional processed run ID; defaults to the first run in the chosen split
- `step_index`: saved-step index; defaults to `0`, so equilibrium inference grabs the
  first trajectory profile and otherwise snaps to the closest valid saved step in that run

When `generation.target_mode = "equilibrium_only"`, the default anchor source is
`"flat"`. When `generation.target_mode = "trajectory"`, the default source is
`"trajectory"`.

If the processed split is unavailable at inference time, `"trajectory"` falls back to the
flat H2/He anchor.

## `training.live_sampling`

- `train_pairs_per_run_per_epoch`
- `eval_pairs_per_run`

These affect training and evaluation behavior but do not change the processed tensors.
