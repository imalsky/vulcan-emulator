# vulcan_emulator

JAX-first chemistry emulator pipeline with one shared workflow and a configurable `chemistry_type × model_type` surface.

Supported chemistry targets:
- `fastchem`: FastChem equilibrium chemistry from PT and profile-global `X/H`
- `vulcan`: final converged VULCAN chemistry from PT, Kzz, surface gravity, planet radius, profile-global `X/H`, runtime science knobs, atmosphere-base flags, and stellar spectrum
- `exogibbs`: ExoGibbs Gibbs free energy equilibrium chemistry (same input contract as FastChem, JAX-native solver)

Supported model families:
- `transformer`

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

`--stage normalization` performs the full raw-to-processed step: split creation, train-only normalization fitting, and processed tensor writing.

Chemistry contract:
- generation still samples chemistry in `[M/H]`, `C/O`, and `S/O`
- preprocessing converts those draws into fixed FastChem/VULCAN element order:

```python
["He_H", "C_H", "O_H", "N_H", "S_H"]
```

- these are hydrogen-normalized absolute abundances `n_X / n_H`

VULCAN-side shipped defaults (in `config/vulcan_luhman16a_10k.json`):
- single default science preset (no `science_presets` block needed)
- photochemistry is off (and required to be off — not yet supported)
- condensation is on for `H2O` and `S8` via the typed
  `vulcan.runtime.condensation` block (no longer hand-rolled through
  `cfg_assignments`)
- settling is on; particle radii / densities for `H2O_l_s` and `S8_l_s`
  live under `vulcan.runtime.condensation.{r_p, rho_p}`
- initial cold trap is on
- eddy diffusion is on
- `vulcan.runtime.chemistry_file` is `thermo/SNCHO_photo_network_2025.txt`
- `vulcan.runtime.regenerate_chem_funs` is enabled so worker-local runs rebuild `chem_funs.py` with photochemistry disabled
- `vulcan.stellar_spectrum` is required because VULCAN reads `sflux_file` unconditionally at startup (even with `use_photochemistry = false`); the pipeline generates a blackbody template and writes it for each worker run
- when every science preset has `use_photochemistry = false`, `stellar_spectrum.template_file` may be omitted and the pipeline will synthesize the default WASP-39 template internally
- Kzz is depth-constant; the per-run value is log-sampled from `sampling.kzz_range_cm2_s`
- VULCAN surface gravity is sampled from `sampling.gravity_range_cm_s2`
- VULCAN planet radius is sampled from `sampling.planet_radius_range_cm`
- `vulcan.runtime.rocky` defaults to `false`
- `generation.vulcan_timeout_seconds` (default `1800`) caps stalled VULCAN runs; timeouts route through the same backfill path as other failures

Data layout:
- each config uses a single dataset root under `data/<run_name>/`
- raw files live in `data/<run_name>/raw`
- shared metadata lives in `data/<run_name>/info`
- processed split tensors live in `data/<run_name>/processed/train`, `data/<run_name>/processed/val`, and `data/<run_name>/processed/test`
- training artifacts stay under `models/`

Contract notes:
- raw VULCAN runs store the layerwise gravity profile in `inputs/gravity_cm_s2`
- learned VULCAN globals use scalar surface gravity plus scalar `planet_radius_cm`
- FastChem does not consume gravity or Kzz

Layout:
- `assets/`: external PT libraries and stellar spectra
- `config/`: canonical JSON configs
- `data/`: dataset runs with `raw/`, `info/`, `processed/{train,val,test}/`, and derived spectrum libraries
- `docs/`: human-facing docs (`config_guide.md`, `training_diary.md`)
- `models/`: checkpoints and exported bundles
- `src/constants.py`: single source of truth for all shared constants
- `src/models/`: architecture, inference, and export logic
- `src/training/`: training loops and evaluation utilities
- `src/data_generation/`: sampling, raw generation, preprocessing, and dataset I/O
- `src/utils/`: config validation, CLI, logging, paths, and provenance helpers
- `supercomputer_cmds/`: HPC submission scripts — submit from anywhere via `qsub supercomputer_cmds/run.pbs` or `sbatch supercomputer_cmds/run_gen.sh`
- `uni_tests/`: minimal test suite (4 files — config, model, export, sampling)

## Inference surface

- `ExportedJAXModel` in `src/models/export_bundle.py` is the single runtime
  class for loaded bundles. It exposes both the canonical
  `predict_fastchem_profile` / `predict_vulcan_profile` (keyword-only,
  profile-level) methods and short positional aliases `predict_fastchem` /
  `predict_vulcan` / `make_compiled_fastchem_predictor` for callers that
  prefer the older names.
- `src/models/standalone_inference.py` is a thin backward-compat shim that
  re-exports `load_model`, `ExportedModel`, and the ExoJAX wrapper
  factories. New code should import from `export_bundle` or `exojax_api`
  directly.
- The ExoJAX-facing factories `make_fastchem_vmr_fn` / `make_vulcan_vmr_fn`
  live in `src/models/exojax_api.py`.

## Running the demos

The notebooks under `exojax_demo/` and `extras/` expect an exported bundle
at `models/<run_name>/best_exported.npz`. No bundle is checked in; produce
one first with `python -m src.utils --config config/<cfg>.json --stage export`
and point the notebook's `BUNDLE_PATH` at the result.
