# vulcan_emulator_photochem

JAX-first VULCAN emulator pipeline with one shared workflow and a configurable `chemistry_type × model_type` surface.

Supported chemistry targets:
- `fastchem`: FastChem equilibrium chemistry from PT and profile-global `X/H`
- `vulcan`: final converged VULCAN chemistry from PT, Kzz, surface gravity, planet radius, profile-global `X/H`, runtime science knobs, atmosphere-base flags, and stellar spectrum

Supported model families:
- `mlp`
- `transformer`

Shipped example configs:
- `config/fastchem_transformer_config.json`
- `config/vulcan_transformer_config.json`

CLI:

```bash
python -m src.utils --config config/fastchem_transformer_config.json --stage generation
python -m src.utils --config config/fastchem_transformer_config.json --stage normalization
python -m src.utils --config config/fastchem_transformer_config.json --stage training
```

`--stage normalization` performs the full raw-to-processed step: split creation, train-only normalization fitting, and processed tensor writing.

Chemistry contract:
- generation still samples chemistry in `[M/H]`, `C/O`, and `S/O`
- preprocessing converts those draws into fixed FastChem/VULCAN element order:

```python
["He_H", "C_H", "O_H", "N_H", "S_H"]
```

- these are hydrogen-normalized absolute abundances `n_X / n_H`

Shipped defaults:
- the VULCAN example config uses one `basic_h2` preset
- photochemistry is off
- condensation is on for `H2O` and `S8`
- eddy diffusion is on
- `vulcan.runtime.chemistry_file` is `thermo/SNCHO_photo_network_2025.txt`
- `vulcan.runtime.regenerate_chem_funs` is enabled so worker-local runs rebuild `chem_funs.py` with photochemistry disabled
- `vulcan.stellar_spectrum` is still part of the VULCAN run contract because the pipeline stores a spectrum and irradiation geometry for every run
- when every science preset has `use_photochemistry = false`, `stellar_spectrum.template_file` may be omitted and the pipeline will synthesize the default WASP-39 template internally
- Kzz is depth-constant from `sampling.kzz_cm2_s`
- VULCAN surface gravity is sampled from `sampling.gravity_range_cm_s2`
- VULCAN planet radius is sampled from `sampling.planet_radius_range_cm`
- `vulcan.runtime.rocky` defaults to `false`

Data layout:
- each config now uses a single dataset root under `data/<run_name>/`
- raw files live in `data/<run_name>/raw`
- shared metadata lives in `data/<run_name>/info`
- processed split tensors live directly in `data/<run_name>/train`, `data/<run_name>/val`, and `data/<run_name>/test`
- training artifacts stay under `models/`

Contract notes:
- raw VULCAN runs store the layerwise gravity profile in `inputs/gravity_cm_s2`
- learned VULCAN globals use scalar surface gravity plus scalar `planet_radius_cm`
- FastChem does not consume gravity or Kzz

Layout:
- `assets/`: external PT libraries and stellar spectra
- `config/`: canonical JSON configs and helper notes
- `data/`: dataset runs with `raw/`, `info/`, `train/`, `val/`, `test/`, and derived spectrum libraries
- `models/`: checkpoints and exported bundles
- `src/models/`: architecture, inference, and export logic
- `src/training/`: training loops and evaluation utilities
- `src/data_generation/`: sampling, raw generation, preprocessing, and dataset I/O
- `src/utils/`: config validation, CLI, logging, paths, and provenance helpers
- `uni_tests/fixtures/`: tiny tracked fixtures used by tests
