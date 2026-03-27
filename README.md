# vulcan_emulator_photochem

JAX-first VULCAN emulator pipeline with one shared data workflow and two explicit tasks:

- `full_vulcan`: trajectory emulation with `dt`, stellar-spectrum conditioning, and a Transformer.
- `equilibrium_only`: direct equilibrium prediction with an MLP.

Both tasks share the same generation, normalization, split, and artifact conventions. The
active task is selected by `task.kind` in the config.

## Shipped Config

- `config/equilibrium_only_config.json`

## CLI

The CLI accepts exactly two parser arguments:

```bash
python -m src.utils --config config/equilibrium_only_config.json --stage generation
python -m src.utils --config config/equilibrium_only_config.json --stage normalization
python -m src.utils --config config/equilibrium_only_config.json --stage training
```

`--stage normalization` performs the full raw-to-processed step: split creation,
train-only normalization fitting, and processed tensor writing.

## Layout

- `assets/`: immutable external inputs such as Roth/PT libraries and stellar spectra templates.
- `config/`: canonical JSON configs and schema notes.
- `data/`: generated raw runs, processed tensors, manifests, and derived spectrum libraries.
- `models/`: checkpoints and exported model bundles.
- `src/models/`: model architecture, inference, and export logic.
- `src/training/`: training loops and live transition sampling.
- `src/data_generation/`: PT/spectrum loading, raw generation, preprocessing, and dataset I/O.
- `src/utils/`: config validation, CLI, logging, paths, and provenance helpers.
- `uni_tests/fixtures/`: tiny tracked fixtures used by tests.

## Temperature-Profile Handling

`temperature_profiles` controls how the pipeline samples TP inputs. For PT-library `.dat`
files, each `(lon, lat)` column is treated as a valid 1D starting profile. The profile is
interpolated onto the emulator/VULCAN pressure grid in log-pressure space, and its source
metadata is preserved through generation artifacts. The analytic branch uses the exposed
`temperature_profiles.analytic_sampler` Line/Robinson-style PT parameterization rather than
the older placeholder logistic profile.

## Asset Policy

Production PT libraries and spectra belong under `assets/` and are intentionally not tracked
in git. Tests use small synthetic fixtures under `uni_tests/fixtures/`.
