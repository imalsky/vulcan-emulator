# vulcan_emulator_photochem

A clean JAX-first rewrite of the `imalsky/vulcan-emulator` training pipeline aimed at
photochemical VULCAN trajectories with a compact sulfur extension and explicit stellar
spectrum conditioning for WASP-39b-like cases.

The shipped model is designed around four requirements:

1. photochemistry is a first-class input rather than a disabled validation toggle;
2. the state vector includes the sulfur carriers and oxidation radicals needed to keep
   sulfur photochemistry closer to Markovian (`H`, `O`, `OH`, `H2S`, `SH`, `S`, `SO`,
   `SO2`, `S2`);
3. a fixed-grid stellar spectrum is part of the conditioning signal, with the shipped
   WASP-39b config reading the Frances stellar surface-flux file used by VULCAN; and
4. the exported transition operator is pure JAX and supports both `jax.grad` and
   `jax.jvp`, so it can be embedded in ExoJAX or any other JAX workflow.

The codebase supports two generation paths:

- `generation.mode = "vulcan"` runs an external VULCAN checkout, patches `vulcan_cfg.py`,
  writes atmosphere and stellar-flux inputs, regenerates `chem_funs.py` when requested,
  and converts the `.vul` pickle output into the raw HDF5 contract expected by preprocessing.
- `generation.mode = "synthetic"` creates a deterministic sulfur-aware photochemical toy
  dataset. This exists only so the repository is testable without running a full VULCAN
  job; the production path is the strict VULCAN-backed workflow.

## Layout

- `config/`: the default project config and a short schema guide.
- `src/`: JAX model, preprocessing, data loading, VULCAN runner, export, and CLI.
- `uni_tests/`: smoke tests for config validation, spectrum handling, synthetic generation,
  preprocessing, JAX autodiff, export, and inference round-trips.

## Typical local workflow

```bash
python -m src.main --config config/config.json gen
python -m src.main --config config/config.json preprocess
python -m src.main --config config/config.json train
```

For a local smoke run, keep the same config and temporarily set `generation.mode` to
`"synthetic"`. For a real VULCAN-backed run, point `paths.vulcan_source_root` at a local
VULCAN checkout.

## Important notes

This repository does **not** bundle VULCAN itself. It interfaces with an external checkout.
The default WASP-39b config expects the Frances stellar surface-flux file already present
in the adjacent `VULCAN-master` tree and writes a `sampling_coverage.json` report for each
generated dataset so the realized parameter-space coverage can be audited.
The shipped config requires `stellar_spectrum.template_file` to exist; there is no
analytic spectrum fallback.

Transition selection is now sampled with weighted, stratified `log10_dt_s` bins for both
training and evaluation. Eval remains reproducible through a fixed RNG seed, but it no
longer overweights the shortest transitions by taking the first sorted rows.

Processed datasets now store the normalized anchor-state trajectory and the normalized
target-species trajectory separately. That keeps the contract correct when
`output_species` is a strict subset or reordering of `state_species`.

The training and inference code intentionally use only JAX, NumPy, and h5py.
