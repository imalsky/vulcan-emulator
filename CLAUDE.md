# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Everything Python in this repo — training, export, and the example notebooks shipped in the standalone `For_Hajime/emulator_tests/` deliverable — runs in the **`vulcan`** conda env. Activate before doing anything:

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate vulcan
```

The env carries ExoJAX + NumPyro + JAX with 64-bit support. Do not install into `base`. On HPC the equivalent env name is set by `CONDA_ENV` in `supercomputer_cmds/run*.sh`.

## Pipeline CLI

The whole project is one four-stage pipeline driven by a single config. Stages must be run in order on a fresh dataset; later ones can be rerun independently if their inputs already exist.

```bash
python -m src.utils --config config/fastchem.json --stage generation
python -m src.utils --config config/fastchem.json --stage normalization   # split + fit + write processed tensors (one stage, not two)
python -m src.utils --config config/fastchem.json --stage training
python -m src.utils --config config/fastchem.json --stage export
```

Each stage prints a JSON artefact summary to stdout. `export` reads `params_best.npz` from `<checkpoints_root>/` and writes `best_exported.npz` next to it. For ad-hoc exports outside the config pipeline, call `from src.models.export_bundle import export_checkpoint_to_npz` directly with a run directory.

Hyperparameter sweeps live under `src/tuning/`:

```bash
python -m src.tuning --config config/fastchem.json --trials 100 --epochs 100
```

## Tests and lint

```bash
pytest                                     # runs uni_tests/ (4 files only, intentionally minimal)
pytest uni_tests/test_model.py             # one file
pytest uni_tests/test_model.py::test_name  # one test
ruff check src uni_tests
```

The four test files cover config validation, JAX model forward/autodiff/training, export bundle round-trip, and atmospheric sampling. Add new tests only when they cover a genuinely distinct failure mode.

## Architecture: the `chemistry_type × model_type` surface

Every config picks two selectors that determine the data contract and the network. Currently shipped: `fastchem|vulcan|exogibbs × transformer`. `chemistry_type` selects which solver the surrogate emulates and therefore which inputs the model consumes; `model_type` selects only the architecture.

Input contracts (sequence + global vectors):

- `fastchem` / `exogibbs`: sequence `[pressure_bar, temperature_k]`, globals `[He_H, C_H, O_H, N_H, S_H]` (5 hydrogen-normalized abundances). ExoGibbs uses the same data contract as FastChem (Gibbs free energy minimizer, same species set).
- `vulcan`: sequence `[pressure_bar, temperature_k, kzz_cm2_s]`, globals expand to 26 entries: gravity, planet radius, the same 5 abundances, irradiation geometry (`r_star_rsun`, `semi_major_axis_au`, `zenith_angle_deg`, `diurnal_factor`), 10 physics toggles, and 5 `atm_base` one-hots.

The fixed elemental order is internal: `["He_H", "C_H", "O_H", "N_H", "S_H"]`. The sampler draws these `X/H` channels directly via Latin-hypercube; there is no `[M/H]` / `C/O` reparameterization in the pipeline. Retrieval-side consumers that prefer a metallicity parameterization should use `src/models/abundance_utils.global_inputs_from_metallicity(...)`.

**Photochemistry is currently off.** The `use_photochemistry` toggle exists in the schema but must be `False` for every science preset.

The transformer is a pre-norm FiLM-conditioned stack: globals go through a conditioning MLP that emits per-layer (γ, β); each block applies attention → FiLM → FFN with optional QK-Norm, RMSNorm, SwiGLU FFN, and zero-init FiLM (AdaLN-Zero) toggled in `model.*`. `src/models/transformer.py` is the single source of truth for the forward pass — the exported bundle is weights + JSON only and **does not** reconstruct the network.

## Variable-grid contract

Per-run vertical level counts vary within `sampling.num_levels_range`. Preprocessing pads every run to `max_num_levels` and writes:

- `valid_mask.npy` `(N, max_num_levels)` — true on real levels, false on padding; gates attention keys.
- `position_coord.npy` `(N, max_num_levels)` — normalized `log10(pressure_bar)` in `[0, 1]` over the union training pressure range; drives the continuous sinusoidal positional encoding.

This is what makes the position encoding physical-coordinate aware (rather than index-aware) and lets the exported bundle accept arbitrary in-range grids at inference time. Both arrays must be threaded through the forward pass — do not bypass them.

## Inference / bundle surface

`src/models/standalone_inference.py` is the **public** import surface for notebooks and external consumers (ExoJAX, retrieval frameworks). It is a thin re-export over the canonical implementations:

- `load_model` → `export_bundle.load_exported_model`
- `ExportedModel` → `export_bundle.ExportedJAXModel` (exposes `predict_fastchem_profile` / `predict_vulcan_profile`, plus short positional aliases)
- `make_fastchem_vmr_fn`, `make_vulcan_vmr_fn` → `exojax_api` (ExoJAX-ready callables)

All of `jax.jit`, `jax.grad`, `jax.jacfwd`, `jax.jacrev`, `jax.jvp`, and `jax.vmap` compose through the predict methods — keep them traceable.

A healthy FastChem bundle prints `fixed_globals == {}`. **Anything else means a global was held constant during training and will break gradient-based sampling** — treat a non-empty `fixed_globals` as a bug, not a feature.

## Data + artefact layout

Each config owns one dataset root:

```
data/<run_name>/
  raw/          per-run HDF5 + chunks/
  info/         normalization.json, data_contract.json, splits.json,
                processed_manifest.json, generation_manifest.json,
                sampling_coverage.json, optional failed_runs.json
  processed/{train,val,test}/   sequence_inputs.npy, target_outputs.npy,
                                global_inputs.npy, valid_mask.npy,
                                position_coord.npy
```

`paths.run_root` is the only user-set dataset path; `raw/`, `info/`, and `processed/` are auto-expanded by `load_and_validate_config()`. Training writes a flat run directory at `paths.checkpoints_root` (default `models/<run_name>/`):

```
models/<run_name>/
  config.json          input config (stripped of runtime annotations)
  history.csv          one row per epoch, ready for pandas/plotting
  metadata.json        model_dimensions, normalization, data_contract, final_metrics
  params_best.npz      flat NPZ of best-epoch weights ({dotted.key: ndarray})
  params_last.npz      same format, last epoch
  best_exported.npz    portable inference bundle (produced by --stage export)
```

**No bundle is checked in** — produce one with `--stage export` before running notebooks.

There is no version tag on bundles or processed data. Cache invalidation is structural: the trainer rebuilds processed tensors whenever the data contract drifts, and stale bundles fail on missing metadata at load time and must be re-exported.

## Single source of truth: `src/constants.py`

All shared constants — physical constants, solar abundances (Asplund 2009 + Lodders 2009 background), species data, chemistry/model enums, config validation allowlists, internal defaults — live in `src/constants.py`. Module-private constants stay in their owning module. Don't introduce parallel constant tables.

## Generation concurrency invariants

`src/data_generation/generation.py` runs workers on a `ThreadPoolExecutor` against subprocess-launched solvers. HDF5 is **not** thread-safe, so the design enforces:

- Each worker writes only to its own per-run file in `runs_dir`. Two threads never share an `h5py.File` handle.
- Per-chunk consolidation runs on a separate single-worker executor (chunk writes are serial).
- Orphan promotion on resume is `fcntl`-locked on a sentinel file in `runs_dir` so two concurrent generation invocations cannot promote the same orphan.
- **Sharded mode (`--shard-id N --num-shards K`)**: each shard owns its own `runs_sNN/` staging dir (node-local under `--staging-root` when set, else under `raw/`), its own `chunks_sNN/` archive on shared FS, its own `worker_base` subtree (`worker_root/shard_sNN/`), and its own `.orphan_promotion_sNN.lock`. The chunk loop iterates only `[shard_start, shard_end)` of a sampling plan built for the **full** target count, preserving seed-byte identity to single-job mode. Backfill IDs live in disjoint per-shard slots of size `_SHARD_BACKFILL_SLOT_SIZE` so all run IDs stay unique. The final consolidated `runs.h5` and `info/generation_manifest.json` are produced exclusively by `--stage merge_shards`, which runs once after every shard succeeds. Each shard writes a per-shard fragment to `info/shards/shard_sNN.json` that the merge stage validates for cross-shard consistency before merging.

Modifications that share an HDF5 writer across workers will silently corrupt the dataset — preserve these invariants when touching this module.

## Classical-reference contract (notebooks in `For_Hajime/emulator_tests/`)

The shipped example notebooks live in the standalone `For_Hajime/emulator_tests/` deliverable (parallel to this repo). They compare the emulator against live FastChem (subprocess) and ExoGibbs (Gibbs minimizer). The training-matched bridge between FastChem's input file and ExoGibbs' 28-element vector is `src/models/classical_reference.py::build_exogibbs_element_vector(..., mode="fastchem_proxy")` — it is the **single source of truth** for that conversion (Lodders refractory background, volatile-weighted [α/H] proxy on the 11 metals, free `He/C/O/N/S` overrides, untracked elements zeroed, mole-fraction normalization). The matching ExoGibbs `ChemicalSetup` must come from `chemsetup_matched_to_fastchem(fastchem_source_root)`, which pins the classical reference to VULCAN-FastChem's shipped `logK_wo_ions.dat`.

Solar anchors must come from `SOLAR_ABUNDANCES` in `src/constants.py` — not from `exojax.utils.zsol.nsol()` AAG21 ratios — or the emulator gets fed a reference point its training distribution wasn't centered on. A residual ~0.1–0.3 dex FC↔EG floor on sulfur polymers (S5/S6/S7) and several O-bearing species is expected and accepted; it is not an emulator defect (NASA-9 vs 5-term-logK polynomial mismatch + species-network differences).

**H₂O is the dominant FC↔EG gap species in retrievals, not sulfur.** Near the CO/H₂O chemical transition (C/O ≈ 0.8), H₂O diverges by ~1.3 dex between FastChem (NASA-9) and ExoGibbs (5-term logK) because H₂O abundance equals total oxygen minus CO-locked oxygen — a small difference in CO thermodynamics amplifies into a large residual H₂O difference. At solar C/O the gap is ~0.19 dex; at C/O = 0.8 it reaches ~1.3 dex. CO itself is negligible (~0.01 dex). The classical-vs-emulator corner plot in `For_Hajime/emulator_tests/07_classical_vs_emulator_retrieval.ipynb` therefore diverges by design — the classical retrieval is fitting a fundamentally different forward model. The apples-to-apples recovery test (emulator retrieval against a live-FastChem mock) is the dedicated emulator-only NUTS path in `For_Hajime/emulator_tests/04_full_nuts_retrieval.ipynb`.

## HPC entry points

PBS and SLURM submission scripts live in `supercomputer_cmds/` and self-locate the project root, so submit from anywhere:

```bash
qsub supercomputer_cmds/run.pbs                                      # full pipeline (default fastchem)
qsub -v CONFIG_PATH=path/to/other.json supercomputer_cmds/run.pbs    # custom config
qsub -v DATA_ONLY=1 supercomputer_cmds/run.pbs                       # generation + normalization only (CPU)
qsub -v SKIP_GEN=1 supercomputer_cmds/run.pbs                        # train + export against existing data
sbatch supercomputer_cmds/run_train.sh                               # SLURM training-only
sbatch supercomputer_cmds/run_gen.sh                                 # SLURM generation-only (single node)
CONFIG_PATH=config/exogibbs_luhman16a.json \
  bash supercomputer_cmds/submit_gen_array.sh                        # SLURM sharded generation (4 nodes) + auto-merge
```

The sharded path (`run_gen_array.sh` + `run_merge.sh`, wired together by `submit_gen_array.sh`) is the right tool for large datasets (ExoGibbs 1M runs, VULCAN condensation); the single-node `run_gen.sh` is fine for FastChem and small runs.

## Reference docs

- `docs/config_guide.md` — key-by-key config reference.
- `docs/training_diary.md` — short log of what each training run taught us about sizing, regularization, and sweep-proxy vs deployment loss. Append a new dated block when finishing a run.
