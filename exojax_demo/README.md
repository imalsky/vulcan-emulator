# FastChem Standalone Demo

Self-contained FastChem equilibrium-chemistry demo for the VULCAN emulator.
Copy this folder anywhere and run it — no installation from this repository required.

The `.npz` bundle is fully self-contained: weights, normalization statistics,
architecture config, and the inference module source code are all embedded
inside `bundle/best_exported.npz`. New bundles produced by the training
pipeline automatically embed the inference code at export time.

The local `vulcan_emulator.py` file is a thin wrapper around that embedded
implementation, so the demo scripts can use ordinary imports instead of
repeating dynamic loading boilerplate.

## Quick start

```bash
cd exojax_demo

# 1. Minimal direct bundle usage (native bundle ordering)
python 01_standalone_fastchem.py

# 2. ExoJAX-facing wrapper (JIT, VJP, gradients)
python 02_exojax_fastchem.py

# 3. Compare against bundled ground-truth test profiles
python 03_compare_test_profiles.py
```

## Requirements

- Python >= 3.9
- JAX >= 0.4
- NumPy >= 1.24
- Matplotlib >= 3.7  (`03_compare_test_profiles.py` only)

## Included files

| File | Description |
|---|---|
| `README.md` | This file |
| `vulcan_emulator.py` | Thin wrapper exposing the embedded inference API as a normal module |
| `01_standalone_fastchem.py` | Minimal direct inference example |
| `02_exojax_fastchem.py` | ExoJAX wrapper example with `jax.jit`, `jax.grad`, `jax.vjp` |
| `03_compare_test_profiles.py` | Saves comparison plots to `plots/` |
| `API_REFERENCE.md` | FastChem interface reference and design notes |
| `bundle/best_exported.npz` | Trained FastChem transformer (~14 MB); includes embedded inference code |
| `bundle/test_profiles.npz` | 3 ground-truth test cases in physical units (~15 KB) |

## How the inference module is loaded

`vulcan_emulator.py` loads the embedded inference source from the bundle once
and re-exports the normal FastChem entrypoints. The scripts import that local
module directly.

## Main entrypoints

- `load_model("bundle/best_exported.npz")` → `ExportedModel`
- `model.predict_fastchem(pressure_bar, temperature_k, global_inputs)`
- `make_fastchem_vmr_fn(model)` → `(vmr_fn, species_labels)`

Use `model.predict_fastchem(...)` for direct standalone inference.
Use `make_fastchem_vmr_fn(model)` for the ExoJAX top-to-bottom API
compatible with `jax.jit`, `jax.grad`, `jax.vjp`, and `jax.vmap`.

## Bundled test data

`bundle/test_profiles.npz` contains 3 ground-truth FastChem profiles:
- pressure grids (50 levels, 100 → 1e-7 bar)
- temperature profiles
- true mixing ratios for all 17 species
- elemental abundances for each case

Running `03_compare_test_profiles.py` creates a `plots/` directory and writes
one comparison figure per test case.

See `API_REFERENCE.md` for the full interface and ordering conventions.
