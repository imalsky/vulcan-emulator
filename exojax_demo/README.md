# ExoJAX demo for the VULCAN FastChem emulator

Self-contained demo showing how to call the exported FastChem emulator
(`best_exported.npz`) from JAX/ExoJAX. Every notebook loads the emulator
via the `standalone_inference.py` source embedded in the NPZ — no project
imports required.

## Contents

| File | Purpose |
| --- | --- |
| `best_exported.npz` | Exported emulator bundle. Copy of `../models/fastchem_no_condensation/best_exported.npz`. |
| `equilibrium_chemistry_transformer.ipynb` | End-to-end ExoJAX retrieval tutorial: load the emulator, build opacities from ExoMol/CIA, run a forward model, and sample with NumPyro NUTS. |
| `exogibbs_vs_emulator.ipynb` | Species-by-species comparison of the emulator against ExoGibbs (native JAX equilibrium chemistry) across several P-T profiles. |
| `gradient_verification.ipynb` | Minimal standalone check that `jax.grad` / `jacfwd` / `jacrev` return finite, consistent values through the emulator — used to certify the HMC-NUTS gradient path. |
| `.database/` | ExoJAX opacity cache (ExoMol CO line list, H2-H2 CIA). Created on first run; kept here so the notebooks do not re-download. |

## Running

Notebooks assume the working directory is this folder, so the relative
paths (`best_exported.npz`, `.database/...`) resolve correctly.

Recommended reading order: `gradient_verification` →
`equilibrium_chemistry_transformer` → `exogibbs_vs_emulator`.

## Refreshing the bundle

When a new emulator is trained:

```bash
cp ../models/fastchem_no_condensation/best_exported.npz ./best_exported.npz
```

The embedded `standalone_inference.py` travels inside the NPZ, so no other
files need to be updated.
