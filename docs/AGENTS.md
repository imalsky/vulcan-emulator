# VULCAN emulator — developer notes for Codex

## Conda environment

All Python in this repo — training, export, and the `exojax_demo/`
notebooks — runs in the **`vulcan`** conda env. When testing code changes,
sanity-checking bundles, or running notebooks, activate it first:

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate vulcan
```

The env has ExoJAX, NumPyro, PyTorch, and JAX with 64-bit support. Do not
install into `base`.

## Quick bundle sanity check

The NPZ bundle holds only weights + metadata; the forward pass lives in
`src/models/transformer.py` and is loaded via `src/models/standalone_inference.py`,
so the `vulcan-emulator` package must be on `sys.path`. No bundle is checked
into the repo — produce one first with
`python -m src.utils --config config/<cfg>.json --stage export`, which writes
to `models/<run_name>/best_exported.npz`. Then:

```bash
python -c "
from src.models.standalone_inference import load_model
b = load_model('models/<run_name>/best_exported.npz')
print(b.chemistry_type, b.data_contract['global_static_feature_order'], b.fixed_globals)
"
```

A healthy FastChem bundle prints:

```
fastchem ['He_H', 'C_H', 'O_H', 'N_H', 'S_H'] {}
```

`fixed_globals` must be `{}`; anything else means a global input was held
constant in training and will break gradient-based sampling.
