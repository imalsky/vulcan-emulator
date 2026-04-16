# VULCAN emulator — developer notes for Claude

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

```bash
cd exojax_demo
python -c "
import sys, types, numpy as np
with np.load('best_exported.npz', allow_pickle=False) as f:
    src = bytes(f['meta/vulcan_emulator_src']).decode()
m = types.ModuleType('_e'); sys.modules['_e'] = m
exec(compile(src, '<e>', 'exec'), m.__dict__)
b = m.load_model('best_exported.npz')
print(b.chemistry_type, b.data_contract['global_static_feature_order'], b.fixed_globals)
"
```

A healthy FastChem bundle prints:

```
fastchem ['He_H', 'C_H', 'O_H', 'N_H', 'S_H'] {}
```

`fixed_globals` must be `{}`; anything else means a global input was held
constant in training and will break gradient-based sampling.
