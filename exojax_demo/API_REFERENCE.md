# FastChem Standalone API Reference

## 1. Load the exported bundle

```python
from vulcan_emulator import load_model, make_fastchem_vmr_fn

model = load_model("bundle/best_exported.npz")
```

This folder ships a standalone wrapper module, `vulcan_emulator.py`, so no
imports from the main repository are required.

## 2. Direct bundle API

```python
vmr = model.predict_fastchem(
    pressure_bar,       # array-like, shape (nz,)
    temperature_k,      # array-like, shape (nz,)
    global_inputs,      # dict[str, float] or array-like, shape (5,)
    return_log10=False,
)
```

Returns a `jax.Array` with shape `(nz, 17)`.

This is the direct exported-bundle interface. It uses the bundle's native
profile ordering, which is the same ordering used in `bundle/test_profiles.npz`
(bottom-to-top, high pressure to low pressure).

## 3. ExoJAX-facing API

```python
vmr_fn, species_labels = make_fastchem_vmr_fn(model)

vmr = vmr_fn(
    temperatures_k,       # jax.Array, shape (nz,), top-to-bottom
    pressures_bar,        # jax.Array, shape (nz,), top-to-bottom
    global_inputs,        # dict or jax.Array, elemental abundances
    gravity_cm_s2=None,   # optional, accepted but unused
)
```

Returns a pure-JAX callable that is compatible with:

- `jax.jit()`
- `jax.grad()`
- `jax.vjp()`
- `jax.jvp()`
- `jax.vmap()`

This wrapper is the ExoJAX-safe interface. It accepts and returns
**top-to-bottom** profiles and handles the internal reversal automatically.

## 4. Global inputs

Elemental abundances are absolute number ratios relative to hydrogen (X/H),
not relative to solar.

Required array order:

```python
["He_H", "C_H", "O_H", "N_H", "S_H"]
```

Approximate solar values:

| Key | Element | Solar value |
|---|---|---|
| `He_H` | Helium | 8.38e-2 |
| `C_H` | Carbon | 2.95e-4 |
| `O_H` | Oxygen | 5.37e-4 |
| `N_H` | Nitrogen | 7.08e-5 |
| `S_H` | Sulfur | 1.41e-5 |

Inputs can be passed as a `dict[str, float]` or as an array in the exact order
above. Use an array if you want the abundances to stay fully differentiable in
JAX.

If you work in `[M/H]` and `C/O`, a simple conversion is:

```python
def solar_to_xh(metallicity_log10, c_to_o, s_to_o=0.025):
    """Convert [M/H] and C/O to absolute X/H abundances."""
    solar_O_H = 5.37e-4
    met_scale = 10.0 ** metallicity_log10
    O_H = solar_O_H * met_scale
    C_H = c_to_o * O_H
    N_H = 7.08e-5 * met_scale
    S_H = s_to_o * O_H
    He_H = 8.38e-2
    return {"He_H": He_H, "C_H": C_H, "O_H": O_H, "N_H": N_H, "S_H": S_H}
```

## 5. Output species

The output species order is fixed:

```text
H2, He, H, O, OH, H2O, CO, CO2, CH4, N2, NH3, H2S, SH, S, SO, SO2, S2
```

Use `model.species` or the `species_labels` returned by
`make_fastchem_vmr_fn(model)` to retrieve the same ordering programmatically.

## 6. Answers to Hajime's questions

### Q: Is any important parameter missing?

No. For FastChem equilibrium chemistry, temperature, pressure, and elemental
abundances fully determine the composition.

### Q: How should abundances be represented?

Use absolute X/H number ratios, not values relative to solar.

### Q: Do we need gravity?

Not for FastChem. The `gravity_cm_s2` argument exists only for API
compatibility with ExoJAX-facing call sites and is not consumed by the model.

### Q: Do we need layer thickness or vertical scale?

No. The emulator works on pressure levels directly and does not require
altitude, layer thickness, or geometric layer spacing.

### Q: Is VJP supported?

Yes. The ExoJAX-facing wrapper returned by `make_fastchem_vmr_fn(model)` is a
pure JAX function and supports `jax.jit`, `jax.grad`, `jax.vjp`, `jax.jvp`,
and `jax.vmap`.

### Q: What is the level ordering convention?

- `model.predict_fastchem(...)`: direct bundle-native ordering
- `make_fastchem_vmr_fn(model)`: top-to-bottom ExoJAX ordering

The shipped examples follow the same split:

- `01_standalone_fastchem.py` and `03_compare_test_profiles.py`: native bundle ordering
- `02_exojax_fastchem.py`: top-to-bottom ExoJAX ordering
