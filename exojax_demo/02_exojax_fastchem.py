#!/usr/bin/env python3
"""ExoJAX integration example: FastChem emulator with JIT, VJP, and gradients.

Demonstrates the ExoJAX-facing API:
  1. Create a pure-JAX vmr_fn via make_fastchem_vmr_fn()
  2. Run inference with top-to-bottom level ordering
  3. JIT-compile for speed
  4. Compute gradients (jax.grad) and VJPs (jax.vjp)

This is what an ExoJAX retrieval would use under the hood.

Usage
-----
    python 02_exojax_fastchem.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from vulcan_emulator import (
    BUNDLE_PATH,
    FASTCHEM_GLOBAL_ORDER,
    SOLAR_ABUNDANCES,
    load_model,
    make_fastchem_vmr_fn,
)

NUM_LEVELS = 50
PRESSURE_TOP_BAR = 1.0e-7
PRESSURE_BOTTOM_BAR = 1.0e2
ISOTHERMAL_TEMPERATURE_K = 1500.0
PHOTOSPHERE_PRESSURE_BAR = 0.1
H2O_SPECIES = "H2O"
JIT_RTOL = 1.0e-5
JIT_ATOL = 1.0e-7

# ===================================================================
# Step 1 -- Load the bundle and create the ExoJAX-facing callable
# ===================================================================
bundle = load_model(BUNDLE_PATH)
vmr_fn, species_labels = make_fastchem_vmr_fn(bundle)
print(f"Loaded: {bundle.chemistry_type} {bundle.model_type}  |  vmr_fn supports jit/grad/vjp/vmap")

# ===================================================================
# Step 2 -- Define inputs in ExoJAX convention (top-to-bottom)
# ===================================================================
# ExoJAX uses top-to-bottom ordering: index 0 = top of atmosphere.
# 50 levels matching the training pressure domain (1e-7 → 100 bar).
pressures_bar = jnp.logspace(
    jnp.log10(jnp.asarray(PRESSURE_TOP_BAR)),
    jnp.log10(jnp.asarray(PRESSURE_BOTTOM_BAR)),
    NUM_LEVELS,
)    # top (1e-7 bar) to bottom (100 bar)
temperatures_k = jnp.full(NUM_LEVELS, ISOTHERMAL_TEMPERATURE_K)
global_inputs = dict(SOLAR_ABUNDANCES)

vmr = vmr_fn(temperatures_k, pressures_bar, global_inputs)
idx = int(jnp.argmin(jnp.abs(pressures_bar - PHOTOSPHERE_PRESSURE_BAR)))
print(f"Eager inference: shape={vmr.shape}  H2O@{PHOTOSPHERE_PRESSURE_BAR:.1f}bar={float(vmr[idx, species_labels.index(H2O_SPECIES)]):.3e}")

# ===================================================================
# JIT compilation -- much faster for repeated calls (e.g. MCMC)
# ===================================================================
global_array = jnp.array([global_inputs[name] for name in FASTCHEM_GLOBAL_ORDER])

vmr_jit = jax.jit(vmr_fn)

# First call includes compilation time.
t0 = time.perf_counter()
vmr_compiled = vmr_jit(temperatures_k, pressures_bar, global_array)
jax.block_until_ready(vmr_compiled)
t_compile = time.perf_counter() - t0

# Second call is pure execution.
t0 = time.perf_counter()
vmr_compiled = vmr_jit(temperatures_k, pressures_bar, global_array)
jax.block_until_ready(vmr_compiled)
t_exec = time.perf_counter() - t0

print(f"JIT: compile={t_compile*1000:.0f} ms,  subsequent calls={t_exec*1000:.1f} ms")
np.testing.assert_allclose(
    np.asarray(vmr),
    np.asarray(vmr_compiled),
    rtol=JIT_RTOL,
    atol=JIT_ATOL,
)

# ===================================================================
# Gradients -- jax.grad and jax.vjp
# ===================================================================

def total_vmr_sum(temps: jax.Array) -> jax.Array:
    """Scalar loss: sum of all predicted mixing ratios."""
    return jnp.sum(vmr_fn(temps, pressures_bar, global_array))

grad_fn = jax.jit(jax.grad(total_vmr_sum))
grad_T = grad_fn(temperatures_k)
print(f"jax.grad  d(ΣV)/dT: shape={grad_T.shape}, max|grad|={float(jnp.max(jnp.abs(grad_T))):.3e}, finite={bool(jnp.all(jnp.isfinite(grad_T)))}")

primals = (temperatures_k, pressures_bar, global_array)
vmr_val, vjp_fn = jax.vjp(vmr_fn, *primals)

# Cotangent: ones everywhere (equivalent to gradient of sum).
cotangent = jnp.ones_like(vmr_val)
g_T, g_P, g_global = vjp_fn(cotangent)
abundance_names = FASTCHEM_GLOBAL_ORDER
print(f"jax.vjp:  dT finite={bool(jnp.all(jnp.isfinite(g_T)))}, dP finite={bool(jnp.all(jnp.isfinite(g_P)))}, d(abundances) finite={bool(jnp.all(jnp.isfinite(g_global)))}")

h2o_idx = species_labels.index(H2O_SPECIES)
level_idx = int(jnp.argmin(jnp.abs(pressures_bar - PHOTOSPHERE_PRESSURE_BAR)))

def h2o_at_photosphere(abundances: jax.Array) -> jax.Array:
    """Predict H2O mixing ratio near the photosphere."""
    vmr = vmr_fn(temperatures_k, pressures_bar, abundances)
    return vmr[level_idx, h2o_idx]

grad_h2o = jax.grad(h2o_at_photosphere)(global_array)
print(
    f"d(H2O@{PHOTOSPHERE_PRESSURE_BAR:.1f}bar)/d(abundances): "
    + ", ".join(f"{n}={v:+.3e}" for n, v in zip(abundance_names, np.asarray(grad_h2o)))
)
print("\nAll checks passed — vmr_fn is JIT-compatible and fully differentiable.")
