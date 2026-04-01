"""
standalone_test.py
==================
End-to-end demonstration of the VULCAN equilibrium chemistry emulator.

What this script shows
----------------------
  1. FORWARD PASS   — Build a synthetic hot-Jupiter T-P profile and get
                      equilibrium mixing ratios from the exported JAX model.
                      Physical units in → physical units out; all normalisation
                      is handled internally by the bundle.

  2. FASTCHEM CHECK — Optionally run the same profile through the bundled
                      FastChem executable for an independent ground truth.
                      Requires the VULCAN-master source tree; the script
                      skips this section gracefully if FastChem is absent.

  3. BACKPROP       — Three JAX differentiation demos showing the model is a
                      fully differentiable program:

                      a) jax.grad   — sensitivity of water abundance to T
                      b) jax.jacobian — how every species responds to chemistry globals
                      c) gradient descent — recover a known in-range C/H target

Run from the vulcan-emulator root:

    conda activate nn
    python extras/standalone_test.py

Plots land in extras/standalone_plots/.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make 'src' importable by inserting the project root into the module search
# path.  This mirrors how all other scripts in this repo are structured.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
# Also add extras/ so we can optionally import compare_test_profile_fastchem.
sys.path.insert(0, str(_ROOT / "extras"))

from src.utils.numpy_compat import patch_numpy_asarray_copy

patch_numpy_asarray_copy()

import numpy as np

# JAX is the main computational framework — it compiles, JITs, and
# differentiates the emulator forward pass automatically.
import jax
import jax.numpy as jnp

# All normalisation helpers live in export_bundle.  They are pure JAX
# functions, which is exactly what makes them compatible with jax.grad /
# jax.jacobian / jax.jit without any special wrappers.
from src.models.export_bundle import (
    _apply_sequence_static_block_jax,   # normalise per-level P and T
    _restore_block_transform_space_jax, # undo z-score but keep log10 space
    apply_mixed_block_jax,              # normalise global scalars (per-feature mixed methods)
    load_exported_model,                # load .npz bundle into ExportedJAXModel
)
from src.models.jax_model import apply_mlp  # the FiLM-conditioned MLP
from src.utils.config import ELEMENT_INPUT_ORDER

_EXPECTED_GLOBAL_ORDER = list(ELEMENT_INPUT_ORDER)
_SOLAR_ELEMENT_ABUNDANCES = {
    "O_H": 5.37e-4,
    "C_H": 2.95e-4,
    "N_H": 7.08e-5,
    "S_H": 1.41e-5,
    "He_H": 8.38e-2,
}

# ---------------------------------------------------------------------------
# Paths — adjust BUNDLE_PATH if your model lives in a different directory.
# VULCAN_SOURCE_ROOT only matters for the optional FastChem comparison.
# ---------------------------------------------------------------------------
BUNDLE_PATH        = _ROOT / "models" / "fastchem_mlp" / "best_exported.npz"
VULCAN_SOURCE_ROOT = _ROOT.parent / "VULCAN-master"   # only needed for FastChem
OUTPUT_DIR         = _ROOT / "extras" / "standalone_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional matplotlib style (lives in extras/ alongside this file).
_MPLSTYLE = _ROOT / "extras" / "science.mplstyle"


# ============================================================================
# SECTION 1 — Load the exported model bundle
# ============================================================================

print("=" * 72)
print("SECTION 1 — Loading exported JAX model bundle")
print("=" * 72)

# The exported .npz bundle is a self-contained inference artefact.
# It holds:
#   params/...   — all model weights as flat arrays
#   meta/...     — JSON-encoded architecture dims, normalisation stats,
#                  data contract (species list, feature order), and config.
#
# load_exported_model re-assembles these into an ExportedJAXModel dataclass
# whose .predict_fastchem_profile() method accepts raw physical inputs.  The training
# codebase is NOT required to load or run this bundle.
model = load_exported_model(BUNDLE_PATH)
if not model.uses_fastchem:
    raise RuntimeError(
        "extras/standalone_test.py supports FastChem export bundles only; "
        f"got a non-FastChem bundle at {BUNDLE_PATH}."
    )

# The data contract records the species order and feature metadata that were
# fixed at preprocessing time.  We use it to know which column is which.
contract      = model.data_contract
species       = list(contract["output_species_order"])
n_species     = len(species)
global_order  = list(contract["global_static_feature_order"])
if global_order != _EXPECTED_GLOBAL_ORDER:
    raise RuntimeError(
        "extras/standalone_test.py requires an exported FastChem bundle with the "
        f"current `X/H` contract {_EXPECTED_GLOBAL_ORDER}, "
        f"but the selected bundle stores {global_order}. Regenerate the example bundle "
        "from a checkpoint trained with the current pipeline."
    )

def _globals_vector(
    globals_map: dict[str, float],
    overrides: dict[str, jax.Array | float] | None = None,
) -> jax.Array:
    """Build a global feature vector in the exported bundle order."""
    merged = dict(globals_map)
    if overrides is not None:
        merged.update(overrides)
    missing = [name for name in global_order if name not in merged]
    if missing:
        raise ValueError(f"global_inputs is missing required keys: {missing}")
    return jnp.stack(
        [jnp.asarray(merged[name], dtype=jnp.float32) for name in global_order]
    )


def _global_label(name: str) -> str:
    """Map exported global feature names to compact plot/table labels."""
    labels = {
        "He_H": "He/H",
        "C_H": "C/H",
        "O_H": "O/H",
        "N_H": "N/H",
        "S_H": "S/H",
    }
    return labels.get(name, name)


def _fastchem_element_globals(globals_map: dict[str, float]) -> dict[str, float]:
    """Return explicit FastChem elemental abundances from the exported `X/H` globals."""
    required = list(ELEMENT_INPUT_ORDER)
    missing = [name for name in required if name not in globals_map]
    if missing:
        raise ValueError(f"global_inputs is missing required elemental abundances: {missing}")
    return {name: float(globals_map[name]) for name in required}


global_labels = [_global_label(name) for name in global_order]

print(f"  Model type      : FastChem {model.model_type}")
print(f"  Bundle version  : {model.export_version}")
print(f"  Output species  : {', '.join(species)}")
print(f"  Global features : {global_order}")


# ============================================================================
# SECTION 2 — Build a synthetic hot-Jupiter T-P profile
# ============================================================================

print("\n" + "=" * 72)
print("SECTION 2 — Constructing a hot-Jupiter T-P profile")
print("=" * 72)

# ------- Pressure grid -------------------------------------------------------
# 50 levels log-spaced from 100 bar (deep photosphere) to 10^-5 bar (top).
# This standalone script calls ExportedJAXModel.predict_fastchem_profile
# directly, so it uses the repository's internal bottom-to-top ordering
# (high pressure -> low pressure).  The ExoJAX wrapper layer is the separate
# public top-to-bottom interface.
N_LEVELS = 50
P_BOTTOM = 100.0    # bar
P_TOP    = 1.0e-5   # bar
pressure_bar = np.logspace(np.log10(P_BOTTOM), np.log10(P_TOP), N_LEVELS)

# ------- Temperature profile -------------------------------------------------
# Simple analytic hot-Jupiter T-P:
#
#   T(P) = T_upper_ref + (T_deep − T_upper_ref) × (P / P_bottom)^α
#
# T_upper_ref is the asymptotic upper-atmosphere reference temperature as
# P → 0, so the finite top layer remains warmer than this reference value.
# Real profiles are more complex (inversions, radiative-convective transition)
# but this is sufficient for demonstration.
T_DEEP      = 2000.0  # K — base of atmosphere
T_UPPER_REF =  800.0  # K — asymptotic upper-atmosphere reference
ALPHA       = 0.08    # power-law index

temperature_k = T_UPPER_REF + (T_DEEP - T_UPPER_REF) * (pressure_bar / P_BOTTOM) ** ALPHA

print(f"  Pressure grid   : {pressure_bar[-1]:.1e} − {pressure_bar[0]:.1f} bar  ({N_LEVELS} levels)")
print(f"  Temperature     : {temperature_k[-1]:.0f} K (top layer) − {temperature_k[0]:.0f} K (bottom)")

# ------- Global chemistry parameters ----------------------------------------
# The learned chemistry inputs are the profile-global FastChem/VULCAN
# elemental abundances in hydrogen-normalized `X/H` form.
global_inputs = {
    "He_H": _SOLAR_ELEMENT_ABUNDANCES["He_H"],
    "C_H": _SOLAR_ELEMENT_ABUNDANCES["C_H"],
    "O_H": _SOLAR_ELEMENT_ABUNDANCES["O_H"],
    "N_H": _SOLAR_ELEMENT_ABUNDANCES["N_H"],
    "S_H": _SOLAR_ELEMENT_ABUNDANCES["S_H"],
}
print(f"  Global inputs   : {global_inputs}")


# ============================================================================
# SECTION 3 — Forward pass: equilibrium mixing ratios from the JAX model
# ============================================================================

print("\n" + "=" * 72)
print("SECTION 3 — Running equilibrium inference (JAX forward pass)")
print("=" * 72)

# predict_fastchem_profile bundles the full pipeline in one call:
#   1. Normalise pressure (log-standard: log10 then z-score)
#   2. Normalise temperature (standard: z-score only)
#   3. Normalise global scalars (mixed per-feature methods)
#   4. Run apply_mlp              → normalised log10 mixing ratios
#   5. Inverse-normalise          → physical mixing ratios (or log10 if asked)
#
# return_log10=True keeps predictions in log10 space, which avoids underflow
# for trace-abundance species and is more convenient for plotting.
mixing_ratios_log10 = model.predict_fastchem_profile(
    pressure_bar=pressure_bar,
    temperature_k=temperature_k,
    global_inputs=global_inputs,
    return_log10=True,
)
mixing_ratios_log10 = np.asarray(mixing_ratios_log10)   # (N_LEVELS, n_species)

print(f"  Output shape    : {mixing_ratios_log10.shape}  (levels × species)")
print()
print("  Key species at photosphere (P ≈ 0.1 bar):")

# Find the pressure level closest to 0.1 bar (photosphere of a hot Jupiter)
phot_idx = int(np.argmin(np.abs(pressure_bar - 0.1)))
for name in ["H2", "H2O", "CO", "CO2", "CH4", "NH3", "H2S"]:
    if name in species:
        i = species.index(name)
        print(f"    {name:>5s}  log10(X) = {mixing_ratios_log10[phot_idx, i]:+.3f}")


# ============================================================================
# SECTION 4 — Optional FastChem comparison
# ============================================================================

print("\n" + "=" * 72)
print("SECTION 4 — FastChem comparison  (requires VULCAN-master)")
print("=" * 72)

# FastChem is the equilibrium chemistry solver embedded inside VULCAN.
# It was used to generate every training profile in this dataset, so running
# it here on our synthetic T-P profile gives the exact ground truth the
# emulator was trained to reproduce.
#
# The FastChem setup below mirrors what VULCAN does internally.  The compare
# helper expects explicit FastChem-native elemental abundances, and the
# exported equilibrium contract already supplies exactly those `X/H` values.
#
# This section is entirely optional — the script continues gracefully if the
# VULCAN-master source tree is absent or FastChem fails to run.
fastchem_ymix: np.ndarray | None = None

try:
    # _run_fastchem_online (from compare_test_profile_fastchem.py) implements
    # exactly the same setup described above: copies the FastChem runtime to a
    # temp directory, writes the T-P profile and element abundances, runs the
    # binary, and parses the output table.  It is the same code path used by
    # the generation pipeline and the validation scripts.
    from compare_test_profile_fastchem import _run_fastchem_online   # type: ignore[import]

    if not VULCAN_SOURCE_ROOT.exists():
        print(f"  Skipped — VULCAN source tree not found at: {VULCAN_SOURCE_ROOT}")
        print("  (clone VULCAN-master alongside vulcan-emulator to enable this section)")
    else:
        print(f"  Running FastChem from: {VULCAN_SOURCE_ROOT}")
        fastchem_ymix = _run_fastchem_online(
            source_root=VULCAN_SOURCE_ROOT,
            pressure_bar=pressure_bar,
            temperature_k=temperature_k,
            globals_map=_fastchem_element_globals(global_inputs),
            output_species=species,      # must match the training species list exactly
        )
        # fastchem_ymix has shape (N_LEVELS, n_species), in *linear* mixing ratio space.
        print(f"  FastChem output shape: {fastchem_ymix.shape}")

        h2o_i = species.index("H2O")
        em   = float(mixing_ratios_log10[phot_idx, h2o_i])
        fc   = float(np.log10(max(fastchem_ymix[phot_idx, h2o_i], 1e-30)))
        print(f"  H2O @ 0.1 bar — emulator: {em:.3f} dex,  FastChem: {fc:.3f} dex,  Δ = {em-fc:+.3f} dex")

except Exception as exc:
    print(f"  Skipped — {type(exc).__name__}: {exc}")


# ============================================================================
# SECTION 5 — Backpropagation with JAX
# ============================================================================
#
# The emulator is a *pure JAX program*.  Every operation from raw physical
# inputs to predicted mixing ratios — normalisation, FiLM-conditioned MLP,
# inverse normalisation — is built from differentiable JAX primitives.
#
# This means jax.grad / jax.jacobian / jax.value_and_grad work through the
# entire pipeline with no extra effort.  No finite differences, no ad-hoc
# Jacobian code.
#
# Why does this matter for atmospheric chemistry?
#   • Sensitivity analysis  — which T levels / abundances drive a given species?
#   • Uncertainty propagation — propagate parameter uncertainties analytically.
#   • Retrieval / inversion  — gradient-descent in chemistry-global space to match
#     an observed spectrum (much faster than MCMC for smooth posteriors).
#   • Physics-informed losses — plug the emulator into a differentiable
#     radiative-transfer code and train end-to-end.
# ============================================================================

print("\n" + "=" * 72)
print("SECTION 5 — Backpropagation examples (JAX automatic differentiation)")
print("=" * 72)

# ---------------------------------------------------------------------------
# Build a *pure JAX forward function* that accepts JAX arrays and returns
# JAX arrays.  This is what jax.grad / jax.jacobian need.
#
# We replicate the four normalisation steps that predict_fastchem_profile
# does internally, but expressed as function arguments so that JAX can trace
# gradients through all of them.
#
# Frozen references (params, normalization stats, dims) are captured in the
# closure — they are constants in the computation graph.
# ---------------------------------------------------------------------------
_params = model.params
_norm   = model.normalization
_dims   = model.dims


def _forward_log10(
    pressure: jax.Array,    # (nz,)   pressure in bar
    temperature: jax.Array, # (nz,)   temperature in K
    globals_vec: jax.Array, # (n_global,) in exported feature order
) -> jax.Array:
    """Pure JAX forward pass: physical inputs → log10(mixing ratios).

    This function is compatible with jax.grad, jax.jacobian, jax.jit,
    jax.vmap, and any other JAX transformation.

    Returns
    -------
    jax.Array of shape (nz, n_species) — log10 mixing ratios
    """
    # ---- Step 1: normalise per-level inputs --------------------------------
    # Stack P and T into a (nz, 2) matrix, then apply per-column normalisation
    # from the stored statistics:
    #   pressure    → log-standard  (log10 first, then z-score)
    #   temperature → standard      (z-score only)
    static_inputs = jnp.stack([pressure, temperature], axis=-1)   # (nz, 2)
    sequence = _apply_sequence_static_block_jax(static_inputs, _norm["sequence_static"])
    sequence = sequence[None, :, :]   # add batch dim → (1, nz, 2)

    # ---- Step 2: normalise global scalars ----------------------------------
    # apply_mixed_block_jax applies the stored per-feature normalisation
    # methods for the exported elemental-abundance features.
    globals_norm = apply_mixed_block_jax(globals_vec[None, :], _norm["global_static"])  # (1, n_global)

    # ---- Step 3: run the FiLM-conditioned MLP ------------------------------
    # apply_mlp is a pure JAX function.  It returns predictions in
    # the model's *normalised* output space (z-scored log10 mixing ratios).
    # All parameters (_params) are constants w.r.t. differentiation here.
    pred_norm, _ = apply_mlp(_params, sequence, globals_norm, _dims)
    pred_norm = pred_norm[0]   # remove batch dim → (nz, n_species)

    # ---- Step 4: undo z-score, stay in log10 space -------------------------
    # _restore_block_transform_space_jax undoes only the affine z-score
    # (pred * std + mean) without applying 10^x.  This gives log10 mixing
    # ratios, which is the most numerically stable form for gradients.
    log10_ratios = _restore_block_transform_space_jax(pred_norm, _norm["target"])

    return log10_ratios   # (nz, n_species)


# ---------------------------------------------------------------------------
# Convert inputs to JAX arrays once, outside of any gradient computation.
# The dict→array conversion for global features is Python-level and can't
# sit inside jax.grad; doing it here is fine.
# ---------------------------------------------------------------------------
pressure_jax    = jnp.asarray(pressure_bar,  dtype=jnp.float32)
temperature_jax = jnp.asarray(temperature_k, dtype=jnp.float32)
global_arr      = _globals_vector(global_inputs)

h2o_idx = species.index("H2O")


# ---- Demo 5a: sensitivity of H2O to temperature at each level -------------
print("\n  Demo 5a — dH2O/dT: sensitivity of water to temperature (jax.grad)")
print("  ─" * 35)

# We want: d(column-mean log10 H2O) / d T_i for every level i.
# jax.grad requires a scalar-valued function, so we average H2O over the
# vertical column first.
def _mean_h2o(temperature: jax.Array) -> jax.Array:
    """Column-averaged log10(H2O) as a function of the temperature profile."""
    log10_ratios = _forward_log10(pressure_jax, temperature, global_arr)
    return jnp.mean(log10_ratios[:, h2o_idx])   # scalar

# jax.grad differentiates w.r.t. the *first* argument by default.
# The output is a (nz,) array: one d(H2O)/dT_i per pressure level.
# jax.jit compiles the gradient function for efficient repeated calls.
grad_h2o_wrt_T = jax.jit(jax.grad(_mean_h2o))
sensitivity_T  = np.asarray(grad_h2o_wrt_T(temperature_jax))

peak_i   = int(np.argmax(np.abs(sensitivity_T)))
peak_val = float(sensitivity_T[peak_i])
direction = "decreases" if peak_val < 0 else "increases"

print(f"    Gradient shape   : {sensitivity_T.shape}")
print(f"    Peak sensitivity : {abs(peak_val):.3e} dex K⁻¹  at  P = {pressure_bar[peak_i]:.2e} bar")
print(f"    Interpretation   : warming by 1 K at P = {pressure_bar[peak_i]:.2e} bar {direction} H2O by {abs(peak_val):.3e} dex")


# ---- Demo 5b: Jacobian of all species w.r.t. global parameters ------------
print("\n  Demo 5b — Full Jacobian: d(species) / d(global params)  (jax.jacobian)")
print("  ─" * 35)

# jax.jacobian computes the full matrix of partial derivatives.
# _column_mean_log10 maps (n_global,) → (n_species,), so the Jacobian is
# (n_species, n_global): each row is how one species responds to the exported
# global conditioning features.
def _column_mean_log10(globals_vec: jax.Array) -> jax.Array:
    """Column-averaged log10 mixing ratio for every species."""
    log10_ratios = _forward_log10(pressure_jax, temperature_jax, globals_vec)
    return jnp.mean(log10_ratios, axis=0)   # (n_species,)

# jax.jacobian uses forward-mode AD by default when the input is small here
# relative to the output (17 species).  You can force
# reverse-mode with argnums + jacrev if the output is small instead.
jacobian_fn = jax.jit(jax.jacobian(_column_mean_log10))
jac = np.asarray(jacobian_fn(global_arr))   # (n_species, n_global)

print(f"    Jacobian shape   : {jac.shape}  (n_species × n_global_features)")
header = f"\n    {'Species':>6}" + "".join(f"  {label:>10}" for label in global_labels)
print(header)
for name in ["H2", "H2O", "CO", "CO2", "CH4", "NH3", "H2S", "SO2", "S2"]:
    if name in species:
        i = species.index(name)
        row = "".join(f"  {value:>10.3f}" for value in jac[i])
        print(f"    {name:>6}{row}")


# ---- Demo 5c: gradient descent — recover a known in-range C/H target -------
print("\n  Demo 5c — Gradient descent: recover a known C/H from its H2O target")
print("  ─" * 35)
print("  (This is a toy in-range inversion demo solved with backprop.)")

# We derive a target column-averaged log10(H2O) from a known in-range C/H value,
# then start from a different in-range C/H and use gradient descent to recover
# the target.  This keeps the demo inside the bundle's training support.
#
# Physical intuition: at fixed O/H, higher C/H tends to tie up more oxygen in
# CO rather than H2O.  If we want more H2O, the optimizer should move C/H down.
#
# We project every update back into the configured training range so the toy
# retrieval stays in-distribution for the exported bundle.
C_TO_O_MIN, C_TO_O_MAX = [float(x) for x in model.config["sampling"]["c_to_o_range"]]
if C_TO_O_MIN >= C_TO_O_MAX:
    raise RuntimeError(
        "Expected the C/O support implied by sampling.c_to_o_range "
        f"to have distinct increasing bounds, got [{C_TO_O_MIN}, {C_TO_O_MAX}]."
    )

C_H_MIN = float(global_inputs["O_H"] * C_TO_O_MIN)
C_H_MAX = float(global_inputs["O_H"] * C_TO_O_MAX)
c_h_span = C_H_MAX - C_H_MIN
c_h_start = C_H_MIN + 0.75 * c_h_span
target_c_h = C_H_MIN + 0.375 * c_h_span
TARGET_H2O_LOG10 = float(
    _column_mean_log10(
        _globals_vector(global_inputs, {"C_H": target_c_h})
    )[h2o_idx]
)
LR            = 1.0e2   # gradient-descent learning rate in C/H space
N_STEPS       = 60      # number of update steps
MAX_GRAD      = 1.0e-3  # gradient clip threshold in the scalar optimization space
H2O_TOL_DEX   = 0.01    # report convergence only within this residual tolerance

def _h2o_loss(c_h_scalar: jax.Array) -> jax.Array:
    """Scalar MSE loss: recover the target H2O abundance by varying C/H only."""
    globals_vec = _globals_vector(global_inputs, {"C_H": c_h_scalar})
    log10_ratios = _forward_log10(pressure_jax, temperature_jax, globals_vec)
    predicted    = jnp.mean(log10_ratios[:, h2o_idx])     # scalar
    return (predicted - TARGET_H2O_LOG10) ** 2

# jax.value_and_grad returns (loss_value, gradient) in a single forward+backward
# pass — more efficient than calling them separately.
loss_and_grad = jax.jit(jax.value_and_grad(_h2o_loss))

co_opt = jnp.asarray(c_h_start, dtype=jnp.float32)

print(f"    C/H training slice: [{C_H_MIN:.3e}, {C_H_MAX:.3e}]")
print(f"    Target C/H        : {target_c_h:.3e}  (used to define the target H2O abundance)")
print(f"    Target log10(H2O) : {TARGET_H2O_LOG10:.3f} dex")
print(f"    Starting C/H      : {c_h_start:.3e}")
print()

for step in range(N_STEPS):
    loss_val, grad = loss_and_grad(co_opt)

    # Clip the gradient to prevent large steps far from the optimum.
    # Without clipping, the first few steps can overshoot into unphysical
    # regions where the model is extrapolating.
    grad_clipped = jnp.clip(grad, -MAX_GRAD, MAX_GRAD)

    # Plain gradient descent with projection back into the training support.
    co_opt = jnp.clip(co_opt - LR * grad_clipped, C_H_MIN, C_H_MAX)

    if step % 10 == 0 or step == N_STEPS - 1:
        current_c_h = float(co_opt)
        cur_h2o = float(
            _column_mean_log10(_globals_vector(global_inputs, {"C_H": current_c_h}))[h2o_idx]
        )
        print(f"    step {step:3d}: loss = {float(loss_val):.2e}  "
              f"C/H = {current_c_h:.3e}  "
              f"H2O = {cur_h2o:.4f} dex")

final_c_h = float(co_opt)
final_h2o = float(
    _column_mean_log10(_globals_vector(global_inputs, {"C_H": final_c_h}))[h2o_idx]
)
final_residual = final_h2o - TARGET_H2O_LOG10

print(f"\n    Final C/H        : {final_c_h:.3e}  (target {target_c_h:.3e})")
print(f"    Final log10(H2O) : {final_h2o:.4f} dex")
print(f"    Final residual   : {final_residual:+.4f} dex")
if abs(final_residual) <= H2O_TOL_DEX:
    print(f"    Status           : converged within {H2O_TOL_DEX:.3f} dex")
else:
    print(f"    Status           : did not converge within {H2O_TOL_DEX:.3f} dex")


# ============================================================================
# SECTION 6 — Plots
# ============================================================================

print("\n" + "=" * 72)
print("SECTION 6 — Generating plots")
print("=" * 72)

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend; safe on servers / clusters
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    if _MPLSTYLE.exists():
        plt.style.use(str(_MPLSTYLE))

    colors = plt.cm.tab20(np.linspace(0, 1, n_species))

    # ---- Plot 1: T-P profile + mixing ratios (always exactly 2 panels) ----
    from matplotlib.lines import Line2D

    # Two square panels side by side.
    fig, (ax_pt, ax_mix) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    # Panel A: T-P profile
    ax_pt.plot(temperature_k, pressure_bar, "k-", lw=2)
    ax_pt.set_xlabel("Temperature [K]")
    ax_pt.set_ylabel("Pressure [bar]")
    ax_pt.set_yscale("log")
    ax_pt.invert_yaxis()
    ax_pt.set_xlim(0, 2500)
    ax_pt.set_title("Synthetic T-P Profile")
    ax_pt.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=8))
    ax_pt.yaxis.set_minor_locator(mticker.NullLocator())
    ax_pt.set_aspect("auto")

    # Panel B: mixing ratios.
    # If FastChem ran successfully, plot both on the same axes:
    #   FastChem → solid lines   (the ground truth the emulator was trained on)
    #   Emulator → dashed lines  (the neural network prediction)
    # If FastChem is unavailable, only the emulator lines are drawn (solid).
    for i, name in enumerate(species):
        c = colors[i]
        if fastchem_ymix is not None:
            # FastChem: solid — the reference solver
            ax_mix.plot(
                np.maximum(fastchem_ymix[:, i], 1e-30),
                pressure_bar,
                color=c, lw=1.6, ls="-", label=name,
            )
            # Emulator: dashed — the neural network
            ax_mix.plot(
                10 ** mixing_ratios_log10[:, i],
                pressure_bar,
                color=c, lw=1.1, ls="--",
            )
        else:
            ax_mix.plot(
                10 ** mixing_ratios_log10[:, i],
                pressure_bar,
                color=c, lw=1.5, label=name,
            )

    ax_mix.set_xscale("log")
    ax_mix.set_xlim(1e-20, 3)
    ax_mix.set_xlabel("Mixing Ratio")
    ax_mix.set_title("Equilibrium Mixing Ratios")

    # Species legend (colours)
    species_legend = ax_mix.legend(fontsize=7, ncol=3, loc="lower left")

    # Line-style legend (only shown when FastChem is present)
    if fastchem_ymix is not None:
        ax_mix.add_artist(species_legend)   # keep the species legend visible
        ax_mix.legend(
            handles=[
                Line2D([0], [0], color="k", lw=1.6, ls="-",  label="FastChem"),
                Line2D([0], [0], color="k", lw=1.1, ls="--", label="Emulator"),
            ],
            fontsize=9, loc="upper right",
        )

    fig.suptitle(
        rf"Hot-Jupiter equilibrium chemistry  "
        rf"(He/H={global_inputs['He_H']:.3e},  "
        rf"C/H={global_inputs['C_H']:.3e},  "
        rf"O/H={global_inputs['O_H']:.3e})",
        fontsize=11,
    )
    fig.tight_layout()
    # Force both panels to the same square size after tight_layout.
    for ax in (ax_pt, ax_mix):
        ax.set_box_aspect(1)
    fig.tight_layout()
    out1 = OUTPUT_DIR / "01_mixing_ratios.png"
    fig.savefig(out1, dpi=160)
    plt.close(fig)
    print(f"  Saved: {out1}")

    # ---- Plot 2: Temperature sensitivity (backprop result) ----------------
    # This plot directly visualises a jax.grad output.
    # The sensitivity d(mean log10 H2O)/dT tells you which atmospheric levels
    # most strongly control the water abundance for this profile.
    fig2, (ax_s, ax_tp2) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    ax_s.plot(sensitivity_T, pressure_bar, color="steelblue", lw=2)
    ax_s.axvline(0, color="k", lw=0.8, alpha=0.4, ls="--")
    # Use a short two-line label so it fits comfortably on the axis.
    ax_s.set_xlabel(
        r"$\partial\,\langle\log_{10}\,X_{\mathrm{H_2O}}\rangle\,/\,\partial T_i$"
        "\n"
        r"[dex K$^{-1}$]",
        fontsize=11,
    )
    ax_s.set_ylabel("Pressure [bar]")
    ax_s.set_yscale("log")
    ax_s.invert_yaxis()
    ax_s.set_title(r"H$_2$O sensitivity to temperature  (jax.grad)", fontsize=11)
    ax_s.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=8))
    ax_s.yaxis.set_minor_locator(mticker.NullLocator())
    # Use scientific notation for the x-axis tick labels so small values are legible.
    ax_s.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax_s.ticklabel_format(axis="x", style="sci", scilimits=(-5, -5))
    ax_s.tick_params(axis="x", labelsize=10)
    ax_s.set_box_aspect(1)

    ax_tp2.plot(temperature_k, pressure_bar, "k-", lw=2)
    ax_tp2.set_xlabel("Temperature [K]")
    ax_tp2.set_title("T-P Profile (reference)")
    ax_tp2.set_xlim(0, 2500)
    ax_tp2.set_box_aspect(1)

    fig2.tight_layout()
    out2 = OUTPUT_DIR / "02_h2o_temperature_sensitivity.png"
    fig2.savefig(out2, dpi=160)
    plt.close(fig2)
    print(f"  Saved: {out2}")

    # ---- Plot 3: Jacobian heatmap (species × global parameters) -----------
    # Each cell [i, j] = d(column-mean log10 species_i) / d(global_j).
    # Red = species increases when the parameter increases.
    # Blue = species decreases when the parameter increases.
    abs_max = float(np.max(np.abs(jac)))
    fig3, ax_j = plt.subplots(figsize=(7, 7))
    im = ax_j.imshow(
        jac, aspect="auto", cmap="RdBu_r",
        vmin=-abs_max, vmax=abs_max,
    )
    ax_j.set_xticks(range(len(global_order)))
    ax_j.set_xticklabels(global_labels, fontsize=12)
    ax_j.set_yticks(range(n_species))
    ax_j.set_yticklabels(species, fontsize=10)
    plt.colorbar(
        im, ax=ax_j,
        label=r"$\partial\langle\log_{10} X_i\rangle\,/\,\partial\,\theta_j$  [dex]",
        shrink=0.7,
    )
    ax_j.set_title("Species sensitivity to global parameters\n(jax.jacobian)", fontsize=11)
    ax_j.set_box_aspect(1)
    fig3.tight_layout()
    out3 = OUTPUT_DIR / "03_jacobian_global_params.png"
    fig3.savefig(out3, dpi=160)
    plt.close(fig3)
    print(f"  Saved: {out3}")

except ImportError as exc:
    print(f"  Matplotlib not available — skipping plots ({exc})")

print("\n" + "=" * 72)
print("Done.  All outputs are in:", OUTPUT_DIR)
print("=" * 72)
