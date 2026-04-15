"""Abundance and composition utilities for VULCAN emulator outputs.

Provides:
- Mean molecular weight computation from VMR profiles
- Solar reference abundances used during model training
- Metallicity-to-fraction conversion for building global_inputs dicts
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ..constants import SOLAR_ABUNDANCES, SPECIES_MOLAR_MASS


def solar_abundances() -> dict[str, float]:
    """Return the solar reference abundances used during model training.

    Returns
    -------
    dict[str, float]
        Mapping from element key (e.g. ``"C_H"``) to number fraction.
        ``He_H`` is fixed in the current model and cannot be varied.
    """
    return dict(SOLAR_ABUNDANCES)


def mean_molecular_weight(
    vmr: jax.Array,
    species_labels: list[str],
) -> jax.Array:
    """Compute mean molecular weight from a VMR profile.

    This function is pure JAX and fully differentiable, suitable for use
    inside JIT-compiled or NUTS-traced code.

    Parameters
    ----------
    vmr : jax.Array
        Linear volume mixing ratios with shape ``(nz, n_species)`` or
        ``(n_species,)``.
    species_labels : list[str]
        Ordered species names matching the columns of *vmr*.

    Returns
    -------
    jax.Array
        Mean molecular weight in g/mol.  Shape ``(nz,)`` when *vmr* is 2-D,
        or a scalar when *vmr* is 1-D.
    """
    masses = jnp.array([SPECIES_MOLAR_MASS[s] for s in species_labels])
    return jnp.sum(vmr * masses, axis=-1)


def global_inputs_from_metallicity(
    log_metallicity: float | jax.Array = 0.0,
    *,
    c_to_o: float | jax.Array | None = None,
    log_c_scale: float | jax.Array | None = None,
    log_o_scale: float | jax.Array | None = None,
    log_n_scale: float | jax.Array | None = None,
    log_s_scale: float | jax.Array | None = None,
) -> dict[str, Any]:
    """Build a ``global_inputs`` dict from metallicity and optional overrides.

    By default every metal (C, O, N, S) is scaled uniformly by
    ``10**log_metallicity`` relative to solar.  Per-element log-scale
    overrides replace the uniform metallicity for that element.

    He is always fixed at the training-constant value (0.0838).

    All arithmetic uses :mod:`jax.numpy` so the result is JAX-traceable
    and can be used directly inside ``jax.jit`` or NumPyro NUTS models.

    Parameters
    ----------
    log_metallicity : float or jax.Array
        Log10 overall metal enrichment relative to solar.  ``0.0`` is solar.
    c_to_o : float or jax.Array, optional
        If given, sets the C/O ratio directly.  O is determined by
        metallicity (or ``log_o_scale``), then ``C_H = c_to_o * O_H``.
        Overrides ``log_c_scale`` when both are provided.
    log_c_scale, log_o_scale, log_n_scale, log_s_scale : optional
        Per-element log10 scale factors relative to solar, overriding the
        uniform ``log_metallicity`` for that element.

    Returns
    -------
    dict[str, jax.Array]
        Global inputs dict with keys ``He_H``, ``C_H``, ``O_H``, ``N_H``,
        ``S_H`` ready to pass to ``bundle.predict_fastchem()``.
    """
    solar = SOLAR_ABUNDANCES
    met = jnp.asarray(log_metallicity, dtype=jnp.float32)

    o_scale = jnp.asarray(log_o_scale, dtype=jnp.float32) if log_o_scale is not None else met
    n_scale = jnp.asarray(log_n_scale, dtype=jnp.float32) if log_n_scale is not None else met
    s_scale = jnp.asarray(log_s_scale, dtype=jnp.float32) if log_s_scale is not None else met

    o_h = solar["O_H"] * 10.0 ** o_scale
    n_h = solar["N_H"] * 10.0 ** n_scale
    s_h = solar["S_H"] * 10.0 ** s_scale

    if c_to_o is not None:
        c_h = jnp.asarray(c_to_o, dtype=jnp.float32) * o_h
    elif log_c_scale is not None:
        c_h = solar["C_H"] * 10.0 ** jnp.asarray(log_c_scale, dtype=jnp.float32)
    else:
        c_h = solar["C_H"] * 10.0 ** met

    return {
        "He_H": jnp.float32(solar["He_H"]),
        "C_H": c_h,
        "O_H": o_h,
        "N_H": n_h,
        "S_H": s_h,
    }
