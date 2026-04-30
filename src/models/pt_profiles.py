"""JAX-traceable PT profile parameterizations for inside-NUTS forward models.

The numpy versions used by the data-generation pipeline live in
``src.data_generation.sampling`` and are not differentiable. This module
provides JAX equivalents so retrieval notebooks (``exojax_demo/``) can use
the same Guillot/Piette functional forms the bundle was trained on.

The shapes here mirror exactly the analytic family the FastChem bundle is
trained against — see ``temperature_profiles.analytic_sampler`` in
``config/fastchem.json`` and the ``_guillot_temperature`` /
``_apply_upper_atmosphere_modification`` helpers in
``src/data_generation/sampling.py``.
"""

from __future__ import annotations

import jax.numpy as jnp


def guillot_temperature(
    pressure_bar,
    *,
    t_int_k,
    t_eq_k,
    log10_delta,
    log10_gamma,
):
    """Guillot 2010 / Piette & Madhusudhan 2019 base PT profile (JAX).

    Parameters
    ----------
    pressure_bar : jax.Array
        Pressure grid in bar, shape ``(nz,)``.
    t_int_k : jax scalar
        Internal temperature in Kelvin.
    t_eq_k : jax scalar
        Equilibrium temperature in Kelvin.
    log10_delta : jax scalar
        ``log10(kappa_IR / g)``. Sets the IR optical-depth scale.
    log10_gamma : jax scalar
        ``log10(kappa_V / kappa_IR)``. Visible-to-IR opacity ratio.

    Returns
    -------
    jax.Array
        Temperature profile in Kelvin, same shape as ``pressure_bar``.

    Notes
    -----
    Implements Eq. 16 from Piette & Madhusudhan (2019)::

        T^4(P) = (3/4) T_int^4 (2/3 + delta P)
               + (3/4) T_eq^4 [2/3 + 1/(gamma sqrt(3))
               +  (gamma/sqrt(3) - 1/(gamma sqrt(3))) exp(-gamma sqrt(3) delta P)]

    No upper-atmosphere modification (alpha) and no boxcar smoothing — both
    of those are awkward to JIT and the alpha=0 / smoothing-off corner of the
    sampler distribution is the fully covered base case the bundle saw.
    """
    delta = 10.0 ** log10_delta
    gamma = 10.0 ** log10_gamma
    sqrt3 = jnp.sqrt(3.0)
    tau = delta * pressure_bar

    t4_internal = 0.75 * t_int_k ** 4 * (2.0 / 3.0 + tau)
    inv_gs3 = 1.0 / (gamma * sqrt3)
    gs3 = gamma * sqrt3
    t4_stellar = 0.75 * t_eq_k ** 4 * (
        2.0 / 3.0
        + inv_gs3
        + (gamma / sqrt3 - inv_gs3) * jnp.exp(-gs3 * tau)
    )
    return jnp.clip(t4_internal + t4_stellar, 0.0, None) ** 0.25


__all__ = ["guillot_temperature"]
