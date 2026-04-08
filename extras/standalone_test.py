"""End-to-end standalone FastChem emulator demo with JAX autodiff.

Usage:
    python extras/standalone_test.py
    python extras/standalone_test.py --bundle models/fastchem_mlp/best_exported.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "extras"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from compare_test_profile_fastchem import _run_fastchem_online  # noqa: E402
from src.models.export_bundle import (  # noqa: E402
    _apply_sequence_static_block_jax,
    _restore_block_transform_space_jax,
    apply_mixed_block_jax,
    load_exported_model,
)
from src.models.jax_model import apply_mlp, apply_transformer_model  # noqa: E402
from src.utils.config import ELEMENT_INPUT_ORDER, load_and_validate_config  # noqa: E402
from src.utils.helpers import resolve_path, resolve_project_root  # noqa: E402

_DEFAULT_CONFIG = _ROOT / "config" / "fastchem_mlp_config.json"
_SOLAR_ELEMENT_ABUNDANCES = {
    "O_H": 5.37e-4,
    "C_H": 2.95e-4,
    "N_H": 7.08e-5,
    "S_H": 1.41e-5,
    "He_H": 8.38e-2,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the standalone autodiff demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="FastChem config used to derive the default export-bundle path.",
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help="Optional explicit path to an exported FastChem bundle.",
    )
    parser.add_argument(
        "--vulcan-source-root",
        default=None,
        help="Optional VULCAN-master checkout for the FastChem comparison.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for generated figures.",
    )
    return parser.parse_args(argv)


def _resolve_bundle_path(config: dict[str, object], project_root: Path, explicit: str | None) -> Path:
    """Resolve the export-bundle path, preferring ``best_exported.npz``."""
    if explicit is not None:
        return resolve_path(explicit, project_root)

    checkpoints_root = resolve_path(config["paths"]["checkpoints_root"], project_root)
    default_bundle = checkpoints_root / "best_exported.npz"
    if default_bundle.exists():
        return default_bundle

    candidates = sorted(checkpoints_root.glob("*_exported.npz"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No exported bundle found under {checkpoints_root}. "
            "Pass --bundle explicitly or export a checkpoint first."
        )
    raise FileNotFoundError(
        f"Multiple exported bundles found under {checkpoints_root}: {candidates}. "
        "Pass --bundle explicitly."
    )


def _globals_vector(
    *,
    global_order: list[str],
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
    return jnp.stack([jnp.asarray(merged[name], dtype=jnp.float32) for name in global_order])


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
    """Return explicit FastChem elemental abundances from exported ``X/H`` globals."""
    required = list(ELEMENT_INPUT_ORDER)
    missing = [name for name in required if name not in globals_map]
    if missing:
        raise ValueError(f"global_inputs is missing required elemental abundances: {missing}")
    return {name: float(globals_map[name]) for name in required}


def main(argv: list[str] | None = None) -> int:
    """Run the standalone FastChem emulator and autodiff demonstration."""
    args = _parse_args(argv)
    project_root = resolve_project_root(Path(__file__).resolve())
    config = load_and_validate_config(resolve_path(args.config, project_root))
    bundle_path = _resolve_bundle_path(config, project_root, args.bundle)
    vulcan_source_root = resolve_path(
        args.vulcan_source_root or config["paths"]["vulcan_source_root"],
        project_root,
    )
    output_dir = (
        resolve_path(args.output_dir, project_root)
        if args.output_dir is not None
        else _ROOT / "extras" / "standalone_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_exported_model(bundle_path)
    if not model.uses_fastchem:
        raise RuntimeError(f"extras/standalone_test.py requires a FastChem bundle, got {bundle_path}.")

    expected_global_order = list(ELEMENT_INPUT_ORDER)
    contract = model.data_contract
    species = list(contract["output_species_order"])
    n_species = len(species)
    global_order = list(contract["global_static_feature_order"])
    if global_order != expected_global_order:
        raise RuntimeError(
            "extras/standalone_test.py requires an exported FastChem bundle with "
            f"global inputs {expected_global_order}, but the bundle stores {global_order}."
        )

    print("=" * 72)
    print("SECTION 1 — Loading exported JAX model bundle")
    print("=" * 72)
    print(f"  Bundle path     : {bundle_path}")
    print(f"  Model type      : FastChem {model.model_type}")
    print(f"  Bundle version  : {model.export_version}")
    print(f"  Output species  : {', '.join(species)}")
    print(f"  Global features : {global_order}")

    print("\n" + "=" * 72)
    print("SECTION 2 — Constructing a hot-Jupiter T-P profile")
    print("=" * 72)

    n_levels = 50
    pressure_bar = np.logspace(np.log10(100.0), np.log10(1.0e-5), n_levels)
    temperature_k = 800.0 + (2000.0 - 800.0) * (pressure_bar / 100.0) ** 0.08
    global_inputs = dict(_SOLAR_ELEMENT_ABUNDANCES)

    print(f"  Pressure grid   : {pressure_bar[-1]:.1e} − {pressure_bar[0]:.1f} bar  ({n_levels} levels)")
    print(f"  Temperature     : {temperature_k[-1]:.0f} K (top) − {temperature_k[0]:.0f} K (bottom)")
    print(f"  Global inputs   : {global_inputs}")

    print("\n" + "=" * 72)
    print("SECTION 3 — Running equilibrium inference (JAX forward pass)")
    print("=" * 72)

    mixing_ratios_log10 = np.asarray(
        model.predict_fastchem_profile(
            pressure_bar=pressure_bar,
            temperature_k=temperature_k,
            global_inputs=global_inputs,
            return_log10=True,
        )
    )
    phot_idx = int(np.argmin(np.abs(pressure_bar - 0.1)))

    print(f"  Output shape    : {mixing_ratios_log10.shape}  (levels × species)")
    print("\n  Key species at photosphere (P ≈ 0.1 bar):")
    for name in ["H2", "H2O", "CO", "CO2", "CH4", "NH3", "H2S"]:
        if name in species:
            species_index = species.index(name)
            print(f"    {name:>5s}  log10(X) = {mixing_ratios_log10[phot_idx, species_index]:+.3f}")

    print("\n" + "=" * 72)
    print("SECTION 4 — FastChem comparison  (requires VULCAN-master)")
    print("=" * 72)

    fastchem_ymix: np.ndarray | None = None
    try:
        if not vulcan_source_root.exists():
            print(f"  Skipped — VULCAN source tree not found at: {vulcan_source_root}")
        else:
            print(f"  Running FastChem from: {vulcan_source_root}")
            fastchem_ymix = _run_fastchem_online(
                source_root=vulcan_source_root,
                pressure_bar=pressure_bar,
                temperature_k=temperature_k,
                globals_map=_fastchem_element_globals(global_inputs),
                output_species=species,
            )
            h2o_i = species.index("H2O")
            emulator_h2o = float(mixing_ratios_log10[phot_idx, h2o_i])
            fastchem_h2o = float(np.log10(max(fastchem_ymix[phot_idx, h2o_i], 1.0e-30)))
            print(f"  FastChem output shape: {fastchem_ymix.shape}")
            print(
                "  H2O @ 0.1 bar — emulator: "
                f"{emulator_h2o:.3f} dex, FastChem: {fastchem_h2o:.3f} dex, "
                f"Δ = {emulator_h2o - fastchem_h2o:+.3f} dex"
            )
    except Exception as exc:
        print(f"  Skipped — {type(exc).__name__}: {exc}")

    print("\n" + "=" * 72)
    print("SECTION 5 — Backpropagation examples (JAX automatic differentiation)")
    print("=" * 72)

    pressure_jax = jnp.asarray(pressure_bar, dtype=jnp.float32)
    temperature_jax = jnp.asarray(temperature_k, dtype=jnp.float32)
    global_arr = _globals_vector(global_order=global_order, globals_map=global_inputs)
    h2o_idx = species.index("H2O")

    def _forward_log10(
        pressure: jax.Array,
        temperature: jax.Array,
        globals_vec: jax.Array,
    ) -> jax.Array:
        """Pure JAX forward pass from physical inputs to log10 mixing ratios."""
        static_inputs = jnp.stack([pressure, temperature], axis=-1)
        sequence = _apply_sequence_static_block_jax(static_inputs, model.normalization["sequence_static"])
        sequence = sequence[None, :, :]
        globals_norm = apply_mixed_block_jax(
            globals_vec[None, :],
            model.normalization["global_static"],
        )
        if model.uses_mlp:
            pred_norm, _ = apply_mlp(model.params, sequence, globals_norm, model.dims)
        else:
            pred_norm, _ = apply_transformer_model(
                model.params,
                sequence,
                globals_norm,
                None,
                model.dims,
            )
        pred_norm = pred_norm[0]
        return _restore_block_transform_space_jax(pred_norm, model.normalization["target"])

    def _mean_h2o(temperature: jax.Array) -> jax.Array:
        """Return the column-averaged log10(H2O) as a function of temperature."""
        return jnp.mean(_forward_log10(pressure_jax, temperature, global_arr)[:, h2o_idx])

    grad_h2o_wrt_t = jax.jit(jax.grad(_mean_h2o))
    sensitivity_t = np.asarray(grad_h2o_wrt_t(temperature_jax))
    peak_i = int(np.argmax(np.abs(sensitivity_t)))
    peak_val = float(sensitivity_t[peak_i])
    direction = "decreases" if peak_val < 0 else "increases"

    print("\n  Demo 5a — dH2O/dT: sensitivity of water to temperature")
    print(f"    Gradient shape   : {sensitivity_t.shape}")
    print(f"    Peak sensitivity : {abs(peak_val):.3e} dex K⁻¹ at P = {pressure_bar[peak_i]:.2e} bar")
    print(
        f"    Interpretation   : warming by 1 K at P = {pressure_bar[peak_i]:.2e} bar "
        f"{direction} H2O by {abs(peak_val):.3e} dex"
    )

    def _column_mean_log10(globals_vec: jax.Array) -> jax.Array:
        """Return the column-mean log10 mixing ratio for every species."""
        return jnp.mean(_forward_log10(pressure_jax, temperature_jax, globals_vec), axis=0)

    jacobian_fn = jax.jit(jax.jacobian(_column_mean_log10))
    jac = np.asarray(jacobian_fn(global_arr))
    global_labels = [_global_label(name) for name in global_order]

    print("\n  Demo 5b — Full Jacobian: d(species) / d(global params)")
    print(f"    Jacobian shape   : {jac.shape}  (n_species × n_global_features)")
    header = f"\n    {'Species':>6}" + "".join(f"  {label:>10}" for label in global_labels)
    print(header)
    for name in ["H2", "H2O", "CO", "CO2", "CH4", "NH3", "H2S", "SO2", "S2"]:
        if name in species:
            species_index = species.index(name)
            row = "".join(f"  {value:>10.3f}" for value in jac[species_index])
            print(f"    {name:>6}{row}")

    print("\n  Demo 5c — Gradient descent: recover a known C/H from its H2O target")

    c_to_o_min, c_to_o_max = [float(x) for x in model.config["sampling"]["c_to_o_range"]]
    c_h_min = float(global_inputs["O_H"] * c_to_o_min)
    c_h_max = float(global_inputs["O_H"] * c_to_o_max)
    c_h_span = c_h_max - c_h_min
    c_h_start = c_h_min + 0.75 * c_h_span
    target_c_h = c_h_min + 0.375 * c_h_span
    target_h2o_log10 = float(
        _column_mean_log10(
            _globals_vector(
                global_order=global_order,
                globals_map=global_inputs,
                overrides={"C_H": target_c_h},
            )
        )[h2o_idx]
    )

    def _h2o_loss(c_h_scalar: jax.Array) -> jax.Array:
        """Return the scalar H2O recovery loss for the toy inversion demo."""
        globals_vec = _globals_vector(
            global_order=global_order,
            globals_map=global_inputs,
            overrides={"C_H": c_h_scalar},
        )
        predicted = jnp.mean(_forward_log10(pressure_jax, temperature_jax, globals_vec)[:, h2o_idx])
        return (predicted - target_h2o_log10) ** 2

    loss_and_grad = jax.jit(jax.value_and_grad(_h2o_loss))
    c_h_opt = jnp.asarray(c_h_start, dtype=jnp.float32)
    lr = 1.0e2
    n_steps = 60
    max_grad = 1.0e-3
    h2o_tol_dex = 0.01

    print(f"    C/H training slice: [{c_h_min:.3e}, {c_h_max:.3e}]")
    print(f"    Target C/H        : {target_c_h:.3e}")
    print(f"    Target log10(H2O) : {target_h2o_log10:.3f} dex")
    print(f"    Starting C/H      : {c_h_start:.3e}\n")

    for step in range(n_steps):
        loss_val, grad = loss_and_grad(c_h_opt)
        grad_clipped = jnp.clip(grad, -max_grad, max_grad)
        c_h_opt = jnp.clip(c_h_opt - lr * grad_clipped, c_h_min, c_h_max)
        if step % 10 == 0 or step == n_steps - 1:
            current_c_h = float(c_h_opt)
            cur_h2o = float(
                _column_mean_log10(
                    _globals_vector(
                        global_order=global_order,
                        globals_map=global_inputs,
                        overrides={"C_H": current_c_h},
                    )
                )[h2o_idx]
            )
            print(
                f"    step {step:3d}: loss = {float(loss_val):.2e}  "
                f"C/H = {current_c_h:.3e}  H2O = {cur_h2o:.4f} dex"
            )

    final_c_h = float(c_h_opt)
    final_h2o = float(
        _column_mean_log10(
            _globals_vector(
                global_order=global_order,
                globals_map=global_inputs,
                overrides={"C_H": final_c_h},
            )
        )[h2o_idx]
    )
    final_residual = final_h2o - target_h2o_log10

    print(f"\n    Final C/H        : {final_c_h:.3e}  (target {target_c_h:.3e})")
    print(f"    Final log10(H2O) : {final_h2o:.4f} dex")
    print(f"    Final residual   : {final_residual:+.4f} dex")
    status = "converged" if abs(final_residual) <= h2o_tol_dex else "did not converge"
    print(f"    Status           : {status} within {h2o_tol_dex:.3f} dex")

    print("\n" + "=" * 72)
    print("SECTION 6 — Generating plots")
    print("=" * 72)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib.lines import Line2D

        mplstyle = _ROOT / "extras" / "science.mplstyle"
        if mplstyle.exists():
            plt.style.use(str(mplstyle))

        colors = plt.cm.tab20(np.linspace(0, 1, n_species))

        fig, (ax_pt, ax_mix) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
        ax_pt.plot(temperature_k, pressure_bar, "k-", lw=2)
        ax_pt.set_xlabel("Temperature [K]")
        ax_pt.set_ylabel("Pressure [bar]")
        ax_pt.set_yscale("log")
        ax_pt.invert_yaxis()
        ax_pt.set_xlim(0, 2500)
        ax_pt.set_title("Synthetic T-P Profile")
        ax_pt.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=8))
        ax_pt.yaxis.set_minor_locator(mticker.NullLocator())

        for species_index, species_name in enumerate(species):
            color = colors[species_index]
            if fastchem_ymix is not None:
                ax_mix.plot(np.maximum(fastchem_ymix[:, species_index], 1.0e-30), pressure_bar, color=color, lw=1.6, ls="-", label=species_name)
                ax_mix.plot(10 ** mixing_ratios_log10[:, species_index], pressure_bar, color=color, lw=1.1, ls="--")
            else:
                ax_mix.plot(10 ** mixing_ratios_log10[:, species_index], pressure_bar, color=color, lw=1.5, label=species_name)

        ax_mix.set_xscale("log")
        ax_mix.set_xlim(1.0e-20, 3.0)
        ax_mix.set_xlabel("Mixing Ratio")
        ax_mix.set_title("Equilibrium Mixing Ratios")
        species_legend = ax_mix.legend(fontsize=7, ncol=3, loc="lower left")
        if fastchem_ymix is not None:
            ax_mix.add_artist(species_legend)
            ax_mix.legend(
                handles=[
                    Line2D([0], [0], color="k", lw=1.6, ls="-", label="FastChem"),
                    Line2D([0], [0], color="k", lw=1.1, ls="--", label="Emulator"),
                ],
                fontsize=9,
                loc="upper right",
            )

        fig.suptitle(
            (
                "Hot-Jupiter equilibrium chemistry  "
                f"(He/H={global_inputs['He_H']:.3e}, "
                f"C/H={global_inputs['C_H']:.3e}, "
                f"O/H={global_inputs['O_H']:.3e})"
            ),
            fontsize=11,
        )
        fig.tight_layout()
        out1 = output_dir / "01_mixing_ratios.png"
        fig.savefig(out1, dpi=160)
        plt.close(fig)
        print(f"  Saved: {out1}")

        fig2, (ax_s, ax_tp2) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
        ax_s.plot(sensitivity_t, pressure_bar, color="steelblue", lw=2)
        ax_s.axvline(0.0, color="k", lw=0.8, alpha=0.4, ls="--")
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
        ax_s.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax_s.ticklabel_format(axis="x", style="sci", scilimits=(-5, -5))

        ax_tp2.plot(temperature_k, pressure_bar, "k-", lw=2)
        ax_tp2.set_xlabel("Temperature [K]")
        ax_tp2.set_title("T-P Profile (reference)")
        ax_tp2.set_xlim(0, 2500)

        fig2.tight_layout()
        out2 = output_dir / "02_h2o_temperature_sensitivity.png"
        fig2.savefig(out2, dpi=160)
        plt.close(fig2)
        print(f"  Saved: {out2}")

        abs_max = float(np.max(np.abs(jac)))
        fig3, ax_j = plt.subplots(figsize=(7, 7))
        im = ax_j.imshow(jac, aspect="auto", cmap="RdBu_r", vmin=-abs_max, vmax=abs_max)
        ax_j.set_xticks(range(len(global_order)))
        ax_j.set_xticklabels(global_labels, fontsize=12)
        ax_j.set_yticks(range(n_species))
        ax_j.set_yticklabels(species, fontsize=10)
        plt.colorbar(
            im,
            ax=ax_j,
            label=r"$\partial\langle\log_{10} X_i\rangle\,/\,\partial\,\theta_j$  [dex]",
            shrink=0.7,
        )
        ax_j.set_title("Species sensitivity to global parameters\n(jax.jacobian)", fontsize=11)
        fig3.tight_layout()
        out3 = output_dir / "03_jacobian_global_params.png"
        fig3.savefig(out3, dpi=160)
        plt.close(fig3)
        print(f"  Saved: {out3}")
    except ImportError as exc:
        print(f"  Matplotlib not available — skipping plots ({exc})")

    print("\n" + "=" * 72)
    print("Done. All outputs are in:", output_dir)
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
