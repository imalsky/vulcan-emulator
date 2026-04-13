from __future__ import annotations

import copy
import sys
from pathlib import Path

import h5py
import jax
import numpy as np
import pytest
from src.data_generation.generation import (
    RunSpecification,
    consolidate_runs_to_single_hdf5,
    write_equilibrium_hdf5,
)
from src.data_generation.preprocess import load_raw_equilibrium_run, preprocess_equilibrium_dataset
from src.models.export_bundle import export_checkpoint_payload
from src.models.jax_model import TransformerDimensions, init_transformer_params
from src.utils.config import ELEMENT_INPUT_ORDER, load_and_validate_config

ROOT = Path(__file__).resolve().parents[1]
EXTRAS_DIR = ROOT / "extras"
if str(EXTRAS_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRAS_DIR))

import _common as extras_common  # noqa: E402
import fastchem_saved_test_profile_demo as fastchem_saved_test_profile_demo_module  # noqa: E402


def _make_fastchem_extras_config(tmp_path: Path) -> dict:
    """Build one small FastChem transformer config for extras tests."""
    config = load_and_validate_config(ROOT / "config" / "fastchem_transformer_config.json")
    config = copy.deepcopy(config)
    config["paths"]["raw_root"] = str(tmp_path / "dataset" / "raw")
    config["paths"]["processed_root"] = str(tmp_path / "dataset" / "processed")
    config["paths"]["checkpoints_root"] = str(tmp_path / "checkpoints")
    config["paths"]["vulcan_source_root"] = str(tmp_path / "VULCAN")
    config["sampling"]["num_levels"] = 12
    config["normalization"]["split"] = {
        "train_fraction": 0.34,
        "val_fraction": 0.16,
        "test_fraction": 0.5,
        "seed": 123,
    }
    config["_project_root"] = ROOT
    return config


def _element_globals(run_index: int) -> dict[str, float]:
    """Return one deterministic elemental/global conditioning payload."""
    metallicity_log10 = 0.1 * run_index
    oxygen_h = 5.37e-4 * (10.0 ** metallicity_log10)
    c_to_o = 0.55 + 0.02 * run_index
    s_to_o = 0.02 + 0.001 * run_index
    return {
        "metallicity_log10": metallicity_log10,
        "c_to_o": c_to_o,
        "s_to_o": s_to_o,
        "He_H": 8.38e-2,
        "C_H": oxygen_h * c_to_o,
        "O_H": oxygen_h,
        "N_H": 7.08e-5 * (10.0 ** metallicity_log10),
        "S_H": oxygen_h * s_to_o,
    }


def _temperature_profile(pressure_bar: np.ndarray, run_index: int) -> np.ndarray:
    """Build one deterministic temperature profile for a synthetic raw run."""
    base = 850.0 + 90.0 * run_index
    span = 700.0 + 40.0 * run_index
    exponent = 0.16 + 0.02 * run_index
    return base + span * (pressure_bar / pressure_bar[0]) ** exponent


def _run_metadata(run_index: int) -> dict[str, float | bool | str]:
    """Return deterministic raw metadata spanning all plot-profile buckets."""
    if run_index in {1, 4}:
        return {
            "temperature_profile_source": "pt_library",
            "temperature_profile_Teq": 1400.0 + 200.0 * run_index,
            "temperature_profile_LogMet": 0.1 * run_index,
            "temperature_profile_LogDrag": float(run_index),
            "temperature_profile_Mstar": 0.8 + 0.1 * run_index,
            "temperature_profile_Rp": 1.3,
            "temperature_profile_logG": 1.0 + 0.1 * run_index,
            "temperature_profile_lon": 10.0 * run_index,
            "temperature_profile_lat": -5.0 * run_index,
        }
    return {
        "temperature_profile_source": "analytic",
        "temperature_profile_analytic_t_int_k": 350.0 + 15.0 * run_index,
        "temperature_profile_analytic_t_irr_k": 1500.0 + 60.0 * run_index,
        "temperature_profile_analytic_log10_kappa_ir_m2_kg": -2.5 + 0.1 * run_index,
        "temperature_profile_analytic_log10_gamma_1": -0.8 + 0.05 * run_index,
        "temperature_profile_analytic_log10_gamma_2": 0.1 + 0.04 * run_index,
        "temperature_profile_analytic_alpha": 0.25 + 0.05 * run_index,
        "temperature_profile_analytic_temperature_shift_k": -50.0 + 20.0 * run_index,
        "temperature_profile_analytic_convective_adjustment_applied": run_index == 3,
        "temperature_profile_analytic_adiabatic_gradient": 0.29 + 0.01 * run_index,
    }


def _equilibrium_state(
    *,
    pressure_bar: np.ndarray,
    output_species: list[str],
    run_index: int,
) -> np.ndarray:
    """Build one deterministic equilibrium abundance table."""
    nz = pressure_bar.size
    profile = np.full((nz, len(output_species)), 1.0e-12, dtype=np.float64)
    profile[:, output_species.index("H2")] = 0.82 - 0.01 * run_index
    profile[:, output_species.index("He")] = 0.16
    profile[:, output_species.index("H2O")] = 8.0e-4 * (1.0 + 0.1 * run_index)
    profile[:, output_species.index("CO")] = 4.0e-4 * (1.0 + 0.12 * run_index)
    profile[:, output_species.index("CO2")] = 1.0e-5 * (1.0 + 0.08 * run_index)
    profile[:, output_species.index("CH4")] = 2.0e-6 * (1.0 + 0.05 * run_index)
    profile[:, output_species.index("NH3")] = 5.0e-7 * (1.0 + 0.03 * run_index)
    profile[:, output_species.index("H2S")] = 4.0e-7 * (1.0 + 0.04 * run_index)
    profile /= np.sum(profile, axis=1, keepdims=True)
    return profile


def _write_fastchem_raw_dataset(config: dict) -> None:
    """Create one tiny deterministic raw FastChem dataset for extras tests."""
    raw_root = Path(config["paths"]["raw_root"])
    runs_dir = raw_root / "runs_tmp"
    runs_dir.mkdir(parents=True, exist_ok=True)

    pressure_bar = np.logspace(
        np.log10(float(config["sampling"]["pressure_bottom_bar"])),
        np.log10(float(config["sampling"]["pressure_top_bar"])),
        int(config["sampling"]["num_levels"]),
        dtype=np.float64,
    )
    output_species = list(config["data_spec"]["output_species"])
    gravity_profile = np.full(pressure_bar.shape, 2.5e3, dtype=np.float64)

    run_files: list[Path] = []
    for run_index in range(6):
        globals_map = _element_globals(run_index)
        element_profile = np.repeat(
            np.array(
                [[float(globals_map[name]) for name in ELEMENT_INPUT_ORDER]],
                dtype=np.float64,
            ),
            pressure_bar.size,
            axis=0,
        )
        spec = RunSpecification(
            run_id=f"run_{run_index:05d}",
            pressure_bar=pressure_bar,
            temperature_k=_temperature_profile(pressure_bar, run_index),
            globals=globals_map,
            metadata=_run_metadata(run_index),
            elemental_abundances_x_h=element_profile,
            gravity_cm_s2=gravity_profile,
        )
        run_path = runs_dir / f"{spec.run_id}.h5"
        write_equilibrium_hdf5(
            run_path,
            spec=spec,
            equilibrium_ymix=_equilibrium_state(
                pressure_bar=pressure_bar,
                output_species=output_species,
                run_index=run_index,
            ),
            state_species=list(config["data_spec"]["state_species"]),
            output_species=output_species,
        )
        run_files.append(run_path)

    raw_root.mkdir(parents=True, exist_ok=True)
    consolidate_runs_to_single_hdf5(run_files, raw_root / "runs.h5")


def _export_fastchem_bundle(
    *,
    config: dict,
    normalization: dict,
    data_contract: dict,
    output_path: Path,
) -> Path:
    """Export one small transformer bundle wired to the synthetic dataset."""
    dims = TransformerDimensions(
        sequence_dim=2,
        global_dim=len(ELEMENT_INPUT_ORDER),
        spectrum_max_tokens=0,
        target_dim=len(data_contract["output_species_order"]),
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        conditioning_hidden_dim=16,
        film_clamp=1.5,
        output_head_divisor=2,
        spectrum_latent_dim=0,
        spectrum_hidden_dim=0,
        spectrum_num_latents=0,
        spectrum_num_layers=0,
        spectrum_num_heads=1,
        spectrum_fourier_features=0,
        spectrum_encoder_mode="none",
        spectrum_floor=1.0e-30,
        activation="gelu",
    )
    params = init_transformer_params(jax.random.PRNGKey(0), dims)
    bundle_config = copy.deepcopy(config)
    bundle_config.pop("_project_root", None)
    bundle_config.pop("_roth_profile_cache", None)
    payload = {
        "params": jax.tree_util.tree_map(np.asarray, params),
        "model_dimensions": dims.to_dict(),
        "normalization": normalization,
        "data_contract": data_contract,
        "config": bundle_config,
    }
    return export_checkpoint_payload(payload, output_path)


def _build_fastchem_extras_fixture(tmp_path: Path) -> tuple[dict, Path]:
    """Create a tiny raw/processed FastChem dataset plus a matching bundle."""
    config = _make_fastchem_extras_config(tmp_path)
    _write_fastchem_raw_dataset(config)
    artifact = preprocess_equilibrium_dataset(config, project_root=ROOT)
    bundle_path = _export_fastchem_bundle(
        config=config,
        normalization=artifact["normalization"],
        data_contract=artifact["data_contract"],
        output_path=Path(config["paths"]["checkpoints_root"]) / "best_exported.npz",
    )
    return config, bundle_path


def _bundle_config(config: dict) -> dict:
    """Return the JSON-serializable config stored inside the exported bundle."""
    payload = copy.deepcopy(config)
    payload.pop("_project_root", None)
    payload.pop("_roth_profile_cache", None)
    return payload


def test_fastchem_test_case_inverse_normalization_matches_raw_run(tmp_path):
    config, bundle_path = _build_fastchem_extras_fixture(tmp_path)
    context = extras_common.load_fastchem_test_context(
        ROOT,
        bundle_path=bundle_path,
        config=_bundle_config(config),
    )
    run_id = context.split.run_ids[0]
    test_case = extras_common.load_fastchem_test_case(context, run_id=run_id)

    with h5py.File(Path(config["paths"]["raw_root"]) / "runs.h5", "r") as handle:
        raw_run = load_raw_equilibrium_run(handle[run_id], config=config, run_id=run_id)

    np.testing.assert_allclose(test_case.pressure_bar, raw_run.pressure_bar, rtol=1.0e-5, atol=1.0e-8)
    np.testing.assert_allclose(test_case.temperature_k, raw_run.temperature_k, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(
        test_case.stored_target_ymix,
        raw_run.equilibrium_ymix,
        rtol=3.0e-5,
        atol=1.0e-12,
    )
    for name in context.contract["global_static_feature_order"]:
        assert test_case.global_inputs[name] == pytest.approx(raw_run.globals[name], rel=1.0e-5)


def test_select_fastchem_test_run_id_supports_random_and_explicit_saved_runs(tmp_path):
    config, bundle_path = _build_fastchem_extras_fixture(tmp_path)
    context = extras_common.load_fastchem_test_context(
        ROOT,
        bundle_path=bundle_path,
        config=_bundle_config(config),
    )

    selected = extras_common.select_fastchem_test_run_id(
        context,
        run_id=None,
        rng=np.random.default_rng(0),
    )
    assert selected in context.split.run_ids

    explicit = context.split.run_ids[0]
    assert extras_common.select_fastchem_test_run_id(context, run_id=explicit) == explicit


def test_saved_test_runs_are_bucketed_from_raw_metadata(tmp_path):
    config, bundle_path = _build_fastchem_extras_fixture(tmp_path)
    context = extras_common.load_fastchem_test_context(
        ROOT,
        bundle_path=bundle_path,
        config=_bundle_config(config),
        require_raw=True,
    )
    assert context.raw_root is not None

    metadata_map = extras_common.load_fastchem_raw_metadata_map(context.raw_root, context.split.run_ids)
    buckets = {
        run_id: extras_common.classify_temperature_profile_bucket(metadata_map[run_id])
        for run_id in context.split.run_ids
    }

    assert set(context.split.run_ids) == {"run_00001", "run_00003", "run_00005"}
    assert buckets["run_00001"] == "pt_library"
    assert buckets["run_00003"] == "analytic_convective"
    assert buckets["run_00005"] == "analytic_radiative"


def test_fastchem_saved_test_profile_demo_only_writes_main_mixing_ratio_figure(tmp_path):
    config, bundle_path = _build_fastchem_extras_fixture(tmp_path)
    context = extras_common.load_fastchem_test_context(
        ROOT,
        bundle_path=bundle_path,
        config=_bundle_config(config),
    )
    output_dir = tmp_path / "plots"
    exit_code = fastchem_saved_test_profile_demo_module.main(
        [
            "--bundle",
            str(bundle_path),
            "--run-id",
            context.split.run_ids[0],
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "01_mixing_ratios.png").is_file()
    assert not (output_dir / "02_h2o_temperature_sensitivity.png").exists()
    assert not (output_dir / "03_jacobian_global_params.png").exists()
