"""VULCAN subprocess execution and raw trajectory extraction.

Manages the lifecycle of individual VULCAN chemistry runs:

1. **Preflight and compatibility checks**: Validates the VULCAN source tree,
   checks configured species against ``chem_funs.py``, and runs a one-shot
   smoke test.
2. **Worker isolation**: Each process gets its own copy of the VULCAN source
   tree to avoid file conflicts during parallel execution.
3. **Config generation**: Patches ``vulcan_cfg.py`` with sampled parameters
   (P, T, Kzz, gravity, abundances, physics toggles) via regex replacement.
4. **Execution**: Runs VULCAN as a subprocess with timeout enforcement.
5. **Extraction**: Loads the ``.vul`` pickle output, converts number densities
   to mixing ratios (``ymix = n_i / sum(n)``), deduplicates time steps, and
   writes the validated trajectory into a standardized HDF5 layout.

Current physics coverage: non-photochemical VULCAN thermochemistry plus
configurable transport, boundary conditions, condensation/settling, cold-trap,
and selected solver/runtime toggles. Photochemistry and ion chemistry remain
intentionally disabled in this emulator branch (see ``spec.md`` for the full
coverage map and all exposed toggles).
"""

from __future__ import annotations

import ast
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

from sampling import RunSpec


class VulcanRuntimeError(RuntimeError):
    """Raised when a VULCAN run or extraction step fails."""


@dataclass(frozen=True)
class BoundaryConditionSettings:
    """Explicit boundary-condition settings passed through to generated VULCAN configs."""

    use_topflux: bool
    use_botflux: bool
    top_BC_flux_file: str
    bot_BC_flux_file: str
    use_fix_sp_bot: dict[str, float]


@dataclass(frozen=True)
class SpeciesSelection:
    """Ordered species lists used for model state and supervised outputs."""

    state_species: tuple[str, ...]
    output_species: tuple[str, ...]


@dataclass(frozen=True)
class WorkerSettings:
    """Immutable worker/runtime settings shared across VULCAN tasks."""

    vulcan_source: str
    worker_root: str
    runs_root: str
    species: SpeciesSelection
    boundary_conditions: BoundaryConditionSettings | None
    use_eddy_diffusion: bool
    use_molecular_diffusion: bool
    use_upwind_molecular_diffusion: bool
    use_condensation: bool
    use_settling: bool
    use_initial_cold_trap: bool
    use_sat_surface_h2o: bool
    use_lowT_limit_rates: bool
    use_adaptive_rtol: bool
    ini_mix: str
    atm_base: str
    save_evo_frq: int
    keep_vulcan_outputs_debug: bool
    run_timeout_seconds: int
    num_workers: int
    runtime: float
    dt_min: float
    dt_max: float
    count_max: int
    trun_min: float
    count_min: int


@dataclass(frozen=True)
class RunResult:
    """Result record for one successfully extracted run."""

    run_id: int
    run_file: str


_WORKER_CACHE: dict[str, Any] = {}


def _load_available_species(vulcan_source: Path) -> list[str]:
    """Parse the bundled VULCAN species list from ``chem_funs.py``."""
    chem_funs_path = vulcan_source / "chem_funs.py"
    if not chem_funs_path.is_file():
        raise VulcanRuntimeError(
            "VULCAN runtime is missing chem_funs.py, which is required to validate "
            f"configured species: {chem_funs_path}"
        )

    text = chem_funs_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(chem_funs_path))
    except SyntaxError as exc:
        raise VulcanRuntimeError(
            f"Failed to parse VULCAN chemistry file: {chem_funs_path}."
        ) from exc

    parsed: Any | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "spec_list":
                    parsed = ast.literal_eval(node.value)
                    break
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "spec_list":
                parsed = ast.literal_eval(node.value)
        if parsed is not None:
            break

    if parsed is None:
        raise VulcanRuntimeError(
            "Failed to parse VULCAN species list from chem_funs.py. Expected a top-level "
            f"spec_list assignment in {chem_funs_path}."
        )

    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise VulcanRuntimeError(
            f"Invalid VULCAN species list in {chem_funs_path}; expected a list[str]."
        )
    return parsed


def validate_species_available(
    vulcan_source: Path,
    *,
    state_species: tuple[str, ...],
    output_species: tuple[str, ...],
) -> None:
    """Fail fast when configured species are not present in the VULCAN network."""
    available = set(_load_available_species(vulcan_source))
    missing_state = [species for species in state_species if species not in available]
    missing_output = [species for species in output_species if species not in available]
    if missing_state or missing_output:
        parts: list[str] = []
        if missing_state:
            parts.append(f"state_species missing={missing_state}")
        if missing_output:
            parts.append(f"output_species missing={missing_output}")
        joined = "; ".join(parts)
        raise VulcanRuntimeError(
            "Configured species are not available in the bundled VULCAN chemistry network: "
            f"{joined}. Update data_spec state/output species or regenerate the chemistry network."
        )


def resolve_boundary_conditions(
    config: dict[str, Any],
    vulcan_source: Path,
) -> BoundaryConditionSettings | None:
    """Resolve and validate optional boundary-condition settings against the VULCAN tree."""
    if not bool(config["physics_toggles"]["use_boundary_conditions"]):
        return None

    boundary_cfg = config.get("boundary_conditions")
    if not isinstance(boundary_cfg, dict):
        raise VulcanRuntimeError(
            "physics_toggles.use_boundary_conditions=true requires a top-level "
            "'boundary_conditions' section."
        )

    use_topflux = bool(boundary_cfg["use_topflux"])
    use_botflux = bool(boundary_cfg["use_botflux"])
    top_file = str(boundary_cfg["top_BC_flux_file"])
    bot_file = str(boundary_cfg["bot_BC_flux_file"])
    fixed_bottom = {
        str(species_name): float(value)
        for species_name, value in dict(boundary_cfg["use_fix_sp_bot"]).items()
    }

    if not use_topflux and not use_botflux and not fixed_bottom:
        raise VulcanRuntimeError(
            "Boundary conditions are enabled, but no top flux, bottom flux, or fixed bottom "
            "mixing ratios were configured."
        )

    if use_topflux and not (vulcan_source / top_file).is_file():
        raise VulcanRuntimeError(
            "Configured boundary condition file does not exist: "
            f"{vulcan_source / top_file}"
        )
    if use_botflux and not (vulcan_source / bot_file).is_file():
        raise VulcanRuntimeError(
            "Configured boundary condition file does not exist: "
            f"{vulcan_source / bot_file}"
        )

    return BoundaryConditionSettings(
        use_topflux=use_topflux,
        use_botflux=use_botflux,
        top_BC_flux_file=top_file,
        bot_BC_flux_file=bot_file,
        use_fix_sp_bot=fixed_bottom,
    )


def _build_preflight_run_spec() -> RunSpec:
    """Return a tiny deterministic run spec for runtime smoke validation."""
    pressure_bar = np.logspace(3.0, -8.0, 16, dtype=np.float64)
    temperature_k = np.full(pressure_bar.shape, 1400.0, dtype=np.float64)
    kzz_cm2_s = np.full(pressure_bar.shape, 1.0e10, dtype=np.float64)
    return RunSpec(
        run_id=-1,
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        kzz_cm2_s=kzz_cm2_s,
        gravity_cm_s2=1.0e3,
        metallicity_log10=0.0,
        c_to_o=0.55,
        abundances={
            "O_H": 5.37e-4,
            "C_H": 2.95e-4,
            "N_H": 7.08e-5,
            "S_H": 1.41e-5,
            "He_H": 8.38e-2,
            "fastchem_met_scale": 1.0,
        },
        tp_params={},
        kzz_params={},
    )


def _run_preflight_smoke(
    *,
    vulcan_source: Path,
    settings: WorkerSettings,
    timeout_seconds: int,
) -> None:
    """Run a one-shot VULCAN smoke test inside an isolated worker copy."""
    preflight_spec = _build_preflight_run_spec()

    with tempfile.TemporaryDirectory(prefix="vulcan_preflight_") as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        worker_dir = tmpdir / "worker"
        shutil.copytree(vulcan_source, worker_dir)
        _patch_op_convergence_compat(worker_dir)

        generated_atm_dir = worker_dir / "atm" / "generated"
        generated_atm_dir.mkdir(parents=True, exist_ok=True)
        atm_relpath = "atm/generated/preflight_profile.txt"
        _write_generated_atm_file(preflight_spec, generated_atm_dir / "preflight_profile.txt")

        baseline_cfg = (worker_dir / "vulcan_cfg.py").read_text(encoding="utf-8")
        smoke_settings = replace(
            settings,
            vulcan_source=str(vulcan_source),
            worker_root=str(tmpdir),
            runs_root=str(tmpdir / "runs"),
            save_evo_frq=1,
            keep_vulcan_outputs_debug=False,
            run_timeout_seconds=timeout_seconds,
            num_workers=1,
            runtime=1.0e-8,
            dt_min=1.0e-14,
            dt_max=1.0e-8,
            count_max=1,
            trun_min=1.0,
            count_min=10,
        )
        smoke_cfg = _apply_run_config(
            baseline_cfg=baseline_cfg,
            run_spec=preflight_spec,
            settings=smoke_settings,
            atm_relpath=atm_relpath,
            out_name="preflight_smoke.vul",
        )
        smoke_cfg = _replace_assignment(smoke_cfg, "save_evolution", False)
        (worker_dir / "vulcan_cfg.py").write_text(smoke_cfg, encoding="utf-8")

        try:
            completed = subprocess.run(
                [sys.executable, "vulcan.py", "-n"],
                cwd=worker_dir,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise VulcanRuntimeError(
                "VULCAN preflight failed: smoke execution timed out after "
                f"{timeout_seconds} seconds."
            ) from exc
        if completed.returncode != 0:
            stderr_tail = (completed.stderr or "")[-4000:]
            stdout_tail = (completed.stdout or "")[-4000:]
            raise VulcanRuntimeError(
                "VULCAN preflight failed: smoke execution did not complete cleanly "
                f"(exit code {completed.returncode}).\n"
                f"STDOUT tail:\n{stdout_tail}\n"
                f"STDERR tail:\n{stderr_tail}"
            )

        output_file = worker_dir / "output" / "preflight_smoke.vul"
        if not output_file.is_file():
            raise VulcanRuntimeError(
                "VULCAN preflight failed: smoke execution did not produce the expected output "
                f"file {output_file}."
            )



def preflight_vulcan_source(
    vulcan_source: Path,
    *,
    settings: WorkerSettings,
    timeout_seconds: int = 120,
) -> None:
    """Verify VULCAN runtime prerequisites without auto-install behavior."""
    fastchem_binary = vulcan_source / "fastchem_vulcan" / "fastchem"
    required_paths = [
        vulcan_source,
        vulcan_source / "vulcan.py",
        vulcan_source / "vulcan_cfg.py",
        vulcan_source / "fastchem_vulcan",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise VulcanRuntimeError(
            "VULCAN preflight failed. Missing required runtime paths: "
            f"{missing}. Install/build VULCAN before running --gen."
        )
    if not fastchem_binary.is_file():
        raise VulcanRuntimeError(
            "VULCAN preflight failed. Missing compiled FastChem binary: "
            f"{fastchem_binary}. Build FastChem inside VULCAN before running --gen."
        )
    if not os.access(fastchem_binary, os.X_OK):
        raise VulcanRuntimeError(
            "VULCAN preflight failed. FastChem binary is not executable: "
            f"{fastchem_binary}. Fix permissions or rebuild FastChem before running --gen."
        )

    _run_preflight_smoke(
        vulcan_source=vulcan_source,
        settings=settings,
        timeout_seconds=timeout_seconds,
    )


def _py_literal(value: Any) -> str:
    """Render one Python literal exactly as it should appear in ``vulcan_cfg.py``."""
    if isinstance(value, bool):
        return "True" if value else "False"
    return repr(value)


def _replace_assignment(text: str, key: str, value: Any) -> str:
    """Replace one top-level assignment in a copied ``vulcan_cfg.py`` file."""
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=).*$", re.MULTILINE)
    replacement = rf"\1 {_py_literal(value)}"
    replaced, count = pattern.subn(replacement, text)
    if count == 0:
        raise VulcanRuntimeError(f"Failed to set '{key}' in generated vulcan_cfg.py")
    return replaced


def _patch_op_convergence_compat(worker_dir: Path) -> None:
    """Patch the worker copy of ``op.py`` for numpy array compatibility.

    Some VULCAN versions assign ``t_time = var.t_time`` (a Python list) in the
    convergence checker without converting to a numpy array first, which causes
    a ``TypeError`` on ``list - float``.  This adds the missing ``np.asarray``
    conversion when needed.  The patch is a no-op if the conversion already
    exists.
    """
    op_path = worker_dir / "op.py"
    if not op_path.is_file():
        return
    text = op_path.read_text(encoding="utf-8")
    if "np.asarray(var.t_time" in text:
        return
    patched = re.sub(
        r"(t_time\s*=\s*)var\.t_time\b(?!\s*[.\[])",
        r"\1np.asarray(var.t_time, dtype=float)",
        text,
    )
    patched = re.sub(
        r"(y_time\s*=\s*)var\.y_time\b(?!\s*[.\[])",
        r"\1np.asarray(var.y_time)",
        patched,
    )
    if patched != text:
        op_path.write_text(patched, encoding="utf-8")


def _ensure_worker_context(settings: WorkerSettings) -> dict[str, Any]:
    """Create or reuse the per-process copied VULCAN worker tree."""
    cache_key = "context"
    existing = _WORKER_CACHE.get(cache_key)
    if existing is not None:
        return existing

    pid = os.getpid()
    source = Path(settings.vulcan_source)
    worker_root = Path(settings.worker_root)
    worker_root.mkdir(parents=True, exist_ok=True)
    worker_dir = worker_root / f"worker_{pid}"

    if worker_dir.exists():
        shutil.rmtree(worker_dir)
    shutil.copytree(source, worker_dir)
    _patch_op_convergence_compat(worker_dir)

    baseline_cfg_path = worker_dir / "vulcan_cfg.py"
    baseline_cfg_text = baseline_cfg_path.read_text(encoding="utf-8")

    generated_atm_dir = worker_dir / "atm" / "generated"
    generated_atm_dir.mkdir(parents=True, exist_ok=True)

    context = {
        "worker_dir": worker_dir,
        "baseline_cfg_text": baseline_cfg_text,
        "generated_atm_dir": generated_atm_dir,
    }
    _WORKER_CACHE[cache_key] = context
    return context


def _write_generated_atm_file(run_spec: RunSpec, atm_file: Path) -> None:
    """Write one sampled atmosphere profile in the VULCAN text input format."""
    pressure_dyn_cm2 = run_spec.pressure_bar * 1.0e6
    with atm_file.open("w", encoding="utf-8") as handle:
        handle.write("#(dyne/cm2) (K) (cm2/s)\n")
        handle.write("Pressure\tTemp\tKzz\n")
        for p_dyn, temp, kzz in zip(
            pressure_dyn_cm2,
            run_spec.temperature_k,
            run_spec.kzz_cm2_s,
            strict=True,
        ):
            handle.write(f"{p_dyn:.6E}\t{temp:.6f}\t{kzz:.6E}\n")


def _apply_run_config(
    baseline_cfg: str,
    run_spec: RunSpec,
    settings: WorkerSettings,
    atm_relpath: str,
    out_name: str,
) -> str:
    """Apply one sampled run configuration to the copied VULCAN config template."""
    cfg = baseline_cfg
    replacements: dict[str, Any] = {
        "use_lowT_limit_rates": bool(settings.use_lowT_limit_rates),
        "use_photo": False,
        "use_ion": False,
        "use_live_plot": False,
        "use_live_flux": False,
        "use_plot_end": False,
        "use_plot_evo": False,
        "use_save_movie": False,
        "use_flux_movie": False,
        "use_print_prog": False,
        "output_humanread": False,
        "save_evolution": True,
        "save_evo_frq": int(settings.save_evo_frq),
        "runtime": float(settings.runtime),
        "dt_min": float(settings.dt_min),
        "dt_max": float(settings.dt_max),
        "count_max": int(settings.count_max),
        "trun_min": float(settings.trun_min),
        "count_min": int(settings.count_min),
        "ini_mix": str(settings.ini_mix),
        "use_ini_cold_trap": bool(settings.use_initial_cold_trap),
        "atm_base": str(settings.atm_base),
        "use_Kzz": bool(settings.use_eddy_diffusion),
        "use_moldiff": bool(settings.use_molecular_diffusion),
        "use_vm_mol": bool(settings.use_upwind_molecular_diffusion),
        "use_vz": False,
        "atm_type": "file",
        "Kzz_prof": "file",
        "vz_prof": "const",
        "const_vz": 0.0,
        "atm_file": atm_relpath,
        "out_name": out_name,
        "output_dir": "output/",
        "plot_dir": "plot/",
        "movie_dir": "plot/movie/",
        "use_solar": False,
        "use_topflux": False,
        "use_botflux": False,
        "use_fix_sp_bot": {},
        "use_sat_surfaceH2O": bool(settings.use_sat_surface_h2o),
        "use_condense": bool(settings.use_condensation),
        "use_settling": bool(settings.use_settling),
        "use_adapt_rtol": bool(settings.use_adaptive_rtol),
        "nz": int(run_spec.pressure_bar.size),
        "P_b": float(np.max(run_spec.pressure_bar) * 1.0e6),
        "P_t": float(np.min(run_spec.pressure_bar) * 1.0e6),
        "gs": float(run_spec.gravity_cm_s2),
        "O_H": float(run_spec.abundances["O_H"]),
        "C_H": float(run_spec.abundances["C_H"]),
        "N_H": float(run_spec.abundances["N_H"]),
        "S_H": float(run_spec.abundances["S_H"]),
        "He_H": float(run_spec.abundances["He_H"]),
        "fastchem_met_scale": float(run_spec.abundances["fastchem_met_scale"]),
    }

    if settings.boundary_conditions is not None:
        replacements["use_topflux"] = bool(settings.boundary_conditions.use_topflux)
        replacements["use_botflux"] = bool(settings.boundary_conditions.use_botflux)
        replacements["top_BC_flux_file"] = settings.boundary_conditions.top_BC_flux_file
        replacements["bot_BC_flux_file"] = settings.boundary_conditions.bot_BC_flux_file
        replacements["use_fix_sp_bot"] = dict(settings.boundary_conditions.use_fix_sp_bot)

    for key, value in replacements.items():
        cfg = _replace_assignment(cfg, key, value)
    return cfg


def _deduplicate_strictly_increasing(
    times: np.ndarray,
    states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the first state and subsequent strictly increasing saved times only."""
    if times.ndim != 1 or states.ndim != 3:
        raise VulcanRuntimeError("Invalid trajectory ranks in VULCAN output.")
    if states.shape[0] != times.size:
        raise VulcanRuntimeError("Trajectory time/state length mismatch in VULCAN output.")

    keep_indices = [0]
    last_time = float(times[0])
    for idx in range(1, times.size):
        candidate = float(times[idx])
        if candidate > last_time:
            keep_indices.append(idx)
            last_time = candidate
    keep = np.asarray(keep_indices, dtype=np.int64)
    return times[keep], states[keep]


def _extract_run_payload(output_file: Path, run_spec: RunSpec, settings: WorkerSettings) -> dict[str, Any]:
    """Extract the raw full-trajectory payload required by the transition-model pipeline."""
    with output_file.open("rb") as handle:
        data = pickle.load(handle)  # noqa: S301 - trusted, self-generated VULCAN artifact

    variable = data.get("variable")
    if not isinstance(variable, dict):
        raise VulcanRuntimeError("Invalid .vul structure: missing 'variable' dictionary.")

    species = list(variable.get("species", []))
    if not species:
        raise VulcanRuntimeError("Invalid .vul structure: missing species list.")

    state_species = list(settings.species.state_species)
    output_species = list(settings.species.output_species)
    missing_state = [sp for sp in state_species if sp not in species]
    missing_output = [sp for sp in output_species if sp not in species]
    if missing_state or missing_output:
        parts: list[str] = []
        if missing_state:
            parts.append(f"state_species={missing_state}")
        if missing_output:
            parts.append(f"output_species={missing_output}")
        raise VulcanRuntimeError(
            f"Configured species missing in run {run_spec.run_id}: {'; '.join(parts)}"
        )

    y_ini = np.asarray(variable.get("y_ini"), dtype=np.float64)
    y_time = np.asarray(variable.get("y_time"), dtype=np.float64)
    t_time = np.asarray(variable.get("t_time"), dtype=np.float64)
    if y_ini.ndim != 2:
        raise VulcanRuntimeError("Invalid y_ini shape in .vul output.")
    if y_time.ndim != 3 or t_time.ndim != 1:
        raise VulcanRuntimeError(
            "save_evolution output missing or malformed (expected y_time[t,z,s], t_time[t])."
        )
    if y_time.shape[0] != t_time.shape[0]:
        raise VulcanRuntimeError("Inconsistent y_time/t_time lengths in .vul output.")
    if y_ini.shape[0] != run_spec.pressure_bar.size:
        raise VulcanRuntimeError("y_ini vertical dimension does not match the sampled atmosphere.")
    if y_ini.shape[1] != len(species):
        raise VulcanRuntimeError("y_ini species dimension does not match the VULCAN species list.")
    if y_time.shape[1:] != y_ini.shape:
        raise VulcanRuntimeError("y_time state shape does not match y_ini shape.")

    states = np.concatenate([y_ini[None, ...], y_time], axis=0)
    times = np.concatenate([np.array([0.0], dtype=np.float64), t_time], axis=0)
    times, states = _deduplicate_strictly_increasing(times, states)
    if times.size < 2:
        raise VulcanRuntimeError(
            "VULCAN output does not contain at least one positive saved time after deduplication."
        )
    if float(times[0]) != 0.0:
        raise VulcanRuntimeError("Extracted trajectory must start at t=0.")

    # Convert number densities to mixing ratios: ymix_i = n_i / sum(n_all)
    row_sums = np.sum(states, axis=2, keepdims=True)
    if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
        raise VulcanRuntimeError("Encountered non-positive or non-finite total number density.")
    ymix = states / row_sums

    state_indices = np.asarray([species.index(sp) for sp in state_species], dtype=np.int64)
    output_indices = np.asarray([species.index(sp) for sp in output_species], dtype=np.int64)
    ymix_state = np.take(ymix, state_indices, axis=2)
    ymix_output = np.take(ymix, output_indices, axis=2)

    arrays_to_check = [
        run_spec.pressure_bar,
        run_spec.temperature_k,
        run_spec.kzz_cm2_s,
        times,
        ymix_state,
        ymix_output,
    ]
    if any(np.any(~np.isfinite(array)) for array in arrays_to_check):
        raise VulcanRuntimeError("Non-finite values detected in extracted run payload.")

    return {
        "pressure_bar": np.asarray(run_spec.pressure_bar, dtype=np.float64),
        "temperature_k": np.asarray(run_spec.temperature_k, dtype=np.float64),
        "kzz_cm2_s": np.asarray(run_spec.kzz_cm2_s, dtype=np.float64),
        "time_s": np.asarray(times, dtype=np.float64),
        "ymix_state": np.asarray(ymix_state, dtype=np.float64),
        "ymix_output": np.asarray(ymix_output, dtype=np.float64),
        "state_species": state_species,
        "output_species": output_species,
    }


def _write_run_hdf5(run_path: Path, run_spec: RunSpec, payload: dict[str, Any]) -> None:
    """Persist one extracted full trajectory into the raw HDF5 contract layout."""
    run_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(run_path, "w") as handle:
        handle.attrs["run_id"] = int(run_spec.run_id)

        inputs = handle.create_group("inputs")
        inputs.create_dataset("pressure_bar", data=payload["pressure_bar"])
        inputs.create_dataset("temperature_k", data=payload["temperature_k"])
        inputs.create_dataset("kzz_cm2_s", data=payload["kzz_cm2_s"])
        inputs.create_dataset(
            "state_species",
            data=np.asarray(payload["state_species"], dtype=str_dtype),
        )
        inputs.create_dataset(
            "output_species",
            data=np.asarray(payload["output_species"], dtype=str_dtype),
        )

        globals_group = handle.create_group("globals")
        globals_group.create_dataset("gravity_cm_s2", data=np.float64(run_spec.gravity_cm_s2))
        globals_group.create_dataset(
            "metallicity_log10",
            data=np.float64(run_spec.metallicity_log10),
        )
        globals_group.create_dataset("c_to_o", data=np.float64(run_spec.c_to_o))

        trajectory = handle.create_group("trajectory")
        trajectory.create_dataset("time_s", data=payload["time_s"])
        trajectory.create_dataset("ymix_state", data=payload["ymix_state"])
        trajectory.create_dataset("ymix_output", data=payload["ymix_output"])

        sampler = handle.create_group("sampler")
        for key, value in run_spec.tp_params.items():
            sampler.create_dataset(f"tp_{key}", data=np.float64(value))
        for key, value in run_spec.kzz_params.items():
            sampler.create_dataset(f"kzz_{key}", data=np.float64(value))


def _run_single(spec: RunSpec, settings: WorkerSettings) -> RunResult:
    """Execute one sampled VULCAN run end-to-end inside one worker copy."""
    context = _ensure_worker_context(settings)
    worker_dir: Path = context["worker_dir"]
    baseline_cfg: str = context["baseline_cfg_text"]
    generated_atm_dir: Path = context["generated_atm_dir"]

    atm_filename = f"run_{spec.run_id:06d}.txt"
    atm_relpath = f"atm/generated/{atm_filename}"
    out_name = f"run_{spec.run_id:06d}.vul"

    atm_path = generated_atm_dir / atm_filename
    _write_generated_atm_file(spec, atm_path)

    cfg_text = _apply_run_config(
        baseline_cfg=baseline_cfg,
        run_spec=spec,
        settings=settings,
        atm_relpath=atm_relpath,
        out_name=out_name,
    )
    (worker_dir / "vulcan_cfg.py").write_text(cfg_text, encoding="utf-8")

    try:
        completed = subprocess.run(
            [sys.executable, "vulcan.py", "-n"],
            cwd=worker_dir,
            text=True,
            capture_output=True,
            timeout=int(settings.run_timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise VulcanRuntimeError(
            f"VULCAN run timed out for run_id={spec.run_id} after "
            f"{settings.run_timeout_seconds} seconds."
        ) from exc
    if completed.returncode != 0:
        stderr_tail = (completed.stderr or "")[-4000:]
        stdout_tail = (completed.stdout or "")[-4000:]
        raise VulcanRuntimeError(
            f"VULCAN run failed for run_id={spec.run_id} with exit code {completed.returncode}.\n"
            f"STDOUT tail:\n{stdout_tail}\nSTDERR tail:\n{stderr_tail}"
        )

    output_file = worker_dir / "output" / out_name
    if not output_file.is_file():
        raise VulcanRuntimeError(f"Expected VULCAN output file not found: {output_file}")

    payload = _extract_run_payload(output_file, spec, settings)
    run_file = Path(settings.runs_root) / f"run_{spec.run_id:06d}.h5"
    _write_run_hdf5(run_file, spec, payload)

    if not settings.keep_vulcan_outputs_debug:
        output_file.unlink(missing_ok=True)

    return RunResult(run_id=spec.run_id, run_file=str(run_file))


def run_vulcan_jobs(
    run_specs: list[RunSpec],
    *,
    settings: WorkerSettings,
) -> list[RunResult]:
    """Run all VULCAN jobs with fail-fast policy."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if not run_specs:
        raise VulcanRuntimeError("No run specs provided to VULCAN runner.")
    if int(settings.num_workers) <= 0:
        raise VulcanRuntimeError("settings.num_workers must be > 0.")

    runs_root = Path(settings.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    try:
        if int(settings.num_workers) == 1:
            for spec in tqdm(run_specs, desc="VULCAN runs", unit="run"):
                results.append(_run_single(spec, settings))
            return sorted(results, key=lambda item: item.run_id)

        # Fan out runs across worker processes; each worker gets an isolated VULCAN tree copy.
        # Fail-fast: cancel remaining futures on the first worker error.
        with ProcessPoolExecutor(max_workers=int(settings.num_workers)) as pool:
            futures = {pool.submit(_run_single, spec, settings): spec.run_id for spec in run_specs}
            try:
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="VULCAN runs",
                    unit="run",
                ):
                    results.append(future.result())
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                raise VulcanRuntimeError(
                    f"Generation aborted due to worker failure: {exc}"
                ) from exc
        return sorted(results, key=lambda item: item.run_id)
    finally:
        _cleanup_worker_dirs(settings)


def _cleanup_worker_dirs(settings: WorkerSettings) -> None:
    """Remove copied worker trees unless debug retention is enabled."""
    if settings.keep_vulcan_outputs_debug:
        return
    _WORKER_CACHE.clear()
    worker_root = Path(settings.worker_root)
    if not worker_root.is_dir():
        return
    for worker_dir in worker_root.glob("worker_*"):
        if worker_dir.is_dir():
            shutil.rmtree(worker_dir, ignore_errors=True)
