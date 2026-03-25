from __future__ import annotations

import functools
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .numpy_compat import patch_numpy_asarray_copy

patch_numpy_asarray_copy()

import jax
import jax.numpy as jnp
import numpy as np

from .anchor_states import build_flat_h2_he_anchor
from .data_loader import load_processed_split
from .export_jax import load_export_bundle
from .jax_model import ModelDimensions, apply_model
from .path_utils import resolve_path, resolve_project_root


def _to_jax_tree(obj: Any) -> Any:
    return jax.tree_util.tree_map(jnp.asarray, obj)


def _apply_block_jax(x: jax.Array, block: dict[str, Any]) -> jax.Array:
    method = block["method"]
    mean = jnp.asarray(block["mean"], dtype=jnp.float32)
    std = jnp.asarray(block["std"], dtype=jnp.float32)
    if method == "none":
        return x
    if method == "standard":
        return (x - mean) / std
    if method == "log-standard":
        floor = float(block["floor"])
        return (jnp.log10(jnp.clip(x, floor)) - mean) / std
    raise ValueError(f"Unsupported normalization block method: {method}")


def _inverse_block_jax(x: jax.Array, block: dict[str, Any]) -> jax.Array:
    method = block["method"]
    mean = jnp.asarray(block["mean"], dtype=jnp.float32)
    std = jnp.asarray(block["std"], dtype=jnp.float32)
    if method == "none":
        return x
    if method == "standard":
        return x * std + mean
    if method == "log-standard":
        return jnp.power(10.0, x * std + mean)
    raise ValueError(f"Unsupported normalization block method: {method}")


def _apply_mixed_block_jax(x: jax.Array, block: dict[str, Any]) -> jax.Array:
    cols = []
    for i, method in enumerate(block["methods"]):
        col = x[..., i]
        mean = float(block["mean"][i])
        std = float(block["std"][i])
        floor = block["floor"][i]
        if method == "none":
            cols.append(col)
        elif method == "standard":
            cols.append((col - mean) / std)
        elif method == "log-standard":
            cols.append((jnp.log10(jnp.clip(col, float(floor))) - mean) / std)
        else:
            raise ValueError(f"Unsupported mixed normalization method: {method}")
    return jnp.stack(cols, axis=-1)


def _inverse_block_numpy(x: np.ndarray, block: dict[str, Any]) -> np.ndarray:
    method = block["method"]
    mean = np.asarray(block["mean"], dtype=np.float64)
    std = np.asarray(block["std"], dtype=np.float64)
    if method == "none":
        return np.asarray(x, dtype=np.float64)
    if method == "standard":
        return np.asarray(x, dtype=np.float64) * std + mean
    if method == "log-standard":
        return np.power(10.0, np.asarray(x, dtype=np.float64) * std + mean)
    raise ValueError(f"Unsupported normalization block method: {method}")


@dataclass
class PhysicalSpaceStandaloneModel:
    params: Any
    dims: ModelDimensions
    normalization: dict[str, Any]
    contract: dict[str, Any]
    config: dict[str, Any]
    project_root: Path | None = None

    def _build_jax_transition_fn(self):
        dims = self.dims
        params = self.params
        normalization = self.normalization
        dt_feature_index = int(self.contract["dt_feature_index"])

        def transition_fn(
            pressure_bar: jax.Array,
            temperature_k: jax.Array,
            kzz_cm2_s: jax.Array,
            anchor_ymix: jax.Array,
            global_static_vector: jax.Array,
            spectrum_inputs: jax.Array,
            log10_dt_s: jax.Array,
        ) -> jax.Array:
            static = jnp.stack([pressure_bar, temperature_k, kzz_cm2_s], axis=-1)
            sequence_parts = []
            for i, block in enumerate(normalization["sequence_static"]["blocks"]):
                sequence_parts.append(_apply_block_jax(static[..., i : i + 1], block))
            static_norm = jnp.concatenate(sequence_parts, axis=-1)
            anchor_norm = _apply_block_jax(anchor_ymix, normalization["state"])
            sequence = jnp.concatenate([static_norm, anchor_norm], axis=-1)[None, :, :]
            global_static_norm = _apply_mixed_block_jax(global_static_vector[None, :], normalization["global_static"])
            dt_block = normalization["log10_dt_s"]
            dt_norm = _apply_block_jax(jnp.asarray([[log10_dt_s]], dtype=jnp.float32), dt_block)
            globals_full = jnp.concatenate(
                [
                    global_static_norm[:, :dt_feature_index],
                    dt_norm,
                    global_static_norm[:, dt_feature_index:],
                ],
                axis=-1,
            )
            spectrum_norm = _apply_block_jax(spectrum_inputs[None, :], normalization["spectrum"])
            pred_norm, _ = apply_model(params, sequence, globals_full, spectrum_norm, dims)
            pred = _inverse_block_jax(pred_norm[0], normalization["target"])
            return pred

        return jax.jit(transition_fn)

    @functools.cached_property
    def transition_fn(self):
        """JIT-compiled transition function mapping physical-space inputs to predictions."""
        return self._build_jax_transition_fn()

    def global_static_vector_from_mapping(self, values: Mapping[str, float]) -> np.ndarray:
        order = list(self.contract["global_static_feature_order"])
        return np.asarray([float(values[name]) for name in order], dtype=np.float32)

    def _equilibrium_anchor_config(self) -> dict[str, Any]:
        inference_cfg = self.config.get("inference", {})
        eq_anchor_cfg = inference_cfg.get("equilibrium_anchor", {})
        default_source = (
            "flat"
            if str(self.config.get("generation", {}).get("target_mode", "")).lower() == "equilibrium_only"
            else "trajectory"
        )
        return {
            "source": str(eq_anchor_cfg.get("source", default_source)).lower(),
            "split": str(eq_anchor_cfg.get("split", "train")).lower(),
            "run_id": eq_anchor_cfg.get("run_id"),
            "step_index": eq_anchor_cfg.get("step_index", 0),
        }

    def _flat_initial_state(self, nz: int) -> np.ndarray:
        """Build an H2/He-dominated flat initial state for the configured species."""
        return build_flat_h2_he_anchor(
            list(self.contract["state_species_order"]),
            nz=int(nz),
        )

    @functools.cached_property
    def equilibrium_anchor_profile(self) -> np.ndarray | None:
        """Load a reference anchor profile from the processed trajectory set when configured."""
        anchor_cfg = self._equilibrium_anchor_config()
        if anchor_cfg["source"] != "trajectory" or self.project_root is None:
            return None
        processed_root = resolve_path(self.config["paths"]["processed_root"], self.project_root)
        split_dir = processed_root / str(anchor_cfg["split"])
        if not (split_dir / "metadata.json").exists():
            return None
        split = load_processed_split(split_dir)
        if not split.run_ids:
            return None
        run_id = anchor_cfg["run_id"]
        if run_id is None:
            run_index = 0
        else:
            try:
                run_index = split.run_ids.index(str(run_id))
            except ValueError as exc:
                raise ValueError(
                    f"Configured equilibrium anchor run_id {run_id!r} was not found in split {split.name!r}."
                ) from exc
        valid_steps = np.flatnonzero(split.valid_steps_mask[run_index])
        if valid_steps.size == 0:
            raise ValueError(
                f"Configured equilibrium anchor run {split.run_ids[run_index]!r} has no valid saved steps."
            )
        requested_step_index = int(anchor_cfg["step_index"])
        distances = np.abs(valid_steps - requested_step_index)
        selected_step = int(valid_steps[int(np.argmin(distances))])
        anchor_state_norm = np.asarray(
            split.state_trajectories[run_index, selected_step],
            dtype=np.float64,
        )
        return _inverse_block_numpy(anchor_state_norm, self.normalization["state"])

    def predict(
        self,
        *,
        pressure_bar: np.ndarray,
        temperature_K: np.ndarray,
        eddy_diffusion_cm2_s: np.ndarray,
        ymix_state: np.ndarray,
        global_inputs: Mapping[str, float] | np.ndarray,
        log10_dt_s: float,
        spectrum_inputs: np.ndarray,
    ) -> np.ndarray:
        if isinstance(global_inputs, Mapping):
            global_static = self.global_static_vector_from_mapping(global_inputs)
        else:
            global_static = np.asarray(global_inputs, dtype=np.float32)
        pred = self.transition_fn(
            jnp.asarray(pressure_bar, dtype=jnp.float32),
            jnp.asarray(temperature_K, dtype=jnp.float32),
            jnp.asarray(eddy_diffusion_cm2_s, dtype=jnp.float32),
            jnp.asarray(ymix_state, dtype=jnp.float32),
            jnp.asarray(global_static, dtype=jnp.float32),
            jnp.asarray(spectrum_inputs, dtype=jnp.float32),
            jnp.asarray(log10_dt_s, dtype=jnp.float32),
        )
        return np.asarray(pred, dtype=np.float64)

    def equilibrium(
        self,
        *,
        pressure_bar: np.ndarray,
        temperature_K: np.ndarray,
        eddy_diffusion_cm2_s: np.ndarray,
        global_inputs: Mapping[str, float] | np.ndarray,
        spectrum_inputs: np.ndarray,
    ) -> np.ndarray:
        """Return near-equilibrium abundances for the given PT profile.

        Uses the surrogate with a minimal timestep (dt = 1 s) from the
        configured equilibrium anchor. By default this is the earliest valid
        saved state from the configured processed split, with a flat
        H2/He-dominated fallback if the processed data is unavailable.
        """
        nz = np.asarray(pressure_bar).shape[0]
        anchor = self.equilibrium_anchor_profile
        if anchor is None or anchor.shape[0] != nz:
            anchor = self._flat_initial_state(nz)
        return self.predict(
            pressure_bar=pressure_bar,
            temperature_K=temperature_K,
            eddy_diffusion_cm2_s=eddy_diffusion_cm2_s,
            ymix_state=anchor,
            global_inputs=global_inputs,
            log10_dt_s=0.0,
            spectrum_inputs=spectrum_inputs,
        )

    def step_from_equilibrium(
        self,
        *,
        dt_s: float,
        pressure_bar: np.ndarray,
        temperature_K: np.ndarray,
        eddy_diffusion_cm2_s: np.ndarray,
        global_inputs: Mapping[str, float] | np.ndarray,
        spectrum_inputs: np.ndarray,
    ) -> np.ndarray:
        """Compute equilibrium, then step forward by *dt_s* seconds.

        Convenience wrapper that first calls :meth:`equilibrium` to obtain
        the anchor state, then evolves it forward using the user-supplied dt.
        """
        eq_state = self.equilibrium(
            pressure_bar=pressure_bar,
            temperature_K=temperature_K,
            eddy_diffusion_cm2_s=eddy_diffusion_cm2_s,
            global_inputs=global_inputs,
            spectrum_inputs=spectrum_inputs,
        )
        return self.predict(
            pressure_bar=pressure_bar,
            temperature_K=temperature_K,
            eddy_diffusion_cm2_s=eddy_diffusion_cm2_s,
            ymix_state=eq_state,
            global_inputs=global_inputs,
            log10_dt_s=math.log10(float(dt_s)),
            spectrum_inputs=spectrum_inputs,
        )


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def load_physical_space_model(path: str | Path) -> PhysicalSpaceStandaloneModel:
    path_obj = Path(path)
    resolved_project_root: Path | None
    if path_obj.is_dir():
        bundle = load_export_bundle(path_obj)
        params = _to_jax_tree(bundle["params"])
        dims = ModelDimensions.from_dict(bundle["model_dimensions"])
        if "_project_root" in bundle["config"]:
            resolved_project_root = Path(bundle["config"]["_project_root"]).resolve()
        else:
            try:
                resolved_project_root = resolve_project_root(path_obj.resolve())
            except FileNotFoundError:
                resolved_project_root = None
        return PhysicalSpaceStandaloneModel(
            params=params,
            dims=dims,
            normalization=bundle["normalization"],
            contract=bundle["data_contract"],
            config=bundle["config"],
            project_root=resolved_project_root,
        )
    payload = load_checkpoint_payload(path_obj)
    params = _to_jax_tree(payload["params"])
    dims = ModelDimensions.from_dict(payload["model_dimensions"])
    if "_project_root" in payload["config"]:
        resolved_project_root = Path(payload["config"]["_project_root"]).resolve()
    else:
        try:
            resolved_project_root = resolve_project_root(path_obj.resolve())
        except FileNotFoundError:
            resolved_project_root = None
    return PhysicalSpaceStandaloneModel(
        params=params,
        dims=dims,
        normalization=payload["normalization"],
        contract=payload["data_contract"],
        config=payload["config"],
        project_root=resolved_project_root,
    )
