"""Physical-space inference helpers for trained VULCAN trajectory surrogates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from model import VulcanTransformer

_TORCH_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


class InferenceError(RuntimeError):
    """Raised when trained-artifact loading or physical-space inference fails."""


def _load_json_dict(path: Path) -> dict[str, Any]:
    """Load one JSON file and require an object payload."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise InferenceError(f"Expected JSON object in {path}, found {type(payload).__name__}.")
    return payload


def _torch_dtype_from_name(name: str) -> torch.dtype:
    """Resolve one serialized dtype name back to a Torch dtype."""
    lowered = str(name).lower()
    if lowered not in _TORCH_DTYPES:
        raise InferenceError(f"Unsupported torch dtype name: {name}")
    return _TORCH_DTYPES[lowered]


def _log10_safe_torch(values: Tensor, epsilon: Tensor) -> Tensor:
    """Apply the project-standard base-10 log transform with a tensor epsilon floor.

    Uses tensor-only operations to remain compatible with torch.compile tracing.
    """
    return torch.log10(torch.clamp(values, min=epsilon))


def _denormalize_numpy(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    """Convert normalized NumPy arrays back into physical space.

    Args:
        values: Normalized NumPy array of any broadcast-compatible shape.
        stats: Serialized normalization statistics dictionary containing `method` plus the
            method-specific fields such as `mean/std` or `min/max`.

    Returns:
        Physical-space NumPy array with the same shape as `values` and dtype `float64`.
    """
    method = str(stats["method"])
    data = np.asarray(values, dtype=np.float64)
    if method == "none":
        return data
    if method == "standard":
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        return data * std + mean
    if method == "log-standard":
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        return np.power(10.0, data * std + mean)
    if method == "log-min-max":
        lower = np.asarray(stats["min"], dtype=np.float64)
        upper = np.asarray(stats["max"], dtype=np.float64)
        width = np.where((upper - lower) > 0.0, upper - lower, 1.0)
        return np.power(10.0, data * width + lower)
    raise InferenceError(f"Unsupported normalization method: {method}")


def _normalize_with_stats(
    values: Tensor,
    *,
    method: str,
    epsilon: Tensor,
    mean: Tensor,
    std: Tensor,
    lower: Tensor,
    upper: Tensor,
) -> Tensor:
    """Normalize one tensor using the serialized stats contract.

    Args:
        values: Input tensor in physical space.
        method: Normalization method name from serialized metadata.
        epsilon: Scalar tensor floor used by log-based methods.
        mean: Per-channel mean tensor.
        std: Per-channel standard-deviation tensor.
        lower: Per-channel minimum tensor.
        upper: Per-channel maximum tensor.

    Returns:
        Normalized tensor with the same shape as `values`.
    """
    if method == "none":
        return values
    if method == "standard":
        return (values - mean) / std
    if method == "log-standard":
        return (_log10_safe_torch(values, epsilon) - mean) / std
    if method == "log-min-max":
        width = torch.where((upper - lower) > 0.0, upper - lower, torch.ones_like(upper))
        return (_log10_safe_torch(values, epsilon) - lower) / width
    raise InferenceError(f"Unsupported normalization method: {method}")


def _denormalize_with_stats(
    values: Tensor,
    *,
    method: str,
    mean: Tensor,
    std: Tensor,
    lower: Tensor,
    upper: Tensor,
) -> Tensor:
    """Denormalize one tensor using the serialized stats contract.

    Args:
        values: Normalized tensor.
        method: Normalization method name from serialized metadata.
        mean: Per-channel mean tensor.
        std: Per-channel standard-deviation tensor.
        lower: Per-channel minimum tensor.
        upper: Per-channel maximum tensor.

    Returns:
        Physical-space tensor with the same shape as `values`.
    """
    if method == "none":
        return values
    if method == "standard":
        return values * std + mean
    if method == "log-standard":
        return torch.pow(10.0, values * std + mean)
    if method == "log-min-max":
        width = torch.where((upper - lower) > 0.0, upper - lower, torch.ones_like(upper))
        return torch.pow(10.0, values * width + lower)
    raise InferenceError(f"Unsupported normalization method: {method}")


class PhysicalSpaceStandaloneModel(nn.Module):
    """Wrapper that normalizes physical inputs, applies the model, and denormalizes outputs.

    The wrapped `forward` interface accepts:
    - profile tensors `pressure_bar`, `temperature_k`, and `kzz_cm2_s` with shape `[batch, nz]`
    - `initial_ymix` with shape `[batch, nz, target_dim]`
    - scalar conditioning tensors `gravity_cm_s2`, `metallicity_log10`, `c_to_o`, and `time_s`
      with shape `[batch]`

    It returns predicted physical-space `ymix` trajectories with shape
    `[batch, nz, target_dim]`.
    """

    def __init__(
        self,
        *,
        model: VulcanTransformer,
        normalization_metadata: dict[str, Any],
        data_contract: dict[str, Any],
    ) -> None:
        """Cache normalization metadata as non-persistent Torch buffers."""
        super().__init__()
        self.model = model
        self.sequence_length = int(data_contract["sequence_length"])
        self.target_dim = int(data_contract["target_dim"])
        self.target_species = list(data_contract["target_species_order"])

        epsilon = float(normalization_metadata["epsilon"])
        self.register_buffer(
            "epsilon",
            torch.tensor(epsilon, dtype=torch.float32),
            persistent=False,
        )

        self.pressure_method = str(normalization_metadata["sequence"]["pressure_bar"]["method"])
        self.temperature_method = str(
            normalization_metadata["sequence"]["temperature_k"]["method"]
        )
        self.kzz_method = str(normalization_metadata["sequence"]["kzz_cm2_s"]["method"])
        self.initial_ymix_method = str(
            normalization_metadata["sequence"]["initial_ymix"]["method"]
        )
        self.gravity_method = str(normalization_metadata["globals"]["gravity_cm_s2"]["method"])
        self.metallicity_method = str(
            normalization_metadata["globals"]["metallicity_log10"]["method"]
        )
        self.c_to_o_method = str(normalization_metadata["globals"]["c_to_o"]["method"])
        self.time_method = str(normalization_metadata["globals"]["log10_time_s"]["method"])
        self.target_method = str(normalization_metadata["targets"]["ymix"]["method"])

        self._register_stat_block(
            "pressure",
            normalization_metadata["sequence"]["pressure_bar"],
            size=1,
        )
        self._register_stat_block(
            "temperature",
            normalization_metadata["sequence"]["temperature_k"],
            size=1,
        )
        self._register_stat_block(
            "kzz",
            normalization_metadata["sequence"]["kzz_cm2_s"],
            size=1,
        )
        self._register_stat_block(
            "initial",
            normalization_metadata["sequence"]["initial_ymix"],
            size=self.target_dim,
        )
        self._register_stat_block(
            "gravity",
            normalization_metadata["globals"]["gravity_cm_s2"],
            size=1,
        )
        self._register_stat_block(
            "metallicity",
            normalization_metadata["globals"]["metallicity_log10"],
            size=1,
        )
        self._register_stat_block(
            "c_to_o",
            normalization_metadata["globals"]["c_to_o"],
            size=1,
        )
        self._register_stat_block(
            "time",
            normalization_metadata["globals"]["log10_time_s"],
            size=1,
        )
        self._register_stat_block(
            "target",
            normalization_metadata["targets"]["ymix"],
            size=self.target_dim,
        )

    def _register_stats_buffer(self, name: str, values: Any) -> None:
        """Register one stats buffer with a consistent float32 backing type."""
        array = np.asarray(values, dtype=np.float32)
        self.register_buffer(name, torch.as_tensor(array), persistent=False)

    def _register_stat_block(
        self,
        prefix: str,
        stats: dict[str, Any],
        *,
        size: int,
    ) -> None:
        """Register the `{mean,std,min,max}` buffer set for one normalized variable."""
        defaults = {
            "mean": [0.0] * size,
            "std": [1.0] * size,
            "min": [0.0] * size,
            "max": [1.0] * size,
        }
        for field_name, default_values in defaults.items():
            self._register_stats_buffer(
                f"{prefix}_{field_name}",
                stats.get(field_name, default_values),
            )

    def forward(
        self,
        pressure_bar: Tensor,
        temperature_k: Tensor,
        kzz_cm2_s: Tensor,
        initial_ymix: Tensor,
        gravity_cm_s2: Tensor,
        metallicity_log10: Tensor,
        c_to_o: Tensor,
        time_s: Tensor,
    ) -> Tensor:
        """Run physical-space inference for one batch of conditioning inputs.

        Args:
            pressure_bar: Pressure profile tensor with shape `[batch, nz]`.
            temperature_k: Temperature profile tensor with shape `[batch, nz]`.
            kzz_cm2_s: Eddy-diffusion profile tensor with shape `[batch, nz]`.
            initial_ymix: Initial species mixing ratios with shape `[batch, nz, target_dim]`.
            gravity_cm_s2: Gravity values with shape `[batch]`.
            metallicity_log10: Log10 metallicity values with shape `[batch]`.
            c_to_o: Carbon-to-oxygen ratio values with shape `[batch]`.
            time_s: Physical query times in seconds with shape `[batch]`.

        Returns:
            Predicted physical-space mixing ratios with shape `[batch, nz, target_dim]`.
        """
        if pressure_bar.ndim != 2 or temperature_k.ndim != 2 or kzz_cm2_s.ndim != 2:
            raise ValueError(
                "pressure_bar, temperature_k, and kzz_cm2_s must have shape [batch, nz]."
            )
        if initial_ymix.ndim != 3:
            raise ValueError("initial_ymix must have shape [batch, nz, species].")
        if pressure_bar.shape != temperature_k.shape or pressure_bar.shape != kzz_cm2_s.shape:
            raise ValueError(
                "pressure_bar, temperature_k, and kzz_cm2_s must share the same shape."
            )
        if int(pressure_bar.shape[1]) != self.sequence_length:
            raise ValueError(
                "Expected sequence length "
                f"{self.sequence_length}, got {int(pressure_bar.shape[1])}."
            )
        if initial_ymix.shape[:2] != pressure_bar.shape:
            raise ValueError("initial_ymix must align with the profile dimensions [batch, nz].")
        if int(initial_ymix.shape[-1]) != self.target_dim:
            raise ValueError(
                f"initial_ymix last dimension must equal target_dim={self.target_dim}, "
                f"got {int(initial_ymix.shape[-1])}."
            )

        batch_size = int(pressure_bar.shape[0])
        for name, tensor in {
            "gravity_cm_s2": gravity_cm_s2,
            "metallicity_log10": metallicity_log10,
            "c_to_o": c_to_o,
            "time_s": time_s,
        }.items():
            if tensor.ndim != 1 or int(tensor.shape[0]) != batch_size:
                raise ValueError(f"{name} must have shape [batch].")

        dtype = pressure_bar.dtype
        device = pressure_bar.device
        epsilon = self.epsilon.to(dtype=dtype, device=device)
        if not torch.compiler.is_compiling() and torch.any(time_s <= 0):
            raise ValueError("time_s must be strictly positive for log10 conditioning.")

        pressure_norm = _normalize_with_stats(
            pressure_bar,
            method=self.pressure_method,
            epsilon=epsilon,
            mean=self.pressure_mean.to(dtype=dtype, device=device),
            std=self.pressure_std.to(dtype=dtype, device=device),
            lower=self.pressure_min.to(dtype=dtype, device=device),
            upper=self.pressure_max.to(dtype=dtype, device=device),
        )
        temperature_norm = _normalize_with_stats(
            temperature_k,
            method=self.temperature_method,
            epsilon=epsilon,
            mean=self.temperature_mean.to(dtype=dtype, device=device),
            std=self.temperature_std.to(dtype=dtype, device=device),
            lower=self.temperature_min.to(dtype=dtype, device=device),
            upper=self.temperature_max.to(dtype=dtype, device=device),
        )
        kzz_norm = _normalize_with_stats(
            kzz_cm2_s,
            method=self.kzz_method,
            epsilon=epsilon,
            mean=self.kzz_mean.to(dtype=dtype, device=device),
            std=self.kzz_std.to(dtype=dtype, device=device),
            lower=self.kzz_min.to(dtype=dtype, device=device),
            upper=self.kzz_max.to(dtype=dtype, device=device),
        )
        initial_norm = _normalize_with_stats(
            initial_ymix,
            method=self.initial_ymix_method,
            epsilon=epsilon,
            mean=self.initial_mean.to(dtype=dtype, device=device),
            std=self.initial_std.to(dtype=dtype, device=device),
            lower=self.initial_min.to(dtype=dtype, device=device),
            upper=self.initial_max.to(dtype=dtype, device=device),
        )

        log10_time_s = torch.log10(time_s)
        gravity_norm = _normalize_with_stats(
            gravity_cm_s2,
            method=self.gravity_method,
            epsilon=epsilon,
            mean=self.gravity_mean.to(dtype=dtype, device=device),
            std=self.gravity_std.to(dtype=dtype, device=device),
            lower=self.gravity_min.to(dtype=dtype, device=device),
            upper=self.gravity_max.to(dtype=dtype, device=device),
        )
        metallicity_norm = _normalize_with_stats(
            metallicity_log10,
            method=self.metallicity_method,
            epsilon=epsilon,
            mean=self.metallicity_mean.to(dtype=dtype, device=device),
            std=self.metallicity_std.to(dtype=dtype, device=device),
            lower=self.metallicity_min.to(dtype=dtype, device=device),
            upper=self.metallicity_max.to(dtype=dtype, device=device),
        )
        c_to_o_norm = _normalize_with_stats(
            c_to_o,
            method=self.c_to_o_method,
            epsilon=epsilon,
            mean=self.c_to_o_mean.to(dtype=dtype, device=device),
            std=self.c_to_o_std.to(dtype=dtype, device=device),
            lower=self.c_to_o_min.to(dtype=dtype, device=device),
            upper=self.c_to_o_max.to(dtype=dtype, device=device),
        )
        time_norm = _normalize_with_stats(
            log10_time_s,
            method=self.time_method,
            epsilon=epsilon,
            mean=self.time_mean.to(dtype=dtype, device=device),
            std=self.time_std.to(dtype=dtype, device=device),
            lower=self.time_min.to(dtype=dtype, device=device),
            upper=self.time_max.to(dtype=dtype, device=device),
        )

        sequence_inputs = torch.cat(
            [
                pressure_norm.unsqueeze(-1),
                temperature_norm.unsqueeze(-1),
                kzz_norm.unsqueeze(-1),
                initial_norm,
            ],
            dim=-1,
        )
        global_inputs = torch.stack(
            [gravity_norm, metallicity_norm, c_to_o_norm, time_norm],
            dim=-1,
        )
        normalized_prediction = self.model(sequence_inputs, global_inputs, padding_mask=None)
        return _denormalize_with_stats(
            normalized_prediction,
            method=self.target_method,
            mean=self.target_mean.to(dtype=dtype, device=device),
            std=self.target_std.to(dtype=dtype, device=device),
            lower=self.target_min.to(dtype=dtype, device=device),
            upper=self.target_max.to(dtype=dtype, device=device),
        )


def load_physical_space_model(
    run_dir: Path,
    *,
    checkpoint_name: str = "best.pt",
    device: torch.device | str = "cpu",
) -> tuple[PhysicalSpaceStandaloneModel, dict[str, Any], dict[str, Any]]:
    """Load a trained checkpoint and wrap it with physical-space IO transforms.

    Args:
        run_dir: Model run directory containing checkpoints and exported metadata JSON files.
        checkpoint_name: Checkpoint filename inside `run_dir`.
        device: Torch device or device string used to materialize the model.

    Returns:
        Tuple `(wrapper, normalization_metadata, data_contract)` where `wrapper` is a
        `PhysicalSpaceStandaloneModel`, and the dictionaries are the parsed JSON artifacts used
        for inference.
    """
    resolved_run_dir = Path(run_dir).resolve()
    checkpoint_path = resolved_run_dir / checkpoint_name
    if not checkpoint_path.is_file():
        raise InferenceError(f"Missing checkpoint: {checkpoint_path}")

    # weights_only=False is safe here: checkpoints are self-generated by this pipeline.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise InferenceError(f"Invalid checkpoint payload: {checkpoint_path}")

    normalization_metadata = _load_json_dict(resolved_run_dir / "normalization_metadata.json")
    data_contract = _load_json_dict(resolved_run_dir / "data_contract.json")
    config = checkpoint["config"]
    model_cfg = config["training"]["model"]
    precision_cfg = config["precision"]

    model = VulcanTransformer(
        input_dim=int(data_contract["input_dim"]),
        global_dim=int(data_contract["global_dim"]),
        target_dim=int(data_contract["target_dim"]),
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        num_layers=int(model_cfg["num_layers"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        dropout=float(model_cfg["dropout"]),
        film_clamp=float(model_cfg["film_clamp"]),
        output_head_divisor=int(model_cfg["output_head_divisor"]),
        max_sequence_length=int(model_cfg["max_sequence_length"]),
    ).to(device=device, dtype=_torch_dtype_from_name(str(precision_cfg["model_dtype"])))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    wrapper = PhysicalSpaceStandaloneModel(
        model=model,
        normalization_metadata=normalization_metadata,
        data_contract=data_contract,
    ).to(device=device, dtype=_torch_dtype_from_name(str(precision_cfg["forward_dtype"])))
    wrapper.eval()
    return wrapper, normalization_metadata, data_contract


def physical_inputs_from_processed_arrays(
    *,
    sequence_inputs: np.ndarray,
    global_inputs: np.ndarray,
    normalization_metadata: dict[str, Any],
    data_contract: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Invert processed arrays back into physical-space inference inputs.

    Args:
        sequence_inputs: Processed sequence array with shape `[nz, input_dim]` or
            `[batch, nz, input_dim]`.
        global_inputs: Processed global array with shape `[global_dim]` or `[batch, global_dim]`.
        normalization_metadata: Serialized normalization metadata used during preprocessing.
        data_contract: Serialized feature-order and shape contract for the trained model.

    Returns:
        Dictionary containing physical-space NumPy arrays for predictor inputs. Profile values are
        returned with shape `[batch, nz]` / `[batch, nz, target_dim]` unless the input was
        unbatched, in which case the batch dimension is removed. Global scalars are returned as
        `[batch]` arrays or scalars with the batch dimension removed.
    """
    seq = np.asarray(sequence_inputs, dtype=np.float64)
    glb = np.asarray(global_inputs, dtype=np.float64)
    squeeze_batch = False
    if seq.ndim == 2:
        seq = seq[None, ...]
        squeeze_batch = True
    if glb.ndim == 1:
        glb = glb[None, ...]
    if seq.ndim != 3 or glb.ndim != 2:
        raise InferenceError(
            "Processed sequence/global arrays must have ranks "
            "[batch, nz, input_dim] and [batch, global_dim]."
        )
    if seq.shape[0] != glb.shape[0]:
        raise InferenceError(
            "Processed sequence/global arrays must share the same batch dimension."
        )

    species = list(data_contract["target_species_order"])
    if int(seq.shape[-1]) != int(data_contract["input_dim"]):
        raise InferenceError("Processed sequence array does not match data_contract input_dim.")
    if int(glb.shape[-1]) != int(data_contract["global_dim"]):
        raise InferenceError("Processed global array does not match data_contract global_dim.")
    if int(seq.shape[1]) != int(data_contract["sequence_length"]):
        raise InferenceError(
            "Processed sequence array does not match data_contract sequence_length."
        )
    expected_sequence_order = [
        "pressure_bar",
        "temperature_k",
        "kzz_cm2_s",
        *[f"initial_ymix:{species_name}" for species_name in species],
    ]
    expected_global_order = ["gravity_cm_s2", "metallicity_log10", "c_to_o", "log10_time_s"]
    if list(data_contract["sequence_feature_order"]) != expected_sequence_order:
        raise InferenceError(
            "data_contract sequence_feature_order is incompatible with v1 inference."
        )
    if list(data_contract["global_feature_order"]) != expected_global_order:
        raise InferenceError(
            "data_contract global_feature_order is incompatible with v1 inference."
        )

    pressure = _denormalize_numpy(
        seq[..., 0],
        normalization_metadata["sequence"]["pressure_bar"],
    )
    temperature = _denormalize_numpy(
        seq[..., 1],
        normalization_metadata["sequence"]["temperature_k"],
    )
    kzz = _denormalize_numpy(seq[..., 2], normalization_metadata["sequence"]["kzz_cm2_s"])
    initial_ymix = _denormalize_numpy(
        seq[..., 3:],
        normalization_metadata["sequence"]["initial_ymix"],
    )

    gravity = _denormalize_numpy(
        glb[..., 0],
        normalization_metadata["globals"]["gravity_cm_s2"],
    )
    metallicity = _denormalize_numpy(
        glb[..., 1],
        normalization_metadata["globals"]["metallicity_log10"],
    )
    c_to_o = _denormalize_numpy(glb[..., 2], normalization_metadata["globals"]["c_to_o"])
    log10_time_s = _denormalize_numpy(
        glb[..., 3],
        normalization_metadata["globals"]["log10_time_s"],
    )
    time_s = np.power(10.0, log10_time_s)

    result = {
        "pressure_bar": pressure,
        "temperature_k": temperature,
        "kzz_cm2_s": kzz,
        "initial_ymix": initial_ymix,
        "gravity_cm_s2": gravity,
        "metallicity_log10": metallicity,
        "c_to_o": c_to_o,
        "time_s": time_s,
        "target_species": species,
    }
    if squeeze_batch:
        return {
            "pressure_bar": pressure[0],
            "temperature_k": temperature[0],
            "kzz_cm2_s": kzz[0],
            "initial_ymix": initial_ymix[0],
            "gravity_cm_s2": gravity[0],
            "metallicity_log10": metallicity[0],
            "c_to_o": c_to_o[0],
            "time_s": time_s[0],
            "target_species": species,
        }
    return result


class VulcanPredictor:
    """Convenience numpy-facing predictor for physical-space inference."""

    def __init__(self, model: PhysicalSpaceStandaloneModel, *, device: torch.device) -> None:
        """Store a ready-to-run physical-space model on one target device."""
        self.model = model
        self.device = device
        self.target_species = tuple(model.target_species)

    @classmethod
    def from_run_dir(
        cls,
        run_dir: Path,
        *,
        checkpoint_name: str = "best.pt",
        device: torch.device | str = "cpu",
    ) -> "VulcanPredictor":
        """Construct a predictor directly from one trained run directory."""
        resolved_device = torch.device(device)
        model, _normalization_metadata, _data_contract = load_physical_space_model(
            run_dir,
            checkpoint_name=checkpoint_name,
            device=resolved_device,
        )
        return cls(model, device=resolved_device)

    def predict(
        self,
        *,
        pressure_bar: Any,
        temperature_k: Any,
        kzz_cm2_s: Any,
        initial_ymix: Any,
        gravity_cm_s2: Any,
        metallicity_log10: Any,
        c_to_o: Any,
        time_s: Any,
    ) -> np.ndarray:
        """Run NumPy-facing inference and return physical-space `ymix` predictions.

        Args:
            pressure_bar: Array-like pressure profile with shape `[nz]` or `[batch, nz]`.
            temperature_k: Array-like temperature profile with shape `[nz]` or `[batch, nz]`.
            kzz_cm2_s: Array-like Kzz profile with shape `[nz]` or `[batch, nz]`.
            initial_ymix: Array-like initial composition with shape `[nz, species]` or
                `[batch, nz, species]`.
            gravity_cm_s2: Scalar-like gravity input, `[batch]`, or `[batch, 1]`.
            metallicity_log10: Scalar-like metallicity input, `[batch]`, or `[batch, 1]`.
            c_to_o: Scalar-like C/O input, `[batch]`, or `[batch, 1]`.
            time_s: Positive scalar-like time input, `[batch]`, or `[batch, 1]`.

        Returns:
            Physical-space prediction array with shape `[nz, target_dim]` for unbatched inputs or
            `[batch, nz, target_dim]` for batched inputs.
        """
        pressure_np, squeeze_batch = self._coerce_profiles(pressure_bar, "pressure_bar")
        temperature_np, _ = self._coerce_profiles(
            temperature_k,
            "temperature_k",
            batch_size=pressure_np.shape[0],
        )
        kzz_np, _ = self._coerce_profiles(
            kzz_cm2_s,
            "kzz_cm2_s",
            batch_size=pressure_np.shape[0],
        )
        initial_np = self._coerce_initial_ymix(
            initial_ymix,
            batch_size=pressure_np.shape[0],
            sequence_length=pressure_np.shape[1],
        )
        gravity_np = self._coerce_globals(
            gravity_cm_s2,
            "gravity_cm_s2",
            batch_size=pressure_np.shape[0],
        )
        metallicity_np = self._coerce_globals(
            metallicity_log10,
            "metallicity_log10",
            batch_size=pressure_np.shape[0],
        )
        c_to_o_np = self._coerce_globals(c_to_o, "c_to_o", batch_size=pressure_np.shape[0])
        time_np = self._coerce_globals(time_s, "time_s", batch_size=pressure_np.shape[0])
        if np.any(time_np <= 0.0):
            raise InferenceError("time_s must be strictly positive for log10 conditioning.")

        model_dtype = next(self.model.parameters()).dtype
        with torch.inference_mode():
            prediction = self.model(
                torch.as_tensor(pressure_np, dtype=model_dtype, device=self.device),
                torch.as_tensor(temperature_np, dtype=model_dtype, device=self.device),
                torch.as_tensor(kzz_np, dtype=model_dtype, device=self.device),
                torch.as_tensor(initial_np, dtype=model_dtype, device=self.device),
                torch.as_tensor(gravity_np, dtype=model_dtype, device=self.device),
                torch.as_tensor(metallicity_np, dtype=model_dtype, device=self.device),
                torch.as_tensor(c_to_o_np, dtype=model_dtype, device=self.device),
                torch.as_tensor(time_np, dtype=model_dtype, device=self.device),
            )

        output = prediction.detach().cpu().numpy()
        if squeeze_batch:
            return output[0]
        return output

    __call__ = predict

    def _coerce_profiles(
        self,
        values: Any,
        name: str,
        *,
        batch_size: int | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Normalize profile inputs to a `[batch, nz]` NumPy layout."""
        array = np.asarray(values, dtype=np.float64)
        squeeze_batch = False
        if array.ndim == 1:
            array = array[None, :]
            squeeze_batch = True
        if array.ndim != 2:
            raise InferenceError(f"{name} must have shape [nz] or [batch, nz].")
        if batch_size is not None and int(array.shape[0]) != batch_size:
            raise InferenceError(
                f"{name} batch size does not match the other profile inputs."
            )
        return array, squeeze_batch

    def _coerce_initial_ymix(
        self,
        values: Any,
        *,
        batch_size: int,
        sequence_length: int,
    ) -> np.ndarray:
        """Normalize initial-composition inputs to `[batch, nz, species]`."""
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3:
            raise InferenceError(
                "initial_ymix must have shape [nz, species] or [batch, nz, species]."
            )
        if int(array.shape[0]) != batch_size or int(array.shape[1]) != sequence_length:
            raise InferenceError(
                "initial_ymix must match the batch and sequence dimensions "
                "of the profiles."
            )
        if int(array.shape[2]) != len(self.target_species):
            raise InferenceError(
                "initial_ymix species dimension must equal "
                f"{len(self.target_species)} configured target species."
            )
        return array

    def _coerce_globals(self, values: Any, name: str, *, batch_size: int) -> np.ndarray:
        """Normalize scalar-like global inputs to a `[batch]` NumPy vector."""
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0:
            return np.full((batch_size,), float(array), dtype=np.float64)
        if array.ndim == 1:
            if array.size == 1:
                return np.full((batch_size,), float(array[0]), dtype=np.float64)
            if int(array.shape[0]) == batch_size:
                return array.astype(np.float64, copy=False)
        if array.ndim == 2 and int(array.shape[0]) == batch_size and int(array.shape[1]) == 1:
            return array[:, 0].astype(np.float64, copy=False)
        raise InferenceError(f"{name} must be scalar, [batch], or [batch, 1].")


__all__ = [
    "InferenceError",
    "PhysicalSpaceStandaloneModel",
    "VulcanPredictor",
    "load_physical_space_model",
    "physical_inputs_from_processed_arrays",
]
