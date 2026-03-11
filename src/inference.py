"""Physical-space inference helpers for trained VULCAN single-step transition surrogates.

Provides two main interfaces for running predictions in physical units:

- **PhysicalSpaceStandaloneModel** (``nn.Module``): Takes raw physical inputs
  (pressure in bar, temperature in K, mixing ratios, etc.), normalizes them
  using the stored training statistics, runs the transformer, and denormalizes
  the output back to physical mixing ratios.  Supports ``torch.compile``.

- **VulcanPredictor**: NumPy-facing convenience class wrapping the above.
  Accepts numpy arrays / scalars, handles batching, and exposes ``predict()``
  for single-step transitions.

Normalization roundtrip: physical -> log10 (if applicable) -> standardize ->
model -> destandardize -> 10^x (if applicable) -> physical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from config_utils import static_conditioning_defaults
from model import VulcanTransitionTransformer

_TORCH_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


class InferenceError(RuntimeError):
    """Raised when trained-artifact loading or physical-space inference fails."""


def _load_json_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise InferenceError(f"Expected JSON object in {path}, found {type(payload).__name__}.")
    return payload


def _load_dict_from_checkpoint_or_disk(
    *,
    checkpoint: dict[str, Any],
    key: str,
    path: Path,
) -> dict[str, Any]:
    """Load one inference artifact from the checkpoint when embedded, else from disk."""
    embedded = checkpoint.get(key)
    if embedded is not None:
        if not isinstance(embedded, dict):
            raise InferenceError(
                f"Checkpoint field '{key}' must be a JSON-like object, found {type(embedded).__name__}."
            )
        return embedded
    return _load_json_dict(path)


def _torch_dtype_from_name(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered not in _TORCH_DTYPES:
        raise InferenceError(f"Unsupported torch dtype name: {name}")
    return _TORCH_DTYPES[lowered]


def _log10_safe_torch(values: Tensor, epsilon: Tensor) -> Tensor:
    return torch.log10(torch.clamp(values, min=epsilon))


def _denormalize_numpy(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
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
    """Wrapper that normalizes physical inputs, runs the model, and denormalizes outputs.

    All normalization statistics (mean, std, min, max per variable) are stored
    as non-persistent buffers, so the wrapper moves cleanly across devices and
    dtypes and is compatible with ``torch.compile``.

    The forward signature accepts raw physical-unit tensors (pressure in bar,
    temperature in K, Kzz in cm^2/s, mixing ratios, gravity, etc.) and
    returns predicted mixing ratios in physical space.
    """

    def __init__(
        self,
        *,
        model: VulcanTransitionTransformer,
        normalization_metadata: dict[str, Any],
        data_contract: dict[str, Any],
        default_global_inputs: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.sequence_length = int(data_contract["sequence_length"])
        self.state_dim = int(data_contract["state_dim"])
        self.target_dim = int(data_contract["target_dim"])
        self.state_species = list(data_contract["state_species_order"])
        self.target_species = list(data_contract["output_species_order"])
        self.global_feature_order = list(data_contract["global_feature_order"])
        self.default_global_inputs = {
            str(key): float(value)
            for key, value in dict(default_global_inputs or {}).items()
            if key in self.global_feature_order and key != "log10_dt_s"
        }

        epsilon = float(normalization_metadata["epsilon"])
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float32), persistent=False)

        self.pressure_method = str(normalization_metadata["sequence"]["pressure_bar"]["method"])
        self.temperature_method = str(normalization_metadata["sequence"]["temperature_k"]["method"])
        self.kzz_method = str(normalization_metadata["sequence"]["kzz_cm2_s"]["method"])
        self.anchor_method = str(normalization_metadata["sequence"]["anchor_ymix"]["method"])
        self.global_methods = {
            name: str(normalization_metadata["globals"][name]["method"])
            for name in self.global_feature_order
        }
        self.target_method = str(normalization_metadata["targets"]["ymix"]["method"])

        self._register_stat_block("pressure", normalization_metadata["sequence"]["pressure_bar"], size=1)
        self._register_stat_block("temperature", normalization_metadata["sequence"]["temperature_k"], size=1)
        self._register_stat_block("kzz", normalization_metadata["sequence"]["kzz_cm2_s"], size=1)
        self._register_stat_block("anchor", normalization_metadata["sequence"]["anchor_ymix"], size=self.state_dim)
        for name in self.global_feature_order:
            self._register_stat_block(
                self._global_prefix(name),
                normalization_metadata["globals"][name],
                size=1,
            )
        self._register_stat_block("target", normalization_metadata["targets"]["ymix"], size=self.target_dim)

    def _register_stats_buffer(self, name: str, values: Any) -> None:
        array = np.asarray(values, dtype=np.float32)
        self.register_buffer(name, torch.as_tensor(array), persistent=False)

    def _register_stat_block(self, prefix: str, stats: dict[str, Any], *, size: int) -> None:
        defaults = {
            "mean": [0.0] * size,
            "std": [1.0] * size,
            "min": [0.0] * size,
            "max": [1.0] * size,
        }
        for field_name, default_values in defaults.items():
            self._register_stats_buffer(f"{prefix}_{field_name}", stats.get(field_name, default_values))

    def _global_prefix(self, name: str) -> str:
        return f"global_{name}"

    def _coerce_global_tensor(
        self,
        value: Tensor | float,
        *,
        name: str,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value, dtype=dtype, device=device)
        tensor = tensor.to(dtype=dtype, device=device)
        if tensor.ndim == 0:
            tensor = tensor.expand(batch_size)
        if tensor.ndim != 1 or int(tensor.shape[0]) != batch_size:
            raise ValueError(f"{name} must have shape [batch].")
        return tensor

    def _global_stat_tensor(self, name: str, field_name: str, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        return getattr(self, f"{self._global_prefix(name)}_{field_name}").to(dtype=dtype, device=device)

    def forward(
        self,
        pressure_bar: Tensor,
        temperature_k: Tensor,
        kzz_cm2_s: Tensor,
        anchor_ymix: Tensor,
        gravity_cm_s2: Tensor,
        metallicity_log10: Tensor,
        c_to_o: Tensor,
        dt_s: Tensor,
        extra_global_inputs: dict[str, Tensor | float] | None = None,
    ) -> Tensor:
        if pressure_bar.ndim != 2 or temperature_k.ndim != 2 or kzz_cm2_s.ndim != 2:
            raise ValueError("pressure_bar, temperature_k, and kzz_cm2_s must have shape [batch, nz].")
        if anchor_ymix.ndim != 3:
            raise ValueError("anchor_ymix must have shape [batch, nz, state_species].")
        if pressure_bar.shape != temperature_k.shape or pressure_bar.shape != kzz_cm2_s.shape:
            raise ValueError("pressure_bar, temperature_k, and kzz_cm2_s must share the same shape.")
        if int(pressure_bar.shape[1]) != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, got {int(pressure_bar.shape[1])}."
            )
        if anchor_ymix.shape[:2] != pressure_bar.shape:
            raise ValueError("anchor_ymix must align with the profile dimensions [batch, nz].")
        if int(anchor_ymix.shape[-1]) != self.state_dim:
            raise ValueError(
                f"anchor_ymix last dimension must equal state_dim={self.state_dim}, got {int(anchor_ymix.shape[-1])}."
            )

        batch_size = int(pressure_bar.shape[0])
        for name, tensor in {
            "gravity_cm_s2": gravity_cm_s2,
            "metallicity_log10": metallicity_log10,
            "c_to_o": c_to_o,
            "dt_s": dt_s,
        }.items():
            if tensor.ndim != 1 or int(tensor.shape[0]) != batch_size:
                raise ValueError(f"{name} must have shape [batch].")

        dtype = pressure_bar.dtype
        device = pressure_bar.device
        epsilon = self.epsilon.to(dtype=dtype, device=device)
        if not torch.compiler.is_compiling() and torch.any(dt_s <= 0):
            raise ValueError("dt_s must be strictly positive for log10 conditioning.")

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
        anchor_norm = _normalize_with_stats(
            anchor_ymix,
            method=self.anchor_method,
            epsilon=epsilon,
            mean=self.anchor_mean.to(dtype=dtype, device=device),
            std=self.anchor_std.to(dtype=dtype, device=device),
            lower=self.anchor_min.to(dtype=dtype, device=device),
            upper=self.anchor_max.to(dtype=dtype, device=device),
        )

        provided_globals: dict[str, Tensor | float] = {
            "gravity_cm_s2": gravity_cm_s2,
            "metallicity_log10": metallicity_log10,
            "c_to_o": c_to_o,
        }
        if extra_global_inputs is not None:
            provided_globals.update(extra_global_inputs)

        normalized_globals: list[Tensor] = []
        log10_dt_s = torch.log10(dt_s)
        for name in self.global_feature_order:
            if name == "log10_dt_s":
                values = self._coerce_global_tensor(
                    log10_dt_s,
                    name="dt_s",
                    batch_size=batch_size,
                    dtype=dtype,
                    device=device,
                )
            else:
                source = provided_globals.get(name, self.default_global_inputs.get(name))
                if source is None:
                    raise ValueError(f"Missing required global conditioning input '{name}'.")
                values = self._coerce_global_tensor(
                    source,
                    name=name,
                    batch_size=batch_size,
                    dtype=dtype,
                    device=device,
                )
            normalized_globals.append(
                _normalize_with_stats(
                    values,
                    method=self.global_methods[name],
                    epsilon=epsilon,
                    mean=self._global_stat_tensor(name, "mean", dtype=dtype, device=device),
                    std=self._global_stat_tensor(name, "std", dtype=dtype, device=device),
                    lower=self._global_stat_tensor(name, "min", dtype=dtype, device=device),
                    upper=self._global_stat_tensor(name, "max", dtype=dtype, device=device),
                )
            )

        sequence_inputs = torch.cat(
            [pressure_norm.unsqueeze(-1), temperature_norm.unsqueeze(-1), kzz_norm.unsqueeze(-1), anchor_norm],
            dim=-1,
        )
        conditioning_inputs = torch.stack(normalized_globals, dim=-1)
        normalized_prediction = self.model(sequence_inputs, conditioning_inputs, padding_mask=None)
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
    """Load a trained checkpoint and wrap it with physical-space IO transforms."""
    resolved_run_dir = Path(run_dir).resolve()
    checkpoint_path = resolved_run_dir / checkpoint_name
    if not checkpoint_path.is_file():
        raise InferenceError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise InferenceError(f"Invalid checkpoint payload: {checkpoint_path}")

    normalization_metadata = _load_dict_from_checkpoint_or_disk(
        checkpoint=checkpoint,
        key="normalization_metadata",
        path=resolved_run_dir / "normalization_metadata.json",
    )
    data_contract = _load_dict_from_checkpoint_or_disk(
        checkpoint=checkpoint,
        key="data_contract",
        path=resolved_run_dir / "data_contract.json",
    )
    config = checkpoint["config"]
    model_cfg = config["training"]["model"]
    precision_cfg = config["precision"]

    model = VulcanTransitionTransformer(
        state_dim=int(data_contract["state_dim"]),
        output_dim=int(data_contract["target_dim"]),
        output_from_state_indices=list(data_contract["output_from_state_indices"]),
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        num_layers=int(model_cfg["num_layers"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        dropout=float(model_cfg["dropout"]),
        film_clamp=float(model_cfg["film_clamp"]),
        output_head_divisor=int(model_cfg["output_head_divisor"]),
        max_sequence_length=int(model_cfg["max_sequence_length"]),
        conditioning_hidden_dim=int(model_cfg["conditioning_hidden_dim"]),
        num_globals=int(data_contract["global_dim"]),
    ).to(device=device, dtype=_torch_dtype_from_name(str(precision_cfg["model_dtype"])))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    wrapper = PhysicalSpaceStandaloneModel(
        model=model,
        normalization_metadata=normalization_metadata,
        data_contract=data_contract,
        default_global_inputs=static_conditioning_defaults(config),
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
    """Invert processed arrays back into physical-space inference inputs."""
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
            "Processed sequence/global arrays must have ranks [batch, nz, input_dim] and [batch, global_dim]."
        )
    if seq.shape[0] != glb.shape[0]:
        raise InferenceError("Processed sequence/global arrays must share the same batch dimension.")

    state_species = list(data_contract["state_species_order"])
    output_species = list(data_contract["output_species_order"])
    if int(seq.shape[-1]) != int(data_contract["input_dim"]):
        raise InferenceError("Processed sequence array does not match data_contract input_dim.")
    if int(glb.shape[-1]) != int(data_contract["global_dim"]):
        raise InferenceError("Processed global array does not match data_contract global_dim.")
    if int(seq.shape[1]) != int(data_contract["sequence_length"]):
        raise InferenceError("Processed sequence array does not match data_contract sequence_length.")

    expected_sequence_order = [
        "pressure_bar",
        "temperature_k",
        "kzz_cm2_s",
        *[f"anchor_ymix:{species_name}" for species_name in state_species],
    ]
    if list(data_contract["sequence_feature_order"]) != expected_sequence_order:
        raise InferenceError("data_contract sequence_feature_order is incompatible with transition inference.")

    pressure = _denormalize_numpy(seq[..., 0], normalization_metadata["sequence"]["pressure_bar"])
    temperature = _denormalize_numpy(seq[..., 1], normalization_metadata["sequence"]["temperature_k"])
    kzz = _denormalize_numpy(seq[..., 2], normalization_metadata["sequence"]["kzz_cm2_s"])
    anchor_ymix = _denormalize_numpy(seq[..., 3:], normalization_metadata["sequence"]["anchor_ymix"])
    global_values = {
        name: _denormalize_numpy(glb[..., idx], normalization_metadata["globals"][name])
        for idx, name in enumerate(data_contract["global_feature_order"])
    }
    log10_dt_s = np.asarray(global_values.pop("log10_dt_s"), dtype=np.float64)
    dt_s = np.power(10.0, log10_dt_s)

    base_globals = {
        key: np.asarray(global_values.pop(key), dtype=np.float64)
        for key in ("gravity_cm_s2", "metallicity_log10", "c_to_o")
    }

    result = {
        "pressure_bar": pressure,
        "temperature_k": temperature,
        "kzz_cm2_s": kzz,
        "anchor_ymix": anchor_ymix,
        "gravity_cm_s2": base_globals["gravity_cm_s2"],
        "metallicity_log10": base_globals["metallicity_log10"],
        "c_to_o": base_globals["c_to_o"],
        "dt_s": dt_s,
        "state_species": state_species,
        "output_species": output_species,
    }
    if global_values:
        result["extra_global_inputs"] = {key: np.asarray(value, dtype=np.float64) for key, value in global_values.items()}
    if squeeze_batch:
        squeezed = {
            "pressure_bar": pressure[0],
            "temperature_k": temperature[0],
            "kzz_cm2_s": kzz[0],
            "anchor_ymix": anchor_ymix[0],
            "gravity_cm_s2": base_globals["gravity_cm_s2"][0],
            "metallicity_log10": base_globals["metallicity_log10"][0],
            "c_to_o": base_globals["c_to_o"][0],
            "dt_s": dt_s[0],
            "state_species": state_species,
            "output_species": output_species,
        }
        if global_values:
            squeezed["extra_global_inputs"] = {
                key: np.asarray(value, dtype=np.float64)[0]
                for key, value in global_values.items()
            }
        return squeezed
    return result


class VulcanPredictor:
    """Convenience NumPy-facing predictor for physical-space inference.

    Wraps :class:`PhysicalSpaceStandaloneModel` with automatic numpy-to-torch
    conversion, batch dimension handling, and device placement for single-step
    transition prediction.
    """

    def __init__(self, model: PhysicalSpaceStandaloneModel, *, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.state_species = tuple(model.state_species)
        self.target_species = tuple(model.target_species)

    @classmethod
    def from_run_dir(
        cls,
        run_dir: Path,
        *,
        checkpoint_name: str = "best.pt",
        device: torch.device | str = "cpu",
    ) -> "VulcanPredictor":
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
        anchor_ymix: Any,
        gravity_cm_s2: Any,
        metallicity_log10: Any,
        c_to_o: Any,
        dt_s: Any,
        extra_global_inputs: dict[str, Any] | None = None,
    ) -> np.ndarray:
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
        anchor_np = self._coerce_anchor_ymix(
            anchor_ymix,
            batch_size=pressure_np.shape[0],
            sequence_length=pressure_np.shape[1],
        )
        gravity_np = self._coerce_globals(gravity_cm_s2, "gravity_cm_s2", batch_size=pressure_np.shape[0])
        metallicity_np = self._coerce_globals(metallicity_log10, "metallicity_log10", batch_size=pressure_np.shape[0])
        c_to_o_np = self._coerce_globals(c_to_o, "c_to_o", batch_size=pressure_np.shape[0])
        dt_np = self._coerce_globals(dt_s, "dt_s", batch_size=pressure_np.shape[0])
        if np.any(dt_np <= 0.0):
            raise InferenceError("dt_s must be strictly positive for log10 conditioning.")
        extras_np = {
            name: self._coerce_globals(value, name, batch_size=pressure_np.shape[0])
            for name, value in dict(extra_global_inputs or {}).items()
        }

        model_dtype = next(self.model.parameters()).dtype
        with torch.inference_mode():
            prediction = self.model(
                torch.as_tensor(pressure_np, dtype=model_dtype, device=self.device),
                torch.as_tensor(temperature_np, dtype=model_dtype, device=self.device),
                torch.as_tensor(kzz_np, dtype=model_dtype, device=self.device),
                torch.as_tensor(anchor_np, dtype=model_dtype, device=self.device),
                torch.as_tensor(gravity_np, dtype=model_dtype, device=self.device),
                torch.as_tensor(metallicity_np, dtype=model_dtype, device=self.device),
                torch.as_tensor(c_to_o_np, dtype=model_dtype, device=self.device),
                torch.as_tensor(dt_np, dtype=model_dtype, device=self.device),
                extra_global_inputs={
                    name: torch.as_tensor(value, dtype=model_dtype, device=self.device)
                    for name, value in extras_np.items()
                } if extras_np else None,
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
        array = np.asarray(values, dtype=np.float64)
        squeeze_batch = False
        if array.ndim == 1:
            array = array[None, :]
            squeeze_batch = True
        if array.ndim != 2:
            raise InferenceError(f"{name} must have shape [nz] or [batch, nz].")
        if batch_size is not None and int(array.shape[0]) != batch_size:
            raise InferenceError(f"{name} batch size does not match the other profile inputs.")
        return array, squeeze_batch

    def _coerce_anchor_ymix(
        self,
        values: Any,
        *,
        batch_size: int,
        sequence_length: int,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3:
            raise InferenceError(
                "anchor_ymix must have shape [nz, species] or [batch, nz, species]."
            )
        if int(array.shape[0]) != batch_size or int(array.shape[1]) != sequence_length:
            raise InferenceError(
                "anchor_ymix must match the batch and sequence dimensions of the profiles."
            )
        if int(array.shape[2]) != len(self.state_species):
            raise InferenceError(
                "anchor_ymix species dimension must equal "
                f"{len(self.state_species)} configured state species."
            )
        return array

    def _coerce_globals(self, values: Any, name: str, *, batch_size: int) -> np.ndarray:
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
