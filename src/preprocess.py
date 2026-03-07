"""Generation post-processing: split, sample transition pairs, normalize, and shard.

Orchestrates the full ``--gen`` pipeline after VULCAN runs complete:

1. **Run VULCAN jobs** from sampled RunSpecs (P, T, Kzz, gravity, abundances).
2. **Split** raw run IDs into train/val/test by configurable ratios.
3. **Sample transition pairs** (anchor/target time pairs) per run.
4. **Fit normalization stats** on the *train* split only, applying the
   configured method (standard, log-standard, log-min-max, none) per variable.
5. **Normalize and shard** all splits into fixed-size ``.npy`` files for
   efficient DataLoader consumption.

Key invariants:

- Stats are fitted exclusively on the training split to prevent data leakage.
- Target normalization shares stats with ``anchor_ymix`` (subset by output
  species indices) so the residual skip operates in one consistent space.
- All base-10 log transforms use ``np.log10`` with an epsilon floor.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from config_utils import PrecisionConfig
from provenance import PROCESSED_FINGERPRINT_FILENAME, build_processed_fingerprint
from sampling import build_run_specs
from transition_sampling import TransitionPairs, sample_transition_pairs
from vulcan_runner import (
    BoundaryConditionSettings,
    SpeciesSelection,
    VulcanRuntimeError,
    WorkerSettings,
    run_vulcan_jobs,
)

logger = logging.getLogger(__name__)


class PreprocessError(RuntimeError):
    """Raised when preprocessing contracts are violated."""


@dataclass(frozen=True)
class SplitAssignments:
    """Run-id assignments for train/val/test splits."""

    train: list[int]
    val: list[int]
    test: list[int]


@dataclass
class _RunningStats:
    """Streaming stats accumulator over scalar or channel vectors."""

    channels: int
    dtype: type[np.float32] | type[np.float64]

    def __post_init__(self) -> None:
        self.count = 0
        self.sum = np.zeros(self.channels, dtype=self.dtype)
        self.sumsq = np.zeros(self.channels, dtype=self.dtype)
        self.min = np.full(self.channels, np.inf, dtype=self.dtype)
        self.max = np.full(self.channels, -np.inf, dtype=self.dtype)

    def update(self, values: np.ndarray) -> None:
        data = np.asarray(values, dtype=self.dtype)
        if data.ndim == 1:
            data = data[:, None]
        if data.shape[-1] != self.channels:
            raise PreprocessError(
                f"Stats update channel mismatch: expected {self.channels}, got {data.shape[-1]}"
            )
        flat = data.reshape(-1, self.channels)
        if np.any(~np.isfinite(flat)):
            raise PreprocessError("Non-finite values encountered while fitting normalization stats.")
        self.count += int(flat.shape[0])
        self.sum += np.sum(flat, axis=0)
        self.sumsq += np.sum(flat * flat, axis=0)
        self.min = np.minimum(self.min, np.min(flat, axis=0))
        self.max = np.maximum(self.max, np.max(flat, axis=0))

    def finalize(self) -> dict[str, Any]:
        if self.count <= 0:
            raise PreprocessError("Cannot finalize stats with zero samples.")
        mean = self.sum / self.count
        var = np.maximum((self.sumsq / self.count) - mean * mean, 0.0)
        std = np.sqrt(var)
        std = np.where(std > 0.0, std, 1.0)
        return {
            "count": int(self.count),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }


@dataclass(frozen=True)
class _NormPolicy:
    """Normalization policy for one variable family."""

    method: str
    epsilon: float


@dataclass(frozen=True)
class RawRunData:
    """Validated raw trajectory payload loaded from one HDF5 run file."""

    run_id: int
    pressure_bar: np.ndarray
    temperature_k: np.ndarray
    kzz_cm2_s: np.ndarray
    time_s: np.ndarray
    ymix_state: np.ndarray
    ymix_output: np.ndarray
    gravity_cm_s2: float
    metallicity_log10: float
    c_to_o: float


@dataclass(frozen=True)
class PairSpecBundle:
    """One raw run file paired with sampled transition metadata."""

    run_file: Path
    pairs: TransitionPairs


def _log10_safe(values: np.ndarray, eps: float) -> np.ndarray:
    return np.log10(np.maximum(values, eps))


def _fit_for_method(values: np.ndarray, policy: _NormPolicy) -> np.ndarray:
    if policy.method in {"log-standard", "log-min-max"}:
        return _log10_safe(values, policy.epsilon)
    if policy.method in {"standard", "none"}:
        return np.asarray(values, dtype=np.float64)
    raise PreprocessError(f"Unsupported normalization method: {policy.method}")


def _apply_method(values: np.ndarray, stats: dict[str, Any], policy: _NormPolicy) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if policy.method == "none":
        return data
    if policy.method == "standard":
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        return (data - mean) / std
    if policy.method == "log-standard":
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        return (_log10_safe(data, policy.epsilon) - mean) / std
    if policy.method == "log-min-max":
        lower = np.asarray(stats["min"], dtype=np.float64)
        upper = np.asarray(stats["max"], dtype=np.float64)
        width = np.where((upper - lower) > 0.0, upper - lower, 1.0)
        return (_log10_safe(data, policy.epsilon) - lower) / width
    raise PreprocessError(f"Unsupported normalization method: {policy.method}")


def _numpy_stats_dtype(precision: PrecisionConfig) -> type[np.float32] | type[np.float64]:
    return np.float64 if precision.stats_dtype == torch.float64 else np.float32


def _read_species_dataset(dataset: h5py.Dataset) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in dataset[()]]


def load_raw_run_file(
    path: Path,
    *,
    state_species: list[str],
    output_species: list[str],
) -> RawRunData:
    """Load one raw run HDF5 file and validate the transition-model science contract."""
    if not path.is_file():
        raise PreprocessError(f"Missing run file: {path}")

    with h5py.File(path, "r") as handle:
        for required_group in ("inputs", "globals", "trajectory"):
            if required_group not in handle:
                raise PreprocessError(f"Run file missing group '{required_group}': {path}")

        inputs = handle["inputs"]
        globals_group = handle["globals"]
        trajectory = handle["trajectory"]

        found_state_species = _read_species_dataset(inputs["state_species"])
        found_output_species = _read_species_dataset(inputs["output_species"])
        if found_state_species != state_species:
            raise PreprocessError(
                f"State species mismatch in {path}. Expected {state_species}, found {found_state_species}"
            )
        if found_output_species != output_species:
            raise PreprocessError(
                f"Output species mismatch in {path}. Expected {output_species}, found {found_output_species}"
            )

        pressure = np.asarray(inputs["pressure_bar"], dtype=np.float64)
        temperature = np.asarray(inputs["temperature_k"], dtype=np.float64)
        kzz = np.asarray(inputs["kzz_cm2_s"], dtype=np.float64)
        time_s = np.asarray(trajectory["time_s"], dtype=np.float64)
        ymix_state = np.asarray(trajectory["ymix_state"], dtype=np.float64)
        ymix_output = np.asarray(trajectory["ymix_output"], dtype=np.float64)
        gravity = float(np.asarray(globals_group["gravity_cm_s2"]))
        metallicity_log10 = float(np.asarray(globals_group["metallicity_log10"]))
        c_to_o = float(np.asarray(globals_group["c_to_o"]))
        run_id = int(handle.attrs["run_id"])

    arrays_to_check = [pressure, temperature, kzz, time_s, ymix_state, ymix_output]
    if any(np.any(~np.isfinite(array)) for array in arrays_to_check):
        raise PreprocessError(f"Non-finite values found in raw run file: {path}")
    if pressure.ndim != 1 or temperature.ndim != 1 or kzz.ndim != 1:
        raise PreprocessError(f"Invalid sequence shapes in {path}")
    if not (pressure.shape == temperature.shape == kzz.shape):
        raise PreprocessError(f"Sequence length mismatch in {path}")
    if time_s.ndim != 1 or time_s.size < 2:
        raise PreprocessError(f"trajectory/time_s must contain at least t=0 and one future state in {path}")
    if np.any(np.diff(time_s) <= 0.0):
        raise PreprocessError(f"trajectory/time_s must be strictly increasing in {path}")

    nz = int(pressure.size)
    n_state = len(state_species)
    n_output = len(output_species)
    if ymix_state.shape != (time_s.size, nz, n_state):
        raise PreprocessError(f"ymix_state shape mismatch in {path}: {ymix_state.shape}")
    if ymix_output.shape != (time_s.size, nz, n_output):
        raise PreprocessError(f"ymix_output shape mismatch in {path}: {ymix_output.shape}")
    if np.any(time_s < 0.0):
        raise PreprocessError(f"trajectory/time_s must be >= 0 in {path}")

    return RawRunData(
        run_id=run_id,
        pressure_bar=pressure,
        temperature_k=temperature,
        kzz_cm2_s=kzz,
        time_s=time_s,
        ymix_state=ymix_state,
        ymix_output=ymix_output,
        gravity_cm_s2=gravity,
        metallicity_log10=metallicity_log10,
        c_to_o=c_to_o,
    )


def _split_run_ids(
    run_ids: list[int],
    split_ratios: dict[str, float],
    seed: int,
) -> SplitAssignments:
    """Partition run IDs into train/val/test splits with no overlap.

    Splitting by run ID (not by individual transition pair) prevents
    snapshot leakage: all pairs from one VULCAN trajectory stay in the
    same split, so the model never sees a target from a training run
    during evaluation.
    """
    if not run_ids:
        raise PreprocessError("Cannot create splits from empty run id list.")
    rng = np.random.default_rng(seed)
    shuffled = np.array(sorted(run_ids), dtype=np.int64)
    rng.shuffle(shuffled)

    n_total = int(shuffled.size)
    n_train = int(n_total * float(split_ratios["train"]))
    n_val = int(n_total * float(split_ratios["val"]))
    n_test = n_total - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise PreprocessError(
            f"Split produced empty partition: train={n_train}, val={n_val}, test={n_test}."
        )

    train = shuffled[:n_train].tolist()
    val = shuffled[n_train : n_train + n_val].tolist()
    test = shuffled[n_train + n_val :].tolist()
    overlap = (set(train) & set(val)) | (set(train) & set(test)) | (set(val) & set(test))
    if overlap:
        raise PreprocessError(f"Split leakage detected across run ids: {sorted(overlap)}")
    return SplitAssignments(train=train, val=val, test=test)


def _sample_pairs_for_run(
    *,
    raw: RawRunData,
    config: dict[str, Any],
    random_seed: int,
) -> TransitionPairs:
    sampling_cfg = config["trajectory_sampling"]
    rng = np.random.default_rng(int(random_seed) + int(raw.run_id) * 7919)
    return sample_transition_pairs(
        rng=rng,
        times_s=raw.time_s,
        n_pairs=int(sampling_cfg["pairs_per_run"]),
        requested_dt_min_s=float(sampling_cfg["requested_dt_min_s"]),
        requested_dt_max_s=float(sampling_cfg["requested_dt_max_s"]),
        allow_anchor_at_t0=bool(sampling_cfg["allow_anchor_at_t0"]),
        min_future_saved_steps=int(sampling_cfg["min_future_saved_steps"]),
    )


def _build_pair_specs(
    *,
    run_file_by_id: dict[int, Path],
    split_ids: list[int],
    config: dict[str, Any],
    state_species: list[str],
    output_species: list[str],
) -> list[PairSpecBundle]:
    bundles: list[PairSpecBundle] = []
    seed = int(config["generation"]["random_seed"])
    for run_id in split_ids:
        raw = load_raw_run_file(
            run_file_by_id[run_id],
            state_species=state_species,
            output_species=output_species,
        )
        pairs = _sample_pairs_for_run(raw=raw, config=config, random_seed=seed)
        bundles.append(PairSpecBundle(run_file=run_file_by_id[run_id], pairs=pairs))
    return bundles


def _subset_stats(stats: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    subset: dict[str, Any] = {"count": int(stats["count"])}
    for field_name in ("mean", "std", "min", "max"):
        values = np.asarray(stats[field_name], dtype=np.float64)
        subset[field_name] = values[indices].tolist()
    return subset


def _build_normalization_stats(
    *,
    train_bundles: list[PairSpecBundle],
    config: dict[str, Any],
    state_species: list[str],
    output_species: list[str],
    output_from_state_indices: list[int],
    stats_dtype: type[np.float32] | type[np.float64],
) -> dict[str, Any]:
    """Fit normalization statistics from the training split only.

    Streaming accumulators compute mean/std/min/max over the *transformed*
    values (e.g., log10 for log-standard) so that normalization operates
    on the scale the model actually sees.  Target stats are derived from
    the anchor_ymix stats, subsetted to the output species channels, which
    ensures the residual skip connection works in a single shared space.
    """
    norm_cfg = config["normalization"]
    epsilon = float(norm_cfg["epsilon"])

    seq_methods = norm_cfg["sequence_methods"]
    global_methods = norm_cfg["global_methods"]
    target_method = str(norm_cfg["target_method"])

    seq_stats = {
        "pressure_bar": (_NormPolicy(seq_methods["pressure_bar"], epsilon), _RunningStats(1, stats_dtype)),
        "temperature_k": (_NormPolicy(seq_methods["temperature_k"], epsilon), _RunningStats(1, stats_dtype)),
        "kzz_cm2_s": (_NormPolicy(seq_methods["kzz_cm2_s"], epsilon), _RunningStats(1, stats_dtype)),
    }
    state_policy = _NormPolicy(seq_methods["anchor_ymix"], epsilon)
    state_stats = _RunningStats(len(state_species), stats_dtype)
    global_stats = {
        "gravity_cm_s2": (_NormPolicy(global_methods["gravity_cm_s2"], epsilon), _RunningStats(1, stats_dtype)),
        "metallicity_log10": (_NormPolicy(global_methods["metallicity_log10"], epsilon), _RunningStats(1, stats_dtype)),
        "c_to_o": (_NormPolicy(global_methods["c_to_o"], epsilon), _RunningStats(1, stats_dtype)),
        "log10_dt_s": (_NormPolicy(global_methods["log10_dt_s"], epsilon), _RunningStats(1, stats_dtype)),
    }

    for bundle in train_bundles:
        raw = load_raw_run_file(
            bundle.run_file,
            state_species=state_species,
            output_species=output_species,
        )
        pairs = bundle.pairs

        for key in ("pressure_bar", "temperature_k", "kzz_cm2_s"):
            policy, acc = seq_stats[key]
            values = getattr(raw, key).reshape(-1, 1)
            acc.update(_fit_for_method(values, policy))

        anchor_states = raw.ymix_state[pairs.anchor_index]
        target_states = raw.ymix_state[pairs.target_index]
        state_stats.update(_fit_for_method(anchor_states, state_policy))
        state_stats.update(_fit_for_method(target_states, state_policy))

        for key, value in {
            "gravity_cm_s2": raw.gravity_cm_s2,
            "metallicity_log10": raw.metallicity_log10,
            "c_to_o": raw.c_to_o,
        }.items():
            policy, acc = global_stats[key]
            repeated = np.full((pairs.actual_dt_s.size, 1), float(value), dtype=np.float64)
            acc.update(_fit_for_method(repeated, policy))

        dt_policy, dt_acc = global_stats["log10_dt_s"]
        log10_dt_s = np.log10(pairs.actual_dt_s).reshape(-1, 1)
        dt_acc.update(_fit_for_method(log10_dt_s, dt_policy))

    state_final = state_stats.finalize()
    output_final = _subset_stats(state_final, output_from_state_indices)
    return {
        "epsilon": epsilon,
        "sequence": {
            key: {"method": policy.method, **acc.finalize()}
            for key, (policy, acc) in seq_stats.items()
        }
        | {"anchor_ymix": {"method": state_policy.method, **state_final}},
        "globals": {
            key: {"method": policy.method, **acc.finalize()}
            for key, (policy, acc) in global_stats.items()
        },
        "targets": {"ymix": {"method": target_method, **output_final}},
    }


def _write_split_shards(
    *,
    split_name: str,
    bundles: list[PairSpecBundle],
    config: dict[str, Any],
    state_species: list[str],
    output_species: list[str],
    output_from_state_indices: list[int],
    stats: dict[str, Any],
    normalization_fingerprint: str,
    processed_root: Path,
) -> dict[str, Any]:
    """Normalize one split and write fixed-size `.npy` shards plus metadata."""
    split_dir = processed_root / split_name
    seq_dir = split_dir / "sequence_inputs"
    glb_dir = split_dir / "globals"
    tgt_dir = split_dir / "targets"
    dt_dir = split_dir / "dt_s"
    for directory in (split_dir, seq_dir, glb_dir, tgt_dir, dt_dir):
        directory.mkdir(parents=True, exist_ok=True)

    state_dim = len(state_species)
    output_dim = len(output_species)
    input_dim = 3 + state_dim
    global_dim = 4
    shard_size = int(config["generation"]["shard_size"])

    sequence_length = -1
    total_samples = 0
    shard_idx = 0
    write_pos = 0
    dt_min_s = np.inf
    dt_max_s = -np.inf

    seq_buf: np.ndarray | None = None
    glb_buf: np.ndarray | None = None
    tgt_buf: np.ndarray | None = None
    dt_buf: np.ndarray | None = None

    def alloc_buffers(nz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.zeros((shard_size, nz, input_dim), dtype=np.float32),
            np.zeros((shard_size, global_dim), dtype=np.float32),
            np.zeros((shard_size, nz, output_dim), dtype=np.float32),
            np.zeros((shard_size,), dtype=np.float32),
        )

    def flush(current_size: int) -> None:
        nonlocal shard_idx, seq_buf, glb_buf, tgt_buf, dt_buf
        if current_size <= 0 or seq_buf is None or glb_buf is None or tgt_buf is None or dt_buf is None:
            return
        np.save(seq_dir / f"shard_{shard_idx:05d}.npy", seq_buf[:current_size], allow_pickle=False)
        np.save(glb_dir / f"shard_{shard_idx:05d}.npy", glb_buf[:current_size], allow_pickle=False)
        np.save(tgt_dir / f"shard_{shard_idx:05d}.npy", tgt_buf[:current_size], allow_pickle=False)
        np.save(dt_dir / f"shard_{shard_idx:05d}.npy", dt_buf[:current_size], allow_pickle=False)
        shard_idx += 1

    eps = float(stats["epsilon"])
    seq_policies = {
        key: _NormPolicy(stats["sequence"][key]["method"], eps)
        for key in ("pressure_bar", "temperature_k", "kzz_cm2_s", "anchor_ymix")
    }
    global_policies = {
        key: _NormPolicy(stats["globals"][key]["method"], eps)
        for key in ("gravity_cm_s2", "metallicity_log10", "c_to_o", "log10_dt_s")
    }
    target_policy = _NormPolicy(stats["targets"]["ymix"]["method"], eps)

    for bundle in bundles:
        raw = load_raw_run_file(
            bundle.run_file,
            state_species=state_species,
            output_species=output_species,
        )
        pairs = bundle.pairs
        nz = int(raw.pressure_bar.size)
        if sequence_length < 0:
            sequence_length = nz
            seq_buf, glb_buf, tgt_buf, dt_buf = alloc_buffers(nz)
        elif sequence_length != nz:
            raise PreprocessError(
                "Mixed sequence lengths across runs are unsupported in v1: "
                f"{sequence_length} vs {nz}"
            )

        pressure_norm = _apply_method(
            raw.pressure_bar.reshape(nz, 1),
            stats["sequence"]["pressure_bar"],
            seq_policies["pressure_bar"],
        )
        temperature_norm = _apply_method(
            raw.temperature_k.reshape(nz, 1),
            stats["sequence"]["temperature_k"],
            seq_policies["temperature_k"],
        )
        kzz_norm = _apply_method(
            raw.kzz_cm2_s.reshape(nz, 1),
            stats["sequence"]["kzz_cm2_s"],
            seq_policies["kzz_cm2_s"],
        )
        static_block = np.concatenate([pressure_norm, temperature_norm, kzz_norm], axis=1)

        anchor_states = raw.ymix_state[pairs.anchor_index]
        target_outputs = raw.ymix_output[pairs.target_index]
        anchor_norm = _apply_method(
            anchor_states,
            stats["sequence"]["anchor_ymix"],
            seq_policies["anchor_ymix"],
        )
        target_norm = _apply_method(
            target_outputs,
            stats["targets"]["ymix"],
            target_policy,
        )
        sequence_block = np.concatenate(
            [np.broadcast_to(static_block[None, :, :], (anchor_norm.shape[0], nz, 3)), anchor_norm],
            axis=2,
        )

        gravity_norm = float(
            _apply_method(
                np.array([[raw.gravity_cm_s2]], dtype=np.float64),
                stats["globals"]["gravity_cm_s2"],
                global_policies["gravity_cm_s2"],
            )[0, 0]
        )
        metallicity_norm = float(
            _apply_method(
                np.array([[raw.metallicity_log10]], dtype=np.float64),
                stats["globals"]["metallicity_log10"],
                global_policies["metallicity_log10"],
            )[0, 0]
        )
        c_to_o_norm = float(
            _apply_method(
                np.array([[raw.c_to_o]], dtype=np.float64),
                stats["globals"]["c_to_o"],
                global_policies["c_to_o"],
            )[0, 0]
        )
        dt_norm = _apply_method(
            np.log10(pairs.actual_dt_s).reshape(-1, 1),
            stats["globals"]["log10_dt_s"],
            global_policies["log10_dt_s"],
        ).reshape(-1)
        global_block = np.column_stack(
            [
                np.full(pairs.actual_dt_s.size, gravity_norm, dtype=np.float64),
                np.full(pairs.actual_dt_s.size, metallicity_norm, dtype=np.float64),
                np.full(pairs.actual_dt_s.size, c_to_o_norm, dtype=np.float64),
                dt_norm,
            ]
        )

        dt_min_s = min(dt_min_s, float(np.min(pairs.actual_dt_s)))
        dt_max_s = max(dt_max_s, float(np.max(pairs.actual_dt_s)))

        start = 0
        num_pairs = int(pairs.actual_dt_s.size)
        while start < num_pairs:
            if seq_buf is None or glb_buf is None or tgt_buf is None or dt_buf is None:
                raise PreprocessError("Internal buffer allocation failure.")
            if write_pos >= shard_size:
                flush(shard_size)
                write_pos = 0
            room = shard_size - write_pos
            n_copy = min(room, num_pairs - start)
            end_pos = write_pos + n_copy
            next_start = start + n_copy

            seq_buf[write_pos:end_pos] = sequence_block[start:next_start].astype(np.float32, copy=False)
            glb_buf[write_pos:end_pos] = global_block[start:next_start].astype(np.float32, copy=False)
            tgt_buf[write_pos:end_pos] = target_norm[start:next_start].astype(np.float32, copy=False)
            dt_buf[write_pos:end_pos] = pairs.actual_dt_s[start:next_start].astype(np.float32, copy=False)

            write_pos = end_pos
            start = next_start
            total_samples += n_copy

    flush(write_pos)

    metadata = {
        "split": split_name,
        "total_samples": int(total_samples),
        "num_shards": int(shard_idx),
        "sequence_length": int(sequence_length),
        "input_dim": int(input_dim),
        "global_dim": int(global_dim),
        "target_dim": int(output_dim),
        "state_dim": int(state_dim),
        "sequence_feature_order": [
            "pressure_bar",
            "temperature_k",
            "kzz_cm2_s",
            *[f"anchor_ymix:{species}" for species in state_species],
        ],
        "global_feature_order": ["gravity_cm_s2", "metallicity_log10", "c_to_o", "log10_dt_s"],
        "state_species_order": list(state_species),
        "output_species_order": list(output_species),
        "output_from_state_indices": list(output_from_state_indices),
        "normalization_fingerprint": normalization_fingerprint,
        "dt_min_s": float(dt_min_s),
        "dt_max_s": float(dt_max_s),
    }
    with (split_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def run_generation_and_preprocess(
    config: dict[str, Any],
    paths: Any,
    *,
    precision: PrecisionConfig,
    boundary_conditions: BoundaryConditionSettings | None,
) -> None:
    """Execute `--gen`: generate raw trajectories, split by run, normalize, and write shards."""
    generation = config["generation"]
    state_species = list(config["data_spec"]["state_species"])
    output_species = list(config["data_spec"]["output_species"])
    output_from_state_indices = [state_species.index(species) for species in output_species]
    stats_dtype = _numpy_stats_dtype(precision)

    run_specs = build_run_specs(config)
    if not run_specs:
        raise PreprocessError("Run-spec generation returned zero runs.")

    settings = WorkerSettings(
        vulcan_source=str(paths.vulcan_source),
        worker_root=str((paths.root / generation["worker_root"]).resolve()),
        runs_root=str((paths.root / generation["runs_root"]).resolve()),
        species=SpeciesSelection(
            state_species=tuple(state_species),
            output_species=tuple(output_species),
        ),
        use_transport=bool(config["physics_toggles"]["use_transport"]),
        boundary_conditions=boundary_conditions,
        use_condensation_optional=bool(config["physics_toggles"]["use_condensation_optional"]),
        save_evo_frq=int(generation["save_evo_frq"]),
        keep_vulcan_outputs_debug=bool(generation["keep_vulcan_outputs_debug"]),
        run_timeout_seconds=int(generation["run_timeout_seconds"]),
        num_workers=int(generation["num_workers"]),
        runtime=float(config["vulcan_runtime"]["runtime"]),
        dt_min=float(config["vulcan_runtime"]["dt_min"]),
        dt_max=float(config["vulcan_runtime"]["dt_max"]),
        count_max=int(config["vulcan_runtime"]["count_max"]),
        trun_min=float(config["vulcan_runtime"]["trun_min"]),
        count_min=int(config["vulcan_runtime"]["count_min"]),
        y_time_freq=int(config["vulcan_runtime"]["y_time_freq"]),
    )

    logger.info("Running %d VULCAN jobs with %d workers...", len(run_specs), generation["num_workers"])
    try:
        results = run_vulcan_jobs(run_specs, settings=settings)
    except VulcanRuntimeError as exc:
        raise PreprocessError(str(exc)) from exc

    run_files = [Path(result.run_file) for result in results]
    run_ids = [int(result.run_id) for result in results]
    split = _split_run_ids(
        run_ids=run_ids,
        split_ratios=generation["split_ratios"],
        seed=int(generation["random_seed"]),
    )
    split_map = {"train": split.train, "val": split.val, "test": split.test}
    run_file_by_id = {int(result.run_id): Path(result.run_file) for result in results}

    logger.info("Sampling transition pairs from raw trajectories...")
    split_bundles = {
        split_name: _build_pair_specs(
            run_file_by_id=run_file_by_id,
            split_ids=split_ids,
            config=config,
            state_species=state_species,
            output_species=output_species,
        )
        for split_name, split_ids in split_map.items()
    }

    logger.info("Fitting normalization stats from train split (%d runs)...", len(split.train))
    stats = _build_normalization_stats(
        train_bundles=split_bundles["train"],
        config=config,
        state_species=state_species,
        output_species=output_species,
        output_from_state_indices=output_from_state_indices,
        stats_dtype=stats_dtype,
    )
    normalization_fingerprint = sha256(
        json.dumps(stats, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    processed_root = paths.processed_root
    if processed_root.exists():
        shutil.rmtree(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    logger.info("Writing processed transition shards...")
    split_metadata = {}
    for split_name in ("train", "val", "test"):
        split_metadata[split_name] = _write_split_shards(
            split_name=split_name,
            bundles=split_bundles[split_name],
            config=config,
            state_species=state_species,
            output_species=output_species,
            output_from_state_indices=output_from_state_indices,
            stats=stats,
            normalization_fingerprint=normalization_fingerprint,
            processed_root=processed_root,
        )

    manifest = {
        "num_runs": len(results),
        "run_files": [str(path.relative_to(paths.root)) for path in sorted(run_files)],
        "split": split_map,
        "state_species": state_species,
        "output_species": output_species,
    }
    manifest_path = paths.data_root / str(generation["manifest_filename"])
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    split_path = paths.data_root / str(generation["split_filename"])
    with split_path.open("w", encoding="utf-8") as handle:
        json.dump(split_map, handle, indent=2)

    norm_path = processed_root / "normalization_metadata.json"
    with norm_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    summary_path = processed_root / "processed_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(split_metadata, handle, indent=2)

    fingerprint = build_processed_fingerprint(
        config=config,
        project_root=paths.root,
        raw_run_files=sorted(run_files),
        manifest_path=manifest_path,
        split_path=split_path,
        normalization_path=norm_path,
        summary_path=summary_path,
        split_metadata_paths={
            split_name: processed_root / split_name / "metadata.json"
            for split_name in ("train", "val", "test")
        },
    )
    fingerprint_path = processed_root / PROCESSED_FINGERPRINT_FILENAME
    with fingerprint_path.open("w", encoding="utf-8") as handle:
        json.dump(fingerprint, handle, indent=2)

    logger.info(
        "Generation + preprocessing complete. Raw runs: %d | train pairs: %d | val pairs: %d | test pairs: %d",
        len(results),
        split_metadata["train"]["total_samples"],
        split_metadata["val"]["total_samples"],
        split_metadata["test"]["total_samples"],
    )
