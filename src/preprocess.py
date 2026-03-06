"""Generation post-processing: split, normalize, shard writing."""

from __future__ import annotations

import json
import logging
import shutil
from hashlib import sha256
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from config_utils import PrecisionConfig
from provenance import PROCESSED_FINGERPRINT_FILENAME, build_processed_fingerprint
from sampling import build_run_specs
from vulcan_runner import (
    BoundaryConditionSettings,
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
        """Allocate accumulator buffers in the configured stats dtype."""
        self.count = 0
        self.sum = np.zeros(self.channels, dtype=self.dtype)
        self.sumsq = np.zeros(self.channels, dtype=self.dtype)
        self.min = np.full(self.channels, np.inf, dtype=self.dtype)
        self.max = np.full(self.channels, -np.inf, dtype=self.dtype)

    def update(self, values: np.ndarray) -> None:
        """Update streaming statistics from a batch of channel values."""
        data = np.asarray(values, dtype=self.dtype)
        if data.ndim == 1:
            data = data[:, None]
        if data.shape[1] != self.channels:
            raise PreprocessError(
                f"Stats update channel mismatch: expected {self.channels}, got {data.shape[1]}"
            )
        if np.any(~np.isfinite(data)):
            raise PreprocessError(
                "Non-finite values encountered while fitting normalization stats."
            )

        self.count += int(data.shape[0])
        self.sum += np.sum(data, axis=0)
        self.sumsq += np.sum(data * data, axis=0)
        self.min = np.minimum(self.min, np.min(data, axis=0))
        self.max = np.maximum(self.max, np.max(data, axis=0))

    def update_constant(self, values: np.ndarray, repeats: int) -> None:
        """Update statistics for a repeated global value without materializing copies."""
        data = np.asarray(values, dtype=self.dtype)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape != (1, self.channels):
            raise PreprocessError(
                "Constant stats update mismatch: expected shape "
                f"(1,{self.channels}), got {data.shape}."
            )
        if repeats <= 0:
            raise PreprocessError("Constant stats update requires repeats > 0.")
        if np.any(~np.isfinite(data)):
            raise PreprocessError(
                "Non-finite values encountered while fitting normalization stats."
            )

        value = data[0]
        self.count += int(repeats)
        self.sum += value * repeats
        self.sumsq += value * value * repeats
        self.min = np.minimum(self.min, value)
        self.max = np.maximum(self.max, value)

    def finalize(self) -> dict[str, Any]:
        """Convert accumulated statistics into JSON-serializable metadata."""
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


def _log10_safe(values: np.ndarray, eps: float) -> np.ndarray:
    """Apply the project-standard base-10 log transform with an epsilon floor."""
    return np.log10(np.maximum(values, eps))


def _fit_for_method(values: np.ndarray, policy: _NormPolicy) -> np.ndarray:
    """Project values into the space used to fit normalization statistics."""
    if policy.method in {"log-standard", "log-min-max"}:
        return _log10_safe(values, policy.epsilon)
    if policy.method in {"standard", "none"}:
        return values
    raise PreprocessError(f"Unsupported normalization method: {policy.method}")


def _apply_method(values: np.ndarray, stats: dict[str, Any], policy: _NormPolicy) -> np.ndarray:
    """Normalize values with precomputed statistics under one explicit method."""
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
    """Map the configured torch stats dtype to the matching NumPy dtype."""
    return np.float64 if precision.stats_dtype == torch.float64 else np.float32


def _load_run_file(path: Path, expected_species: list[str]) -> dict[str, Any]:
    """Load one raw run HDF5 file and validate its science contract.

    Args:
        path: Raw run HDF5 path.
        expected_species: Ordered target-species list expected in the file.

    Returns:
        Dictionary containing physical-space NumPy arrays/scalars:
        - `pressure_bar`, `temperature_k`, `kzz_cm2_s`: shape `[nz]`, dtype `float64`
        - `initial_ymix`: shape `[nz, species]`, dtype `float64`
        - `time_s`: shape `[snapshots]`, dtype `float64`
        - `target_ymix`: shape `[snapshots, nz, species]`, dtype `float64`
        - `gravity_cm_s2`, `metallicity_log10`, `c_to_o`: Python `float`
    """
    if not path.is_file():
        raise PreprocessError(f"Missing run file: {path}")

    with h5py.File(path, "r") as handle:
        for required_group in ("inputs", "globals", "targets"):
            if required_group not in handle:
                raise PreprocessError(f"Run file missing group '{required_group}': {path}")

        inputs = handle["inputs"]
        globals_group = handle["globals"]
        targets = handle["targets"]

        species = [
            sp.decode("utf-8") if isinstance(sp, bytes) else str(sp)
            for sp in inputs["target_species"][()]
        ]
        if species != expected_species:
            raise PreprocessError(
                f"Target species mismatch in {path}. Expected {expected_species}, found {species}"
            )

        pressure = np.asarray(inputs["pressure_bar"], dtype=np.float64)
        temperature = np.asarray(inputs["temperature_k"], dtype=np.float64)
        kzz = np.asarray(inputs["kzz_cm2_s"], dtype=np.float64)
        initial_ymix = np.asarray(inputs["initial_ymix"], dtype=np.float64)

        time_s = np.asarray(targets["time_s"], dtype=np.float64)
        target_ymix = np.asarray(targets["ymix"], dtype=np.float64)

        gravity = float(np.asarray(globals_group["gravity_cm_s2"]))
        metallicity_log10 = float(np.asarray(globals_group["metallicity_log10"]))
        c_to_o = float(np.asarray(globals_group["c_to_o"]))

    arrays_to_check = [pressure, temperature, kzz, initial_ymix, time_s, target_ymix]
    if any(np.any(~np.isfinite(arr)) for arr in arrays_to_check):
        raise PreprocessError(f"Non-finite values found in raw run file: {path}")

    if pressure.ndim != 1 or temperature.ndim != 1 or kzz.ndim != 1:
        raise PreprocessError(f"Invalid sequence shapes in {path}")
    if not (pressure.shape == temperature.shape == kzz.shape):
        raise PreprocessError(f"Sequence length mismatch in {path}")

    nz = pressure.shape[0]
    n_species = len(expected_species)

    if initial_ymix.shape != (nz, n_species):
        raise PreprocessError(f"initial_ymix shape mismatch in {path}: {initial_ymix.shape}")
    if target_ymix.ndim != 3 or target_ymix.shape[1:] != (nz, n_species):
        raise PreprocessError(f"target_ymix shape mismatch in {path}: {target_ymix.shape}")

    if time_s.ndim != 1 or time_s.size != target_ymix.shape[0]:
        raise PreprocessError(f"time_s length mismatch in {path}")
    if np.any(time_s <= 0.0):
        raise PreprocessError(f"time_s must be > 0 for log10 transform in {path}")

    return {
        "pressure_bar": pressure,
        "temperature_k": temperature,
        "kzz_cm2_s": kzz,
        "initial_ymix": initial_ymix,
        "time_s": time_s,
        "target_ymix": target_ymix,
        "gravity_cm_s2": gravity,
        "metallicity_log10": metallicity_log10,
        "c_to_o": c_to_o,
    }


def _split_run_ids(
    run_ids: list[int], split_ratios: dict[str, float], seed: int
) -> SplitAssignments:
    """Shuffle run ids deterministically and split them without leakage."""
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


def _build_normalization_stats(
    *,
    train_files: list[Path],
    config: dict[str, Any],
    species: list[str],
    stats_dtype: type[np.float32] | type[np.float64],
) -> dict[str, Any]:
    """Fit normalization statistics from the training split only.

    Args:
        train_files: Raw HDF5 run files assigned to the training split.
        config: Fully validated project configuration dictionary.
        species: Ordered target-species list.
        stats_dtype: NumPy accumulator dtype, typically `np.float32` or `np.float64`.

    Returns:
        JSON-serializable normalization metadata dictionary with top-level `sequence`,
        `globals`, and `targets` sections.
    """
    norm_cfg = config["normalization"]
    epsilon = float(norm_cfg["epsilon"])

    seq_methods = norm_cfg["sequence_methods"]
    glob_methods = norm_cfg["global_methods"]
    target_method = str(norm_cfg["target_method"])

    seq_stats = {
        "pressure_bar": (
            _NormPolicy(seq_methods["pressure_bar"], epsilon),
            _RunningStats(1, stats_dtype),
        ),
        "temperature_k": (
            _NormPolicy(seq_methods["temperature_k"], epsilon),
            _RunningStats(1, stats_dtype),
        ),
        "kzz_cm2_s": (
            _NormPolicy(seq_methods["kzz_cm2_s"], epsilon),
            _RunningStats(1, stats_dtype),
        ),
        "initial_ymix": (
            _NormPolicy(seq_methods["initial_ymix"], epsilon),
            _RunningStats(len(species), stats_dtype),
        ),
    }

    glob_stats = {
        "gravity_cm_s2": (
            _NormPolicy(glob_methods["gravity_cm_s2"], epsilon),
            _RunningStats(1, stats_dtype),
        ),
        "metallicity_log10": (
            _NormPolicy(glob_methods["metallicity_log10"], epsilon),
            _RunningStats(1, stats_dtype),
        ),
        "c_to_o": (
            _NormPolicy(glob_methods["c_to_o"], epsilon),
            _RunningStats(1, stats_dtype),
        ),
        "log10_time_s": (
            _NormPolicy(glob_methods["log10_time_s"], epsilon),
            _RunningStats(1, stats_dtype),
        ),
    }

    tgt_stats = (
        _NormPolicy(target_method, epsilon),
        _RunningStats(len(species), stats_dtype),
    )

    for run_file in train_files:
        data = _load_run_file(run_file, expected_species=species)
        n_snap = int(data["time_s"].size)

        for key in ("pressure_bar", "temperature_k", "kzz_cm2_s"):
            policy, acc = seq_stats[key]
            arr = _fit_for_method(data[key].reshape(-1, 1), policy)
            acc.update(arr)

        policy_init, acc_init = seq_stats["initial_ymix"]
        acc_init.update(_fit_for_method(data["initial_ymix"], policy_init))

        g_policy, g_acc = glob_stats["gravity_cm_s2"]
        g_value = _fit_for_method(np.array([[data["gravity_cm_s2"]]], dtype=np.float64), g_policy)
        g_acc.update_constant(g_value, repeats=n_snap)

        m_policy, m_acc = glob_stats["metallicity_log10"]
        m_value = _fit_for_method(
            np.array([[data["metallicity_log10"]]], dtype=np.float64), m_policy
        )
        m_acc.update_constant(m_value, repeats=n_snap)

        c_policy, c_acc = glob_stats["c_to_o"]
        c_value = _fit_for_method(np.array([[data["c_to_o"]]], dtype=np.float64), c_policy)
        c_acc.update_constant(c_value, repeats=n_snap)

        t_policy, t_acc = glob_stats["log10_time_s"]
        log10_time = np.log10(data["time_s"]).reshape(-1, 1)
        t_acc.update(_fit_for_method(log10_time, t_policy))

        tgt_policy, tgt_acc = tgt_stats
        tgt_acc.update(_fit_for_method(data["target_ymix"].reshape(-1, len(species)), tgt_policy))

    return {
        "epsilon": epsilon,
        "sequence": {
            key: {"method": policy.method, **acc.finalize()}
            for key, (policy, acc) in seq_stats.items()
        },
        "globals": {
            key: {"method": policy.method, **acc.finalize()}
            for key, (policy, acc) in glob_stats.items()
        },
        "targets": {"ymix": {"method": tgt_stats[0].method, **tgt_stats[1].finalize()}},
    }


def _write_split_shards(
    *,
    split_name: str,
    run_files: list[Path],
    config: dict[str, Any],
    species: list[str],
    stats: dict[str, Any],
    normalization_fingerprint: str,
    processed_root: Path,
) -> dict[str, Any]:
    """Normalize one split and write fixed-size `.npy` shards plus metadata.

    Args:
        split_name: Split name such as `"train"`, `"val"`, or `"test"`.
        run_files: Ordered raw HDF5 run files assigned to the split.
        config: Fully validated project configuration dictionary.
        species: Ordered target-species list.
        stats: Serialized normalization statistics dictionary.
        normalization_fingerprint: SHA-256 hash of the normalization metadata JSON payload.
        processed_root: Root directory for processed split outputs.

    Returns:
        Split metadata dictionary describing sample counts, feature dimensions, feature ordering,
        and the normalization fingerprint used for the written shards.
    """
    split_dir = processed_root / split_name
    seq_dir = split_dir / "sequence_inputs"
    glb_dir = split_dir / "globals"
    tgt_dir = split_dir / "targets"
    for directory in (split_dir, seq_dir, glb_dir, tgt_dir):
        directory.mkdir(parents=True, exist_ok=True)

    seq_keys = ["pressure_bar", "temperature_k", "kzz_cm2_s"]

    n_species = len(species)
    input_dim = 3 + n_species
    global_dim = 4

    shard_size = int(config["generation"]["shard_size"])

    shard_idx = 0
    write_pos = 0
    total_samples = 0
    sequence_length = -1

    seq_buf: np.ndarray | None = None
    glb_buf: np.ndarray | None = None
    tgt_buf: np.ndarray | None = None

    def alloc_buffers(nz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.zeros((shard_size, nz, input_dim), dtype=np.float32),
            np.zeros((shard_size, global_dim), dtype=np.float32),
            np.zeros((shard_size, nz, n_species), dtype=np.float32),
        )

    def flush(current_size: int) -> None:
        nonlocal shard_idx, seq_buf, glb_buf, tgt_buf
        if current_size <= 0 or seq_buf is None or glb_buf is None or tgt_buf is None:
            return
        np.save(seq_dir / f"shard_{shard_idx:05d}.npy", seq_buf[:current_size], allow_pickle=False)
        np.save(glb_dir / f"shard_{shard_idx:05d}.npy", glb_buf[:current_size], allow_pickle=False)
        np.save(tgt_dir / f"shard_{shard_idx:05d}.npy", tgt_buf[:current_size], allow_pickle=False)
        shard_idx += 1

    eps = float(stats["epsilon"])
    seq_policies = {
        key: _NormPolicy(stats["sequence"][key]["method"], eps)
        for key in ("pressure_bar", "temperature_k", "kzz_cm2_s", "initial_ymix")
    }
    glb_policies = {
        key: _NormPolicy(stats["globals"][key]["method"], eps)
        for key in ("gravity_cm_s2", "metallicity_log10", "c_to_o", "log10_time_s")
    }
    tgt_policy = _NormPolicy(stats["targets"]["ymix"]["method"], eps)

    for run_file in run_files:
        data = _load_run_file(run_file, expected_species=species)
        nz = int(data["pressure_bar"].size)
        if sequence_length < 0:
            sequence_length = nz
            seq_buf, glb_buf, tgt_buf = alloc_buffers(nz)
        elif sequence_length != nz:
            raise PreprocessError(
                "Mixed sequence lengths across runs are unsupported in v1: "
                f"{sequence_length} vs {nz}"
            )

        seq_channels = []
        for key in seq_keys:
            raw = data[key].reshape(nz, 1)
            seq_channels.append(_apply_method(raw, stats["sequence"][key], seq_policies[key]))
        init_norm = _apply_method(
            data["initial_ymix"],
            stats["sequence"]["initial_ymix"],
            seq_policies["initial_ymix"],
        )
        seq_template = np.concatenate([*seq_channels, init_norm], axis=1)

        target_norm = _apply_method(
            data["target_ymix"],
            stats["targets"]["ymix"],
            tgt_policy,
        )

        log10_time = np.log10(data["time_s"])
        gravity_norm = float(
            _apply_method(
                np.array([[data["gravity_cm_s2"]]], dtype=np.float64),
                stats["globals"]["gravity_cm_s2"],
                glb_policies["gravity_cm_s2"],
            )[0, 0]
        )
        metallicity_norm = float(
            _apply_method(
                np.array([[data["metallicity_log10"]]], dtype=np.float64),
                stats["globals"]["metallicity_log10"],
                glb_policies["metallicity_log10"],
            )[0, 0]
        )
        c_to_o_norm = float(
            _apply_method(
                np.array([[data["c_to_o"]]], dtype=np.float64),
                stats["globals"]["c_to_o"],
                glb_policies["c_to_o"],
            )[0, 0]
        )

        time_norm = _apply_method(
            log10_time.reshape(-1, 1),
            stats["globals"]["log10_time_s"],
            glb_policies["log10_time_s"],
        ).reshape(-1)

        n_snap = int(data["time_s"].size)
        glb_block = np.column_stack(
            [
                np.full(n_snap, gravity_norm, dtype=np.float64),
                np.full(n_snap, metallicity_norm, dtype=np.float64),
                np.full(n_snap, c_to_o_norm, dtype=np.float64),
                time_norm,
            ]
        ).astype(np.float32, copy=False)

        seq_template_f32 = seq_template.astype(np.float32, copy=False)
        target_norm_f32 = target_norm.astype(np.float32, copy=False)

        snap_start = 0
        while snap_start < n_snap:
            if seq_buf is None or glb_buf is None or tgt_buf is None:
                raise PreprocessError("Internal buffer allocation failure.")

            if write_pos >= shard_size:
                flush(shard_size)
                write_pos = 0

            room = shard_size - write_pos
            n_copy = min(room, n_snap - snap_start)
            end_pos = write_pos + n_copy
            next_snap = snap_start + n_copy

            seq_buf[write_pos:end_pos] = seq_template_f32
            glb_buf[write_pos:end_pos] = glb_block[snap_start:next_snap]
            tgt_buf[write_pos:end_pos] = target_norm_f32[snap_start:next_snap]

            write_pos = end_pos
            snap_start = next_snap
            total_samples += n_copy

    flush(write_pos)

    metadata = {
        "split": split_name,
        "total_samples": total_samples,
        "num_shards": shard_idx,
        "sequence_length": sequence_length,
        "input_dim": input_dim,
        "global_dim": global_dim,
        "target_dim": n_species,
        "sequence_feature_order": [
            "pressure_bar",
            "temperature_k",
            "kzz_cm2_s",
            *[f"initial_ymix:{sp}" for sp in species],
        ],
        "global_feature_order": ["gravity_cm_s2", "metallicity_log10", "c_to_o", "log10_time_s"],
        "target_species_order": list(species),
        "normalization_fingerprint": normalization_fingerprint,
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
    """Execute `--gen`: generate raw runs, split by run, normalize, and write shards.

    Args:
        config: Fully validated project configuration dictionary.
        paths: Resolved path bundle with raw/processed/model/log locations.
        precision: Resolved precision policy used for normalization-stat accumulation.
        boundary_conditions: Optional resolved VULCAN boundary-condition settings.
    """
    generation = config["generation"]
    species = list(config["data_spec"]["target_species"])
    stats_dtype = _numpy_stats_dtype(precision)

    run_specs = build_run_specs(config)
    if not run_specs:
        raise PreprocessError("Run-spec generation returned zero runs.")

    settings = WorkerSettings(
        vulcan_source=str(paths.vulcan_source),
        worker_root=str((paths.root / generation["worker_root"]).resolve()),
        runs_root=str((paths.root / generation["runs_root"]).resolve()),
        target_species=tuple(species),
        use_transport=bool(config["physics_toggles"]["use_transport"]),
        boundary_conditions=boundary_conditions,
        use_condensation_optional=bool(config["physics_toggles"]["use_condensation_optional"]),
        save_evo_frq=int(generation["save_evo_frq"]),
        snapshots_per_run=int(generation["snapshots_per_run"]),
        keep_vulcan_outputs_debug=bool(generation["keep_vulcan_outputs_debug"]),
        run_timeout_seconds=int(generation["run_timeout_seconds"]),
        num_workers=int(generation["num_workers"]),
    )

    logger.info(
        "Running %d VULCAN jobs with %d workers...", len(run_specs), generation["num_workers"]
    )
    try:
        results = run_vulcan_jobs(run_specs, settings=settings)
    except VulcanRuntimeError as exc:
        raise PreprocessError(str(exc)) from exc

    run_files = [Path(result.run_file) for result in results]
    run_ids = [result.run_id for result in results]

    split = _split_run_ids(
        run_ids=run_ids,
        split_ratios=generation["split_ratios"],
        seed=int(generation["random_seed"]),
    )

    split_map = {
        "train": split.train,
        "val": split.val,
        "test": split.test,
    }

    run_file_by_id = {int(result.run_id): Path(result.run_file) for result in results}
    train_files = [run_file_by_id[idx] for idx in split.train]

    logger.info("Fitting normalization stats from train split (%d runs)...", len(train_files))
    stats = _build_normalization_stats(
        train_files=train_files,
        config=config,
        species=species,
        stats_dtype=stats_dtype,
    )
    normalization_fingerprint = sha256(
        json.dumps(stats, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    processed_root = paths.processed_root
    if processed_root.exists():
        shutil.rmtree(processed_root)

    processed_root.mkdir(parents=True, exist_ok=True)

    logger.info("Writing processed shards...")
    split_metadata = {}
    for split_name, split_ids in split_map.items():
        files = [run_file_by_id[idx] for idx in split_ids]
        split_metadata[split_name] = _write_split_shards(
            split_name=split_name,
            run_files=files,
            config=config,
            species=species,
            stats=stats,
            normalization_fingerprint=normalization_fingerprint,
            processed_root=processed_root,
        )

    manifest = {
        "num_runs": len(results),
        "run_files": [str(path.relative_to(paths.root)) for path in sorted(run_files)],
        "split": split_map,
        "species": species,
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

    logger.info("Generation + preprocessing complete. Raw runs: %d", len(results))
