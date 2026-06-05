#!/bin/bash
# Run pipeline stages: training + (optional) export. GPU job, 1 node.
# Epochs / early stopping come from the config (100k: 1000 epochs, patience 20).
#
# Default invocation (VULCAN-JAX Luhman 16A 100k condensation):
#   sbatch supercomputer_cmds/run_train.sh
#
# ExoGibbs:
#   sbatch --export=ALL,CONFIG_PATH=config/exogibbs_luhman16a_10k.json \
#          supercomputer_cmds/run_train.sh
#
# VULCAN-JAX Luhman 16A 10k:
#   sbatch --job-name=vulcan_train_luhman16a \
#          --export=ALL,CONFIG_PATH=config/vulcan_luhman16a_10k.json \
#          supercomputer_cmds/run_train.sh
#SBATCH -J vulcan_train
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gpu
#SBATCH --mem=60G
#SBATCH -t 24:00:00
#SBATCH --gpus=1
#SBATCH --clusters=edge
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=all
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

set -euo pipefail

CONDA_ENV=${CONDA_ENV:-vulcan}
CONFIG_PATH=${CONFIG_PATH:-config/vulcan_luhman16a_100k.json}
SKIP_INSTALL=${SKIP_INSTALL:-0}
SKIP_EXPORT=${SKIP_EXPORT:-0}

if [ -z "${PROJECT_ROOT:-}" ]; then
  submit_dir="${SLURM_SUBMIT_DIR:-}"
  if [ -n "$submit_dir" ] && [ -d "$submit_dir/supercomputer_cmds" ]; then
    PROJECT_ROOT="$submit_dir"
  elif [ -n "$submit_dir" ] && [ "$(basename "$submit_dir")" = "supercomputer_cmds" ]; then
    PROJECT_ROOT="$(cd -- "$submit_dir/.." && pwd -P)"
  else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
  fi
fi
cd -P "${PROJECT_ROOT}"
CONDA_EXE="$(command -v conda)"
CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  if [ "$SKIP_INSTALL" = "1" ]; then
    echo "ERROR: SKIP_INSTALL=1 but conda env '$CONDA_ENV' does not exist." >&2
    echo "       Build it once on a login node (jax[cuda12] + deps) before submitting." >&2
    exit 2
  fi
  echo "[setup] creating conda env '$CONDA_ENV' (python=3.11, conda-forge)"
  conda create -y -n "$CONDA_ENV" -c conda-forge --override-channels python=3.11 pip
fi
conda activate "$CONDA_ENV"

if command -v module >/dev/null 2>&1; then
  module load cuda12.6/toolkit 2>/dev/null || module load cuda11.8/toolkit 2>/dev/null || true
fi

export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONNOUSERSITE=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH:-}"
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=${JAX_ENABLE_X64:-1}
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"

if [ "$SKIP_INSTALL" != "1" ]; then
  python -m pip install -U pip setuptools wheel
  python -m pip install -U "jax[cuda12]" numpy scipy sympy matplotlib h5py optuna optax orbax-checkpoint pydantic
  python -m pip install -e . --no-deps
fi

PY_MM="$(python -c 'import sys;print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
SITE_PKGS="$CONDA_PREFIX/lib/$PY_MM/site-packages"
NVIDIA_LD_PATHS=""
for pkg_dir in "$SITE_PKGS"/nvidia/*/lib; do
  [ -d "$pkg_dir" ] && NVIDIA_LD_PATHS="${pkg_dir}:${NVIDIA_LD_PATHS}"
done
if [ -n "$NVIDIA_LD_PATHS" ]; then
  export LD_LIBRARY_PATH="${NVIDIA_LD_PATHS}${LD_LIBRARY_PATH:-}"
fi

nvidia-smi

python - <<'PY'
import os, jax
backend = str(jax.default_backend()).lower()
print(f"[preflight] JAX backend: {backend}")
print(f"[preflight] JAX devices: {jax.devices()}")
if backend not in {"gpu", "cuda", "rocm", "metal"}:
    raise SystemExit(f"ERROR: JAX backend is '{backend}', expected a GPU backend")
PY

# Reuse-only contract check: the training stage's ensure_processed silently
# falls through to REGENERATION (which needs vulcan_jax, absent in this GPU env)
# if the processed dataset is missing/incompatible. Fail fast here — before the
# GPU is allocated — instead of detonating mid-run.
python - <<PY
import json
import sys
from pathlib import Path
from src.data_generation.data_loader import processed_info_dir
from src.utils.config import get_chemistry_type, get_model_type, load_and_validate_config

cfg = load_and_validate_config(Path("$CONFIG_PATH").expanduser().resolve())
processed_root = Path(cfg["paths"]["processed_root"])
if not processed_root.is_absolute():
    processed_root = (Path.cwd() / processed_root).resolve()

required = (
    "metadata.json", "run_ids.json", "sequence_inputs.npy", "target_outputs.npy",
    "global_inputs.npy", "valid_mask.npy", "position_coord.npy",
)
problems = []
for split in ("train", "val", "test"):
    split_dir = processed_root / split
    if not split_dir.is_dir():
        problems.append(f"missing split dir {split_dir}")
        continue
    for filename in required:
        if not (split_dir / filename).exists():
            problems.append(f"missing {split_dir / filename}")

contract_path = processed_info_dir(processed_root) / "data_contract.json"
if not contract_path.exists():
    problems.append(f"missing {contract_path}")
else:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if str(contract.get("chemistry_type", "")).lower() != get_chemistry_type(cfg):
        problems.append("chemistry_type mismatch vs processed data_contract")
    if str(contract.get("model_type", "")).lower() != get_model_type(cfg):
        problems.append("model_type mismatch vs processed data_contract")
    if list(contract.get("sequence_static_feature_order", [])) != list(cfg["data_spec"]["sequence_static_feature_order"]):
        problems.append("sequence feature order mismatch vs processed data_contract")
    if list(contract.get("global_static_feature_order", [])) != list(cfg["data_spec"]["global_static_feature_order"]):
        problems.append("global feature order mismatch vs processed data_contract")

if problems:
    print("ERROR: processed dataset missing or incompatible with this config.")
    print("       The training stage would try to REGENERATE (needs vulcan_jax) and fail on the GPU node.")
    print("       Run generation+merge+normalization with this config first.")
    for problem in problems:
        print(f"       - {problem}")
    sys.exit(1)

print(f"[preflight] chemistry_type={cfg['chemistry_type']} model_type={cfg['model_type']}")
print(f"[preflight] processed_root={processed_root} (contract OK, required tensors present)")
print(f"[preflight] checkpoints_root={cfg['paths']['checkpoints_root']}")
PY

srun python -u -m src.utils --config "$CONFIG_PATH" --stage training

if [ "$SKIP_EXPORT" != "1" ]; then
  srun python -u -m src.utils --config "$CONFIG_PATH" --stage export
fi
