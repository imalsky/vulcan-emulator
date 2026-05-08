#!/bin/bash
# Run pipeline stages: training + (optional) export.
#
# Default invocation (FastChem):
#   sbatch supercomputer_cmds/run_train.sh
#
# Condensation emulator (rename job/logs so the condensation and FastChem
# checkpoints land on disjoint paths and concurrent submissions don't collide):
#   sbatch --job-name=vulcan_train_cond \
#          --export=ALL,CONFIG_PATH=config/vulcan_condensation.json \
#          supercomputer_cmds/run_train.sh
#
# No-condensation VULCAN sibling (when you have one):
#   sbatch --job-name=vulcan_train_nocond \
#          --export=ALL,CONFIG_PATH=config/vulcan_nocondensation.json \
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
CONFIG_PATH=${CONFIG_PATH:-config/fastchem.json}
SKIP_INSTALL=${SKIP_INSTALL:-0}
SKIP_EXPORT=${SKIP_EXPORT:-0}

# Under SLURM/PBS the script is copied to a spool directory, so BASH_SOURCE
# points there instead of the original path. Prefer the scheduler's submit dir.
if [ -z "${PROJECT_ROOT:-}" ]; then
  submit_dir="${SLURM_SUBMIT_DIR:-${PBS_O_WORKDIR:-}}"
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
  # sympy + matplotlib aren't strictly needed by training, but keeping the env
  # consistent with run_gen.sh means the same conda env can do both stages
  # without a re-install round-trip.
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

python - <<PY
from pathlib import Path
from src.utils.config import load_and_validate_config
cfg = load_and_validate_config(Path("$CONFIG_PATH").expanduser().resolve())
processed_root = Path(cfg["paths"]["processed_root"])
for split in ("train", "val", "test"):
    if not (processed_root / split).is_dir():
        raise SystemExit(f"ERROR: processed data missing: {processed_root / split}")
print(f"[preflight] chemistry_type={cfg['chemistry_type']} model_type={cfg['model_type']}")
print(f"[preflight] processed_root={processed_root}")
print(f"[preflight] checkpoints_root={cfg['paths']['checkpoints_root']}")
print(f"[preflight] target_dim={cfg['data_spec']['target_dim']}")
if cfg["chemistry_type"] == "vulcan":
    presets = cfg["vulcan"].get("science_presets") or []
    cond_on = any(p["physics_toggles"].get("use_condensation", False) for p in presets)
    print(f"[preflight] use_condensation={cond_on}")
    block = cfg["vulcan_runtime"].get("condensation")
    if block is not None:
        print(f"[preflight] non_gas_sp={block['non_gas_sp']}")
PY

srun python -u -m src.utils --config "$CONFIG_PATH" --stage training

if [ "$SKIP_EXPORT" != "1" ]; then
  srun python -u -m src.utils --config "$CONFIG_PATH" --stage export
fi
