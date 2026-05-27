#!/bin/bash
# Run pipeline stages: generation + (optional) normalization.
#
# Default invocation (VULCAN-JAX Luhman 16A 10k):
#   sbatch supercomputer_cmds/run_gen.sh
#
# ExoGibbs Luhman 16A 10k:
#   sbatch --export=ALL,CONFIG_PATH=config/exogibbs_luhman16a_10k.json \
#          supercomputer_cmds/run_gen.sh
#
# VULCAN-JAX Luhman 16A 10k:
#   sbatch -t 120:00:00 --job-name=vulcan_gen_luhman16a \
#          --export=ALL,CONFIG_PATH=config/vulcan_luhman16a_10k.json \
#          supercomputer_cmds/run_gen.sh
#SBATCH -J vulcan_gen
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p compute
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH --clusters=edge
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mail-type=all
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

set -euo pipefail
JOB_START_SECONDS=$SECONDS
finish() {
  status=$?
  elapsed=$((SECONDS - JOB_START_SECONDS))
  echo "[job] finished status=${status} elapsed=${elapsed}s at $(date -Is)"
}
trap finish EXIT

CONDA_ENV=${CONDA_ENV:-vulcan}
CONFIG_PATH=${CONFIG_PATH:-config/vulcan_luhman16a_10k.json}
SKIP_INSTALL=${SKIP_INSTALL:-0}
SKIP_NORM=${SKIP_NORM:-0}
export CONDA_ENV CONFIG_PATH SKIP_NORM

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
export VULCAN_PROJECT_ROOT="$PROJECT_ROOT"
CONDA_EXE="$(command -v conda)"
CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "[setup] creating conda env '$CONDA_ENV' (python=3.11, conda-forge)"
  conda create -y -n "$CONDA_ENV" -c conda-forge --override-channels python=3.11 pip
fi
conda activate "$CONDA_ENV"

export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONNOUSERSITE=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH:-}"
export MPLBACKEND=Agg
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
export JAX_ENABLE_X64=${JAX_ENABLE_X64:-1}
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SRUN_CPUS_PER_TASK=${SRUN_CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-48}}}
export SRUN_CPUS_PER_TASK
JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/tmp/vulcan_jax_cache_${SLURM_JOB_ID:-local}}
export JAX_COMPILATION_CACHE_DIR
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "========== VULCAN generation preflight =========="
echo "[preflight] date=$(date -Is)"
echo "[preflight] host=$(hostname)"
echo "[preflight] project_root=$PROJECT_ROOT"
echo "[preflight] config_path=$CONFIG_PATH"
echo "[preflight] job_id=${SLURM_JOB_ID:-local} cpus_per_task=$SRUN_CPUS_PER_TASK nodelist=${SLURM_JOB_NODELIST:-unknown}"
echo "[preflight] jax_compilation_cache=$JAX_COMPILATION_CACHE_DIR"
echo "[preflight] xla_flags=$XLA_FLAGS"

if [ "$SKIP_INSTALL" != "1" ]; then
  python -m pip install -U pip setuptools wheel
  python -m pip uninstall -y jax-cuda12-plugin jax-cuda12-pjrt >/dev/null 2>&1 || true
  python -m pip install -U exogibbs jax numpy scipy sympy matplotlib h5py optuna optax orbax-checkpoint pydantic
  echo "[setup] installing vulcan-jax into conda env '$CONDA_ENV' from TestPyPI"
  python -m pip install -U -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --no-deps vulcan-jax
  python -m pip install -e . --no-deps
fi

python - <<'PY'
import os
from pathlib import Path
import sys
from src.utils.config import load_and_validate_config
cfg = load_and_validate_config(Path(os.environ["CONFIG_PATH"]).expanduser().resolve())
backend = cfg.get("vulcan_runtime", {}).get("backend", "n/a")
print(f"[preflight] conda_env={os.environ['CONDA_ENV']}")
print(f"[preflight] python_executable={sys.executable}")
print(f"[preflight] chemistry_type={cfg['chemistry_type']} model_type={cfg['model_type']}")
print(f"[preflight] vulcan_backend={backend}")
print(f"[preflight] num_runs={cfg['generation']['num_runs']}")
print(f"[preflight] raw_root={cfg['paths']['raw_root']}")
print(f"[preflight] parallel_workers={cfg['generation'].get('parallel_workers', 0)}")
print(f"[preflight] cpus_available={os.environ['SRUN_CPUS_PER_TASK']}")
print(f"[preflight] jax_compilation_cache={os.environ['JAX_COMPILATION_CACHE_DIR']}")
print(f"[preflight] normalizing_after_generation={os.environ.get('SKIP_NORM', '0') != '1'}")
if backend == "vulcan_jax":
    import vulcan_jax
    print(f"[preflight] vulcan_jax_version={getattr(vulcan_jax, '__version__', '<unknown>')}")
    print(f"[preflight] vulcan_jax_path={Path(vulcan_jax.__file__).resolve()}")
PY

echo "========== VULCAN generation start =========="
srun --ntasks=1 --cpus-per-task="$SRUN_CPUS_PER_TASK" --cpu-bind=cores python -u -m src.utils --config "$CONFIG_PATH" --stage generation

if [ "$SKIP_NORM" != "1" ]; then
  echo "========== normalization start =========="
  srun --ntasks=1 --cpus-per-task="$SRUN_CPUS_PER_TASK" --cpu-bind=cores python -u -m src.utils --config "$CONFIG_PATH" --stage normalization
fi
