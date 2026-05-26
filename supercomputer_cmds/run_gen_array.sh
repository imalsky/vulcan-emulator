#!/bin/bash
# Sharded generation as a SLURM job array.
#
# Each array task generates one shard's slice of the deterministic sampling
# plan and writes per-shard chunks to $RAW_ROOT/chunks_sNN/. Per-run HDF5
# staging lives on node-local $SLURM_TMPDIR. Chunks live on shared FS so
# they survive a SLURM kill.
#
# After every shard succeeds, run_merge.sh combines them via
# `--stage merge_shards`. Submit both via submit_gen_array.sh.
#
# Usage:
#   CONFIG_PATH=config/exogibbs_luhman16a_10k.json bash supercomputer_cmds/submit_gen_array.sh
#   CONFIG_PATH=config/vulcan_luhman16a_10k.json bash supercomputer_cmds/submit_gen_array.sh
#
# Re-run a failed shard (e.g. shard 2):
#   CONFIG_PATH=config/vulcan_luhman16a_10k.json sbatch --array=2 supercomputer_cmds/run_gen_array.sh
#
#SBATCH -J vulcan_gen_array
#SBATCH -o %x_%A_%a.o
#SBATCH -e %x_%A_%a.e
#SBATCH -p compute
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH --clusters=edge
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --array=0-3
#SBATCH --mail-type=all
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

set -euo pipefail

CONDA_ENV=${CONDA_ENV:-vulcan}
CONFIG_PATH=${CONFIG_PATH:-config/vulcan_luhman16a_10k.json}
SKIP_INSTALL=${SKIP_INSTALL:-0}

# Must match the --array range above and --num-shards in run_merge.sh.
NUM_SHARDS=4

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

# One BLAS thread per worker — the parallelism comes from the Python-level
# ThreadPoolExecutor in the exogibbs/generation backend.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SRUN_CPUS_PER_TASK=${SRUN_CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-64}}}
export SRUN_CPUS_PER_TASK

STAGING_ROOT="${SLURM_TMPDIR:-/tmp/vulcan_gen_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
mkdir -p "$STAGING_ROOT"
JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$STAGING_ROOT/jax_cache}
export JAX_COMPILATION_CACHE_DIR
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

if [ "$SKIP_INSTALL" != "1" ]; then
  python -m pip install -U pip setuptools wheel
  python -m pip uninstall -y jax-cuda12-plugin jax-cuda12-pjrt >/dev/null 2>&1 || true
  python -m pip install -U exogibbs jax numpy scipy sympy matplotlib h5py optuna optax orbax-checkpoint pydantic
  echo "[setup] installing vulcan-jax into conda env '$CONDA_ENV' from TestPyPI"
  python -m pip install -U -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --no-deps vulcan-jax
  python -m pip install -e . --no-deps
fi

python - <<PY
from pathlib import Path
import sys
from src.utils.config import load_and_validate_config
cfg = load_and_validate_config(Path("$CONFIG_PATH").expanduser().resolve())
backend = cfg.get("vulcan_runtime", {}).get("backend", "n/a")
print(f"[preflight] conda_env=$CONDA_ENV")
print(f"[preflight] python_executable={sys.executable}")
print(f"[preflight] chemistry_type={cfg['chemistry_type']} model_type={cfg['model_type']}")
print(f"[preflight] vulcan_backend={backend}")
print(f"[preflight] num_runs={cfg['generation']['num_runs']}")
print(f"[preflight] raw_root={cfg['paths']['raw_root']}")
print(f"[preflight] num_shards=$NUM_SHARDS shard_id=$SLURM_ARRAY_TASK_ID")
print(f"[preflight] staging_root=$STAGING_ROOT")
print(f"[preflight] parallel_workers={cfg['generation'].get('parallel_workers', 0)}")
print(f"[preflight] cpus_available={$SRUN_CPUS_PER_TASK}")
print(f"[preflight] jax_compilation_cache=$JAX_COMPILATION_CACHE_DIR")
if backend == "vulcan_jax":
    import vulcan_jax
    print(f"[preflight] vulcan_jax_version={getattr(vulcan_jax, '__version__', '<unknown>')}")
    print(f"[preflight] vulcan_jax_path={Path(vulcan_jax.__file__).resolve()}")
PY

srun --ntasks=1 --cpus-per-task="$SRUN_CPUS_PER_TASK" --cpu-bind=cores \
     python -u -m src.utils \
       --config "$CONFIG_PATH" \
       --stage generation \
       --shard-id "$SLURM_ARRAY_TASK_ID" \
       --num-shards "$NUM_SHARDS" \
       --staging-root "$STAGING_ROOT"
