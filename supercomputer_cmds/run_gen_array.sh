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
# Usage (default config is the 100k condensation set; 4 shards = 4 CPU nodes):
#   bash supercomputer_cmds/submit_gen_array.sh
#   CONFIG_PATH=config/vulcan_luhman16a_10k.json bash supercomputer_cmds/submit_gen_array.sh
#
# Re-run a failed shard (e.g. shard 2):
#   sbatch --array=2 supercomputer_cmds/run_gen_array.sh
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
#SBATCH --cpus-per-task=48
#SBATCH --array=0-3
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
CONFIG_PATH=${CONFIG_PATH:-config/vulcan_luhman16a_100k.json}
SKIP_INSTALL=${SKIP_INSTALL:-1}
export CONDA_ENV CONFIG_PATH

NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-4}}
if [ "$NUM_SHARDS" -lt 1 ]; then
  echo "ERROR: NUM_SHARDS must be >= 1 (got ${NUM_SHARDS})." >&2
  exit 2
fi
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "ERROR: run_gen_array.sh must run as a SLURM array job." >&2
  exit 2
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_SHARDS" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} outside NUM_SHARDS=${NUM_SHARDS}." >&2
  exit 2
fi
export NUM_SHARDS

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
export JAX_ENABLE_X64=${JAX_ENABLE_X64:-1}

# generation.gpu_batch.enabled → in-process vmapped GPU path (one H100 per
# shard task) instead of the CPU subprocess pool.
GPU_BATCH="$(python - "$CONFIG_PATH" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
print("1" if cfg.get("generation", {}).get("gpu_batch", {}).get("enabled", False) else "0")
PY
)"
if [ "$GPU_BATCH" = "1" ]; then
  unset JAX_PLATFORMS
  export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-true}
  export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}
  export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0}"
else
  export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
  export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1}"
  # One BLAS thread per worker — the parallelism comes from the Python-level
  # ThreadPoolExecutor in the exogibbs/generation backend.
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1
  export BLIS_NUM_THREADS=1
fi
export PYTHONUNBUFFERED=1

SRUN_CPUS_PER_TASK=${SRUN_CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-48}}}
export SRUN_CPUS_PER_TASK

STAGING_ROOT="${SLURM_TMPDIR:-/tmp/vulcan_gen_${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID}}"
export STAGING_ROOT
mkdir -p "$STAGING_ROOT"
JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$STAGING_ROOT/jax_cache}
export JAX_COMPILATION_CACHE_DIR
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "========== VULCAN generation shard preflight =========="
echo "[preflight] date=$(date -Is)"
echo "[preflight] host=$(hostname)"
echo "[preflight] project_root=$PROJECT_ROOT"
echo "[preflight] config_path=$CONFIG_PATH"
echo "[preflight] job_id=${SLURM_JOB_ID:-local} array_job_id=${SLURM_ARRAY_JOB_ID:-local} task_id=${SLURM_ARRAY_TASK_ID}"
echo "[preflight] num_shards=$NUM_SHARDS cpus_per_task=$SRUN_CPUS_PER_TASK nodelist=${SLURM_JOB_NODELIST:-unknown}"
echo "[preflight] staging_root=$STAGING_ROOT"
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
config_path = Path(os.environ["CONFIG_PATH"]).expanduser().resolve()
cfg = load_and_validate_config(config_path)
backend = cfg.get("vulcan_runtime", {}).get("backend", "n/a")
num_runs = int(cfg["generation"]["num_runs"])
num_shards = int(os.environ["NUM_SHARDS"])
shard_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
shard_start = shard_id * num_runs // num_shards
shard_end = (shard_id + 1) * num_runs // num_shards
print(f"[preflight] conda_env={os.environ['CONDA_ENV']}")
print(f"[preflight] python_executable={sys.executable}")
print(f"[preflight] chemistry_type={cfg['chemistry_type']} model_type={cfg['model_type']}")
print(f"[preflight] vulcan_backend={backend}")
print(f"[preflight] num_runs={num_runs}")
print(f"[preflight] raw_root={cfg['paths']['raw_root']}")
print(f"[preflight] shard={shard_id}/{num_shards} slice=[{shard_start}, {shard_end}) runs={shard_end - shard_start}")
print(f"[preflight] staging_root={os.environ['STAGING_ROOT']}")
print(f"[preflight] parallel_workers={cfg['generation'].get('parallel_workers', 0)}")
print(f"[preflight] cpus_available={os.environ['SRUN_CPUS_PER_TASK']}")
print(f"[preflight] jax_compilation_cache={os.environ['JAX_COMPILATION_CACHE_DIR']}")
if backend == "vulcan_jax":
    import vulcan_jax
    print(f"[preflight] vulcan_jax_version={getattr(vulcan_jax, '__version__', '<unknown>')}")
    print(f"[preflight] vulcan_jax_path={Path(vulcan_jax.__file__).resolve()}")
PY

echo "========== VULCAN generation shard start =========="
srun --ntasks=1 --cpus-per-task="$SRUN_CPUS_PER_TASK" --cpu-bind=cores \
     python -u -m src.utils \
       --config "$CONFIG_PATH" \
       --stage generation \
       --shard-id "$SLURM_ARRAY_TASK_ID" \
       --num-shards "$NUM_SHARDS" \
       --staging-root "$STAGING_ROOT"
