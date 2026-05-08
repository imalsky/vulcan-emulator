#!/bin/bash
# Run pipeline stages: generation + (optional) normalization.
#
# Default invocation (FastChem):
#   sbatch supercomputer_cmds/run_gen.sh
#
# Condensation emulator (VULCAN, much slower per run — bump wall time and
# rename job/logs so concurrent FastChem and VULCAN submissions don't collide):
#   sbatch -t 120:00:00 \
#          --job-name=vulcan_gen_cond \
#          --export=ALL,CONFIG_PATH=config/vulcan_condensation.json \
#          supercomputer_cmds/run_gen.sh
#
# No-condensation VULCAN sibling (when you have one):
#   sbatch -t 96:00:00 \
#          --job-name=vulcan_gen_nocond \
#          --export=ALL,CONFIG_PATH=config/vulcan_nocondensation.json \
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
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=all
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

set -euo pipefail

CONDA_ENV=${CONDA_ENV:-vulcan}
CONFIG_PATH=${CONFIG_PATH:-config/fastchem.json}
SKIP_INSTALL=${SKIP_INSTALL:-0}
SKIP_NORM=${SKIP_NORM:-0}

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

export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONNOUSERSITE=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH:-}"
export MPLBACKEND=Agg
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}

# Pin BLAS/OpenMP to one thread per process. With 64 parallel workers each
# calling numpy/scipy, unbounded OpenMP fans out to 64 threads per call and
# the resulting thousands of contending threads can stall progress entirely.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SRUN_CPUS_PER_TASK=${SRUN_CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-64}}}
export SRUN_CPUS_PER_TASK

if [ "$SKIP_INSTALL" != "1" ]; then
  python -m pip install -U pip setuptools wheel
  # sympy + matplotlib are VULCAN-runtime imports (vulcan.py imports both at
  # module load; sympy is also used by make_chem_funs.py when
  # vulcan.runtime.regenerate_chem_funs=true). FastChem doesn't need them but
  # installing always keeps the env consistent across configs.
  python -m pip install -U jax numpy scipy sympy matplotlib h5py optuna optax orbax-checkpoint pydantic
  python -m pip install -e . --no-deps
fi

python - <<PY
from pathlib import Path
from src.utils.config import load_and_validate_config
cfg = load_and_validate_config(Path("$CONFIG_PATH").expanduser().resolve())
print(f"[preflight] chemistry_type={cfg['chemistry_type']} model_type={cfg['model_type']}")
print(f"[preflight] num_runs={cfg['generation']['num_runs']}")
print(f"[preflight] raw_root={cfg['paths']['raw_root']}")
print(f"[preflight] processed_root={cfg['paths']['processed_root']}")
print(f"[preflight] checkpoints_root={cfg['paths']['checkpoints_root']}")
if cfg["chemistry_type"] == "vulcan":
    timeout_s = cfg["generation"].get("vulcan_timeout_seconds", 1800.0)
    print(f"[preflight] vulcan_timeout_seconds={timeout_s}")
    presets = cfg["vulcan"].get("science_presets") or []
    cond_on = any(p["physics_toggles"].get("use_condensation", False) for p in presets)
    print(f"[preflight] use_condensation={cond_on}")
    block = cfg["vulcan_runtime"].get("condensation")
    if block is not None:
        print(f"[preflight] condense_sp={block['condense_sp']}")
        print(f"[preflight] non_gas_sp={block['non_gas_sp']}")
    print(f"[preflight] target_dim={cfg['data_spec']['target_dim']}")
PY

python - <<'PY'
import os

affinity = "unavailable"
if hasattr(os, "sched_getaffinity"):
    try:
        affinity = len(os.sched_getaffinity(0))
    except OSError:
        affinity = "error"
print(
    "[preflight] "
    f"SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK', 'unset')} "
    f"SLURM_CPUS_ON_NODE={os.environ.get('SLURM_CPUS_ON_NODE', 'unset')} "
    f"SRUN_CPUS_PER_TASK={os.environ.get('SRUN_CPUS_PER_TASK', 'unset')} "
    f"os_cpu_count={os.cpu_count()} "
    f"affinity_cpus={affinity}"
)
PY

srun --ntasks=1 --cpus-per-task="$SRUN_CPUS_PER_TASK" --cpu-bind=cores python -u -m src.utils --config "$CONFIG_PATH" --stage generation

if [ "$SKIP_NORM" != "1" ]; then
  srun --ntasks=1 --cpus-per-task="$SRUN_CPUS_PER_TASK" --cpu-bind=cores python -u -m src.utils --config "$CONFIG_PATH" --stage normalization
fi
