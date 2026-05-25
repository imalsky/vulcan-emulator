#!/bin/bash
# Run pipeline stages: generation + (optional) normalization.
#
# Default invocation (FastChem):
#   sbatch supercomputer_cmds/run_gen.sh
#
# ExoGibbs:
#   sbatch --export=ALL,CONFIG_PATH=config/exogibbs_luhman16a.json \
#          supercomputer_cmds/run_gen.sh
#
# VULCAN condensation:
#   sbatch -t 120:00:00 --job-name=vulcan_gen_cond \
#          --export=ALL,CONFIG_PATH=config/vulcan_condensation.json \
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
  python -m pip install -U exogibbs jax numpy scipy sympy matplotlib h5py optuna optax orbax-checkpoint pydantic
  python -m pip install -e . --no-deps
fi

python - <<PY
from pathlib import Path
from src.utils.config import load_and_validate_config
cfg = load_and_validate_config(Path("$CONFIG_PATH").expanduser().resolve())
print(f"[preflight] chemistry_type={cfg['chemistry_type']} model_type={cfg['model_type']}")
print(f"[preflight] num_runs={cfg['generation']['num_runs']}")
print(f"[preflight] raw_root={cfg['paths']['raw_root']}")
print(f"[preflight] parallel_workers={cfg['generation'].get('parallel_workers', 0)}")
PY

srun --ntasks=1 --cpus-per-task="$SRUN_CPUS_PER_TASK" --cpu-bind=cores python -u -m src.utils --config "$CONFIG_PATH" --stage generation

if [ "$SKIP_NORM" != "1" ]; then
  srun --ntasks=1 --cpus-per-task="$SRUN_CPUS_PER_TASK" --cpu-bind=cores python -u -m src.utils --config "$CONFIG_PATH" --stage normalization
fi
