#!/bin/bash
# Merge per-shard chunks into raw/runs.h5 and run normalization.
#
# Runs after every shard from run_gen_array.sh has succeeded. Submit via
# submit_gen_array.sh (which wires up the --dependency=afterok between the
# array and this job) or manually with:
#   sbatch --dependency=afterok:<array_job_id> \
#          --export=ALL,CONFIG_PATH=config/vulcan_luhman16a_10k.json \
#          supercomputer_cmds/run_merge.sh
#
# Set SKIP_NORM=1 to stop after producing runs.h5 (e.g. to inspect failed
# runs before kicking off normalization).
#
#SBATCH -J vulcan_gen_merge
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p compute
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH --clusters=edge
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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
SKIP_NORM=${SKIP_NORM:-0}
export CONDA_ENV CONFIG_PATH SKIP_NORM

NUM_SHARDS=${NUM_SHARDS:-4}
if [ "$NUM_SHARDS" -lt 1 ]; then
  echo "ERROR: NUM_SHARDS must be >= 1 (got ${NUM_SHARDS})." >&2
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
conda activate "$CONDA_ENV"

export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONNOUSERSITE=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH:-}"
export MPLBACKEND=Agg
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SRUN_CPUS_PER_TASK=${SRUN_CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-8}}
export SRUN_CPUS_PER_TASK

echo "========== VULCAN shard merge preflight =========="
echo "[preflight] date=$(date -Is)"
echo "[preflight] host=$(hostname)"
echo "[preflight] project_root=$PROJECT_ROOT"
echo "[preflight] config_path=$CONFIG_PATH"
echo "[preflight] job_id=${SLURM_JOB_ID:-local} num_shards=$NUM_SHARDS cpus_per_task=$SRUN_CPUS_PER_TASK"
python - <<'PY'
import os
from pathlib import Path
from src.utils.config import load_and_validate_config
cfg = load_and_validate_config(Path(os.environ["CONFIG_PATH"]).expanduser().resolve())
print(f"[preflight] chemistry_type={cfg['chemistry_type']} model_type={cfg['model_type']}")
print(f"[preflight] num_runs={cfg['generation']['num_runs']}")
print(f"[preflight] raw_root={cfg['paths']['raw_root']}")
print(f"[preflight] normalizing_after_merge={os.environ.get('SKIP_NORM', '0') != '1'}")
PY

echo "========== VULCAN shard merge start =========="
srun --ntasks=1 --cpus-per-task="$SRUN_CPUS_PER_TASK" \
     python -u -m src.utils \
       --config "$CONFIG_PATH" --stage merge_shards --num-shards "$NUM_SHARDS"

if [ "$SKIP_NORM" != "1" ]; then
  echo "========== normalization start =========="
  srun --ntasks=1 --cpus-per-task="$SRUN_CPUS_PER_TASK" \
       python -u -m src.utils --config "$CONFIG_PATH" --stage normalization
fi
