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

CONDA_ENV=${CONDA_ENV:-vulcan}
CONFIG_PATH=${CONFIG_PATH:-config/vulcan_luhman16a_10k.json}
SKIP_NORM=${SKIP_NORM:-0}

# Must match NUM_SHARDS in run_gen_array.sh.
NUM_SHARDS=${NUM_SHARDS:-4}

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

srun --ntasks=1 --cpus-per-task="$SRUN_CPUS_PER_TASK" \
     python -u -m src.utils \
       --config "$CONFIG_PATH" --stage merge_shards --num-shards "$NUM_SHARDS"

if [ "$SKIP_NORM" != "1" ]; then
  srun --ntasks=1 --cpus-per-task="$SRUN_CPUS_PER_TASK" \
       python -u -m src.utils --config "$CONFIG_PATH" --stage normalization
fi
