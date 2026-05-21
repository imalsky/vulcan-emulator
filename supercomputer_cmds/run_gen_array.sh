#!/bin/bash
# Sharded generation as a SLURM job array.
#
# Each array task generates one shard's slice of the deterministic sampling
# plan (--shard-id $SLURM_ARRAY_TASK_ID --num-shards 8) and writes its
# per-shard chunks to $RAW_ROOT/chunks_sNN/. Per-run HDF5 staging and the
# per-thread VULCAN/FastChem source-tree copies live on node-local
# $SLURM_TMPDIR. Chunks always live on shared FS so they survive a SLURM kill.
#
# After every shard succeeds, run_merge.sh combines them via
# `--stage merge_shards`. Submit both with the dependency wired up via
# submit_gen_array.sh.
#
# Default invocation (VULCAN condensation):
#   sbatch supercomputer_cmds/run_gen_array.sh
#
# Custom config:
#   sbatch --export=ALL,CONFIG_PATH=config/other.json \
#          supercomputer_cmds/run_gen_array.sh
#
# Re-run only a failed shard (e.g. shard 3):
#   sbatch --array=3 \
#          --export=ALL,CONFIG_PATH=config/vulcan_condensation.json \
#          supercomputer_cmds/run_gen_array.sh
#   # then resubmit run_merge.sh manually with --dependency=afterok:<new_id>
#
# Defaults: 500_000 runs across 40 shards (12_500 deterministic + 6_250 backfill
# slots each). --array=0-39%20 caps at 20 concurrent tasks; bump to %40 if the
# cluster lets you grab all 40 nodes at once.
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
#SBATCH --array=0-39%20
#SBATCH --mail-type=all
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

set -euo pipefail

CONDA_ENV=${CONDA_ENV:-vulcan}
CONFIG_PATH=${CONFIG_PATH:-config/vulcan_condensation.json}
SKIP_INSTALL=${SKIP_INSTALL:-0}

# Hard-code NUM_SHARDS rather than relying on SLURM_ARRAY_TASK_COUNT, which
# becomes wrong on sparse re-submissions like --array=3,5,7. The number must
# match the --array=0-N range above and the --num-shards used by run_merge.sh.
NUM_SHARDS=40

# Under SLURM the script is copied to a spool directory, so BASH_SOURCE points
# there instead of the original path. Prefer the scheduler's submit dir.
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

# Pin BLAS/OpenMP to one thread per process. With many parallel workers each
# calling numpy/scipy, unbounded OpenMP fans out per call and the resulting
# thousands of contending threads can stall progress entirely.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SRUN_CPUS_PER_TASK=${SRUN_CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-64}}}
export SRUN_CPUS_PER_TASK

# Node-local scratch for per-run HDF5 staging and worker-tree copies. Falls
# back to /tmp if the cluster does not provide $SLURM_TMPDIR. Either way the
# chunks_sNN/ directory lives on shared FS (under $RAW_ROOT) so chunks survive
# a SLURM kill — node-local only holds in-flight per-run files and source-tree
# copies that get rebuilt on resume.
STAGING_ROOT="${SLURM_TMPDIR:-/tmp/vulcan_gen_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
mkdir -p "$STAGING_ROOT"

if [ "$SKIP_INSTALL" != "1" ]; then
  python -m pip install -U pip setuptools wheel
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
print(f"[preflight] num_shards=$NUM_SHARDS shard_id=$SLURM_ARRAY_TASK_ID")
print(f"[preflight] staging_root=$STAGING_ROOT")
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

srun --ntasks=1 --cpus-per-task="$SRUN_CPUS_PER_TASK" --cpu-bind=cores \
     python -u -m src.utils \
       --config "$CONFIG_PATH" \
       --stage generation \
       --shard-id "$SLURM_ARRAY_TASK_ID" \
       --num-shards "$NUM_SHARDS" \
       --staging-root "$STAGING_ROOT"
