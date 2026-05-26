#!/bin/bash
# One-shot dependency install for the shared conda env.
#
# Submitted automatically by submit_gen_array.sh before the generation
# array. The array tasks depend on this job via --dependency=afterok, so
# they never race on pip install.
#
# After pip install, compiles FastChem natively from ../VULCAN-master and
# patches the installed vulcan-jax package with the resulting binary.
# The TestPyPI wheel ships a macOS binary that cannot run on Linux.
#
# Can also be run standalone:
#   sbatch supercomputer_cmds/run_install.sh
#
#SBATCH -J vulcan_install
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p compute
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH --clusters=edge
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

set -euo pipefail

CONDA_ENV=${CONDA_ENV:-vulcan}

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

export PYTHONNOUSERSITE=1

python -m pip install -U pip setuptools wheel
python -m pip uninstall -y jax-cuda12-plugin jax-cuda12-pjrt >/dev/null 2>&1 || true
python -m pip install -U exogibbs jax numpy scipy sympy matplotlib h5py optuna optax orbax-checkpoint pydantic
echo "[setup] installing vulcan-jax into conda env '$CONDA_ENV' from TestPyPI"
python -m pip install -U -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --no-deps vulcan-jax
python -m pip install -e . --no-deps

# --- Compile FastChem natively and patch the installed vulcan-jax package ---
# The TestPyPI wheel bundles a macOS ARM binary; we need a Linux binary.
VULCAN_SOURCE="${PROJECT_ROOT}/../VULCAN-master"
FC_BUILD_DIR="${VULCAN_SOURCE}/fastchem_vulcan"

if [ -d "$FC_BUILD_DIR/fastchem_src" ]; then
  echo "[setup] compiling FastChem from ${FC_BUILD_DIR} ..."
  make -C "$FC_BUILD_DIR" clean 2>/dev/null || true
  make -C "$FC_BUILD_DIR" all
  if [ -x "$FC_BUILD_DIR/fastchem" ]; then
    VULCAN_JAX_PKG=$(python -c "import vulcan_jax, pathlib; print(pathlib.Path(vulcan_jax.__file__).resolve().parent)")
    cp "$FC_BUILD_DIR/fastchem" "$VULCAN_JAX_PKG/fastchem_vulcan/fastchem"
    chmod +x "$VULCAN_JAX_PKG/fastchem_vulcan/fastchem"
    echo "[setup] patched installed vulcan-jax with native FastChem binary"
    file "$VULCAN_JAX_PKG/fastchem_vulcan/fastchem"
  else
    echo "[WARNING] FastChem compilation produced no executable — VULCAN generation will fail"
  fi
else
  echo "[WARNING] VULCAN-master source not found at ${FC_BUILD_DIR} — skipping FastChem compilation"
fi

echo "[install] done — env '$CONDA_ENV' is ready"
