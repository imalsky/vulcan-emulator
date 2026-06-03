# hpc_env.sh — redirect HOME-quota-bound caches and the Python user-site to
# scratch (/nobackup), so installs and runs never write to the over-quota $HOME.
#
# SOURCE this (it exports variables into the calling shell); do not execute it.
# Sourced by run_install.sh, run.pbs and run_one.pbs. Safe to source repeatedly.
#
# Why: on this system $HOME (/home4/$USER) is small and frequently over quota,
# and the shared conda env (pyt2_8_gh) is read-only. We therefore keep using the
# shared env's interpreter (with its working GPU JAX) and install our extra
# packages into a writable PYTHONUSERBASE on /nobackup, with every cache pointed
# at /nobackup as well.
#
# Override the scratch root by exporting VULCAN_SCRATCH before sourcing.
: "${VULCAN_SCRATCH:=/nobackup/${USER}/.vulcan}"

export PYTHONUSERBASE="${VULCAN_SCRATCH}/userbase"      # pip --user target (importable via user-site)
export PIP_CACHE_DIR="${VULCAN_SCRATCH}/cache/pip"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${VULCAN_SCRATCH}/cache/jax}"
export XDG_CACHE_HOME="${VULCAN_SCRATCH}/cache/xdg"
export MPLCONFIGDIR="${VULCAN_SCRATCH}/cache/mpl"
export CONDA_PKGS_DIRS="${VULCAN_SCRATCH}/conda/pkgs"
export TMPDIR="${VULCAN_SCRATCH}/tmp"

# User-site packages MUST be importable for the PYTHONUSERBASE install to work.
unset PYTHONNOUSERSITE

mkdir -p \
  "${PYTHONUSERBASE}" "${PIP_CACHE_DIR}" "${JAX_COMPILATION_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" "${CONDA_PKGS_DIRS}" "${TMPDIR}"

echo "[hpc_env] VULCAN_SCRATCH=${VULCAN_SCRATCH}"
echo "[hpc_env] PYTHONUSERBASE=${PYTHONUSERBASE}"
echo "[hpc_env] JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR}"
