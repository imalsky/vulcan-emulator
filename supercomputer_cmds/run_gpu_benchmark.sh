#!/bin/bash
# Standalone VULCAN-JAX GPU throughput benchmark on the edge A100 (SLURM) —
# the .sh counterpart of run_gpu_benchmark.pbs (NAS GH200), for when the PBS
# system is down or you want the A100.
#
# Two phases:
#   1. TILED SWEEP — throughput vs batch size with host-side tiling
#      (--device-batch, default 128), the production-safe configuration.
#   2. FIX-B PROBE — one UNTILED batch (--device-batch == batch) in a fresh
#      process, to verify the chunked Jacobian assembly (vulcan-jax >= 0.1.14,
#      chem.py lax.scan over 128-reaction blocks) actually bounds the vmapped
#      transient on device. The 'peak GiB' column is the verdict: the GH200
#      analysis predicted the un-chunked batch-512 transient at ~42-61 GiB
#      (it OOM'd 96 GB); with the fix the untiled peak should sit far below
#      the un-chunked prediction. The probe size adapts to the visible GPU
#      memory (>=70 GiB -> 512, >=35 -> 256, else 128). A probe OOM is
#      reported as a FINDING (XLA un-did the chunking); it does not fail the
#      job if the sweep succeeded.
#
# Default invocation (from the repo root):
#   sbatch supercomputer_cmds/run_gpu_benchmark.sh
#
# Quick test / custom sweep:
#   sbatch --export=ALL,BATCHES="4",PROBE=0 supercomputer_cmds/run_gpu_benchmark.sh
#   sbatch --export=ALL,BATCHES="8 64 256 512" supercomputer_cmds/run_gpu_benchmark.sh
#
# Also runs WITHOUT SLURM on any box with an NVIDIA GPU + conda:
#   bash supercomputer_cmds/run_gpu_benchmark.sh
#
# Env vars (all optional):
#   BATCHES       sweep batch sizes               (default "1 8 32 128 512")
#   DEVICE_BATCH  max lanes per device call       (default 128)
#   PROBE         1 = run the untiled Fix-B probe (default 1)
#   PROBE_BATCH   probe size, or "auto" by GPU mem (default auto)
#   NZ            layers per planet               (default 150 = real HD189)
#   COUNT_MAX     Ros2 step cap per planet        (default 2500; converges ~600)
#   CHUNK         accepted steps per device call  (default 250)
#   WORKERS       host-setup CPU processes        (default 0 = all visible)
#   CONDA_ENV     interpreter env                 (default vulcan)
#   SKIP_INSTALL  1 = don't touch the env         (default 0)
#SBATCH -J vjax_gpu_bench
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gpu
#SBATCH --mem=120G
#SBATCH -t 06:00:00
#SBATCH --gpus=1
#SBATCH --clusters=edge
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=all
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

set -uo pipefail
JOB_START_SECONDS=$SECONDS
finish() {
  status=$?
  elapsed=$((SECONDS - JOB_START_SECONDS))
  echo "[job] finished status=${status} elapsed=${elapsed}s at $(date -Is)"
}
trap finish EXIT

CONDA_ENV=${CONDA_ENV:-vulcan}
BATCHES=${BATCHES:-"1 8 32 128 512"}
DEVICE_BATCH=${DEVICE_BATCH:-128}
PROBE=${PROBE:-1}
PROBE_BATCH=${PROBE_BATCH:-auto}
NZ=${NZ:-150}
COUNT_MAX=${COUNT_MAX:-2500}
CHUNK=${CHUNK:-250}
WORKERS=${WORKERS:-0}
SKIP_INSTALL=${SKIP_INSTALL:-0}

# --- locate the repo (works under sbatch, or run directly with bash) ----------
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
SCRIPT_DIR="${PROJECT_ROOT}/supercomputer_cmds"

# --- stream everything to a tail-able log --------------------------------------
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
_jobid="${SLURM_JOB_ID:-manual_$$}"
LOG_FILE="${LOG_DIR}/vjax_gpu_bench_${_jobid}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Live log: tail -f ${LOG_FILE}"
echo "Job ${SLURM_JOB_ID:-manual} started $(date -Is) on $(hostname) ($(uname -m))"

# --- conda env (same pattern as run_train.sh) -----------------------------------
CONDA_EXE="$(command -v conda)" || { echo "ERROR: conda not on PATH."; exit 2; }
CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  if [ "$SKIP_INSTALL" = "1" ]; then
    echo "ERROR: SKIP_INSTALL=1 but conda env '$CONDA_ENV' does not exist." >&2
    exit 2
  fi
  echo "[setup] creating conda env '$CONDA_ENV' (python=3.11, conda-forge)"
  conda create -y -n "$CONDA_ENV" -c conda-forge --override-channels python=3.11 pip
fi
conda activate "$CONDA_ENV"

if [ "$SKIP_INSTALL" != "1" ]; then
  echo "[setup] installing GPU jax + vulcan-jax (TestPyPI) into '$CONDA_ENV'"
  python -m pip install -U pip setuptools wheel
  python -m pip install -U "jax[cuda12]" numpy scipy
  python -m pip install -U -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --no-deps vulcan-jax
fi

# vulcan-jax must carry the chunked Jacobian + run_batch tiling era (>= 0.1.14).
python - <<'PY' || exit 1
import sys
import vulcan_jax
ver = tuple(int(x) for x in vulcan_jax.__version__.split(".")[:3])
print(f"[preflight] vulcan_jax {vulcan_jax.__version__}")
if ver < (0, 1, 14):
    sys.exit(
        f"ERROR: vulcan_jax {vulcan_jax.__version__} predates the chunked "
        "Jacobian (Fix B); the Fix-B probe would be meaningless. Need >= 0.1.14."
    )
PY

# FastChem binary must be native to this node (x86_64 on edge; the package may
# have been installed elsewhere). Rebuilds in place from bundled source if not.
bash "${SCRIPT_DIR}/ensure_vulcan_jax.sh" || exit 1

# pip's jax[cuda12] ships its own CUDA runtime in the nvidia-* wheels; a
# cluster CUDA toolkit module (or inherited LD_LIBRARY_PATH) can shadow them
# with an incompatible version — symptom: "Unable to load cuSPARSE", JAX
# falls back to CPU. Do NOT module-load cuda here; put the wheel libs FIRST.
NVLIB="$(python - <<'PY'
import pathlib
try:
    import nvidia
except ImportError:
    raise SystemExit
root = pathlib.Path(nvidia.__path__[0])
print(":".join(str(d / "lib") for d in sorted(root.iterdir()) if (d / "lib").is_dir()))
PY
)"
if [ -n "${NVLIB:-}" ]; then
  export LD_LIBRARY_PATH="${NVLIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "[setup] pip CUDA wheel libs prepended to LD_LIBRARY_PATH"
fi

# --- GPU runtime env (same knobs as run_gen.sh's GPU branch) --------------------
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
unset JAX_PLATFORMS
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0}"
JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/tmp/vulcan_jax_cache_${_jobid}}
export JAX_COMPILATION_CACHE_DIR
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: no visible NVIDIA GPU in this job (did you land on the gpu partition?)."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GPU_MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
python - <<'PY' || { echo "ERROR: JAX is not on the GPU in this env."; exit 1; }
import jax, sys
be = jax.default_backend()
print(f"[preflight] jax {jax.__version__}  backend={be}  devices={jax.devices()}")
sys.exit(0 if be in ("gpu", "cuda", "rocm") else 1)
PY

# --- phase 1: tiled throughput sweep --------------------------------------------
echo "========== phase 1: tiled sweep (batches=${BATCHES}, device_batch=${DEVICE_BATCH}, nz=${NZ}, count_max=${COUNT_MAX}, chunk=${CHUNK}) =========="
# shellcheck disable=SC2086
python "${SCRIPT_DIR}/gpu_benchmark.py" \
  --batches ${BATCHES} --device-batch "${DEVICE_BATCH}" \
  --nz "${NZ}" --count-max "${COUNT_MAX}" --chunk "${CHUNK}" --workers "${WORKERS}"
sweep_rc=$?
echo "[phase 1] sweep rc=${sweep_rc}"

# --- phase 2: untiled Fix-B probe (fresh process -> clean allocator peak) -------
probe_rc=0
if [ "${PROBE}" = "1" ]; then
  if [ "${PROBE_BATCH}" = "auto" ]; then
    # The pre-fix batch-512 stack/Jacobian transients needed ~42-61 GiB on top
    # of ~3 GiB persistent; size the probe so a HEALTHY (chunked) run fits the
    # card while a chunking regression would still blow past it.
    if   [ "${GPU_MEM_MIB}" -ge 71680 ]; then PROBE_BATCH=512
    elif [ "${GPU_MEM_MIB}" -ge 35840 ]; then PROBE_BATCH=256
    else PROBE_BATCH=128; fi
  fi
  echo "========== phase 2: Fix-B probe — UNTILED batch ${PROBE_BATCH} on a ${GPU_MEM_MIB} MiB GPU =========="
  echo "[probe] watch the 'peak GiB' column: chunked-Jacobian-healthy means it"
  echo "[probe] stays far below the un-chunked prediction (~42-61 GiB at 512)."
  python "${SCRIPT_DIR}/gpu_benchmark.py" \
    --batches "${PROBE_BATCH}" --device-batch "${PROBE_BATCH}" \
    --nz "${NZ}" --count-max "${COUNT_MAX}" --chunk "${CHUNK}" --workers "${WORKERS}"
  probe_rc=$?
  if [ "${probe_rc}" -ne 0 ]; then
    echo "[probe] FINDING: untiled batch ${PROBE_BATCH} FAILED (rc=${probe_rc}, likely OOM)."
    echo "[probe] If the sweep above succeeded, XLA may have un-done the lax.scan"
    echo "[probe] chunking in chem.chem_jac_analytical_per_layer — see"
    echo "[probe] VULCAN-JAX/docs/notes.md (add jax.checkpoint on the chunk body"
    echo "[probe] or shrink _JAC_CHUNK_REACTIONS). Production runs are unaffected"
    echo "[probe] as long as --device-batch stays tiled (default 128)."
  else
    echo "[probe] untiled batch ${PROBE_BATCH} completed — Fix B holds on this device."
  fi
fi

echo "sweep rc=${sweep_rc}, probe rc=${probe_rc} at $(date -Is)"
exit "${sweep_rc}"
