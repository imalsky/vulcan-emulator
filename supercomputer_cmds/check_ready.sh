#!/bin/bash
# check_ready.sh — read-only readiness check for the VULCAN emulator + vulcan_jax
# backend on THIS node. Installs nothing, builds nothing. Exit 0 only if ready.
#
#   bash supercomputer_cmds/check_ready.sh
#
# Env: CONDA_ENV (default pyt2_8_gh), VULCAN_SCRATCH (default /nobackup/$USER/.vulcan)
set -uo pipefail   # not -e: run every check and report, don't abort on first miss

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# Set PYTHONUSERBASE/caches the same way the real scripts do, so import checks
# see the /nobackup user-site.
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/hpc_env.sh" >/dev/null

module purge 2>/dev/null || true
module use -a /swbuild/analytix/tools/modulefiles 2>/dev/null || true
module load miniconda3/gh2 2>/dev/null || true
CONDA_ENV="${CONDA_ENV:-pyt2_8_gh}"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1090
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}" 2>/dev/null || true
fi

fail=0
ok()   { printf '  [ OK ]  %s\n' "$1"; }
miss() { printf '  [MISS]  %s\n' "$1"; fail=1; }

echo "env:            ${CONDA_PREFIX:-<none>}"
echo "python:         $(command -v python || echo MISSING)"
echo "PYTHONUSERBASE: ${PYTHONUSERBASE}"
echo "arch:           $(uname -m)"
echo "--- packages ---"
for m in jax numpy scipy h5py sympy optuna optax pydantic; do
  if python -c "import $m" 2>/dev/null; then ok "import $m"; else miss "import $m  (run run_install.sh)"; fi
done

echo "--- vulcan_jax + FastChem ---"
if VJ="$(python -c 'import vulcan_jax,pathlib;print(pathlib.Path(vulcan_jax.__file__).resolve().parent)' 2>/dev/null)"; then
  ok "import vulcan_jax  (${VJ})"
  arch="$(python - "${VJ}/fastchem_vulcan/fastchem" <<'PY'
import sys
try: b = open(sys.argv[1], "rb").read(20)
except FileNotFoundError: print("missing"); raise SystemExit
print("notelf" if b[:4]!=b"\x7fELF" else {62:"x86_64",183:"aarch64",40:"arm"}.get(int.from_bytes(b[18:20],"little"),"elf?"))
PY
)"
  want="$(uname -m)"; [ "${want}" = "arm64" ] && want="aarch64"
  if [ "${arch}" = "${want}" ]; then ok "FastChem binary native (${arch})"
  else miss "FastChem binary = '${arch}', need '${want}'  (run ensure_vulcan_jax.sh)"; fi
else
  miss "import vulcan_jax  (run run_install.sh on a login node)"
fi

echo "--- emulator source ---"
[ -d "${SCRIPT_DIR}/../src" ] && ok "src/ present" || miss "src/ missing (wrong cwd / incomplete clone)"
[ -f "${SCRIPT_DIR}/../config/vulcan_luhman16a_10k.json" ] && ok "default config present" || miss "default config missing"

CHECK_GPU="${CHECK_GPU:-0}"
echo "--- jax backend ---"
backend="$(python -c 'import jax; print(jax.default_backend())' 2>/dev/null || echo error)"
if [ "${CHECK_GPU}" = "1" ]; then
  # On a compute node (via check_ready.pbs): a GPU backend is REQUIRED for training.
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then ok "nvidia-smi sees a GPU"
  else miss "no usable GPU (nvidia-smi failed)"; fi
  case "${backend}" in
    gpu|cuda|rocm) ok "jax backend = ${backend}" ;;
    *) miss "jax backend = '${backend}' (expected gpu/cuda)" ;;
  esac
else
  echo "  backend: ${backend}  (informational on a login node; run check_ready.pbs to verify GPU)"
fi

echo "----------------------------------------"
if [ "${fail}" = 0 ]; then
  echo "READY — qsub supercomputer_cmds/run.pbs"
else
  echo "NOT READY — see [MISS] above; run: bash supercomputer_cmds/run_install.sh"
fi
exit "${fail}"
