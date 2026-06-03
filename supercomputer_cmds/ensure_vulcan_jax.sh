#!/bin/bash
# ensure_vulcan_jax.sh — make the active conda env's vulcan_jax usable on THIS node.
#
# Idempotent. Two guarantees:
#   1. `import vulcan_jax` works in the active env. If not, it prints a clear
#      error pointing at run_install.sh (pip needs internet -> a login node).
#   2. The bundled FastChem binary is a native executable for `$(uname -m)`.
#      If it is missing or was built for a different architecture, recompile it
#      in place from the package's own `fastchem_src/` (no internet needed; just
#      `make` + a C++ compiler). VULCAN-master is NOT required.
#
# Run it (do not source) AFTER activating the conda env:
#   supercomputer_cmds/ensure_vulcan_jax.sh
#
# Used by run_install.sh (login node) and run.pbs / run_one.pbs (compute node).
# Because the VULCAN-JAX wheel is pure-python and ships FastChem source only,
# the compute-node call rebuilds FastChem for the compute arch when the login
# node had a different one (e.g. x86_64 login vs aarch64 GH200) — automatically.
#
# Env vars:
#   CXX   C++ compiler for the FastChem build (default: g++)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# --- 1. vulcan_jax must be importable in the active env -----------------------
VJ_PKG="$(python -c 'import vulcan_jax, pathlib; print(pathlib.Path(vulcan_jax.__file__).resolve().parent)' 2>/dev/null)" || {
  echo "ERROR: 'vulcan_jax' is not importable in env '${CONDA_PREFIX:-<none>}'."
  echo "       Install it on a LOGIN node (internet required), then resubmit:"
  echo "         bash \"${SCRIPT_DIR}/run_install.sh\""
  exit 3
}

FC_DIR="${VJ_PKG}/fastchem_vulcan"
FC_BIN="${FC_DIR}/fastchem"
ARCH="$(uname -m)"
echo "[ensure] vulcan_jax=${VJ_PKG}"
echo "[ensure] node arch=${ARCH}"

# --- 2. FastChem binary must be native to this node's arch --------------------
# Read the ELF e_machine field directly so we don't depend on `file`/`readelf`.
elf_arch() {
  python - "$1" <<'PY'
import sys
try:
    with open(sys.argv[1], "rb") as fh:
        head = fh.read(20)
except FileNotFoundError:
    print("missing"); raise SystemExit
if head[:4] != b"\x7fELF":
    print("notelf"); raise SystemExit
machine = int.from_bytes(head[18:20], "little")
print({62: "x86_64", 183: "aarch64", 40: "arm"}.get(machine, f"elf{machine}"))
PY
}

want="${ARCH}"; [ "${ARCH}" = "arm64" ] && want="aarch64"
have="$(elf_arch "${FC_BIN}")"

if [ "${have}" = "${want}" ]; then
  echo "[ensure] FastChem binary already native (${have}); nothing to do."
  exit 0
fi
echo "[ensure] FastChem binary is '${have}', need '${want}' — recompiling from bundled source."

# --- 3. Compile in place from fastchem_src/ (no VULCAN-master, no internet) ----
if [ ! -w "${FC_DIR}" ]; then
  echo "ERROR: ${FC_DIR} is not writable — cannot compile FastChem here."
  echo "       Install into a writable env (personal or cloned). See run_install.sh."
  exit 4
fi
command -v make >/dev/null 2>&1 || { echo "ERROR: 'make' not found (module load a build toolchain)."; exit 5; }
CXX_BIN="${CXX:-g++}"
command -v "${CXX_BIN}" >/dev/null 2>&1 || {
  echo "ERROR: C++ compiler '${CXX_BIN}' not found (module load gcc, or set CXX=...)."
  exit 5
}

echo "[ensure] building FastChem (CXX=${CXX_BIN}) in ${FC_DIR} ..."
make -C "${FC_DIR}" clean >/dev/null 2>&1 || true
make -C "${FC_DIR}" CXX="${CXX_BIN}" CC="${CXX_BIN}" all
[ -x "${FC_BIN}" ] || { echo "ERROR: FastChem build produced no executable at ${FC_BIN}"; exit 6; }

echo "[ensure] built FastChem ($(elf_arch "${FC_BIN}")) at ${FC_BIN}"
