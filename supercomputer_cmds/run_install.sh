#!/bin/bash
# run_install.sh — one-time setup for the VULCAN emulator + VULCAN-JAX backend.
#
# Run this ONCE on a LOGIN / front-end node (pip needs internet, which the PBS
# compute nodes do not have):
#
#   bash supercomputer_cmds/run_install.sh
#
# Design for this system (NAS GH200):
#   * $HOME (/home4/$USER) is small and often OVER QUOTA, and the shared conda
#     env (pyt2_8_gh) is READ-ONLY. So we do NOT write to either:
#       - hpc_env.sh redirects every cache (pip/conda/JAX/mpl/tmp) AND the
#         Python user-site (PYTHONUSERBASE) to /nobackup.
#       - we keep using the shared env's interpreter (with its working GPU JAX)
#         and install our extra packages with `pip install --user`, which lands
#         in the writable PYTHONUSERBASE on /nobackup.
#   * The login node is aarch64 (same as the GH200 compute nodes), so the
#     FastChem binary compiled here also runs on the compute nodes.
#
# After this, just `qsub supercomputer_cmds/run.pbs` (or run_one.pbs). Those
# scripts source hpc_env.sh too, so they see the same PYTHONUSERBASE and caches.
#
# vulcan-jax is published on TestPyPI only (not on PyPI), hence the pinned index.
#
# Env vars:
#   VULCAN_SCRATCH   scratch root for caches + user-site (default /nobackup/$USER/.vulcan)
#   CONDA_ENV        interpreter env to use (default pyt2_8_gh)
#   CXX              C++ compiler for the FastChem build (default g++)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# Redirect all HOME-bound caches + the user-site to /nobackup BEFORE anything
# touches $HOME (must precede conda/pip).
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/hpc_env.sh"

# --- Conda interpreter (shared, read-only env is fine; we install --user) -----
module purge 2>/dev/null || true
module use -a /swbuild/analytix/tools/modulefiles 2>/dev/null || true
module load miniconda3/gh2 2>/dev/null || true
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Run 'module load miniconda3/gh2' first."
  exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-pyt2_8_gh}"
conda activate "${CONDA_ENV}" || {
  echo "ERROR: could not activate ${CONDA_ENV}."; exit 2;
}
echo "[install] interpreter: $(command -v python)  (env ${CONDA_ENV})"
echo "[install] user-site target: ${PYTHONUSERBASE}"

# Verify the user-site really is on /nobackup and writable.
if [ ! -w "${PYTHONUSERBASE}" ]; then
  echo "ERROR: PYTHONUSERBASE not writable: ${PYTHONUSERBASE}"
  echo "       Set VULCAN_SCRATCH to a writable /nobackup path and retry."
  exit 2
fi

FORCE="${FORCE:-0}"

# --- 1. vulcan-jax (TestPyPI, --user, --no-deps so GPU JAX is untouched) ------
# Skip the network install entirely if it already imports (FORCE=1 to override).
if [ "${FORCE}" != "1" ] && python -c "import vulcan_jax" 2>/dev/null; then
  echo "[install] vulcan_jax already importable — skipping pip install (FORCE=1 to reinstall)"
else
  echo "[install] pip install --user -U vulcan-jax (TestPyPI, --no-deps)"
  # -U so FORCE=1 actually upgrades to a newer release (e.g. 0.1.7 -> 0.1.8);
  # the skip-if-importable guard above keeps the no-FORCE re-run offline/fast.
  python -m pip install --user -U \
    -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    --no-deps vulcan-jax
fi

# --- 2. JAX: the emulator needs a *GPU* JAX. pyt2_8_gh is PyTorch-only (ships
#        no jax), so install one into the user-site. On Python 3.10 + aarch64
#        (GH200) the newest jax with cp310 aarch64 CUDA12 wheels is 0.6.2; newer
#        jax dropped Python 3.10. If a future env already provides its own GPU
#        jax, we detect and reuse it instead.
JAX_SPEC="${JAX_SPEC:-jax[cuda12]==0.6.2}"   # installed when the env lacks jax
ENV_JAX_VER="$(PYTHONNOUSERSITE=1 python -c 'import importlib.metadata as m;print(m.version("jax"))' 2>/dev/null || true)"
if [ -n "${ENV_JAX_VER}" ]; then
  JAX_PIN="${ENV_JAX_VER}"
  JAXLIB_PIN="$(PYTHONNOUSERSITE=1 python -c 'import importlib.metadata as m;print(m.version("jaxlib"))' 2>/dev/null || echo "${JAX_PIN}")"
  STRIP_USERSITE_JAX=1                        # env jax wins; drop any user-site shadow
  echo "[install] env '${CONDA_ENV}' provides jax ${ENV_JAX_VER}; reusing it (pinning deps to it)."
else
  echo "[install] env has no jax; installing GPU ${JAX_SPEC} into the user-site (aarch64 GH200)."
  python -m pip install --user -U "${JAX_SPEC}"
  JAX_PIN="0.6.2"; JAXLIB_PIN="0.6.2"
  STRIP_USERSITE_JAX=0                        # the user-site jax IS our GPU jax — keep it
fi

# --- 3. Other deps, PINNED to the chosen jax so pip can't bump/replace it ------
MISSING="$(python - <<'PY'
import importlib.util as u
req = {"sympy":"sympy","optuna":"optuna","optax":"optax",
       "pydantic":"pydantic","matplotlib":"matplotlib"}
print(" ".join(pip for mod, pip in req.items() if u.find_spec(mod) is None))
PY
)"
if [ -n "${MISSING}" ]; then
  CONSTRAINTS="${VULCAN_SCRATCH}/pip-constraints.txt"
  printf 'jax==%s\njaxlib==%s\n' "${JAX_PIN}" "${JAXLIB_PIN}" > "${CONSTRAINTS}"
  echo "[install] pip install --user deps (${MISSING}) pinned to jax==${JAX_PIN}"
  python -m pip install --user -U -c "${CONSTRAINTS}" ${MISSING}
else
  echo "[install] all emulator Python deps already present"
fi

# If the ENV ships its own jax, strip any user-site jax so the env GPU build
# wins on import. When we installed jax into the user-site ourselves, keep it.
if [ "${STRIP_USERSITE_JAX}" = "1" ]; then
  USERSITE="$(python -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
  if [ -n "${USERSITE}" ] && [ -d "${USERSITE}" ]; then
    shopt -s nullglob
    for stray in "${USERSITE}"/jax "${USERSITE}"/jaxlib "${USERSITE}"/jax_plugins \
                 "${USERSITE}"/jax-*.dist-info "${USERSITE}"/jaxlib-*.dist-info \
                 "${USERSITE}"/jax_cuda*; do
      echo "[install] removing user-site $(basename "${stray}") so the env GPU JAX is used"
      rm -rf "${stray}"
    done
    shopt -u nullglob
  fi
fi

# Sanity: core imports resolve, and report the JAX backend. This install runs
# on a GH200 node, so a healthy GPU JAX prints backend=gpu here.
python - <<'PY'
import importlib.util as u
core = ["jax","numpy","scipy","h5py","sympy","optuna","optax","pydantic","vulcan_jax"]
missing = [m for m in core if u.find_spec(m) is None]
if missing:
    raise SystemExit(f"ERROR: still missing required modules: {missing}")
print("[install] core imports OK:", ", ".join(core))
import jax
be = jax.default_backend()
print(f"[install] jax {jax.__version__}  backend={be}  devices={jax.devices()}")
if be not in ("gpu", "cuda", "rocm"):
    print("[install] WARNING: JAX is NOT on the GPU. If this ran on a GH200 node, the")
    print("[install]          CUDA plugin is missing/incompatible — run.pbs will fail its GPU preflight.")
PY

# --- 4. Compile FastChem for this node's arch (into the user-site copy) --------
"${SCRIPT_DIR}/ensure_vulcan_jax.sh"

echo ""
echo "[install] done. Everything lives under VULCAN_SCRATCH=${VULCAN_SCRATCH} (on /nobackup)."
echo "[install] submit a run with:"
echo "    qsub supercomputer_cmds/run.pbs        # generation + tuning"
echo "    qsub supercomputer_cmds/run_one.pbs    # single full training run"
