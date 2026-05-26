#!/bin/bash
# Submit install → generation array → merge as a three-stage dependency chain.
#
# The install job runs once on a single node and sets up the shared conda
# env before any array task starts, preventing pip-install race conditions
# on NFS.
#
# Usage:
#   bash supercomputer_cmds/submit_gen_array.sh
#   CONFIG_PATH=config/exogibbs_luhman16a_10k.json bash supercomputer_cmds/submit_gen_array.sh
#   CONFIG_PATH=config/vulcan_luhman16a_10k.json bash supercomputer_cmds/submit_gen_array.sh
#   SKIP_INSTALL=1 bash supercomputer_cmds/submit_gen_array.sh   # env already set up
#
# Prints all job IDs. The merge job fires automatically once every array
# task succeeds (--dependency=afterok). If any shard fails, the merge job
# stays queued in DependencyNeverSatisfied until the failing shard is
# re-submitted (sbatch --array=K) and merge is re-submitted manually with
# --dependency=afterok:<new_array_id>.

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-config/vulcan_luhman16a_10k.json}
SKIP_INSTALL=${SKIP_INSTALL:-0}

if [ -z "${PROJECT_ROOT:-}" ]; then
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
fi
cd -P "${PROJECT_ROOT}"

# --- Step 1: install job (single node, runs once) -----------------------
# Skippable with SKIP_INSTALL=1 if the env is already set up.
if [ "$SKIP_INSTALL" != "1" ]; then
  INSTALL_RAW=$(sbatch --parsable \
    --export=ALL,CONFIG_PATH="$CONFIG_PATH" \
    supercomputer_cmds/run_install.sh)
  INSTALL_ID="${INSTALL_RAW%%;*}"
  INSTALL_CLUSTER="${INSTALL_RAW##*;}"
  if [ "$INSTALL_CLUSTER" = "$INSTALL_RAW" ]; then
    INSTALL_CLUSTER=""
  fi
  echo "Submitted install job: $INSTALL_RAW"
  ARRAY_DEP="--dependency=afterok:$INSTALL_ID"
else
  INSTALL_CLUSTER=""
  ARRAY_DEP=""
  echo "Skipping install job (SKIP_INSTALL=1)"
fi

# --- Step 2: generation array (depends on install) ----------------------
ARRAY_RAW=$(sbatch --parsable \
  ${ARRAY_DEP:+"$ARRAY_DEP"} \
  --export=ALL,CONFIG_PATH="$CONFIG_PATH",SKIP_INSTALL=1 \
  supercomputer_cmds/run_gen_array.sh)
# On multi-cluster SLURM setups (--clusters=edge), --parsable returns
# "JOBID;CLUSTER". Strip the suffix so --dependency=afterok:<jobid> parses.
ARRAY_ID="${ARRAY_RAW%%;*}"
ARRAY_CLUSTER="${ARRAY_RAW##*;}"
if [ "$ARRAY_CLUSTER" = "$ARRAY_RAW" ]; then
  ARRAY_CLUSTER=""
fi
echo "Submitted generation array: $ARRAY_RAW"

# --- Step 3: merge job (depends on array) -------------------------------
MERGE_RAW=$(sbatch --parsable \
  ${ARRAY_CLUSTER:+--clusters="$ARRAY_CLUSTER"} \
  --dependency=afterok:"$ARRAY_ID" \
  --export=ALL,CONFIG_PATH="$CONFIG_PATH" \
  supercomputer_cmds/run_merge.sh)
echo "Submitted merge job: $MERGE_RAW (depends on array $ARRAY_ID)"
