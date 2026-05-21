#!/bin/bash
# Submit the sharded generation array + dependent merge job in one shot.
#
# Usage:
#   bash supercomputer_cmds/submit_gen_array.sh
#   CONFIG_PATH=config/other.json bash supercomputer_cmds/submit_gen_array.sh
#
# Prints both job IDs. The merge job fires automatically once every array
# task succeeds (--dependency=afterok). If any shard fails, the merge job
# stays queued in DependencyNeverSatisfied until the failing shard is
# re-submitted (sbatch --array=K) and merge is re-submitted manually with
# --dependency=afterok:<new_array_id>.

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-config/vulcan_condensation.json}

if [ -z "${PROJECT_ROOT:-}" ]; then
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
fi
cd -P "${PROJECT_ROOT}"

ARRAY_ID=$(sbatch --parsable \
  --export=ALL,CONFIG_PATH="$CONFIG_PATH" \
  supercomputer_cmds/run_gen_array.sh)
echo "Submitted generation array: $ARRAY_ID"

MERGE_ID=$(sbatch --parsable \
  --dependency=afterok:"$ARRAY_ID" \
  --export=ALL,CONFIG_PATH="$CONFIG_PATH" \
  supercomputer_cmds/run_merge.sh)
echo "Submitted merge job: $MERGE_ID (depends on array $ARRAY_ID)"
