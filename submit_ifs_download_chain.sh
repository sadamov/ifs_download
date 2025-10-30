#!/usr/bin/env bash

set -euo pipefail

# submit_ifs_bulk_chain.sh
#
# Submit multiple copies of submit_ifs_bulk_master.sh in a dependency chain so
# that only one runs at a time, and the next starts when the previous finishes
# (e.g., due to TIMEOUT or completion).
#
# Usage examples:
#   ./submit_ifs_bulk_chain.sh -n 4                    # 4 chained jobs, default dep afterany
#   ./submit_ifs_bulk_chain.sh -n 6 -d afternotok      # start next only if previous did NOT complete OK
#   ./submit_ifs_bulk_chain.sh -n 3 -- --partition=normal --time=12:00:00
#
# Flags:
#   -n N           Number of jobs to queue (default: 4)
#   -d DEP         Dependency type: afterany | afternotok | afterok (default: afterany)
#   -h             Show help
#
# Any arguments after "--" are forwarded to sbatch (and override #SBATCH lines if present).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_SCRIPT="${SCRIPT_DIR}/submit_ifs_bulk_master.sh"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not found in PATH." >&2
  exit 1
fi

if [ ! -x "$MASTER_SCRIPT" ]; then
  echo "Error: $MASTER_SCRIPT not found or not executable." >&2
  exit 1
fi

N=4
DEP_TYPE="afterany"

print_help() {
  sed -n '1,60p' "$0" | sed -n 's/^# \{0,1\}//p'
}

SBATCH_EXTRA=()

while (( "$#" )); do
  case "$1" in
    -n)
      N="$2"; shift 2;;
    -d)
      DEP_TYPE="$2"; shift 2;;
    -h|--help)
      print_help; exit 0;;
    --)
      shift
      SBATCH_EXTRA=("$@")
      break;;
    *)
      echo "Unknown option: $1" >&2
      print_help
      exit 1;;
  esac
done

case "$DEP_TYPE" in
  afterany|afternotok|afterok) :;;
  *)
    echo "Error: invalid dependency type '$DEP_TYPE' (use afterany|afternotok|afterok)" >&2
    exit 1;;
esac

if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: -n must be a positive integer" >&2
  exit 1
fi

echo "Submitting $N jobs in a '$DEP_TYPE' chain for: $MASTER_SCRIPT"
if [ ${#SBATCH_EXTRA[@]} -gt 0 ]; then
  echo "Extra sbatch args: ${SBATCH_EXTRA[*]}"
fi

submit_and_get_id() {
  # shellcheck disable=SC2068
  local out
  out=$(sbatch ${SBATCH_EXTRA[@]:-} "$MASTER_SCRIPT") || return 1
  # Expected format: "Submitted batch job <JOBID>"
  echo "$out" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+$/){print $i; exit}}}'
}

submit_with_dep_and_get_id() {
  local dep_job_id="$1"
  # shellcheck disable=SC2068
  local out
  out=$(sbatch --dependency="${DEP_TYPE}:${dep_job_id}" ${SBATCH_EXTRA[@]:-} "$MASTER_SCRIPT") || return 1
  echo "$out" | awk '{for(i=1;i<=NF;i++){if($i~/^[0-9]+$/){print $i; exit}}}'
}

first_job_id="$(submit_and_get_id)"
if [ -z "$first_job_id" ]; then
  echo "Error: failed to parse first job ID" >&2
  exit 1
fi
echo "[#1] Submitted job $first_job_id"

prev_job_id="$first_job_id"

for ((i=2; i<=N; i++)); do
  jid="$(submit_with_dep_and_get_id "$prev_job_id")"
  if [ -z "$jid" ]; then
    echo "Error: failed to parse job ID for submission #$i" >&2
    exit 1
  fi
  echo "[#${i}] Submitted job $jid (dependency: ${DEP_TYPE}:${prev_job_id})"
  prev_job_id="$jid"
done

echo "Chain submitted successfully. First job: $first_job_id, Last job: $prev_job_id"
echo "Tip: use 'squeue -j $first_job_id,$prev_job_id' to monitor endpoints of the chain."
