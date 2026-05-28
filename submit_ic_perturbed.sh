#!/bin/bash
#SBATCH --job-name=ic_perturbed_dl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --output=logs/ic_perturbed_%j.out
#SBATCH --error=logs/ic_perturbed_%j.err
#SBATCH --requeue

# IFS ENS analysis (type=pf step=0) MARS download for AI baseline IC use.
#
# Submits MARS requests one init_time at a time (PL + SFC), so 224 inits ->
# ~448 MARS requests across (PL, SFC). Each request is one tape mount per
# MARS efficiency best practices: a single tape mount per (date, time,
# levtype) covering all params/levels/members.
#
# Output: /capstor/store/cscs/swissai/a122/IFS/ifs_analysis_perturbed_ic.zarr
# Storage estimate: ~3.8 TB (50 members x 224 inits x 85 fields x 4 MB)
#
# Resume: the script is idempotent -- already-written init_time slices are
# skipped. Just rerun the same sbatch and missing inits will be filled in.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"
mkdir -p logs

# Load config (only OUT_ZARR matters for the IC variant; rest is in the python)
CONFIG_FILE="${CONFIG_FILE:-$REPO_DIR/config_ic_perturbed.env}"
if [[ -f "$CONFIG_FILE" ]]; then
    while IFS= read -r line; do
        key="${line%%=*}"
        val="${line#*=}"
        printf -v "$key" "%s" "$val"
        export "$key"
    done < <(sed -e 's/\r$//' -n -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' -e '/^[A-Za-z_][A-Za-z0-9_]*=.*/p' -- "$CONFIG_FILE")
fi

OUT_ZARR="${OUT_ZARR:-/capstor/store/cscs/swissai/a122/IFS/ifs_analysis_perturbed_ic.zarr}"

# Always redirect tmp + dask spill to iopsstor (project-wide rule)
export TMPDIR=/iopsstor/scratch/cscs/sadamov/dask-tmp
export DASK_TEMPORARY_DIRECTORY=/iopsstor/scratch/cscs/sadamov/dask-tmp
mkdir -p "$TMPDIR"

source "$REPO_DIR/.venv/bin/activate"

# Point Python's SSL to certifi's CA bundle -- system bundles don't exist on
# the compute nodes, so ECMWF API HTTPS calls fail certificate verification
# otherwise. certifi is in requirements.txt.
export SSL_CERT_FILE="$($REPO_DIR/.venv/bin/python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

# Unbuffered stdout so live progress is visible in the SLURM log file.
export PYTHONUNBUFFERED=1

python -u "$REPO_DIR/download_ic_perturbed.py" --out-zarr "$OUT_ZARR" "$@"
